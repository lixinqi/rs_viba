//! Unparse AST nodes back to Viba source (was viba/viba_ast/unparse.py).

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyFloat, PyInt, PyString};

use crate::pyast;

fn is_binary_name(name: &str) -> bool {
    matches!(
        name,
        "Sum" | "Product" | "Exponent" | "SumChain" | "ProductChain" | "ExponentChain"
    )
}

fn dedent(text: &str, prefix: &str) -> String {
    if let Some(index) = text.find('\n') {
        let (first, rest) = text.split_at(index);
        let mut first = first.to_string();
        if !prefix.is_empty() && first.starts_with(prefix) {
            first = first[prefix.len()..].to_string();
        }
        format!("{}{}", first, rest)
    } else {
        let mut first = text.to_string();
        if !prefix.is_empty() && first.starts_with(prefix) {
            first = first[prefix.len()..].to_string();
        }
        first
    }
}

fn tag_spelling(tag: &str) -> String {
    if tag.starts_with('$') {
        tag.to_string()
    } else {
        format!("${}", tag)
    }
}

fn int_spelling_ok(text: &str) -> bool {
    let digits = text.strip_prefix('-').unwrap_or(text);
    !digits.is_empty() && digits.chars().all(|c| c.is_ascii_digit())
}

fn float_spelling_ok(text: &str) -> bool {
    let digits = text.strip_prefix('-').unwrap_or(text);
    if let Some((before, after)) = digits.split_once('.') {
        if before.is_empty() && after.is_empty() {
            return false;
        }
        return before.chars().all(|c| c.is_ascii_digit())
            && after.chars().all(|c| c.is_ascii_digit());
    }
    false
}

pub fn literal_spelling(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<String> {
    if value.is_exact_instance_of::<PyBool>() {
        let flag: bool = value.extract()?;
        return Ok(if flag { "true" } else { "false" }.to_string());
    }
    if value.is_exact_instance_of::<PyString>() {
        let text: String = value.extract()?;
        return quote_string(&text);
    }
    if value.is_exact_instance_of::<PyInt>() {
        let spelling = value.str()?.to_string();
        if !int_spelling_ok(&spelling) {
            return Err(PyValueError::new_err(format!(
                "viba has no literal for this number: {}",
                spelling
            )));
        }
        return Ok(spelling);
    }
    if value.is_exact_instance_of::<PyFloat>() {
        let spelling = value.str()?.to_string();
        if !float_spelling_ok(&spelling) {
            return Err(PyValueError::new_err(format!(
                "viba has no literal for this number: {}",
                spelling
            )));
        }
        return Ok(spelling);
    }
    let name = value.get_type().name()?.to_string();
    Err(PyValueError::new_err(format!(
        "viba has no literal for {}",
        name
    )))
}

fn quote_string(value: &str) -> PyResult<String> {
    if !value.contains('\n') && !value.ends_with('\\') {
        if !value.contains('"') {
            return Ok(format!("\"{}\"", value));
        }
        if !value.contains('\'') {
            return Ok(format!("'{}'", value));
        }
    }
    if !value.contains("'''") && !value.ends_with('\'') {
        return Ok(format!("'''{}'''", value));
    }
    Err(PyValueError::new_err(
        "no viba string literal holds this text",
    ))
}

fn needs_parens(child: &str, parent: &str, position: &str) -> bool {
    if matches!(child, "SumChain" | "ProductChain" | "ExponentChain") {
        return true;
    }
    if parent == "Sum" || parent == "Product" {
        if child == parent {
            return false;
        }
        if matches!(child, "Sum" | "Product" | "Exponent") {
            return true;
        }
    }
    if parent == "Exponent" {
        if child == "Exponent" && position == "argument" {
            return false;
        }
        if matches!(child, "Sum" | "Product" | "Exponent") {
            return true;
        }
    }
    false
}

fn tagged_body_parens(node: &Bound<'_, PyAny>, unparsed: &str, prefix: &str) -> PyResult<String> {
    let body = dedent(unparsed, prefix);
    let name = pyast::class_name(node)?;
    if is_binary_name(&name) || name == "Tagged" || name == "Partial" {
        Ok(format!(" ({})", body))
    } else {
        Ok(format!(" {}", body))
    }
}

fn unparse_binary(
    py: Python<'_>,
    node: &Bound<'_, PyAny>,
    op: &str,
    indent: usize,
    depth: usize,
) -> PyResult<String> {
    let left = pyast::field(node, "left")?;
    let right = pyast::field(node, "right")?;
    let mut left_text = unparse_node(py, &left, indent, depth + 1)?;
    let mut right_text = unparse_node(py, &right, indent, depth + 1)?;
    let parent = pyast::class_name(node)?;
    if needs_parens(&pyast::class_name(&left)?, &parent, "left") {
        left_text = format!("({})", left_text);
    }
    if needs_parens(&pyast::class_name(&right)?, &parent, "right") {
        right_text = format!("({})", right_text);
    }
    Ok(format!("{}{}{}", left_text, op, right_text))
}

fn unparse_node(
    py: Python<'_>,
    node: &Bound<'_, PyAny>,
    indent: usize,
    depth: usize,
) -> PyResult<String> {
    let name = pyast::class_name(node)?;
    match name.as_str() {
        "Module" => {
            let body = pyast::field_list(node, "body")?;
            let mut parts = Vec::new();
            for item in body {
                parts.push(unparse_node(py, &item, indent, depth)?);
            }
            Ok(parts.join("\n\n"))
        }
        "TypeDefinition" => {
            let name = pyast::field_string(node, "name")?;
            let body = pyast::field(node, "body")?;
            let prefix = " ".repeat(indent * (depth + 1));
            let body_text = dedent(&unparse_node(py, &body, indent, depth + 1)?, &prefix);
            Ok(format!("{} =\n{}{}", name, prefix, body_text))
        }
        "GenericDefinition" => {
            let name = pyast::field_string(node, "name")?;
            let params: Vec<String> = pyast::field_list(node, "generic_params")?
                .into_iter()
                .map(|p| p.extract::<String>())
                .collect::<PyResult<_>>()?;
            let body = pyast::field(node, "body")?;
            let prefix = " ".repeat(indent * (depth + 1));
            let body_text = dedent(&unparse_node(py, &body, indent, depth + 1)?, &prefix);
            Ok(format!(
                "{}[{}] =\n{}{}",
                name,
                params.join(", "),
                prefix,
                body_text
            ))
        }
        "Import" => {
            let module = pyast::field_string(node, "module")?;
            match pyast::field_opt_string(node, "alias")? {
                Some(alias) => Ok(format!("import {} as {}", module, alias)),
                None => Ok(format!("import {}", module)),
            }
        }
        "Pattern" => {
            let pattern = pyast::field(node, "pattern")?;
            let prefix = " ".repeat(indent * (depth + 1));
            let body = dedent(&unparse_node(py, &pattern, indent, depth + 1)?, &prefix);
            Ok(format!("pattern\n{}{}", prefix, body))
        }
        "Sum" => unparse_binary(py, node, " | ", indent, depth),
        "Product" => unparse_binary(py, node, " * ", indent, depth),
        "Exponent" => {
            let result = pyast::field(node, "result")?;
            let argument = pyast::field(node, "argument")?;
            let mut result_text = unparse_node(py, &result, indent, depth + 1)?;
            let mut argument_text = unparse_node(py, &argument, indent, depth + 1)?;
            if is_binary_name(&pyast::class_name(&result)?) {
                result_text = format!("({})", result_text);
            }
            if is_binary_name(&pyast::class_name(&argument)?) {
                argument_text = format!("({})", argument_text);
            }
            Ok(format!("{} <- {}", result_text, argument_text))
        }
        "Partial" => {
            let function = pyast::field(node, "function")?;
            let argument = pyast::field(node, "argument")?;
            let mut function_text = unparse_node(py, &function, indent, depth + 1)?;
            let mut argument_text = unparse_node(py, &argument, indent, depth + 1)?;
            if is_binary_name(&pyast::class_name(&function)?) {
                function_text = format!("({})", function_text);
            }
            let argument_name = pyast::class_name(&argument)?;
            if is_binary_name(&argument_name) || argument_name == "Partial" {
                argument_text = format!("({})", argument_text);
            }
            Ok(format!("{} << {}", function_text, argument_text))
        }
        "Member" => {
            let tag = pyast::field_string(node, "tag")?;
            Ok(tag_spelling(&tag))
        }
        "MemberTaken" => {
            let owner = pyast::field(node, "owner")?;
            let member = pyast::field_string(node, "name")?;
            let owner_text = unparse_node(py, &owner, indent, depth)?;
            let owner_name = pyast::class_name(&owner)?;
            if matches!(owner_name.as_str(), "TypeRef" | "MemberTaken" | "TypeApp") {
                Ok(format!("{}.{}", owner_text, member))
            } else {
                Ok(format!("({}).{}", owner_text, member))
            }
        }
        "Tagged" => {
            let tag = tag_spelling(&pyast::field_string(node, "tag")?);
            let body = pyast::field(node, "type")?;
            let body_text = unparse_node(py, &body, indent, depth)?;
            let prefix = " ".repeat(indent * depth);
            Ok(format!(
                "{}{}",
                tag,
                tagged_body_parens(&body, &body_text, &prefix)?
            ))
        }
        "TypeApp" => {
            let constructor = pyast::field_string(node, "constructor")?;
            let args = pyast::field_list(node, "args")?;
            let inner_prefix = " ".repeat(indent * (depth + 1));
            let mut rendered = Vec::new();
            for arg in args {
                let text = unparse_node(py, &arg, indent, depth + 1)?;
                rendered.push(dedent(&text, &inner_prefix));
            }
            Ok(format!("{}[{}]", constructor, rendered.join(", ")))
        }
        "Tuple" => {
            let elements = pyast::field_list(node, "elements")?;
            let mut rendered = Vec::new();
            for element in &elements {
                rendered.push(unparse_node(py, element, indent, depth + 1)?);
            }
            let mut text = rendered.join(", ");
            if elements.len() == 1 {
                text.push(',');
            }
            Ok(format!("({})", text))
        }
        "TypeRef" => pyast::field_string(node, "name"),
        "Constant" => {
            let value = pyast::field(node, "value")?;
            literal_spelling(py, &value)
        }
        "Nil" => Ok("nil".to_string()),
        "Never" => Ok("never".to_string()),
        "Any" => Ok("Any".to_string()),
        "Ellipsis" => Ok("...".to_string()),
        "HintBlock" => Ok(format!("{{{}}}", pyast::field_string(node, "code")?)),
        "SumChain" => {
            let elements = pyast::field_list(node, "elements")?;
            if elements.is_empty() {
                return Ok("never".to_string());
            }
            let line_indent = " ".repeat(indent * depth);
            let inner_indent = " ".repeat(indent * (depth + 1));
            let mut lines = Vec::new();
            for (index, element) in elements.iter().enumerate() {
                let mut text = dedent(
                    &unparse_node(py, element, indent, depth + 1)?,
                    &inner_indent,
                );
                let element_name = pyast::class_name(element)?;
                if element_name == "SumChain" || element_name == "Partial" {
                    text = format!("({})", text);
                }
                if index == 0 {
                    lines.push(format!("{}{}", line_indent, text));
                } else {
                    lines.push(format!("{}| {}", line_indent, text));
                }
            }
            Ok(lines.join("\n"))
        }
        "ProductChain" => {
            let elements = pyast::field_list(node, "elements")?;
            if elements.is_empty() {
                return Ok("nil".to_string());
            }
            let line_indent = " ".repeat(indent * depth);
            let inner_indent = " ".repeat(indent * (depth + 1));
            let mut lines = Vec::new();
            for (index, element) in elements.iter().enumerate() {
                let mut text = dedent(
                    &unparse_node(py, element, indent, depth + 1)?,
                    &inner_indent,
                );
                let element_name = pyast::class_name(element)?;
                if matches!(
                    element_name.as_str(),
                    "SumChain" | "ProductChain" | "Partial"
                ) {
                    text = format!("({})", text);
                }
                if index == 0 {
                    lines.push(format!("{}{}", line_indent, text));
                } else {
                    lines.push(format!("{}* {}", line_indent, text));
                }
            }
            Ok(lines.join("\n"))
        }
        "ExponentChain" => {
            let elements = pyast::field_list(node, "elements")?;
            if elements.is_empty() {
                return Ok("never".to_string());
            }
            let line_indent = " ".repeat(indent * depth);
            let inner_indent = " ".repeat(indent * (depth + 1));
            let mut lines = Vec::new();
            for (index, element) in elements.iter().enumerate() {
                let mut text = dedent(
                    &unparse_node(py, element, indent, depth + 1)?,
                    &inner_indent,
                );
                let element_name = pyast::class_name(element)?;
                if is_binary_name(&element_name) || element_name == "Partial" {
                    text = format!("({})", text);
                }
                if index == 0 {
                    lines.push(format!("{}{}", line_indent, text));
                } else {
                    lines.push(format!("{}<- {}", line_indent, text));
                }
            }
            Ok(lines.join("\n"))
        }
        _ => Err(PyTypeError::new_err(format!(
            "No handler matched for node: {}",
            name
        ))),
    }
}

pub fn unparse_module(py: Python<'_>, tree: &Bound<'_, PyAny>, indent: usize) -> PyResult<String> {
    let body: Vec<Bound<'_, PyAny>> = if pyast::class_name(tree)? == "Module" {
        pyast::field_list(tree, "body")?
    } else {
        let mut items = Vec::new();
        for item in tree.try_iter()? {
            items.push(item?);
        }
        items
    };
    let mut parts = Vec::new();
    for item in body {
        parts.push(unparse_node(py, &item, indent, 0)?);
    }
    Ok(parts.join("\n\n"))
}

pub fn unparse_type(py: Python<'_>, node: &Bound<'_, PyAny>, indent: usize) -> PyResult<String> {
    unparse_node(py, node, indent, 0)
}
