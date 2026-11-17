//! Chain-style canonicalization (was viba/viba_ast/chain.py).

use pyo3::prelude::*;

use crate::pyast::{self, Nodes};

fn is_class(node: &Bound<'_, PyAny>, name: &str) -> PyResult<bool> {
    Ok(pyast::class_name(node)? == name)
}

fn elements(node: &Bound<'_, PyAny>) -> PyResult<Vec<Py<PyAny>>> {
    Ok(pyast::field_list(node, "elements")?
        .into_iter()
        .map(|item| item.unbind())
        .collect())
}

pub fn convert_to_chain_style<'py>(
    nodes: &Nodes<'py>,
    node: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    let py = nodes.module.py();
    let name = pyast::class_name(node)?;
    match name.as_str() {
        "Sum" => {
            let mut items: Vec<Py<PyAny>> = Vec::new();
            let left = pyast::field(node, "left")?;
            let converted = convert_to_chain_style(nodes, &left)?;
            let converted_bound = converted.bind(py);
            if is_class(converted_bound, "SumChain")? {
                items.extend(elements(converted_bound)?);
            } else {
                items.push(converted);
            }
            let right = pyast::field(node, "right")?;
            items.push(convert_to_chain_style(nodes, &right)?);
            let items = pyast::pylist(py, items)?;
            nodes.make("SumChain", &[("elements", items)])
        }
        "Product" => {
            let mut items: Vec<Py<PyAny>> = Vec::new();
            let left = pyast::field(node, "left")?;
            let converted = convert_to_chain_style(nodes, &left)?;
            let converted_bound = converted.bind(py);
            if is_class(converted_bound, "ProductChain")? {
                items.extend(elements(converted_bound)?);
            } else {
                items.push(converted);
            }
            let right = pyast::field(node, "right")?;
            items.push(convert_to_chain_style(nodes, &right)?);
            let items = pyast::pylist(py, items)?;
            nodes.make("ProductChain", &[("elements", items)])
        }
        "Exponent" => {
            let exponent = pyast::field(node, "argument")?;
            let result = pyast::field(node, "result")?;
            let argument = convert_to_chain_style(nodes, &exponent)?;
            let result = convert_to_chain_style(nodes, &result)?;
            let mut items: Vec<Py<PyAny>> = Vec::new();
            if is_class(result.bind(py), "ExponentChain")? {
                items.extend(elements(result.bind(py))?);
            } else {
                items.push(result);
            }
            items.push(argument);
            let items = pyast::pylist(py, items)?;
            nodes.make("ExponentChain", &[("elements", items)])
        }
        "Partial" => {
            let function = pyast::field(node, "function")?;
            let argument = pyast::field(node, "argument")?;
            let function = convert_to_chain_style(nodes, &function)?;
            let argument = convert_to_chain_style(nodes, &argument)?;
            nodes.make("Partial", &[("function", function), ("argument", argument)])
        }
        "TypeDefinition" => {
            let name = pyast::field_string(node, "name")?;
            let body = pyast::field(node, "body")?;
            let body = convert_to_chain_style(nodes, &body)?;
            nodes.make(
                "TypeDefinition",
                &[("name", pyast::pystr(py, &name)), ("body", body)],
            )
        }
        "GenericDefinition" => {
            let name = pyast::field_string(node, "name")?;
            let params = pyast::field(node, "generic_params")?;
            let body = pyast::field(node, "body")?;
            let body = convert_to_chain_style(nodes, &body)?;
            nodes.make(
                "GenericDefinition",
                &[
                    ("name", pyast::pystr(py, &name)),
                    ("generic_params", params.unbind()),
                    ("body", body),
                ],
            )
        }
        "Pattern" => {
            let pattern = pyast::field(node, "pattern")?;
            let pattern = convert_to_chain_style(nodes, &pattern)?;
            nodes.make("Pattern", &[("pattern", pattern)])
        }
        "Tagged" => {
            let tag = pyast::field(node, "tag")?;
            let body = pyast::field(node, "type")?;
            let body = convert_to_chain_style(nodes, &body)?;
            nodes.make("Tagged", &[("tag", tag.unbind()), ("type", body)])
        }
        "Member" => Ok(node.clone().unbind()),
        "MemberTaken" => {
            let owner = pyast::field(node, "owner")?;
            let name = pyast::field(node, "name")?;
            let owner = convert_to_chain_style(nodes, &owner)?;
            nodes.make("MemberTaken", &[("owner", owner), ("name", name.unbind())])
        }
        "TypeApp" => {
            let constructor = pyast::field(node, "constructor")?;
            let args = pyast::field_list(node, "args")?;
            let mut converted = Vec::new();
            for arg in args {
                converted.push(convert_to_chain_style(nodes, &arg)?);
            }
            let args = pyast::pylist(py, converted)?;
            nodes.make(
                "TypeApp",
                &[("constructor", constructor.unbind()), ("args", args)],
            )
        }
        "Tuple" => {
            let items = pyast::field_list(node, "elements")?;
            let mut converted = Vec::new();
            for item in items {
                converted.push(convert_to_chain_style(nodes, &item)?);
            }
            let items = pyast::pylist(py, converted)?;
            nodes.make("Tuple", &[("elements", items)])
        }
        _ => Ok(node.clone().unbind()),
    }
}

pub fn convert_from_chain_style<'py>(
    nodes: &Nodes<'py>,
    node: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    let py = nodes.module.py();
    let name = pyast::class_name(node)?;
    match name.as_str() {
        "SumChain" => {
            let items = elements(node)?;
            if items.is_empty() {
                return nodes.make0("Never");
            }
            let mut result = convert_from_chain_style(nodes, items[0].bind(py))?;
            for item in &items[1..] {
                let right = convert_from_chain_style(nodes, item.bind(py))?;
                result = nodes.make("Sum", &[("left", result), ("right", right)])?;
            }
            Ok(result)
        }
        "ProductChain" => {
            let items = elements(node)?;
            if items.is_empty() {
                return nodes.make0("Nil");
            }
            let mut result = convert_from_chain_style(nodes, items[0].bind(py))?;
            for item in &items[1..] {
                let right = convert_from_chain_style(nodes, item.bind(py))?;
                result = nodes.make("Product", &[("left", result), ("right", right)])?;
            }
            Ok(result)
        }
        "ExponentChain" => {
            let items = elements(node)?;
            if items.is_empty() {
                return nodes.make0("Never");
            }
            let mut result = convert_from_chain_style(nodes, items[0].bind(py))?;
            for item in &items[1..] {
                let argument = convert_from_chain_style(nodes, item.bind(py))?;
                result = nodes.make("Exponent", &[("result", result), ("argument", argument)])?;
            }
            Ok(result)
        }
        "TypeDefinition" => {
            let name = pyast::field_string(node, "name")?;
            let body = pyast::field(node, "body")?;
            let body = convert_from_chain_style(nodes, &body)?;
            nodes.make(
                "TypeDefinition",
                &[("name", pyast::pystr(py, &name)), ("body", body)],
            )
        }
        "GenericDefinition" => {
            let name = pyast::field_string(node, "name")?;
            let params = pyast::field(node, "generic_params")?;
            let body = pyast::field(node, "body")?;
            let body = convert_from_chain_style(nodes, &body)?;
            nodes.make(
                "GenericDefinition",
                &[
                    ("name", pyast::pystr(py, &name)),
                    ("generic_params", params.unbind()),
                    ("body", body),
                ],
            )
        }
        "Pattern" => {
            let pattern = pyast::field(node, "pattern")?;
            let pattern = convert_from_chain_style(nodes, &pattern)?;
            nodes.make("Pattern", &[("pattern", pattern)])
        }
        "Tagged" => {
            let tag = pyast::field(node, "tag")?;
            let body = pyast::field(node, "type")?;
            let body = convert_from_chain_style(nodes, &body)?;
            nodes.make("Tagged", &[("tag", tag.unbind()), ("type", body)])
        }
        "MemberTaken" => {
            let owner = pyast::field(node, "owner")?;
            let name = pyast::field(node, "name")?;
            let owner = convert_from_chain_style(nodes, &owner)?;
            nodes.make("MemberTaken", &[("owner", owner), ("name", name.unbind())])
        }
        "TypeApp" => {
            let constructor = pyast::field(node, "constructor")?;
            let args = pyast::field_list(node, "args")?;
            let mut converted = Vec::new();
            for arg in args {
                converted.push(convert_from_chain_style(nodes, &arg)?);
            }
            let args = pyast::pylist(py, converted)?;
            nodes.make(
                "TypeApp",
                &[("constructor", constructor.unbind()), ("args", args)],
            )
        }
        "Tuple" => {
            let items = pyast::field_list(node, "elements")?;
            let mut converted = Vec::new();
            for item in items {
                converted.push(convert_from_chain_style(nodes, &item)?);
            }
            let items = pyast::pylist(py, converted)?;
            nodes.make("Tuple", &[("elements", items)])
        }
        _ => Ok(node.clone().unbind()),
    }
}

pub fn canonical(py: Python<'_>, tree: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let nodes = Nodes::new(py)?;
    let body = pyast::field_list(tree, "body")?;
    let mut converted = Vec::new();
    for item in body {
        converted.push(convert_to_chain_style(&nodes, &item)?);
    }
    let body = pyast::pylist(py, converted)?;
    nodes.make("Module", &[("body", body)])
}
