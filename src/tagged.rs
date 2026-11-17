//! `tagged[...]`: a tag given as a symbol (was viba/viba_ast/tagged.py).

use pyo3::prelude::*;

use crate::ctx::Ctx;

pub const TAGGED_NAME: &str = "tagged";
pub const SELF_NAME: &str = "__type__";
pub const GET_ATTR_TAG: &str = "$get_attr";
pub const GET_ITEM_TAG: &str = "$get_item";
pub const IN_TAG: &str = "$in";
pub const LEN_TAG: &str = "$len";
pub const KEYS_TAG: &str = "$keys";
pub const FIRST_CASE_INDEX_TAG: &str = "$first_case_index";

pub const INTERPRETER_MEMBER_TAGS: &[&str] = &[
    GET_ATTR_TAG,
    GET_ITEM_TAG,
    IN_TAG,
    LEN_TAG,
    KEYS_TAG,
    FIRST_CASE_INDEX_TAG,
];

pub fn symbol_of_text(text: &str) -> Option<String> {
    let name = text.strip_prefix('$').unwrap_or(text);
    if name.is_empty() {
        return None;
    }
    let mut chars = name.chars();
    let first = chars.next()?;
    if !(first.is_alphabetic() || first == '_') {
        return None;
    }
    if !name.chars().all(|c| c.is_alphanumeric() || c == '_') {
        return None;
    }
    Some(name.to_string())
}

pub fn symbol_of(value: &Bound<'_, PyAny>) -> Option<String> {
    let text: String = value.extract().ok()?;
    symbol_of_text(&text)
}

pub fn symbol_problem(value: &Bound<'_, PyAny>) -> Option<String> {
    if symbol_of(value).is_none() {
        Some(format!(
            "{} is no symbol for a tag",
            crate::pyast::repr_of(value.py(), value)
        ))
    } else {
        None
    }
}

pub fn tag_of(name: &str) -> String {
    format!("${}", name)
}

pub fn string_literal(ctx: &Ctx<'_>, node: &Bound<'_, PyAny>) -> PyResult<bool> {
    if !ctx.is(node, "Constant")? {
        return Ok(false);
    }
    let value = node.getattr("value")?;
    Ok(value.is_instance_of::<pyo3::types::PyString>())
}

pub fn literal_symbol(ctx: &Ctx<'_>, node: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
    if !string_literal(ctx, node)? {
        return Ok(None);
    }
    Ok(symbol_of(&node.getattr("value")?))
}

pub fn tagged_node(ctx: &Ctx<'_>, symbol: &str, arguments: &[Py<PyAny>]) -> PyResult<Py<PyAny>> {
    let tag = tag_of(symbol);
    if arguments.len() == 1 {
        return ctx.make("Member", &[("tag", ctx.pystr(&tag))]);
    }
    ctx.make(
        "Tagged",
        &[
            ("tag", ctx.pystr(&tag)),
            ("type", arguments[1].clone_ref(ctx.py)),
        ],
    )
}

pub fn tagged_problem(ctx: &Ctx<'_>, arguments: &[Bound<'_, PyAny>]) -> PyResult<Option<String>> {
    if arguments.len() != 1 && arguments.len() != 2 {
        return Ok(Some(format!(
            "{} takes one argument (the symbol) or two (the symbol and the type it marks), not {}",
            TAGGED_NAME,
            arguments.len()
        )));
    }
    let first = &arguments[0];
    if ctx.is(first, "Constant")? {
        let value = first.getattr("value")?;
        return Ok(symbol_problem(&value));
    }
    Ok(None)
}

fn member_chain_source(tag: &str) -> Option<&'static str> {
    match tag {
        GET_ATTR_TAG => Some("Any <- $env Env <- $value Any <- $name str"),
        GET_ITEM_TAG => Some("Any <- $env Env <- $container Any <- $address Any"),
        IN_TAG => Some("bool <- $env Env <- $container Any <- $piece Any"),
        LEN_TAG => Some("int <- $env Env <- $container Any"),
        KEYS_TAG => Some("list[str] <- $env Env <- $table Any"),
        FIRST_CASE_INDEX_TAG => {
            Some("(int | DispatchError) <- $env Env <- $args Any <- $func_arg_types list[Any]")
        }
        _ => None,
    }
}

pub fn interpreter_member_chain(ctx: &Ctx<'_>, tag: &str) -> PyResult<Option<Py<PyAny>>> {
    let source = match member_chain_source(tag) {
        Some(source) => source,
        None => return Ok(None),
    };
    let program = crate::parser::parse(ctx.py, &format!("__decl__ =\n  {}\n", source))?;
    let body = crate::pyast::field_list(program.bind(ctx.py), "body")?;
    let definition = &body[0];
    Ok(Some(definition.getattr("body")?.unbind()))
}
