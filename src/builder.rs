//! The builder: Viba source spelled with Python operators (was viba/builder.py).
//!
//! The classes in `rs_viba.builder` are the Python face — they hold what the
//! source has, and the operator methods hand each step here. Everything the
//! builder decides lives in this file: turning a Python value into an
//! expression, splicing an exponent, taking an expression back to an AST node,
//! and spelling the file's source.

use std::collections::HashSet;

use pyo3::exceptions::{PySyntaxError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyModule, PyString, PyTuple};

use crate::pyast::{self, Nodes};

/// The keywords a definition name may not be.
const RESERVED: &[&str] = &[
    "true", "false", "nil", "void", "None", "never", "Any", "import", "as", "pattern",
];

/// The builtin containers, with what the source calls each of them: the parser
/// refuses them as definition names and as generic parameters
/// (viba/parser.py: BUILTIN_TYPE_NAMES), so the builder does too.
const BUILTIN_TYPE_NAMES: &[(&str, &str)] = &[
    ("list", "a builtin container"),
    ("set", "a builtin container"),
    ("dict", "a builtin container"),
    ("ListLiteral", "a builtin literal constructor"),
    ("SetLiteral", "a builtin literal constructor"),
    ("DictLiteral", "a builtin literal constructor"),
    ("tagged", "a builtin tag constructor"),
];

/// Python's own types that are the language's names.
const BUILTIN_NAMES: &[(&str, &str)] = &[
    ("str", "str"),
    ("int", "int"),
    ("float", "float"),
    ("bool", "bool"),
    ("list", "list"),
    ("set", "set"),
    ("dict", "dict"),
];

/// The keywords that are not the language's own spelling of a value.
const SLIPS: &[(&str, &str)] = &[("true", "True"), ("false", "False")];

fn builder_module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    PyModule::import(py, "rs_viba.builder")
}

fn builder_class<'py>(py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    builder_module(py)?.getattr(name)
}

fn is_class_of(py: Python<'_>, value: &Bound<'_, PyAny>, name: &str) -> PyResult<bool> {
    value.is_instance(&builder_class(py, name)?)
}

fn is_expr(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<bool> {
    is_class_of(py, value, "_Expr")
}

fn make_expr(py: Python<'_>, name: &str, args: &Bound<'_, PyTuple>) -> PyResult<Py<PyAny>> {
    Ok(builder_class(py, name)?.call1(args)?.unbind())
}

fn expr0(py: Python<'_>, name: &str) -> PyResult<Py<PyAny>> {
    let args = PyTuple::empty(py);
    make_expr(py, name, &args)
}

fn expr1(py: Python<'_>, name: &str, first: Py<PyAny>) -> PyResult<Py<PyAny>> {
    let args = PyTuple::new(py, [first])?;
    make_expr(py, name, &args)
}

fn expr2(py: Python<'_>, name: &str, first: Py<PyAny>, second: Py<PyAny>) -> PyResult<Py<PyAny>> {
    let args = PyTuple::new(py, [first, second])?;
    make_expr(py, name, &args)
}

fn expr_list(py: Python<'_>, name: &str, items: Vec<Py<PyAny>>) -> PyResult<Py<PyAny>> {
    expr1(py, name, pyast::pylist(py, items)?)
}

fn pyrepr(py: Python<'_>, value: &Bound<'_, PyAny>) -> String {
    pyast::repr_of(py, value)
}

fn text_repr(py: Python<'_>, text: &str) -> String {
    pyrepr(py, PyString::new(py, text).as_any())
}

fn is_instance_of_names(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    names: &[&str],
) -> PyResult<bool> {
    let builtins = py.import("builtins")?;
    let mut classes = Vec::new();
    for name in names {
        classes.push(builtins.getattr(*name)?);
    }
    let tuple = PyTuple::new(py, classes)?;
    Ok(builtins
        .getattr("isinstance")?
        .call1((value, tuple))?
        .extract()?)
}

fn builtin_type_name(name: &str) -> Option<&'static str> {
    BUILTIN_TYPE_NAMES
        .iter()
        .find(|(builtin, _)| *builtin == name)
        .map(|(_, what)| *what)
}

// ----------------------------------------------------------------------
// A Python value as an expression (_wrap)
// ----------------------------------------------------------------------

/// A Python value as an expression.
///
/// The builtins are the language's names, `None` is `nil`, `...` is the
/// ellipsis, a tuple is a tuple type, and any other value is a literal.
pub fn wrap(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    if is_expr(py, value)? {
        return Ok(value.clone().unbind());
    }
    let none_type = py.import("types")?.getattr("NoneType")?;
    if value.is_none() || value.is_instance(&none_type)? || value.is(&none_type) {
        return expr0(py, "_Nil");
    }
    let types = py.import("types")?;
    if value.is_instance(&types.getattr("UnionType")?)? {
        // What Python's own `int | str` gives: a sum, branches and all.
        let mut items = Vec::new();
        for argument in value.getattr("__args__")?.try_iter()? {
            items.push(wrap(py, &argument?)?);
        }
        return expr_list(py, "_Sum", items);
    }
    if let Some(applied) = applied(py, value)? {
        return Ok(applied);
    }
    let builtins = py.import("builtins")?;
    if value.is(&builtins.getattr("Ellipsis")?) {
        return expr0(py, "_Ellipsis");
    }
    for (class, text) in BUILTIN_NAMES {
        if value.is(&builtins.getattr(*class)?) {
            return expr1(py, "_Name", pyast::pystr(py, text));
        }
    }
    if is_class_of(py, value, "_Tag")? {
        let spelled = pyrepr(py, value);
        return Err(PyTypeError::new_err(format!(
            "{spelled} is a tag with no body: spell {spelled}(body)"
        )));
    }
    if is_instance_of_names(py, value, &["bool", "int", "float", "str"])? {
        return expr1(py, "_Literal", value.clone().unbind());
    }
    if value.is_instance_of::<PyTuple>() {
        return expr1(py, "_Tuple", value.clone().unbind());
    }
    if value.is_instance_of::<PyList>() {
        return expr2(
            py,
            "_TypeApp",
            pyast::pystr(py, "ListLiteral"),
            value.clone().unbind(),
        );
    }
    if value.is_instance_of::<pyo3::types::PySet>()
        || value.is_instance_of::<pyo3::types::PyFrozenSet>()
    {
        // A set has no order of its own: give the members sorted, so the same
        // set always gives the same source.
        let mut members: Vec<(String, Py<PyAny>)> = Vec::new();
        for member in value.try_iter()? {
            let member = member?;
            let node = ast(py, &wrap(py, &member)?.bind(py))?;
            let key = crate::unparser::unparse_type(py, node.bind(py), 2)?;
            members.push((key, member.unbind()));
        }
        members.sort_by(|left, right| left.0.cmp(&right.0));
        let items = members.into_iter().map(|(_, member)| member).collect();
        return expr2(
            py,
            "_TypeApp",
            pyast::pystr(py, "SetLiteral"),
            pyast::pylist(py, items)?,
        );
    }
    if value.is_instance_of::<pyo3::types::PyDict>() {
        let mut pairs = Vec::new();
        for item in value.call_method0("items")?.try_iter()? {
            let item = item?;
            let pair = item.cast::<PyTuple>()?;
            let key = pair.get_item(0)?.unbind();
            let element = pair.get_item(1)?.unbind();
            pairs.push(PyTuple::new(py, [key, element])?.into_any().unbind());
        }
        return expr2(
            py,
            "_TypeApp",
            pyast::pystr(py, "DictLiteral"),
            pyast::pylist(py, pairs)?,
        );
    }
    Err(PyTypeError::new_err(format!(
        "cannot spell {} as a viba type expression",
        pyrepr(py, value)
    )))
}

/// Python's own applied types: `list[int]` (PEP 585), `typing.List[int]`,
/// `typing.Optional[int]` — an application, or a sum for a union.
fn applied(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Option<Py<PyAny>>> {
    let origin = match value.getattr("__origin__") {
        Ok(origin) if !origin.is_none() => origin,
        _ => return Ok(None),
    };
    let arguments = match value.getattr("__args__") {
        Ok(arguments) if !arguments.is_none() => arguments,
        _ => return Ok(None),
    };
    let typing = py.import("typing")?;
    if origin.is(&typing.getattr("Union")?) {
        let mut items = Vec::new();
        for argument in arguments.try_iter()? {
            items.push(wrap(py, &argument?)?);
        }
        return Ok(Some(expr_list(py, "_Sum", items)?));
    }
    let name = origin_name(py, &origin)?;
    Ok(Some(expr2(
        py,
        "_TypeApp",
        pyast::pystr(py, &name),
        arguments.unbind(),
    )?))
}

/// The name a Python generic hangs off.
fn origin_name(py: Python<'_>, origin: &Bound<'_, PyAny>) -> PyResult<String> {
    let builtins = py.import("builtins")?;
    for (class, text) in BUILTIN_NAMES {
        if origin.is(&builtins.getattr(*class)?) {
            return Ok((*text).to_string());
        }
    }
    if let Ok(name) = origin.getattr("__name__") {
        if let Ok(text) = name.extract::<String>() {
            return Ok(text);
        }
    }
    Err(PyTypeError::new_err(format!(
        "{} has no name to give",
        pyrepr(py, origin)
    )))
}

pub fn elements(py: Python<'_>, values: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let mut items = Vec::new();
    for value in values.try_iter()? {
        items.push(wrap(py, &value?)?);
    }
    pyast::pylist(py, items)
}

// ----------------------------------------------------------------------
// An expression back to an AST node (_ast)
// ----------------------------------------------------------------------

fn chain_ast(py: Python<'_>, node_name: &str, expr: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let nodes = Nodes::new(py)?;
    let mut items = Vec::new();
    for element in expr.getattr("elements")?.try_iter()? {
        items.push(ast(py, &element?)?);
    }
    let items = pyast::pylist(py, items)?;
    nodes.make(node_name, &[("elements", items)])
}

/// An expression as the AST node the source has it as.
pub fn ast(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let nodes = Nodes::new(py)?;
    let name = pyast::class_name(value)?;
    match name.as_str() {
        "_Sum" => chain_ast(py, "SumChain", value),
        "_Product" => chain_ast(py, "ProductChain", value),
        "_Exponent" => chain_ast(py, "ExponentChain", value),
        "_Branch" => ast(py, &value.getattr("inner")?),
        "_Tagged" => {
            let tag = format!("${}", value.getattr("name")?.extract::<String>()?);
            let body = ast(py, &value.getattr("body")?)?;
            nodes.make("Tagged", &[("tag", pyast::pystr(py, &tag)), ("type", body)])
        }
        "_TypeApp" => {
            let constructor: String = value.getattr("constructor")?.extract()?;
            let mut args = Vec::new();
            for argument in value.getattr("args")?.try_iter()? {
                args.push(ast(py, &argument?)?);
            }
            let args = pyast::pylist(py, args)?;
            nodes.make(
                "TypeApp",
                &[
                    ("constructor", pyast::pystr(py, &constructor)),
                    ("args", args),
                ],
            )
        }
        "_Partial" => {
            let function = ast(py, &value.getattr("function")?)?;
            let argument = ast(py, &value.getattr("argument")?)?;
            nodes.make("Partial", &[("function", function), ("argument", argument)])
        }
        "_Literal" => {
            let literal = value.getattr("value")?.unbind();
            nodes.make("Constant", &[("value", literal)])
        }
        "_Tuple" => {
            let mut items = Vec::new();
            for element in value.getattr("elements")?.try_iter()? {
                items.push(ast(py, &element?)?);
            }
            let items = pyast::pylist(py, items)?;
            nodes.make("Tuple", &[("elements", items)])
        }
        "_Nil" => nodes.make0("Nil"),
        "_Ellipsis" => nodes.make0("Ellipsis"),
        "_Hint" => {
            let text = value.getattr("text")?.unbind();
            nodes.make("HintBlock", &[("code", text)])
        }
        "_Name" => name_ast(py, value, &nodes),
        _ => Err(PyTypeError::new_err(format!(
            "cannot spell {} as a viba type expression",
            pyrepr(py, value)
        ))),
    }
}

fn name_ast(py: Python<'_>, value: &Bound<'_, PyAny>, nodes: &Nodes<'_>) -> PyResult<Py<PyAny>> {
    let path: String = value.getattr("path")?.extract()?;
    // `nil` and `never` are keywords: however the source has them, what comes
    // out is the unit itself. The other keywords are slips here — Python has a
    // spelling for them — and are refused.
    match path.as_str() {
        "nil" | "void" | "None" => nodes.make0("Nil"),
        "never" => nodes.make0("Never"),
        "Any" => nodes.make0("Any"),
        _ => {
            if let Some(spelling) = SLIPS
                .iter()
                .find(|(keyword, _)| *keyword == path)
                .map(|(_, spelling)| *spelling)
            {
                return Err(PyTypeError::new_err(format!(
                    "spell {spelling}, not vb.{path}"
                )));
            }
            nodes.make("TypeRef", &[("name", pyast::pystr(py, &path))])
        }
    }
}

// ----------------------------------------------------------------------
// The operators
// ----------------------------------------------------------------------

fn power_argument(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    if is_class_of(py, value, "_Tagged")?
        || is_class_of(py, value, "_Branch")?
        || is_class_of(py, value, "_Exponent")?
    {
        return Ok(value.clone().unbind());
    }
    Err(PyTypeError::new_err(format!(
        "the right side of ** is a tagged field — tag.name(body), or tag(body) for a group — not {}",
        pyrepr(py, value)
    )))
}

fn partial_argument(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    if is_class_of(py, value, "_Tagged")? || is_class_of(py, value, "_Branch")? {
        return Ok(value.clone().unbind());
    }
    Err(PyTypeError::new_err(format!(
        "the right side of << is a tagged field — tag.name(body), or tag(body) for a group — not {}",
        pyrepr(py, value)
    )))
}

/// The source chain: the run on the left, then a right `**` chain's own
/// elements, so that A ** (B ** C) is the flat A <- B <- C.
fn splice(
    py: Python<'_>,
    mut elements: Vec<Py<PyAny>>,
    other: Py<PyAny>,
) -> PyResult<Vec<Py<PyAny>>> {
    let bound = other.bind(py);
    if is_class_of(py, bound, "_Exponent")? {
        for element in bound.getattr("elements")?.try_iter()? {
            elements.push(element?.unbind());
        }
    } else {
        elements.push(other);
    }
    Ok(elements)
}

fn elements_of(_py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Vec<Py<PyAny>>> {
    let mut items = Vec::new();
    for element in value.getattr("elements")?.try_iter()? {
        items.push(element?.unbind());
    }
    Ok(items)
}

pub fn sum_of(
    py: Python<'_>,
    left: &Bound<'_, PyAny>,
    right: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    expr_list(
        py,
        "_Sum",
        vec![left.clone().unbind(), right.clone().unbind()],
    )
}

pub fn sum_append(
    py: Python<'_>,
    chain: &Bound<'_, PyAny>,
    other: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let mut elements = elements_of(py, chain)?;
    elements.push(other.clone().unbind());
    expr_list(py, "_Sum", elements)
}

pub fn product_of(
    py: Python<'_>,
    left: &Bound<'_, PyAny>,
    right: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    expr_list(
        py,
        "_Product",
        vec![left.clone().unbind(), right.clone().unbind()],
    )
}

pub fn product_append(
    py: Python<'_>,
    chain: &Bound<'_, PyAny>,
    other: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let mut elements = elements_of(py, chain)?;
    elements.push(other.clone().unbind());
    expr_list(py, "_Product", elements)
}

pub fn power_of(
    py: Python<'_>,
    base: &Bound<'_, PyAny>,
    other: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let argument = power_argument(py, other)?;
    let spliced = splice(py, vec![base.clone().unbind()], argument)?;
    expr_list(py, "_Exponent", spliced)
}

pub fn exponent_append(
    py: Python<'_>,
    chain: &Bound<'_, PyAny>,
    other: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let argument = power_argument(py, other)?;
    let spliced = splice(py, elements_of(py, chain)?, argument)?;
    expr_list(py, "_Exponent", spliced)
}

pub fn lshift(
    py: Python<'_>,
    function: &Bound<'_, PyAny>,
    other: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let argument = partial_argument(py, other)?;
    expr2(py, "_Partial", function.clone().unbind(), argument)
}

/// `tag(body)` keeps a group whole; `tag(name, body)` is a field.
pub fn tag_call(py: Python<'_>, args: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let length = args.len()?;
    let tuple = args.cast::<PyTuple>()?;
    match length {
        1 => expr1(py, "_Branch", tuple.get_item(0)?.unbind()),
        2 => expr2(
            py,
            "_Tagged",
            tuple.get_item(0)?.unbind(),
            tuple.get_item(1)?.unbind(),
        ),
        _ => Err(PyTypeError::new_err(
            "tag(body) keeps a group whole; tag(name, body) is a field",
        )),
    }
}

// ----------------------------------------------------------------------
// Names, definitions, and the file
// ----------------------------------------------------------------------

fn items(_py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Vec<Py<PyAny>>> {
    // The subscript as a list: `vb.X[a, b]` hands over a tuple.
    if key.is_instance_of::<PyTuple>() {
        let mut out = Vec::new();
        for item in key.try_iter()? {
            out.push(item?.unbind());
        }
        return Ok(out);
    }
    Ok(vec![key.clone().unbind()])
}

fn param(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<String> {
    let name = if is_class_of(py, value, "_Name")? {
        value.getattr("path")?
    } else {
        value.clone()
    };
    if let Ok(text) = name.extract::<String>() {
        if let Some(what) = builtin_type_name(&text) {
            return Err(PyTypeError::new_err(format!(
                "{text} is {what}, not a generic parameter"
            )));
        }
        return Ok(text);
    }
    Err(PyTypeError::new_err(format!(
        "a generic parameter is a name, not {}",
        pyrepr(py, value)
    )))
}

/// The name in the source an application hangs off.
pub fn path(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<String> {
    if is_class_of(py, value, "_Name")? {
        return Ok(value.getattr("path")?.extract()?);
    }
    if let Ok(text) = value.extract::<String>() {
        return Ok(text);
    }
    Err(PyTypeError::new_err(format!(
        "an application needs a name, not {}",
        pyrepr(py, value)
    )))
}

/// A definition's name: one plain name, not a keyword, not a builtin.
pub fn definition_name(py: Python<'_>, name: &str) -> PyResult<String> {
    if let Some(what) = builtin_type_name(name) {
        return Err(PyTypeError::new_err(format!(
            "vb.{name} is {what}, not a definition name"
        )));
    }
    let identifier: bool = PyString::new(py, name)
        .call_method0("isidentifier")?
        .extract()?;
    if identifier && !RESERVED.contains(&name) {
        return Ok(name.to_string());
    }
    Err(PyTypeError::new_err(format!(
        "{} is not a definition name",
        text_repr(py, name)
    )))
}

fn ast_module<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyModule>> {
    PyModule::import(py, "rs_viba.viba_ast")
}

fn parse_source<'py>(py: Python<'py>, source: &str) -> PyResult<Bound<'py, PyAny>> {
    ast_module(py)?.getattr("parse")?.call1((source,))
}

/// The definition names a file already has: appending one is a redefinition.
pub fn existing_names(py: Python<'_>, source: &str) -> PyResult<Py<PyAny>> {
    let builtins = py.import("builtins")?;
    if source.trim().is_empty() {
        return Ok(builtins.getattr("frozenset")?.call0()?.unbind());
    }
    let parsed = match parse_source(py, source) {
        Ok(parsed) => parsed,
        Err(error) => return Err(as_value_error(py, error, "not Viba source")),
    };
    let mut names = Vec::new();
    for node in parsed.getattr("body")?.try_iter()? {
        let node = node?;
        let kind = pyast::class_name(&node)?;
        if kind == "TypeDefinition" || kind == "GenericDefinition" {
            names.push(node.getattr("name")?.unbind());
        }
    }
    Ok(builtins
        .getattr("frozenset")?
        .call1((pyast::pylist(py, names)?,))?
        .unbind())
}

/// A syntax error the source side made is a ValueError on this side, with what
/// the parser said.
fn as_value_error(py: Python<'_>, error: PyErr, what: &str) -> PyErr {
    if error.value(py).is_instance_of::<PySyntaxError>() {
        let message = error
            .value(py)
            .str()
            .map(|text| text.to_string())
            .unwrap_or_default();
        PyValueError::new_err(format!("{what}: {message}"))
    } else {
        error
    }
}

/// The name, unless the file this builder continues already has it.
pub fn free(py: Python<'_>, builder: &Bound<'_, PyAny>, name: &str) -> PyResult<String> {
    let taken = builder.getattr("_taken")?;
    let contains = taken
        .call_method1("__contains__", (PyString::new(py, name),))?
        .is_truthy()?;
    if contains {
        return Err(PyTypeError::new_err(format!(
            "{} is already defined: only new definitions can be appended",
            text_repr(py, name)
        )));
    }
    Ok(name.to_string())
}

fn add_node(builder: &Bound<'_, PyAny>, node: Py<PyAny>) -> PyResult<()> {
    builder.call_method1("_add", (node,))?;
    Ok(())
}

/// `vb.Name = body` spells `Name = body`.
pub fn define(
    py: Python<'_>,
    builder: &Bound<'_, PyAny>,
    name: &str,
    value: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let name = definition_name(py, name)?;
    let name = free(py, builder, &name)?;
    let body = ast(py, wrap(py, value)?.bind(py))?;
    let nodes = Nodes::new(py)?;
    let node = nodes.make(
        "TypeDefinition",
        &[("name", pyast::pystr(py, &name)), ("body", body)],
    )?;
    add_node(builder, node)
}

/// `vb.Name[T] = body` spells `Name[T] = body`.
pub fn define_generic(
    py: Python<'_>,
    name_expr: &Bound<'_, PyAny>,
    key: &Bound<'_, PyAny>,
    body: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let slot_path: String = name_expr.getattr("path")?.extract()?;
    let owner = name_expr.getattr("owner")?;
    if owner.is_none() {
        return Err(PyTypeError::new_err(format!(
            "{} is a name, not a definition slot",
            text_repr(py, &slot_path)
        )));
    }
    let mut params = Vec::new();
    for item in items(py, key)? {
        params.push(param(py, item.bind(py))?);
    }
    let unique: HashSet<&String> = params.iter().collect();
    if unique.len() != params.len() {
        let mut given = Vec::new();
        for name in &params {
            given.push(pyast::pystr(py, name));
        }
        return Err(PyTypeError::new_err(format!(
            "the parameters of {} repeat: {}",
            slot_path,
            pyrepr(py, pyast::pylist(py, given)?.bind(py))
        )));
    }
    let name = definition_name(py, &slot_path)?;
    let name = free(py, &owner, &name)?;
    let body = ast(py, wrap(py, body)?.bind(py))?;
    let mut given = Vec::new();
    for param in &params {
        given.push(pyast::pystr(py, param));
    }
    let nodes = Nodes::new(py)?;
    let node = nodes.make(
        "GenericDefinition",
        &[
            ("name", pyast::pystr(py, &name)),
            ("generic_params", pyast::pylist(py, given)?),
            ("body", body),
        ],
    )?;
    add_node(&owner, node)
}

/// The whole file: what it started from, then the new definitions.
pub fn source(py: Python<'_>, builder: &Bound<'_, PyAny>) -> PyResult<String> {
    let existing: String = builder.getattr("_existing")?.extract()?;
    let body = builder.getattr("_body")?;
    let asts = ast_module(py)?;
    let import_class = asts.getattr("Import")?;
    let comment_class = builder_class(py, "_Comment")?;
    let module_class = asts.getattr("Module")?;
    let unparse = asts.getattr("unparse")?;
    let mut imports: Vec<Py<PyAny>> = Vec::new();
    let mut rest: Vec<Py<PyAny>> = Vec::new();
    // A file's imports stand first, however late they were added.
    for item in body.try_iter()? {
        let item = item?;
        if item.is_instance(&import_class)? {
            imports.push(item.unbind());
        } else {
            rest.push(item.unbind());
        }
    }
    let mut parts: Vec<String> = Vec::new();
    if !existing.is_empty() {
        parts.push(existing);
    }
    for item in imports.into_iter().chain(rest) {
        let item = item.bind(py);
        if item.is_instance(&comment_class)? {
            let text = comment_class
                .getattr("text")?
                .call1((item,))?
                .extract::<String>()?;
            parts.push(text);
        } else {
            let one = pyast::pylist(py, vec![item.clone().unbind()])?;
            let module = module_class.call1((one,))?;
            parts.push(unparse.call1((module,))?.extract()?);
        }
    }
    if parts.is_empty() {
        return Ok(String::new());
    }
    Ok(format!("{}\n", parts.join("\n\n")))
}

// ----------------------------------------------------------------------
// The free helpers
// ----------------------------------------------------------------------

/// `Name` in the source, from the text of its path.
pub fn name_text(py: Python<'_>, text: &str) -> PyResult<Py<PyAny>> {
    expr1(py, "_Name", pyast::pystr(py, text))
}

/// `Constructor[args]`: an application as the builder spells it.
pub fn type_app(py: Python<'_>, constructor: &str, args: Vec<Py<PyAny>>) -> PyResult<Py<PyAny>> {
    expr2(
        py,
        "_TypeApp",
        pyast::pystr(py, constructor),
        pyast::pylist(py, args)?,
    )
}

pub fn hint_expr(py: Python<'_>, text: &str) -> PyResult<Py<PyAny>> {
    expr1(py, "_Hint", pyast::pystr(py, text))
}

pub fn name_expr(py: Python<'_>, path: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let text: String = match path.extract() {
        Ok(text) => text,
        Err(_) => return Err(not_a_name(py, path)),
    };
    if text.is_empty() {
        return Err(not_a_name(py, path));
    }
    expr1(py, "_Name", pyast::pystr(py, &text))
}

fn not_a_name(py: Python<'_>, path: &Bound<'_, PyAny>) -> PyErr {
    PyTypeError::new_err(format!(
        "a name in the source is a non-empty str, not {}",
        pyrepr(py, path)
    ))
}

pub fn name_attr(py: Python<'_>, name_expr: &Bound<'_, PyAny>, attr: &str) -> PyResult<Py<PyAny>> {
    let path: String = name_expr.getattr("path")?.extract()?;
    let owner = name_expr.getattr("owner")?;
    expr2(
        py,
        "_Name",
        pyast::pystr(py, &format!("{path}.{attr}")),
        owner.unbind(),
    )
}

pub fn name_setattr(
    _py: Python<'_>,
    name_expr: &Bound<'_, PyAny>,
    attr: &str,
    _body: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let path: String = name_expr.getattr("path")?.extract()?;
    Err(PyTypeError::new_err(format!(
        "a definition's name is one name: give vb.{attr} = ..., not vb.{path}.{attr} = ..."
    )))
}

pub fn slot(py: Python<'_>, builder: &Bound<'_, PyAny>, name: &str) -> PyResult<Py<PyAny>> {
    expr2(
        py,
        "_Name",
        pyast::pystr(py, name),
        builder.clone().unbind(),
    )
}

pub fn branch(py: Python<'_>, expr: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    expr1(py, "_Branch", expr.clone().unbind())
}

pub fn tagged(
    py: Python<'_>,
    name: &Bound<'_, PyAny>,
    body: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    expr2(py, "_Tagged", name.clone().unbind(), body.clone().unbind())
}

pub fn comment_body(text: &str) -> String {
    text.trim().trim_start_matches('#').trim().to_string()
}

pub fn add_import(
    py: Python<'_>,
    builder: &Bound<'_, PyAny>,
    module: &str,
    alias: Option<&str>,
) -> PyResult<Py<PyAny>> {
    if builder.getattr("_existing")?.is_truthy()? {
        return Err(PyTypeError::new_err(
            "this builder continues an existing file: an import would have to stand above it",
        ));
    }
    let nodes = Nodes::new(py)?;
    let alias = match alias {
        Some(alias) => pyast::pystr(py, alias),
        None => py.None(),
    };
    let node = nodes.make(
        "Import",
        &[("module", pyast::pystr(py, module)), ("alias", alias)],
    )?;
    add_node(builder, node)?;
    Ok(builder.clone().unbind())
}

pub fn comment(py: Python<'_>, builder: &Bound<'_, PyAny>, text: &str) -> PyResult<Py<PyAny>> {
    let node = builder_class(py, "_Comment")?.call1((text,))?.unbind();
    add_node(builder, node)?;
    Ok(builder.clone().unbind())
}

pub fn check(py: Python<'_>, builder: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let source = builder.str()?.extract::<String>()?;
    match parse_source(py, &source) {
        Ok(tree) => Ok(tree.unbind()),
        Err(error) => Err(as_value_error(py, error, "the source form does not parse")),
    }
}

// ----------------------------------------------------------------------
// The Python functions (_native)
// ----------------------------------------------------------------------

#[pyfunction]
fn builder_wrap_py(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    wrap(py, value)
}

#[pyfunction]
fn builder_elements_py(py: Python<'_>, values: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    elements(py, values)
}

#[pyfunction]
fn builder_ast_py(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    ast(py, value)
}

#[pyfunction]
fn builder_sum_py(
    py: Python<'_>,
    left: &Bound<'_, PyAny>,
    right: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    sum_of(py, left, right)
}

#[pyfunction]
fn builder_sum_append_py(
    py: Python<'_>,
    chain: &Bound<'_, PyAny>,
    other: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    sum_append(py, chain, other)
}

#[pyfunction]
fn builder_product_py(
    py: Python<'_>,
    left: &Bound<'_, PyAny>,
    right: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    product_of(py, left, right)
}

#[pyfunction]
fn builder_product_append_py(
    py: Python<'_>,
    chain: &Bound<'_, PyAny>,
    other: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    product_append(py, chain, other)
}

#[pyfunction]
fn builder_power_py(
    py: Python<'_>,
    base: &Bound<'_, PyAny>,
    other: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    power_of(py, base, other)
}

#[pyfunction]
fn builder_rpower_py(
    py: Python<'_>,
    base: &Bound<'_, PyAny>,
    other: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    power_of(py, base, other)
}

#[pyfunction]
fn builder_exponent_append_py(
    py: Python<'_>,
    chain: &Bound<'_, PyAny>,
    other: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    exponent_append(py, chain, other)
}

#[pyfunction]
fn builder_lshift_py(
    py: Python<'_>,
    function: &Bound<'_, PyAny>,
    other: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    lshift(py, function, other)
}

#[pyfunction]
fn builder_tag_call_py(py: Python<'_>, args: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    tag_call(py, args)
}

#[pyfunction]
fn builder_branch_py(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    branch(py, value)
}

#[pyfunction]
fn builder_tagged_py(
    py: Python<'_>,
    name: &Bound<'_, PyAny>,
    body: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    tagged(py, name, body)
}

#[pyfunction]
fn builder_items_py(py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let items = items(py, key)?;
    pyast::pylist(py, items)
}

#[pyfunction]
fn builder_define_py(
    py: Python<'_>,
    builder: &Bound<'_, PyAny>,
    name: &str,
    value: &Bound<'_, PyAny>,
) -> PyResult<()> {
    define(py, builder, name, value)
}

#[pyfunction]
fn builder_define_generic_py(
    py: Python<'_>,
    name_expr: &Bound<'_, PyAny>,
    key: &Bound<'_, PyAny>,
    body: &Bound<'_, PyAny>,
) -> PyResult<()> {
    define_generic(py, name_expr, key, body)
}

#[pyfunction]
fn builder_source_py(py: Python<'_>, builder: &Bound<'_, PyAny>) -> PyResult<String> {
    source(py, builder)
}

#[pyfunction]
fn builder_check_py(py: Python<'_>, builder: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    check(py, builder)
}

#[pyfunction]
fn builder_existing_names_py(py: Python<'_>, source: &str) -> PyResult<Py<PyAny>> {
    existing_names(py, source)
}

#[pyfunction]
fn builder_free_py(py: Python<'_>, builder: &Bound<'_, PyAny>, name: &str) -> PyResult<String> {
    free(py, builder, name)
}

#[pyfunction]
fn builder_definition_name_py(py: Python<'_>, name: &str) -> PyResult<String> {
    definition_name(py, name)
}

#[pyfunction]
fn builder_param_py(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<String> {
    param(py, value)
}

#[pyfunction]
fn builder_path_py(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<String> {
    path(py, value)
}

#[pyfunction]
fn builder_origin_name_py(py: Python<'_>, origin: &Bound<'_, PyAny>) -> PyResult<String> {
    origin_name(py, origin)
}

#[pyfunction]
fn builder_literal_py(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    wrap(py, value)
}

#[pyfunction]
fn builder_hint_py(py: Python<'_>, text: &str) -> PyResult<Py<PyAny>> {
    hint_expr(py, text)
}

#[pyfunction]
fn builder_name_py(py: Python<'_>, path: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    name_expr(py, path)
}

#[pyfunction]
fn builder_slot_py(py: Python<'_>, builder: &Bound<'_, PyAny>, name: &str) -> PyResult<Py<PyAny>> {
    slot(py, builder, name)
}

#[pyfunction]
fn builder_name_attr_py(
    py: Python<'_>,
    name_expr: &Bound<'_, PyAny>,
    attr: &str,
) -> PyResult<Py<PyAny>> {
    name_attr(py, name_expr, attr)
}

#[pyfunction]
fn builder_name_setattr_py(
    py: Python<'_>,
    name_expr: &Bound<'_, PyAny>,
    attr: &str,
    body: &Bound<'_, PyAny>,
) -> PyResult<()> {
    name_setattr(py, name_expr, attr, body)
}

#[pyfunction]
fn builder_comment_body_py(text: &str) -> String {
    comment_body(text)
}

#[pyfunction(signature = (builder, module, alias = None))]
fn builder_add_import_py(
    py: Python<'_>,
    builder: &Bound<'_, PyAny>,
    module: &str,
    alias: Option<&str>,
) -> PyResult<Py<PyAny>> {
    add_import(py, builder, module, alias)
}

#[pyfunction]
fn builder_comment_py(
    py: Python<'_>,
    builder: &Bound<'_, PyAny>,
    text: &str,
) -> PyResult<Py<PyAny>> {
    comment(py, builder, text)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(builder_wrap_py, m)?)?;
    m.add_function(wrap_pyfunction!(builder_elements_py, m)?)?;
    m.add_function(wrap_pyfunction!(builder_ast_py, m)?)?;
    m.add_function(wrap_pyfunction!(builder_sum_py, m)?)?;
    m.add_function(wrap_pyfunction!(builder_sum_append_py, m)?)?;
    m.add_function(wrap_pyfunction!(builder_product_py, m)?)?;
    m.add_function(wrap_pyfunction!(builder_product_append_py, m)?)?;
    m.add_function(wrap_pyfunction!(builder_power_py, m)?)?;
    m.add_function(wrap_pyfunction!(builder_rpower_py, m)?)?;
    m.add_function(wrap_pyfunction!(builder_exponent_append_py, m)?)?;
    m.add_function(wrap_pyfunction!(builder_lshift_py, m)?)?;
    m.add_function(wrap_pyfunction!(builder_tag_call_py, m)?)?;
    m.add_function(wrap_pyfunction!(builder_branch_py, m)?)?;
    m.add_function(wrap_pyfunction!(builder_tagged_py, m)?)?;
    m.add_function(wrap_pyfunction!(builder_items_py, m)?)?;
    m.add_function(wrap_pyfunction!(builder_define_py, m)?)?;
    m.add_function(wrap_pyfunction!(builder_define_generic_py, m)?)?;
    m.add_function(wrap_pyfunction!(builder_source_py, m)?)?;
    m.add_function(wrap_pyfunction!(builder_check_py, m)?)?;
    m.add_function(wrap_pyfunction!(builder_existing_names_py, m)?)?;
    m.add_function(wrap_pyfunction!(builder_free_py, m)?)?;
    m.add_function(wrap_pyfunction!(builder_definition_name_py, m)?)?;
    m.add_function(wrap_pyfunction!(builder_param_py, m)?)?;
    m.add_function(wrap_pyfunction!(builder_path_py, m)?)?;
    m.add_function(wrap_pyfunction!(builder_origin_name_py, m)?)?;
    m.add_function(wrap_pyfunction!(builder_literal_py, m)?)?;
    m.add_function(wrap_pyfunction!(builder_hint_py, m)?)?;
    m.add_function(wrap_pyfunction!(builder_name_py, m)?)?;
    m.add_function(wrap_pyfunction!(builder_slot_py, m)?)?;
    m.add_function(wrap_pyfunction!(builder_name_attr_py, m)?)?;
    m.add_function(wrap_pyfunction!(builder_name_setattr_py, m)?)?;
    m.add_function(wrap_pyfunction!(builder_comment_body_py, m)?)?;
    m.add_function(wrap_pyfunction!(builder_add_import_py, m)?)?;
    m.add_function(wrap_pyfunction!(builder_comment_py, m)?)?;
    Ok(())
}
