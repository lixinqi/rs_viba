//! Serialize: a viba data back into viba source (was viba/serialize.py).
//!
//! Every value is asked for through the protocol's own cells — the same ones
//! that take values — so this knows nothing about a design beyond what the
//! source has. What comes out is canonical viba source: the walking builds
//! `rs_viba.builder` expressions, and `builder.check` parses the result back
//! before it is handed over.

use std::collections::HashSet;

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyFloat, PyInt, PyModule, PyString, PyTuple};

use crate::builder;
use crate::pyast;
use crate::reflect;
use crate::type_system;

/// A piece the design has no spelling for, or the Python error the builder
/// raised while spelling it.
enum SerializeError {
    Gap(String),
    Python(PyErr),
}

type SResult<T> = Result<T, SerializeError>;

impl From<PyErr> for SerializeError {
    fn from(error: PyErr) -> Self {
        SerializeError::Python(error)
    }
}

fn gap<T>(message: impl Into<String>) -> SResult<T> {
    Err(SerializeError::Gap(message.into()))
}

fn module<'py>(py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyModule>> {
    PyModule::import(py, name)
}

fn vtd(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    module(py, "rs_viba.viba_type_descriptor")
}

fn reflect_module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    module(py, "rs_viba.reflect")
}

fn builder_module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    module(py, "rs_viba.builder")
}

fn is_class(node: &Bound<'_, PyAny>, name: &str) -> PyResult<bool> {
    Ok(pyast::class_name(node)? == name)
}

fn pyrepr(py: Python<'_>, value: &Bound<'_, PyAny>) -> String {
    pyast::repr_of(py, value)
}

fn text_repr(py: Python<'_>, text: &str) -> String {
    pyrepr(py, PyString::new(py, text).as_any())
}

fn attr<'py>(value: &Bound<'py, PyAny>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    value.getattr(name)
}

fn opt_attr<'py>(value: &Bound<'py, PyAny>, name: &str) -> Option<Bound<'py, PyAny>> {
    match value.getattr(name) {
        Ok(given) if !given.is_none() => Some(given),
        _ => None,
    }
}

fn descriptor_kind(descriptor: &Bound<'_, PyAny>) -> PyResult<String> {
    descriptor.getattr("kind")?.extract()
}

fn payload<'py>(descriptor: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    attr(descriptor, "payload")
}

fn payload_list(descriptor: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<Py<PyAny>>> {
    let mut out = Vec::new();
    for item in payload(descriptor)?.getattr(name)?.try_iter()? {
        out.push(item?.unbind());
    }
    Ok(out)
}

fn result_is_err(value: &Bound<'_, PyAny>) -> PyResult<bool> {
    is_class(value, "Err")
}

fn result_ok_value<'py>(value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    value.getattr("ok_value")
}

fn result_err_message(value: &Bound<'_, PyAny>) -> PyResult<String> {
    value.getattr("msg")?.extract()
}

fn step_repr(py: Python<'_>, step: &Bound<'_, PyAny>) -> String {
    pyrepr(py, step)
}

/// A step of the source's address: `at_index(i)` / `at_key(k)`.
fn step_of(py: Python<'_>, kind: &str, value: &Bound<'_, PyAny>) -> SResult<Py<PyAny>> {
    let function = reflect_module(py)?.getattr(kind)?;
    Ok(function.call1((value.clone(),))?.unbind())
}

// --- the accessor's cells, as the source spells them ---------------------

fn unfold(
    py: Python<'_>,
    access: &Bound<'_, PyAny>,
    descriptor: &Bound<'_, PyAny>,
) -> SResult<Py<PyAny>> {
    let _ = py;
    Ok(access
        .call_method1("unfold", (descriptor.clone(),))?
        .unbind())
}

fn inline_cycle(
    py: Python<'_>,
    access: &Bound<'_, PyAny>,
    descriptor: &Bound<'_, PyAny>,
) -> SResult<Option<String>> {
    let _ = py;
    Ok(access
        .call_method1("inline_cycle", (descriptor.clone(),))?
        .extract()?)
}

fn member_steps(
    py: Python<'_>,
    access: &Bound<'_, PyAny>,
    node: &Bound<'_, PyAny>,
) -> SResult<Vec<(Option<String>, Py<PyAny>, Py<PyAny>)>> {
    let _ = py;
    let steps = access.call_method1("member_steps", (node.clone(),))?;
    let mut out = Vec::new();
    for step in steps.try_iter()? {
        let step = step?;
        let tag = step.get_item(0)?;
        let tag = if tag.is_none() {
            None
        } else {
            Some(tag.extract::<String>()?)
        };
        out.push((tag, step.get_item(1)?.unbind(), step.get_item(2)?.unbind()));
    }
    Ok(out)
}

fn may_be_nil(
    py: Python<'_>,
    access: &Bound<'_, PyAny>,
    step: &Bound<'_, PyAny>,
    node: &Bound<'_, PyAny>,
) -> SResult<bool> {
    let _ = py;
    let members = access.call_method1("members", (node.clone(),))?;
    let target = access.call_method1("target_of", (node.clone(), step.clone(), members))?;
    if target.is_none() {
        return Ok(false);
    }
    Ok(access.call_method1("carries_nil", (target,))?.extract()?)
}

fn is_unit_descriptor(
    py: Python<'_>,
    access: &Bound<'_, PyAny>,
    descriptor: &Bound<'_, PyAny>,
) -> SResult<bool> {
    let _ = py;
    Ok(access
        .call_method1("_is_unit_descriptor", (descriptor.clone(),))?
        .extract()?)
}

fn get(
    py: Python<'_>,
    access: &Bound<'_, PyAny>,
    node: &Bound<'_, PyAny>,
    step: &Bound<'_, PyAny>,
) -> SResult<Py<PyAny>> {
    let _ = py;
    Ok(access
        .call_method1("get", (node.clone(), step.clone()))?
        .unbind())
}

fn leaf(py: Python<'_>, access: &Bound<'_, PyAny>, node: &Bound<'_, PyAny>) -> SResult<Py<PyAny>> {
    let _ = py;
    Ok(access.call_method1("leaf", (node.clone(),))?.unbind())
}

fn child(
    py: Python<'_>,
    access: &Bound<'_, PyAny>,
    node: &Bound<'_, PyAny>,
    step: &Bound<'_, PyAny>,
) -> SResult<Py<PyAny>> {
    let given = get(py, access, node, step)?;
    let given = given.bind(py);
    if result_is_err(given)? {
        return gap(format!("this step has no value: {}", step_repr(py, step)));
    }
    let value = result_ok_value(given)?;
    if value.is_none() {
        return gap(format!("this step has no value: {}", step_repr(py, step)));
    }
    Ok(value.unbind())
}

fn container_kind(
    py: Python<'_>,
    access: &Bound<'_, PyAny>,
    descriptor: &Bound<'_, PyAny>,
) -> SResult<Option<String>> {
    let _ = py;
    Ok(access
        .call_method1("container_kind", (descriptor.clone(),))?
        .extract()?)
}

// --- the sources the serializer builds -----------------------------------

fn source_module_of<'py>(
    _py: Python<'py>,
    node: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let descriptor = attr(node, "descriptor")?;
    let resolvable = match opt_attr(&descriptor, "resolvable_type") {
        Some(resolvable) => resolvable,
        None => return Ok(None),
    };
    Ok(opt_attr(&resolvable, "container_module"))
}

fn bare<'py>(
    py: Python<'py>,
    piece: &Bound<'py, PyAny>,
    access: &Bound<'py, PyAny>,
    source_module: Option<&Bound<'py, PyAny>>,
) -> SResult<Py<PyAny>> {
    let source_module = match source_module {
        Some(module) => module.clone(),
        None => py.None().into_bound(py),
    };
    let node_type = reflect::ast_node_type(py, piece, &source_module)?;
    let descriptor = vtd(py)?.getattr("descriptor_of")?.call1((node_type,))?;
    let object = reflect_module(py)?.getattr("VObject")?.call1((
        access.clone(),
        descriptor,
        piece.clone(),
    ))?;
    Ok(object.unbind())
}

fn call_parts(py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<(Py<PyAny>, Vec<Py<PyAny>>)> {
    let mut arguments = Vec::new();
    let mut current = data.clone().unbind();
    loop {
        let bound = current.clone_ref(py).into_bound(py);
        if !is_class(&bound, "Partial")? {
            arguments.reverse();
            return Ok((bound.unbind(), arguments));
        }
        arguments.push(attr(&bound, "argument")?.unbind());
        current = attr(&bound, "function")?.unbind();
    }
}

fn product_head(py: Python<'_>) -> SResult<Py<PyAny>> {
    Ok(builder::name_text(py, "Object")?)
}

fn unit_name(py: Python<'_>, name: &str) -> SResult<Py<PyAny>> {
    Ok(builder::name_text(py, name)?)
}

fn tag_expr(py: Python<'_>, tag: &str, value: &Bound<'_, PyAny>) -> SResult<Py<PyAny>> {
    Ok(builder::tagged(py, PyString::new(py, tag).as_any(), value)?)
}

fn argument(py: Python<'_>, inner: &Bound<'_, PyAny>, tag: Option<&str>) -> SResult<Py<PyAny>> {
    match tag {
        None => Ok(builder::branch(py, inner)?),
        Some(tag) => tag_expr(py, tag, inner),
    }
}

/// The literal a leaf spells: nil, or one of the four literals.
fn emit_leaf(py: Python<'_>, value: &Bound<'_, PyAny>) -> SResult<Py<PyAny>> {
    if value.is_none() {
        return Ok(py.None());
    }
    if value.is_instance_of::<PyBool>()
        || value.is_instance_of::<PyInt>()
        || value.is_instance_of::<PyFloat>()
        || value.is_instance_of::<PyString>()
    {
        return Ok(value.clone().unbind());
    }
    gap(format!(
        "cannot serialize this leaf out: {}",
        value.get_type().name()?
    ))
}

// --- one piece at a time -------------------------------------------------

fn emit(py: Python<'_>, access: &Bound<'_, PyAny>, node: &Bound<'_, PyAny>) -> SResult<Py<PyAny>> {
    let data = attr(node, "data")?;
    if is_class(&data, "Partial")? {
        return emit_closure(py, access, node);
    }
    let descriptor = attr(node, "descriptor")?;
    let unfolded = unfold(py, access, &descriptor)?;
    let unfolded = unfolded.bind(py);
    let unfolded_kind = descriptor_kind(unfolded)?;
    if unfolded_kind == reflect::ANY {
        // `Any` is the top type: it says nothing about this piece, and no member
        // under it can be walked. What the piece is has to come from the piece.
        let source_module = source_module_of(py, node)?;
        let as_itself = bare(py, &data, access, source_module.as_ref())?;
        let descriptor = attr(as_itself.bind(py), "descriptor")?;
        let unfolded = unfold(py, access, &descriptor)?;
        if descriptor_kind(unfolded.bind(py))? == reflect::ANY {
            return gap(format!("this piece is {unfolded_kind} itself, not a value"));
        }
        return emit(py, access, as_itself.bind(py));
    }
    if is_class(&data, "TypeRef")? && unfolded_kind == reflect::EXPONENT {
        // A name the design calls a function — a call that wants its arguments
        // still. Given as the name, it parses back as that same call.
        return emit_closure(py, access, node);
    }
    if unfolded_kind == reflect::NEVER {
        return gap("nothing resides in never");
    }
    if unfolded_kind == reflect::PRODUCT {
        if let Some(cycle) = inline_cycle(py, access, unfolded)? {
            return gap(format!(
                "the inline chain comes back to {}",
                text_repr(py, &cycle)
            ));
        }
    }
    let given = leaf(py, access, node)?;
    if !result_is_err(given.bind(py))? {
        let value = result_ok_value(given.bind(py))?;
        return emit_leaf(py, &value);
    }
    if unfolded_kind == reflect::SUM {
        return emit_sum(py, access, node);
    }
    if unfolded_kind == reflect::PRODUCT {
        return emit_product(py, access, node);
    }
    if unfolded_kind == reflect::TUPLE {
        return emit_tuple(py, access, node);
    }
    if let Some(container) = container_kind(py, access, unfolded)? {
        return emit_container(py, access, node, &container, unfolded);
    }
    if unfolded_kind == reflect::TAGGED {
        return emit_tagged(py, access, node);
    }
    if unfolded_kind == reflect::EXPONENT {
        return emit_exponent(py, access, node, unfolded);
    }
    if unfolded_kind == reflect::HINT_BLOCK {
        // `{ ... }`: opaque, and the viba data's own text is not reachable
        // through the protocol. What cannot be taken is not invented.
        return Ok(py.None());
    }
    if unfolded_kind == reflect::NIL {
        return Ok(py.None());
    }
    gap(format!("cannot serialize this piece out: {unfolded_kind}"))
}

fn emit_closure(
    py: Python<'_>,
    access: &Bound<'_, PyAny>,
    node: &Bound<'_, PyAny>,
) -> SResult<Py<PyAny>> {
    let data = attr(node, "data")?;
    let (head, arguments) = call_parts(py, &data)?;
    let head = head.bind(py);
    if !is_class(head, "TypeRef")? {
        return gap("a chain headed by a member has no form in the source here");
    }
    let name = attr(head, "name")?.extract::<String>()?;
    let mut expression = builder::name_text(py, &name)?;
    let source_module = source_module_of(py, node)?;
    for argument in arguments {
        let argument = argument.bind(py);
        let piece = bare(py, argument, access, source_module.as_ref())?;
        let piece = emit(py, access, piece.bind(py))?;
        let piece = if is_class(argument, "Tagged")? {
            piece
        } else {
            builder::branch(py, piece.bind(py))?
        };
        expression = builder::lshift(py, expression.bind(py), piece.bind(py))?;
    }
    Ok(expression)
}

fn emit_sum(
    py: Python<'_>,
    access: &Bound<'_, PyAny>,
    node: &Bound<'_, PyAny>,
) -> SResult<Py<PyAny>> {
    for (tag, step, _descriptor) in member_steps(py, access, node)? {
        let given = get(py, access, node, step.bind(py))?;
        let given = given.bind(py);
        if result_is_err(given)? {
            continue;
        }
        let value = result_ok_value(given)?;
        if value.is_none() {
            continue;
        }
        let inner = emit(py, access, &value)?;
        return match tag {
            None => Ok(inner),
            Some(tag) => tag_expr(py, &tag, inner.bind(py)),
        };
    }
    gap(format!(
        "no branch of this sum carries a value: {}",
        pyrepr(py, node)
    ))
}

fn emit_product(
    py: Python<'_>,
    access: &Bound<'_, PyAny>,
    node: &Bound<'_, PyAny>,
) -> SResult<Py<PyAny>> {
    let mut parts: Vec<Py<PyAny>> = vec![product_head(py)?];
    let mut seen_tags: HashSet<String> = HashSet::new();
    for (tag, step, descriptor) in member_steps(py, access, node)? {
        if let Some(tag) = &tag {
            if !seen_tags.insert(tag.clone()) {
                return gap(format!("the tag {tag} appears twice in one product"));
            }
        }
        let unit = is_unit_descriptor(py, access, descriptor.bind(py))?;
        let unfolded = unfold(py, access, descriptor.bind(py))?;
        if unit && descriptor_kind(unfolded.bind(py))? != reflect::NEVER {
            let unit = py.None();
            parts.push(match &tag {
                None => unit,
                Some(tag) => tag_expr(py, tag, unit.bind(py))?,
            });
            continue;
        }
        let given = get(py, access, node, step.bind(py))?;
        let given = given.bind(py);
        let inner = if !result_is_err(given)? {
            let value = result_ok_value(given)?;
            if value.is_none() {
                if may_be_nil(py, access, step.bind(py), node)? {
                    py.None()
                } else {
                    return gap(no_value_message(py, tag.as_deref(), step.bind(py)));
                }
            } else {
                emit(py, access, &value)?
            }
        } else if may_be_nil(py, access, step.bind(py), node)? {
            py.None()
        } else {
            return gap(no_value_message(py, tag.as_deref(), step.bind(py)));
        };
        parts.push(match &tag {
            None => inner,
            Some(tag) => tag_expr(py, tag, inner.bind(py))?,
        });
    }
    let mut product = builder::wrap(py, parts[0].bind(py))?;
    let mut started = false;
    for element in &parts[1..] {
        let element = builder::wrap(py, element.bind(py))?;
        // The first `*` starts the chain; every one after it grows that chain.
        product = if started {
            builder::product_append(py, product.bind(py), element.bind(py))?
        } else {
            builder::product_of(py, product.bind(py), element.bind(py))?
        };
        started = true;
    }
    Ok(product)
}

fn no_value_message(py: Python<'_>, tag: Option<&str>, step: &Bound<'_, PyAny>) -> String {
    match tag {
        Some(tag) => format!("no value here: {tag}"),
        None => format!("no value here: {}", step_repr(py, step)),
    }
}

fn emit_tuple(
    py: Python<'_>,
    access: &Bound<'_, PyAny>,
    node: &Bound<'_, PyAny>,
) -> SResult<Py<PyAny>> {
    let length = access.call_method1("length", (node.clone(),))?;
    if result_is_err(&length)? {
        return gap(result_err_message(&length)?);
    }
    let length = result_ok_value(&length)?.extract::<usize>()?;
    let mut items = Vec::new();
    for index in 0..length {
        let step = step_of(py, "at_index", pyast::pyint(py, index as i64).bind(py))?;
        let child = child(py, access, node, step.bind(py))?;
        items.push(emit(py, access, child.bind(py))?);
    }
    Ok(PyTuple::new(py, items)?.into_any().unbind())
}

fn emit_tagged(
    py: Python<'_>,
    access: &Bound<'_, PyAny>,
    node: &Bound<'_, PyAny>,
) -> SResult<Py<PyAny>> {
    for (tag, step, _descriptor) in member_steps(py, access, node)? {
        let tag = match tag {
            Some(tag) => tag,
            None => {
                return gap(format!(
                    "this tagged piece has no member: {}",
                    pyrepr(py, node)
                ))
            }
        };
        let child = child(py, access, node, step.bind(py))?;
        let inner = emit(py, access, child.bind(py))?;
        return tag_expr(py, &tag, inner.bind(py));
    }
    gap(format!(
        "this tagged piece has no member: {}",
        pyrepr(py, node)
    ))
}

fn emit_exponent(
    py: Python<'_>,
    access: &Bound<'_, PyAny>,
    node: &Bound<'_, PyAny>,
    unfolded: &Bound<'_, PyAny>,
) -> SResult<Py<PyAny>> {
    let elements = payload_list(unfolded, "elements")?;
    let head = match elements.first() {
        Some(head) => head.clone_ref(py),
        None => return gap("no value for the result of exponent"),
    };
    let mut steps = member_steps(py, access, node)?;
    let head = head.bind(py);
    let mut chain: Py<PyAny> = if is_unit_descriptor(py, access, head)? {
        let unfolded_head = unfold(py, access, head)?;
        if descriptor_kind(unfolded_head.bind(py))? == reflect::NEVER {
            unit_name(py, "never")?
        } else {
            unit_name(py, "nil")?
        }
    } else {
        if steps.is_empty() {
            return gap(format!(
                "no value for the result of {}",
                descriptor_kind(unfolded)?
            ));
        }
        let (_, step, _descriptor) = steps.remove(0);
        let child = child(py, access, node, step.bind(py))?;
        emit(py, access, child.bind(py))?
    };
    for (tag, step, _descriptor) in steps {
        let given = get(py, access, node, step.bind(py))?;
        let given = given.bind(py);
        let inner = if !result_is_err(given)? {
            let value = result_ok_value(given)?;
            if value.is_none() {
                if may_be_nil(py, access, step.bind(py), node)? {
                    py.None()
                } else {
                    return gap(no_value_message(py, tag.as_deref(), step.bind(py)));
                }
            } else {
                emit(py, access, &value)?
            }
        } else if may_be_nil(py, access, step.bind(py), node)? {
            py.None()
        } else {
            return gap(no_value_message(py, tag.as_deref(), step.bind(py)));
        };
        let argument = argument(py, inner.bind(py), tag.as_deref())?;
        // `<-` is left-associative: a result that is already a chain grows it,
        // and anything else starts one.
        chain = if is_class(chain.bind(py), "_Exponent")? {
            builder::exponent_append(py, chain.bind(py), argument.bind(py))?
        } else {
            builder::power_of(py, chain.bind(py), argument.bind(py))?
        };
    }
    Ok(chain)
}

fn is_literal_descriptor(descriptor: &Bound<'_, PyAny>) -> PyResult<bool> {
    if descriptor_kind(descriptor)? != reflect::TYPE_APP {
        return Ok(false);
    }
    let constructor: String = payload(descriptor)?
        .getattr("constructor_name")?
        .extract()?;
    Ok(reflect::literal_kind(&constructor).is_some())
}

fn literal_keys_are_strings(py: Python<'_>, node: &Bound<'_, PyAny>) -> PyResult<bool> {
    let data = attr(node, "data")?;
    if !is_class(&data, "TypeApp")? {
        return Ok(false);
    }
    if attr(&data, "constructor")?.extract::<String>()? != "DictLiteral" {
        return Ok(false);
    }
    for pair in attr(&data, "args")?.try_iter()? {
        let pair = pair?;
        if !is_class(&pair, "Tuple")? {
            return Ok(false);
        }
        let elements = attr(&pair, "elements")?;
        let elements: Vec<Bound<'_, PyAny>> = elements.try_iter()?.collect::<PyResult<Vec<_>>>()?;
        if elements.is_empty() || !is_class(&elements[0], "Constant")? {
            return Ok(false);
        }
        let value = attr(&elements[0], "value")?;
        if !value.is_instance_of::<PyString>() {
            return Ok(false);
        }
    }
    let _ = py;
    Ok(true)
}

fn emit_container(
    py: Python<'_>,
    access: &Bound<'_, PyAny>,
    node: &Bound<'_, PyAny>,
    container: &str,
    unfolded: &Bound<'_, PyAny>,
) -> SResult<Py<PyAny>> {
    if container == "dict" {
        let keys = access.call_method1("keys", (node.clone(),))?;
        if result_is_err(&keys)? {
            return gap(result_err_message(&keys)?);
        }
        let keys: Vec<Bound<'_, PyAny>> = result_ok_value(&keys)?
            .try_iter()?
            .collect::<PyResult<Vec<_>>>()?;
        let string_keys_gap = "the protocol hands dict keys over as strings; \
                               this one has another key type";
        if is_literal_descriptor(unfolded)? {
            // The design is the literal itself, so there is no key type to ask.
            if !literal_keys_are_strings(py, node)? {
                return gap(string_keys_gap);
            }
        } else {
            // The key type in the source, through however many names.
            let args = payload_list(unfolded, "args")?;
            let mut good = false;
            if let Some(first) = args.first() {
                let key_type = unfold(py, access, first.bind(py))?;
                let key_type = key_type.bind(py);
                if descriptor_kind(key_type)? == reflect::TYPE_REF {
                    good = payload(key_type)?
                        .getattr("type_name")?
                        .extract::<String>()?
                        == "str";
                }
            }
            if !good {
                return gap(string_keys_gap);
            }
        }
        let mut pairs = Vec::new();
        for key in keys {
            let step = step_of(py, "at_key", &key)?;
            let child = child(py, access, node, step.bind(py))?;
            let value = emit(py, access, child.bind(py))?;
            pairs.push(PyTuple::new(py, [key.unbind(), value])?.into_any().unbind());
        }
        return Ok(builder::type_app(py, "DictLiteral", pairs)?);
    }
    let length = access.call_method1("length", (node.clone(),))?;
    if result_is_err(&length)? {
        return gap(result_err_message(&length)?);
    }
    let length = result_ok_value(&length)?.extract::<usize>()?;
    let mut payloads = Vec::new();
    for index in 0..length {
        let step = step_of(py, "at_index", pyast::pyint(py, index as i64).bind(py))?;
        let child = child(py, access, node, step.bind(py))?;
        payloads.push(emit(py, access, child.bind(py))?);
    }
    match container {
        "list" => Ok(builder::type_app(py, "ListLiteral", payloads)?),
        "set" => Ok(builder::type_app(py, "SetLiteral", payloads)?),
        _ => gap(format!("cannot serialize this container out: {container}")),
    }
}

// --- the entry point -----------------------------------------------------

fn serialize_source(
    py: Python<'_>,
    access: &Bound<'_, PyAny>,
    name: &str,
    node: &Bound<'_, PyAny>,
) -> SResult<Py<PyAny>> {
    let expression = emit(py, access, node)?;
    let vb = builder_module(py)?.getattr("Builder")?.call0()?;
    builder::define(py, &vb, name, expression.bind(py))?;
    builder::check(py, &vb)?;
    Ok(pyast::pystr(py, &vb.str()?.extract::<String>()?))
}

/// A piece of the design in hand (a VObject) -> viba source: the definition
/// named `name` whose body is that piece's spelling.
fn serialize_impl(py: Python<'_>, name: &str, node: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let access = attr(node, "_access")?;
    match serialize_source(py, &access, name, node) {
        Ok(source) => type_system::ok(py, source),
        Err(SerializeError::Gap(message)) => type_system::err(py, &message),
        Err(SerializeError::Python(error)) => {
            let value = error.value(py);
            if value.is_instance_of::<PyTypeError>() || value.is_instance_of::<PyValueError>() {
                let message: String = value.str()?.extract()?;
                type_system::err(py, &message)
            } else {
                Err(error)
            }
        }
    }
}

#[pyfunction]
fn serialize_py(py: Python<'_>, name: &str, node: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    serialize_impl(py, name, node)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(serialize_py, m)?)?;
    Ok(())
}
