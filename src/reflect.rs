//! Reflect: treat a design as a map and take a viba data through it
//! (was viba/reflect.py).
//!
//! The map comes from the descriptor side walking the design; the data is one
//! viba data's type expression; access walks the map's addresses one step at a
//! time. `VibaAccess` is the accessor (the protocol's VibaRoot / VibaHas /
//! VibaGet / VibaLeaf / VibaLength / VibaKeys / VibaResolve / VibaGetByPath /
//! VibaListFields, plus this layer's map-walking helpers), and the data classes
//! the protocol names are the Python face in `rs_viba.reflect`.

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyModule, PySet, PyString, PyTuple};

use crate::pyast;
use crate::type_system;

// Branch names of the spec's VibaTypeDescriptor sum.
pub const TYPE_REF: &str = "type_ref";
pub const MEMBER_TAKEN: &str = "member_taken";
pub const TYPE_APP: &str = "type_app";
pub const TUPLE: &str = "tuple";
pub const TAGGED: &str = "tagged";
pub const SUM: &str = "sum";
pub const PRODUCT: &str = "product";
pub const EXPONENT: &str = "exponent";
pub const LITERAL: &str = "literal";
pub const NIL: &str = "nil";
pub const NEVER: &str = "never";
pub const ANY: &str = "any";
pub const ELLIPSIS: &str = "ellipsis";
pub const HINT_BLOCK: &str = "code_block";

pub const FIXED_INT_NAMES: &[&str] = &[
    "int8_t", "uint8_t", "int16_t", "uint16_t", "int32_t", "uint32_t", "int64_t", "uint64_t",
];
pub const SCALAR_NAMES: &[&str] = &["bool", "int", "float", "str"];
pub const CONTAINERS: &[&str] = &["list", "set", "dict"];
pub const LITERAL_CTORS: &[&str] = &["ListLiteral", "SetLiteral", "DictLiteral"];

/// The container a literal spells, or None when the name is no literal
/// constructor. `ListLiteral[a, b]` is a resident of `list[a | b]`, so the
/// piece is taken the way any other piece of that container is.
pub fn literal_kind(name: &str) -> Option<&'static str> {
    match name {
        "ListLiteral" => Some("list"),
        "SetLiteral" => Some("set"),
        "DictLiteral" => Some("dict"),
        _ => None,
    }
}

fn scalar_name_of(value: &Bound<'_, PyAny>) -> PyResult<Option<&'static str>> {
    if value.is_instance_of::<PyBool>() {
        return Ok(Some("bool"));
    }
    if value.is_instance_of::<PyInt>() {
        return Ok(Some("int"));
    }
    if value.is_instance_of::<PyFloat>() {
        return Ok(Some("float"));
    }
    if value.is_instance_of::<PyString>() {
        return Ok(Some("str"));
    }
    Ok(None)
}

fn key_text_of(py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    if key.is_none() {
        return Ok(py.None());
    }
    if key.is_instance_of::<PyBool>() {
        return Ok(pyast::pystr(
            py,
            if key.is_truthy()? { "true" } else { "false" },
        ));
    }
    let text: String = key.str()?.extract()?;
    Ok(pyast::pystr(py, &text))
}

fn type_name_of(value: &Bound<'_, PyAny>) -> PyResult<String> {
    if let Ok(text) = value.extract::<String>() {
        return Ok(text);
    }
    for attribute in ["type_name", "name", "full_name"] {
        if let Some(given) = opt_attr(value, attribute) {
            if let Ok(text) = given.extract::<String>() {
                return Ok(text);
            }
        }
    }
    Ok(value.str()?.extract()?)
}

fn tag_of(py: Python<'_>, name: &Bound<'_, PyAny>) -> PyResult<String> {
    if let Ok(text) = name.extract::<String>() {
        return Ok(if text.starts_with('$') {
            text
        } else {
            format!("${text}")
        });
    }
    Err(PyTypeError::new_err(format!(
        "a tag name is a string, not {}",
        name.get_type().name()?
    )))
}

fn pyrepr(py: Python<'_>, value: &Bound<'_, PyAny>) -> String {
    pyast::repr_of(py, value)
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

fn ast_module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    module(py, "rs_viba.viba_ast")
}

fn pattern_module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    module(py, "rs_viba.pattern")
}

fn type_module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    module(py, "rs_viba.type")
}

fn is_class(node: &Bound<'_, PyAny>, name: &str) -> PyResult<bool> {
    Ok(pyast::class_name(node)? == name)
}

fn opt_attr<'py>(value: &Bound<'py, PyAny>, name: &str) -> Option<Bound<'py, PyAny>> {
    match value.getattr(name) {
        Ok(given) if !given.is_none() => Some(given),
        _ => None,
    }
}

fn attr<'py>(value: &Bound<'py, PyAny>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    value.getattr(name)
}

fn attr_text(value: &Bound<'_, PyAny>, name: &str) -> PyResult<String> {
    value.getattr(name)?.extract()
}

fn descriptor_kind(descriptor: &Bound<'_, PyAny>) -> PyResult<String> {
    attr_text(descriptor, "kind")
}

fn payload<'py>(descriptor: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    attr(descriptor, "payload")
}

fn payload_attr<'py>(descriptor: &Bound<'py, PyAny>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    payload(descriptor)?.getattr(name)
}

fn payload_opt<'py>(descriptor: &Bound<'py, PyAny>, name: &str) -> Option<Bound<'py, PyAny>> {
    payload(descriptor).ok().and_then(|p| opt_attr(&p, name))
}

fn chain_elements(descriptor: &Bound<'_, PyAny>) -> PyResult<Vec<Py<PyAny>>> {
    let mut out = Vec::new();
    for element in payload_attr(descriptor, "elements")?.try_iter()? {
        out.push(element?.unbind());
    }
    Ok(out)
}

fn new_descriptor<'py>(
    py: Python<'py>,
    kind: &str,
    payload: Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    Ok(vtd(py)?
        .getattr("VibaTypeDescriptor")?
        .call1((kind, payload))?
        .unbind())
}

fn chain_descriptor(
    py: Python<'_>,
    kind: &str,
    pool: &Bound<'_, PyAny>,
    resolvable: &Bound<'_, PyAny>,
    elements: Vec<Py<PyAny>>,
) -> PyResult<Py<PyAny>> {
    let payload = vtd(py)?.getattr("VibaChainDescriptor")?.call1((
        pool.clone(),
        resolvable.clone(),
        pyast::pylist(py, elements)?,
    ))?;
    new_descriptor(py, kind, payload)
}

fn type_app_descriptor(
    py: Python<'_>,
    pool: &Bound<'_, PyAny>,
    resolvable: &Bound<'_, PyAny>,
    constructor: &str,
    args: Vec<Py<PyAny>>,
) -> PyResult<Py<PyAny>> {
    let payload = vtd(py)?.getattr("VibaTypeAppDescriptor")?.call1((
        pool.clone(),
        resolvable.clone(),
        pyast::pystr(py, constructor),
        pyast::pylist(py, args)?,
    ))?;
    new_descriptor(py, TYPE_APP, payload)
}

fn tuple_descriptor(
    py: Python<'_>,
    pool: &Bound<'_, PyAny>,
    resolvable: &Bound<'_, PyAny>,
    elements: Vec<Py<PyAny>>,
) -> PyResult<Py<PyAny>> {
    let payload = vtd(py)?.getattr("VibaTupleDescriptor")?.call1((
        pool.clone(),
        resolvable.clone(),
        pyast::pylist(py, elements)?,
    ))?;
    new_descriptor(py, TUPLE, payload)
}

fn tagged_descriptor_fn(
    py: Python<'_>,
    pool: &Bound<'_, PyAny>,
    constructor: &str,
    args: &[Py<PyAny>],
    resolvable: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let args: Vec<Py<PyAny>> = args.iter().map(|argument| argument.clone_ref(py)).collect();
    Ok(vtd(py)?
        .getattr("tagged_descriptor")?
        .call1((
            pool.clone(),
            constructor,
            pyast::pylist(py, args)?,
            resolvable.clone(),
        ))?
        .unbind())
}

fn build_type(
    py: Python<'_>,
    pool: &Bound<'_, PyAny>,
    module: &Bound<'_, PyAny>,
    node: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    Ok(vtd(py)?
        .getattr("_build_type")?
        .call1((pool.clone(), module.clone(), node.clone()))?
        .unbind())
}

/// `AstNodeType(node, module)`: the type value a piece of syntax stands for.
pub fn ast_node_type(
    py: Python<'_>,
    node: &Bound<'_, PyAny>,
    module: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    Ok(type_module(py)?
        .getattr("AstNodeType")?
        .call1((node.clone(), module.clone()))?
        .unbind())
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

fn ok(py: Python<'_>, value: Py<PyAny>) -> PyResult<Py<PyAny>> {
    type_system::ok(py, value)
}

fn err(py: Python<'_>, message: &str) -> PyResult<Py<PyAny>> {
    type_system::err(py, message)
}

fn make_step(py: Python<'_>, kind: &str, value: Py<PyAny>) -> PyResult<Py<PyAny>> {
    Ok(reflect_module(py)?
        .getattr("VibaStep")?
        .call1((kind, value))?
        .unbind())
}

fn step_kind(step: &Bound<'_, PyAny>) -> PyResult<String> {
    attr_text(step, "kind")
}

fn step_text(step: &Bound<'_, PyAny>) -> PyResult<String> {
    attr_text(step, "value")
}

fn step_index(step: &Bound<'_, PyAny>) -> PyResult<i64> {
    attr(step, "value")?.extract()
}

fn make_tuple(py: Python<'_>, items: Vec<Py<PyAny>>) -> PyResult<Py<PyAny>> {
    Ok(PyTuple::new(py, items)?.into_any().unbind())
}

fn make_list(py: Python<'_>, items: Vec<Py<PyAny>>) -> PyResult<Py<PyAny>> {
    pyast::pylist(py, items)
}

fn new_set_with(
    py: Python<'_>,
    seen: &Bound<'_, PyAny>,
    key: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let one = PySet::new(py, [key.clone()])?;
    Ok(seen.call_method1("__or__", (one,))?.unbind())
}

fn is_empty_set(py: Python<'_>) -> PyResult<Py<PyAny>> {
    Ok(PySet::empty(py)?.into_any().unbind())
}

fn set_contains(seen: &Bound<'_, PyAny>, key: &Bound<'_, PyAny>) -> PyResult<bool> {
    seen.call_method1("__contains__", (key.clone(),))?.extract()
}

fn id_of(value: &Bound<'_, PyAny>) -> usize {
    value.as_ptr() as usize
}

fn key_of_kind(py: Python<'_>, kind: &str, id: usize) -> PyResult<Py<PyAny>> {
    make_tuple(py, vec![pyast::pystr(py, kind), pyint(py, id)])
}

fn key_of_two(py: Python<'_>, kind: &str, first: usize, second: usize) -> PyResult<Py<PyAny>> {
    make_tuple(
        py,
        vec![pyast::pystr(py, kind), pyint(py, first), pyint(py, second)],
    )
}

fn key_of_member(
    py: Python<'_>,
    kind: &str,
    first: usize,
    second: usize,
    name: &str,
) -> PyResult<Py<PyAny>> {
    make_tuple(
        py,
        vec![
            pyast::pystr(py, kind),
            pyint(py, first),
            pyint(py, second),
            pyast::pystr(py, name),
        ],
    )
}

fn pyint(py: Python<'_>, value: usize) -> Py<PyAny> {
    (value as u64)
        .into_pyobject(py)
        .unwrap()
        .into_any()
        .unbind()
}

// ----------------------------------------------------------------------
// Members: [(tag or None, descriptor), ...]
// ----------------------------------------------------------------------

type Slot = (Option<String>, Py<PyAny>);

fn slots_to_py(py: Python<'_>, slots: Option<Vec<Slot>>) -> PyResult<Py<PyAny>> {
    let slots = match slots {
        Some(slots) => slots,
        None => return Ok(py.None()),
    };
    let mut items = Vec::new();
    for (tag, descriptor) in slots {
        let tag = match tag {
            Some(tag) => pyast::pystr(py, &tag),
            None => py.None(),
        };
        items.push(make_tuple(py, vec![tag, descriptor])?);
    }
    make_list(py, items)
}

fn slots_of(value: &Bound<'_, PyAny>) -> PyResult<Vec<Slot>> {
    let mut out = Vec::new();
    for item in value.try_iter()? {
        let item = item?;
        let pair = item.cast::<PyTuple>()?;
        let tag = pair.get_item(0)?;
        let tag = if tag.is_none() {
            None
        } else {
            Some(tag.extract::<String>()?)
        };
        out.push((tag, pair.get_item(1)?.unbind()));
    }
    Ok(out)
}

fn optional_slots(value: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Vec<Slot>>> {
    match value {
        Some(value) if !value.is_none() => Ok(Some(slots_of(value)?)),
        _ => Ok(None),
    }
}

// ----------------------------------------------------------------------
// The Python-visible data classes the accessor builds
// ----------------------------------------------------------------------

fn make_vobject(
    py: Python<'_>,
    access: &Bound<'_, PyAny>,
    descriptor: &Bound<'_, PyAny>,
    data: &Bound<'_, PyAny>,
    path: Vec<Py<PyAny>>,
    data_module: Option<&Bound<'_, PyAny>>,
    function: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let class = reflect_module(py)?.getattr("VObject")?;
    let path = PyTuple::new(py, path)?;
    let data_module = match data_module {
        Some(value) => value.clone(),
        None => py.None().into_bound(py),
    };
    let function = match function {
        Some(value) => value.clone(),
        None => py.None().into_bound(py),
    };
    Ok(class
        .call1((
            access.clone(),
            descriptor.clone(),
            data.clone(),
            path,
            data_module,
            function,
        ))?
        .unbind())
}

fn reflect_error(py: Python<'_>, message: &str) -> PyErr {
    match reflect_module(py)
        .and_then(|module| module.getattr("VibaReflectError"))
        .and_then(|class| class.call1((message,)))
    {
        Ok(error) => PyErr::from_value(error),
        Err(error) => error,
    }
}

// ----------------------------------------------------------------------
// The accessor
// ----------------------------------------------------------------------

/// An accessor: the protocol's cells plus this layer's map-walking helpers, all
/// sharing one VibaReflectConfig.
#[pyclass(name = "VibaAccess")]
pub struct VibaAccess {
    config: Py<PyAny>,
}

impl VibaAccess {
    fn config_value<'py>(&self, py: Python<'py>) -> Py<PyAny> {
        self.config.clone_ref(py)
    }
}

// ---- the map: unfolding ----

fn d_unfold(
    py: Python<'_>,
    access: &Bound<'_, PyAny>,
    descriptor: &Bound<'_, PyAny>,
    seen: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let owned;
    let seen: &Bound<'_, PyAny> = match seen {
        Some(seen) => seen,
        None => {
            owned = is_empty_set(py)?.into_bound(py);
            &owned
        }
    };
    let (result, _) = d_unfold_seen(py, access, descriptor, seen)?;
    Ok(result)
}

fn d_unfold_seen(
    py: Python<'_>,
    access: &Bound<'_, PyAny>,
    descriptor: &Bound<'_, PyAny>,
    seen: &Bound<'_, PyAny>,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    if let Some((result, seen)) = d_decided(py, access, descriptor, seen)? {
        return Ok((result, seen));
    }
    if let Some((result, seen)) = d_decided_member(py, access, descriptor, seen)? {
        return Ok((result, seen));
    }
    let found = match d_definition_target(py, descriptor)? {
        Some(found) => found,
        None => return Ok((descriptor.clone().unbind(), seen.clone().unbind())),
    };
    let (key, target) = found;
    if set_contains(seen, key.bind(py))? {
        return Ok((descriptor.clone().unbind(), seen.clone().unbind()));
    }
    let kind = descriptor_kind(descriptor)?;
    if kind == TYPE_REF {
        let body_node = ast_node_type_body(py, target.bind(py))?;
        let module = attr(target.bind(py), "container_module")?;
        let pool = payload_opt(descriptor, "pool").unwrap_or(py.None().into_bound(py));
        let body = build_type(py, &pool, &module, &body_node)?;
        let next = new_set_with(py, seen, key.bind(py))?;
        return d_unfold_seen(py, access, body.bind(py), next.bind(py));
    }
    match d_apply_generic(py, descriptor, target.bind(py))? {
        Some(applied) => {
            let next = new_set_with(py, seen, key.bind(py))?;
            d_unfold_seen(py, access, applied.bind(py), next.bind(py))
        }
        None => Ok((descriptor.clone().unbind(), seen.clone().unbind())),
    }
}

fn ast_node_type_body<'py>(
    _py: Python<'py>,
    target: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let node = attr(target, "ast_node")?;
    Ok(node.getattr("body")?)
}

fn d_decided(
    py: Python<'_>,
    access: &Bound<'_, PyAny>,
    descriptor: &Bound<'_, PyAny>,
    seen: &Bound<'_, PyAny>,
) -> PyResult<Option<(Py<PyAny>, Py<PyAny>)>> {
    if descriptor_kind(descriptor)? != TYPE_APP {
        return Ok(None);
    }
    let resolvable = match payload_opt(descriptor, "resolvable_type") {
        Some(resolvable) => resolvable,
        None => return Ok(None),
    };
    let node = match opt_attr(&resolvable, "ast_node") {
        Some(node) => node,
        None => return Ok(None),
    };
    let module = attr(&resolvable, "container_module")?;
    let decision = pattern_module(py)?
        .getattr("reduce_application")?
        .call1((node.clone(), module.clone()))?;
    if result_is_err(&decision)? {
        return Ok(None);
    }
    let chosen = result_ok_value(&decision)?;
    if chosen.is_none() {
        return Ok(None);
    }
    let key = key_of_two(py, "decided", id_of(&node), id_of(&module))?;
    if set_contains(seen, key.bind(py))? {
        return Ok(Some((descriptor.clone().unbind(), seen.clone().unbind())));
    }
    let pool = payload_attr(descriptor, "pool")?;
    let bindings = build_bindings(py, &pool, &chosen)?;
    let chosen_module = attr(&chosen, "module")?;
    let body_node = attr(&chosen, "body")?;
    let body = build_type(py, &pool, &chosen_module, &body_node)?;
    let substituted = d_substitute(py, body.bind(py), &bindings)?;
    let next = new_set_with(py, seen, key.bind(py))?;
    d_unfold_seen(py, access, substituted.bind(py), next.bind(py)).map(Some)
}

fn build_bindings<'py>(
    py: Python<'py>,
    pool: &Bound<'py, PyAny>,
    chosen: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let bindings = PyDict::new(py);
    let given = attr(chosen, "bindings")?;
    for item in given.call_method0("items")?.try_iter()? {
        let item = item?;
        let pair = item.cast::<PyTuple>()?;
        let name = pair.get_item(0)?;
        let bound = pair.get_item(1)?;
        let module = attr(&bound, "container_module")?;
        let node = attr(&bound, "ast_node")?;
        let descriptor = build_type(py, pool, &module, &node)?;
        bindings.set_item(name, descriptor)?;
    }
    Ok(bindings.into_any())
}

fn d_decided_member(
    py: Python<'_>,
    access: &Bound<'_, PyAny>,
    descriptor: &Bound<'_, PyAny>,
    seen: &Bound<'_, PyAny>,
) -> PyResult<Option<(Py<PyAny>, Py<PyAny>)>> {
    if descriptor_kind(descriptor)? != MEMBER_TAKEN {
        return Ok(None);
    }
    let owner = payload_attr(descriptor, "owner")?;
    if descriptor_kind(&owner)? != TYPE_APP {
        return Ok(None);
    }
    let resolvable = match payload_opt(&owner, "resolvable_type") {
        Some(resolvable) => resolvable,
        None => return Ok(None),
    };
    let node = match opt_attr(&resolvable, "ast_node") {
        Some(node) => node,
        None => return Ok(None),
    };
    let module = attr(&resolvable, "container_module")?;
    let decision = pattern_module(py)?
        .getattr("reduce_application")?
        .call1((node.clone(), module.clone()))?;
    if result_is_err(&decision)? {
        return Ok(None);
    }
    let chosen = result_ok_value(&decision)?;
    if chosen.is_none() {
        return Ok(None);
    }
    let name = payload_attr(descriptor, "name")?.extract::<String>()?;
    let entry = attr(&chosen, "entry")?;
    let entry_module = attr(&entry, "module")?;
    let file_module = attr(&entry_module, "module")?;
    let body = attr(&file_module, "body")?;
    let mut definition: Option<Bound<'_, PyAny>> = None;
    for statement in body.try_iter()? {
        let statement = statement?;
        let kind = pyast::class_name(&statement)?;
        if (kind == "TypeDefinition" || kind == "GenericDefinition")
            && attr_text(&statement, "name")? == name
        {
            definition = Some(statement);
            break;
        }
    }
    let definition = match definition {
        Some(definition) => definition,
        None => return Ok(None),
    };
    let key = key_of_member(py, "member", id_of(&node), id_of(&module), &name)?;
    if set_contains(seen, key.bind(py))? {
        return Ok(Some((descriptor.clone().unbind(), seen.clone().unbind())));
    }
    let pool = payload_attr(&owner, "pool")?;
    let bindings = build_bindings(py, &pool, &chosen)?;
    let chosen_module = attr(&chosen, "module")?;
    let definition_body = attr(&definition, "body")?;
    let built = build_type(py, &pool, &chosen_module, &definition_body)?;
    let substituted = d_substitute(py, built.bind(py), &bindings)?;
    let next = new_set_with(py, seen, key.bind(py))?;
    d_unfold_seen(py, access, substituted.bind(py), next.bind(py)).map(Some)
}

fn d_definition_target(
    py: Python<'_>,
    descriptor: &Bound<'_, PyAny>,
) -> PyResult<Option<(Py<PyAny>, Py<PyAny>)>> {
    let kind = descriptor_kind(descriptor)?;
    let (key_kind, name) = if kind == TYPE_REF {
        (
            "ref",
            payload_attr(descriptor, "type_name")?.extract::<String>()?,
        )
    } else if kind == TYPE_APP {
        (
            "app",
            payload_attr(descriptor, "constructor_name")?.extract::<String>()?,
        )
    } else {
        return Ok(None);
    };
    let target = match d_source_target(py, descriptor, &name)? {
        Some(target) => target,
        None => return Ok(None),
    };
    let node = attr(target.bind(py), "ast_node")?;
    let node_kind = pyast::class_name(&node)?;
    let wanted = if key_kind == "ref" {
        "TypeDefinition"
    } else {
        "GenericDefinition"
    };
    if node_kind != wanted {
        return Ok(None);
    }
    let key = key_of_kind(py, key_kind, id_of(&node))?;
    Ok(Some((key, target)))
}

fn d_source_target(
    py: Python<'_>,
    descriptor: &Bound<'_, PyAny>,
    name: &str,
) -> PyResult<Option<Py<PyAny>>> {
    let resolvable = match payload_opt(descriptor, "resolvable_type") {
        Some(resolvable) => resolvable,
        None => return Ok(None),
    };
    let module = attr(&resolvable, "container_module")?;
    let resolved = type_system::module_get_type(py, &module, name, None)?;
    let resolved = resolved.bind(py);
    if result_is_err(resolved)? {
        return Ok(None);
    }
    let value = result_ok_value(resolved)?;
    if !is_class(&value, "AstNodeType")? {
        return Ok(None);
    }
    Ok(Some(value.unbind()))
}

fn d_apply_generic(
    py: Python<'_>,
    descriptor: &Bound<'_, PyAny>,
    target: &Bound<'_, PyAny>,
) -> PyResult<Option<Py<PyAny>>> {
    let payload_value = payload(descriptor)?;
    let node = attr(target, "ast_node")?;
    let mut params = Vec::new();
    for param in attr(&node, "generic_params")?.try_iter()? {
        params.push(param?);
    }
    let mut args = Vec::new();
    for argument in attr(&payload_value, "args")?.try_iter()? {
        args.push(argument?);
    }
    if params.len() != args.len() {
        return Ok(None);
    }
    // The builtin library is loaded as the source has it (never canonicalized), so
    // canonicalize here first.
    let body_node = ast_module(py)?
        .getattr("convert_to_chain_style")?
        .call1((attr(&node, "body")?,))?;
    let pool = attr(&payload_value, "pool")?;
    let module = attr(target, "container_module")?;
    let body = build_type(py, &pool, &module, &body_node)?;
    let bindings = PyDict::new(py);
    for (param, argument) in params.iter().zip(args.iter()) {
        bindings.set_item(param, argument)?;
    }
    let substituted = d_substitute(py, body.bind(py), &bindings.into_any())?;
    Ok(Some(substituted))
}

fn d_substitute(
    py: Python<'_>,
    descriptor: &Bound<'_, PyAny>,
    bindings: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let kind = descriptor_kind(descriptor)?;
    if kind == TYPE_REF {
        let name = payload_attr(descriptor, "type_name")?;
        return Ok(bindings
            .call_method1("get", (name, descriptor.clone()))?
            .unbind());
    }
    let payload_value = payload(descriptor)?;
    if kind == PRODUCT || kind == SUM || kind == EXPONENT {
        let mut elements = Vec::new();
        for element in attr(&payload_value, "elements")?.try_iter()? {
            elements.push(d_substitute(py, &element?, bindings)?);
        }
        return chain_descriptor(
            py,
            &kind,
            &attr(&payload_value, "pool")?,
            &attr(&payload_value, "resolvable_type")?,
            elements,
        );
    }
    if kind == TYPE_APP {
        let mut args = Vec::new();
        for argument in attr(&payload_value, "args")?.try_iter()? {
            args.push(d_substitute(py, &argument?, bindings)?);
        }
        let constructor = attr_text(&payload_value, "constructor_name")?;
        let pool = attr(&payload_value, "pool")?;
        let resolvable = attr(&payload_value, "resolvable_type")?;
        let folded = tagged_descriptor_fn(py, &pool, &constructor, &args, &resolvable)?;
        if !folded.bind(py).is_none() {
            return Ok(folded);
        }
        return type_app_descriptor(py, &pool, &resolvable, &constructor, args);
    }
    if kind == TUPLE {
        let mut elements = Vec::new();
        for element in attr(&payload_value, "elements")?.try_iter()? {
            elements.push(d_substitute(py, &element?, bindings)?);
        }
        return tuple_descriptor(
            py,
            &attr(&payload_value, "pool")?,
            &attr(&payload_value, "resolvable_type")?,
            elements,
        );
    }
    if kind == TAGGED {
        let tag = attr_text(&payload_value, "tag")?;
        let inner = d_substitute(py, &attr(&payload_value, "tagged_type")?, bindings)?;
        let payload = vtd(py)?.getattr("VibaTaggedDescriptor")?.call1((
            attr(&payload_value, "pool")?,
            attr(&payload_value, "resolvable_type")?,
            pyast::pystr(py, &tag),
            inner,
        ))?;
        return new_descriptor(py, TAGGED, payload);
    }
    Ok(descriptor.clone().unbind())
}

// ---- the map: members ----

fn d_carries_nil(
    py: Python<'_>,
    access: &Bound<'_, PyAny>,
    descriptor: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    let unfolded = d_unfold(py, access, descriptor, None)?;
    if descriptor_kind(unfolded.bind(py))? != SUM {
        return Ok(false);
    }
    for element in chain_elements(unfolded.bind(py))? {
        if d_is_unit_descriptor(py, access, element.bind(py))? {
            return Ok(true);
        }
    }
    Ok(false)
}

fn d_container_kind(
    _py: Python<'_>,
    access: &Bound<'_, PyAny>,
    descriptor: &Bound<'_, PyAny>,
) -> PyResult<Option<String>> {
    let _ = access;
    if descriptor_kind(descriptor)? == TYPE_APP {
        let name = payload_attr(descriptor, "constructor_name")?.extract::<String>()?;
        if CONTAINERS.contains(&name.as_str()) {
            return Ok(Some(name));
        }
        return Ok(literal_kind(&name).map(|kind| kind.to_string()));
    }
    Ok(None)
}

fn d_has_elements_by_index(
    py: Python<'_>,
    access: &Bound<'_, PyAny>,
    descriptor: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    if let Some(container) = d_container_kind(py, access, descriptor)? {
        return Ok(container == "list" || container == "set");
    }
    Ok(descriptor_kind(descriptor)? == TUPLE)
}

fn d_design_members(
    py: Python<'_>,
    access: &Bound<'_, PyAny>,
    descriptor: &Bound<'_, PyAny>,
) -> PyResult<Option<Vec<Slot>>> {
    let descriptor = d_unfold(py, access, descriptor, None)?;
    let descriptor = descriptor.bind(py);
    let kind = descriptor_kind(descriptor)?;
    if kind == PRODUCT {
        return d_product_members(py, access, descriptor, None, None, None).map(Some);
    }
    if kind == TAGGED {
        let tag = payload_attr(descriptor, "tag")?.extract::<String>()?;
        let tagged_type = payload_attr(descriptor, "tagged_type")?.unbind();
        return Ok(Some(vec![(Some(tag), tagged_type)]));
    }
    let mut elements = if kind == SUM || kind == EXPONENT {
        chain_elements(descriptor)?
    } else {
        return Ok(None);
    };
    if !elements.is_empty() && d_is_unit_descriptor(py, access, elements[0].bind(py))? {
        elements.remove(0);
    }
    let mut slots = Vec::new();
    for element in elements {
        let element = element.bind(py);
        if descriptor_kind(element)? == TAGGED {
            slots.push((
                Some(payload_attr(element, "tag")?.extract::<String>()?),
                payload_attr(element, "tagged_type")?.unbind(),
            ));
        } else {
            slots.push((None, element.clone().unbind()));
        }
    }
    Ok(Some(slots))
}

fn d_product_members(
    py: Python<'_>,
    access: &Bound<'_, PyAny>,
    descriptor: &Bound<'_, PyAny>,
    seen: Option<&Bound<'_, PyAny>>,
    mut cycles: Option<&mut Vec<String>>,
    collect: Option<bool>,
) -> PyResult<Vec<Slot>> {
    let seen_owned;
    let seen: &Bound<'_, PyAny> = match seen {
        Some(seen) => seen,
        None => {
            seen_owned = is_empty_set(py)?;
            seen_owned.bind(py)
        }
    };
    let collect = collect.unwrap_or(cycles.is_some());
    let mut out = Vec::new();
    for element in chain_elements(descriptor)? {
        let element = element.bind(py);
        if descriptor_kind(element)? == TAGGED {
            out.push((
                Some(payload_attr(element, "tag")?.extract::<String>()?),
                payload_attr(element, "tagged_type")?.unbind(),
            ));
            continue;
        }
        let (unfolded, inner_seen) = d_unfold_seen(py, access, element, seen)?;
        let unfolded = unfolded.bind(py);
        if descriptor_kind(unfolded)? == PRODUCT {
            let inner_seen = inner_seen.bind(py);
            out.extend(d_product_members(
                py,
                access,
                unfolded,
                Some(inner_seen),
                cycles.as_deref_mut(),
                Some(collect),
            )?);
        } else if descriptor_kind(unfolded)? == TAGGED {
            out.push((
                Some(payload_attr(unfolded, "tag")?.extract::<String>()?),
                payload_attr(unfolded, "tagged_type")?.unbind(),
            ));
        } else if d_is_product_unit(py, access, element)? {
            continue;
        } else {
            if collect {
                if let Some(found) = d_definition_target(py, element)? {
                    if set_contains(seen, found.0.bind(py))? {
                        if let Some(cycles) = cycles.as_deref_mut() {
                            cycles.push(d_source_name(element)?);
                        }
                    }
                }
            }
            out.push((None, element.clone().unbind()));
        }
    }
    Ok(out)
}

fn d_source_name(descriptor: &Bound<'_, PyAny>) -> PyResult<String> {
    if descriptor_kind(descriptor)? == TYPE_REF {
        payload_attr(descriptor, "type_name")?.extract()
    } else {
        payload_attr(descriptor, "constructor_name")?.extract()
    }
}

fn d_inline_cycle(
    py: Python<'_>,
    access: &Bound<'_, PyAny>,
    descriptor: &Bound<'_, PyAny>,
) -> PyResult<Option<String>> {
    let unfolded = d_unfold(py, access, descriptor, None)?;
    if descriptor_kind(unfolded.bind(py))? != PRODUCT {
        return Ok(None);
    }
    let mut cycles = Vec::new();
    d_product_members(
        py,
        access,
        unfolded.bind(py),
        None,
        Some(&mut cycles),
        Some(true),
    )?;
    Ok(cycles.into_iter().next())
}

fn d_bare_sum(
    py: Python<'_>,
    access: &Bound<'_, PyAny>,
    descriptor: &Bound<'_, PyAny>,
) -> PyResult<Option<Vec<(usize, String, Py<PyAny>)>>> {
    let descriptor = d_unfold(py, access, descriptor, None)?;
    let descriptor = descriptor.bind(py);
    if descriptor_kind(descriptor)? != SUM {
        return Ok(None);
    }
    let mut elements = chain_elements(descriptor)?;
    if !elements.is_empty() && d_is_unit_descriptor(py, access, elements[0].bind(py))? {
        elements.remove(0);
    }
    let mut members = Vec::new();
    let mut inner = 0;
    for (index, element) in elements.iter().enumerate() {
        let element = element.bind(py);
        if descriptor_kind(element)? == TAGGED {
            return Ok(None);
        }
        let role = d_branch_role(py, access, element)?;
        if role == "inner" {
            inner += 1;
        }
        members.push((index, role, element.clone().unbind()));
    }
    if inner <= 1 {
        Ok(Some(members))
    } else {
        Ok(None)
    }
}

fn d_branch_role(
    py: Python<'_>,
    access: &Bound<'_, PyAny>,
    descriptor: &Bound<'_, PyAny>,
) -> PyResult<String> {
    let descriptor = d_unfold(py, access, descriptor, None)?;
    let descriptor = descriptor.bind(py);
    let kind = descriptor_kind(descriptor)?;
    if kind == NIL || kind == NEVER {
        return Ok("unit".to_string());
    }
    if kind == LITERAL {
        return Ok("leaf".to_string());
    }
    if kind == TYPE_REF {
        let name = payload_attr(descriptor, "type_name")?.extract::<String>()?;
        if is_scalar_name(&name) {
            return Ok("leaf".to_string());
        }
    }
    Ok("inner".to_string())
}

fn is_scalar_name(name: &str) -> bool {
    SCALAR_NAMES.contains(&name) || FIXED_INT_NAMES.contains(&name)
}

fn d_bare_sum_tag_target(
    py: Python<'_>,
    access: &Bound<'_, PyAny>,
    descriptor: &Bound<'_, PyAny>,
    tag: &str,
) -> PyResult<Option<Py<PyAny>>> {
    let members = match d_bare_sum(py, access, descriptor)? {
        Some(members) => members,
        None => return Ok(None),
    };
    for (_, role, branch) in members {
        if role != "inner" {
            continue;
        }
        if let Some(slots) = d_design_members(py, access, branch.bind(py))? {
            for (branch_tag, branch_type) in slots {
                if branch_tag.as_deref() == Some(tag) {
                    return Ok(Some(branch_type));
                }
            }
        }
    }
    Ok(None)
}

fn d_knows(
    py: Python<'_>,
    access: &Bound<'_, PyAny>,
    node: &Bound<'_, PyAny>,
    step: &Bound<'_, PyAny>,
    slots: &Option<Vec<Slot>>,
) -> PyResult<bool> {
    let kind = step_kind(step)?;
    match kind.as_str() {
        "by_tag" => {
            let value = step_text(step)?;
            if let Some(slots) = slots {
                if slots
                    .iter()
                    .any(|(tag, _)| tag.as_deref() == Some(value.as_str()))
                {
                    return Ok(true);
                }
            }
            let descriptor = attr(node, "descriptor")?;
            Ok(d_bare_sum_tag_target(py, access, &descriptor, &value)?.is_some())
        }
        "by_field_index" => {
            let positional = slots
                .as_ref()
                .map(|slots| slots.iter().filter(|(tag, _)| tag.is_none()).count())
                .unwrap_or(0);
            let value = step_index(step)?;
            Ok(value >= 0 && (value as usize) < positional)
        }
        "at_index" => {
            let descriptor = attr(node, "descriptor")?;
            let unfolded = d_unfold(py, access, &descriptor, None)?;
            d_has_elements_by_index(py, access, unfolded.bind(py))
        }
        "at_key" => {
            let descriptor = attr(node, "descriptor")?;
            let unfolded = d_unfold(py, access, &descriptor, None)?;
            Ok(d_container_kind(py, access, unfolded.bind(py))?.as_deref() == Some("dict"))
        }
        _ => Ok(false),
    }
}

fn d_target_of(
    py: Python<'_>,
    access: &Bound<'_, PyAny>,
    node: &Bound<'_, PyAny>,
    step: &Bound<'_, PyAny>,
    slots: &Option<Vec<Slot>>,
) -> PyResult<Option<Py<PyAny>>> {
    let kind = step_kind(step)?;
    match kind.as_str() {
        "by_tag" => {
            let value = step_text(step)?;
            if let Some(slots) = slots {
                for (tag, descriptor) in slots {
                    if tag.as_deref() == Some(value.as_str()) {
                        return Ok(Some(descriptor.clone_ref(py)));
                    }
                }
            }
            let descriptor = attr(node, "descriptor")?;
            d_bare_sum_tag_target(py, access, &descriptor, &value)
        }
        "by_field_index" => {
            let value = step_index(step)?;
            let positional: Vec<&Py<PyAny>> = match slots {
                Some(slots) => slots
                    .iter()
                    .filter(|(tag, _)| tag.is_none())
                    .map(|(_, descriptor)| descriptor)
                    .collect(),
                None => Vec::new(),
            };
            if value >= 0 && (value as usize) < positional.len() {
                return Ok(Some(positional[value as usize].clone_ref(py)));
            }
            Ok(None)
        }
        "at_index" => {
            let descriptor = attr(node, "descriptor")?;
            let unfolded = d_unfold(py, access, &descriptor, None)?;
            let unfolded = unfolded.bind(py);
            if !d_has_elements_by_index(py, access, unfolded)? {
                return Ok(None);
            }
            let value = step_index(step)?;
            if descriptor_kind(unfolded)? == TUPLE {
                let elements = chain_elements(unfolded)?;
                if value >= 0 && (value as usize) < elements.len() {
                    return Ok(Some(elements[value as usize].clone_ref(py)));
                }
                return Ok(None);
            }
            if is_literal_descriptor(unfolded)? {
                let args = payload_attr(unfolded, "args")?;
                let args: Vec<Bound<'_, PyAny>> = args.try_iter()?.collect::<PyResult<Vec<_>>>()?;
                if value >= 0 && (value as usize) < args.len() {
                    return Ok(Some(args[value as usize].clone().unbind()));
                }
                return Ok(None);
            }
            let args = payload_attr(unfolded, "args")?;
            let first = args.get_item(0)?;
            Ok(Some(first.unbind()))
        }
        "at_key" => {
            let descriptor = attr(node, "descriptor")?;
            let unfolded = d_unfold(py, access, &descriptor, None)?;
            let unfolded = unfolded.bind(py);
            if d_container_kind(py, access, unfolded)?.as_deref() != Some("dict") {
                return Ok(None);
            }
            let value = step_text(step)?;
            if is_literal_descriptor(unfolded)? {
                for pair in payload_attr(unfolded, "args")?.try_iter()? {
                    let pair = pair?;
                    if let Some(found) = d_pair_of(&pair, &value)? {
                        return Ok(Some(found));
                    }
                }
                return Ok(None);
            }
            let args = payload_attr(unfolded, "args")?;
            if args.len()? < 2 {
                return Ok(None);
            }
            Ok(Some(args.get_item(1)?.unbind()))
        }
        _ => Ok(None),
    }
}

fn is_literal_descriptor(descriptor: &Bound<'_, PyAny>) -> PyResult<bool> {
    if descriptor_kind(descriptor)? != TYPE_APP {
        return Ok(false);
    }
    let name = payload_attr(descriptor, "constructor_name")?.extract::<String>()?;
    Ok(literal_kind(&name).is_some())
}

fn d_pair_of(pair: &Bound<'_, PyAny>, key: &str) -> PyResult<Option<Py<PyAny>>> {
    if descriptor_kind(pair).ok().as_deref() != Some(TUPLE) {
        return Ok(None);
    }
    let elements = match payload_attr(pair, "elements") {
        Ok(elements) => {
            let mut out = Vec::new();
            for element in elements.try_iter()? {
                out.push(element?);
            }
            out
        }
        Err(_) => return Ok(None),
    };
    if elements.len() < 2 {
        return Ok(None);
    }
    if descriptor_kind(&elements[0])? != LITERAL {
        return Ok(None);
    }
    let value = payload_attr(&elements[0], "value")?;
    let text = key_text_of(pair.py(), &value)?;
    let text = text.bind(pair.py());
    if text.is_none() || !text.eq(PyString::new(pair.py(), key))? {
        return Ok(None);
    }
    Ok(Some(elements[1].clone().unbind()))
}

// ---- the data side ----

fn d_value_at(
    py: Python<'_>,
    access: &Bound<'_, PyAny>,
    data: &Bound<'_, PyAny>,
    step: &Bound<'_, PyAny>,
    design: Option<&Bound<'_, PyAny>>,
    data_module: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<Py<PyAny>>> {
    let matched = d_match(py, access, data, step, design, data_module)?;
    Ok(as_value(py, matched))
}

fn as_value(_py: Python<'_>, piece: Option<Bound<'_, PyAny>>) -> Option<Py<PyAny>> {
    match piece {
        None => None,
        Some(piece) => {
            if pyast::class_name(&piece).ok().as_deref() == Some("Ellipsis") {
                None
            } else {
                Some(piece.unbind())
            }
        }
    }
}

fn module_or_design<'py>(
    data_module: Option<&Bound<'py, PyAny>>,
    design: Option<&Bound<'py, PyAny>>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    if let Some(module) = data_module {
        return Ok(Some(module.clone()));
    }
    match design {
        Some(design) => design_module(design),
        None => Ok(None),
    }
}

fn design_module<'py>(descriptor: &Bound<'py, PyAny>) -> PyResult<Option<Bound<'py, PyAny>>> {
    let resolvable = match opt_attr(descriptor, "resolvable_type") {
        Some(resolvable) => resolvable,
        None => return Ok(None),
    };
    Ok(opt_attr(&resolvable, "container_module"))
}

fn d_expand_data(
    py: Python<'_>,
    access: &Bound<'_, PyAny>,
    data: &Bound<'_, PyAny>,
    design: Option<&Bound<'_, PyAny>>,
    data_module: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let seen = is_empty_set(py)?.into_bound(py);
    d_expand_data_seen(py, access, data, design, &seen, data_module)
}

fn d_expand_data_seen(
    py: Python<'_>,
    access: &Bound<'_, PyAny>,
    data: &Bound<'_, PyAny>,
    design: Option<&Bound<'_, PyAny>>,
    seen: &Bound<'_, PyAny>,
    data_module: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let class = pyast::class_name(data)?;
    if class == "TypeRef" {
        let module = match module_or_design(data_module, design)? {
            Some(module) => module,
            None => return Ok(data.clone().unbind()),
        };
        let name = attr_text(data, "name")?;
        let body = match definition_body(py, &module, &name)? {
            Some(body) => body,
            None => return Ok(data.clone().unbind()),
        };
        let key = match d_data_definition_key(py, data, &module)? {
            Some(key) => key,
            None => return Ok(data.clone().unbind()),
        };
        if set_contains(seen, key.bind(py))? {
            return Ok(data.clone().unbind());
        }
        let next = new_set_with(py, seen, key.bind(py))?;
        return d_expand_data_seen(
            py,
            access,
            body.bind(py),
            design,
            next.bind(py),
            data_module,
        );
    }
    if class == "TypeApp" {
        let module = match module_or_design(data_module, design)? {
            Some(module) => module,
            None => return Ok(data.clone().unbind()),
        };
        let constructor = attr_text(data, "constructor")?;
        let definition = match generic_definition(py, &module, &constructor)? {
            Some(definition) => definition,
            None => return Ok(data.clone().unbind()),
        };
        let definition = definition.bind(py);
        let mut params = Vec::new();
        for param in attr(definition, "generic_params")?.try_iter()? {
            params.push(param?);
        }
        let mut args = Vec::new();
        for argument in attr(data, "args")?.try_iter()? {
            args.push(argument?);
        }
        if params.len() != args.len() {
            return Ok(data.clone().unbind());
        }
        let key = match d_data_definition_key(py, data, &module)? {
            Some(key) => key,
            None => return Ok(data.clone().unbind()),
        };
        if set_contains(seen, key.bind(py))? {
            return Ok(data.clone().unbind());
        }
        let bindings = PyDict::new(py);
        for (param, argument) in params.iter().zip(args.iter()) {
            bindings.set_item(param, argument)?;
        }
        let body = fill_params(py, &attr(definition, "body")?, &bindings.into_any())?;
        let next = new_set_with(py, seen, key.bind(py))?;
        return d_expand_data_seen(
            py,
            access,
            body.bind(py),
            design,
            next.bind(py),
            data_module,
        );
    }
    Ok(data.clone().unbind())
}

fn definition_body(
    py: Python<'_>,
    module: &Bound<'_, PyAny>,
    name: &str,
) -> PyResult<Option<Py<PyAny>>> {
    let resolved = type_system::module_get_type(py, module, name, None)?;
    let resolved = resolved.bind(py);
    if result_is_err(resolved)? {
        return Ok(None);
    }
    let value = result_ok_value(resolved)?;
    if !is_class(&value, "AstNodeType")? {
        return Ok(None);
    }
    let node = attr(&value, "ast_node")?;
    if is_class(&node, "TypeDefinition")? {
        return Ok(Some(attr(&node, "body")?.unbind()));
    }
    Ok(None)
}

fn generic_definition(
    py: Python<'_>,
    module: &Bound<'_, PyAny>,
    name: &str,
) -> PyResult<Option<Py<PyAny>>> {
    let resolved = type_system::module_get_type(py, module, name, None)?;
    let resolved = resolved.bind(py);
    if result_is_err(resolved)? {
        return Ok(None);
    }
    let value = result_ok_value(resolved)?;
    if !is_class(&value, "AstNodeType")? {
        return Ok(None);
    }
    let node = attr(&value, "ast_node")?;
    if is_class(&node, "GenericDefinition")? {
        return Ok(Some(node.unbind()));
    }
    Ok(None)
}

fn d_data_definition_key(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    module: &Bound<'_, PyAny>,
) -> PyResult<Option<Py<PyAny>>> {
    let class = pyast::class_name(data)?;
    if class == "TypeRef" {
        let name = attr_text(data, "name")?;
        let resolved = type_system::module_get_type(py, module, &name, None)?;
        let resolved = resolved.bind(py);
        if result_is_err(resolved)? {
            return Ok(None);
        }
        let value = result_ok_value(resolved)?;
        if !is_class(&value, "AstNodeType")? {
            return Ok(None);
        }
        let node = attr(&value, "ast_node")?;
        if is_class(&node, "TypeDefinition")? {
            return Ok(Some(key_of_kind(py, "ref", id_of(&node))?));
        }
        return Ok(None);
    }
    if class == "TypeApp" {
        let constructor = attr_text(data, "constructor")?;
        let definition = match generic_definition(py, module, &constructor)? {
            Some(definition) => definition,
            None => return Ok(None),
        };
        let definition = definition.bind(py);
        let mut params = 0;
        for _ in attr(definition, "generic_params")?.try_iter()? {
            params += 1;
        }
        let mut args = 0;
        for _ in attr(data, "args")?.try_iter()? {
            args += 1;
        }
        if params != args {
            return Ok(None);
        }
        return Ok(Some(key_of_kind(py, "app", id_of(definition))?));
    }
    Ok(None)
}

fn d_match<'py>(
    py: Python<'py>,
    access: &Bound<'py, PyAny>,
    data: &Bound<'py, PyAny>,
    step: &Bound<'py, PyAny>,
    design: Option<&Bound<'py, PyAny>>,
    data_module: Option<&Bound<'py, PyAny>>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let kind = step_kind(step)?;
    if kind == "by_tag" || kind == "by_field_index" {
        let slots = d_data_members(py, access, data, design, data_module)?;
        if let Some(slots) = &slots {
            if kind == "by_tag" {
                let value = step_text(step)?;
                for (tag, piece) in slots {
                    if tag.as_deref() == Some(value.as_str()) {
                        return Ok(Some(piece.clone_ref(py).into_bound(py)));
                    }
                }
                return Ok(None);
            }
            let positional: Vec<&Py<PyAny>> = slots
                .iter()
                .filter(|(tag, _)| tag.is_none())
                .map(|(_, piece)| piece)
                .collect();
            let value = step_index(step)?;
            if value >= 0 && (value as usize) < positional.len() {
                return Ok(Some(
                    positional[value as usize].clone_ref(py).into_bound(py),
                ));
            }
        }
        if kind == "by_field_index" {
            return d_bare_sum_piece(py, access, data, design, step_index(step)?);
        }
        return Ok(None);
    }
    if kind == "at_index" {
        let elements = match elements_of(py, data)? {
            Some(elements) => elements,
            None => return Ok(None),
        };
        let value = step_index(step)?;
        if value < 0 || (value as usize) >= elements.len() {
            return Ok(None);
        }
        return Ok(Some(elements[value as usize].clone_ref(py).into_bound(py)));
    }
    if kind == "at_key" {
        if !is_class(data, "TypeApp")? || attr_text(data, "constructor")? != "DictLiteral" {
            return Ok(None);
        }
        let value = step_text(step)?;
        for pair in attr(data, "args")?.try_iter()? {
            let pair = pair?;
            if let Some(key) = d_key_of(&pair)? {
                let text = key_text_of(py, &key)?;
                let text = text.bind(py);
                if !text.is_none() && text.eq(PyString::new(py, &value))? {
                    let elements = attr(&pair, "elements")?;
                    if elements.len()? > 1 {
                        return Ok(Some(elements.get_item(1)?));
                    }
                    return Ok(None);
                }
            }
        }
        return Ok(None);
    }
    Ok(None)
}

fn d_bare_sum_piece<'py>(
    py: Python<'py>,
    access: &Bound<'py, PyAny>,
    data: &Bound<'py, PyAny>,
    design: Option<&Bound<'py, PyAny>>,
    index: i64,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let design = match design {
        Some(design) => design,
        None => return Ok(None),
    };
    let members = match d_bare_sum(py, access, design)? {
        Some(members) => members,
        None => return Ok(None),
    };
    for (position, role, branch) in members {
        if position as i64 != index {
            continue;
        }
        if role == "inner" {
            return Ok(if is_leaf_data(data)? {
                None
            } else {
                Some(data.clone())
            });
        }
        return Ok(if d_data_fits_leaf(py, access, data, branch.bind(py))? {
            Some(data.clone())
        } else {
            None
        });
    }
    Ok(None)
}

fn is_leaf_data(data: &Bound<'_, PyAny>) -> PyResult<bool> {
    let kind = pyast::class_name(data)?;
    Ok(kind == "Constant" || kind == "Nil" || kind == "Never")
}

fn d_data_fits_leaf(
    _py: Python<'_>,
    access: &Bound<'_, PyAny>,
    data: &Bound<'_, PyAny>,
    branch: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    let data_kind = pyast::class_name(data)?;
    let branch_kind = descriptor_kind(branch)?;
    if data_kind == "Nil" {
        return Ok(branch_kind == NIL);
    }
    if data_kind == "Never" {
        return Ok(branch_kind == NEVER);
    }
    if data_kind != "Constant" {
        return Ok(false);
    }
    let value = attr(data, "value")?;
    if branch_kind == LITERAL {
        let other = payload_attr(branch, "value")?;
        return Ok(other.get_type().is(&value.get_type()) && other.eq(value)?);
    }
    if branch_kind != TYPE_REF {
        return Ok(false);
    }
    let name = payload_attr(branch, "type_name")?.extract::<String>()?;
    let _ = access;
    let kind = scalar_name_of(&value)?;
    Ok(kind == Some(name.as_str())
        || (FIXED_INT_NAMES.contains(&name.as_str()) && kind == Some("int")))
}

fn d_key_of<'py>(pair: &Bound<'py, PyAny>) -> PyResult<Option<Bound<'py, PyAny>>> {
    if !is_class(pair, "Tuple")? {
        return Ok(None);
    }
    let elements = attr(pair, "elements")?;
    let elements: Vec<Bound<'_, PyAny>> = elements.try_iter()?.collect::<PyResult<Vec<_>>>()?;
    if elements.is_empty() {
        return Ok(None);
    }
    if !is_class(&elements[0], "Constant")? {
        return Ok(None);
    }
    Ok(Some(attr(&elements[0], "value")?))
}

fn elements_of<'py>(py: Python<'py>, data: &Bound<'py, PyAny>) -> PyResult<Option<Vec<Py<PyAny>>>> {
    let _ = py;
    if is_class(data, "Tuple")? {
        let mut out = Vec::new();
        for element in attr(data, "elements")?.try_iter()? {
            out.push(element?.unbind());
        }
        return Ok(Some(out));
    }
    if is_class(data, "TypeApp")? {
        let constructor = attr_text(data, "constructor")?;
        if LITERAL_CTORS.contains(&constructor.as_str()) {
            let mut out = Vec::new();
            for element in attr(data, "args")?.try_iter()? {
                out.push(element?.unbind());
            }
            return Ok(Some(out));
        }
    }
    Ok(None)
}

// ---- the map: units ----

fn config_is_unit_name(py: Python<'_>, access: &Bound<'_, PyAny>, name: &str) -> PyResult<bool> {
    let _ = py;
    let config = attr(access, "config")?;
    Ok(config
        .call_method1("is_unit_name", (PyString::new(access.py(), name),))?
        .extract()?)
}

fn config_contains(access: &Bound<'_, PyAny>, attribute: &str, name: &str) -> PyResult<bool> {
    let config = attr(access, "config")?;
    let names = attr(&config, attribute)?;
    names
        .call_method1("__contains__", (PyString::new(access.py(), name),))?
        .extract()
}

fn d_is_unit_descriptor(
    py: Python<'_>,
    access: &Bound<'_, PyAny>,
    descriptor: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    let descriptor = d_unfold(py, access, descriptor, None)?;
    let descriptor = descriptor.bind(py);
    let kind = descriptor_kind(descriptor)?;
    if kind == NIL || kind == NEVER {
        return Ok(true);
    }
    if kind == TYPE_REF {
        let name = payload_attr(descriptor, "type_name")?.extract::<String>()?;
        return config_is_unit_name(py, access, &name);
    }
    Ok(false)
}

fn d_is_product_unit(
    py: Python<'_>,
    access: &Bound<'_, PyAny>,
    descriptor: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    let unfolded = d_unfold(py, access, descriptor, None)?;
    let unfolded = unfolded.bind(py);
    let kind = descriptor_kind(unfolded)?;
    if kind == NIL {
        return Ok(true);
    }
    if kind == TYPE_REF {
        let name = payload_attr(unfolded, "type_name")?.extract::<String>()?;
        return config_contains(access, "nil_eqv", &name);
    }
    Ok(false)
}

fn d_is_unit_data(
    py: Python<'_>,
    access: &Bound<'_, PyAny>,
    node: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    let kind = pyast::class_name(node)?;
    if kind == "Nil" || kind == "Never" {
        return Ok(true);
    }
    if kind == "TypeRef" {
        let name = attr_text(node, "name")?;
        return config_is_unit_name(py, access, &name);
    }
    Ok(false)
}

fn d_is_product_unit_data(
    _py: Python<'_>,
    access: &Bound<'_, PyAny>,
    node: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    let kind = pyast::class_name(node)?;
    if kind == "Nil" {
        return Ok(true);
    }
    if kind == "TypeRef" {
        let name = attr_text(node, "name")?;
        return config_contains(access, "nil_eqv", &name);
    }
    Ok(false)
}

// ---- the data side: members ----

fn d_data_members(
    py: Python<'_>,
    access: &Bound<'_, PyAny>,
    data: &Bound<'_, PyAny>,
    design: Option<&Bound<'_, PyAny>>,
    data_module: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<Vec<Slot>>> {
    let expanded = d_expand_data(py, access, data, design, data_module)?;
    if let Some(design) = design {
        let unfolded = d_unfold(py, access, design, None)?;
        if descriptor_kind(unfolded.bind(py))? == PRODUCT {
            let members =
                d_product_data_members(py, access, expanded.bind(py), design, None, data_module)?;
            return Ok(Some(members));
        }
    }
    let elements = match flatten(py, expanded.bind(py))? {
        Some(elements) => elements,
        None => {
            if is_class(expanded.bind(py), "Tagged")? {
                return Ok(Some(vec![(
                    Some(attr_text(expanded.bind(py), "tag")?),
                    attr(expanded.bind(py), "type")?.unbind(),
                )]));
            }
            return Ok(None);
        }
    };
    let mut elements = elements;
    if !elements.is_empty() && d_is_unit_data(py, access, elements[0].bind(py))? {
        elements.remove(0);
    }
    let mut slots = Vec::new();
    for element in elements {
        let element = element.bind(py);
        if is_class(element, "Tagged")? {
            slots.push((
                Some(attr_text(element, "tag")?),
                attr(element, "type")?.unbind(),
            ));
        } else {
            slots.push((None, element.clone().unbind()));
        }
    }
    Ok(Some(slots))
}

fn d_product_data_members(
    py: Python<'_>,
    access: &Bound<'_, PyAny>,
    data: &Bound<'_, PyAny>,
    design: &Bound<'_, PyAny>,
    seen: Option<&Bound<'_, PyAny>>,
    data_module: Option<&Bound<'_, PyAny>>,
) -> PyResult<Vec<Slot>> {
    let seen_owned;
    let seen: &Bound<'_, PyAny> = match seen {
        Some(seen) => seen,
        None => {
            seen_owned = is_empty_set(py)?;
            seen_owned.bind(py)
        }
    };
    let elements = match flatten(py, data)? {
        Some(elements) => elements,
        None => {
            if is_class(data, "Tagged")? {
                return Ok(vec![(
                    Some(attr_text(data, "tag")?),
                    attr(data, "type")?.unbind(),
                )]);
            }
            return Ok(Vec::new());
        }
    };
    let module = module_or_design(data_module, Some(design))?;
    let mut out = Vec::new();
    for element in elements {
        let element = element.bind(py);
        if is_class(element, "Tagged")? {
            out.push((
                Some(attr_text(element, "tag")?),
                attr(element, "type")?.unbind(),
            ));
            continue;
        }
        let key = match &module {
            Some(module) => d_data_definition_key(py, element, module)?,
            None => None,
        };
        let expanded = match &key {
            Some(key) if !set_contains(seen, key.bind(py))? => {
                d_expand_data(py, access, element, Some(design), data_module)?
            }
            _ => element.clone().unbind(),
        };
        let expanded = expanded.bind(py);
        if is_class(expanded, "Product")? || is_class(expanded, "ProductChain")? {
            let next_seen = match &key {
                Some(key) => Some(new_set_with(py, seen, key.bind(py))?),
                None => None,
            };
            let next_seen = next_seen.as_ref().map(|value| value.bind(py));
            out.extend(d_product_data_members(
                py,
                access,
                expanded,
                design,
                next_seen,
                data_module,
            )?);
        } else if is_class(expanded, "Tagged")? {
            out.push((
                Some(attr_text(expanded, "tag")?),
                attr(expanded, "type")?.unbind(),
            ));
        } else if d_is_product_unit_data(py, access, expanded)? {
            continue;
        } else {
            out.push((None, element.clone().unbind()));
        }
    }
    Ok(out)
}

// ---- the data side: flattening ----

fn flatten(py: Python<'_>, node: &Bound<'_, PyAny>) -> PyResult<Option<Vec<Py<PyAny>>>> {
    let kind = pyast::class_name(node)?;
    if kind == "ProductChain" || kind == "SumChain" || kind == "ExponentChain" {
        let mut out = Vec::new();
        for element in attr(node, "elements")?.try_iter()? {
            out.push(element?.unbind());
        }
        return Ok(Some(out));
    }
    if kind == "Product" || kind == "Sum" {
        let (head, tail) = if kind == "Product" {
            (attr(node, "left")?, attr(node, "right")?)
        } else {
            (attr(node, "left")?, attr(node, "right")?)
        };
        let mut elements = match flatten(py, &head)? {
            Some(elements) => elements,
            None => vec![head.unbind()],
        };
        elements.push(tail.unbind());
        return Ok(Some(elements));
    }
    if kind == "Exponent" {
        let result = attr(node, "result")?;
        let argument = attr(node, "argument")?;
        let mut elements = match flatten(py, &result)? {
            Some(elements) => elements,
            None => vec![result.unbind()],
        };
        elements.push(argument.unbind());
        return Ok(Some(elements));
    }
    Ok(None)
}

// ---- filling generic parameters in a data body ----

fn fill_params(
    py: Python<'_>,
    body: &Bound<'_, PyAny>,
    bindings: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let copied = py
        .import("copy")?
        .getattr("deepcopy")?
        .call1((body.clone(),))?;
    transform_node(py, &copied, bindings)
}

fn transform_node(
    py: Python<'_>,
    node: &Bound<'_, PyAny>,
    bindings: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    if is_class(node, "TypeRef")? {
        let name = attr(node, "name")?;
        let bound = bindings.call_method1("get", (name,))?;
        if !bound.is_none() {
            return Ok(bound.unbind());
        }
        return Ok(node.clone().unbind());
    }
    if !node.get_type().hasattr("_fields")? {
        return Ok(node.clone().unbind());
    }
    for field in node.get_type().getattr("_fields")?.try_iter()? {
        let field = field?;
        let field = field.extract::<String>()?;
        let value = attr(node, &field)?;
        if value.is_instance_of::<PyList>() {
            let mut items = Vec::new();
            for item in value.try_iter()? {
                let item = item?;
                if item.get_type().hasattr("_fields")? {
                    items.push(transform_node(py, &item, bindings)?);
                } else {
                    items.push(item.unbind());
                }
            }
            node.setattr(&field, pyast::pylist(py, items)?)?;
        } else if value.get_type().hasattr("_fields")? {
            node.setattr(&field, transform_node(py, &value, bindings)?)?;
        }
    }
    Ok(node.clone().unbind())
}

// ----------------------------------------------------------------------
// The protocol cells
// ----------------------------------------------------------------------

fn d_root(
    py: Python<'_>,
    access: &Bound<'_, PyAny>,
    definition: &Bound<'_, PyAny>,
    data: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let body = attr(definition, "body")?;
    let node = attr(data, "node")?;
    let module = attr(data, "module")?;
    let object = make_vobject(py, access, &body, &node, Vec::new(), Some(&module), None)?;
    ok(py, object)
}

fn d_has(
    py: Python<'_>,
    access: &Bound<'_, PyAny>,
    node: &Bound<'_, PyAny>,
    step: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let descriptor = attr(node, "descriptor")?;
    let slots = d_design_members(py, access, &descriptor)?;
    if !d_knows(py, access, node, step, &slots)? {
        return ok(py, pyast::pybool(py, false));
    }
    let data = attr(node, "data")?;
    let data_module = opt_attr(node, "data_module");
    let piece = d_value_at(
        py,
        access,
        &data,
        step,
        Some(&descriptor),
        data_module.as_ref(),
    )?;
    ok(py, pyast::pybool(py, piece.is_some()))
}

fn d_get(
    py: Python<'_>,
    access: &Bound<'_, PyAny>,
    node: &Bound<'_, PyAny>,
    step: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let descriptor = attr(node, "descriptor")?;
    let slots = d_design_members(py, access, &descriptor)?;
    let missing = || {
        err(
            py,
            &format!(
                "the design has no such address: {} ({})",
                pyrepr(py, step),
                pyrepr(py, node)
            ),
        )
    };
    if !d_knows(py, access, node, step, &slots)? {
        return missing();
    }
    let target = match d_target_of(py, access, node, step, &slots)? {
        Some(target) => target,
        None => return missing(),
    };
    let data = attr(node, "data")?;
    let data_module = opt_attr(node, "data_module");
    let piece = d_value_at(
        py,
        access,
        &data,
        step,
        Some(&descriptor),
        data_module.as_ref(),
    )?;
    let piece = match piece {
        Some(piece) => piece,
        None => return ok(py, py.None()),
    };
    let mut path = Vec::new();
    for item in attr(node, "path")?.try_iter()? {
        path.push(item?.unbind());
    }
    path.push(step.clone().unbind());
    let object = make_vobject(
        py,
        access,
        target.bind(py),
        piece.bind(py),
        path,
        data_module.as_ref(),
        None,
    )?;
    ok(py, object)
}

fn d_leaf(py: Python<'_>, node: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let data = attr(node, "data")?;
    if is_class(&data, "Constant")? {
        return ok(py, attr(&data, "value")?.unbind());
    }
    if is_class(&data, "Nil")? {
        return ok(py, py.None());
    }
    err(
        py,
        &format!("this piece is not a leaf: {}", pyrepr(py, node)),
    )
}

fn d_length(py: Python<'_>, node: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let data = attr(node, "data")?;
    match elements_of(py, &data)? {
        Some(elements) => ok(py, pyint(py, elements.len())),
        None => err(
            py,
            &format!("this piece is not a container: {}", pyrepr(py, node)),
        ),
    }
}

fn d_keys(py: Python<'_>, node: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let data = attr(node, "data")?;
    if !is_class(&data, "TypeApp")? || attr_text(&data, "constructor")? != "DictLiteral" {
        return err(
            py,
            &format!("this piece is not a dict: {}", pyrepr(py, node)),
        );
    }
    let mut keys = Vec::new();
    for pair in attr(&data, "args")?.try_iter()? {
        let pair = pair?;
        match d_key_of(&pair)? {
            Some(key) => keys.push(key_text_of(py, &key)?),
            None => {
                return err(
                    py,
                    &format!("this dict key is not a literal: {}", pyrepr(py, node)),
                )
            }
        }
    }
    ok(py, make_list(py, keys)?)
}

fn d_resolve(
    py: Python<'_>,
    access: &Bound<'_, PyAny>,
    node: &Bound<'_, PyAny>,
    path: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let mut current: Option<Py<PyAny>> = Some(node.clone().unbind());
    for step in path.try_iter()? {
        let step = step?;
        let current_bound = match &current {
            Some(current) => current.clone_ref(py).into_bound(py),
            None => return err(py, "this step has no value"),
        };
        let given = d_get(py, access, &current_bound, &step)?;
        if result_is_err(given.bind(py))? {
            return Ok(given);
        }
        let value = result_ok_value(given.bind(py))?;
        current = if value.is_none() {
            None
        } else {
            Some(value.unbind())
        };
    }
    ok(py, current.unwrap_or_else(|| py.None()))
}

fn d_get_by_path(
    py: Python<'_>,
    access: &Bound<'_, PyAny>,
    node: &Bound<'_, PyAny>,
    path: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let resolved = d_resolve(py, access, node, path)?;
    let resolved_bound = resolved.bind(py);
    if result_is_err(resolved_bound)? {
        return Ok(resolved);
    }
    let value = result_ok_value(resolved_bound)?;
    if value.is_none() {
        return err(py, "this step has no value");
    }
    d_leaf(py, &value)
}

fn d_list_fields(
    py: Python<'_>,
    access: &Bound<'_, PyAny>,
    node: &Bound<'_, PyAny>,
    definition: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let mut out = Vec::new();
    let mut positional = 0i64;
    let body = attr(definition, "body")?;
    if let Some(members) = d_design_members(py, access, &body)? {
        for (tag, _descriptor) in members {
            let step = match tag {
                Some(tag) => make_step(py, "by_tag", pyast::pystr(py, &tag))?,
                None => {
                    let step = make_step(py, "by_field_index", pyint(py, positional as usize))?;
                    positional += 1;
                    step
                }
            };
            let given = d_get(py, access, node, step.bind(py))?;
            let given_bound = given.bind(py);
            if !result_is_err(given_bound)? {
                let value = result_ok_value(given_bound)?;
                if !value.is_none() {
                    out.push(value.unbind());
                }
            }
        }
    }
    ok(py, make_list(py, out)?)
}

fn d_unwrap(py: Python<'_>, given: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    if result_is_err(given)? {
        return Err(reflect_error(py, &result_err_message(given)?));
    }
    let value = result_ok_value(given)?;
    if value.is_none() {
        return Err(reflect_error(py, "this piece has no value"));
    }
    Ok(value.unbind())
}

fn d_walk(
    py: Python<'_>,
    access: &Bound<'_, PyAny>,
    node: &Bound<'_, PyAny>,
) -> PyResult<Vec<Py<PyAny>>> {
    let mut out = vec![node.clone().unbind()];
    let steps = member_steps(py, access, node)?;
    for (_tag, step, _descriptor) in steps {
        let given = d_get(py, access, node, step.bind(py))?;
        let given_bound = given.bind(py);
        if !result_is_err(given_bound)? {
            let value = result_ok_value(given_bound)?;
            if !value.is_none() {
                out.extend(d_walk(py, access, &value)?);
            }
        }
    }
    let descriptor = attr(node, "descriptor")?;
    let unfolded = d_unfold(py, access, &descriptor, None)?;
    let unfolded = unfolded.bind(py);
    if d_container_kind(py, access, unfolded)?.as_deref() == Some("dict") {
        let given_keys = d_keys(py, node)?;
        let given_keys = given_keys.bind(py);
        if !result_is_err(given_keys)? {
            for key in result_ok_value(given_keys)?.try_iter()? {
                let key = key?;
                let step = make_step(py, "at_key", key.unbind())?;
                let given = d_get(py, access, node, step.bind(py))?;
                let given_bound = given.bind(py);
                if !result_is_err(given_bound)? {
                    let value = result_ok_value(given_bound)?;
                    if !value.is_none() {
                        out.extend(d_walk(py, access, &value)?);
                    }
                }
            }
        }
    } else if d_has_elements_by_index(py, access, unfolded)? {
        let given_length = d_length(py, node)?;
        let given_length = given_length.bind(py);
        if !result_is_err(given_length)? {
            let length = result_ok_value(given_length)?.extract::<usize>()?;
            for index in 0..length {
                let step = make_step(py, "at_index", pyint(py, index))?;
                let given = d_get(py, access, node, step.bind(py))?;
                let given_bound = given.bind(py);
                if !result_is_err(given_bound)? {
                    let value = result_ok_value(given_bound)?;
                    if !value.is_none() {
                        out.extend(d_walk(py, access, &value)?);
                    }
                }
            }
        }
    }
    Ok(out)
}

fn member_steps(
    py: Python<'_>,
    access: &Bound<'_, PyAny>,
    node: &Bound<'_, PyAny>,
) -> PyResult<Vec<(Option<String>, Py<PyAny>, Py<PyAny>)>> {
    let descriptor = attr(node, "descriptor")?;
    let members = d_design_members(py, access, &descriptor)?;
    let mut out = Vec::new();
    let mut positional = 0i64;
    if let Some(members) = members {
        for (tag, member) in members {
            match tag {
                Some(tag) => {
                    let step = make_step(py, "by_tag", pyast::pystr(py, &tag))?;
                    out.push((Some(tag), step, member));
                }
                None => {
                    let step = make_step(py, "by_field_index", pyint(py, positional as usize))?;
                    positional += 1;
                    out.push((None, step, member));
                }
            }
        }
    }
    Ok(out)
}

fn d_dir(py: Python<'_>, node: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    let mut names: Vec<String> = [
        "value",
        "leaf",
        "is_list",
        "is_set",
        "is_dict",
        "keys",
        "values",
        "items",
        "path",
        "descriptor",
        "data",
    ]
    .iter()
    .map(|name| name.to_string())
    .collect();
    let access = attr(node, "_access")?;
    let mut positional = 0;
    for (tag, _step, _descriptor) in member_steps(py, &access, node)? {
        match tag {
            Some(tag) => {
                let short = tag.strip_prefix('$').unwrap_or(&tag);
                names.push(format!("get_{short}"));
                names.push(format!("has_{short}"));
                names.push(format!("try_get_{short}"));
            }
            None => {
                names.push(format!("get_field_{positional}"));
                names.push(format!("has_field_{positional}"));
                names.push(format!("try_get_field_{positional}"));
                positional += 1;
            }
        }
    }
    names.sort();
    Ok(names)
}

// ----------------------------------------------------------------------
// The Python methods on the accessor
// ----------------------------------------------------------------------

fn access_object<'py>(py: Python<'py>, slf: PyRef<'py, VibaAccess>) -> Bound<'py, PyAny> {
    let value: Py<VibaAccess> = slf.into();
    value.into_bound(py).into_any()
}

#[pymethods]
impl VibaAccess {
    #[new]
    fn new(config: Py<PyAny>) -> Self {
        VibaAccess { config }
    }

    #[getter]
    fn config(&self, py: Python<'_>) -> Py<PyAny> {
        self.config.clone_ref(py)
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        format!("VibaAccess({})", pyrepr(py, self.config_value(py).bind(py)))
    }

    fn root(
        slf: PyRef<'_, Self>,
        definition: &Bound<'_, PyAny>,
        data: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let access = access_object(py, slf);
        d_root(py, &access, definition, data)
    }

    fn has(
        slf: PyRef<'_, Self>,
        node: &Bound<'_, PyAny>,
        step: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let access = access_object(py, slf);
        d_has(py, &access, node, step)
    }

    fn get(
        slf: PyRef<'_, Self>,
        node: &Bound<'_, PyAny>,
        step: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let access = access_object(py, slf);
        d_get(py, &access, node, step)
    }

    fn leaf(slf: PyRef<'_, Self>, node: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let _access = access_object(py, slf);
        d_leaf(py, node)
    }

    fn length(slf: PyRef<'_, Self>, node: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let _access = access_object(py, slf);
        d_length(py, node)
    }

    fn keys(slf: PyRef<'_, Self>, node: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let _access = access_object(py, slf);
        d_keys(py, node)
    }

    fn resolve(
        slf: PyRef<'_, Self>,
        node: &Bound<'_, PyAny>,
        path: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let access = access_object(py, slf);
        d_resolve(py, &access, node, path)
    }

    fn get_by_path(
        slf: PyRef<'_, Self>,
        node: &Bound<'_, PyAny>,
        path: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let access = access_object(py, slf);
        d_get_by_path(py, &access, node, path)
    }

    fn list_fields(
        slf: PyRef<'_, Self>,
        node: &Bound<'_, PyAny>,
        definition: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let access = access_object(py, slf);
        d_list_fields(py, &access, node, definition)
    }

    fn _unwrap(slf: PyRef<'_, Self>, given: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let _access = access_object(py, slf);
        d_unwrap(py, given)
    }

    #[pyo3(signature = (descriptor, seen = None))]
    fn unfold(
        slf: PyRef<'_, Self>,
        descriptor: &Bound<'_, PyAny>,
        seen: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let access = access_object(py, slf);
        d_unfold(py, &access, descriptor, seen)
    }

    fn _unfold_seen(
        slf: PyRef<'_, Self>,
        descriptor: &Bound<'_, PyAny>,
        seen: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let access = access_object(py, slf);
        let (result, seen) = d_unfold_seen(py, &access, descriptor, seen)?;
        make_tuple(py, vec![result, seen])
    }

    #[pyo3(signature = (descriptor, seen))]
    fn _decided(
        slf: PyRef<'_, Self>,
        descriptor: &Bound<'_, PyAny>,
        seen: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let access = access_object(py, slf);
        match d_decided(py, &access, descriptor, seen)? {
            Some((result, seen)) => make_tuple(py, vec![result, seen]),
            None => Ok(py.None()),
        }
    }

    #[pyo3(signature = (descriptor, seen))]
    fn _decided_member(
        slf: PyRef<'_, Self>,
        descriptor: &Bound<'_, PyAny>,
        seen: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let access = access_object(py, slf);
        match d_decided_member(py, &access, descriptor, seen)? {
            Some((result, seen)) => make_tuple(py, vec![result, seen]),
            None => Ok(py.None()),
        }
    }

    fn _definition_target(
        slf: PyRef<'_, Self>,
        descriptor: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let _access = access_object(py, slf);
        match d_definition_target(py, descriptor)? {
            Some((key, target)) => make_tuple(py, vec![key, target]),
            None => Ok(py.None()),
        }
    }

    fn _source_target(
        slf: PyRef<'_, Self>,
        descriptor: &Bound<'_, PyAny>,
        name: &str,
    ) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let _access = access_object(py, slf);
        match d_source_target(py, descriptor, name)? {
            Some(target) => Ok(target),
            None => Ok(py.None()),
        }
    }

    fn _apply_generic(
        slf: PyRef<'_, Self>,
        descriptor: &Bound<'_, PyAny>,
        target: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let _access = access_object(py, slf);
        match d_apply_generic(py, descriptor, target)? {
            Some(applied) => Ok(applied),
            None => Ok(py.None()),
        }
    }

    fn _substitute(
        slf: PyRef<'_, Self>,
        descriptor: &Bound<'_, PyAny>,
        bindings: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let _access = access_object(py, slf);
        d_substitute(py, descriptor, bindings)
    }

    fn carries_nil(slf: PyRef<'_, Self>, descriptor: &Bound<'_, PyAny>) -> PyResult<bool> {
        let py = slf.py();
        let access = access_object(py, slf);
        d_carries_nil(py, &access, descriptor)
    }

    fn container_kind(
        slf: PyRef<'_, Self>,
        descriptor: &Bound<'_, PyAny>,
    ) -> PyResult<Option<String>> {
        let py = slf.py();
        let access = access_object(py, slf);
        d_container_kind(py, &access, descriptor)
    }

    fn _has_elements_by_index(
        slf: PyRef<'_, Self>,
        descriptor: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        let py = slf.py();
        let access = access_object(py, slf);
        d_has_elements_by_index(py, &access, descriptor)
    }

    fn members(slf: PyRef<'_, Self>, node: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let access = access_object(py, slf);
        let descriptor = attr(node, "descriptor")?;
        slots_to_py(py, d_design_members(py, &access, &descriptor)?)
    }

    fn members_of(slf: PyRef<'_, Self>, descriptor: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let access = access_object(py, slf);
        slots_to_py(py, d_design_members(py, &access, descriptor)?)
    }

    fn member_steps(slf: PyRef<'_, Self>, node: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let access = access_object(py, slf);
        let steps = member_steps(py, &access, node)?;
        let mut items = Vec::new();
        for (tag, step, descriptor) in steps {
            let tag = match tag {
                Some(tag) => pyast::pystr(py, &tag),
                None => py.None(),
            };
            items.push(make_tuple(py, vec![tag, step, descriptor])?);
        }
        make_list(py, items)
    }

    fn _design_members(slf: PyRef<'_, Self>, descriptor: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let access = access_object(py, slf);
        slots_to_py(py, d_design_members(py, &access, descriptor)?)
    }

    #[pyo3(signature = (descriptor, seen = None, cycles = None))]
    fn _product_members(
        slf: PyRef<'_, Self>,
        descriptor: &Bound<'_, PyAny>,
        seen: Option<&Bound<'_, PyAny>>,
        mut cycles: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let access = access_object(py, slf);
        let mut collected: Vec<String> = Vec::new();
        let collect = cycles.is_some();
        let slots = d_product_members(
            py,
            &access,
            descriptor,
            seen,
            Some(&mut collected),
            Some(collect),
        )?;
        if let Some(cycles) = cycles.as_mut() {
            for name in collected {
                cycles.call_method1("append", (pyast::pystr(py, &name),))?;
            }
        }
        slots_to_py(py, Some(slots))
    }

    fn inline_cycle(
        slf: PyRef<'_, Self>,
        descriptor: &Bound<'_, PyAny>,
    ) -> PyResult<Option<String>> {
        let py = slf.py();
        let access = access_object(py, slf);
        d_inline_cycle(py, &access, descriptor)
    }

    fn _bare_sum(slf: PyRef<'_, Self>, descriptor: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let access = access_object(py, slf);
        match d_bare_sum(py, &access, descriptor)? {
            None => Ok(py.None()),
            Some(members) => {
                let mut items = Vec::new();
                for (index, role, branch) in members {
                    items.push(make_tuple(
                        py,
                        vec![pyint(py, index), pyast::pystr(py, &role), branch],
                    )?);
                }
                make_list(py, items)
            }
        }
    }

    fn _branch_role(slf: PyRef<'_, Self>, descriptor: &Bound<'_, PyAny>) -> PyResult<String> {
        let py = slf.py();
        let access = access_object(py, slf);
        d_branch_role(py, &access, descriptor)
    }

    fn _bare_sum_tag_target(
        slf: PyRef<'_, Self>,
        descriptor: &Bound<'_, PyAny>,
        tag: &str,
    ) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let access = access_object(py, slf);
        match d_bare_sum_tag_target(py, &access, descriptor, tag)? {
            Some(target) => Ok(target),
            None => Ok(py.None()),
        }
    }

    fn _knows(
        slf: PyRef<'_, Self>,
        node: &Bound<'_, PyAny>,
        step: &Bound<'_, PyAny>,
        slots: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        let py = slf.py();
        let access = access_object(py, slf);
        let slots = optional_slots(slots)?;
        d_knows(py, &access, node, step, &slots)
    }

    fn target_of(
        slf: PyRef<'_, Self>,
        node: &Bound<'_, PyAny>,
        step: &Bound<'_, PyAny>,
        slots: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let access = access_object(py, slf);
        let slots = optional_slots(slots)?;
        match d_target_of(py, &access, node, step, &slots)? {
            Some(target) => Ok(target),
            None => Ok(py.None()),
        }
    }

    fn _value_at(
        slf: PyRef<'_, Self>,
        data: &Bound<'_, PyAny>,
        step: &Bound<'_, PyAny>,
        design: Option<&Bound<'_, PyAny>>,
        data_module: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let access = access_object(py, slf);
        match d_value_at(py, &access, data, step, design, data_module)? {
            Some(piece) => Ok(piece),
            None => Ok(py.None()),
        }
    }

    #[pyo3(signature = (data, design, data_module = None))]
    fn _expand_data(
        slf: PyRef<'_, Self>,
        data: &Bound<'_, PyAny>,
        design: Option<&Bound<'_, PyAny>>,
        data_module: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let access = access_object(py, slf);
        d_expand_data(py, &access, data, design, data_module)
    }

    #[pyo3(signature = (data, design, seen, data_module = None))]
    fn _expand_data_seen(
        slf: PyRef<'_, Self>,
        data: &Bound<'_, PyAny>,
        design: Option<&Bound<'_, PyAny>>,
        seen: &Bound<'_, PyAny>,
        data_module: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let access = access_object(py, slf);
        d_expand_data_seen(py, &access, data, design, seen, data_module)
    }

    fn _match(
        slf: PyRef<'_, Self>,
        data: &Bound<'_, PyAny>,
        step: &Bound<'_, PyAny>,
        design: Option<&Bound<'_, PyAny>>,
        data_module: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let access = access_object(py, slf);
        match d_match(py, &access, data, step, design, data_module)? {
            Some(piece) => Ok(piece.unbind()),
            None => Ok(py.None()),
        }
    }

    fn _bare_sum_piece(
        slf: PyRef<'_, Self>,
        data: &Bound<'_, PyAny>,
        design: Option<&Bound<'_, PyAny>>,
        index: i64,
    ) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let access = access_object(py, slf);
        match d_bare_sum_piece(py, &access, data, design, index)? {
            Some(piece) => Ok(piece.unbind()),
            None => Ok(py.None()),
        }
    }

    fn _data_fits_leaf(
        slf: PyRef<'_, Self>,
        data: &Bound<'_, PyAny>,
        branch: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        let py = slf.py();
        let access = access_object(py, slf);
        d_data_fits_leaf(py, &access, data, branch)
    }

    fn _key_of(slf: PyRef<'_, Self>, pair: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let _access = access_object(py, slf);
        match d_key_of(&pair)? {
            Some(key) => Ok(key.unbind()),
            None => Ok(py.None()),
        }
    }

    fn _elements_of(slf: PyRef<'_, Self>, data: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let _access = access_object(py, slf);
        match elements_of(py, data)? {
            Some(elements) => make_list(py, elements),
            None => Ok(py.None()),
        }
    }

    fn _is_unit_descriptor(slf: PyRef<'_, Self>, descriptor: &Bound<'_, PyAny>) -> PyResult<bool> {
        let py = slf.py();
        let access = access_object(py, slf);
        d_is_unit_descriptor(py, &access, descriptor)
    }

    fn _is_product_unit(slf: PyRef<'_, Self>, descriptor: &Bound<'_, PyAny>) -> PyResult<bool> {
        let py = slf.py();
        let access = access_object(py, slf);
        d_is_product_unit(py, &access, descriptor)
    }

    fn _is_unit_data(slf: PyRef<'_, Self>, node: &Bound<'_, PyAny>) -> PyResult<bool> {
        let py = slf.py();
        let access = access_object(py, slf);
        d_is_unit_data(py, &access, node)
    }

    fn _is_product_unit_data(slf: PyRef<'_, Self>, node: &Bound<'_, PyAny>) -> PyResult<bool> {
        let py = slf.py();
        let access = access_object(py, slf);
        d_is_product_unit_data(py, &access, node)
    }

    fn _data_members(
        slf: PyRef<'_, Self>,
        data: &Bound<'_, PyAny>,
        design: Option<&Bound<'_, PyAny>>,
        data_module: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let access = access_object(py, slf);
        slots_to_py(py, d_data_members(py, &access, data, design, data_module)?)
    }

    fn _product_data_members(
        slf: PyRef<'_, Self>,
        data: &Bound<'_, PyAny>,
        design: &Bound<'_, PyAny>,
        seen: Option<&Bound<'_, PyAny>>,
        data_module: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let access = access_object(py, slf);
        slots_to_py(
            py,
            Some(d_product_data_members(
                py,
                &access,
                data,
                design,
                seen,
                data_module,
            )?),
        )
    }

    fn _data_definition_key(
        slf: PyRef<'_, Self>,
        data: &Bound<'_, PyAny>,
        module: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let _access = access_object(py, slf);
        match d_data_definition_key(py, data, module)? {
            Some(key) => Ok(key),
            None => Ok(py.None()),
        }
    }

    fn _walk(slf: PyRef<'_, Self>, node: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let access = access_object(py, slf);
        let walked = d_walk(py, &access, node)?;
        make_list(py, walked)
    }
}

// ----------------------------------------------------------------------
// The module's own helpers
// ----------------------------------------------------------------------

#[pyfunction]
fn reflect_type_name_py(value: &Bound<'_, PyAny>) -> PyResult<String> {
    type_name_of(value)
}

#[pyfunction]
fn reflect_tag_of_py(py: Python<'_>, name: &Bound<'_, PyAny>) -> PyResult<String> {
    tag_of(py, name)
}

#[pyfunction]
fn reflect_key_text_py(py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    key_text_of(py, key)
}

#[pyfunction]
fn reflect_scalar_name_py(value: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
    Ok(scalar_name_of(value)?.map(|name| name.to_string()))
}

#[pyfunction]
fn reflect_dir_py(py: Python<'_>, node: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    d_dir(py, node)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<VibaAccess>()?;
    m.add_function(wrap_pyfunction!(reflect_type_name_py, m)?)?;
    m.add_function(wrap_pyfunction!(reflect_tag_of_py, m)?)?;
    m.add_function(wrap_pyfunction!(reflect_key_text_py, m)?)?;
    m.add_function(wrap_pyfunction!(reflect_scalar_name_py, m)?)?;
    m.add_function(wrap_pyfunction!(reflect_dir_py, m)?)?;
    Ok(())
}
