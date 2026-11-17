//! viba.interpret — the interpreter runtime (was viba/interpret.py).
//!
//! The runtime runs one viba module: a file is a function (`__decl__` is that
//! function, the environment among its parameters) and its output is
//! `__impl__`. What it answers is the declared `InterpretResult` as one node of
//! viba data: `$ok` carries the answer, `$err` why the run stopped.
//!
//! The design-side modules (`rs_viba.viba_ast`, `rs_viba.type`,
//! `rs_viba.partial`, `rs_viba.pattern`, `rs_viba.reflect`,
//! `rs_viba.viba_type_descriptor`, `rs_viba.serialize`, `rs_viba.is_sub_type`)
//! are Rust as well; this layer takes their Python API, the same way the Python
//! original took `viba.*`. What lives here is the run itself: the host side of
//! an environment, the calls it makes, and what it answers.
//!
//! The data classes this layer threads (`_Runner`, `_Activation`, `_Pending`,
//! `_VibaData`, ...) are defined in `python/rs_viba/interpret.py`: they are the
//! Python-visible spelling of a run's data, and every operation on them is a
//! function below.

use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyAny, PyCFunction, PyDict, PyList, PyModule, PyString, PyTuple};

type P = Py<PyAny>;

const OK_TAG: &str = "$ok";
const ERR_TAG: &str = "$err";
const FAILURE_TAG: &str = "$underlying_viba_op_err";
const NOT_IMPLEMENTED_TAG: &str = "$not_implemented_err";
const PROGRAM_ERR_TAG: &str = "$viba_program_err";
const ENVIRONMENT_API_TAG: &str = "$environment_api_invalid_argument_err";
const REASON_NO_IMPLEMENTATION: &str = "no implementation";
const REASON_GET_FUNC_RAISED: &str = "get_func raised";
const REASON_RAISED: &str = "raised";
const REASON_NO_LEAF: &str = "no leaf";

const TMP_PREFIX: &str = "tmp_";
const RET_NAME: &str = "__impl__";
const DEF_NAME: &str = "__decl__";
const GET_ARGS_NAME: &str = "__get_args__";
const IS_SUB_TYPE_VALUE_NAME: &str = "__is_sub_type_value__";
const ENVIRON_TAG: &str = "$env";
const ENVIRON_TYPE: &str = "Environment";
const ENV_TYPE: &str = "Env";
const DYN_CALL_NAME: &str = "__dyn_call__";
const DYN_METHOD_NAME: &str = "__dyn_method__";
const CODE_LABEL: &str = "<viba_code>";
const SNAPSHOT_NAME: &str = "value";
const SNAPSHOT_SUFFIX: &str = ".viba";
const RESULT_MODULE_NAME: &str = "interpret_result";
const RESULT_TYPE_NAME: &str = "InterpretResult";
const TAGGED_NAME: &str = "tagged";
const GET_ATTR_TAG: &str = "$get_attr";
const GET_ITEM_TAG: &str = "$get_item";
const IN_TAG: &str = "$in";
const LEN_TAG: &str = "$len";
const KEYS_TAG: &str = "$keys";
const FIRST_CASE_INDEX_TAG: &str = "$first_case_index";

const AST_MODULE: &str = "rs_viba.viba_ast";
const TYPE_MODULE: &str = "rs_viba.type";
const PARTIAL_MODULE: &str = "rs_viba.partial";
const PATTERN_MODULE: &str = "rs_viba.pattern";
const REFLECT_MODULE: &str = "rs_viba.reflect";
const DESCRIPTOR_MODULE: &str = "rs_viba.viba_type_descriptor";
const SERIALIZE_MODULE: &str = "rs_viba.serialize";
const IS_SUB_TYPE_MODULE: &str = "rs_viba.is_sub_type";
const SHIM_MODULE: &str = "rs_viba.interpret";

// ----------------------------------------------------------------------
// Talking to the other layers
// ----------------------------------------------------------------------

fn module<'py>(py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyModule>> {
    PyModule::import(py, name)
}

fn ast_mod<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyModule>> {
    module(py, AST_MODULE)
}

fn type_mod<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyModule>> {
    module(py, TYPE_MODULE)
}

fn partial_mod<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyModule>> {
    module(py, PARTIAL_MODULE)
}

fn pattern_mod<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyModule>> {
    module(py, PATTERN_MODULE)
}

fn reflect_mod<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyModule>> {
    module(py, REFLECT_MODULE)
}

fn descriptor_mod<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyModule>> {
    module(py, DESCRIPTOR_MODULE)
}

fn serialize_mod<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyModule>> {
    module(py, SERIALIZE_MODULE)
}

fn is_sub_type_mod<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyModule>> {
    module(py, IS_SUB_TYPE_MODULE)
}

fn shim<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyModule>> {
    module(py, SHIM_MODULE)
}

fn ast_class<'py>(py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    ast_mod(py)?.getattr(name)
}

fn type_class<'py>(py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    type_mod(py)?.getattr(name)
}

fn shim_class<'py>(py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    shim(py)?.getattr(name)
}

fn is_a(py: Python<'_>, obj: &Bound<'_, PyAny>, module_name: &str, name: &str) -> PyResult<bool> {
    obj.is_instance(&module(py, module_name)?.getattr(name)?)
}

fn is_ast(py: Python<'_>, obj: &Bound<'_, PyAny>, name: &str) -> PyResult<bool> {
    is_a(py, obj, AST_MODULE, name)
}

fn is_shim(py: Python<'_>, obj: &Bound<'_, PyAny>, name: &str) -> PyResult<bool> {
    is_a(py, obj, SHIM_MODULE, name)
}

fn is_type(py: Python<'_>, obj: &Bound<'_, PyAny>, name: &str) -> PyResult<bool> {
    is_a(py, obj, TYPE_MODULE, name)
}

fn is_ast_any(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    is_ast(py, obj, "AST")
}

/// `getattr`, spelled short: the runtime walks Python objects, as the original
/// walked the same attributes.
fn g<'py>(obj: &Bound<'py, PyAny>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    obj.getattr(name)
}

fn gs(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<String> {
    obj.getattr(name)?.extract()
}

fn has(obj: &Bound<'_, PyAny>, name: &str) -> bool {
    obj.hasattr(name).unwrap_or(false)
}

fn none(py: Python<'_>) -> Bound<'_, PyAny> {
    py.None().into_bound(py)
}

fn py_string(py: Python<'_>, text: &str) -> P {
    PyString::new(py, text).into_any().unbind()
}

fn pylist(py: Python<'_>, items: Vec<P>) -> PyResult<P> {
    Ok(PyList::new(py, items)?.into_any().unbind())
}

fn pytuple(py: Python<'_>, items: Vec<P>) -> PyResult<P> {
    Ok(PyTuple::new(py, items)?.into_any().unbind())
}

fn class_name(py: Python<'_>, obj: &Bound<'_, PyAny>) -> String {
    obj.get_type()
        .name()
        .map(|name| name.to_string())
        .unwrap_or_else(|_| String::from("?"))
}

fn pyrepr(py: Python<'_>, obj: &Bound<'_, PyAny>) -> String {
    obj.repr()
        .map(|text| text.to_string())
        .unwrap_or_else(|_| String::from("<unrepr>"))
}

fn format_exc(py: Python<'_>, error: &PyErr) -> String {
    let value = error.value(py);
    pyrepr(py, value.as_any())
}

fn is_py_exception(py: Python<'_>, error: &PyErr) -> bool {
    error
        .value(py)
        .as_any()
        .is_instance(&py.get_type::<pyo3::exceptions::PyException>())
        .unwrap_or(false)
}

fn is_raised(py: Python<'_>, error: &PyErr) -> PyResult<Option<P>> {
    let raised = shim_class(py, "_Raised")?;
    let value = error.value(py);
    if value.as_any().is_instance(&raised)? {
        return Ok(Some(value.getattr("result")?.unbind()));
    }
    Ok(None)
}

fn is_compact<'py>(py: Python<'py>, error: &PyErr) -> PyResult<Option<Bound<'py, PyAny>>> {
    let compact = shim_class(py, "_Compact")?;
    let value = error.value(py);
    if value.as_any().is_instance(&compact)? {
        return Ok(Some(value.clone().into_any()));
    }
    Ok(None)
}

fn runtime_error(message: &str) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(message.to_string())
}

fn type_error(message: &str) -> PyErr {
    pyo3::exceptions::PyTypeError::new_err(message.to_string())
}

fn make_ok(py: Python<'_>, value: P) -> PyResult<P> {
    Ok(type_class(py, "Ok")?.call1((value,))?.unbind())
}

fn make_ok_ref(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<P> {
    make_ok(py, value.clone().unbind())
}

fn make_ok_none(py: Python<'_>) -> PyResult<P> {
    make_ok(py, py.None())
}

fn is_ok(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<bool> {
    value.is_instance(&type_class(py, "Ok")?)
}

fn ok_value<'py>(value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    value.getattr("ok_value")
}

fn program_err(py: Python<'_>, message: &str) -> PyResult<P> {
    Ok(type_class(py, "VibaProgramErr")?
        .call1((message,))?
        .unbind())
}

fn under_op_err(
    py: Python<'_>,
    message: &str,
    module_path: &str,
    full_name: &str,
    call: Option<&Bound<'_, PyAny>>,
    tag: Option<&str>,
) -> PyResult<P> {
    let call = match call {
        Some(call) => call.clone().unbind(),
        None => py.None(),
    };
    let made = type_class(py, "UnderlyingOpErr")?.call1((
        PyString::new(py, message),
        PyString::new(py, module_path),
        PyString::new(py, full_name),
        call,
    ))?;
    if let Some(tag) = tag {
        made.setattr("tag", tag)?;
    }
    Ok(made.unbind())
}

fn api_err(py: Python<'_>, message: &str, api_name: &str, arguments: Option<P>) -> PyResult<P> {
    let arguments = match arguments {
        Some(arguments) => arguments,
        None => py.None(),
    };
    Ok(type_class(py, "EnvironmentApiInvalidArgumentErr")?
        .call1((
            PyString::new(py, message),
            PyString::new(py, api_name),
            arguments,
        ))?
        .unbind())
}

fn make_frame(py: Python<'_>, file_path: &str, lineno: i64) -> PyResult<P> {
    Ok(type_class(py, "Frame")?
        .call1((PyString::new(py, file_path), lineno))?
        .unbind())
}

fn make_viba_data(py: Python<'_>, node: P) -> PyResult<P> {
    Ok(shim_class(py, "_VibaData")?.call1((node,))?.unbind())
}

/// `VObject(access, descriptor, data)` with the two optional fields set after.
fn make_vobject(
    py: Python<'_>,
    descriptor: &Bound<'_, PyAny>,
    data: &Bound<'_, PyAny>,
    data_module: Option<&Bound<'_, PyAny>>,
    function: Option<&Bound<'_, PyAny>>,
) -> PyResult<P> {
    let access = reflect_mod(py)?.getattr("access")?;
    let made = reflect_mod(py)?
        .getattr("VObject")?
        .call1((access, descriptor, data))?;
    if let Some(module) = data_module {
        made.setattr("data_module", module)?;
    }
    if let Some(function) = function {
        made.setattr("function", function)?;
    }
    Ok(made.unbind())
}

fn make_ast_node_type(
    py: Python<'_>,
    node: &Bound<'_, PyAny>,
    source_module: &Bound<'_, PyAny>,
) -> PyResult<P> {
    Ok(type_class(py, "AstNodeType")?
        .call1((node, source_module))?
        .unbind())
}

fn descriptor_of(
    py: Python<'_>,
    node: &Bound<'_, PyAny>,
    source_module: &Bound<'_, PyAny>,
) -> PyResult<P> {
    let node_type = make_ast_node_type(py, node, source_module)?;
    Ok(descriptor_mod(py)?
        .getattr("descriptor_of")?
        .call1((node_type,))?
        .unbind())
}

fn descriptor_of_node_type(py: Python<'_>, node_type: &Bound<'_, PyAny>) -> PyResult<P> {
    Ok(descriptor_mod(py)?
        .getattr("descriptor_of")?
        .call1((node_type,))?
        .unbind())
}

fn descriptor_of_tagged(
    py: Python<'_>,
    tag: &str,
    inner: &Bound<'_, PyAny>,
    node: &Bound<'_, PyAny>,
    source_module: &Bound<'_, PyAny>,
) -> PyResult<P> {
    Ok(descriptor_mod(py)?
        .getattr("descriptor_of_tagged")?
        .call1((tag, inner, node, source_module))?
        .unbind())
}

fn descriptor_of_values(
    py: Python<'_>,
    node: &Bound<'_, PyAny>,
    source_module: &Bound<'_, PyAny>,
    members: Vec<P>,
) -> PyResult<P> {
    Ok(descriptor_mod(py)?
        .getattr("descriptor_of_values")?
        .call1((node, source_module, PyList::new(py, members)?))?
        .unbind())
}

fn descriptor_kind(py: Python<'_>, descriptor: &Bound<'_, PyAny>) -> PyResult<String> {
    g(descriptor, "kind")?.extract()
}

fn empty_pool(py: Python<'_>) -> PyResult<P> {
    Ok(descriptor_mod(py)?.getattr("empty_pool")?.call0()?.unbind())
}

fn viba_chain_descriptor(
    py: Python<'_>,
    pool: P,
    source_node_type: P,
    elements: Vec<P>,
) -> PyResult<P> {
    Ok(descriptor_mod(py)?
        .getattr("VibaChainDescriptor")?
        .call1((pool, source_node_type, PyList::new(py, elements)?))?
        .unbind())
}

fn viba_type_descriptor(py: Python<'_>, kind: &str, payload: P) -> PyResult<P> {
    Ok(descriptor_mod(py)?
        .getattr("VibaTypeDescriptor")?
        .call1((kind, payload))?
        .unbind())
}

fn language_config(py: Python<'_>) -> PyResult<P> {
    Ok(reflect_mod(py)?.getattr("language_config")?.unbind())
}

fn reflect_access<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
    reflect_mod(py)?.getattr("access")
}

fn by_tag(py: Python<'_>, tag: &str) -> PyResult<P> {
    Ok(reflect_mod(py)?.getattr("by_tag")?.call1((tag,))?.unbind())
}

fn reflect_get<'py>(
    py: Python<'py>,
    node: &Bound<'py, PyAny>,
    tag: &str,
) -> PyResult<Bound<'py, PyAny>> {
    reflect_access(py)?.call_method1("get", (node, by_tag(py, tag)?))
}

/// `descriptor_of(AstNodeType(...))` on one AST piece of a module.
fn piece_descriptor(
    py: Python<'_>,
    node: &Bound<'_, PyAny>,
    source_module: &Bound<'_, PyAny>,
) -> PyResult<P> {
    descriptor_of(py, node, source_module)
}

// ----------------------------------------------------------------------
// Small spellings of the source's own helpers
// ----------------------------------------------------------------------

fn one_line(py: Python<'_>, node: &Bound<'_, PyAny>) -> PyResult<String> {
    let text: String = ast_mod(py)?
        .getattr("unparse_type")?
        .call1((node,))?
        .extract()?;
    Ok(text.split_whitespace().collect::<Vec<_>>().join(" "))
}

fn unparse_type(py: Python<'_>, node: &Bound<'_, PyAny>) -> PyResult<String> {
    ast_mod(py)?
        .getattr("unparse_type")?
        .call1((node,))?
        .extract()
}

fn make_ast(py: Python<'_>, name: &str, fields: Vec<(&str, P)>) -> PyResult<P> {
    let made = ast_class(py, name)?.call0()?;
    for (field, value) in fields {
        made.setattr(field, value)?;
    }
    Ok(made.unbind())
}

fn make_ast0(py: Python<'_>, name: &str) -> PyResult<P> {
    Ok(ast_class(py, name)?.call0()?.unbind())
}

fn make_ast1(py: Python<'_>, name: &str, field: &str, value: P) -> PyResult<P> {
    make_ast(py, name, vec![(field, value)])
}

fn make_tagged(py: Python<'_>, tag: &str, inner: P) -> PyResult<P> {
    Ok(ast_class(py, "Tagged")?
        .call1((PyString::new(py, tag), inner))?
        .unbind())
}

fn make_partial(py: Python<'_>, function: P, argument: P) -> PyResult<P> {
    Ok(ast_class(py, "Partial")?
        .call1((function, argument))?
        .unbind())
}

fn make_type_ref(py: Python<'_>, name: &str) -> PyResult<P> {
    Ok(ast_class(py, "TypeRef")?
        .call1((PyString::new(py, name),))?
        .unbind())
}

fn source_path(py: Python<'_>, node: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
    let found = ast_mod(py)?.getattr("source_path")?.call1((node,))?;
    if found.is_none() {
        return Ok(None);
    }
    Ok(Some(found.extract()?))
}

fn scope_frames(py: Python<'_>, scope: &Bound<'_, PyAny>) -> PyResult<Vec<P>> {
    let mut frames = Vec::new();
    for frame in scope.try_iter()? {
        frames.push(frame?.unbind());
    }
    Ok(frames)
}

fn scope_object(py: Python<'_>, frames: Vec<P>) -> PyResult<P> {
    pytuple(py, frames)
}

fn in_scope(py: Python<'_>, frames: &[P], name: &str) -> PyResult<Option<P>> {
    for frame in frames.iter().rev() {
        let frame = frame.bind(py);
        if frame
            .call_method1("__contains__", (name,))?
            .extract::<bool>()?
        {
            return Ok(Some(frame.get_item(name)?.unbind()));
        }
    }
    Ok(None)
}

fn is_never(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<bool> {
    if !is_shim(py, value, "_VibaData")? {
        return Ok(false);
    }
    let node = g(value, "node")?;
    is_ast(py, &g(&node, "data")?, "Never")
}

fn is_nil(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<bool> {
    if !is_shim(py, value, "_VibaData")? {
        return Ok(false);
    }
    let node = g(value, "node")?;
    is_ast(py, &g(&node, "data")?, "Nil")
}

fn is_environ_value(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<bool> {
    if !is_shim(py, value, "_Host")? {
        return Ok(false);
    }
    let obj = g(value, "obj")?;
    is_shim(py, &obj, "Environment")
}

fn is_stopped(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(!is_ok(py, value)?)
}

// ----------------------------------------------------------------------
// The host side: storage, chains, modules of the builtin directory
// ----------------------------------------------------------------------

fn default_store_root(py: Python<'_>) -> PyResult<String> {
    let tempdir: String = module(py, "tempfile")?
        .call_method0("gettempdir")?
        .extract()?;
    Ok(format!("{tempdir}/viba-store"))
}

fn store_path_of(
    py: Python<'_>,
    storage: &Bound<'_, PyAny>,
    file_path: &str,
    root_dir: Option<&str>,
) -> PyResult<P> {
    let root = match root_dir {
        Some(root) => root.to_string(),
        None => gs(storage, "store_root_dir")?,
    };
    let path_module = module(py, "pathlib")?;
    let parts: Vec<P> = file_path
        .split('/')
        .filter(|part| !part.is_empty())
        .map(|part| py_string(py, part))
        .collect();
    let relative = path_module
        .getattr("Path")?
        .call1(PyTuple::new(py, parts)?)?;
    let root = path_module.getattr("Path")?.call1((root,))?;
    Ok(root.call_method1("__truediv__", (relative,))?.unbind())
}

fn storage_sub(py: Python<'_>, storage: &Bound<'_, PyAny>, name: &str) -> PyResult<P> {
    let sub_storage = g(storage, "sub_storage")?;
    if sub_storage
        .call_method1("__contains__", (name,))?
        .extract::<bool>()?
    {
        return Ok(sub_storage.get_item(name)?.unbind());
    }
    let prefix: String = gs(storage, "cur_storage_path")?;
    let path = if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{prefix}/{name}")
    };
    let store_root = g(storage, "store_root_dir")?;
    let cls = shim_class(py, "EnvironmentStorage")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("parent", storage)?;
    let made = cls.call((path, py.None(), store_root), Some(&kwargs))?;
    sub_storage.set_item(name, &made)?;
    Ok(made.unbind())
}

fn storage_tmp(py: Python<'_>, storage: &Bound<'_, PyAny>) -> PyResult<P> {
    loop {
        let uuid = module(py, "uuid")?;
        let hex: String = uuid.getattr("uuid4")?.call0()?.getattr("hex")?.extract()?;
        let name = format!("{TMP_PREFIX}{}", &hex[..12]);
        let sub_storage = g(storage, "sub_storage")?;
        if !sub_storage
            .call_method1("__contains__", (&name,))?
            .extract::<bool>()?
        {
            return storage_sub(py, storage, &name);
        }
    }
}

fn storage_get_text(
    py: Python<'_>,
    storage: &Bound<'_, PyAny>,
    file_path: &str,
) -> PyResult<Option<String>> {
    let path = store_path_of(py, storage, file_path, None)?;
    let path = path.bind(py);
    match path.call_method0("read_text") {
        Ok(text) => Ok(Some(text.extract()?)),
        Err(error) => {
            if error.is_instance_of::<pyo3::exceptions::PyFileNotFoundError>(py) {
                Ok(None)
            } else {
                Err(error)
            }
        }
    }
}

fn storage_put_text(
    py: Python<'_>,
    storage: &Bound<'_, PyAny>,
    file_path: &str,
    content: &str,
) -> PyResult<()> {
    let path = store_path_of(py, storage, file_path, None)?;
    let path = path.bind(py);
    let mkdir_kwargs = PyDict::new(py);
    mkdir_kwargs.set_item("parents", true)?;
    mkdir_kwargs.set_item("exist_ok", true)?;
    path.getattr("parent")?
        .call_method("mkdir", (), Some(&mkdir_kwargs))?;
    let parent = path.getattr("parent")?;
    let name: String = path.getattr("name")?.extract()?;
    let mkstemp = module(py, "tempfile")?.getattr("mkstemp")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("dir", parent)?;
    kwargs.set_item("prefix", format!("{name}."))?;
    kwargs.set_item("suffix", ".partial")?;
    let made = mkstemp.call((), Some(&kwargs))?;
    let (handle, staging) = made.extract::<(P, P)>()?;
    let os = module(py, "os")?;
    let write = (|| -> PyResult<()> {
        let stream = os.getattr("fdopen")?.call1((handle, py_string(py, "w")))?;
        stream.call_method1("write", (content,))?;
        stream.call_method0("close")?;
        Ok(())
    })();
    if let Err(error) = write {
        let _ = os.call_method1("unlink", (staging,));
        return Err(error);
    }
    if let Err(error) = os.call_method1("replace", (staging.clone_ref(py), path)) {
        let _ = os.call_method1("unlink", (staging,));
        return Err(error);
    }
    Ok(())
}

fn an_environment(py: Python<'_>, value: &Bound<'_, PyAny>, name: &str) -> PyResult<P> {
    if !is_shim(py, value, "Environment")? {
        return Err(runtime_error(&format!(
            "{name} takes an Environment, not {}",
            class_name(py, value)
        )));
    }
    Ok(value.clone().unbind())
}

fn an_environment_or_nil(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    name: &str,
) -> PyResult<Option<P>> {
    if is_a(py, value, REFLECT_MODULE, "VObject")? {
        if is_ast(py, &g(value, "data")?, "Nil")? {
            return Ok(None);
        }
        return Err(runtime_error(&format!(
            "{name} takes an Environment or nil, not a value in the source"
        )));
    }
    if value.is_none() {
        return Ok(None);
    }
    Ok(Some(an_environment(py, value, name)?))
}

fn storage_path(py: Python<'_>, environ: &Bound<'_, PyAny>) -> PyResult<String> {
    let storage = g(environ, "storage")?;
    if storage.is_none() {
        return Ok(String::new());
    }
    Ok(gs(&storage, "cur_storage_path")?)
}

fn the_storage(py: Python<'_>, environ: &Bound<'_, PyAny>) -> PyResult<P> {
    let storage = g(environ, "storage")?;
    if storage.is_none() {
        return Err(runtime_error(
            "this environment has no storage to keep a snapshot in",
        ));
    }
    Ok(storage.unbind())
}

fn get_root(py: Python<'_>, current: &Bound<'_, PyAny>) -> PyResult<Option<P>> {
    let mut current = match an_environment_or_nil(py, current, "get_root")? {
        Some(current) => Some(current),
        None => return Ok(None),
    };
    loop {
        let bound = current.as_ref().unwrap().bind(py);
        let parent = g(bound, "parent")?;
        if parent.is_none() {
            return Ok(current);
        }
        current = Some(parent.unbind());
    }
}

fn get_parent(py: Python<'_>, environ: &Bound<'_, PyAny>) -> PyResult<P> {
    let environ = an_environment(py, environ, "get_parent")?;
    Ok(g(environ.bind(py), "parent")?.unbind())
}

fn env_chain_length(py: Python<'_>, environ: &Bound<'_, PyAny>) -> PyResult<i64> {
    let mut current = Some(an_environment(py, environ, "env_chain_length")?);
    let mut length = 0;
    while let Some(one) = current {
        length += 1;
        let parent = g(one.bind(py), "parent")?;
        current = if parent.is_none() {
            None
        } else {
            Some(parent.unbind())
        };
    }
    Ok(length)
}

fn get_relative_path(
    py: Python<'_>,
    current: &Bound<'_, PyAny>,
    root: &Bound<'_, PyAny>,
) -> PyResult<String> {
    let current = an_environment(py, current, "get_relative_path")?;
    let root = match an_environment_or_nil(py, root, "get_relative_path")? {
        Some(root) => root,
        None => match get_root(py, current.bind(py))? {
            Some(root) => root,
            None => return Ok(String::new()),
        },
    };
    let here = storage_path(py, current.bind(py))?;
    let base = storage_path(py, root.bind(py))?;
    if here == base {
        return Ok(String::new());
    }
    let prefix = if base.is_empty() {
        String::new()
    } else {
        format!("{base}/")
    };
    if !here.starts_with(&prefix) {
        let here = if here.is_empty() { "<no path>" } else { &here };
        let base = if base.is_empty() { "<no path>" } else { &base };
        return Err(runtime_error(&format!("{here} is no child of {base}")));
    }
    Ok(here[prefix.len()..].to_string())
}

fn the_relative_path(py: Python<'_>, relative_path: &Bound<'_, PyAny>) -> PyResult<String> {
    let mut text = relative_path;
    let value;
    if is_a(py, relative_path, REFLECT_MODULE, "VObject")? {
        value = g(relative_path, "value")?;
        text = &value;
    }
    if text.is_instance_of::<PyString>() {
        return Ok(text.extract()?);
    }
    Err(runtime_error(&format!(
        "find_by_relative_path takes a relative path, not {}: give it as \
         args.env.find_by_relative_path << \"a/b\"",
        class_name(py, text)
    )))
}

fn find_by_relative_path(
    py: Python<'_>,
    relative_path: &Bound<'_, PyAny>,
    root: &Bound<'_, PyAny>,
) -> PyResult<P> {
    let text = the_relative_path(py, relative_path)?;
    let base = an_environment(py, root, "find_by_relative_path")?;
    let mut storage = the_storage(py, base.bind(py))?;
    for part in text.split('/').filter(|one| !one.is_empty()) {
        if part == "." || part == ".." {
            return Err(runtime_error(&format!(
                "'{text}' is no relative path: it has a '{part}' segment"
            )));
        }
        storage = storage.bind(py).call_method1("sub", (part,))?.unbind();
    }
    if storage
        .bind(py)
        .is(&the_storage(py, base.bind(py))?.bind(py).clone())
    {
        return Ok(base);
    }
    let compute = g(base.bind(py), "compute")?;
    let viba_path = g(base.bind(py), "viba_path")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("parent", base.bind(py))?;
    Ok(shim_class(py, "Environment")?
        .call((storage, compute, viba_path), Some(&kwargs))?
        .unbind())
}

fn find_by_relative_path_member(
    py: Python<'_>,
    environ: &Bound<'_, PyAny>,
    relative_path: &Bound<'_, PyAny>,
    root: &Bound<'_, PyAny>,
) -> PyResult<P> {
    let text = the_relative_path(py, relative_path)?;
    let root = match an_environment_or_nil(py, root, "find_by_relative_path")? {
        Some(root) => root,
        None => environ.clone().unbind(),
    };
    find_by_relative_path(py, &py_string(py, &text).into_bound(py), root.bind(py))
}

fn a_name_piece(py: Python<'_>, name: &Bound<'_, PyAny>) -> PyResult<String> {
    let mut name = name.clone();
    if is_a(py, &name, REFLECT_MODULE, "VObject")? {
        name = g(&name, "data")?;
    }
    if name.is_instance_of::<PyString>() {
        return Ok(name.extract()?);
    }
    if is_ast(py, &name, "Constant")? {
        let value = g(&name, "value")?;
        if value.is_instance_of::<PyString>() {
            return Ok(value.extract()?);
        }
        return Ok(String::new());
    }
    if is_ast(py, &name, "TypeRef")? {
        return Ok(gs(&name, "name")?);
    }
    if is_ast(py, &name, "Member")? {
        return Ok(a_tag_text(&gs(&name, "tag")?));
    }
    if is_ast(py, &name, "MemberTaken")? {
        return Ok(source_path(py, &name)?.unwrap_or_default());
    }
    if is_ast(py, &name, "Partial")? {
        let mut head = g(&name, "function")?;
        while is_ast(py, &head, "Partial")? {
            head = g(&head, "function")?;
        }
        return a_name_piece(py, &head);
    }
    if is_ast(py, &name, "Product")? || is_ast(py, &name, "ProductChain")? {
        let mut pieces = Vec::new();
        for piece in product_elements(py, &name)? {
            let text = a_name_piece(py, &piece)?;
            if !text.is_empty() {
                pieces.push(text);
            }
        }
        return Ok(pieces.join("/"));
    }
    Ok(String::new())
}

fn a_tag_text(tag: &str) -> String {
    tag.strip_prefix('$').unwrap_or(tag).to_string()
}

fn the_storage_name(py: Python<'_>, name: &Bound<'_, PyAny>) -> PyResult<String> {
    let mut name = name.clone();
    if is_a(py, &name, REFLECT_MODULE, "VObject")? {
        name = g(&name, "data")?;
    }
    if !is_ast_any(py, &name)? {
        return Ok(name.str()?.to_string());
    }
    if is_ast(py, &name, "Constant")? {
        let value = g(&name, "value")?;
        if value.is_instance_of::<PyString>() {
            return Ok(value.extract()?);
        }
        return unparse_type(py, &name);
    }
    let text = a_name_piece(py, &name)?;
    if text.is_empty() {
        return Err(runtime_error(&format!(
            "a storage name is a string or pieces of the source that name \
             something, and this names nothing: {}",
            one_line(py, &name)?
        )));
    }
    Ok(text)
}

fn child_environment(py: Python<'_>, environ: &Bound<'_, PyAny>, storage: P) -> PyResult<P> {
    let compute = g(environ, "compute")?;
    let viba_path = g(environ, "viba_path")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("parent", environ)?;
    if is_shim(py, environ, "ReferencedEnvironment")? {
        kwargs.set_item(
            "reference_store_root_dir",
            g(environ, "reference_store_root_dir")?,
        )?;
        return Ok(shim_class(py, "ReferencedEnvironment")?
            .call((storage, compute, viba_path), Some(&kwargs))?
            .unbind());
    }
    Ok(shim_class(py, "Environment")?
        .call((storage, compute, viba_path), Some(&kwargs))?
        .unbind())
}

fn sub_env(py: Python<'_>, environ: &Bound<'_, PyAny>, name: &Bound<'_, PyAny>) -> PyResult<P> {
    let environ = an_environment(py, environ, "sub_env")?;
    let storage_name = the_storage_name(py, name)?;
    let storage = the_storage(py, environ.bind(py))?
        .bind(py)
        .call_method1("sub", (storage_name,))?
        .unbind();
    child_environment(py, environ.bind(py), storage)
}

fn tmp_env(py: Python<'_>, environ: &Bound<'_, PyAny>) -> PyResult<P> {
    let storage = the_storage(py, environ)?
        .bind(py)
        .call_method0("tmp")?
        .unbind();
    child_environment(py, environ, storage)
}

fn make_referenced_env(
    py: Python<'_>,
    environ: &Bound<'_, PyAny>,
    reference_store_root_dir: &Bound<'_, PyAny>,
) -> PyResult<P> {
    let env = an_environment(py, environ, "make_referenced_env")?;
    let root = text_of(py, reference_store_root_dir, "make_referenced_env")?;
    let env = env.bind(py);
    let kwargs = PyDict::new(py);
    kwargs.set_item("parent", g(env, "parent")?)?;
    kwargs.set_item("reference_store_root_dir", root)?;
    Ok(shim_class(py, "ReferencedEnvironment")?
        .call(
            (g(env, "storage")?, g(env, "compute")?, g(env, "viba_path")?),
            Some(&kwargs),
        )?
        .unbind())
}

fn compress_env_path(
    py: Python<'_>,
    sup: &Bound<'_, PyAny>,
    sub: &Bound<'_, PyAny>,
) -> PyResult<P> {
    let sup = an_environment(py, sup, "compress_env_path")?;
    let sub = an_environment(py, sub, "compress_env_path")?;
    let above = storage_path(py, sup.bind(py))?;
    let under = storage_path(py, sub.bind(py))?;
    if !under.starts_with(&format!("{above}_")) && !under.starts_with(&format!("{above}/")) {
        let above = if above.is_empty() {
            "<no path>"
        } else {
            &above
        };
        let under = if under.is_empty() {
            "<no path>"
        } else {
            &under
        };
        return Err(runtime_error(&format!(
            "{under} does not begin with {above}: compress_env_path compresses a \
             path under the sup it is taken from"
        )));
    }
    let storage = the_storage(py, sup.bind(py))?;
    let beside = g(storage.bind(py), "parent")?;
    if beside.is_none() {
        let above = if above.is_empty() {
            "<no path>"
        } else {
            &above
        };
        return Err(runtime_error(&format!(
            "{above} is the root: there is no directory beside it to put the path in"
        )));
    }
    let digest: String = module(py, "hashlib")?
        .call_method1("sha1", (under.as_bytes(),))?
        .call_method0("hexdigest")?
        .extract()?;
    let last = above.rsplit('/').next().unwrap_or("");
    let name = format!("{last}_{digest}");
    let new_storage = beside.call_method1("sub", (name,))?.unbind();
    let kwargs = PyDict::new(py);
    kwargs.set_item("parent", sub.bind(py))?;
    let made = shim_class(py, "Environment")?
        .call(
            (
                new_storage,
                g(sup.bind(py), "compute")?,
                g(sup.bind(py), "viba_path")?,
            ),
            Some(&kwargs),
        )?
        .unbind();
    made.bind(py).setattr("uncompress_relative_path", under)?;
    Ok(made)
}

fn env_chain_limit(py: Python<'_>, limit: &Bound<'_, PyAny>) -> PyResult<i64> {
    let mut source = limit.clone();
    if is_a(py, limit, REFLECT_MODULE, "VObject")? {
        source = g(limit, "data")?;
    }
    if is_ast(py, &source, "Constant")? {
        let value = g(&source, "value")?;
        if value.is_instance_of::<pyo3::types::PyInt>()
            && !value.is_instance_of::<pyo3::types::PyBool>()
        {
            return value.extract();
        }
    }
    if source.is_instance_of::<pyo3::types::PyInt>()
        && !source.is_instance_of::<pyo3::types::PyBool>()
    {
        return source.extract();
    }
    let said = if is_ast_any(py, &source)? {
        one_line(py, &source)?
    } else {
        pyrepr(py, &source)
    };
    Err(runtime_error(&format!(
        "try_compact takes the chain length limit as an int, not {said}"
    )))
}

fn current_run(py: Python<'_>) -> PyResult<Option<P>> {
    let run = shim(py)?.getattr("_CURRENT_RUN")?.call_method0("get")?;
    if run.is_none() {
        return Ok(None);
    }
    Ok(Some(run.unbind()))
}

fn is_running_there(
    py: Python<'_>,
    runner: &Bound<'_, PyAny>,
    environ: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    let calls = g(runner, "running_calls")?;
    for record in calls.try_iter()? {
        let record = record?;
        if g(&record, "environ")?.is(environ) {
            return Ok(true);
        }
    }
    Ok(false)
}

fn resumable_stack(
    py: Python<'_>,
    runner: &Bound<'_, PyAny>,
    environ: &Bound<'_, PyAny>,
) -> PyResult<P> {
    let calls = g(runner, "running_calls")?;
    let mut at_env: Vec<(usize, Bound<'_, PyAny>)> = Vec::new();
    for record in calls.try_iter()? {
        let record = record?;
        let key = g(&record, "environ")?.as_ptr() as usize;
        at_env.push((key, record));
    }
    let mut on_the_chain: Vec<P> = Vec::new();
    let mut current = Some(environ.clone().unbind());
    while let Some(one) = current {
        let key = one.bind(py).as_ptr() as usize;
        if let Some(index) = at_env.iter().position(|(other, _)| *other == key) {
            let (_, record) = at_env.remove(index);
            on_the_chain.push(record.unbind());
        }
        let parent = g(one.bind(py), "parent")?;
        current = if parent.is_none() {
            None
        } else {
            Some(parent.unbind())
        };
    }
    on_the_chain.reverse();
    let mut elsewhere: Vec<P> = at_env
        .into_iter()
        .map(|(_, record)| record.unbind())
        .collect();
    elsewhere.extend(on_the_chain);
    pylist(py, elsewhere)
}

fn try_compact(
    py: Python<'_>,
    environ: &Bound<'_, PyAny>,
    env_chain_length_limit: &Bound<'_, PyAny>,
) -> PyResult<P> {
    let current = an_environment(py, environ, "try_compact")?;
    let limit = env_chain_limit(py, env_chain_length_limit)?;
    let runner = current_run(py)?;
    let floor: i64 = match &runner {
        Some(runner) => g(runner.bind(py), "compacted_chain_length")?
            .extract()
            .unwrap_or(0),
        None => 0,
    };
    if env_chain_length(py, current.bind(py))? <= limit.max(floor) {
        return Ok(current);
    }
    let runner = match runner {
        Some(runner) => runner,
        None => {
            return Err(runtime_error(
                "try_compact compacts the run it is called in, and no run is here",
            ))
        }
    };
    if !is_running_there(py, runner.bind(py), current.bind(py))? {
        return Ok(current);
    }
    let tasks = resumable_stack(py, runner.bind(py), current.bind(py))?;
    let compact = shim_class(py, "_Compact")?
        .call1((tasks, current))?
        .unbind();
    Err(PyErr::from_value(compact.bind(py).clone()))
}

// ----------------------------------------------------------------------
// Snapshots
// ----------------------------------------------------------------------

fn storage_cur_path(py: Python<'_>, storage: &Bound<'_, PyAny>) -> String {
    storage
        .getattr("cur_storage_path")
        .ok()
        .and_then(|path| path.extract().ok())
        .unwrap_or_default()
}

fn snapshot_path(py: Python<'_>, environ: &Bound<'_, PyAny>, name: &str) -> PyResult<String> {
    let storage = the_storage(py, environ)?;
    let path = storage_cur_path(py, storage.bind(py));
    let tail = format!("{name}{SNAPSHOT_SUFFIX}");
    Ok([path.as_str(), tail.as_str()]
        .iter()
        .filter(|part| !part.is_empty())
        .cloned()
        .collect::<Vec<_>>()
        .join("/"))
}

fn snapshot_of(py: Python<'_>, text: &str) -> PyResult<P> {
    let module = type_mod(py)?
        .getattr("custom_module")?
        .call1((text,))?
        .unbind();
    let stored = match definition(py, module.bind(py), SNAPSHOT_NAME)? {
        Some(stored) => stored,
        None => {
            return Err(runtime_error(&format!(
                "the snapshot has no {SNAPSHOT_NAME}: '{text}'"
            )))
        }
    };
    let node = g(stored.bind(py), "body")?;
    let descriptor = piece_descriptor(py, &node, module.bind(py))?;
    make_vobject(py, descriptor.bind(py), &node, None, None)
}

fn get_snapshot(py: Python<'_>, environ: &Bound<'_, PyAny>, name: &str) -> PyResult<Option<P>> {
    let storage = the_storage(py, environ)?;
    let path = snapshot_path(py, environ, name)?;
    let text = storage
        .bind(py)
        .call_method1("get_text", (path,))?
        .extract::<Option<String>>()?;
    match text {
        Some(text) => Ok(Some(snapshot_of(py, &text)?)),
        None => Ok(None),
    }
}

fn get_snapshot_from(
    py: Python<'_>,
    environ: &Bound<'_, PyAny>,
    name: &str,
    root: &str,
) -> PyResult<Option<P>> {
    let storage = the_storage(py, environ)?;
    let path = storage_cur_path(py, storage.bind(py));
    let kwargs = PyDict::new(py);
    kwargs.set_item("store_root_dir", root)?;
    let reader = shim_class(py, "EnvironmentStorage")?.call((path, py.None()), Some(&kwargs))?;
    let path = snapshot_path(py, environ, name)?;
    let text = reader
        .call_method1("get_text", (path,))?
        .extract::<Option<String>>()?;
    match text {
        Some(text) => Ok(Some(snapshot_of(py, &text)?)),
        None => Ok(None),
    }
}

fn put_snapshot(
    py: Python<'_>,
    environ: &Bound<'_, PyAny>,
    value: &Bound<'_, PyAny>,
    name: &str,
) -> PyResult<()> {
    let data = viba_data(py, value)?.bind(py).clone();
    let source = serialize_mod(py)?
        .getattr("serialize")?
        .call1((SNAPSHOT_NAME, data))?;
    if is_type(py, &source, "Err")? {
        return Err(runtime_error(&format!(
            "cannot snapshot this value: {}",
            gs(&source, "msg")?
        )));
    }
    let text: String = ok_value(&source)?.extract()?;
    let storage = the_storage(py, environ)?;
    let path = snapshot_path(py, environ, name)?;
    storage.bind(py).call_method1("put_text", (path, text))?;
    Ok(())
}

fn replayed(
    py: Python<'_>,
    environ: &Bound<'_, PyAny>,
    compute: &Bound<'_, PyAny>,
    name: &str,
) -> PyResult<P> {
    if let Some(stored) = get_snapshot(py, environ, name)? {
        return Ok(stored);
    }
    let value = compute.call0()?;
    put_snapshot(py, environ, &value, name)?;
    Ok(value.unbind())
}

// ----------------------------------------------------------------------
// Recording and replay (viba/exec/viba-reenact.md)
// ----------------------------------------------------------------------

fn text_of(py: Python<'_>, value: &Bound<'_, PyAny>, name: &str) -> PyResult<String> {
    let mut value = value.clone();
    if is_a(py, &value, REFLECT_MODULE, "VObject")? {
        value = g(&value, "data")?;
    }
    if is_ast(py, &value, "Constant")? {
        let inner = g(&value, "value")?;
        if inner.is_instance_of::<PyString>() {
            return Ok(inner.extract()?);
        }
    }
    if value.is_instance_of::<PyString>() {
        return Ok(value.extract()?);
    }
    Err(runtime_error(&format!(
        "{name} takes a str, not {}",
        class_name(py, &value)
    )))
}

fn join_store_path(base: &str, rest: &str) -> String {
    [base, rest]
        .iter()
        .filter(|part| !part.is_empty())
        .cloned()
        .collect::<Vec<_>>()
        .join("/")
}

fn stored_count(py: Python<'_>, storage: &Bound<'_, PyAny>, path: &str) -> PyResult<i64> {
    let text = storage
        .call_method1("get_text", (path,))?
        .extract::<Option<String>>()?;
    let Some(text) = text else { return Ok(0) };
    match text.trim().parse::<i64>() {
        Ok(count) => Ok(count),
        Err(_) => Err(runtime_error(&format!(
            "the reenact count at '{path}' is no integer: '{text}'"
        ))),
    }
}

fn answer_list(py: Python<'_>, values: Vec<P>) -> PyResult<P> {
    let mut elements = Vec::new();
    for value in values {
        let data = if is_a(py, value.bind(py), REFLECT_MODULE, "VObject")? {
            g(value.bind(py), "data")?
        } else {
            g(&viba_data(py, value.bind(py))?.bind(py).clone(), "data")?
        };
        elements.push(data.unbind());
    }
    let node = ast_class(py, "TypeApp")?
        .call1((py_string(py, "ListLiteral"), PyList::new(py, elements)?))?
        .unbind();
    viba_data(py, node.bind(py))
}

fn referenced(py: Python<'_>, environ: &Bound<'_, PyAny>, name: &str) -> PyResult<(P, String)> {
    let env = an_environment(py, environ, name)?;
    if !is_shim(py, env.bind(py), "ReferencedEnvironment")? {
        return Err(runtime_error(&format!(
            "{name} runs on a referenced environment: make one with make_referenced_env"
        )));
    }
    let root = g(env.bind(py), "reference_store_root_dir")?;
    if root.is_none() {
        return Err(runtime_error(
            "a referenced environment has no reference store",
        ));
    }
    Ok((env, root.extract()?))
}

fn found_data(py: Python<'_>, found: bool, value: Option<&Bound<'_, PyAny>>) -> PyResult<P> {
    let inner = match value {
        Some(value) => g(&viba_data(py, value)?.bind(py).clone(), "data")?,
        None => make_ast0(py, "Nil")?.bind(py).clone(),
    };
    let tagged_found = make_tagged(
        py,
        "$found",
        ast_class(py, "Constant")?.call1((found,))?.unbind(),
    )?;
    let tagged_value = make_tagged(py, "$value", inner.unbind())?;
    let node = ast_class(py, "ProductChain")?
        .call1((PyList::new(py, [tagged_found, tagged_value])?,))?
        .unbind();
    viba_data(py, node.bind(py))
}

fn entry_name(py: Python<'_>, f: &Bound<'_, PyAny>) -> PyResult<String> {
    let mut f = f.clone();
    if is_a(py, &f, REFLECT_MODULE, "VObject")? {
        f = g(&f, "data")?;
    }
    if is_ast(py, &f, "TypeRef")? {
        return Ok(gs(&f, "name")?);
    }
    if is_ast(py, &f, "Partial")? {
        let (head, _arguments) = call_parts(py, &f)?;
        return entry_name(py, head.bind(py));
    }
    Ok(f.str()?.to_string())
}

fn record_reenact_call(
    py: Python<'_>,
    environ: &Bound<'_, PyAny>,
    tag: &Bound<'_, PyAny>,
    f: &Bound<'_, PyAny>,
    args: &Bound<'_, PyAny>,
) -> PyResult<String> {
    let env = an_environment(py, environ, "record_reenact_call")?;
    let tag = text_of(py, tag, "record_reenact_call")?;
    let storage = the_storage(py, env.bind(py))?;
    let base = storage_cur_path(py, storage.bind(py));
    let count_path = join_store_path(&base, &format!("reenact/{tag}/count.viba"));
    let count = stored_count(py, storage.bind(py), &count_path)?;
    storage
        .bind(py)
        .call_method1("put_text", (&count_path, (count + 1).to_string()))?;
    let invocation = sub_env(
        py,
        env.bind(py),
        &py_string(py, &format!("reenact/{tag}/{count}")).into_bound(py),
    )?;
    let name = entry_name(py, f)?;
    put_snapshot(
        py,
        invocation.bind(py),
        &py_string(py, &name).into_bound(py),
        "entry",
    )?;
    put_snapshot(py, invocation.bind(py), args, "args")?;
    Ok(format!("reenact/{tag}/{count}"))
}

fn reenact_reference(
    py: Python<'_>,
    environ: &Bound<'_, PyAny>,
    tag: &Bound<'_, PyAny>,
) -> PyResult<P> {
    let env = an_environment(py, environ, "reenact_reference")?;
    let tag = text_of(py, tag, "reenact_reference")?;
    if is_shim(py, env.bind(py), "ReferencedEnvironment")? {
        let root = g(env.bind(py), "reference_store_root_dir")?;
        if root.is_none() {
            return Err(runtime_error(
                "a referenced environment has no reference store",
            ));
        }
        let root: String = root.extract()?;
        let stored = get_snapshot_from(py, env.bind(py), &tag, &root)?;
        let Some(stored) = stored else {
            return Err(runtime_error(&format!(
                "no reference '{tag}' at {}",
                snapshot_path(py, env.bind(py), &tag)?
            )));
        };
        return found_data(py, true, Some(stored.bind(py)));
    }
    found_data(py, false, None)
}

fn store_reenact_reference(
    py: Python<'_>,
    environ: &Bound<'_, PyAny>,
    tag: &Bound<'_, PyAny>,
    value: &Bound<'_, PyAny>,
) -> PyResult<P> {
    let env = an_environment(py, environ, "store_reenact_reference")?;
    let tag = text_of(py, tag, "store_reenact_reference")?;
    if is_shim(py, env.bind(py), "ReferencedEnvironment")? {
        return Ok(value.clone().unbind());
    }
    put_snapshot(py, env.bind(py), value, &tag)?;
    Ok(value.clone().unbind())
}

fn reenact_calls(
    py: Python<'_>,
    environ: &Bound<'_, PyAny>,
    tag: &Bound<'_, PyAny>,
    entry: &Bound<'_, PyAny>,
) -> PyResult<P> {
    let env = an_environment(py, environ, "reenact_calls")?;
    if !is_shim(py, env.bind(py), "ReferencedEnvironment")? {
        return Err(runtime_error(
            "reenact_calls runs on a referenced environment: make one with make_referenced_env",
        ));
    }
    let root = g(env.bind(py), "reference_store_root_dir")?;
    if root.is_none() {
        return Err(runtime_error(
            "a referenced environment has no reference store",
        ));
    }
    let root: String = root.extract()?;
    let tag = text_of(py, tag, "reenact_calls")?;
    let base_env = g(env.bind(py), "parent")?;
    if base_env.is_none() {
        return Err(runtime_error(
            "reenact_calls needs the layer above it as the recording's base",
        ));
    }
    let base = storage_path(py, &base_env)?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("store_root_dir", &root)?;
    let reader_storage = shim_class(py, "EnvironmentStorage")?
        .call((py_string(py, &base), py.None()), Some(&kwargs))?;
    let count_path = join_store_path(&base, &format!("reenact/{tag}/count.viba"));
    let count = stored_count(py, &reader_storage, &count_path)?;
    let mut answers: Vec<P> = Vec::new();
    for index in 0..count {
        let invocation = sub_env(
            py,
            &base_env,
            &py_string(py, &format!("reenact/{tag}/{index}")).into_bound(py),
        )?;
        let path = join_store_path(&base, &format!("reenact/{tag}/{index}/args.viba"));
        let text = reader_storage
            .call_method1("get_text", (path.clone(),))?
            .extract::<Option<String>>()?;
        let Some(text) = text else {
            return Err(runtime_error(&format!("no recorded call at '{path}'")));
        };
        let arguments = snapshot_of(py, &text)?;
        let answer = run_entry(py, entry, invocation.bind(py), arguments.bind(py))?;
        put_snapshot(py, invocation.bind(py), answer.bind(py), "answer")?;
        answers.push(answer);
    }
    answer_list(py, answers)
}

fn run_entry(
    py: Python<'_>,
    entry: &Bound<'_, PyAny>,
    environ: &Bound<'_, PyAny>,
    arguments: &Bound<'_, PyAny>,
) -> PyResult<P> {
    if !is_a(py, entry, REFLECT_MODULE, "VObject")? {
        return Err(runtime_error(
            "the entry is a name or a call given to reenact, and this is no such value",
        ));
    }
    let runner = match current_run(py)? {
        Some(runner) => runner,
        None => return Err(runtime_error("reenact runs inside a run, and none is here")),
    };
    let args_node = if is_a(py, arguments, REFLECT_MODULE, "VObject")? {
        g(arguments, "data")?
    } else if is_ast_any(py, arguments)? {
        arguments.clone()
    } else {
        g(&viba_data(py, arguments)?.bind(py).clone(), "data")?
    };
    let entry_data = g(entry, "data")?;
    let call = make_partial(
        py,
        make_partial(
            py,
            make_type_ref(py, "apply")?,
            make_tagged(py, "$f", entry_data.unbind())?,
        )?,
        make_tagged(py, "$args", args_node.unbind())?,
    )?;
    let module = g(entry, "data_module")?;
    let module = if module.is_none() {
        no_module(py)?
    } else {
        module.unbind()
    };
    let caller = new_activation(
        py,
        runner.bind(py),
        module.bind(py),
        environ,
        "reenact",
        py.None().into_bound(py).as_any(),
        None,
        None,
        None,
    )?;
    let host = shim_class(py, "_Host")?.call1((environ,))?.unbind();
    let answered = act_apply_without_environ(
        py,
        caller.bind(py),
        call.bind(py),
        &[],
        host.bind(py),
        Vec::new(),
    )?;
    if is_ok(py, answered.bind(py))? {
        let value = ok_value(answered.bind(py))?;
        if is_shim(py, &value, "_VibaData")? {
            return Ok(g(&value, "node")?.unbind());
        }
        if is_shim(py, &value, "_Host")? {
            return Ok(g(&value, "obj")?.unbind());
        }
        return Ok(value.unbind());
    }
    if is_type(py, answered.bind(py), "Err")? {
        return Err(runtime_error(&gs(answered.bind(py), "msg")?));
    }
    let raised = shim_class(py, "_Raised")?.call1((answered,))?.unbind();
    Err(PyErr::from_value(raised.bind(py).clone()))
}

// ----------------------------------------------------------------------
// What a run answers, as viba data
// ----------------------------------------------------------------------

fn builtin_dir(py: Python<'_>) -> PyResult<String> {
    type_mod(py)?.getattr("BUILTIN_DIR")?.str()?.extract()
}

fn builtin_concept_dir(py: Python<'_>) -> PyResult<String> {
    type_mod(py)?
        .getattr("BUILTIN_CONCEPT_DIR")?
        .str()?
        .extract()
}

fn builtin_module(py: Python<'_>) -> PyResult<P> {
    Ok(type_mod(py)?.getattr("BUILTIN_MODULE")?.unbind())
}

static NO_MODULE: PyOnceLock<P> = PyOnceLock::new();

fn no_module(py: Python<'_>) -> PyResult<P> {
    if let Some(module) = NO_MODULE.get(py) {
        return Ok(module.clone_ref(py));
    }
    let made = type_mod(py)?
        .getattr("custom_module")?
        .call1(("",))?
        .unbind();
    let _ = NO_MODULE.set(py, made.clone_ref(py));
    Ok(made)
}

fn result_module(py: Python<'_>) -> PyResult<P> {
    static RESULT: PyOnceLock<P> = PyOnceLock::new();
    if let Some(module) = RESULT.get(py) {
        return Ok(module.clone_ref(py));
    }
    let file = format!("{}/{RESULT_MODULE_NAME}.viba", builtin_dir(py)?);
    let source: String = module(py, "pathlib")?
        .getattr("Path")?
        .call1((file.clone(),))?
        .call_method0("read_text")?
        .extract()?;
    let tree = ast_mod(py)?.getattr("parse")?.call1((source,))?.unbind();
    let kwargs = PyDict::new(py);
    kwargs.set_item("name", RESULT_MODULE_NAME)?;
    kwargs.set_item("file_path", file)?;
    kwargs.set_item("root", builtin_dir(py)?)?;
    let made = type_class(py, "CustomModuleType")?
        .call((tree,), Some(&kwargs))?
        .unbind();
    let _ = RESULT.set(py, made.clone_ref(py));
    Ok(made)
}

fn interpret_result_type(py: Python<'_>) -> PyResult<P> {
    static TYPE: PyOnceLock<P> = PyOnceLock::new();
    if let Some(found) = TYPE.get(py) {
        return Ok(found.clone_ref(py));
    }
    let module = result_module(py)?;
    let node = make_type_ref(py, RESULT_TYPE_NAME)?;
    let made = make_ast_node_type(py, node.bind(py), module.bind(py))?;
    let _ = TYPE.set(py, made.clone_ref(py));
    Ok(made)
}

fn interpret_error_descriptor(py: Python<'_>) -> PyResult<P> {
    static DESCRIPTOR: PyOnceLock<P> = PyOnceLock::new();
    if let Some(found) = DESCRIPTOR.get(py) {
        return Ok(found.clone_ref(py));
    }
    let module = result_module(py)?;
    let node = make_type_ref(py, "InterpretError")?;
    let node_type = make_ast_node_type(py, node.bind(py), module.bind(py))?;
    let made = descriptor_of_node_type(py, node_type.bind(py))?;
    let _ = DESCRIPTOR.set(py, made.clone_ref(py));
    Ok(made)
}

fn underlying_op_err_descriptor(py: Python<'_>) -> PyResult<P> {
    static DESCRIPTOR: PyOnceLock<P> = PyOnceLock::new();
    if let Some(found) = DESCRIPTOR.get(py) {
        return Ok(found.clone_ref(py));
    }
    let module = result_module(py)?;
    let node = make_type_ref(py, "UnderlyingOpErr")?;
    let node_type = make_ast_node_type(py, node.bind(py), module.bind(py))?;
    let made = descriptor_of_node_type(py, node_type.bind(py))?;
    let _ = DESCRIPTOR.set(py, made.clone_ref(py));
    Ok(made)
}

fn member_tag(name: &str) -> &'static str {
    match name {
        "msg" => "$msg",
        "stack" => "$stack",
        "file_path" => "$file_path",
        "lineno" => "$lineno",
        "module_path" => "$module_path",
        "full_qualified_func_name" => "$full_qualified_func_name",
        "call" => "$call",
        "api_name" => "$api_name",
        "args" => "$args",
        _ => "",
    }
}

fn builtin_unit(py: Python<'_>, node: &Bound<'_, PyAny>) -> PyResult<Option<P>> {
    if is_ast(py, node, "Nil")? {
        return Ok(Some(type_class(py, "NilType")?.unbind()));
    }
    if is_ast(py, node, "Never")? {
        return Ok(Some(type_class(py, "NeverType")?.unbind()));
    }
    if !is_ast(py, node, "TypeRef")? {
        return Ok(None);
    }
    let lookup = builtin_module(py)?.bind(py).clone();
    let found = lookup.call_method1("lookup", (g(node, "name")?,))?;
    if is_ok(py, &found)? {
        let value = ok_value(&found)?;
        if is_type(py, &value, "NilType")? {
            return Ok(Some(type_class(py, "NilType")?.unbind()));
        }
        if is_type(py, &value, "NeverType")? {
            return Ok(Some(type_class(py, "NeverType")?.unbind()));
        }
    }
    Ok(None)
}

fn tagged_piece<'py>(
    py: Python<'py>,
    body: &Bound<'py, PyAny>,
    tag: &str,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let product = if is_ast(py, body, "TypeDefinition")? {
        g(body, "body")?
    } else {
        body.clone()
    };
    for factor in product_elements(py, &product)? {
        let factor = factor.clone();
        if is_ast(py, &factor, "Tagged")? && gs(&factor, "tag")? == format!("${tag}") {
            return Ok(Some(g(&factor, "type")?));
        }
    }
    Ok(None)
}

fn builtin_member_data(py: Python<'_>, inner: &Bound<'_, PyAny>) -> PyResult<P> {
    let descriptor = descriptor_of(py, inner, builtin_module(py)?.bind(py))?;
    let node = make_vobject(py, descriptor.bind(py), inner, None, None)?;
    let data = make_viba_data(py, node)?;
    make_ok(py, data)
}

fn never_viba_data(py: Python<'_>) -> PyResult<P> {
    let node = make_ast0(py, "Never")?;
    let module = no_module(py)?;
    let descriptor = piece_descriptor(py, node.bind(py), module.bind(py))?;
    let object = make_vobject(py, descriptor.bind(py), node.bind(py), None, None)?;
    make_viba_data(py, object)
}

fn sum_branches(py: Python<'_>, node: &Bound<'_, PyAny>) -> PyResult<Vec<P>> {
    let mut out = Vec::new();
    if is_ast(py, node, "Sum")? {
        out.extend(sum_branches(py, &g(node, "left")?)?);
        out.extend(sum_branches(py, &g(node, "right")?)?);
        return Ok(out);
    }
    if is_ast(py, node, "SumChain")? {
        for element in g(node, "elements")?.try_iter()? {
            out.push(element?.unbind());
        }
        return Ok(out);
    }
    out.push(node.clone().unbind());
    Ok(out)
}

fn descriptor_module(py: Python<'_>, descriptor: &Bound<'_, PyAny>) -> PyResult<Option<P>> {
    if descriptor.is_none() {
        return Ok(None);
    }
    let mut source = g(descriptor, "container_module").unwrap_or_else(|_| py.None().into_bound(py));
    if source.is_none() {
        if let Ok(resolvable) = g(descriptor, "resolvable_type") {
            source =
                g(&resolvable, "container_module").unwrap_or_else(|_| py.None().into_bound(py));
        }
    }
    if source.is_none() {
        Ok(None)
    } else {
        Ok(Some(source.unbind()))
    }
}

fn source_module_of(py: Python<'_>, node: &Bound<'_, PyAny>) -> PyResult<Option<P>> {
    match g(node, "descriptor") {
        Ok(descriptor) => descriptor_module(py, &descriptor),
        Err(_) => Ok(None),
    }
}

fn viba_data(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<P> {
    if is_a(py, value, REFLECT_MODULE, "VObject")? {
        return Ok(value.clone().unbind());
    }
    if is_ast_any(py, value)? {
        let module = no_module(py)?;
        let descriptor = piece_descriptor(py, value, module.bind(py))?;
        return make_vobject(py, descriptor.bind(py), value, None, None);
    }
    let scalar = value.is_none()
        || value.is_instance_of::<pyo3::types::PyBool>()
        || value.is_instance_of::<pyo3::types::PyInt>()
        || value.is_instance_of::<pyo3::types::PyFloat>()
        || value.is_instance_of::<PyString>();
    if !scalar {
        return Err(runtime_error(&format!(
            "cannot take {} as viba data: only a VObject, an AST piece, a scalar, or None",
            class_name(py, value)
        )));
    }
    let node = if value.is_none() {
        make_ast0(py, "Nil")?
    } else {
        let constant = ast_class(py, "Constant")?.call1((value,))?;
        constant.unbind()
    };
    let module = no_module(py)?;
    let descriptor = piece_descriptor(py, node.bind(py), module.bind(py))?;
    make_vobject(py, descriptor.bind(py), node.bind(py), None, None)
}

fn failure_viba_data(
    py: Python<'_>,
    tag: &str,
    msg: &str,
    module_path: &str,
    full_qualified_func_name: &str,
    call: &Bound<'_, PyAny>,
) -> PyResult<P> {
    let members = members_viba_data(
        py,
        vec![
            (
                member_tag("msg"),
                viba_data(py, &py_string(py, msg).into_bound(py))?,
            ),
            (
                member_tag("module_path"),
                viba_data(py, &py_string(py, module_path).into_bound(py))?,
            ),
            (
                member_tag("full_qualified_func_name"),
                viba_data(py, &py_string(py, full_qualified_func_name).into_bound(py))?,
            ),
            (member_tag("call"), viba_data(py, call)?),
        ],
    )?;
    tagged_viba_data(py, tag, members.bind(py))
}

fn result_viba_data(py: Python<'_>, answer: &Bound<'_, PyAny>) -> PyResult<P> {
    let (chosen, ok_branch, err_branch);
    if is_ok(py, answer)? {
        let payload = ok_value(answer)?;
        let branch = descriptor_of_tagged(
            py,
            OK_TAG,
            &g(&payload, "descriptor")?,
            &make_tagged(py, OK_TAG, g(&payload, "data")?.unbind())?
                .bind(py)
                .clone(),
            result_module(py)?.bind(py),
        )?;
        chosen = make_tagged(py, OK_TAG, g(&payload, "data")?.unbind())?;
        ok_branch = branch;
        err_branch = declared_branch(py, ERR_TAG)?;
    } else {
        let payload = error_viba_data(py, answer)?;
        chosen = make_tagged(py, ERR_TAG, g(payload.bind(py), "data")?.unbind())?;
        ok_branch = declared_branch(py, OK_TAG)?;
        err_branch = descriptor_of_tagged(
            py,
            ERR_TAG,
            interpret_error_descriptor(py)?.bind(py),
            &make_tagged(py, ERR_TAG, g(payload.bind(py), "data")?.unbind())?
                .bind(py)
                .clone(),
            result_module(py)?.bind(py),
        )?;
    }
    let chain = viba_chain_descriptor(
        py,
        empty_pool(py)?,
        make_ast_node_type(py, chosen.bind(py), result_module(py)?.bind(py))?,
        vec![ok_branch, err_branch],
    )?;
    let descriptor = viba_type_descriptor(py, "sum", chain)?;
    make_vobject(py, descriptor.bind(py), chosen.bind(py), None, None)
}

fn declared_branch(py: Python<'_>, tag: &str) -> PyResult<P> {
    let module = result_module(py)?;
    let definition = module
        .bind(py)
        .call_method1("lookup_local", (RESULT_TYPE_NAME,))?;
    if !is_ok(py, &definition)? {
        return Err(type_error(&format!(
            "cannot take {RESULT_TYPE_NAME}: {}",
            gs(&definition, "msg")?
        )));
    }
    let body = g(&ok_value(&definition)?, "ast_node")?;
    let body = g(&body, "body")?;
    for branch in sum_branches(py, &body)? {
        let branch = branch.bind(py);
        if is_ast(py, branch, "Tagged")? && gs(branch, "tag")? == tag {
            let inner = descriptor_of(py, &g(branch, "type")?, module.bind(py))?;
            return descriptor_of_tagged(py, tag, inner.bind(py), branch, module.bind(py));
        }
    }
    Err(type_error(&format!(
        "{RESULT_TYPE_NAME} declares no branch {tag}"
    )))
}

fn error_viba_data(py: Python<'_>, error: &Bound<'_, PyAny>) -> PyResult<P> {
    if is_type(py, error, "Err")? && !is_type(py, error, "InterpretError")? {
        let members = members_viba_data(
            py,
            vec![
                (member_tag("msg"), viba_data(py, &g(error, "msg")?)?),
                (
                    member_tag("stack"),
                    stack_viba_data(py, &pylist(py, Vec::new())?.into_bound(py))?,
                ),
            ],
        )?;
        return tagged_viba_data(py, PROGRAM_ERR_TAG, members.bind(py));
    }
    if is_type(py, error, "VibaProgramErr")? {
        let stack = g(error, "stack")?;
        let members = members_viba_data(
            py,
            vec![
                (member_tag("msg"), viba_data(py, &g(error, "msg")?)?),
                (member_tag("stack"), stack_viba_data(py, &stack)?),
            ],
        )?;
        return tagged_viba_data(py, PROGRAM_ERR_TAG, members.bind(py));
    }
    if is_type(py, error, "UnderlyingOpErr")? {
        let call = g(error, "call")?;
        let members = members_viba_data(
            py,
            vec![
                (member_tag("msg"), viba_data(py, &g(error, "msg")?)?),
                (
                    member_tag("module_path"),
                    viba_data(py, &g(error, "module_path")?)?,
                ),
                (
                    member_tag("full_qualified_func_name"),
                    viba_data(py, &g(error, "full_qualified_func_name")?)?,
                ),
                (member_tag("call"), viba_data(py, &call)?),
            ],
        )?;
        let tag: String = gs(error, "tag")?;
        return tagged_viba_data(py, &tag, members.bind(py));
    }
    if is_type(py, error, "EnvironmentApiInvalidArgumentErr")? {
        let arguments = g(error, "arguments")?;
        let members = members_viba_data(
            py,
            vec![
                (member_tag("msg"), viba_data(py, &g(error, "msg")?)?),
                (
                    member_tag("api_name"),
                    viba_data(py, &g(error, "api_name")?)?,
                ),
                (member_tag("args"), viba_data(py, &arguments)?),
            ],
        )?;
        return tagged_viba_data(py, ENVIRONMENT_API_TAG, members.bind(py));
    }
    Err(type_error(&format!(
        "no viba data for {}",
        class_name(py, error)
    )))
}

fn stack_viba_data(py: Python<'_>, stack: &Bound<'_, PyAny>) -> PyResult<P> {
    let mut frames: Vec<P> = Vec::new();
    for frame in stack.try_iter()? {
        let frame = frame?;
        frames.push(members_viba_data(
            py,
            vec![
                (
                    member_tag("file_path"),
                    viba_data(py, &g(&frame, "file_path")?)?,
                ),
                (member_tag("lineno"), viba_data(py, &g(&frame, "lineno")?)?),
            ],
        )?);
    }
    let mut elements = Vec::new();
    for frame in &frames {
        elements.push(g(frame.bind(py), "data")?.unbind());
    }
    let node = ast_class(py, "TypeApp")?
        .call1((py_string(py, "ListLiteral"), PyList::new(py, elements)?))?
        .unbind();
    let descriptor = descriptor_of(py, node.bind(py), builtin_module(py)?.bind(py))?;
    make_vobject(py, descriptor.bind(py), node.bind(py), None, None)
}

fn members_viba_data(py: Python<'_>, members: Vec<(&str, P)>) -> PyResult<P> {
    let mut factors = Vec::new();
    let mut values = Vec::new();
    for (tag, member) in &members {
        let inner = g(member.bind(py), "data")?.unbind();
        factors.push(make_tagged(py, tag, inner)?);
        values.push(make_viba_data(py, member.clone_ref(py))?);
    }
    let node = ast_class(py, "ProductChain")?
        .call1((PyList::new(py, factors)?,))?
        .unbind();
    let module = no_module(py)?;
    let descriptor = descriptor_of_values(py, node.bind(py), module.bind(py), values)?;
    make_vobject(py, descriptor.bind(py), node.bind(py), None, None)
}

fn tagged_viba_data(py: Python<'_>, tag: &str, inner: &Bound<'_, PyAny>) -> PyResult<P> {
    let node = make_tagged(py, tag, g(inner, "data")?.unbind())?;
    let module = no_module(py)?;
    let descriptor = descriptor_of_tagged(
        py,
        tag,
        &g(inner, "descriptor")?,
        node.bind(py),
        module.bind(py),
    )?;
    make_vobject(py, descriptor.bind(py), node.bind(py), None, None)
}

fn handed_back_failure(py: Python<'_>, answer: &Bound<'_, PyAny>) -> PyResult<Option<P>> {
    let piece = if is_a(py, answer, REFLECT_MODULE, "VObject")? {
        g(answer, "data")?
    } else {
        answer.clone()
    };
    if !is_ast(py, &piece, "Tagged")? {
        return Ok(None);
    }
    let tag: String = gs(&piece, "tag")?;
    if tag != NOT_IMPLEMENTED_TAG && tag != FAILURE_TAG {
        return Ok(None);
    }
    let node = make_vobject(
        py,
        underlying_op_err_descriptor(py)?.bind(py),
        &g(&piece, "type")?,
        None,
        None,
    )?;
    let reason = if tag == NOT_IMPLEMENTED_TAG {
        REASON_NO_IMPLEMENTATION
    } else {
        REASON_RAISED
    };
    let mut call = node_member(py, node.bind(py), "call")?;
    if let Some(found) = &call {
        if is_a(py, found.bind(py), REFLECT_MODULE, "VObject")?
            && is_ast(py, &g(found.bind(py), "data")?, "Nil")?
        {
            call = None;
        }
    }
    let msg = text_member(py, node.bind(py), "msg")?;
    let msg = if msg.is_empty() {
        reason.to_string()
    } else {
        msg
    };
    let module_path = text_member(py, node.bind(py), "module_path")?;
    let full_name = text_member(py, node.bind(py), "full_qualified_func_name")?;
    let made = under_op_err(
        py,
        &msg,
        &module_path,
        &full_name,
        call.map(|call| call.bind(py).clone()).as_ref(),
        Some(&tag),
    )?;
    Ok(Some(made))
}

fn text_member(py: Python<'_>, node: &Bound<'_, PyAny>, member: &str) -> PyResult<String> {
    let given = reflect_get(py, node, member_tag(member))?;
    let value = if is_ok(py, &given)? {
        ok_value(&given)?
    } else {
        return Ok(String::new());
    };
    if value.is_none() {
        return Ok(String::new());
    }
    let leaf = reflect_access(py)?.call_method1("leaf", (value,))?;
    if is_ok(py, &leaf)? {
        let leaf = ok_value(&leaf)?;
        if leaf.is_instance_of::<PyString>() {
            return leaf.extract();
        }
    }
    Ok(String::new())
}

fn node_member(py: Python<'_>, node: &Bound<'_, PyAny>, member: &str) -> PyResult<Option<P>> {
    let given = reflect_get(py, node, member_tag(member))?;
    let value = if is_ok(py, &given)? {
        ok_value(&given)?
    } else {
        return Ok(None);
    };
    if value.is_none() {
        return Ok(None);
    }
    Ok(Some(viba_data(py, &value)?))
}

// ----------------------------------------------------------------------
// Partial's own helpers, taken from the Rust layer behind its Python API
// ----------------------------------------------------------------------

fn product_elements<'py>(
    py: Python<'py>,
    node: &Bound<'py, PyAny>,
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let mut out = Vec::new();
    for element in partial_mod(py)?
        .getattr("product_elements")?
        .call1((node,))?
        .try_iter()?
    {
        out.push(element?);
    }
    Ok(out)
}

fn parameters_of<'py>(
    py: Python<'py>,
    chain: &Bound<'py, PyAny>,
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let mut out = Vec::new();
    for element in partial_mod(py)?
        .getattr("parameters_of")?
        .call1((chain,))?
        .try_iter()?
    {
        out.push(element?);
    }
    Ok(out)
}

fn names_the_environment(py: Python<'_>, node: &Bound<'_, PyAny>) -> PyResult<bool> {
    partial_mod(py)?
        .getattr("names_the_environment")?
        .call1((node,))?
        .extract()
}

fn file_environment_result_problem(
    py: Python<'_>,
    tree: &Bound<'_, PyAny>,
) -> PyResult<Option<String>> {
    let found = partial_mod(py)?
        .getattr("file_environment_result_problem")?
        .call1((tree,))?;
    if found.is_none() {
        return Ok(None);
    }
    Ok(Some(found.extract()?))
}

fn file_pattern_problem(
    py: Python<'_>,
    tree: &Bound<'_, PyAny>,
    file_name: &str,
) -> PyResult<Option<String>> {
    let found = pattern_mod(py)?
        .getattr("file_pattern_problem")?
        .call1((tree, file_name))?;
    if found.is_none() {
        return Ok(None);
    }
    Ok(Some(found.extract()?))
}

fn takes_a_product(py: Python<'_>, node: &Bound<'_, PyAny>) -> PyResult<bool> {
    type_mod(py)?
        .getattr("takes_a_product")?
        .call1((node,))?
        .extract()
}

fn builtin_directory_name(py: Python<'_>, name: &str) -> PyResult<Option<String>> {
    let found = type_mod(py)?
        .getattr("builtin_directory_name")?
        .call1((name,))?;
    if found.is_none() {
        return Ok(None);
    }
    Ok(Some(found.extract()?))
}

fn module_get_type(py: Python<'_>, module: &Bound<'_, PyAny>, name: &str) -> PyResult<P> {
    Ok(type_mod(py)?
        .getattr("module_get_type")?
        .call1((module, name))?
        .unbind())
}

fn reduce_application(
    py: Python<'_>,
    node: &Bound<'_, PyAny>,
    module: &Bound<'_, PyAny>,
    argument_modules: &Bound<'_, PyAny>,
    parts: &Bound<'_, PyAny>,
) -> PyResult<P> {
    Ok(pattern_mod(py)?
        .getattr("reduce_application")?
        .call1((node, module, argument_modules, parts))?
        .unbind())
}

fn tagged_type_of(
    py: Python<'_>,
    node: &Bound<'_, PyAny>,
    module: &Bound<'_, PyAny>,
) -> PyResult<P> {
    Ok(pattern_mod(py)?
        .getattr("tagged_type_of")?
        .call1((node, module))?
        .unbind())
}

fn load_generic(
    py: Python<'_>,
    directory: &str,
    name: &str,
    source_of: &Bound<'_, PyAny>,
    list_of: &Bound<'_, PyAny>,
    module_of: &Bound<'_, PyAny>,
) -> PyResult<P> {
    Ok(pattern_mod(py)?
        .getattr("load_generic")?
        .call1((directory, name, source_of, list_of, module_of))?
        .unbind())
}

fn is_sub_type_value(
    py: Python<'_>,
    sub_type: &Bound<'_, PyAny>,
    sup_type: &Bound<'_, PyAny>,
) -> PyResult<P> {
    let kwargs = PyDict::new(py);
    kwargs.set_item("config", language_config(py)?)?;
    Ok(is_sub_type_mod(py)?
        .getattr("is_sub_type")?
        .call((sub_type, sup_type), Some(&kwargs))?
        .unbind())
}

// ----------------------------------------------------------------------
// One run: the files it has loaded, and where it looks for more
// ----------------------------------------------------------------------

fn path_of_string(py: Python<'_>, text: &str) -> PyResult<P> {
    Ok(module(py, "pathlib")?
        .getattr("Path")?
        .call1((text,))?
        .unbind())
}

fn path_text(py: Python<'_>, path: &Bound<'_, PyAny>) -> PyResult<String> {
    Ok(path.str()?.to_string())
}

fn path_name(py: Python<'_>, path: &Bound<'_, PyAny>) -> PyResult<String> {
    g(path, "name")?.extract()
}

fn path_suffix(py: Python<'_>, path: &Bound<'_, PyAny>) -> PyResult<String> {
    g(path, "suffix")?.extract()
}

fn path_parent(py: Python<'_>, path: &Bound<'_, PyAny>) -> PyResult<P> {
    Ok(g(path, "parent")?.unbind())
}

fn path_div(py: Python<'_>, path: &Bound<'_, PyAny>, part: &str) -> PyResult<P> {
    Ok(path.call_method1("__truediv__", (part,))?.unbind())
}

fn path_join(py: Python<'_>, path: &Bound<'_, PyAny>, parts: Vec<String>) -> PyResult<P> {
    let args: Vec<P> = parts.iter().map(|part| py_string(py, part)).collect();
    let tuple = PyTuple::new(py, args)?;
    Ok(path.call_method("joinpath", tuple, None)?.unbind())
}

fn path_equals(
    py: Python<'_>,
    left: &Bound<'_, PyAny>,
    right: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    Ok(left.eq(right)?)
}

fn runner_get<'py>(runner: &Bound<'py, PyAny>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    g(runner, name)
}

fn runner_key(py: Python<'_>, path: &Bound<'_, PyAny>) -> PyResult<String> {
    let os = module(py, "os")?;
    let text = path_text(py, path)?;
    Ok(os
        .getattr("path")?
        .call_method1("normpath", (text,))?
        .extract()?)
}

fn runner_source(
    py: Python<'_>,
    runner: &Bound<'_, PyAny>,
    path: &Bound<'_, PyAny>,
) -> PyResult<(Option<String>, Option<String>)> {
    let get_file = runner_get(runner, "get_file")?;
    if !get_file.is_none() {
        let text = path_text(py, path)?;
        match get_file.call1((text,)) {
            Ok(source) => {
                if source.is_none() {
                    return Ok((None, None));
                }
                if !source.is_instance_of::<PyString>() {
                    return Ok((
                        None,
                        Some(format!(
                            "get_file({}) answered {}, not the file's text",
                            path_text(py, path)?,
                            class_name(py, &source)
                        )),
                    ));
                }
                return Ok((Some(source.extract()?), None));
            }
            Err(error) => {
                if error.is_instance_of::<pyo3::exceptions::PyFileNotFoundError>(py) {
                    return Ok((None, None));
                }
                if is_py_exception(py, &error) {
                    return Ok((
                        None,
                        Some(format!(
                            "get_file({}) raised {}",
                            path_text(py, path)?,
                            format_exc(py, &error)
                        )),
                    ));
                }
                return Err(error);
            }
        }
    }
    match path.call_method0("read_text") {
        Ok(source) => Ok((Some(source.extract()?), None)),
        Err(error) => {
            if error.is_instance_of::<pyo3::exceptions::PyFileNotFoundError>(py) {
                Ok((None, None))
            } else if error.is_instance_of::<pyo3::exceptions::PyOSError>(py) {
                Ok((
                    None,
                    Some(format!(
                        "cannot take {}: {}",
                        path_text(py, path)?,
                        format_exc(py, &error)
                    )),
                ))
            } else {
                Err(error)
            }
        }
    }
}

fn runner_bind(
    py: Python<'_>,
    runner: &Bound<'_, PyAny>,
    module: P,
    path: &Bound<'_, PyAny>,
    name: &str,
) -> PyResult<P> {
    runner_get(runner, "by_name")?.set_item(name, module.bind(py))?;
    runner_get(runner, "path_of")?.set_item(name, path_text(py, path)?)?;
    make_ok(py, module)
}

fn runner_file_of(
    py: Python<'_>,
    runner: &Bound<'_, PyAny>,
    path: &Bound<'_, PyAny>,
    name: &str,
    source: &str,
    file_path: Option<&str>,
    root: Option<&str>,
    near: Option<&str>,
) -> PyResult<P> {
    let key = runner_key(py, path)?;
    let by_path = runner_get(runner, "by_path")?;
    if !by_path
        .call_method1("__contains__", (&key,))?
        .extract::<bool>()?
    {
        let built = runner_module_of(py, runner, path, name, source, near, file_path, root)?;
        if is_stopped(py, built.bind(py))? {
            return Ok(built);
        }
        by_path.set_item(&key, ok_value(built.bind(py))?)?;
    }
    let module = by_path.get_item(&key)?.unbind();
    runner_bind(py, runner, module, path, name)
}

#[allow(clippy::too_many_arguments)]
fn runner_module_of(
    py: Python<'_>,
    runner: &Bound<'_, PyAny>,
    path: &Bound<'_, PyAny>,
    name: &str,
    source: &str,
    near: Option<&str>,
    file_path: Option<&str>,
    root: Option<&str>,
) -> PyResult<P> {
    let file_name = path_name(py, path)?;
    if file_name
        == pattern_mod(py)?
            .getattr("GENERIC_FILE")?
            .extract::<String>()?
            .as_str()
    {
        return runner_generic_of(py, runner, path, name, file_path, root);
    }
    if path_suffix(py, path)? == ".md" {
        return runner_generic_of_markdown(py, runner, path, name, source, root);
    }
    let tree = match ast_mod(py)?.getattr("parse")?.call1((source,)) {
        Ok(tree) => tree.unbind(),
        Err(error) => {
            if error.is_instance_of::<pyo3::exceptions::PySyntaxError>(py) {
                return program_err(
                    py,
                    &format!(
                        "cannot parse {}: {}",
                        path_text(py, path)?,
                        error
                            .value(py)
                            .str()
                            .map(|text| text.to_string())
                            .unwrap_or_default()
                    ),
                );
            }
            return Err(error);
        }
    };
    let kwargs = PyDict::new(py);
    kwargs.set_item("name", name)?;
    if let Some(file_path) = file_path {
        kwargs.set_item("file_path", file_path)?;
    }
    if let Some(root) = root {
        kwargs.set_item("root", root)?;
    }
    runner_built(py, runner, path, tree.bind(py), near, &kwargs)
}

fn runner_built(
    py: Python<'_>,
    runner: &Bound<'_, PyAny>,
    path: &Bound<'_, PyAny>,
    tree: &Bound<'_, PyAny>,
    near: Option<&str>,
    kwargs: &Bound<'_, PyDict>,
) -> PyResult<P> {
    if let Some(problem) = file_environment_result_problem(py, tree)? {
        return program_err(py, &format!("{}: {problem}", path_text(py, path)?));
    }
    let file_name = path_name(py, path)?;
    if let Some(problem) = file_pattern_problem(py, tree, &file_name)? {
        return program_err(py, &format!("{}: {problem}", path_text(py, path)?));
    }
    let imports = PyDict::new(py);
    for statement in g(tree, "body")?.try_iter()? {
        let statement = statement?;
        if !is_ast(py, &statement, "Import")? {
            continue;
        }
        let module_name: String = g(&statement, "module")?.extract()?;
        let alias = g(&statement, "alias")?;
        let bound = if alias.is_none() {
            module_name.clone()
        } else {
            alias.extract::<String>()?
        };
        imports.set_item(bound, module_name)?;
    }
    let module = type_class(py, "CustomModuleType")?
        .call((tree,), Some(kwargs))?
        .unbind();
    let beside = match near {
        Some(near) => near.to_string(),
        None => path_text(py, path)?,
    };
    let runner_ref = runner.clone().unbind();
    let environment = PyCFunction::new_closure(py, None, None, move |args, _kwargs| {
        Python::attach(|py| {
            let asked = args.get_item(0)?;
            let asked: String = asked.extract()?;
            let imported = runner_imported(py, runner_ref.bind(py), &asked, Some(&beside))?;
            if is_ok(py, imported.bind(py))? {
                Ok::<P, PyErr>(imported)
            } else {
                let message: String = g(imported.bind(py), "msg")?.extract()?;
                Ok::<P, PyErr>(type_class(py, "Err")?.call1((message,))?.unbind())
            }
        })
    })?;
    module.bind(py).setattr("module_environment", environment)?;
    module.bind(py).setattr("imports", imports)?;
    make_ok(py, module)
}

fn runner_generic_of(
    py: Python<'_>,
    runner: &Bound<'_, PyAny>,
    path: &Bound<'_, PyAny>,
    name: &str,
    file_path: Option<&str>,
    root: Option<&str>,
) -> PyResult<P> {
    let parent = path_parent(py, path)?;
    let parent_text = path_text(py, parent.bind(py))?;
    let root_text = match root {
        Some(root) => root.to_string(),
        None => parent_text.clone(),
    };
    let runner_ref = runner.clone().unbind();
    let source_of = PyCFunction::new_closure(py, None, None, move |args, _kwargs| {
        Python::attach(|py| {
            let file_path: String = args.get_item(0)?.extract()?;
            runner_source_text(py, runner_ref.bind(py), &file_path)
        })
    })?;
    let runner_ref = runner.clone().unbind();
    let list_of = PyCFunction::new_closure(py, None, None, move |args, _kwargs| {
        Python::attach(|py| {
            let directory: String = args.get_item(0)?.extract()?;
            runner_list_files(py, runner_ref.bind(py), &directory)
        })
    })?;
    let runner_ref = runner.clone().unbind();
    let root_ref = root_text.clone();
    let module_of = PyCFunction::new_closure(py, None, None, move |args, _kwargs| {
        Python::attach(|py| {
            let entry_path: String = args.get_item(0)?.extract()?;
            let entry_name: String = args.get_item(1)?.extract()?;
            let text: String = args.get_item(2)?.extract()?;
            let path = path_of_string(py, &entry_path)?.bind(py).clone();
            runner_file_of(
                py,
                runner_ref.bind(py),
                &path,
                &entry_name,
                &text,
                Some(&entry_path),
                Some(&root_ref),
                None,
            )
        })
    })?;
    let generic = load_generic(
        py,
        &parent_text,
        name,
        source_of.as_any(),
        list_of.as_any(),
        module_of.as_any(),
    )?;
    if is_stopped(py, generic.bind(py))? {
        return Ok(generic);
    }
    let key = runner_key(py, path)?;
    runner_get(runner, "by_path")?.set_item(&key, ok_value(generic.bind(py))?)?;
    runner_bind(py, runner, ok_value(generic.bind(py))?.unbind(), path, name)
}

fn runner_generic_of_markdown(
    py: Python<'_>,
    runner: &Bound<'_, PyAny>,
    path: &Bound<'_, PyAny>,
    name: &str,
    source: &str,
    root: Option<&str>,
) -> PyResult<P> {
    let files = match module(py, "rs_viba.markdown")?
        .getattr("files_of")?
        .call1((source,))
    {
        Ok(files) => files,
        Err(error) => {
            if error.is_instance_of::<pyo3::exceptions::PySyntaxError>(py) {
                return program_err(
                    py,
                    &format!(
                        "cannot parse {}: {}",
                        path_text(py, path)?,
                        error
                            .value(py)
                            .str()
                            .map(|text| text.to_string())
                            .unwrap_or_default()
                    ),
                );
            }
            return Err(error);
        }
    };
    let by_name = PyDict::new(py);
    for pair in files.try_iter()? {
        let pair = pair?;
        let file_name: String = pair.get_item(0)?.extract()?;
        let text = pair.get_item(1)?;
        by_name.set_item(file_name, text)?;
    }
    let path_text_value = path_text(py, path)?;
    let root_text = match root {
        Some(root) => root.to_string(),
        None => path_parent(py, path).and_then(|parent| path_text(py, parent.bind(py)))?,
    };
    let runner_ref = runner.clone().unbind();
    let by_name_ref = by_name.clone().unbind();
    let source_of = PyCFunction::new_closure(py, None, None, move |args, _kwargs| {
        Python::attach(|py| {
            let file_path: String = args.get_item(0)?.extract()?;
            let base: String = module(py, "os")?
                .getattr("path")?
                .call_method1("basename", (file_path,))?
                .extract()?;
            let by_name = by_name_ref.bind(py);
            let found: Bound<'_, PyAny> = by_name.call_method1("get", (&base,))?;
            if found.is_none() {
                Ok::<P, PyErr>(py.None())
            } else {
                Ok::<P, PyErr>(found.unbind())
            }
        })
    })?;
    let by_name_ref = by_name.clone().unbind();
    let listing = PyCFunction::new_closure(py, None, None, move |_args, _kwargs| {
        Python::attach(|py| {
            let by_name = by_name_ref.bind(py);
            let names = PyList::new(py, by_name.keys())?;
            let sorted = module(py, "builtins")?.getattr("sorted")?.call1((names,))?;
            make_ok_ref(py, &sorted)
        })
    })?;
    let runner_ref = runner.clone().unbind();
    let near_ref = path_text_value.clone();
    let root_ref = root_text.clone();
    let module_of = PyCFunction::new_closure(py, None, None, move |args, _kwargs| {
        Python::attach(|py| {
            let entry_path: String = args.get_item(0)?.extract()?;
            let entry_name: String = args.get_item(1)?.extract()?;
            let text: String = args.get_item(2)?.extract()?;
            let path = path_of_string(py, &entry_path)?.bind(py).clone();
            runner_file_of(
                py,
                runner_ref.bind(py),
                &path,
                &entry_name,
                &text,
                None,
                Some(&root_ref),
                Some(&near_ref),
            )
        })
    })?;
    let generic = load_generic(
        py,
        &path_text_value,
        name,
        source_of.as_any(),
        listing.as_any(),
        module_of.as_any(),
    )?;
    if is_stopped(py, generic.bind(py))? {
        return Ok(generic);
    }
    let key = runner_key(py, path)?;
    runner_get(runner, "by_path")?.set_item(&key, ok_value(generic.bind(py))?)?;
    runner_bind(py, runner, ok_value(generic.bind(py))?.unbind(), path, name)
}

fn runner_source_text(py: Python<'_>, runner: &Bound<'_, PyAny>, file_path: &str) -> PyResult<P> {
    let path = path_of_string(py, file_path)?;
    let (source, _problem) = runner_source(py, runner, path.bind(py))?;
    match source {
        Some(source) => Ok(py_string(py, &source)),
        None => Ok(py.None()),
    }
}

fn runner_list_files(py: Python<'_>, runner: &Bound<'_, PyAny>, directory: &str) -> PyResult<P> {
    let list_files = runner_get(runner, "list_files")?;
    if !list_files.is_none() {
        match list_files.call1((directory,)) {
            Ok(names) => {
                if names.is_none() {
                    return program_err(py, &format!("cannot take {directory}: no such directory"));
                }
                if !names.is_instance_of::<PyList>() && !names.is_instance_of::<PyTuple>() {
                    return program_err(
                        py,
                        &format!(
                            "list_files({directory}) answered {}, not the names inside it",
                            class_name(py, &names)
                        ),
                    );
                }
                let mut out = Vec::new();
                for name in names.try_iter()? {
                    out.push(py_string(py, &name?.str()?.to_string()));
                }
                return make_ok(py, pylist(py, out)?);
            }
            Err(error) => {
                if error.is_instance_of::<pyo3::exceptions::PyFileNotFoundError>(py) {
                    return program_err(py, &format!("cannot take {directory}: no such directory"));
                }
                if is_py_exception(py, &error) {
                    return program_err(
                        py,
                        &format!("list_files({directory}) raised {}", format_exc(py, &error)),
                    );
                }
                return Err(error);
            }
        }
    }
    if !runner_get(runner, "get_file")?.is_none() {
        return program_err(
            py,
            &format!(
                "cannot take {directory}: a host that serves the files itself serves \
                 the names inside a directory too (list_files)"
            ),
        );
    }
    match module(py, "os")?.call_method1("listdir", (directory,)) {
        Ok(names) => make_ok_ref(py, &names),
        Err(error) => {
            if error.is_instance_of::<pyo3::exceptions::PyOSError>(py) {
                program_err(
                    py,
                    &format!("cannot take {directory}: {}", format_exc(py, &error)),
                )
            } else {
                Err(error)
            }
        }
    }
}

fn runner_node_of(
    py: Python<'_>,
    runner: &Bound<'_, PyAny>,
    node: &Bound<'_, PyAny>,
    path: &Bound<'_, PyAny>,
) -> PyResult<P> {
    let data = g(node, "data")?;
    let tree = if is_ast(py, &data, "Module")? {
        data.clone()
    } else {
        let definition = ast_class(py, "TypeDefinition")?
            .call1((RET_NAME, data))?
            .unbind();
        ast_class(py, "Module")?.call1((PyList::new(py, [definition])?,))?
    };
    let kwargs = PyDict::new(py);
    kwargs.set_item("name", CODE_LABEL)?;
    runner_built(py, runner, path, &tree, Some(""), &kwargs)
}

fn runner_builtin_module(
    py: Python<'_>,
    runner: &Bound<'_, PyAny>,
    name: &str,
) -> PyResult<Option<P>> {
    let place = path_of_string(py, &format!("{}/{}", builtin_dir(py)?, name))?;
    let place = g(place.bind(py), "with_suffix")?.call1((".viba",))?;
    let key = runner_key(py, &place)?;
    let by_path = runner_get(runner, "by_path")?;
    if !by_path
        .call_method1("__contains__", (&key,))?
        .extract::<bool>()?
    {
        let (source, problem) = runner_source(py, runner, &place)?;
        if let Some(problem) = problem {
            return Ok(Some(program_err(py, &problem)?));
        }
        let Some(source) = source else {
            return Ok(None);
        };
        let built = runner_module_of(
            py,
            runner,
            &place,
            name,
            &source,
            None,
            Some(&path_text(py, &place)?),
            Some(&builtin_dir(py)?),
        )?;
        if is_stopped(py, built.bind(py))? {
            return Ok(Some(built));
        }
        by_path.set_item(&key, ok_value(built.bind(py))?)?;
    }
    Ok(Some(make_ok(py, by_path.get_item(&key)?.unbind())?))
}

fn runner_bases(py: Python<'_>, runner: &Bound<'_, PyAny>, near: Option<&str>) -> PyResult<Vec<P>> {
    let mut bases: Vec<P> = Vec::new();
    let mut texts: Vec<String> = Vec::new();
    if let Some(near) = near {
        if !near.is_empty() {
            let parent = path_parent(py, path_of_string(py, near)?.bind(py))?;
            texts.push(path_text(py, parent.bind(py))?);
        }
    }
    for path in runner_get(runner, "paths")?.try_iter()? {
        texts.push(path_text(py, &path?)?);
    }
    texts.push(builtin_dir(py)?);
    texts.push(builtin_concept_dir(py)?);
    let mut seen: Vec<String> = Vec::new();
    for text in texts {
        if seen.contains(&text) {
            continue;
        }
        seen.push(text.clone());
        bases.push(path_of_string(py, &text)?);
    }
    Ok(bases)
}

fn runner_sections(
    py: Python<'_>,
    runner: &Bound<'_, PyAny>,
    document: &Bound<'_, PyAny>,
    source: Option<&str>,
) -> PyResult<(Option<P>, Option<String>)> {
    let key = runner_key(py, document)?;
    let documents = runner_get(runner, "documents")?;
    if let Ok(cached) = documents.get_item(&key) {
        if !cached.is_none() {
            return Ok((Some(cached.unbind()), None));
        }
    }
    let source = match source {
        Some(source) => source.to_string(),
        None => {
            let (found, problem) = runner_source(py, runner, document)?;
            if let Some(problem) = problem {
                return Ok((None, Some(problem)));
            }
            match found {
                Some(found) => found,
                None => return Ok((None, None)),
            }
        }
    };
    let files = match module(py, "rs_viba.markdown")?
        .getattr("files_of")?
        .call1((source,))
    {
        Ok(files) => files,
        Err(error) => {
            if error.is_instance_of::<pyo3::exceptions::PySyntaxError>(py) {
                return Ok((
                    None,
                    Some(format!(
                        "cannot parse {}: {}",
                        path_text(py, document)?,
                        error
                            .value(py)
                            .str()
                            .map(|text| text.to_string())
                            .unwrap_or_default()
                    )),
                ));
            }
            return Err(error);
        }
    };
    let sections = PyDict::new(py);
    for pair in files.try_iter()? {
        let pair = pair?;
        let file_name: String = pair.get_item(0)?.extract()?;
        sections.set_item(file_name, pair.get_item(1)?)?;
    }
    documents.set_item(&key, &sections)?;
    Ok((Some(sections.unbind().into()), None))
}

fn runner_module_of_section(
    py: Python<'_>,
    runner: &Bound<'_, PyAny>,
    document: &Bound<'_, PyAny>,
    section: &str,
    text: &str,
    name: &str,
    base: &Bound<'_, PyAny>,
) -> PyResult<P> {
    let virtual_path = path_div(py, document, section)?;
    let key = runner_key(py, virtual_path.bind(py))?;
    let by_path = runner_get(runner, "by_path")?;
    if !by_path
        .call_method1("__contains__", (&key,))?
        .extract::<bool>()?
    {
        let virtual_path_bound = virtual_path.bind(py);
        let document_text = path_text(py, document)?;
        let base_text = path_text(py, base)?;
        let near = path_div(py, base, section)?;
        let near_text = path_text(py, near.bind(py))?;
        return runner_file_of(
            py,
            runner,
            virtual_path_bound,
            name,
            text,
            Some(&document_text),
            Some(&base_text),
            Some(&near_text),
        );
    }
    let module = by_path.get_item(&key)?.unbind();
    runner_bind(py, runner, module, virtual_path.bind(py), name)
}

fn runner_in_base(
    py: Python<'_>,
    runner: &Bound<'_, PyAny>,
    base: &Bound<'_, PyAny>,
    name: &str,
) -> PyResult<Option<P>> {
    let parts: Vec<String> = name.split('.').map(|part| part.to_string()).collect();
    let base_name = path_name(py, base)?;
    if base_name.ends_with(".viba.md") {
        let (sections, problem) = runner_sections(py, runner, base, None)?;
        if let Some(problem) = problem {
            return Ok(Some(program_err(py, &problem)?));
        }
        let Some(sections) = sections else {
            return Ok(None);
        };
        let section = format!("{}.viba", parts.last().unwrap());
        let sections = sections.bind(py);
        if !sections
            .call_method1("__contains__", (&section,))?
            .extract::<bool>()?
        {
            return Ok(None);
        }
        let text: String = sections.get_item(&section)?.extract()?;
        return Ok(Some(runner_module_of_section(
            py, runner, base, &section, &text, name, base,
        )?));
    }
    let rel = path_join(py, base, parts.clone())?;
    let section = format!("{}.viba", parts.last().unwrap());
    let parent = path_parent(py, rel.bind(py))?;
    let parent_doc = path_of_string(py, &format!("{}.viba.md", path_text(py, parent.bind(py))?))?;
    let (sections, problem) = runner_sections(py, runner, parent_doc.bind(py), None)?;
    if let Some(problem) = problem {
        return Ok(Some(program_err(py, &problem)?));
    }
    if let Some(sections) = sections {
        let sections = sections.bind(py);
        if !sections
            .call_method1("__contains__", (&section,))?
            .extract::<bool>()?
        {
            return Ok(None);
        }
        let text: String = sections.get_item(&section)?.extract()?;
        return Ok(Some(runner_module_of_section(
            py,
            runner,
            parent_doc.bind(py),
            &section,
            &text,
            name,
            base,
        )?));
    }
    let mut places: Vec<P> = Vec::new();
    let mut texts: Vec<String> = Vec::new();
    let rel_text = path_text(py, rel.bind(py))?;
    for place in [format!("{rel_text}.viba"), format!("{rel_text}.viba.md")] {
        texts.push(place);
    }
    let generic_file: String = pattern_mod(py)?.getattr("GENERIC_FILE")?.extract()?;
    let generic_place = path_div(py, rel.bind(py), &generic_file)?;
    texts.push(path_text(py, generic_place.bind(py))?);
    let base_text = path_text(py, base)?;
    texts.push(format!("{base_text}/{name}.viba"));
    texts.push(format!("{base_text}/{name}.viba.md"));
    let mut seen: Vec<String> = Vec::new();
    for text in texts {
        if seen.contains(&text) {
            continue;
        }
        seen.push(text.clone());
        places.push(path_of_string(py, &text)?);
    }
    for place in places {
        let place = place.bind(py);
        let key = runner_key(py, place)?;
        let by_path = runner_get(runner, "by_path")?;
        let cached = by_path.get_item(&key);
        if let Ok(cached) = cached {
            if !cached.is_none() {
                return Ok(Some(runner_bind(py, runner, cached.unbind(), place, name)?));
            }
        }
        let (source, problem) = runner_source(py, runner, place)?;
        if let Some(problem) = problem {
            return Ok(Some(program_err(py, &problem)?));
        }
        let Some(source) = source else { continue };
        let place_name = path_name(py, place)?;
        if place_name.ends_with(".viba.md") {
            let (sections, problem) = runner_sections(py, runner, place, Some(&source))?;
            if let Some(problem) = problem {
                return Ok(Some(program_err(py, &problem)?));
            }
            if let Some(sections) = sections {
                if !sections
                    .bind(py)
                    .call_method1("__contains__", (&generic_file,))?
                    .extract::<bool>()?
                {
                    return Ok(None);
                }
            }
        }
        let place_text = path_text(py, place)?;
        let base_text = path_text(py, base)?;
        return Ok(Some(runner_file_of(
            py,
            runner,
            place,
            name,
            &source,
            Some(&place_text),
            Some(&base_text),
            None,
        )?));
    }
    Ok(None)
}

fn runner_imported(
    py: Python<'_>,
    runner: &Bound<'_, PyAny>,
    name: &str,
    near: Option<&str>,
) -> PyResult<P> {
    let by_name = runner_get(runner, "by_name")?;
    if by_name
        .call_method1("__contains__", (name,))?
        .extract::<bool>()?
    {
        return make_ok(py, by_name.get_item(name)?.unbind());
    }
    for base in runner_bases(py, runner, near)? {
        if let Some(found) = runner_in_base(py, runner, base.bind(py), name)? {
            return Ok(found);
        }
    }
    let where_ = if near.is_some() {
        format!("next to {} and on VIBA_PATH", near.unwrap())
    } else {
        "on VIBA_PATH".to_string()
    };
    program_err(py, &format!("module '{name}' not found ({where_})"))
}

fn runner_run_file(
    py: Python<'_>,
    runner: &Bound<'_, PyAny>,
    file: &str,
    environ: &Bound<'_, PyAny>,
) -> PyResult<P> {
    let path = path_of_string(py, file)?;
    let (source, problem) = runner_source(py, runner, path.bind(py))?;
    if let Some(problem) = problem {
        return program_err(py, &problem);
    }
    let Some(source) = source else {
        return program_err(py, &format!("no such file: {file}"));
    };
    let stem = g(path.bind(py), "stem")?.extract::<String>()?;
    let file_path = path_text(py, path.bind(py))?;
    let module = runner_file_of(
        py,
        runner,
        path.bind(py),
        &stem,
        &source,
        Some(&file_path),
        None,
        None,
    )?;
    if is_stopped(py, module.bind(py))? {
        return Ok(module);
    }
    let file_value = py_string(py, &file_path).into_bound(py);
    run_module(
        py,
        runner,
        &ok_value(module.bind(py))?,
        environ,
        &stem,
        Some(&file_value),
        None,
        None,
        &pytuple(py, Vec::new())?.into_bound(py),
        false,
        0,
        false,
        false,
    )
}

fn runner_run_code(
    py: Python<'_>,
    runner: &Bound<'_, PyAny>,
    viba_code: &str,
    environ: &Bound<'_, PyAny>,
) -> PyResult<P> {
    let path = path_of_string(py, CODE_LABEL)?;
    let module = runner_module_of(
        py,
        runner,
        path.bind(py),
        CODE_LABEL,
        viba_code,
        Some(""),
        None,
        None,
    )?;
    if is_stopped(py, module.bind(py))? {
        return Ok(module);
    }
    run_module(
        py,
        runner,
        &ok_value(module.bind(py))?,
        environ,
        "",
        None,
        None,
        None,
        &pytuple(py, Vec::new())?.into_bound(py),
        false,
        0,
        false,
        false,
    )
}

fn runner_run_node(
    py: Python<'_>,
    runner: &Bound<'_, PyAny>,
    node: &Bound<'_, PyAny>,
    environ: &Bound<'_, PyAny>,
) -> PyResult<P> {
    let path = path_of_string(py, CODE_LABEL)?;
    let module = runner_node_of(py, runner, node, path.bind(py))?;
    if is_stopped(py, module.bind(py))? {
        return Ok(module);
    }
    run_module(
        py,
        runner,
        &ok_value(module.bind(py))?,
        environ,
        "",
        None,
        None,
        None,
        &pytuple(py, Vec::new())?.into_bound(py),
        false,
        0,
        true,
        false,
    )
}

// ----------------------------------------------------------------------
// The module as a function, and the calls it makes
// ----------------------------------------------------------------------

fn file_near(py: Python<'_>, file: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
    if file.is_none() {
        return Ok(None);
    }
    let text: String = file.str()?.to_string();
    if text.is_empty() {
        return Ok(None);
    }
    Ok(Some(text))
}

fn definition(py: Python<'_>, module: &Bound<'_, PyAny>, name: &str) -> PyResult<Option<P>> {
    let tree = g(module, "module")?;
    let body = g(&tree, "body")?;
    let mut found: Option<P> = None;
    for node in body.try_iter()? {
        let node = node?;
        if let Ok(node_name) = g(&node, "name") {
            if let Ok(node_name) = node_name.extract::<String>() {
                if node_name == name {
                    found = Some(node.unbind());
                }
            }
        }
    }
    Ok(found)
}

fn module_params<'py>(
    py: Python<'py>,
    module: &Bound<'py, PyAny>,
) -> PyResult<(Option<Vec<Bound<'py, PyAny>>>, Option<String>)> {
    let Some(found) = definition(py, module, DEF_NAME)? else {
        return Ok((None, None));
    };
    let body = g(found.bind(py), "body")?;
    if !is_ast(py, &body, "Exponent")? && !is_ast(py, &body, "ExponentChain")? {
        return Ok((
            None,
            Some(format!(
                "{DEF_NAME} is not a function type: {}",
                unparse_type(py, &body)?
            )),
        ));
    }
    let params = parameters_of(py, &body)?;
    let envs: Vec<&Bound<'_, PyAny>> = params
        .iter()
        .filter(|one| {
            one.get_item(2)
                .and_then(|env| env.extract::<bool>())
                .unwrap_or(false)
        })
        .collect();
    if envs.len() > 1 {
        return Ok((
            None,
            Some(format!(
                "{DEF_NAME} has {} {ENVIRON_TAG} {ENV_TYPE} parameters: a module that runs takes exactly one",
                envs.len()
            )),
        ));
    }
    for env in envs {
        let source = env.get_item(1)?;
        if !names_the_environment(py, &source)? {
            return Ok((
                None,
                Some(format!(
                    "the {ENVIRON_TAG} parameter of {DEF_NAME} must be {ENV_TYPE}, not {}",
                    unparse_type(py, &source)?
                )),
            ));
        }
    }
    Ok((Some(params), None))
}

fn module_arg_slots(
    py: Python<'_>,
    module: &Bound<'_, PyAny>,
) -> PyResult<(Option<Vec<(Option<String>, P)>>, Option<String>)> {
    let (params, problem) = module_params(py, module)?;
    if let Some(problem) = problem {
        return Ok((None, Some(problem)));
    }
    let Some(params) = params else {
        if definition(py, module, RET_NAME)?.is_none() {
            return Ok((Some(Vec::new()), None));
        }
        return Ok((
            None,
            Some(format!(
                "module has no {DEF_NAME}: a module that runs declares its parameters there"
            )),
        ));
    };
    let mut slots = Vec::new();
    for param in params {
        let is_env: bool = param.get_item(2)?.extract()?;
        if is_env {
            continue;
        }
        let tag = param.get_item(0)?;
        let tag = if tag.is_none() {
            None
        } else {
            Some(tag.extract::<String>()?)
        };
        slots.push((tag, param.get_item(1)?.unbind()));
    }
    Ok((Some(slots), None))
}

fn viba_data_factors<'py>(
    py: Python<'py>,
    node: &Bound<'py, PyAny>,
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let data = g(node, "data")?;
    if is_ast(py, &data, "Product")? || is_ast(py, &data, "ProductChain")? {
        return product_elements(py, &data);
    }
    Ok(vec![data])
}

fn call_signature(py: Python<'_>, members: &Bound<'_, PyAny>) -> PyResult<P> {
    if members.is_none() {
        return pytuple(py, Vec::new());
    }
    let mut out = Vec::new();
    for member in members.try_iter()? {
        let member = member?;
        let value = member.get_item(1)?;
        if is_environ_value(py, &value)? {
            continue;
        }
        out.push(signature_of(py, &value)?);
    }
    pytuple(py, out)
}

fn signature_of(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<P> {
    if is_shim(py, value, "_VibaData")? {
        let data = g(&g(value, "node")?, "data")?;
        let text: String = ast_mod(py)?
            .getattr("unparse_type")?
            .call1((data,))?
            .extract()?;
        return Ok(py_string(py, &text));
    }
    Ok(py_string(py, &class_name(py, value)))
}

fn stack_of(py: Python<'_>, runner: &Bound<'_, PyAny>) -> PyResult<P> {
    let running = g(runner, "running")?;
    let entries: Vec<Bound<'_, PyAny>> = running.try_iter()?.collect::<PyResult<Vec<_>>>()?;
    let mut frames: Vec<P> = Vec::new();
    for (index, entry) in entries.iter().enumerate() {
        let at: Option<i64> = entry.get_item(4)?.extract().unwrap_or(None);
        if index == 0 {
            let file: Option<String> = entry.get_item(3)?.extract().unwrap_or(None);
            frames.push(make_frame(py, file.as_deref().unwrap_or(""), 0)?);
        } else {
            let file: Option<String> = entries[index - 1].get_item(3)?.extract().unwrap_or(None);
            frames.push(make_frame(
                py,
                file.as_deref().unwrap_or(""),
                at.unwrap_or(0),
            )?);
        }
    }
    pylist(py, frames)
}

fn stopped_result(py: Python<'_>, runner: &Bound<'_, PyAny>, result: P) -> PyResult<P> {
    if is_type(py, result.bind(py), "VibaProgramErr")? {
        let stack = g(result.bind(py), "stack")?;
        let empty = stack.call_method0("__len__")?.extract::<i64>()? == 0;
        if empty {
            result.bind(py).setattr("stack", stack_of(py, runner)?)?;
        }
    }
    Ok(result)
}

fn new_activation(
    py: Python<'_>,
    runner: &Bound<'_, PyAny>,
    module: &Bound<'_, PyAny>,
    environ: &Bound<'_, PyAny>,
    name: &str,
    file: &Bound<'_, PyAny>,
    args: Option<&Bound<'_, PyAny>>,
    members: Option<&Bound<'_, PyAny>>,
    bindings: Option<&Bound<'_, PyAny>>,
) -> PyResult<P> {
    Ok(shim_class(py, "_Activation")?
        .call1((
            runner,
            module,
            environ,
            name,
            file,
            args.map(|value| value.clone().unbind())
                .unwrap_or_else(|| py.None()),
            members
                .map(|value| value.clone().unbind())
                .unwrap_or_else(|| py.None()),
            bindings
                .map(|value| value.clone().unbind())
                .unwrap_or_else(|| py.None()),
        ))?
        .unbind())
}

fn taking_a_design(py: Python<'_>, take: impl FnOnce() -> PyResult<P>) -> PyResult<P> {
    match take() {
        Ok(value) => Ok(value),
        Err(error) => {
            let refused = type_class(py, "PartialError")?;
            if error.value(py).as_any().is_instance(&refused)? {
                let text = error
                    .value(py)
                    .str()
                    .map(|text| text.to_string())
                    .unwrap_or_default();
                program_err(py, &text)
            } else {
                Err(error)
            }
        }
    }
}

fn argument_value(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<P> {
    if is_shim(py, value, "_VibaData")? {
        let node = g(value, "node")?;
        let data_module = g(&node, "data_module")?;
        let source_module = g(value, "source_module")?;
        if data_module.is_none() && !source_module.is_none() {
            let rebuilt = reflect_mod(py)?.getattr("VObject")?.call1((
                g(&node, "_access")?,
                g(&node, "descriptor")?,
                g(&node, "data")?,
            ))?;
            rebuilt.setattr("path", g(&node, "path")?)?;
            rebuilt.setattr("data_module", source_module)?;
            rebuilt.setattr("function", g(&node, "function")?)?;
            return Ok(rebuilt.unbind());
        }
        return Ok(node.unbind());
    }
    if is_shim(py, value, "_HostFunction")? {
        return Ok(value.call_method0("as_callable")?.unbind());
    }
    if is_shim(py, value, "_Host")? {
        return Ok(g(value, "obj")?.unbind());
    }
    Ok(value.clone().unbind())
}

#[allow(clippy::too_many_arguments)]
fn run_module_value(
    py: Python<'_>,
    runner: &Bound<'_, PyAny>,
    module: &Bound<'_, PyAny>,
    environ: &Bound<'_, PyAny>,
    name: &str,
    file: &Bound<'_, PyAny>,
    args: Option<&Bound<'_, PyAny>>,
    members: Option<&Bound<'_, PyAny>>,
    bindings: &Bound<'_, PyAny>,
    passes_environ: bool,
    give_environ: bool,
    ret: &Bound<'_, PyAny>,
    path: &str,
) -> PyResult<P> {
    let activation = new_activation(
        py,
        runner,
        module,
        environ,
        name,
        file,
        args,
        members,
        Some(bindings),
    )?;
    let body = g(ret, "body")?;
    let activation_bound = activation.bind(py).clone();
    let mut value = stopped_result(
        py,
        runner,
        taking_a_design(py, || {
            act_evaluate(py, &activation_bound, &body, &[], false)
        })?,
    )?;
    {
        let value_ref = value.bind(py);
        if !is_stopped(py, value_ref)? {
            let answered = ok_value(value_ref)?;
            if is_shim(py, &answered, "_Pending")? || is_shim(py, &answered, "_DynCall")? {
                if is_shim(py, &answered, "_DynCall")? {
                    let handed = if give_environ {
                        Some(environ.clone().unbind())
                    } else {
                        None
                    };
                    value = stopped_result(
                        py,
                        runner,
                        act_finish_dyn(
                            py,
                            &activation_bound,
                            &answered,
                            handed.as_ref().map(|handed| handed.bind(py)),
                            Vec::new(),
                        )?,
                    )?;
                } else {
                    let pending = answered.clone();
                    if (passes_environ || give_environ) && pending_takes_environ(py, &pending)? {
                        let mut handed = environ.clone().unbind();
                        if gs(&pending, "kind")? == "module"
                            && pending_declares_environ(py, &pending)?
                        {
                            let source = gs(&pending, "source")?;
                            handed = sub_env(py, environ, &py_string(py, &source).into_bound(py))?;
                        }
                        let host = shim_class(py, "_Host")?.call1((handed,))?.unbind();
                        let given = pending_give(py, &pending, Some(ENVIRON_TAG), host.bind(py))?;
                        if is_stopped(py, given.bind(py))? {
                            return Ok(given);
                        }
                        let pending = ok_value(given.bind(py))?.unbind();
                        value = stopped_result(py, runner, act_finish(py, pending.bind(py))?)?;
                    } else {
                        value = stopped_result(py, runner, act_finish(py, &pending)?)?;
                    }
                }
            }
        }
    }
    {
        let value_ref = value.bind(py);
        if !is_stopped(py, value_ref)? {
            let answered = ok_value(value_ref)?;
            if is_shim(py, &answered, "_HostFunction")?
                || !(is_shim(py, &answered, "_VibaData")? || is_shim(py, &answered, "_Host")?)
            {
                if is_shim(py, &answered, "_Pending")? {}
                return stopped_result(
                    py,
                    runner,
                    program_err(
                        py,
                        &format!("{name}.{RET_NAME} is a function still waiting for arguments"),
                    )?,
                );
            }
        }
    }
    if is_stopped(py, value.bind(py))? {
        if !is_type(py, value.bind(py), "Err")? {}
        return Ok(value);
    }
    let answer = argument_value(py, &ok_value(value.bind(py))?)?;
    let done = g(runner, "done")?;
    done.set_item(
        path,
        PyTuple::new(py, [py_string(py, name), answer.clone_ref(py)])?,
    )?;
    make_ok(py, answer)
}

#[allow(clippy::too_many_arguments)]
fn run_module_impl(
    py: Python<'_>,
    runner: &Bound<'_, PyAny>,
    module: &Bound<'_, PyAny>,
    environ: &Bound<'_, PyAny>,
    name: &str,
    file: Option<&Bound<'_, PyAny>>,
    args: Option<&Bound<'_, PyAny>>,
    members: Option<&Bound<'_, PyAny>>,
    bindings: &Bound<'_, PyAny>,
    passes_environ: bool,
    at: i64,
    give_environ: bool,
    driving: bool,
) -> PyResult<P> {
    let file_owned: Option<P> = match file {
        Some(file) => Some(file.clone().unbind()),
        None => match g(runner, "path_of")?.call_method1("get", (name,)) {
            Ok(found) => {
                if found.is_none() {
                    None
                } else {
                    Some(found.unbind())
                }
            }
            Err(_) => None,
        },
    };
    let file: Bound<'_, PyAny> = match &file_owned {
        Some(file) => file.bind(py).clone(),
        None => py.None().into_bound(py),
    };
    let Some(ret) = definition(py, module, RET_NAME)? else {
        return stopped_result(
            py,
            runner,
            program_err(
                py,
                &format!("module '{name}' has no {RET_NAME}: it is design, not a program"),
            )?,
        );
    };
    if let (_, Some(problem)) = module_params(py, module)? {
        return stopped_result(
            py,
            runner,
            program_err(py, &format!("module '{name}': {problem}"))?,
        );
    }
    let members_owned: P = match members {
        Some(members) => members.clone().unbind(),
        None => {
            let host = shim_class(py, "_Host")?.call1((environ,))?.unbind();
            PyTuple::new(py, [PyTuple::new(py, [py_string(py, ENVIRON_TAG), host])?])?
                .into_any()
                .unbind()
        }
    };
    let args_owned: P = match args {
        Some(args) => args.clone().unbind(),
        None => {
            let nil = viba_data(py, &py.None().into_bound(py))?;
            make_viba_data(py, nil)?
        }
    };
    let bindings: Bound<'_, PyAny> = if bindings.is_none() {
        pytuple(py, Vec::new())?.into_bound(py)
    } else {
        bindings.clone()
    };
    let path = storage_path(py, environ)?;
    let signature = call_signature(py, &members_owned.bind(py).clone())?;
    {
        let running = g(runner, "running")?;
        for seen in running.try_iter()? {
            let seen = seen?;
            let running_path: String = seen.get_item(0)?.extract()?;
            if running_path == path {
                return stopped_result(
                    py,
                    runner,
                    program_err(
                        py,
                        &format!(
                            "the storage path '{path}' is already running a call, so '{name}' \
                             cannot run there too: give each module call a sub-environment of its \
                             own (args.env.sub_env << args.env << ...)"
                        ),
                    )?,
                );
            }
            let running_name: String = seen.get_item(1)?.extract()?;
            let running_signature = seen.get_item(2)?;
            if running_name == name && running_signature.eq(&signature.bind(py).clone())? {
                return stopped_result(
                    py,
                    runner,
                    program_err(
                        py,
                        &format!(
                            "module '{name}' is already running with the same arguments: a \
                             module call cycle"
                        ),
                    )?,
                );
            }
        }
    }
    {
        let held = g(runner, "held_paths")?;
        if held
            .call_method1("__contains__", (&path,))?
            .extract::<bool>()?
        {
            return stopped_result(
                py,
                runner,
                program_err(
                    py,
                    &format!(
                        "the storage path '{path}' is already running a call, so '{name}' cannot \
                         run there too: give each module call a sub-environment of its own \
                         (args.env.sub_env << args.env << ...)"
                    ),
                )?,
            );
        }
    }
    {
        let done = g(runner, "done")?;
        if done
            .call_method1("__contains__", (&path,))?
            .extract::<bool>()?
        {
            let answered = done.get_item(&path)?;
            let answered_as: String = answered.get_item(0)?.extract()?;
            if answered_as == name {
                return make_ok_ref(py, &answered.get_item(1)?);
            }
            return stopped_result(
                py,
                runner,
                program_err(
                    py,
                    &format!(
                        "module '{name}' was handed the storage path '{path}', which another \
                         module call already used: give each module call a sub-environment of its \
                         own (args.env.sub_env << args.env << ...)"
                    ),
                )?,
            );
        }
    }
    let call_run = |py: Python<'_>| -> PyResult<P> {
        let resumable = shim_class(py, "_ResumableCall")?
            .call1((
                runner,
                module,
                environ,
                name,
                file.clone().unbind(),
                args_owned.clone_ref(py),
                members_owned.clone_ref(py),
                bindings.clone().unbind(),
                passes_environ,
                give_environ,
                at,
            ))?
            .unbind();
        let running = g(runner, "running")?;
        running.call_method1(
            "append",
            (PyTuple::new(
                py,
                [
                    py_string(py, &path),
                    py_string(py, name),
                    signature.clone_ref(py),
                    file.clone().unbind(),
                    at.into_pyobject(py)?.into_any().unbind(),
                ],
            )?,),
        )?;
        let running_calls = g(runner, "running_calls")?;
        running_calls.call_method1("append", (resumable.bind(py),))?;
        let outcome = run_module_value(
            py,
            runner,
            module,
            environ,
            name,
            &file,
            Some(&args_owned.bind(py).clone()),
            Some(&members_owned.bind(py).clone()),
            &bindings,
            passes_environ,
            give_environ,
            ret.bind(py),
            &path,
        );
        running.call_method0("pop")?;
        running_calls.call_method0("pop")?;
        outcome
    };
    let running_len = g(runner, "running")?
        .call_method0("__len__")?
        .extract::<i64>()?;
    if driving || running_len > 0 {
        return call_run(py);
    }
    let own = shim_class(py, "_ResumableCall")?
        .call1((
            runner,
            module,
            environ,
            name,
            file.clone().unbind(),
            args_owned,
            members_owned,
            bindings.unbind(),
            passes_environ,
            give_environ,
            at,
        ))?
        .unbind();
    let current_run = shim(py)?.getattr("_CURRENT_RUN")?;
    let token = current_run.call_method1("set", (runner,))?;
    let outcome = driven(py, runner, own.bind(py));
    current_run.call_method1("reset", (token,))?;
    outcome
}

fn hold_impl(py: Python<'_>, runner: &Bound<'_, PyAny>, tasks: &Bound<'_, PyAny>) -> PyResult<()> {
    let held = g(runner, "held_paths")?;
    let stack = g(runner, "global_resumable_stack")?;
    for task in tasks.try_iter()? {
        let task = task?;
        let path: String = g(&task, "current_env_path")?.extract()?;
        if held
            .call_method1("__contains__", (&path,))?
            .extract::<bool>()?
        {
            continue;
        }
        held.set_item(&path, task.clone())?;
        stack.call_method1("append", (task,))?;
    }
    Ok(())
}

fn driven(py: Python<'_>, runner: &Bound<'_, PyAny>, own: &Bound<'_, PyAny>) -> PyResult<P> {
    let stack = g(runner, "global_resumable_stack")?;
    let root = get_root(py, &g(own, "environ")?)?;
    let mut own = own.clone().unbind();
    let mut own_environ = g(own.bind(py), "environ")?.unbind();
    loop {
        while stack.call_method0("__len__")?.extract::<i64>()? > 0 {
            let task = stack.call_method0("pop")?.unbind();
            let task_ref = task.bind(py).clone();
            let path: String = g(&task_ref, "current_env_path")?.extract()?;
            g(runner, "held_paths")?.call_method1("pop", (&path, py.None()))?;
            let root_bound = root.as_ref().map(|root| root.bind(py).clone());
            let task_environ = g(&task_ref, "restored")?
                .call1((root_bound
                    .as_ref()
                    .map(|root| root.clone().unbind())
                    .unwrap_or_else(|| py.None()),))?
                .unbind();
            let chain_length = env_chain_length(py, task_environ.bind(py))?;
            runner.setattr("compacted_chain_length", chain_length)?;
            match g(&task_ref, "run")?.call1((task_environ.clone_ref(py),)) {
                Ok(result) => {
                    if is_stopped(py, &result)? {
                        return Ok(result.unbind());
                    }
                }
                Err(error) => {
                    if let Some(compact) = is_compact(py, &error)? {
                        let tasks = g(&compact, "tasks")?;
                        shim(py)?.getattr("_hold")?.call1((runner, tasks))?;
                        continue;
                    }
                    return Err(error);
                }
            }
        }
        match g(own.bind(py), "run")?.call1((own_environ.clone_ref(py),)) {
            Ok(result) => return Ok(result.unbind()),
            Err(error) => {
                if let Some(compact) = is_compact(py, &error)? {
                    let own_path: String = g(own.bind(py), "current_env_path")?.extract()?;
                    let mut rest: Vec<P> = Vec::new();
                    for task in g(&compact, "tasks")?.try_iter()? {
                        let task = task?;
                        let path: String = g(&task, "current_env_path")?.extract()?;
                        if path == own_path {
                            own = task.unbind();
                        } else {
                            rest.push(task.unbind());
                        }
                    }
                    let rest = pylist(py, rest)?;
                    shim(py)?.getattr("_hold")?.call1((runner, rest.bind(py)))?;
                    let root_bound = root.as_ref().map(|root| root.bind(py).clone());
                    own_environ = g(own.bind(py), "restored")?
                        .call1((root_bound
                            .as_ref()
                            .map(|root| root.clone().unbind())
                            .unwrap_or_else(|| py.None()),))?
                        .unbind();
                    let chain_length = env_chain_length(py, own_environ.bind(py))?;
                    runner.setattr("compacted_chain_length", chain_length)?;
                    continue;
                }
                return Err(error);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_module(
    py: Python<'_>,
    runner: &Bound<'_, PyAny>,
    module: &Bound<'_, PyAny>,
    environ: &Bound<'_, PyAny>,
    name: &str,
    file: Option<&Bound<'_, PyAny>>,
    args: Option<&Bound<'_, PyAny>>,
    members: Option<&Bound<'_, PyAny>>,
    bindings: &Bound<'_, PyAny>,
    passes_environ: bool,
    at: i64,
    give_environ: bool,
    driving: bool,
) -> PyResult<P> {
    let kwargs = PyDict::new(py);
    kwargs.set_item(
        "args",
        args.map(|value| value.clone().unbind())
            .unwrap_or_else(|| py.None()),
    )?;
    kwargs.set_item(
        "members",
        members
            .map(|value| value.clone().unbind())
            .unwrap_or_else(|| py.None()),
    )?;
    kwargs.set_item("bindings", bindings)?;
    kwargs.set_item("passes_environ", passes_environ)?;
    kwargs.set_item("at", at)?;
    kwargs.set_item("give_environ", give_environ)?;
    kwargs.set_item("driving", driving)?;
    let file_value = match file {
        Some(file) => file.clone().unbind(),
        None => py.None(),
    };
    Ok(shim(py)?
        .getattr("_run_module")?
        .call((runner, module, environ, name, file_value), Some(&kwargs))?
        .unbind())
}

// ----------------------------------------------------------------------
// One module running with one environment
// ----------------------------------------------------------------------

fn act_attr<'py>(act: &Bound<'py, PyAny>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    g(act, name)
}

fn act_bindings(py: Python<'_>, act: &Bound<'_, PyAny>) -> PyResult<Vec<P>> {
    let bindings = act_attr(act, "bindings")?;
    if bindings.call_method0("__len__")?.extract::<i64>()? == 0 {
        return Ok(Vec::new());
    }
    Ok(vec![bindings.unbind()])
}

fn scope_with_bindings(py: Python<'_>, act: &Bound<'_, PyAny>, scope: &[P]) -> PyResult<Vec<P>> {
    let mut frames = act_bindings(py, act)?;
    frames.extend(scope.iter().map(|frame| frame.clone_ref(py)));
    Ok(frames)
}

fn act_evaluate(
    py: Python<'_>,
    act: &Bound<'_, PyAny>,
    node: &Bound<'_, PyAny>,
    scope: &[P],
    finish: bool,
) -> PyResult<P> {
    let outcome = act_evaluate_inner(py, act, node, scope, finish);
    if let Err(error) = &outcome {
        let text = format_exc(py, error);
        if text.contains("not callable") {
            let node_text = ast_mod(py)
                .and_then(|module| module.getattr("dump"))
                .and_then(|dump| dump.call1((node,)))
                .and_then(|text| text.extract::<String>())
                .unwrap_or_default();
        }
    }
    outcome
}

fn act_evaluate_inner(
    py: Python<'_>,
    act: &Bound<'_, PyAny>,
    node: &Bound<'_, PyAny>,
    scope: &[P],
    finish: bool,
) -> PyResult<P> {
    let scope = scope_with_bindings(py, act, scope)?;
    if is_ast(py, node, "Constant")?
        || is_ast(py, node, "Nil")?
        || is_ast(py, node, "Never")?
        || is_ast(py, node, "Any")?
        || is_ast(py, node, "Tuple")?
        || is_ast(py, node, "Tagged")?
    {
        let node = substituted(py, node, &scope)?;
        let module = act_attr(act, "module")?;
        let descriptor = piece_descriptor(py, node.bind(py), &module)?;
        let object = make_vobject(py, descriptor.bind(py), node.bind(py), None, None)?;
        let data = make_viba_data(py, object)?;
        return make_ok(py, data);
    }
    if is_ast(py, node, "Product")? || is_ast(py, node, "ProductChain")? {
        return act_product(py, act, node, &scope);
    }
    if is_ast(py, node, "Sum")? || is_ast(py, node, "SumChain")? {
        return act_sum(py, act, node, &scope);
    }
    if is_ast(py, node, "Partial")? {
        let out = act_apply_chain(py, act, node, &scope, finish, None)?;
        return Ok(out);
    }
    if is_ast(py, node, "TypeApp")? {
        return act_generic_application(py, act, node, &scope);
    }
    if is_ast(py, node, "TypeRef")? {
        return act_resolve(py, act, &gs(node, "name")?, &scope);
    }
    if is_ast(py, node, "MemberTaken")? {
        if let Some(path) = source_path(py, node)? {
            return act_resolve(py, act, &path, &scope);
        }
        return act_application_member(py, act, node, &scope);
    }
    if is_ast(py, node, "HintBlock")? {
        return program_err(py, "a hint block is documentation: it is not a value");
    }
    program_err(py, &format!("cannot compute {}", class_name(py, node)))
}

fn substituted(py: Python<'_>, node: &Bound<'_, PyAny>, frames: &[P]) -> PyResult<P> {
    if frames.is_empty() {
        return Ok(node.clone().unbind());
    }
    let copied = module(py, "copy")?.call_method1("deepcopy", (node,))?;
    substitute_visit(py, &copied, frames)
}

fn substitute_visit(py: Python<'_>, node: &Bound<'_, PyAny>, frames: &[P]) -> PyResult<P> {
    if is_ast(py, node, "TypeRef")? {
        let name: String = gs(node, "name")?;
        if let Some(bound) = in_scope(py, frames, &name)? {
            if is_shim(py, bound.bind(py), "_VibaData")? {
                return Ok(g(&g(bound.bind(py), "node")?, "data")?.unbind());
            }
        }
        return Ok(node.clone().unbind());
    }
    for field in g(node, "_fields")?.try_iter()? {
        let field: String = field?.extract()?;
        let value = g(node, &field)?;
        if is_ast_any(py, &value)? {
            let replaced = substitute_visit(py, &value, frames)?;
            node.setattr(&field, replaced)?;
        } else if value.is_instance_of::<PyList>() {
            let mut items: Vec<P> = Vec::new();
            let mut changed = false;
            for item in value.try_iter()? {
                let item = item?;
                if is_ast_any(py, &item)? {
                    items.push(substitute_visit(py, &item, frames)?);
                    changed = true;
                } else {
                    items.push(item.unbind());
                }
            }
            if changed {
                node.setattr(&field, PyList::new(py, items)?)?;
            }
        }
    }
    Ok(node.clone().unbind())
}

fn act_descriptor(py: Python<'_>, act: &Bound<'_, PyAny>, node: &Bound<'_, PyAny>) -> PyResult<P> {
    let module = act_attr(act, "module")?;
    piece_descriptor(py, node, &module)
}

fn act_product(
    py: Python<'_>,
    act: &Bound<'_, PyAny>,
    node: &Bound<'_, PyAny>,
    scope: &[P],
) -> PyResult<P> {
    let nil_type = type_class(py, "NilType")?.unbind();
    let mut kept: Vec<Bound<'_, PyAny>> = Vec::new();
    for factor in product_elements(py, node)? {
        if let Some(unit) = builtin_unit(py, &factor)? {
            if unit.bind(py).is(nil_type.bind(py)) {
                continue;
            }
        }
        let value = act_evaluate(py, act, &factor, scope, true)?;
        if is_stopped(py, value.bind(py))? {
            return Ok(value);
        }
        let answered = ok_value(value.bind(py))?;
        if is_never(py, &answered)? {
            return Ok(make_ok(py, never_viba_data(py)?)?);
        }
        if is_nil(py, &answered)? {
            continue;
        }
        kept.extend(act_inlined_members(py, act, &answered)?);
    }
    if kept.is_empty() {
        let nil = viba_data(py, &py.None().into_bound(py))?;
        return make_ok(py, make_viba_data(py, nil)?);
    }
    if kept.len() == 1 {
        return make_ok(py, kept[0].clone().unbind());
    }
    let mut factors = Vec::new();
    for one in &kept {
        factors.push(g(&g(one, "node")?, "data")?.unbind());
    }
    let chain = ast_class(py, "ProductChain")?
        .call1((PyList::new(py, factors)?,))?
        .unbind();
    let mut pairs: Vec<(Option<String>, P)> = Vec::new();
    for one in &kept {
        for pair in the_members(py, one)? {
            pairs.push(pair);
        }
    }
    let mut members = Vec::new();
    for (_, member) in &pairs {
        members.push(member.clone_ref(py));
    }
    let module = act_attr(act, "module")?;
    let descriptor = descriptor_of_values(py, chain.bind(py), &module, members)?;
    let object = make_vobject(py, descriptor.bind(py), chain.bind(py), None, None)?;
    let data = make_viba_data(py, object)?;
    data.bind(py).setattr(
        "members",
        PyList::new(
            py,
            pairs
                .into_iter()
                .map(|(tag, member)| {
                    Ok::<P, PyErr>(
                        PyTuple::new(
                            py,
                            [
                                match tag {
                                    Some(tag) => py_string(py, &tag),
                                    None => py.None(),
                                },
                                member,
                            ],
                        )?
                        .into_any()
                        .unbind(),
                    )
                })
                .collect::<PyResult<Vec<_>>>()?,
        )?,
    )?;
    make_ok(py, data)
}

fn act_inlined_members<'py>(
    py: Python<'py>,
    act: &Bound<'py, PyAny>,
    piece: &Bound<'py, PyAny>,
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let data = g(&g(piece, "node")?, "data")?;
    if !is_ast(py, &data, "Product")? && !is_ast(py, &data, "ProductChain")? {
        return Ok(vec![piece.clone()]);
    }
    let piece_module = g(piece, "source_module")?;
    let source_module = if piece_module.is_none() {
        act_attr(act, "module")?
    } else {
        piece_module.clone()
    };
    let bindings = g(piece, "bindings")?;
    let mut out = Vec::new();
    for member in product_elements(py, &data)? {
        let descriptor = piece_descriptor(py, &member, &source_module)?;
        let object = make_vobject(py, descriptor.bind(py), &member, None, None)?;
        let value = make_viba_data(py, object)?;
        let value = value.bind(py).clone();
        if !piece_module.is_none() {
            value.setattr("source_module", &piece_module)?;
        }
        value.setattr("bindings", &bindings)?;
        out.push(value);
    }
    Ok(out)
}

fn piece_module_is_none(py: Python<'_>, module: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(module.is_none())
}

fn act_sum(
    py: Python<'_>,
    act: &Bound<'_, PyAny>,
    node: &Bound<'_, PyAny>,
    scope: &[P],
) -> PyResult<P> {
    let never_type = type_class(py, "NeverType")?.unbind();
    let mut kept: Vec<Bound<'_, PyAny>> = Vec::new();
    for branch in sum_branches(py, node)? {
        let branch = branch.bind(py);
        if let Some(unit) = builtin_unit(py, branch)? {
            if unit.bind(py).is(never_type.bind(py)) {
                continue;
            }
        }
        let value = act_evaluate(py, act, branch, scope, true)?;
        if is_stopped(py, value.bind(py))? {
            return Ok(value);
        }
        let answered = ok_value(value.bind(py))?;
        if is_never(py, &answered)? {
            continue;
        }
        kept.push(answered);
    }
    if kept.is_empty() {
        return make_ok(py, never_viba_data(py)?);
    }
    if kept.len() == 1 {
        return make_ok(py, kept[0].clone().unbind());
    }
    let mut elements = Vec::new();
    for branch in &kept {
        elements.push(g(&g(branch, "node")?, "data")?.unbind());
    }
    let chain = ast_class(py, "SumChain")?
        .call1((PyList::new(py, elements)?,))?
        .unbind();
    let descriptor = act_descriptor(py, act, chain.bind(py))?;
    make_ok(
        py,
        make_viba_data(
            py,
            make_vobject(py, descriptor.bind(py), chain.bind(py), None, None)?,
        )?,
    )
}

fn act_imports(py: Python<'_>, act: &Bound<'_, PyAny>) -> PyResult<P> {
    let module = act_attr(act, "module")?;
    let tree = g(&module, "module")?;
    let imports = PyDict::new(py);
    for statement in g(&tree, "body")?.try_iter()? {
        let statement = statement?;
        if !is_ast(py, &statement, "Import")? {
            continue;
        }
        let module_name: String = g(&statement, "module")?.extract()?;
        let alias = g(&statement, "alias")?;
        let bound = if alias.is_none() {
            module_name.clone()
        } else {
            alias.extract::<String>()?
        };
        imports.set_item(bound, module_name)?;
    }
    Ok(imports.into_any().unbind())
}

fn act_resolve(py: Python<'_>, act: &Bound<'_, PyAny>, name: &str, scope: &[P]) -> PyResult<P> {
    if let Some(bound) = in_scope(py, scope, name)? {
        return make_ok(py, bound);
    }
    if name == GET_ARGS_NAME {
        let made = shim_class(py, "_GetArgs")?.call1((act,))?.unbind();
        return make_ok(py, made);
    }
    if name == IS_SUB_TYPE_VALUE_NAME {
        let made = shim_class(py, "_IsSubType")?.call1((py.None(),))?.unbind();
        return make_ok(py, made);
    }
    if name == DEF_NAME {
        let module = act_attr(act, "module")?;
        if let Some(definition) = definition(py, &module, DEF_NAME)? {
            let node = g(definition.bind(py), "body")?;
            let descriptor = piece_descriptor(py, &node, &module)?;
            let object = make_vobject(py, descriptor.bind(py), &node, None, None)?;
            return make_ok(py, make_viba_data(py, object)?);
        }
    }
    let module = act_attr(act, "module")?;
    if let Some(definition) = definition(py, &module, name)? {
        return act_value_of_definition(py, act, name, definition.bind(py), &module, None, None);
    }
    if let Some((module_name, rest)) = act_imported_name(py, act, name)? {
        let runner = act_attr(act, "runner")?;
        let file = act_attr(act, "file")?;
        let near = file_near(py, &file)?;
        let imported = runner_imported(py, &runner, &module_name, near.as_deref())?;
        if is_stopped(py, imported.bind(py))? {
            return Ok(imported);
        }
        if !rest.is_empty() {
            return act_member_of(
                py,
                act,
                &ok_value(imported.bind(py))?,
                &module_name,
                &rest,
                name,
            );
        }
        let node = make_type_ref(py, name)?;
        let descriptor = piece_descriptor(py, node.bind(py), &module)?;
        let object = make_vobject(py, descriptor.bind(py), node.bind(py), None, None)?;
        return make_ok(py, make_viba_data(py, object)?);
    }
    if let Some(module_name) = builtin_directory_name(py, name)? {
        let runner = act_attr(act, "runner")?;
        let found = runner_builtin_module(py, &runner, &module_name)?;
        if let Some(found) = found {
            if is_stopped(py, found.bind(py))? {
                return Ok(found);
            }
            let node = make_type_ref(py, name)?;
            let descriptor = piece_descriptor(py, node.bind(py), &module)?;
            let object = make_vobject(py, descriptor.bind(py), node.bind(py), None, None)?;
            return make_ok(py, make_viba_data(py, object)?);
        }
    }
    if let Some(member) = act_tagged_member(py, act, name, scope)? {
        return Ok(member);
    }
    if let Some(member) = act_builtin_member(py, name)? {
        return Ok(member);
    }
    if let Some(member) = act_builtin_bare_member(py, name)? {
        return Ok(member);
    }
    if let Some(member) = act_environment_member(py, act, name, scope)? {
        return Ok(member);
    }
    let act_name: String = act_attr(act, "name")?.extract()?;
    program_err(
        py,
        &format!("no definition named '{name}' in module '{act_name}'"),
    )
}

fn act_tagged_member(
    py: Python<'_>,
    act: &Bound<'_, PyAny>,
    name: &str,
    scope: &[P],
) -> PyResult<Option<P>> {
    let Some((head, tag)) = name.rsplit_once('.') else {
        return Ok(None);
    };
    if head.is_empty() {
        return Ok(None);
    }
    let value = act_resolve(py, act, head, scope)?;
    if is_stopped(py, value.bind(py))? || !is_shim(py, &ok_value(value.bind(py))?, "_VibaData")? {
        return Ok(None);
    }
    let value = ok_value(value.bind(py))?;
    let wanted = format!("${tag}");
    let args = act_attr(act, "args")?;
    if value.is(&args) {
        if let Some(given) = act_as_given(py, act, &wanted)? {
            return Ok(Some(make_ok(py, given)?));
        }
        if wanted == ENVIRON_TAG && !act_declares_environ(py, act)? {
            let act_name: String = act_attr(act, "name")?.extract()?;
            return Ok(Some(program_err(
                py,
                &format!(
                    "module '{act_name}' declares no {ENVIRON_TAG} {ENV_TYPE} parameter: its \
                     body cannot use args.env"
                ),
            )?));
        }
    }
    for factor in viba_data_factors(py, &g(&value, "node")?)? {
        if is_ast(py, &factor, "Tagged")? && gs(&factor, "tag")? == wanted {
            if let Some(member) = member_under_tag(py, &value, &wanted)? {
                return Ok(Some(make_ok(py, member)?));
            }
        }
    }
    Ok(Some(program_err(
        py,
        &format!("'{head}' has no member tagged '{wanted}'"),
    )?))
}

fn act_imported_name(
    py: Python<'_>,
    act: &Bound<'_, PyAny>,
    name: &str,
) -> PyResult<Option<(String, String)>> {
    let imports = act_imports(py, act)?;
    let imports = imports.bind(py);
    if imports
        .call_method1("__contains__", (name,))?
        .extract::<bool>()?
    {
        return Ok(Some((imports.get_item(name)?.extract()?, String::new())));
    }
    let mut keys: Vec<String> = imports
        .call_method0("keys")?
        .try_iter()?
        .map(|key| key?.extract::<String>())
        .collect::<PyResult<Vec<_>>>()?;
    keys.sort_by_key(|key| std::cmp::Reverse(key.len()));
    for prefix in keys {
        let with_dot = format!("{prefix}.");
        if name.starts_with(&with_dot) {
            return Ok(Some((
                imports.get_item(&prefix)?.extract()?,
                name[with_dot.len()..].to_string(),
            )));
        }
    }
    Ok(None)
}

fn act_defined(
    py: Python<'_>,
    act: &Bound<'_, PyAny>,
    name: &str,
    definition: &Bound<'_, PyAny>,
) -> PyResult<P> {
    let defined = act_attr(act, "defined")?;
    if defined
        .call_method1("__contains__", (name,))?
        .extract::<bool>()?
    {
        return Ok(defined.get_item(name)?.unbind());
    }
    let act_name: String = act_attr(act, "name")?.extract()?;
    let signature = act_attr(act, "signature")?;
    let mine = PyTuple::new(
        py,
        [
            py_string(py, &act_name),
            py_string(py, name),
            signature.unbind(),
        ],
    )?;
    let runner = act_attr(act, "runner")?;
    let computing = g(&runner, "computing")?;
    for seen in computing.try_iter()? {
        let seen = seen?;
        if seen.eq(&mine)? {
            let message = act_goes_round(py, act, name)?;
            return program_err(py, &message);
        }
    }
    computing.call_method1("append", (&mine,))?;
    let value = act_compute(py, act, name, definition);
    computing.call_method0("pop")?;
    let value = value?;
    defined.set_item(name, value.bind(py))?;
    Ok(value)
}

fn act_goes_round(py: Python<'_>, act: &Bound<'_, PyAny>, name: &str) -> PyResult<String> {
    let act_name: String = act_attr(act, "name")?.extract()?;
    let signature = act_attr(act, "signature")?;
    let mine = PyTuple::new(
        py,
        [
            py_string(py, &act_name),
            py_string(py, name),
            signature.clone().unbind(),
        ],
    )?;
    let runner = act_attr(act, "runner")?;
    let computing = g(&runner, "computing")?;
    let entries: Vec<Bound<'_, PyAny>> = computing.try_iter()?.collect::<PyResult<Vec<_>>>()?;
    let mut start = 0;
    for (index, entry) in entries.iter().enumerate() {
        if entry.eq(&mine)? {
            start = index;
            break;
        }
    }
    let mut path: Vec<Bound<'_, PyAny>> = entries[start..].to_vec();
    path.push(mine.into_any());
    let mut modules = Vec::new();
    for entry in &path {
        modules.push(entry.get_item(0)?.extract::<String>()?);
    }
    let unique: std::collections::HashSet<&String> = modules.iter().collect();
    if unique.len() == 1 {
        let defined: Vec<String> = path
            .iter()
            .map(|entry| entry.get_item(1)?.extract::<String>())
            .collect::<PyResult<Vec<_>>>()?;
        return Ok(format!(
            "{}: one file's definitions may not go round — recursion takes two files",
            defined.join(" -> ")
        ));
    }
    let mut described = Vec::new();
    for entry in &path {
        let module: String = entry.get_item(0)?.extract()?;
        let defined: String = entry.get_item(1)?.extract()?;
        let short = defined.rsplit('.').next().unwrap_or(&defined);
        described.push(format!("{module}.{short}"));
    }
    Ok(format!(
        "{}: the run came back to where it started",
        described.join(" -> ")
    ))
}

fn act_compute(
    py: Python<'_>,
    act: &Bound<'_, PyAny>,
    _name: &str,
    definition: &Bound<'_, PyAny>,
) -> PyResult<P> {
    let body = g(definition, "body")?;
    act_evaluate(py, act, &body, &[], true)
}

fn act_member_of(
    py: Python<'_>,
    act: &Bound<'_, PyAny>,
    module: &Bound<'_, PyAny>,
    module_name: &str,
    rest: &str,
    source: &str,
) -> PyResult<P> {
    let Some(definition) = definition(py, module, rest)? else {
        return program_err(py, &format!("module '{module_name}' has no '{rest}'"));
    };
    let runner = act_attr(act, "runner")?;
    let environ = act_attr(act, "environ")?;
    let file = g(&runner, "path_of")?
        .call_method1("get", (module_name,))?
        .unbind();
    let other = new_activation(
        py,
        &runner,
        module,
        &environ,
        module_name,
        file.bind(py),
        None,
        None,
        None,
    )?;
    let current_module = act_attr(act, "module")?;
    let spelled = if source.is_empty() { rest } else { source };
    let spelled = make_type_ref(py, spelled)?;
    act_value_of_definition(
        py,
        other.bind(py),
        rest,
        definition.bind(py),
        module,
        Some(&current_module),
        Some(spelled.bind(py)),
    )
}

fn act_value_of_definition(
    py: Python<'_>,
    act: &Bound<'_, PyAny>,
    name: &str,
    definition: &Bound<'_, PyAny>,
    owner_module: &Bound<'_, PyAny>,
    source_module: Option<&Bound<'_, PyAny>>,
    source: Option<&Bound<'_, PyAny>>,
) -> PyResult<P> {
    let current_module = act_attr(act, "module")?;
    let source_module = match source_module {
        Some(module) => module.clone(),
        None => current_module,
    };
    let body = g(definition, "body")?;
    if is_ast(py, &body, "Exponent")? || is_ast(py, &body, "ExponentChain")? {
        let node: P = match source {
            Some(source) if is_ast_any(py, source)? => source.clone().unbind(),
            Some(source) => source.clone().unbind(),
            None => make_type_ref(py, name)?,
        };
        let descriptor = piece_descriptor(py, node.bind(py), &source_module)?;
        let definition_name: String = g(definition, "name")?.extract()?;
        let function = reflect_mod(py)?
            .getattr("VibaFunction")?
            .call1((definition_name, owner_module))?
            .unbind();
        let object = make_vobject(
            py,
            descriptor.bind(py),
            node.bind(py),
            None,
            Some(function.bind(py)),
        )?;
        let data = make_viba_data(py, object)?;
        data.bind(py).setattr("source_module", source_module)?;
        return make_ok(py, data);
    }
    act_defined(py, act, name, definition)
}

fn act_declares_environ(py: Python<'_>, act: &Bound<'_, PyAny>) -> PyResult<bool> {
    let module = act_attr(act, "module")?;
    let (params, _problem) = module_params(py, &module)?;
    for param in params.unwrap_or_default() {
        if param.get_item(2)?.extract::<bool>()? {
            return Ok(true);
        }
    }
    Ok(false)
}

fn act_as_given(py: Python<'_>, act: &Bound<'_, PyAny>, tag: &str) -> PyResult<Option<P>> {
    let members = act_attr(act, "members")?;
    for member in members.try_iter()? {
        let member = member?;
        let member_tag = member.get_item(0)?;
        if !member_tag.is_none() && member_tag.extract::<String>()? == tag {
            let value = member.get_item(1)?;
            return Ok(Some(value.unbind()));
        }
    }
    Ok(None)
}

fn act_application_member(
    py: Python<'_>,
    act: &Bound<'_, PyAny>,
    node: &Bound<'_, PyAny>,
    scope: &[P],
) -> PyResult<P> {
    let owner = g(node, "owner")?;
    let (source, modules, parts) = act_application_piece(py, act, &owner, scope)?;
    let module = act_attr(act, "module")?;
    let decision = reduce_application(
        py,
        source.bind(py),
        &module,
        modules.bind(py),
        parts.bind(py),
    )?;
    if is_stopped(py, decision.bind(py))? {
        return Ok(decision);
    }
    let chosen = ok_value(decision.bind(py))?;
    if chosen.is_none() {
        return program_err(py, &format!("cannot compute {}", class_name(py, &owner)));
    }
    let chosen_module = g(&chosen, "module")?;
    let node_name: String = g(node, "name")?.extract()?;
    let Some(definition) = definition(py, &chosen_module, &node_name)? else {
        let entry = g(&chosen, "entry")?;
        return program_err(
            py,
            &format!(
                "module '{}' has no '{node_name}'",
                g(&entry, "name")?.extract::<String>()?
            ),
        );
    };
    let runner = act_attr(act, "runner")?;
    let environ = act_attr(act, "environ")?;
    let entry = g(&chosen, "entry")?;
    let bindings = act_generic_bindings(py, &chosen)?;
    let other = new_activation(
        py,
        &runner,
        &chosen_module,
        &environ,
        &g(&entry, "name")?.extract::<String>()?,
        &g(&entry, "path")?,
        None,
        None,
        Some(bindings.bind(py)),
    )?;
    let taken = ast_class(py, "MemberTaken")?
        .call1((source, node_name.clone()))?
        .unbind();
    let current_module = act_attr(act, "module")?;
    act_value_of_definition(
        py,
        other.bind(py),
        &node_name,
        definition.bind(py),
        &chosen_module,
        Some(&current_module),
        Some(taken.bind(py)),
    )
}

fn act_shorthand_chain(
    py: Python<'_>,
    act: &Bound<'_, PyAny>,
    node: &Bound<'_, PyAny>,
    scope: &[P],
) -> PyResult<P> {
    let constructor: String = g(node, "constructor")?.extract()?;
    let container = act_resolve(py, act, &constructor, scope)?;
    if is_stopped(py, container.bind(py))? {
        return Ok(container);
    }
    let nil = make_ast0(py, "Nil")?;
    let tagged_env = make_tagged(py, ENVIRON_TAG, nil)?;
    let member = ast_class(py, "Member")?
        .call1((PyString::new(py, GET_ITEM_TAG),))?
        .unbind();
    let mut chain = make_partial(py, member, tagged_env)?;
    chain = make_partial(py, chain, make_type_ref(py, &constructor)?)?;
    for argument in g(node, "args")?.try_iter()? {
        chain = make_partial(py, chain, argument?.unbind())?;
    }
    act_apply_chain(py, act, chain.bind(py), scope, true, None)
}

fn act_product_member_given(
    py: Python<'_>,
    act: &Bound<'_, PyAny>,
    argument: &Bound<'_, PyAny>,
) -> PyResult<Option<P>> {
    let Some(path) = source_path(py, argument)? else {
        return Ok(None);
    };
    let Some((head, tag)) = path.rsplit_once('.') else {
        return Ok(None);
    };
    if head.is_empty() {
        return Ok(None);
    }
    let module = act_attr(act, "module")?;
    let Some(definition) = definition(py, &module, head)? else {
        return Ok(None);
    };
    if !is_get_args_call(py, &g(definition.bind(py), "body")?)? {
        return Ok(None);
    }
    act_as_given(py, act, &format!("${tag}"))
}

fn act_bound_value(
    py: Python<'_>,
    act: &Bound<'_, PyAny>,
    argument: &Bound<'_, PyAny>,
    scope: &[P],
) -> PyResult<Option<P>> {
    if !is_ast(py, argument, "TypeRef")? {
        return Ok(None);
    }
    let mut frames: Vec<P> = scope.iter().map(|frame| frame.clone_ref(py)).collect();
    let bindings = act_attr(act, "bindings")?;
    if bindings.call_method0("__len__")?.extract::<i64>()? > 0 {
        frames.push(bindings.unbind());
    }
    let name: String = g(argument, "name")?.extract()?;
    match in_scope(py, &frames, &name)? {
        Some(found) if is_shim(py, found.bind(py), "_VibaData")? => Ok(Some(found)),
        _ => Ok(None),
    }
}

fn act_argument_pieces(
    py: Python<'_>,
    act: &Bound<'_, PyAny>,
    node: &Bound<'_, PyAny>,
    scope: &[P],
) -> PyResult<Option<(P, P, P)>> {
    let mut nodes: Vec<P> = Vec::new();
    let mut modules: Vec<P> = Vec::new();
    let mut parts: Vec<P> = Vec::new();
    let mut bound = false;
    for argument in g(node, "args")?.try_iter()? {
        let argument = argument?;
        let mut value = act_bound_value(py, act, &argument, scope)?;
        if value.is_none() {
            value = act_product_member_given(py, act, &argument)?;
        }
        let Some(value) = value else {
            nodes.push(argument.unbind());
            modules.push(act_attr(act, "module")?.unbind());
            parts.push(py.None());
            continue;
        };
        bound = true;
        let value_bound = value.bind(py);
        let data = g(&g(value_bound, "node")?, "data")?.unbind();
        nodes.push(data);
        let source_module = g(value_bound, "source_module")?;
        if source_module.is_none() {
            modules.push(act_attr(act, "module")?.unbind());
        } else {
            modules.push(source_module.unbind());
        }
        let member_modules = member_modules(py, value_bound)?;
        parts.push(match member_modules {
            Some(modules) => pylist(py, modules)?,
            None => py.None(),
        });
    }
    if !bound {
        return Ok(None);
    }
    Ok(Some((
        pylist(py, nodes)?,
        pylist(py, modules)?,
        pylist(py, parts)?,
    )))
}

fn act_application_piece(
    py: Python<'_>,
    act: &Bound<'_, PyAny>,
    node: &Bound<'_, PyAny>,
    scope: &[P],
) -> PyResult<(P, P, P)> {
    match act_argument_pieces(py, act, node, scope)? {
        Some((nodes, modules, parts)) => {
            let constructor = g(node, "constructor")?;
            let built = ast_class(py, "TypeApp")?
                .call1((constructor, nodes.bind(py)))?
                .unbind();
            Ok((built, modules, parts))
        }
        None => {
            let empty = py.None();
            Ok((node.clone().unbind(), py.None(), py.None()))
        }
    }
}

fn act_generic_application(
    py: Python<'_>,
    act: &Bound<'_, PyAny>,
    node: &Bound<'_, PyAny>,
    scope: &[P],
) -> PyResult<P> {
    let constructor: String = g(node, "constructor")?.extract()?;
    if constructor == TAGGED_NAME {
        return act_tagged_data(py, act, node, scope);
    }
    let literal_ctors: Vec<String> = reflect_mod(py)?.getattr("LITERAL_CTORS")?.extract()?;
    if literal_ctors.contains(&constructor) {
        let node = substituted(py, node, scope)?;
        let descriptor = act_descriptor(py, act, node.bind(py))?;
        let object = make_vobject(py, descriptor.bind(py), node.bind(py), None, None)?;
        return make_ok(py, make_viba_data(py, object)?);
    }
    let (source, modules, parts) = act_application_piece(py, act, node, scope)?;
    let module = act_attr(act, "module")?;
    let decision = reduce_application(
        py,
        source.bind(py),
        &module,
        modules.bind(py),
        parts.bind(py),
    )?;
    if is_stopped(py, decision.bind(py))? {
        return Ok(decision);
    }
    let chosen = ok_value(decision.bind(py))?;
    if chosen.is_none() {
        if let Some(applied) = act_definition_generic(py, act, node, scope)? {
            return Ok(applied);
        }
        return act_shorthand_chain(py, act, node, scope);
    }
    let body = g(&chosen, "body")?;
    let entry = g(&chosen, "entry")?;
    if body.is_none() {
        return program_err(
            py,
            &format!(
                "{} answers a module: take one of its members by name, `{}.value`",
                g(&entry, "path")?.extract::<String>()?,
                one_line(py, node)?
            ),
        );
    }
    if is_ast(py, &body, "Exponent")? || is_ast(py, &body, "ExponentChain")? {
        let descriptor = act_descriptor(py, act, node)?;
        let object = make_vobject(py, descriptor.bind(py), node, None, None)?;
        let data = make_viba_data(py, object)?;
        data.bind(py).setattr("source_module", module)?;
        let bindings = act_attr(act, "bindings")?;
        if bindings.call_method0("__len__")?.extract::<i64>()? > 0 {
            data.bind(py).setattr("bindings", bindings)?;
        }
        return make_ok(py, data);
    }
    let runner = act_attr(act, "runner")?;
    let environ = act_attr(act, "environ")?;
    let bindings = act_generic_bindings(py, &chosen)?;
    let other = new_activation(
        py,
        &runner,
        &g(&chosen, "module")?,
        &environ,
        &g(&entry, "name")?.extract::<String>()?,
        &g(&entry, "path")?,
        None,
        None,
        Some(bindings.bind(py)),
    )?;
    act_evaluate(py, other.bind(py), &body, &[], true)
}

fn act_definition_generic(
    py: Python<'_>,
    act: &Bound<'_, PyAny>,
    node: &Bound<'_, PyAny>,
    _scope: &[P],
) -> PyResult<Option<P>> {
    let constructor: String = g(node, "constructor")?.extract()?;
    let module = act_attr(act, "module")?;
    let resolved = module_get_type(py, &module, &constructor)?;
    if !is_ok(py, resolved.bind(py))? {
        return Ok(None);
    }
    let found = ok_value(resolved.bind(py))?;
    if !is_type(py, &found, "AstNodeType")? {
        return Ok(None);
    }
    let target = g(&found, "ast_node")?;
    if !is_ast(py, &target, "GenericDefinition")? {
        return Ok(None);
    }
    let params = g(&target, "generic_params")?;
    let params: Vec<Bound<'_, PyAny>> = if params.is_none() {
        Vec::new()
    } else {
        params.try_iter()?.collect::<PyResult<Vec<_>>>()?
    };
    let args: Vec<Bound<'_, PyAny>> = g(node, "args")?.try_iter()?.collect::<PyResult<Vec<_>>>()?;
    if params.len() != args.len() {
        return Ok(Some(program_err(
            py,
            &format!(
                "'{constructor}' takes {} parameters, not {}",
                params.len(),
                args.len()
            ),
        )?));
    }
    let bindings = PyDict::new(py);
    for (param, argument) in params.iter().zip(args.iter()) {
        let descriptor = piece_descriptor(py, argument, &module)?;
        let object = make_vobject(py, descriptor.bind(py), argument, None, None)?;
        bindings.set_item(param, make_viba_data(py, object)?)?;
    }
    let runner = act_attr(act, "runner")?;
    let environ = act_attr(act, "environ")?;
    let file = act_attr(act, "file")?;
    let other = new_activation(
        py,
        &runner,
        &g(&found, "container_module")?,
        &environ,
        &constructor,
        &file,
        None,
        None,
        Some(bindings.as_any()),
    )?;
    let body = g(&target, "body")?;
    Ok(Some(act_evaluate(py, other.bind(py), &body, &[], true)?))
}

fn act_tagged_data(
    py: Python<'_>,
    act: &Bound<'_, PyAny>,
    node: &Bound<'_, PyAny>,
    scope: &[P],
) -> PyResult<P> {
    let args: Vec<Bound<'_, PyAny>> = g(node, "args")?.try_iter()?.collect::<PyResult<Vec<_>>>()?;
    if args.len() != 1 && args.len() != 2 {
        return program_err(
            py,
            &format!(
                "{TAGGED_NAME} takes one argument (the symbol) or two (the symbol and \
                 the type it marks), not {}",
                args.len()
            ),
        );
    }
    let taken = act_evaluate(py, act, &args[0], scope, true)?;
    if is_stopped(py, taken.bind(py))? {
        return Ok(taken);
    }
    let text = match symbol_text(py, &ok_value(taken.bind(py))?)? {
        Some(text) => text,
        None => {
            return program_err(
                py,
                &format!(
                    "{TAGGED_NAME} asks for a symbol (a string), not {}",
                    one_line(py, &args[0])?
                ),
            )
        }
    };
    let Some(symbol) = symbol_of(&text) else {
        return program_err(py, &symbol_problem(&text));
    };
    let mut arguments = vec![args[0].clone()];
    for argument in args.iter().skip(1) {
        let replaced = substituted(py, argument, scope)?;
        arguments.push(replaced.bind(py).clone());
    }
    let built = tagged_node(py, &symbol, &arguments)?;
    let described = if is_ast(py, built.bind(py), "Member")? {
        node.clone().unbind()
    } else {
        built.clone_ref(py)
    };
    let module = act_attr(act, "module")?;
    let descriptor = piece_descriptor(py, described.bind(py), &module)?;
    let object = make_vobject(py, descriptor.bind(py), built.bind(py), None, None)?;
    let data = make_viba_data(py, object)?;
    data.bind(py).setattr("source_module", module)?;
    make_ok(py, data)
}

fn act_generic_bindings(py: Python<'_>, chosen: &Bound<'_, PyAny>) -> PyResult<P> {
    let bindings = g(chosen, "bindings")?;
    let out = PyDict::new(py);
    for item in bindings.call_method0("items")?.try_iter()? {
        let item = item?;
        let name = item.get_item(0)?;
        let bound = item.get_item(1)?;
        let container_module = g(&bound, "container_module")?;
        let node = g(&bound, "ast_node")?;
        let descriptor = piece_descriptor(py, &node, &container_module)?;
        let object = make_vobject(py, descriptor.bind(py), &node, None, None)?;
        let value = make_viba_data(py, object)?;
        value.bind(py).setattr("source_module", container_module)?;
        out.set_item(name, value)?;
    }
    Ok(out.into_any().unbind())
}

// ----------------------------------------------------------------------
// The members of a piece, and what a product built out of values carries
// ----------------------------------------------------------------------

fn the_members<'py>(
    py: Python<'py>,
    value: &Bound<'py, PyAny>,
) -> PyResult<Vec<(Option<String>, P)>> {
    let members = g(value, "members")?;
    if !members.is_none() {
        let mut out = Vec::new();
        for member in members.try_iter()? {
            let member = member?;
            let tag = member.get_item(0)?;
            let tag = if tag.is_none() {
                None
            } else {
                Some(tag.extract::<String>()?)
            };
            out.push((tag, member.get_item(1)?.unbind()));
        }
        return Ok(out);
    }
    let node = g(value, "node")?;
    let data = g(&node, "data")?;
    let factors: Vec<Bound<'_, PyAny>>;
    let elements: Vec<Bound<'_, PyAny>>;
    if is_ast(py, &data, "Product")? || is_ast(py, &data, "ProductChain")? {
        factors = product_elements(py, &data)?;
        let descriptor = g(&node, "descriptor")?;
        elements = match g(&descriptor, "elements") {
            Ok(elements) => elements.try_iter()?.collect::<PyResult<Vec<_>>>()?,
            Err(_) => Vec::new(),
        };
    } else {
        factors = vec![data.clone()];
        elements = vec![g(&node, "descriptor")?];
    }
    let whole = factors.len() == 1;
    let source_module = g(value, "source_module")?;
    let given = g(value, "given")?;
    let bindings = g(value, "bindings")?;
    let function = g(&node, "function")?;
    let mut out = Vec::new();
    for (index, factor) in factors.iter().enumerate() {
        let descriptor = if index < elements.len() {
            elements[index].clone()
        } else {
            g(&node, "descriptor")?
        };
        let mut tag = None;
        let inner: Bound<'_, PyAny>;
        let mut descriptor = descriptor;
        if is_ast(py, factor, "Tagged")? {
            tag = Some(gs(factor, "tag")?);
            inner = g(factor, "type")?;
            let under = g(&descriptor, "tagged_type").unwrap_or_else(|_| py.None().into_bound(py));
            if !under.is_none() {
                descriptor = under;
            } else {
                descriptor = piece_descriptor(py, &inner, &source_module)?
                    .bind(py)
                    .clone();
            }
        } else {
            inner = factor.clone();
        }
        let member_source = if source_module.is_none() {
            descriptor_module(py, &descriptor)?
        } else {
            Some(source_module.clone().unbind())
        };
        let member_function = if whole { Some(function.clone()) } else { None };
        let object = make_vobject(py, &descriptor, &inner, None, member_function.as_ref())?;
        let value = make_viba_data(py, object)?;
        let value_bound = value.bind(py);
        if let Some(member_source) = member_source {
            value_bound.setattr("source_module", member_source)?;
        }
        let member_given = if whole {
            given.clone()
        } else {
            py.None().into_bound(py)
        };
        if !member_given.is_none() {
            value_bound.setattr("given", member_given)?;
        }
        if !bindings.is_none() {
            value_bound.setattr("bindings", &bindings)?;
        }
        out.push((tag, value));
    }
    Ok(out)
}

fn member_under_tag(py: Python<'_>, value: &Bound<'_, PyAny>, tag: &str) -> PyResult<Option<P>> {
    for (member_tag, member) in the_members(py, value)? {
        if member_tag.as_deref() == Some(tag) {
            return Ok(Some(member));
        }
    }
    Ok(None)
}

fn member_modules(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Option<Vec<P>>> {
    if !is_shim(py, value, "_VibaData")? {
        return Ok(None);
    }
    if g(value, "members")?.is_none() {
        return Ok(None);
    }
    let mut out = Vec::new();
    for (_tag, member) in the_members(py, value)? {
        out.push(g(member.bind(py), "source_module")?.unbind());
    }
    Ok(Some(out))
}

fn one_more_member(
    py: Python<'_>,
    product: &Bound<'_, PyAny>,
    tag: Option<&str>,
    value: &Bound<'_, PyAny>,
    source_module: &Bound<'_, PyAny>,
) -> PyResult<(Option<P>, Option<P>)> {
    if !is_shim(py, product, "_VibaData")? || !is_shim(py, value, "_VibaData")? {
        return Ok((
            None,
            Some(program_err(
                py,
                "a parameter spelled '...' holds one product, so what joins it has to \
                 be viba data: this argument is not",
            )?),
        ));
    }
    let mut members = the_members(py, product)?;
    members.push((tag.map(|tag| tag.to_string()), value.clone().unbind()));
    let mut nodes = Vec::new();
    let mut kept = Vec::new();
    for (member_tag, member) in &members {
        let data = g(&g(member.bind(py), "node")?, "data")?.unbind();
        nodes.push(match member_tag {
            Some(tag) => make_tagged(py, tag, data)?,
            None => data,
        });
        kept.push(member.clone_ref(py));
    }
    let chain = ast_class(py, "ProductChain")?
        .call1((PyList::new(py, nodes)?,))?
        .unbind();
    let descriptor = descriptor_of_values(py, chain.bind(py), source_module, kept)?;
    let object = make_vobject(py, descriptor.bind(py), chain.bind(py), None, None)?;
    let made = make_viba_data(py, object)?;
    made.bind(py).setattr("source_module", source_module)?;
    let bindings = g(product, "bindings")?;
    if !bindings.is_none() {
        made.bind(py).setattr("bindings", bindings)?;
    }
    made.bind(py).setattr(
        "members",
        PyList::new(
            py,
            members
                .into_iter()
                .map(|(tag, member)| {
                    Ok::<P, PyErr>(
                        PyTuple::new(
                            py,
                            [
                                match tag {
                                    Some(tag) => py_string(py, &tag),
                                    None => py.None(),
                                },
                                member,
                            ],
                        )?
                        .into_any()
                        .unbind(),
                    )
                })
                .collect::<PyResult<Vec<_>>>()?,
        )?,
    )?;
    Ok((Some(made), None))
}

fn is_get_args_call(py: Python<'_>, node: &Bound<'_, PyAny>) -> PyResult<bool> {
    if !is_ast(py, node, "Partial")? {
        return Ok(false);
    }
    let function = g(node, "function")?;
    Ok(is_ast(py, &function, "TypeRef")? && gs(&function, "name")? == GET_ARGS_NAME)
}

fn takes_the_rest(py: Python<'_>, source: &Bound<'_, PyAny>) -> PyResult<bool> {
    is_ast(py, source, "Ellipsis")
}

fn symbol_text(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
    if !is_shim(py, value, "_VibaData")? {
        return Ok(None);
    }
    let source = g(&g(value, "node")?, "data")?;
    if is_ast(py, &source, "Constant")? {
        let value = g(&source, "value")?;
        if value.is_instance_of::<PyString>() {
            return Ok(Some(value.extract()?));
        }
    }
    Ok(None)
}

fn symbol_of(text: &str) -> Option<String> {
    let name = text.strip_prefix('$').unwrap_or(text);
    if name.is_empty() {
        return None;
    }
    let mut chars = name.chars();
    let first = chars.next()?;
    if !(first.is_alphabetic() || first == '_') {
        return None;
    }
    if !name
        .chars()
        .all(|part| part.is_alphanumeric() || part == '_')
    {
        return None;
    }
    Some(name.to_string())
}

fn symbol_problem(text: &str) -> String {
    format!("'{text}' is no symbol for a tag")
}

fn tag_of(symbol: &str) -> String {
    format!("${symbol}")
}

fn tagged_node(py: Python<'_>, symbol: &str, arguments: &[Bound<'_, PyAny>]) -> PyResult<P> {
    if arguments.len() == 1 {
        return Ok(ast_class(py, "Member")?
            .call1((PyString::new(py, &tag_of(symbol)),))?
            .unbind());
    }
    Ok(ast_class(py, "Tagged")?
        .call1((PyString::new(py, &tag_of(symbol)), arguments[1].clone()))?
        .unbind())
}

fn member_tag_of(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<P> {
    let Some(text) = symbol_text(py, value)? else {
        return program_err(
            py,
            "a member is taken by the name a string spells, and this is no string",
        );
    };
    let Some(symbol) = symbol_of(&text) else {
        return program_err(py, &symbol_problem(&text));
    };
    make_ok(py, py_string(py, &tag_of(&symbol)))
}

fn address_of(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
) -> PyResult<(Option<&'static str>, Option<P>)> {
    if !is_shim(py, value, "_VibaData")? {
        return Ok((None, None));
    }
    let source = g(&g(value, "node")?, "data")?;
    if is_ast(py, &source, "Constant")? {
        let inner = g(&source, "value")?;
        if inner.is_instance_of::<PyString>() {
            return Ok((Some("key"), Some(inner.unbind())));
        }
        if inner.is_instance_of::<pyo3::types::PyInt>()
            && !inner.is_instance_of::<pyo3::types::PyBool>()
        {
            return Ok((Some("index"), Some(inner.unbind())));
        }
    }
    Ok((None, None))
}

fn required_arguments(py: Python<'_>, func: &Bound<'_, PyAny>) -> PyResult<i64> {
    let inspect = module(py, "inspect")?;
    let Ok(parameters) = inspect
        .getattr("signature")
        .and_then(|signature| signature.call1((func,)))
        .and_then(|signature| signature.getattr("parameters"))
        .and_then(|parameters| parameters.call_method0("values"))
    else {
        return Ok(1);
    };
    let empty = inspect.getattr("Parameter")?.getattr("empty")?;
    let mut count = 0;
    for parameter in parameters.try_iter()? {
        let parameter = parameter?;
        let default = g(&parameter, "default")?;
        if !default.is(&empty) {
            continue;
        }
        let kind = g(&parameter, "kind")?;
        let value: i64 = kind.getattr("value")?.extract()?;
        if value == 0 || value == 1 {
            count += 1;
        }
    }
    Ok(count)
}

fn host_answer(
    py: Python<'_>,
    name: &str,
    value: &Bound<'_, PyAny>,
    module_path: Option<&str>,
    full_qualified_func_name: Option<&str>,
    call: Option<&Bound<'_, PyAny>>,
) -> PyResult<P> {
    if is_shim(py, value, "_HostArgument")? {
        let closure = g(value, "closure_of")?.call0()?;
        if !closure.is_none() {
            return make_ok(py, make_viba_data(py, closure.unbind())?);
        }
        let node = g(value, "node")?;
        let module = g(value, "module")?;
        let descriptor = piece_descriptor(py, &node, &module)?;
        let object = make_vobject(py, descriptor.bind(py), &node, None, None)?;
        return make_ok(py, make_viba_data(py, object)?);
    }
    if is_a(py, value, REFLECT_MODULE, "VObject")? {
        return make_ok(py, make_viba_data(py, value.clone().unbind())?);
    }
    if is_shim(py, value, "Environment")? {
        let host = shim_class(py, "_Host")?.call1((value,))?.unbind();
        return make_ok(py, host);
    }
    let scalar = value.is_none()
        || value.is_instance_of::<pyo3::types::PyBool>()
        || value.is_instance_of::<pyo3::types::PyInt>()
        || value.is_instance_of::<pyo3::types::PyFloat>()
        || value.is_instance_of::<PyString>();
    if !scalar {
        let message = format!(
            "{name} answered {}, which is no leaf: answer a VObject, a scalar, or None",
            class_name(py, value)
        );
        if let Some(module_path) = module_path {
            return under_op_err(
                py,
                &format!("{REASON_NO_LEAF}: {message}"),
                module_path,
                full_qualified_func_name.unwrap_or(""),
                call,
                None,
            );
        }
        return program_err(py, &message);
    }
    let node = if value.is_none() {
        make_ast0(py, "Nil")?
    } else {
        ast_class(py, "Constant")?.call1((value,))?.unbind()
    };
    let module = no_module(py)?;
    let descriptor = piece_descriptor(py, node.bind(py), module.bind(py))?;
    let object = make_vobject(py, descriptor.bind(py), node.bind(py), None, None)?;
    let data = make_viba_data(py, object)?;
    make_ok(py, data)
}

fn act_in_container(
    py: Python<'_>,
    act: &Bound<'_, PyAny>,
    container: &Bound<'_, PyAny>,
    asked: &Bound<'_, PyAny>,
) -> PyResult<P> {
    let _ = act;
    if !is_shim(py, container, "_VibaData")? || !is_shim(py, asked, "_VibaData")? {
        return program_err(
            py,
            "an element is looked for in viba data, and this is no viba data",
        );
    }
    let node = g(container, "node")?;
    let access = reflect_access(py)?;
    let unfolded = access.call_method1("unfold", (&g(&node, "descriptor")?,))?;
    let kind: Option<String> = access
        .call_method1("container_kind", (&unfolded,))?
        .extract()?;
    let kind = kind.unwrap_or_default();
    if kind == "dict" {
        let Some(key) = symbol_text(py, asked)? else {
            return program_err(
                py,
                "a dict holds keys, and a key is the string the protocol hands keys over as",
            );
        };
        let holds = node
            .call_method1("keys", ())?
            .call_method1("__contains__", (key,))?;
        return make_ok(py, make_viba_data(py, viba_data(py, &holds)?)?);
    }
    let tuple_kind = descriptor_mod(py)?.getattr("TUPLE")?.extract::<String>()?;
    if kind == "list" || kind == "set" || gs(&unfolded, "kind")? == tuple_kind {
        let wanted = unparse_type(py, &g(&g(asked, "node")?, "data")?)?;
        for one in node.try_iter()? {
            let one = one?;
            if unparse_type(py, &g(&one, "data")?)? == wanted {
                let truth = pyo3::types::PyBool::new(py, true);
                return make_ok(py, make_viba_data(py, viba_data(py, truth.as_any())?)?);
            }
        }
        let falsity = pyo3::types::PyBool::new(py, false);
        return make_ok(py, make_viba_data(py, viba_data(py, falsity.as_any())?)?);
    }
    program_err(
        py,
        "`$in` asks about a list, a set, a tuple or a dict, and this is none of them",
    )
}

fn act_first_case_index(
    py: Python<'_>,
    act: &Bound<'_, PyAny>,
    arguments: &Bound<'_, PyAny>,
    func_arg_types: &Bound<'_, PyAny>,
) -> PyResult<P> {
    let _ = act;
    if !is_shim(py, arguments, "_VibaData")? || !is_shim(py, func_arg_types, "_VibaData")? {
        return program_err(
            py,
            "$first_case_index takes the arguments and a list of case types",
        );
    }
    let mut member_types: Vec<Bound<'_, PyAny>> = Vec::new();
    for (tag, member) in the_members(py, arguments)? {
        if tag.as_deref() == Some(ENVIRON_TAG) {
            continue;
        }
        let member = member.bind(py);
        let descriptor = match g(member, "descriptor") {
            Ok(descriptor) => descriptor,
            Err(_) => g(&g(member, "node")?, "descriptor")?,
        };
        let resolvable = g(&descriptor, "resolvable_type")?;
        if !resolvable.is_none() {
            member_types.push(resolvable);
        }
    }
    if member_types.is_empty() {
        return program_err(py, "$first_case_index got no argument to judge");
    }
    let source_module_of_arguments = g(arguments, "source_module")?;
    let args_type: P = if member_types.len() == 1 {
        member_types[0].clone().unbind()
    } else {
        let mut nodes = Vec::new();
        for one in &member_types {
            nodes.push(g(one, "ast_node")?.unbind());
        }
        let chain = ast_class(py, "ProductChain")?
            .call1((PyList::new(py, nodes)?,))?
            .unbind();
        make_ast_node_type(py, chain.bind(py), &source_module_of_arguments)?
    };
    let node = g(func_arg_types, "node")?;
    let mut index = 0;
    for item in node.try_iter()? {
        let item = item?;
        let descriptor = g(&item, "descriptor")?;
        let mut case_type = g(&descriptor, "resolvable_type")?;
        if case_type.is_none() {
            let module = {
                let a = g(func_arg_types, "source_module")?;
                if a.is_none() {
                    source_module_of_arguments.clone()
                } else {
                    a
                }
            };
            case_type = make_ast_node_type(py, &g(&item, "data")?, &module)?
                .bind(py)
                .clone();
        }
        let judged = is_sub_type_value(py, args_type.bind(py), &case_type)?;
        if is_type(py, judged.bind(py), "Err")? {
            return Ok(judged);
        }
        if ok_value(judged.bind(py))?.extract::<bool>()? {
            let index_value = pyo3::types::PyInt::new(py, index as i64);
            return make_ok(
                py,
                make_viba_data(py, viba_data(py, index_value.as_any())?)?,
            );
        }
        index += 1;
    }
    let message = viba_data(py, &py_string(py, "no handler matches").into_bound(py))?;
    let tagged = tagged_viba_data(py, "$dispatch_error_message", message.bind(py))?;
    make_ok(py, make_viba_data(py, tagged)?)
}

fn act_container_len(
    py: Python<'_>,
    act: &Bound<'_, PyAny>,
    container: &Bound<'_, PyAny>,
) -> PyResult<P> {
    let _ = act;
    if !is_shim(py, container, "_VibaData")? {
        return program_err(py, "a container is counted, and this is no viba data");
    }
    let node = g(container, "node")?;
    let access = reflect_access(py)?;
    let unfolded = access.call_method1("unfold", (&g(&node, "descriptor")?,))?;
    let kind: Option<String> = access
        .call_method1("container_kind", (&unfolded,))?
        .extract()?;
    let kind = kind.unwrap_or_default();
    let tuple_kind = descriptor_mod(py)?.getattr("TUPLE")?.extract::<String>()?;
    if !(kind == "list" || kind == "set" || kind == "dict" || gs(&unfolded, "kind")? == tuple_kind)
    {
        return program_err(
            py,
            "`$len` counts a list, a set, a tuple or a dict, and this is none of them",
        );
    }
    let counted = access.call_method1("length", (&node,))?;
    if is_stopped(py, &counted)? {
        return Ok(counted.unbind());
    }
    let data = viba_data(py, &ok_value(&counted)?)?;
    let made = make_viba_data(py, data)?;
    let made_bound = made.bind(py);
    let source_module = g(container, "source_module")?;
    if !source_module.is_none() {
        made_bound.setattr("source_module", source_module)?;
    }
    let bindings = g(container, "bindings")?;
    if !bindings.is_none() {
        made_bound.setattr("bindings", bindings)?;
    }
    make_ok(py, made)
}

fn act_container_keys(
    py: Python<'_>,
    act: &Bound<'_, PyAny>,
    container: &Bound<'_, PyAny>,
) -> PyResult<P> {
    let _ = act;
    if !is_shim(py, container, "_VibaData")? {
        return program_err(
            py,
            "the keys of a container are taken, and this is no viba data",
        );
    }
    let node = g(container, "node")?;
    let access = reflect_access(py)?;
    let unfolded = access.call_method1("unfold", (&g(&node, "descriptor")?,))?;
    let kind: Option<String> = access
        .call_method1("container_kind", (&unfolded,))?
        .extract()?;
    let kind = kind.unwrap_or_default();
    if kind != "dict" {
        return program_err(py, "`$keys` gives the keys of a dict, and this is not one");
    }
    let keys = access.call_method1("keys", (&node,))?;
    if is_stopped(py, &keys)? {
        return Ok(keys.unbind());
    }
    let mut elements = Vec::new();
    for key in ok_value(&keys)?.try_iter()? {
        elements.push(ast_class(py, "Constant")?.call1((key?,))?.unbind());
    }
    let node = ast_class(py, "TypeApp")?
        .call1((py_string(py, "ListLiteral"), PyList::new(py, elements)?))?
        .unbind();
    let data = viba_data(py, node.bind(py))?;
    make_ok(py, make_viba_data(py, data)?)
}

fn act_take_item(
    py: Python<'_>,
    act: &Bound<'_, PyAny>,
    container: &Bound<'_, PyAny>,
    address: &Bound<'_, PyAny>,
) -> PyResult<P> {
    let _ = act;
    if !is_shim(py, container, "_VibaData")? {
        return program_err(
            py,
            "an element is taken out of viba data, and this is no viba data",
        );
    }
    let (kind, asked) = address_of(py, address)?;
    let Some(kind) = kind else {
        return program_err(
            py,
            "an element is taken by position (an int) or by key (a str), and this is neither",
        );
    };
    let node = g(container, "node")?;
    let taken = if kind == "index" {
        node.call_method1("at_index", (asked.unwrap(),))
    } else {
        node.call_method1("at_key", (asked.unwrap(),))
    };
    let taken = match taken {
        Ok(taken) => taken,
        Err(error) => {
            if error
                .value(py)
                .is_instance(&reflect_mod(py)?.getattr("VibaReflectError")?)?
            {
                let text = error
                    .value(py)
                    .str()
                    .map(|text| text.to_string())
                    .unwrap_or_default();
                return program_err(py, &text);
            }
            return Err(error);
        }
    };
    let made = make_viba_data(py, taken.unbind())?;
    let source_module = g(container, "source_module")?;
    if !source_module.is_none() {
        made.bind(py).setattr("source_module", source_module)?;
    }
    let bindings = g(container, "bindings")?;
    if !bindings.is_none() {
        made.bind(py).setattr("bindings", bindings)?;
    }
    make_ok(py, made)
}

fn act_take_member(
    py: Python<'_>,
    act: &Bound<'_, PyAny>,
    member: &Bound<'_, PyAny>,
    value: &Bound<'_, PyAny>,
) -> PyResult<P> {
    let tag: String = g(member, "tag")?.extract()?;
    let name = tag.strip_prefix('$').unwrap_or(&tag).to_string();
    if is_environ_value(py, value)? {
        let obj = g(value, "obj")?;
        let attributed = match g(&obj, &name) {
            Ok(attributed) => attributed,
            Err(error) => {
                if error.is_instance_of::<pyo3::exceptions::PyAttributeError>(py) {
                    return program_err(py, &format!("the environment has no '{name}'"));
                }
                return Err(error);
            }
        };
        if !attributed.is_callable() {
            return host_answer(
                py,
                &format!("the environment's {name}"),
                &attributed,
                None,
                None,
                None,
            );
        }
        let slots = required_arguments(py, &attributed)?;
        let given = PyList::new(py, [obj.clone().unbind()])?;
        let pieces = PyList::new(py, [PyTuple::new(py, [py.None(), py.None()])?])?;
        return make_ok(
            py,
            shim_class(py, "_HostFunction")?
                .call1((name, attributed, slots, given, pieces))?
                .unbind(),
        );
    }
    if is_shim(py, value, "_VibaData")? {
        let args = act_attr(act, "args")?;
        if value.is(&args) {
            if let Some(given) = act_as_given(py, act, &tag)? {
                return make_ok(py, given);
            }
        }
        for factor in viba_data_factors(py, &g(value, "node")?)? {
            if is_ast(py, &factor, "Tagged")? && gs(&factor, "tag")? == tag {
                if let Some(member) = member_under_tag(py, value, &tag)? {
                    return make_ok(py, member);
                }
            }
        }
        return program_err(py, &format!("no member tagged '{tag}' to take from it"));
    }
    program_err(
        py,
        &format!("{} has no member tagged '{tag}'", class_name(py, value)),
    )
}

fn act_environment_member(
    py: Python<'_>,
    act: &Bound<'_, PyAny>,
    name: &str,
    scope: &[P],
) -> PyResult<Option<P>> {
    let Some((head, rest)) = name.rsplit_once('.') else {
        return Ok(None);
    };
    if head.is_empty() {
        return Ok(None);
    }
    let value = act_resolve(py, act, head, scope)?;
    if is_stopped(py, value.bind(py))? {
        return Ok(None);
    }
    if !is_environ_value(py, &ok_value(value.bind(py))?)? {
        return Ok(None);
    }
    let environ = g(&ok_value(value.bind(py))?, "obj")?;
    let member = match g(&environ, rest) {
        Ok(member) => member,
        Err(error) => {
            if error.is_instance_of::<pyo3::exceptions::PyAttributeError>(py) {
                return Ok(Some(program_err(
                    py,
                    &format!("the environment has no '{rest}'"),
                )?));
            }
            return Err(error);
        }
    };
    if !member.is_callable() {
        return Ok(Some(host_answer(
            py,
            &format!("the environment's {rest}"),
            &member,
            None,
            None,
            None,
        )?));
    }
    let slots = required_arguments(py, &member)?;
    Ok(Some(make_ok(
        py,
        shim_class(py, "_HostFunction")?
            .call1((rest, member, slots))?
            .unbind(),
    )?))
}

// ----------------------------------------------------------------------
// Calls: giving arguments, and finishing a chain
// ----------------------------------------------------------------------

fn call_parts(py: Python<'_>, node: &Bound<'_, PyAny>) -> PyResult<(P, Vec<P>)> {
    let mut source: Vec<P> = Vec::new();
    let mut current = node.clone();
    while is_ast(py, &current, "Partial")? {
        source.push(g(&current, "argument")?.unbind());
        current = g(&current, "function")?;
    }
    source.reverse();
    Ok((current.unbind(), source))
}

fn elements(py: Python<'_>, node: &Bound<'_, PyAny>) -> PyResult<Vec<P>> {
    if is_ast(py, node, "ExponentChain")? {
        let mut out = Vec::new();
        for element in g(node, "elements")?.try_iter()? {
            out.push(element?.unbind());
        }
        return Ok(out);
    }
    if is_ast(py, node, "Exponent")? {
        let mut out = elements(py, &g(node, "result")?)?;
        out.push(g(node, "argument")?.unbind());
        return Ok(out);
    }
    Ok(vec![node.clone().unbind()])
}

fn slots_of(py: Python<'_>, chain: &Bound<'_, PyAny>) -> PyResult<Vec<P>> {
    let mut found = elements(py, chain)?;
    found.remove(0);
    if let Some(last) = found.last() {
        if is_ast(py, last.bind(py), "HintBlock")? || is_ast(py, last.bind(py), "Partial")? {
            found.pop();
        }
    }
    let mut out = Vec::new();
    for element in found {
        if !is_ast(py, element.bind(py), "HintBlock")? {
            out.push(element);
        }
    }
    Ok(out)
}

fn is_a_source_call(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<bool> {
    if !is_shim(py, value, "_VibaData")? {
        return Ok(false);
    }
    let data = g(&g(value, "node")?, "data")?;
    Ok(is_ast(py, &data, "Partial")?
        || is_ast(py, &data, "TypeRef")?
        || is_ast(py, &data, "TypeApp")?)
}

fn module_in_source(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Option<P>> {
    if !is_shim(py, value, "_VibaData")? {
        return Ok(None);
    }
    let module = g(value, "source_module")?;
    if module.is_none() {
        Ok(None)
    } else {
        Ok(Some(module.unbind()))
    }
}

fn bindings_in_source<'py>(
    py: Python<'py>,
    value: &Bound<'py, PyAny>,
) -> Option<Bound<'py, PyAny>> {
    if !is_shim(py, value, "_VibaData").ok()? {
        return None;
    }
    let bindings = g(value, "bindings").ok()?;
    if bindings.is_none() {
        None
    } else {
        Some(bindings)
    }
}

fn stored_arguments(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<(P, Vec<P>)> {
    let (head, stored) = call_parts(py, &g(&g(value, "node")?, "data")?)?;
    let mut stored_values: Vec<Bound<'_, PyAny>> = Vec::new();
    let given = g(value, "given")?;
    if !given.is_none() {
        stored_values = given.try_iter()?.collect::<PyResult<Vec<_>>>()?;
    }
    let mut arguments = Vec::new();
    for (index, piece) in stored.into_iter().enumerate() {
        let (tag, remembered) = if index < stored_values.len() {
            let pair = &stored_values[index];
            let tag = pair.get_item(0)?;
            let tag = if tag.is_none() {
                None
            } else {
                Some(tag.extract::<String>()?)
            };
            let remembered = pair.get_item(1)?;
            (
                tag,
                if remembered.is_none() {
                    None
                } else {
                    Some(remembered.unbind())
                },
            )
        } else {
            (None, None)
        };
        match remembered {
            Some(remembered) => {
                let given = shim_class(py, "_Given")?
                    .call1((
                        match &tag {
                            Some(tag) => py_string(py, tag),
                            None => py.None(),
                        },
                        remembered,
                    ))?
                    .unbind();
                arguments.push(given);
            }
            None => arguments.push(piece),
        }
    }
    Ok((head, arguments))
}

fn function_slot(py: Python<'_>, pending: &Bound<'_, PyAny>, index: usize) -> PyResult<Option<P>> {
    let element_list = g(pending, "elements")?;
    let element_list: Vec<Bound<'_, PyAny>> =
        element_list.try_iter()?.collect::<PyResult<Vec<_>>>()?;
    if index >= element_list.len() {
        return Ok(None);
    }
    let source = &element_list[index];
    let inner = if is_ast(py, source, "Tagged")? {
        g(source, "type")?
    } else {
        source.clone()
    };
    if is_ast(py, &inner, "Exponent")? || is_ast(py, &inner, "ExponentChain")? {
        let parts = elements(py, &inner)?;
        if parts.is_empty() {
            return Ok(None);
        }
        return Ok(Some(parts[0].clone_ref(py)));
    }
    Ok(None)
}

fn handed_to_host(
    py: Python<'_>,
    pending: &Bound<'_, PyAny>,
    index: usize,
    value: &Bound<'_, PyAny>,
) -> PyResult<P> {
    if is_shim(py, value, "_HostArgument")? {
        return Ok(value.clone().unbind());
    }
    if function_slot(py, pending, index)?.is_some() && is_a_source_call(py, value)? {
        let activation = g(pending, "activation")?;
        let source_module = if activation.is_none() {
            py.None()
        } else {
            g(&activation, "module")?.unbind()
        };
        let name = if activation.is_none() {
            String::new()
        } else {
            gs(&activation, "name")?
        };
        let module = g(pending, "module")?;
        let source = g(pending, "source")?;
        let made = shim_class(py, "_HostArgument")?.call1((
            g(&g(value, "node")?, "data")?,
            g(pending, "activation")?,
            pending,
            module,
            source,
            py.None(),
            source_module,
            name,
            if activation.is_none() {
                py.None()
            } else {
                g(pending, "activation")?.unbind()
            },
        ))?;
        return Ok(made.unbind());
    }
    argument_value(py, value)
}

fn host_give(py: Python<'_>, function: &Bound<'_, PyAny>, item: &Bound<'_, PyAny>) -> PyResult<P> {
    if is_shim(py, function, "_HostFunction")? {
        return pending_give_host_function(py, function, item);
    }
    if is_shim(py, function, "_GetArgs")? {
        return get_args_give(py, function, item);
    }
    if is_shim(py, function, "_IsSubType")? {
        return is_sub_type_give(py, function, item);
    }
    if is_shim(py, function, "_VibaData")? {
        return program_err(
            py,
            &format!(
                "{} is not a function: it is a value",
                pyrepr(py, &g(function, "node")?)
            ),
        );
    }
    if is_shim(py, function, "_Host")? {
        return program_err(
            py,
            &format!("{} is not a function", class_name(py, &g(function, "obj")?)),
        );
    }
    program_err(
        py,
        &format!("{} is not a function", class_name(py, function)),
    )
}

fn act_is_the_environment(py: Python<'_>, argument: &Bound<'_, PyAny>) -> PyResult<bool> {
    if is_ast(py, argument, "Tagged")? {
        return Ok(gs(argument, "tag")? == ENVIRON_TAG);
    }
    Ok(false)
}

fn act_argument_is_lazy(
    py: Python<'_>,
    act: &Bound<'_, PyAny>,
    current: &Bound<'_, PyAny>,
    argument: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    if !is_shim(py, current, "_Pending")? || gs(current, "kind")? != "func" {
        return Ok(false);
    }
    if act_is_the_environment(py, argument)? {
        return Ok(false);
    }
    let Some(index) = pending_index_for(py, current, argument)? else {
        return Ok(false);
    };
    Ok(function_slot(py, current, index)?.is_some())
}

fn act_argument(
    py: Python<'_>,
    act: &Bound<'_, PyAny>,
    node: &Bound<'_, PyAny>,
    scope: &[P],
) -> PyResult<P> {
    if is_ast(py, node, "HintBlock")? {
        return make_ok_none(py);
    }
    let (tag, inner) = addressed(py, node)?;
    let tag = match tag {
        Some(tag) => Some(tag),
        None => act_tag_from_get_attr(py, act, node, scope)?,
    };
    let value = act_evaluate(py, act, &inner, scope, true)?;
    if is_stopped(py, value.bind(py))? {
        return Ok(value);
    }
    let given = shim_class(py, "_Given")?.call1((
        match tag {
            Some(tag) => py_string(py, &tag),
            None => py.None(),
        },
        ok_value(value.bind(py))?,
    ))?;
    make_ok(py, given.unbind())
}

fn act_lazy_argument(
    py: Python<'_>,
    act: &Bound<'_, PyAny>,
    node: &Bound<'_, PyAny>,
    scope: &[P],
) -> PyResult<P> {
    if is_ast(py, node, "HintBlock")? {
        return make_ok_none(py);
    }
    let (tag, inner) = addressed(py, node)?;
    let scope = scope_object(py, scope.iter().map(|frame| frame.clone_ref(py)).collect())?;
    let getter = shim_class(py, "_Getter")?
        .call1((act, inner, scope))?
        .unbind();
    let given = shim_class(py, "_Given")?.call1((
        match tag {
            Some(tag) => py_string(py, &tag),
            None => py.None(),
        },
        getter,
    ))?;
    make_ok(py, given.unbind())
}

fn act_tag_from_get_attr(
    py: Python<'_>,
    act: &Bound<'_, PyAny>,
    node: &Bound<'_, PyAny>,
    scope: &[P],
) -> PyResult<Option<String>> {
    let (head, arguments) = call_parts(py, node)?;
    if !is_ast(py, head.bind(py), "Member")? || gs(head.bind(py), "tag")? != GET_ATTR_TAG {
        return Ok(None);
    }
    if arguments.len() < 2 {
        return Ok(None);
    }
    let last = arguments.last().unwrap();
    if !is_ast(py, last.bind(py), "TypeRef")? {
        return Ok(None);
    }
    let spelled = act_evaluate(py, act, last.bind(py), scope, true)?;
    if is_stopped(py, spelled.bind(py))? {
        return Ok(None);
    }
    let named = member_tag_of(py, &ok_value(spelled.bind(py))?)?;
    if is_stopped(py, named.bind(py))? {
        return Ok(None);
    }
    let tag: String = ok_value(named.bind(py))?.extract()?;
    Ok(Some(tag))
}

fn addressed<'py>(
    py: Python<'py>,
    node: &Bound<'py, PyAny>,
) -> PyResult<(Option<String>, Bound<'py, PyAny>)> {
    if is_ast(py, node, "Tagged")? {
        return Ok((Some(gs(node, "tag")?), g(node, "type")?));
    }
    Ok((None, node.clone()))
}

fn new_given(py: Python<'_>, tag: Option<&str>, value: P) -> PyResult<P> {
    Ok(shim_class(py, "_Given")?
        .call1((
            match tag {
                Some(tag) => py_string(py, tag),
                None => py.None(),
            },
            value,
        ))?
        .unbind())
}

fn new_member(py: Python<'_>, tag: &str) -> PyResult<P> {
    Ok(shim_class(py, "_Member")?
        .call1((py_string(py, tag),))?
        .unbind())
}

fn new_host(py: Python<'_>, obj: P) -> PyResult<P> {
    Ok(shim_class(py, "_Host")?.call1((obj,))?.unbind())
}

fn interpreter_member_tags() -> [&'static str; 6] {
    [
        GET_ATTR_TAG,
        GET_ITEM_TAG,
        IN_TAG,
        LEN_TAG,
        KEYS_TAG,
        FIRST_CASE_INDEX_TAG,
    ]
}

fn interpreter_member_source(tag: &str) -> Option<&'static str> {
    Some(match tag {
        GET_ATTR_TAG => "Any <- $env Env <- $value Any <- $name str",
        GET_ITEM_TAG => "Any <- $env Env <- $container Any <- $address Any",
        IN_TAG => "bool <- $env Env <- $container Any <- $piece Any",
        LEN_TAG => "int <- $env Env <- $container Any",
        KEYS_TAG => "list[str] <- $env Env <- $table Any",
        FIRST_CASE_INDEX_TAG => {
            "(int | DispatchError) <- $env Env <- $args Any <- $func_arg_types list[Any]"
        }
        _ => return None,
    })
}

fn interpreter_member_chain(py: Python<'_>, tag: &str) -> PyResult<Option<P>> {
    static CHAINS: PyOnceLock<Py<PyDict>> = PyOnceLock::new();
    let Some(source) = interpreter_member_source(tag) else {
        return Ok(None);
    };
    let chains = CHAINS.get_or_try_init(py, || -> PyResult<Py<PyDict>> {
        Ok(PyDict::new(py).unbind())
    })?;
    let chains = chains.bind(py);
    if let Some(found) = chains.get_item(tag)? {
        if !found.is_none() {
            return Ok(Some(found.unbind()));
        }
    }
    let program = ast_mod(py)?
        .getattr("parse")?
        .call1((format!("__decl__ =\n  {source}\n"),))?;
    let body = program.getattr("body")?.get_item(0)?.getattr("body")?;
    chains.set_item(tag, &body)?;
    Ok(Some(body.unbind()))
}

fn act_member_pending(
    py: Python<'_>,
    act: &Bound<'_, PyAny>,
    head: &Bound<'_, PyAny>,
    node: &Bound<'_, PyAny>,
) -> PyResult<P> {
    let tag: String = g(head, "tag")?.extract()?;
    let chain = match interpreter_member_chain(py, &tag)? {
        Some(chain) => chain,
        None => return program_err(py, &format!("no member chain for '{tag}'")),
    };
    let slots = slots_of(py, chain.bind(py))?;
    let runner = act_attr(act, "runner")?;
    let module = act_attr(act, "module")?;
    let source_bindings = act_attr(act, "bindings")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("kind", "func")?;
    kwargs.set_item("activation", act)?;
    kwargs.set_item("head", head)?;
    kwargs.set_item("source", one_line(py, node)?)?;
    kwargs.set_item("name", &tag)?;
    kwargs.set_item("module", builtin_module(py)?.bind(py))?;
    kwargs.set_item("source_module", &module)?;
    kwargs.set_item("elements", PyList::new(py, slots)?)?;
    kwargs.set_item(
        "source_bindings",
        if source_bindings.call_method0("__len__")?.extract::<i64>()? > 0 {
            source_bindings.clone().unbind()
        } else {
            py.None()
        },
    )?;
    let _ = runner;
    let pending = shim_class(py, "_Pending")?
        .call((), Some(&kwargs))?
        .unbind();
    pending.bind(py).setattr("member_tag", &tag)?;
    make_ok(py, pending)
}

fn act_func_pending(
    py: Python<'_>,
    act: &Bound<'_, PyAny>,
    name_node: &Bound<'_, PyAny>,
    source: &str,
    owner_module: &Bound<'_, PyAny>,
    chain: &Bound<'_, PyAny>,
    name: &str,
    source_module: Option<&Bound<'_, PyAny>>,
) -> PyResult<P> {
    let slots = slots_of(py, chain)?;
    let envs: Vec<&P> = slots
        .iter()
        .filter(|one| {
            is_ast(py, one.bind(py), "Tagged").unwrap_or(false)
                && gs(one.bind(py), "tag")
                    .map(|tag| tag == ENVIRON_TAG)
                    .unwrap_or(false)
        })
        .collect();
    if envs.is_empty() {
        return program_err(
            py,
            &format!(
                "{source} takes no {ENVIRON_TAG} {ENV_TYPE} parameter: every function that runs \
                 depends on the environment"
            ),
        );
    }
    let env_type = g(envs[0].bind(py), "type")?;
    if !names_the_environment(py, &env_type)? {
        return program_err(
            py,
            &format!(
                "{source}: the {ENVIRON_TAG} parameter must be {ENV_TYPE}, not {}",
                unparse_type(py, &env_type)?
            ),
        );
    }
    let kwargs = PyDict::new(py);
    kwargs.set_item("kind", "func")?;
    kwargs.set_item("activation", act)?;
    kwargs.set_item("head", name_node)?;
    kwargs.set_item("source", source)?;
    kwargs.set_item("name", name)?;
    kwargs.set_item("module", owner_module)?;
    let source_module_value = match source_module {
        Some(source_module) => source_module.clone().unbind(),
        None => owner_module.clone().unbind(),
    };
    kwargs.set_item("source_module", source_module_value)?;
    kwargs.set_item("elements", PyList::new(py, slots)?)?;
    let bindings = act_attr(act, "bindings")?;
    kwargs.set_item(
        "source_bindings",
        if bindings.call_method0("__len__")?.extract::<i64>()? > 0 {
            bindings.unbind()
        } else {
            py.None()
        },
    )?;
    make_ok(
        py,
        shim_class(py, "_Pending")?
            .call((), Some(&kwargs))?
            .unbind(),
    )
}

fn act_pending_of(
    py: Python<'_>,
    act: &Bound<'_, PyAny>,
    definition: &Bound<'_, PyAny>,
    name_node: &Bound<'_, PyAny>,
    name: &str,
    owner_module: &Bound<'_, PyAny>,
    source: &str,
    local_name: &str,
    source_module: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<P>> {
    let body = g(definition, "body")?;
    if is_ast(py, &body, "Exponent")? || is_ast(py, &body, "ExponentChain")? {
        let used = if local_name.is_empty() {
            name
        } else {
            local_name
        };
        return Ok(Some(act_func_pending(
            py,
            act,
            name_node,
            source,
            owner_module,
            &body,
            used,
            source_module,
        )?));
    }
    Ok(None)
}

fn act_generic_target(
    py: Python<'_>,
    act: &Bound<'_, PyAny>,
    node: &Bound<'_, PyAny>,
    scope: &[P],
) -> PyResult<Option<P>> {
    let (source, modules, parts) = act_application_piece(py, act, node, scope)?;
    let module = act_attr(act, "module")?;
    let decision = reduce_application(
        py,
        source.bind(py),
        &module,
        modules.bind(py),
        parts.bind(py),
    )?;
    if is_stopped(py, decision.bind(py))? {
        return Ok(Some(decision));
    }
    let chosen = ok_value(decision.bind(py))?;
    if chosen.is_none() {
        return Ok(None);
    }
    let body = g(&chosen, "body")?;
    if !is_ast(py, &body, "Exponent")? && !is_ast(py, &body, "ExponentChain")? {
        return Ok(None);
    }
    let bindings = act_generic_bindings(py, &chosen)?;
    let chain = substituted(py, &body, &[bindings])?;
    let chain_module = g(&chosen, "module")?;
    let chain = source_tags(py, chain.bind(py), &chain_module)?;
    if definition(py, &chain_module, RET_NAME)?.is_some() {
        let kwargs = PyDict::new(py);
        kwargs.set_item("kind", "module")?;
        kwargs.set_item("runner", act_attr(act, "runner")?)?;
        kwargs.set_item("module", &chain_module)?;
        let entry = g(&chosen, "entry")?;
        kwargs.set_item("module_name", g(&entry, "name")?)?;
        kwargs.set_item("head", node)?;
        kwargs.set_item("source", g(&entry, "name")?)?;
        kwargs.set_item("source_module", &module)?;
        let generic_bindings = act_generic_bindings(py, &chosen)?;
        kwargs.set_item("bindings", generic_bindings)?;
        kwargs.set_item("pattern_env", g(&entry, "order")?.str()?.to_string())?;
        let params = parameters_of(py, chain.bind(py))?;
        kwargs.set_item("slots", PyList::new(py, params)?)?;
        let bindings = act_attr(act, "bindings")?;
        kwargs.set_item(
            "source_bindings",
            if bindings.call_method0("__len__")?.extract::<i64>()? > 0 {
                bindings.unbind()
            } else {
                py.None()
            },
        )?;
        return Ok(Some(make_ok(
            py,
            shim_class(py, "_Pending")?
                .call((), Some(&kwargs))?
                .unbind(),
        )?));
    }
    let constructor: String = g(node, "constructor")?.extract()?;
    let source_text = one_line(py, node)?;
    Ok(Some(act_func_pending(
        py,
        act,
        node,
        &source_text,
        &chain_module,
        chain.bind(py),
        &constructor,
        Some(&module),
    )?))
}

fn act_builtin_member(py: Python<'_>, name: &str) -> PyResult<Option<P>> {
    let Some((head, tag)) = name.rsplit_once('.') else {
        return Ok(None);
    };
    let found = builtin_module(py)?
        .bind(py)
        .clone()
        .call_method1("lookup", (head,))?;
    if !is_ok(py, &found)? {
        return Ok(None);
    }
    let found = ok_value(&found)?;
    if !is_type(py, &found, "AstNodeType")? {
        return Ok(None);
    }
    let inner = tagged_piece(py, &g(&found, "ast_node")?, tag)?;
    match inner {
        Some(inner) => Ok(Some(builtin_member_data(py, &inner)?)),
        None => Ok(Some(program_err(
            py,
            &format!("'{head}' has no member tagged '${tag}'"),
        )?)),
    }
}

fn act_builtin_bare_member(py: Python<'_>, name: &str) -> PyResult<Option<P>> {
    let piece = builtin_module(py)?
        .bind(py)
        .clone()
        .call_method1("operator", (name,))?;
    if piece.is_none() {
        return Ok(None);
    }
    Ok(Some(builtin_member_data(py, &piece)?))
}

fn act_builtin_bare_call(
    py: Python<'_>,
    act: &Bound<'_, PyAny>,
    name_node: &Bound<'_, PyAny>,
    name: &str,
) -> PyResult<Option<P>> {
    let member = act_builtin_bare_member(py, name)?;
    let Some(member) = member else {
        return Ok(None);
    };
    if !is_ok(py, member.bind(py))? || !is_shim(py, &ok_value(member.bind(py))?, "_VibaData")? {
        return Ok(None);
    }
    let chain = g(&g(&ok_value(member.bind(py))?, "node")?, "data")?;
    if !is_ast(py, &chain, "Exponent")? && !is_ast(py, &chain, "ExponentChain")? {
        return Ok(None);
    }
    let module = act_attr(act, "module")?;
    Ok(Some(act_func_pending(
        py,
        act,
        name_node,
        name,
        builtin_module(py)?.bind(py),
        &chain,
        name,
        Some(&module),
    )?))
}

fn act_builtin_module_call(
    py: Python<'_>,
    act: &Bound<'_, PyAny>,
    name_node: &Bound<'_, PyAny>,
    name: &str,
) -> PyResult<Option<P>> {
    let Some(module_name) = builtin_directory_name(py, name)? else {
        return Ok(None);
    };
    let runner = act_attr(act, "runner")?;
    let module = runner_builtin_module(py, &runner, &module_name)?;
    let Some(module) = module else {
        return Ok(None);
    };
    if is_stopped(py, module.bind(py))? {
        return Ok(Some(module));
    }
    let module = ok_value(module.bind(py))?.unbind();
    let kwargs = PyDict::new(py);
    kwargs.set_item("kind", "module")?;
    kwargs.set_item("runner", &runner)?;
    kwargs.set_item("module", &module)?;
    kwargs.set_item("module_name", &module_name)?;
    kwargs.set_item("head", name_node)?;
    kwargs.set_item("source", &module_name)?;
    kwargs.set_item("source_module", act_attr(act, "module")?)?;
    let bindings = act_attr(act, "bindings")?;
    kwargs.set_item(
        "source_bindings",
        if bindings.call_method0("__len__")?.extract::<i64>()? > 0 {
            bindings.unbind()
        } else {
            py.None()
        },
    )?;
    Ok(Some(make_ok(
        py,
        shim_class(py, "_Pending")?
            .call((), Some(&kwargs))?
            .unbind(),
    )?))
}

fn act_member_function<'py>(
    py: Python<'py>,
    act: &Bound<'py, PyAny>,
    name: &str,
) -> PyResult<Option<(P, P, String)>> {
    if !name.contains('.') {
        return Ok(None);
    }
    let own = name.rsplit('.').next().unwrap().to_string();
    let mut member = act_tagged_member(py, act, name, &[])?;
    let declared_in: P;
    if member.is_some() && is_ok(py, member.as_ref().unwrap().bind(py))? {
        let value = ok_value(member.as_ref().unwrap().bind(py))?;
        let source_module = g(&value, "source_module")?;
        declared_in = if source_module.is_none() {
            act_attr(act, "module")?.unbind()
        } else {
            source_module.unbind()
        };
    } else {
        member = act_builtin_member(py, name)?;
        declared_in = builtin_module(py)?;
    }
    let Some(member) = member else {
        return Ok(None);
    };
    if !is_ok(py, member.bind(py))? {
        return Ok(None);
    }
    let value = ok_value(member.bind(py))?;
    if !is_shim(py, &value, "_VibaData")? {
        return Ok(None);
    }
    let piece = g(&g(&value, "node")?, "data")?;
    if is_ast(py, &piece, "Exponent")? || is_ast(py, &piece, "ExponentChain")? {
        return Ok(Some((piece.unbind(), declared_in, own)));
    }
    Ok(None)
}

fn act_call_target(
    py: Python<'_>,
    act: &Bound<'_, PyAny>,
    name_node: &Bound<'_, PyAny>,
    scope: &[P],
) -> PyResult<Option<P>> {
    if is_ast(py, name_node, "TypeApp")? {
        return act_generic_target(py, act, name_node, scope);
    }
    if is_ast(py, name_node, "MemberTaken")? {
        if let Some(path) = source_path(py, name_node)? {
            let node = make_type_ref(py, &path)?;
            return act_call_target(py, act, node.bind(py), scope);
        }
        return act_application_member_target(py, act, name_node, scope);
    }
    let name: String = g(name_node, "name")?.extract()?;
    if name == DYN_CALL_NAME || name == DYN_METHOD_NAME {
        let kind = if name == DYN_CALL_NAME {
            "call"
        } else {
            "method"
        };
        let made = shim_class(py, "_DynCall")?
            .call1((act, kind, name_node, &name))?
            .unbind();
        return Ok(Some(make_ok(py, made)?));
    }
    let module = act_attr(act, "module")?;
    if let Some(definition) = definition(py, &module, &name)? {
        return act_pending_of(
            py,
            act,
            definition.bind(py),
            name_node,
            &name,
            &module,
            &name,
            "",
            None,
        );
    }
    if let Some((chain, declared_in, own)) = act_member_function(py, act, &name)? {
        let module = act_attr(act, "module")?;
        return Ok(Some(act_func_pending(
            py,
            act,
            name_node,
            &name,
            declared_in.bind(py),
            chain.bind(py),
            &own,
            Some(&module),
        )?));
    }
    let Some((module_name, rest)) = act_imported_name(py, act, &name)? else {
        if let Some(member) = act_builtin_bare_call(py, act, name_node, &name)? {
            return Ok(Some(member));
        }
        return act_builtin_module_call(py, act, name_node, &name);
    };
    let runner = act_attr(act, "runner")?;
    let file = act_attr(act, "file")?;
    let near = file_near(py, &file)?;
    let imported = runner_imported(py, &runner, &module_name, near.as_deref())?;
    if is_stopped(py, imported.bind(py))? {
        return Ok(Some(imported));
    }
    if !rest.is_empty() {
        return act_member_target(
            py,
            act,
            &ok_value(imported.bind(py))?,
            &module_name,
            &rest,
            name_node,
            &name,
            Some(&act_attr(act, "module")?),
        );
    }
    if is_a(
        py,
        &ok_value(imported.bind(py))?,
        PATTERN_MODULE,
        "GenericModuleType",
    )? {
        return Ok(Some(program_err(
            py,
            &format!(
                "'{name}' is a generic: it answers an application ({name}[T, ...]), not a call"
            ),
        )?));
    }
    Ok(Some(pending_module(
        py,
        act,
        &runner,
        &ok_value(imported.bind(py))?,
        &module_name,
        name_node,
        None,
        Some(&act_attr(act, "module")?),
        None,
        None,
    )?))
}

#[allow(clippy::too_many_arguments)]
fn pending_module(
    py: Python<'_>,
    act: &Bound<'_, PyAny>,
    runner: &Bound<'_, PyAny>,
    module: &Bound<'_, PyAny>,
    module_name: &str,
    head: &Bound<'_, PyAny>,
    bindings: Option<&Bound<'_, PyAny>>,
    source_module: Option<&Bound<'_, PyAny>>,
    pattern_env: Option<&str>,
    slots: Option<&Bound<'_, PyAny>>,
) -> PyResult<P> {
    let kwargs = PyDict::new(py);
    kwargs.set_item("kind", "module")?;
    kwargs.set_item("runner", runner)?;
    kwargs.set_item("module", module)?;
    kwargs.set_item("module_name", module_name)?;
    kwargs.set_item("head", head)?;
    kwargs.set_item("source", module_name)?;
    if let Some(source_module) = source_module {
        kwargs.set_item("source_module", source_module)?;
    }
    if let Some(bindings) = bindings {
        kwargs.set_item("bindings", bindings)?;
    }
    if let Some(pattern_env) = pattern_env {
        kwargs.set_item("pattern_env", pattern_env)?;
    }
    if let Some(slots) = slots {
        kwargs.set_item("slots", slots)?;
    }
    let bindings = act_attr(act, "bindings")?;
    kwargs.set_item(
        "source_bindings",
        if bindings.call_method0("__len__")?.extract::<i64>()? > 0 {
            bindings.unbind()
        } else {
            py.None()
        },
    )?;
    make_ok(
        py,
        shim_class(py, "_Pending")?
            .call((), Some(&kwargs))?
            .unbind(),
    )
}

#[allow(clippy::too_many_arguments)]
fn act_member_target(
    py: Python<'_>,
    act: &Bound<'_, PyAny>,
    module: &Bound<'_, PyAny>,
    module_name: &str,
    rest: &str,
    name_node: &Bound<'_, PyAny>,
    source: &str,
    source_module: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<P>> {
    let Some(definition) = definition(py, module, rest)? else {
        return Ok(Some(program_err(
            py,
            &format!("module '{module_name}' has no '{rest}'"),
        )?));
    };
    let runner = act_attr(act, "runner")?;
    let environ = act_attr(act, "environ")?;
    let file = g(&runner, "path_of")?
        .call_method1("get", (module_name,))?
        .unbind();
    let other = new_activation(
        py,
        &runner,
        module,
        &environ,
        module_name,
        file.bind(py),
        None,
        None,
        None,
    )?;
    act_pending_of(
        py,
        other.bind(py),
        definition.bind(py),
        name_node,
        rest,
        module,
        source,
        "",
        source_module,
    )
}

fn act_application_member_target(
    py: Python<'_>,
    act: &Bound<'_, PyAny>,
    node: &Bound<'_, PyAny>,
    scope: &[P],
) -> PyResult<Option<P>> {
    let owner = g(node, "owner")?;
    let (source, modules, parts) = act_application_piece(py, act, &owner, scope)?;
    let module = act_attr(act, "module")?;
    let decision = reduce_application(
        py,
        source.bind(py),
        &module,
        modules.bind(py),
        parts.bind(py),
    )?;
    if is_stopped(py, decision.bind(py))? {
        return Ok(Some(decision));
    }
    let chosen = ok_value(decision.bind(py))?;
    if chosen.is_none() {
        return Ok(None);
    }
    let chosen_module = g(&chosen, "module")?;
    let node_name: String = g(node, "name")?.extract()?;
    let Some(definition) = definition(py, &chosen_module, &node_name)? else {
        let entry = g(&chosen, "entry")?;
        return Ok(Some(program_err(
            py,
            &format!(
                "module '{}' has no '{node_name}'",
                g(&entry, "name")?.extract::<String>()?
            ),
        )?));
    };
    let bindings = act_generic_bindings(py, &chosen)?;
    let body = substituted(py, &g(definition.bind(py), "body")?, &[bindings])?;
    if !is_ast(py, body.bind(py), "Exponent")? && !is_ast(py, body.bind(py), "ExponentChain")? {
        return Ok(None);
    }
    let source_name = format!("{}.{}", one_line(py, &owner)?, node_name);
    let constructor: String = g(&owner, "constructor")?.extract()?;
    let current_module = act_attr(act, "module")?;
    Ok(Some(act_func_pending(
        py,
        act,
        node,
        &source_name,
        &chosen_module,
        body.bind(py),
        &constructor,
        Some(&current_module),
    )?))
}

// ----------------------------------------------------------------------
// Giving arguments, and finishing a chain (the activation's side)
// ----------------------------------------------------------------------

fn act_apply_chain(
    py: Python<'_>,
    act: &Bound<'_, PyAny>,
    node: &Bound<'_, PyAny>,
    scope: &[P],
    finish: bool,
    member: Option<(String, P)>,
) -> PyResult<P> {
    let (target, source) = act_target_and_arguments(py, act, node, scope)?;
    let Some(source) = source else {
        return Ok(target);
    };
    if let Some((name, owner)) = member {
        let target_value = ok_value(target.bind(py))?;
        if is_shim(py, &target_value, "_Pending")? {
            let pair = PyTuple::new(py, [py_string(py, &name), owner])?;
            target_value.setattr("member_of", pair)?;
        }
    }
    let target_value = ok_value(target.bind(py))?;
    let out = act_give_all(py, act, &target_value, &source, scope, finish)?;
    Ok(out)
}

fn act_give_all(
    py: Python<'_>,
    act: &Bound<'_, PyAny>,
    current: &Bound<'_, PyAny>,
    source: &[P],
    scope: &[P],
    finish: bool,
) -> PyResult<P> {
    let mut current = current.clone().unbind();
    let mut index = 0;
    while index < source.len() {
        let argument = source[index].bind(py).clone();
        let member_tag = if is_shim(py, current.bind(py), "_Pending")? {
            pending_member_tag(py, current.bind(py))?
        } else {
            None
        };
        if member_tag.is_some() && pending_ready(py, current.bind(py))? {
            let member = current.bind(py).clone();
            let run = pending_fire(py, &member)?;
            if is_stopped(py, run.bind(py))? {
                return Ok(run);
            }
            current = ok_value(run.bind(py))?.unbind();
            let further: Vec<P> = source[index..]
                .iter()
                .map(|one| one.clone_ref(py))
                .collect();
            if !further.is_empty() && is_a_source_call(py, current.bind(py))? {
                return act_give_the_rest(
                    py,
                    act,
                    &member,
                    current.bind(py),
                    &further,
                    scope,
                    finish,
                );
            }
            index += 1;
            continue;
        }
        if is_a_source_call(py, current.bind(py))? {
            let target = act_stored_call(py, act, current.bind(py), scope, false, finish)?;
            if let Some(target) = target {
                if is_stopped(py, target.bind(py))? {
                    return Ok(target);
                }
                let rest: Vec<P> = source[index..]
                    .iter()
                    .map(|one| one.clone_ref(py))
                    .collect();
                return act_give_all(py, act, &ok_value(target.bind(py))?, &rest, scope, finish);
            }
            let data = g(&g(current.bind(py), "node")?, "data")?.unbind();
            let mut node = data;
            for later in &source[index..] {
                node = make_partial(py, node, later.clone_ref(py))?;
            }
            let source_module = module_in_source(py, current.bind(py))?;
            let other = match &source_module {
                Some(module) => act_in_module(
                    py,
                    act,
                    module.bind(py),
                    bindings_in_source(py, current.bind(py)).as_ref(),
                )?,
                None => None,
            };
            let act_module = act_attr(act, "module")?;
            let same_module = match &source_module {
                Some(module) => module.bind(py).is(&act_module),
                None => true,
            };
            let Some(other) = other.filter(|_| !same_module) else {
                return act_apply_chain(py, act, node.bind(py), scope, true, None);
            };
            let stored = act_stored_call(py, act, current.bind(py), scope, true, finish)?;
            let Some(stored) = stored else {
                return act_apply_chain(py, act, node.bind(py), scope, true, None);
            };
            if is_stopped(py, stored.bind(py))? {
                return Ok(stored);
            }
            return Ok(stored);
        }
        let value = if is_shim(py, &argument, "_Given")? {
            make_ok(py, argument.clone().unbind())?
        } else if act_argument_is_lazy(py, act, current.bind(py), &argument)? {
            act_lazy_argument(py, act, &argument, scope)?
        } else {
            act_argument(py, act, &argument, scope)?
        };
        if is_stopped(py, value.bind(py))? {
            return Ok(value);
        }
        let value_object = ok_value(value.bind(py))?;
        if value_object.is_none() {
            index += 1;
            continue;
        }
        if is_shim(py, current.bind(py), "_Member")? {
            let owner = g(&value_object, "value")?;
            let tag: String = gs(current.bind(py), "tag")?;
            let name = a_tag_text(&tag);
            let as_member = (name, owner.clone().unbind());
            let taken = act_take_member(py, act, current.bind(py), &owner)?;
            if is_stopped(py, taken.bind(py))? {
                return Ok(taken);
            }
            current = ok_value(taken.bind(py))?.unbind();
            if is_a_source_call(py, current.bind(py))? {
                let mut node = g(&g(current.bind(py), "node")?, "data")?.unbind();
                for later in &source[index..] {
                    node = make_partial(py, node, later.clone_ref(py))?;
                }
                return act_apply_chain(py, act, node.bind(py), scope, true, Some(as_member));
            }
            if is_shim(py, current.bind(py), "_HostFunction")?
                && is_shim(py, current.bind(py), "_HostFunction")?
                && host_function_filled(py, current.bind(py))?
            {
                let run = host_function_run(py, current.bind(py))?;
                if is_stopped(py, run.bind(py))? {
                    return Ok(run);
                }
                current = ok_value(run.bind(py))?.unbind();
            }
            index += 1;
            continue;
        }
        if is_shim(py, current.bind(py), "_Pending")? {
            if gs(current.bind(py), "kind")? == "module" {
                let inner = g(&value_object, "value")?;
                let (member, stop) = act_members_as_values(py, act, &inner)?;
                if let Some(stop) = stop {
                    return Ok(stop);
                }
                let tag = g(&value_object, "tag")?;
                let tag_string = if tag.is_none() {
                    None
                } else {
                    Some(tag.extract::<String>()?)
                };
                let given = new_given(py, tag_string.as_deref(), member)?;
                let given_value = g(given.bind(py), "value")?;
                current = pending_give(py, current.bind(py), tag_string.as_deref(), &given_value)?;
            } else {
                let tag = g(&value_object, "tag")?;
                let tag = if tag.is_none() {
                    None
                } else {
                    Some(tag.extract::<String>()?)
                };
                current = pending_give(
                    py,
                    current.bind(py),
                    tag.as_deref(),
                    &g(&value_object, "value")?,
                )?;
            }
        } else if is_shim(py, current.bind(py), "_DynCall")? {
            let tag = g(&value_object, "tag")?;
            let tag = if tag.is_none() {
                None
            } else {
                Some(tag.extract::<String>()?)
            };
            current = dyn_call_give(
                py,
                current.bind(py),
                tag.as_deref(),
                &g(&value_object, "value")?,
            )?;
        } else {
            current = host_give(py, current.bind(py), &value_object)?;
        }
        if is_stopped(py, current.bind(py))? {
            return Ok(current);
        }
        current = ok_value(current.bind(py))?.unbind();
        index += 1;
    }
    if is_shim(py, current.bind(py), "_Pending")? {
        return if finish {
            act_finish(py, current.bind(py))
        } else {
            make_ok(py, current)
        };
    }
    if is_shim(py, current.bind(py), "_DynCall")? {
        return if finish {
            act_finish_dyn(py, act, current.bind(py), None, Vec::new())
        } else {
            make_ok(py, current)
        };
    }
    if is_shim(py, current.bind(py), "_HostFunction")? {
        if host_function_filled(py, current.bind(py))? {
            return host_function_run(py, current.bind(py));
        }
        let name: String = gs(current.bind(py), "name")?;
        let given = g(current.bind(py), "given")?
            .call_method0("__len__")?
            .extract::<i64>()?;
        let slots: i64 = g(current.bind(py), "slots")?.extract()?;
        return program_err(
            py,
            &format!("the environment's {name} was given {given} of its {slots} arguments"),
        );
    }
    if is_shim(py, current.bind(py), "_GetArgs")? {
        return program_err(
            py,
            &format!("{GET_ARGS_NAME} asks for the module's {DEF_NAME}, and none was given"),
        );
    }
    if is_shim(py, current.bind(py), "_IsSubType")? {
        if is_sub_type_filled(py, current.bind(py))? {
            return is_sub_type_run(py, current.bind(py));
        }
        let given = g(current.bind(py), "given")?
            .call_method0("__len__")?
            .extract::<i64>()?;
        return program_err(
            py,
            &format!("{IS_SUB_TYPE_VALUE_NAME} takes two arguments, and {given} were given"),
        );
    }
    make_ok(py, current)
}

fn act_give_the_rest(
    py: Python<'_>,
    act: &Bound<'_, PyAny>,
    member: &Bound<'_, PyAny>,
    answered: &Bound<'_, PyAny>,
    further: &[P],
    scope: &[P],
    finish: bool,
) -> PyResult<P> {
    let mut node = g(&g(answered, "node")?, "data")?.unbind();
    for later in further {
        node = make_partial(py, node, later.clone_ref(py))?;
    }
    let source_module = module_in_source(py, answered)?;
    let elsewhere = match &source_module {
        Some(module) if !module.bind(py).is(&act_attr(act, "module")?) => act_in_module(
            py,
            act,
            module.bind(py),
            bindings_in_source(py, answered).as_ref(),
        )?,
        _ => None,
    };
    let holder = match &elsewhere {
        Some(other) => other.bind(py).clone(),
        None => act.clone(),
    };
    let (target, arguments) = act_target_and_arguments(py, &holder, node.bind(py), scope)?;
    let Some(mut source) = arguments else {
        return Ok(target);
    };
    if is_stopped(py, target.bind(py))? {
        return Ok(target);
    }
    let target = ok_value(target.bind(py))?.unbind();
    let member_tag: Option<String> = g(member, "member_tag")?.extract().unwrap_or(None);
    if member_tag.as_deref() == Some(GET_ATTR_TAG) {
        let named = member_tag_of(py, &pending_member_given(py, member, "$name")?)?;
        if is_stopped(py, named.bind(py))? {
            return Ok(named);
        }
        let owner = pending_member_given(py, member, "$value")?;
        if is_shim(py, target.bind(py), "_Pending")? {
            let tag: String = ok_value(named.bind(py))?.extract()?;
            let pair = PyTuple::new(
                py,
                [py_string(py, &a_tag_text(&tag)), owner.clone().unbind()],
            )?;
            target.bind(py).setattr("member_of", pair)?;
        }
        source.insert(0, new_given(py, None, owner.unbind())?);
    }
    let environ = g(member, "environ")?;
    let environ = if is_shim(py, &environ, "Environment")? {
        new_host(py, environ.unbind())?
    } else {
        environ.unbind()
    };
    source.push(new_given(py, Some(ENVIRON_TAG), environ)?);
    act_give_all(py, act, target.bind(py), &source, scope, finish)
}

fn act_finish(py: Python<'_>, pending: &Bound<'_, PyAny>) -> PyResult<P> {
    let environ = g(pending, "environ")?;
    if environ.is_none() {
        return pending_as_viba_data(py, pending);
    }
    if let Some(missing) = pending_missing(py, pending)? {
        return program_err(py, &missing);
    }
    pending_fire(py, pending)
}

fn act_finish_dyn(
    py: Python<'_>,
    act: &Bound<'_, PyAny>,
    target: &Bound<'_, PyAny>,
    environ: Option<&Bound<'_, PyAny>>,
    arguments: Vec<P>,
) -> PyResult<P> {
    let mut target = target.clone().unbind();
    let target_environ = g(target.bind(py), "environ")?;
    if target_environ.is_none() {
        if let Some(environ) = environ {
            let handed = if is_environ_value(py, environ)? {
                environ.clone().unbind()
            } else {
                new_host(py, environ.clone().unbind())?
            };
            let given = dyn_call_give(py, target.bind(py), Some(ENVIRON_TAG), handed.bind(py))?;
            if is_stopped(py, given.bind(py))? {
                return Ok(given);
            }
            target = ok_value(given.bind(py))?.unbind();
        }
    }
    for argument in arguments {
        let value = given_value(py, argument.bind(py))?;
        let given = dyn_call_give(py, target.bind(py), None, value.bind(py))?;
        if is_stopped(py, given.bind(py))? {
            return Ok(given);
        }
        target = ok_value(given.bind(py))?.unbind();
    }
    if g(target.bind(py), "environ")?.is_none() {
        return make_ok(py, dyn_call_as_viba_data(py, target.bind(py))?);
    }
    if let Some(missing) = dyn_call_missing(py, target.bind(py))? {
        return program_err(py, &missing);
    }
    dyn_call_fire(py, target.bind(py))
}

fn act_apply_without_environ(
    py: Python<'_>,
    act: &Bound<'_, PyAny>,
    node: &Bound<'_, PyAny>,
    scope: &[P],
    environ: &Bound<'_, PyAny>,
    arguments: Vec<P>,
) -> PyResult<P> {
    let (target, source) = act_target_and_arguments(py, act, node, scope)?;
    let Some(source) = source else {
        return Ok(target);
    };
    let given = act_give_all(py, act, &ok_value(target.bind(py))?, &source, scope, false)?;
    if is_stopped(py, given.bind(py))? {
        return Ok(given);
    }
    let mut current = ok_value(given.bind(py))?.unbind();
    if is_shim(py, current.bind(py), "_DynCall")? {
        return act_finish_dyn(py, act, current.bind(py), Some(environ), arguments);
    }
    if !is_shim(py, current.bind(py), "_Pending")? {
        if !arguments.is_empty() {
            return program_err(
                py,
                "the host gave this argument arguments, but it is already the answer",
            );
        }
        return Ok(current);
    }
    if g(current.bind(py), "environ")?.is_none() {
        let handed = if is_environ_value(py, environ)? {
            environ.clone().unbind()
        } else {
            new_host(py, environ.clone().unbind())?
        };
        let given = pending_give(py, current.bind(py), Some(ENVIRON_TAG), handed.bind(py))?;
        if is_stopped(py, given.bind(py))? {
            return Ok(given);
        }
        current = ok_value(given.bind(py))?.unbind();
    }
    for argument in arguments {
        let value = given_value(py, argument.bind(py))?;
        let given = pending_give(py, current.bind(py), None, value.bind(py))?;
        if is_stopped(py, given.bind(py))? {
            return Ok(given);
        }
        current = ok_value(given.bind(py))?.unbind();
    }
    act_finish(py, current.bind(py))
}

fn act_target_and_arguments(
    py: Python<'_>,
    act: &Bound<'_, PyAny>,
    node: &Bound<'_, PyAny>,
    scope: &[P],
) -> PyResult<(P, Option<Vec<P>>)> {
    let (mut head, mut arguments) = call_parts(py, node)?;
    loop {
        if is_ast(py, head.bind(py), "Member")? {
            let tag: String = gs(head.bind(py), "tag")?;
            if interpreter_member_tags().contains(&tag.as_str()) {
                let pending = act_member_pending(py, act, head.bind(py), node)?;
                return Ok((pending, Some(arguments)));
            }
            let member = new_member(py, &tag)?;
            return Ok((make_ok(py, member)?, Some(arguments)));
        }
        if is_ast(py, head.bind(py), "TypeRef")?
            || is_ast(py, head.bind(py), "TypeApp")?
            || is_ast(py, head.bind(py), "MemberTaken")?
        {
            if let Some(target) = act_call_target(py, act, head.bind(py), scope)? {
                if is_stopped(py, target.bind(py))? {
                    return Ok((target, None));
                }
                return Ok((target, Some(arguments)));
            }
        }
        let value = act_evaluate(py, act, head.bind(py), scope, true)?;
        if is_stopped(py, value.bind(py))? {
            return Ok((value, None));
        }
        let got = ok_value(value.bind(py))?.unbind();
        if is_shim(py, got.bind(py), "_VibaData")? && !g(got.bind(py), "given")?.is_none() {
            let source_module = module_in_source(py, got.bind(py))?;
            let (stored_head, passed) = stored_arguments(py, got.bind(py))?;
            if let Some(source_module) = &source_module {
                if !source_module.bind(py).is(&act_attr(act, "module")?) {
                    let other = act_in_module(
                        py,
                        act,
                        source_module.bind(py),
                        bindings_in_source(py, got.bind(py)).as_ref(),
                    )?;
                    let is_name = is_ast(py, stored_head.bind(py), "TypeRef")?
                        || is_ast(py, stored_head.bind(py), "TypeApp")?
                        || is_ast(py, stored_head.bind(py), "MemberTaken")?;
                    let target = match &other {
                        Some(other) if is_name => {
                            act_call_target(py, other.bind(py), stored_head.bind(py), scope)?
                        }
                        _ => None,
                    };
                    if let Some(target) = target {
                        if is_stopped(py, target.bind(py))? {
                            return Ok((target, None));
                        }
                        let mut all = passed;
                        all.extend(arguments);
                        return Ok((target, Some(all)));
                    }
                }
            }
            head = stored_head;
            let mut all = passed;
            all.extend(arguments);
            arguments = all;
            continue;
        }
        if is_shim(py, got.bind(py), "_VibaData")?
            && is_ast(py, &g(got.bind(py), "node")?.getattr("data")?, "Partial")?
        {
            let (stored_head, stored) = call_parts(py, &g(got.bind(py), "node")?.getattr("data")?)?;
            head = stored_head;
            let mut all = stored;
            all.extend(arguments);
            arguments = all;
            continue;
        }
        if is_shim(py, got.bind(py), "_VibaData")?
            && is_ast(py, &g(got.bind(py), "node")?.getattr("data")?, "Member")?
        {
            let data = g(got.bind(py), "node")?.getattr("data")?;
            let tag: String = gs(&data, "tag")?;
            if interpreter_member_tags().contains(&tag.as_str()) {
                let pending = act_member_pending(py, act, &data, node)?;
                return Ok((pending, Some(arguments)));
            }
            let member = new_member(py, &tag)?;
            return Ok((make_ok(py, member)?, Some(arguments)));
        }
        if is_shim(py, got.bind(py), "_VibaData")? {
            let data = g(got.bind(py), "node")?.getattr("data")?;
            if is_ast(py, &data, "TypeRef")?
                || is_ast(py, &data, "TypeApp")?
                || is_ast(py, &data, "MemberTaken")?
            {
                let source_module = module_in_source(py, got.bind(py))?;
                if let Some(source_module) = &source_module {
                    if !source_module.bind(py).is(&act_attr(act, "module")?) {
                        let other = act_in_module(
                            py,
                            act,
                            source_module.bind(py),
                            bindings_in_source(py, got.bind(py)).as_ref(),
                        )?;
                        if let Some(other) = &other {
                            if let Some(target) = act_call_target(py, other.bind(py), &data, scope)?
                            {
                                if is_stopped(py, target.bind(py))? {
                                    return Ok((target, None));
                                }
                                return Ok((target, Some(arguments)));
                            }
                        }
                    }
                }
                head = data.unbind();
                continue;
            }
        }
        return Ok((make_ok(py, got)?, Some(arguments)));
    }
}

fn act_stored_call(
    py: Python<'_>,
    act: &Bound<'_, PyAny>,
    value: &Bound<'_, PyAny>,
    scope: &[P],
    whole: bool,
    finish: bool,
) -> PyResult<Option<P>> {
    if g(value, "given")?.is_none() {
        return Ok(None);
    }
    let (head, passed) = stored_arguments(py, value)?;
    let source_module = module_in_source(py, value)?;
    let different = match &source_module {
        Some(module) => !module.bind(py).is(&act_attr(act, "module")?),
        None => false,
    };
    if whole || different {
        let node = g(value, "node")?;
        let function = g(&node, "function")?;
        if !function.is_none() {
            if let Some(defining) = act_in_module(
                py,
                act,
                &g(&function, "module")?,
                bindings_in_source(py, value).as_ref(),
            )? {
                let name_node = make_type_ref(py, &gs(&function, "name")?)?;
                let target = act_call_target(py, defining.bind(py), name_node.bind(py), scope)?;
                if let Some(target) = target {
                    if !is_stopped(py, target.bind(py))? {
                        return Ok(Some(act_give_all(
                            py,
                            defining.bind(py),
                            &ok_value(target.bind(py))?,
                            &passed,
                            scope,
                            finish,
                        )?));
                    }
                }
            }
        }
        if let Some(source_module) = &source_module {
            let other = act_in_module(
                py,
                act,
                source_module.bind(py),
                bindings_in_source(py, value).as_ref(),
            )?;
            let is_name =
                is_ast(py, head.bind(py), "TypeRef")? || is_ast(py, head.bind(py), "TypeApp")?;
            if let (Some(other), true) = (other, is_name) {
                let target = act_call_target(py, other.bind(py), head.bind(py), scope)?;
                let Some(target) = target else {
                    return Ok(None);
                };
                if is_stopped(py, target.bind(py))? {
                    return Ok(Some(target));
                }
                return Ok(Some(act_give_all(
                    py,
                    other.bind(py),
                    &ok_value(target.bind(py))?,
                    &passed,
                    scope,
                    finish,
                )?));
            }
        }
    }
    Ok(None)
}

fn act_in_module(
    py: Python<'_>,
    act: &Bound<'_, PyAny>,
    module: &Bound<'_, PyAny>,
    bindings: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<P>> {
    let runner = act_attr(act, "runner")?;
    let by_name = g(&runner, "by_name")?;
    for item in by_name.call_method0("items")?.try_iter()? {
        let item = item?;
        let known = item.get_item(1)?;
        if known.is(module) {
            let name: String = item.get_item(0)?.extract()?;
            let file = g(&runner, "path_of")?
                .call_method1("get", (&name,))?
                .unbind();
            return Ok(Some(new_activation(
                py,
                &runner,
                module,
                &act_attr(act, "environ")?,
                &name,
                file.bind(py),
                None,
                None,
                bindings,
            )?));
        }
    }
    Ok(None)
}

// ----------------------------------------------------------------------
// _Pending: a call not yet given an environment
// ----------------------------------------------------------------------

fn pending_member_tag(py: Python<'_>, pending: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
    let tag = g(pending, "member_tag")?;
    if tag.is_none() {
        return Ok(None);
    }
    Ok(Some(tag.extract()?))
}

fn pending_params<'py>(
    py: Python<'py>,
    pending: &Bound<'py, PyAny>,
) -> PyResult<(Option<Vec<Bound<'py, PyAny>>>, Option<String>)> {
    let override_ = g(pending, "params_override")?;
    if !override_.is_none() {
        let mut params = Vec::new();
        for param in override_.try_iter()? {
            params.push(param?);
        }
        return Ok((Some(params), None));
    }
    module_params(py, &g(pending, "module")?)
}

fn pending_arg_slots(
    py: Python<'_>,
    pending: &Bound<'_, PyAny>,
) -> PyResult<(Option<Vec<(Option<String>, P)>>, Option<String>)> {
    let override_ = g(pending, "params_override")?;
    if override_.is_none() {
        return module_arg_slots(py, &g(pending, "module")?);
    }
    let mut slots = Vec::new();
    for param in override_.try_iter()? {
        let param = param?;
        let is_env: bool = param.get_item(2)?.extract()?;
        if is_env {
            continue;
        }
        let tag = param.get_item(0)?;
        let tag = if tag.is_none() {
            None
        } else {
            Some(tag.extract::<String>()?)
        };
        slots.push((tag, param.get_item(1)?.unbind()));
    }
    Ok((Some(slots), None))
}

fn pending_takes_environ(py: Python<'_>, pending: &Bound<'_, PyAny>) -> PyResult<bool> {
    let environ = g(pending, "environ")?;
    if !environ.is_none() {
        return Ok(false);
    }
    if gs(pending, "kind")? == "module" {
        return Ok(true);
    }
    let slot = pending_environ_slot(py, pending)?;
    match slot {
        Some(slot) => Ok(!pending_given_contains(py, pending, slot)?),
        None => Ok(false),
    }
}

fn pending_declares_environ(py: Python<'_>, pending: &Bound<'_, PyAny>) -> PyResult<bool> {
    let (params, _problem) = pending_params(py, pending)?;
    for param in params.unwrap_or_default() {
        if param.get_item(2)?.extract::<bool>()? {
            return Ok(true);
        }
    }
    Ok(false)
}

fn ends_with_name(py: Python<'_>, environ: &Bound<'_, PyAny>, name: &str) -> PyResult<bool> {
    let path = storage_path(py, environ)?;
    Ok(path == name || path.ends_with(&format!("/{name}")))
}

fn pending_the_env_name(
    py: Python<'_>,
    pending: &Bound<'_, PyAny>,
    parent: &Bound<'_, PyAny>,
) -> PyResult<String> {
    let pattern_env = g(pending, "pattern_env")?;
    let source = gs(pending, "source")?;
    if !pattern_env.is_none() && ends_with_name(py, parent, &source)? {
        return pattern_env.extract();
    }
    Ok(source)
}

fn pending_slots(py: Python<'_>, pending: &Bound<'_, PyAny>) -> PyResult<Vec<Option<String>>> {
    let mut out = Vec::new();
    for element in g(pending, "elements")?.try_iter()? {
        let element = element?;
        if is_ast(py, &element, "Tagged")? {
            out.push(Some(gs(&element, "tag")?));
        } else {
            out.push(None);
        }
    }
    Ok(out)
}

fn pending_slot_tags(py: Python<'_>, pending: &Bound<'_, PyAny>) -> PyResult<Vec<Option<String>>> {
    if gs(pending, "kind")? == "module" {
        let (slots, _problem) = pending_arg_slots(py, pending)?;
        return Ok(slots
            .unwrap_or_default()
            .into_iter()
            .map(|(tag, _source)| tag)
            .collect());
    }
    pending_slots(py, pending)
}

fn pending_environ_slot(py: Python<'_>, pending: &Bound<'_, PyAny>) -> PyResult<Option<usize>> {
    if gs(pending, "kind")? != "func" {
        return Ok(None);
    }
    let elements = g(pending, "elements")?;
    for (index, element) in elements.try_iter()?.enumerate() {
        let element = element?;
        if is_ast(py, &element, "Tagged")? && gs(&element, "tag")? == ENVIRON_TAG {
            return Ok(Some(index));
        }
    }
    Ok(None)
}

fn pending_given_contains(
    py: Python<'_>,
    pending: &Bound<'_, PyAny>,
    index: usize,
) -> PyResult<bool> {
    Ok(g(pending, "given")?
        .call_method1("__contains__", (index,))?
        .extract()?)
}

fn pending_given_get<'py>(
    py: Python<'py>,
    pending: &Bound<'py, PyAny>,
    index: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let given = g(pending, "given")?;
    given.call_method1("get", (index,))
}

fn pending_index_for(
    py: Python<'_>,
    pending: &Bound<'_, PyAny>,
    node: &Bound<'_, PyAny>,
) -> PyResult<Option<usize>> {
    let (tag, _inner) = addressed(py, node)?;
    let (index, _problem) = pending_slot_for(py, pending, tag.as_deref())?;
    Ok(index)
}

fn pending_slot_for(
    py: Python<'_>,
    pending: &Bound<'_, PyAny>,
    tag: Option<&str>,
) -> PyResult<(Option<usize>, Option<String>)> {
    let slots = pending_slots(py, pending)?;
    let given = g(pending, "given")?;
    let given_count = given.call_method0("__len__")?.extract::<i64>()? as usize;
    let source: String = gs(pending, "source")?;
    if given_count == slots.len() {
        return Ok((None, Some(format!("{source} takes no more arguments"))));
    }
    if let Some(tag) = tag {
        let position = slots.iter().position(|slot| slot.as_deref() == Some(tag));
        if let Some(index) = position {
            if pending_given_contains(py, pending, index)? {
                return Ok((None, Some(format!("{source} was given {tag} twice"))));
            }
            return Ok((Some(index), None));
        }
        let mut free = Vec::new();
        for index in 0..slots.len() {
            if !pending_given_contains(py, pending, index)? {
                let source = pending_slot_in_source(py, pending, index)?;
                if takes_a_product(py, source.bind(py))? {
                    free.push(index);
                }
            }
        }
        if let Some(index) = free.first() {
            return Ok((Some(*index), None));
        }
        return Ok((None, Some(format!("{source} takes no {tag} argument"))));
    }
    let mut free = Vec::new();
    for index in 0..slots.len() {
        if !pending_given_contains(py, pending, index)? {
            free.push(index);
        }
    }
    match free.first() {
        Some(index) => Ok((Some(*index), None)),
        None => Ok((None, Some(format!("{source} takes no more arguments")))),
    }
}

fn pending_slot_in_source(py: Python<'_>, pending: &Bound<'_, PyAny>, index: usize) -> PyResult<P> {
    if gs(pending, "kind")? == "module" {
        let (slots, _problem) = pending_arg_slots(py, pending)?;
        let slots = slots.unwrap_or_default();
        if index < slots.len() {
            return Ok(slots[index].1.clone_ref(py));
        }
        return Ok(py.None());
    }
    let elements = g(pending, "elements")?;
    let elements: Vec<Bound<'_, PyAny>> = elements.try_iter()?.collect::<PyResult<Vec<_>>>()?;
    if index < elements.len() {
        return Ok(elements[index].clone().unbind());
    }
    Ok(py.None())
}

fn pending_ready(py: Python<'_>, pending: &Bound<'_, PyAny>) -> PyResult<bool> {
    if g(pending, "environ")?.is_none() {
        return Ok(false);
    }
    let given_count = g(pending, "given")?
        .call_method0("__len__")?
        .extract::<i64>()? as usize;
    if gs(pending, "kind")? == "module" {
        let (slots, _problem) = pending_arg_slots(py, pending)?;
        return Ok(given_count == slots.unwrap_or_default().len());
    }
    Ok(given_count
        == g(pending, "elements")?
            .call_method0("__len__")?
            .extract::<i64>()? as usize)
}

fn slot_name(index: usize, tag: Option<&str>) -> String {
    match tag {
        Some(tag) => tag.to_string(),
        None => format!("#{}", index + 1),
    }
}

fn source_slots(slots: &[(Option<String>, P)]) -> String {
    slots
        .iter()
        .enumerate()
        .map(|(index, (tag, _source))| slot_name(index, tag.as_deref()))
        .collect::<Vec<_>>()
        .join(", ")
}

fn pending_missing(py: Python<'_>, pending: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
    if gs(pending, "kind")? == "module" {
        return pending_module_missing(py, pending);
    }
    let source: String = gs(pending, "source")?;
    if pending_member_tag(py, pending)?.is_some() {
        if let Some(wanted) = pending_member_missing(py, pending)? {
            return Ok(Some(format!("{source}: {wanted}")));
        }
    }
    let slots = pending_slots(py, pending)?;
    let mut left = Vec::new();
    for index in 0..slots.len() {
        if !pending_given_contains(py, pending, index)? {
            left.push(slot_name(index, slots[index].as_deref()));
        }
    }
    if left.is_empty() {
        return Ok(None);
    }
    let given = g(pending, "given")?
        .call_method0("__len__")?
        .extract::<i64>()?;
    Ok(Some(format!(
        "{source}: was given {given} of its {} arguments: {} missing",
        slots.len(),
        left.join(", ")
    )))
}

fn pending_member_missing(py: Python<'_>, pending: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
    let Some(tag) = pending_member_tag(py, pending)? else {
        return Ok(None);
    };
    if tag == GET_ATTR_TAG {
        if pending_member_given_or_none(py, pending, "$value")?.is_none() {
            return Ok(None);
        }
        if pending_member_given_or_none(py, pending, "$name")?.is_none() {
            return Ok(Some(
                "a member is taken by a name, and none was given".to_string(),
            ));
        }
        return Ok(None);
    }
    if tag == FIRST_CASE_INDEX_TAG {
        if pending_member_given_or_none(py, pending, "$args")?.is_none() {
            return Ok(Some(
                "a case index asks for the arguments, and none were given".to_string(),
            ));
        }
        if pending_member_given_or_none(py, pending, "$func_arg_types")?.is_none() {
            return Ok(Some(
                "a case index asks for the list of argument types, and none was given".to_string(),
            ));
        }
        return Ok(None);
    }
    if tag == GET_ITEM_TAG || tag == IN_TAG || tag == LEN_TAG {
        if pending_member_given_or_none(py, pending, "$container")?.is_none() {
            return Ok(None);
        }
    } else if pending_member_given_or_none(py, pending, "$table")?.is_none() {
        return Ok(None);
    }
    if tag == GET_ITEM_TAG && pending_member_given_or_none(py, pending, "$address")?.is_none() {
        return Ok(Some(
            "an element is taken by an address — an int or a str — and none was given".to_string(),
        ));
    }
    if tag == IN_TAG && pending_member_given_or_none(py, pending, "$piece")?.is_none() {
        return Ok(Some(
            "a container is asked about a piece, and none was given".to_string(),
        ));
    }
    Ok(None)
}

fn pending_module_missing(py: Python<'_>, pending: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
    let (slots, problem) = pending_arg_slots(py, pending)?;
    if let Some(problem) = problem {
        let module_name: String = gs(pending, "module_name")?;
        return Ok(Some(format!("module '{module_name}': {problem}")));
    }
    let slots = slots.unwrap_or_default();
    if slots.is_empty() {
        return Ok(None);
    }
    let mut left = Vec::new();
    for (index, (tag, _source)) in slots.iter().enumerate() {
        if !pending_given_contains(py, pending, index)? {
            left.push(slot_name(index, tag.as_deref()));
        }
    }
    if left.is_empty() {
        return Ok(None);
    }
    let module_name: String = gs(pending, "module_name")?;
    let given = g(pending, "given")?
        .call_method0("__len__")?
        .extract::<i64>()?;
    Ok(Some(format!(
        "module '{module_name}' was given {given} of its {} parameters: {} missing",
        slots.len(),
        left.join(", ")
    )))
}

fn pending_give(
    py: Python<'_>,
    pending: &Bound<'_, PyAny>,
    tag: Option<&str>,
    value: &Bound<'_, PyAny>,
) -> PyResult<P> {
    if gs(pending, "kind")? == "module" {
        return pending_give_module(py, pending, tag, value);
    }
    pending_give_func(py, pending, tag, value)
}

fn pending_member_given_or_none<'py>(
    py: Python<'py>,
    pending: &Bound<'py, PyAny>,
    tag: &str,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let slots = pending_slot_tags(py, pending)?;
    let Some(index) = slots.iter().position(|slot| slot.as_deref() == Some(tag)) else {
        return Ok(None);
    };
    let value = pending_given_get(py, pending, index)?;
    if value.is_none() {
        return Ok(None);
    }
    Ok(Some(value))
}

fn pending_member_given<'py>(
    py: Python<'py>,
    pending: &Bound<'py, PyAny>,
    tag: &str,
) -> PyResult<Bound<'py, PyAny>> {
    match pending_member_given_or_none(py, pending, tag)? {
        Some(value) => Ok(value),
        None => Ok(py.None().into_bound(py)),
    }
}

fn pending_give_func(
    py: Python<'_>,
    pending: &Bound<'_, PyAny>,
    tag: Option<&str>,
    value: &Bound<'_, PyAny>,
) -> PyResult<P> {
    let environ = g(pending, "environ")?;
    let source: String = gs(pending, "source")?;
    if !environ.is_none()
        && (tag == Some(ENVIRON_TAG) || (tag.is_none() && is_environ_value(py, value)?))
    {
        return program_err(py, &format!("{source} was given {ENVIRON_TAG} twice"));
    }
    let member_tag = pending_member_tag(py, pending)?;
    let (index, problem) =
        if member_tag.is_some() && !is_environ_value(py, value)? && tag != Some(ENVIRON_TAG) {
            pending_member_slot_for(py, pending, tag)?
        } else {
            pending_slot_for(py, pending, tag)?
        };
    let Some(index) = index else {
        return program_err(py, &problem.unwrap_or_default());
    };
    if Some(index) == pending_environ_slot(py, pending)? {
        if !is_environ_value(py, value)? && !(member_tag.is_some() && is_nil(py, value)?) {
            return program_err(py, &format!("{source} was not given an {ENVIRON_TYPE}"));
        }
        let given = g(pending, "given")?;
        given.set_item(index, value)?;
        let given_tags = g(pending, "given_tags")?;
        given_tags.set_item(
            index,
            match tag {
                Some(tag) => py_string(py, tag),
                None => py.None(),
            },
        )?;
        let environ_value = if is_environ_value(py, value)? {
            g(value, "obj")?.unbind()
        } else {
            value.clone().unbind()
        };
        pending.setattr("environ", environ_value)?;
        return make_ok(py, pending.clone().unbind());
    }
    if function_slot(py, pending, index)?.is_some() && is_shim(py, value, "_Getter")? {
        let origin = g(value, "origin")?;
        let made = shim_class(py, "_HostArgument")?.call1((
            g(value, "node")?,
            g(value, "scope")?,
            pending,
            g(pending, "module")?,
            g(pending, "source")?,
            g(value, "file")?,
            g(value, "source_module")?,
            g(value, "name")?,
            if origin.is_none() {
                py.None()
            } else {
                origin.unbind()
            },
        ))?;
        g(pending, "given")?.set_item(index, &made)?;
        g(pending, "given_tags")?.set_item(
            index,
            match tag {
                Some(tag) => py_string(py, tag),
                None => py.None(),
            },
        )?;
        return make_ok(py, pending.clone().unbind());
    }
    g(pending, "given")?.set_item(index, value)?;
    g(pending, "given_tags")?.set_item(
        index,
        match tag {
            Some(tag) => py_string(py, tag),
            None => py.None(),
        },
    )?;
    make_ok(py, pending.clone().unbind())
}

fn pending_member_slot_for(
    py: Python<'_>,
    pending: &Bound<'_, PyAny>,
    tag: Option<&str>,
) -> PyResult<(Option<usize>, Option<String>)> {
    let slots = pending_slots(py, pending)?;
    let source: String = gs(pending, "source")?;
    let environ = pending_environ_slot(py, pending)?;
    let mut own = Vec::new();
    for index in 0..slots.len() {
        if Some(index) != environ && !pending_given_contains(py, pending, index)? {
            own.push(index);
        }
    }
    if own.is_empty() {
        return Ok((None, Some(format!("{source} takes no more arguments"))));
    }
    if let Some(tag) = tag {
        if let Some(index) = slots.iter().position(|slot| slot.as_deref() == Some(tag)) {
            if Some(index) == environ {
                return Ok((Some(index), None));
            }
            if pending_given_contains(py, pending, index)? {
                return Ok((None, Some(format!("{source} was given {tag} twice"))));
            }
            return Ok((Some(index), None));
        }
    }
    Ok((Some(own[0]), None))
}

fn pending_the_parameter_taking_the_rest(
    py: Python<'_>,
    pending: &Bound<'_, PyAny>,
    slots: &[(Option<String>, P)],
) -> PyResult<Option<usize>> {
    for index in 0..slots.len() {
        if takes_the_rest(py, slots[index].1.bind(py))?
            && pending_given_contains(py, pending, index)?
        {
            return Ok(Some(index));
        }
    }
    Ok(None)
}

fn pending_give_module(
    py: Python<'_>,
    pending: &Bound<'_, PyAny>,
    tag: Option<&str>,
    value: &Bound<'_, PyAny>,
) -> PyResult<P> {
    let module_name: String = gs(pending, "module_name")?;
    let (slots, problem) = pending_arg_slots(py, pending)?;
    if let Some(problem) = problem {
        return program_err(py, &format!("module '{module_name}': {problem}"));
    }
    let slots = slots.unwrap_or_default();
    let environ = g(pending, "environ")?;
    if environ.is_none() && (tag == Some(ENVIRON_TAG) || is_environ_value(py, value)?) {
        if !is_environ_value(py, value)? {
            return program_err(
                py,
                &format!(
                    "module '{module_name}' was not given an {ENVIRON_TYPE}: its {ENVIRON_TAG} \
                     parameter asks for {ENV_TYPE}"
                ),
            );
        }
        let obj = g(value, "obj")?;
        if pending_declares_environ(py, pending)? {
            let new_environ = if !g(pending, "pattern_env")?.is_none() {
                let name = pending_the_env_name(py, pending, &obj)?;
                sub_env(py, &obj, &py_string(py, &name).into_bound(py))?
            } else {
                obj.unbind()
            };
            pending.setattr("environ", new_environ)?;
            return make_ok(py, pending.clone().unbind());
        }
        let name = pending_the_env_name(py, pending, &obj)?;
        let new_environ = sub_env(py, &obj, &py_string(py, &name).into_bound(py))?;
        pending.setattr("environ", new_environ)?;
        pending.setattr("appends_environ", true)?;
        return make_ok(py, pending.clone().unbind());
    }
    if slots.is_empty() {
        return program_err(
            py,
            &format!(
                "module '{module_name}' needs an {ENVIRON_TYPE}: its {DEF_NAME} takes no other \
                 parameters"
            ),
        );
    }
    let tags: Vec<Option<&str>> = slots.iter().map(|(tag, _source)| tag.as_deref()).collect();
    let mut own_tag: Option<String> = None;
    let mut index: Option<usize> = None;
    if let Some(tag) = tag {
        let position = tags.iter().position(|slot| *slot == Some(tag));
        if let Some(position) = position {
            if pending_given_contains(py, pending, position)? {
                return program_err(py, &format!("module '{module_name}' was given {tag} twice"));
            }
            index = Some(position);
        } else {
            let mut free = Vec::new();
            for one in 0..slots.len() {
                if !pending_given_contains(py, pending, one)?
                    && takes_a_product(py, slots[one].1.bind(py))?
                {
                    free.push(one);
                }
            }
            if let Some(free) = free.first() {
                index = Some(*free);
                own_tag = Some(tag.to_string());
            } else if let Some(rest) = pending_the_parameter_taking_the_rest(py, pending, &slots)? {
                return pending_give_to_the_rest(py, pending, rest, Some(tag), value);
            } else {
                return program_err(
                    py,
                    &format!(
                        "module '{module_name}' takes no {tag} argument: its {DEF_NAME} \
                         parameters are {}",
                        source_slots(&slots)
                    ),
                );
            }
        }
    } else {
        let mut free = Vec::new();
        for one in 0..slots.len() {
            if !pending_given_contains(py, pending, one)? {
                free.push(one);
            }
        }
        if let Some(free) = free.first() {
            index = Some(*free);
        } else if let Some(rest) = pending_the_parameter_taking_the_rest(py, pending, &slots)? {
            return pending_give_to_the_rest(py, pending, rest, None, value);
        } else {
            return program_err(
                py,
                &format!(
                    "module '{module_name}' takes no more arguments: its {DEF_NAME} parameters \
                     are all given"
                ),
            );
        }
    }
    let index = index.unwrap();
    let kept = kept_as_its_own_tag(py, own_tag.as_deref(), value)?;
    g(pending, "given")?.set_item(index, kept)?;
    make_ok(py, pending.clone().unbind())
}

fn pending_give_to_the_rest(
    py: Python<'_>,
    pending: &Bound<'_, PyAny>,
    index: usize,
    tag: Option<&str>,
    value: &Bound<'_, PyAny>,
) -> PyResult<P> {
    let product = pending_given_get(py, pending, index)?;
    let source_module = g(pending, "source_module")?;
    let (made, problem) = one_more_member(py, &product, tag, value, &source_module)?;
    if let Some(problem) = problem {
        return Ok(problem);
    }
    g(pending, "given")?.set_item(index, made.unwrap())?;
    make_ok(py, pending.clone().unbind())
}

fn kept_as_its_own_tag(py: Python<'_>, tag: Option<&str>, value: &Bound<'_, PyAny>) -> PyResult<P> {
    let Some(tag) = tag else {
        return Ok(value.clone().unbind());
    };
    if !is_shim(py, value, "_VibaData")? {
        return Ok(value.clone().unbind());
    }
    let node = make_tagged(py, tag, g(&g(value, "node")?, "data")?.unbind())?;
    let source_module = g(value, "source_module")?;
    let descriptor = descriptor_of_tagged(
        py,
        tag,
        &g(&g(value, "node")?, "descriptor")?,
        node.bind(py),
        &source_module,
    )?;
    let function = g(&g(value, "node")?, "function")?;
    let object = make_vobject(
        py,
        descriptor.bind(py),
        node.bind(py),
        None,
        if function.is_none() {
            None
        } else {
            Some(&function)
        },
    )?;
    let made = make_viba_data(py, object)?;
    let made_bound = made.bind(py);
    if !source_module.is_none() {
        made_bound.setattr("source_module", source_module)?;
    }
    let given = g(value, "given")?;
    if !given.is_none() {
        made_bound.setattr("given", given)?;
    }
    let bindings = g(value, "bindings")?;
    if !bindings.is_none() {
        made_bound.setattr("bindings", bindings)?;
    }
    Ok(made)
}

fn pending_fire(py: Python<'_>, pending: &Bound<'_, PyAny>) -> PyResult<P> {
    let out = if pending_member_tag(py, pending)?.is_some() {
        pending_answer_member(py, pending)
    } else if gs(pending, "kind")? == "module" {
        pending_run_module_call(py, pending)
    } else {
        pending_call_host(py, pending)
    }?;
    Ok(out)
}

fn pending_answer_member(py: Python<'_>, pending: &Bound<'_, PyAny>) -> PyResult<P> {
    let activation = g(pending, "activation")?;
    let tag = pending_member_tag(py, pending)?.unwrap_or_default();
    if tag == LEN_TAG {
        return act_container_len(
            py,
            &activation,
            &pending_member_given(py, pending, "$container")?,
        );
    }
    if tag == KEYS_TAG {
        return act_container_keys(
            py,
            &activation,
            &pending_member_given(py, pending, "$table")?,
        );
    }
    if tag == IN_TAG {
        return act_in_container(
            py,
            &activation,
            &pending_member_given(py, pending, "$container")?,
            &pending_member_given(py, pending, "$piece")?,
        );
    }
    if tag == GET_ITEM_TAG {
        return act_take_item(
            py,
            &activation,
            &pending_member_given(py, pending, "$container")?,
            &pending_member_given(py, pending, "$address")?,
        );
    }
    if tag == FIRST_CASE_INDEX_TAG {
        return act_first_case_index(
            py,
            &activation,
            &pending_member_given(py, pending, "$args")?,
            &pending_member_given(py, pending, "$func_arg_types")?,
        );
    }
    let owner = pending_member_given(py, pending, "$value")?;
    let named = member_tag_of(py, &pending_member_given(py, pending, "$name")?)?;
    if is_stopped(py, named.bind(py))? {
        return Ok(named);
    }
    let tag: String = ok_value(named.bind(py))?.extract()?;
    let out = act_take_member(py, &activation, new_member(py, &tag)?.bind(py), &owner)?;
    Ok(out)
}

fn pending_sorted_given(py: Python<'_>, pending: &Bound<'_, PyAny>) -> PyResult<Vec<i64>> {
    let keys = g(pending, "given")?.call_method0("keys")?;
    let mut out: Vec<i64> = keys
        .try_iter()?
        .map(|key| key?.extract::<i64>())
        .collect::<PyResult<Vec<_>>>()?;
    out.sort();
    Ok(out)
}

fn pending_member_call_data(py: Python<'_>, pending: &Bound<'_, PyAny>) -> PyResult<P> {
    let tag = pending_member_tag(py, pending)?.unwrap_or_default();
    let mut node = ast_class(py, "Member")?
        .call1((PyString::new(py, &tag),))?
        .unbind();
    let environ_slot = pending_environ_slot(py, pending)?;
    let activation = g(pending, "activation")?;
    for index in pending_sorted_given(py, pending)? {
        if Some(index as usize) == environ_slot {
            continue;
        }
        let value = pending_given_get(py, pending, index as usize)?;
        let piece = travelling_piece(py, &value, Some(&activation))?;
        let Some(piece) = piece else { continue };
        let given_tags = g(pending, "given_tags")?;
        let tag = given_tags.call_method1("get", (index,))?;
        node = if tag.is_none() {
            make_partial(py, node, piece)?
        } else {
            make_partial(py, node, make_tagged(py, &tag.extract::<String>()?, piece)?)?
        };
    }
    let descriptor = pending_descriptor(py, pending, node.bind(py))?;
    make_vobject(py, descriptor.bind(py), node.bind(py), None, None)
}

fn pending_descriptor(
    py: Python<'_>,
    pending: &Bound<'_, PyAny>,
    node: &Bound<'_, PyAny>,
) -> PyResult<P> {
    let source_module = g(pending, "source_module")?;
    if let Some(tag) = pending_member_tag(py, pending)? {
        let chain = match interpreter_member_chain(py, &tag)? {
            Some(chain) => chain,
            None => return Err(runtime_error("no member chain")),
        };
        let node_type = make_ast_node_type(py, chain.bind(py), builtin_module(py)?.bind(py))?;
        return descriptor_of_node_type(py, node_type.bind(py));
    }
    if is_ast(py, node, "Partial")? {
        let head = g(pending, "head")?;
        return piece_descriptor(py, &head, &source_module);
    }
    piece_descriptor(py, node, &source_module)
}

fn pending_as_viba_data(py: Python<'_>, pending: &Bound<'_, PyAny>) -> PyResult<P> {
    if pending_member_tag(py, pending)?.is_some() {
        return make_ok(
            py,
            make_viba_data(py, pending_member_call_data(py, pending)?)?,
        );
    }
    let source: String = gs(pending, "source")?;
    let tags = pending_slot_tags(py, pending)?;
    let mut node = g(pending, "head")?.unbind();
    let mut kept: Vec<(Option<String>, Option<P>)> = Vec::new();
    for index in pending_sorted_given(py, pending)? {
        let value = pending_given_get(py, pending, index as usize)?;
        let piece: P;
        let kept_value: Option<P>;
        if is_shim(py, &value, "_Getter")? {
            return program_err(
                py,
                &format!(
                    "{source}: the {} argument is computed only when it is wanted, so it cannot \
                     be stored: give it in the chain that runs",
                    slot_name(
                        index as usize,
                        tags.get(index as usize).and_then(|one| one.as_deref())
                    )
                ),
            );
        }
        if is_shim(py, &value, "_HostArgument")? {
            piece = g(&value, "node")?.unbind();
            kept_value = None;
        } else if is_shim(py, &value, "_VibaData")? {
            piece = g(&g(&value, "node")?, "data")?.unbind();
            kept_value = Some(value.unbind());
        } else {
            return program_err(
                py,
                &format!(
                    "{source}: a closure holds viba_data only; the {} argument is not",
                    slot_name(
                        index as usize,
                        tags.get(index as usize).and_then(|one| one.as_deref())
                    )
                ),
            );
        }
        let tag = tags.get(index as usize).cloned().flatten();
        kept.push((tag.clone(), kept_value));
        node = if let Some(tag) = tag {
            make_partial(py, node, make_tagged(py, &tag, piece)?)?
        } else {
            make_partial(py, node, piece)?
        };
    }
    let descriptor = pending_descriptor(py, pending, node.bind(py))?;
    let function = reflect_mod(py)?
        .getattr("VibaFunction")?
        .call1((gs(pending, "name")?, g(pending, "module")?))?
        .unbind();
    let object = make_vobject(
        py,
        descriptor.bind(py),
        node.bind(py),
        None,
        Some(function.bind(py)),
    )?;
    let made = make_viba_data(py, object)?;
    let made_bound = made.bind(py);
    let source_module = g(pending, "source_module")?;
    if !source_module.is_none() {
        made_bound.setattr("source_module", source_module)?;
    }
    if !kept.is_empty() {
        made_bound.setattr(
            "given",
            PyList::new(
                py,
                kept.into_iter()
                    .map(|(tag, value)| {
                        Ok::<P, PyErr>(
                            PyTuple::new(
                                py,
                                [
                                    match tag {
                                        Some(tag) => py_string(py, &tag),
                                        None => py.None(),
                                    },
                                    value.unwrap_or_else(|| py.None()),
                                ],
                            )?
                            .into_any()
                            .unbind(),
                        )
                    })
                    .collect::<PyResult<Vec<_>>>()?,
            )?,
        )?;
    }
    let source_bindings = g(pending, "source_bindings")?;
    if !source_bindings.is_none() {
        made_bound.setattr("bindings", source_bindings)?;
    }
    make_ok(py, made)
}

fn pending_declaring_module_name(py: Python<'_>, pending: &Bound<'_, PyAny>) -> PyResult<String> {
    let module = g(pending, "module")?;
    Ok(g(&module, "name")?.extract::<String>().unwrap_or_default())
}

fn pending_declaring_module_path<'py>(
    py: Python<'py>,
    pending: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    g(&g(pending, "module")?, "module_path")
}

fn pending_full_name(py: Python<'_>, pending: &Bound<'_, PyAny>) -> PyResult<String> {
    let module = g(&g(pending, "module")?, "name")?.extract::<String>()?;
    let name: String = gs(pending, "name")?;
    Ok(if module.is_empty() {
        name
    } else {
        format!("{module}.{name}")
    })
}

fn pending_the_value_itself(
    py: Python<'_>,
    pending: &Bound<'_, PyAny>,
    value: &Bound<'_, PyAny>,
) -> PyResult<Option<i64>> {
    let indices = pending_sorted_given(py, pending)?;
    for index in &indices {
        let given = pending_given_get(py, pending, *index as usize)?;
        if given.is(value) {
            return Ok(Some(*index));
        }
    }
    Ok(indices.first().copied())
}

fn pending_call_viba_data(py: Python<'_>, pending: &Bound<'_, PyAny>) -> PyResult<P> {
    if pending_member_tag(py, pending)?.is_some() {
        return pending_member_call_data(py, pending);
    }
    let member_of = g(pending, "member_of")?;
    let activation = g(pending, "activation")?;
    let mut node = if member_of.is_none() {
        make_type_ref(py, DYN_CALL_NAME)?
    } else {
        make_type_ref(py, DYN_METHOD_NAME)?
    };
    let mut taken_from: Option<i64> = None;
    if !member_of.is_none() {
        let member = member_of.get_item(0)?;
        let value = member_of.get_item(1)?;
        node = make_partial(
            py,
            node,
            make_constant_str(py, &member.extract::<String>()?)?,
        )?;
        let travelled = act_travelling_data(py, &activation, &value)?;
        node = make_partial(py, node, travelled)?;
        taken_from = pending_the_value_itself(py, pending, &value)?;
    } else {
        node = make_partial(
            py,
            node,
            make_constant_str(py, &pending_full_name(py, pending)?)?,
        )?;
    }
    let environ_slot = pending_environ_slot(py, pending)?;
    for index in pending_sorted_given(py, pending)? {
        if Some(index as usize) == environ_slot || Some(index) == taken_from {
            continue;
        }
        let value = pending_given_get(py, pending, index as usize)?;
        let piece = travelling_piece(py, &value, Some(&activation))?;
        let Some(piece) = piece else { continue };
        let given_tags = g(pending, "given_tags")?;
        let tag = given_tags.call_method1("get", (index,))?;
        node = if tag.is_none() {
            make_partial(py, node, piece)?
        } else {
            make_partial(py, node, make_tagged(py, &tag.extract::<String>()?, piece)?)?
        };
    }
    let descriptor = pending_descriptor(py, pending, node.bind(py))?;
    make_vobject(py, descriptor.bind(py), node.bind(py), None, None)
}

fn make_constant_str(py: Python<'_>, value: &str) -> PyResult<P> {
    Ok(ast_class(py, "Constant")?
        .call1((PyString::new(py, value),))?
        .unbind())
}

fn completed(
    py: Python<'_>,
    answered: &Bound<'_, PyAny>,
    module_path: &str,
    full_qualified_func_name: &str,
    call: &Bound<'_, PyAny>,
) -> PyResult<P> {
    let msg: String = g(answered, "msg")?.extract()?;
    let answered_module: String = g(answered, "module_path")?.extract()?;
    let answered_full: String = g(answered, "full_qualified_func_name")?.extract()?;
    let answered_call = g(answered, "call")?;
    let tag: String = g(answered, "tag")?.extract()?;
    let module_path = if answered_module.is_empty() {
        module_path
    } else {
        &answered_module
    };
    let full_name = if answered_full.is_empty() {
        full_qualified_func_name
    } else {
        &answered_full
    };
    let chosen_call = if answered_call.is_none() {
        call
    } else {
        &answered_call
    };
    under_op_err(
        py,
        &msg,
        module_path,
        full_name,
        Some(chosen_call),
        Some(&tag),
    )
}

fn not_implemented_for(
    py: Python<'_>,
    module_path: &str,
    full_qualified_func_name: &str,
    call: &Bound<'_, PyAny>,
) -> PyResult<P> {
    under_op_err(
        py,
        REASON_NO_IMPLEMENTATION,
        module_path,
        full_qualified_func_name,
        Some(call),
        Some(NOT_IMPLEMENTED_TAG),
    )
}

fn pending_call_host(py: Python<'_>, pending: &Bound<'_, PyAny>) -> PyResult<P> {
    let environ = g(pending, "environ")?;
    let compute = g(&environ, "compute")?;
    let source: String = gs(pending, "source")?;
    if compute.is_none() {
        return program_err(
            py,
            &format!("{source}: the environment carries no compute side"),
        );
    }
    let module_path: String = pending_declaring_module_path(py, pending)?.extract()?;
    let module_name = pending_declaring_module_name(py, pending)?;
    let name = pending_full_name(py, pending)?;
    let local_name: String = gs(pending, "name")?;
    let call = pending_call_viba_data(py, pending)?;
    let get_func = g(&compute, "get_func")?;
    let host = match get_func.call1((&module_name, &local_name)) {
        Ok(host) => host,
        Err(error) => {
            if error
                .value(py)
                .as_any()
                .is_instance(&type_class(py, "EnvironmentApiInvalidArgumentErr")?)?
            {
                return Ok(error.value(py).clone().into_any().unbind());
            }
            if is_py_exception(py, &error) {
                return under_op_err(
                    py,
                    &format!(
                        "{REASON_GET_FUNC_RAISED}: get_func('{module_name}', '{local_name}') \
                         raised {}",
                        format_exc(py, &error)
                    ),
                    &module_path,
                    &name,
                    Some(call.bind(py)),
                    None,
                );
            }
            return Err(error);
        }
    };
    if let Some(failure) = handed_back_failure(py, &host)? {
        return completed(py, failure.bind(py), &module_path, &name, call.bind(py));
    }
    if host.is_none() {
        return not_implemented_for(py, &module_path, &name, call.bind(py));
    }
    let elements = g(pending, "elements")?;
    let count = elements.call_method0("__len__")?.extract::<i64>()?;
    let mut handed: Vec<P> = Vec::new();
    for index in 0..count {
        let value = pending_given_get(py, pending, index as usize)?;
        handed.push(handed_to_host(py, pending, index as usize, &value)?);
    }
    let answer = match host.call1(PyTuple::new(py, handed)?) {
        Ok(answer) => answer,
        Err(error) => {
            if let Some(stopped) = is_raised(py, &error)? {
                return Ok(stopped);
            }
            if is_py_exception(py, &error) {
                return under_op_err(
                    py,
                    &format!(
                        "{REASON_RAISED}: {local_name} raised {}",
                        format_exc(py, &error)
                    ),
                    &module_path,
                    &name,
                    Some(call.bind(py)),
                    None,
                );
            }
            return Err(error);
        }
    };
    if let Some(failure) = handed_back_failure(py, &answer)? {
        return completed(py, failure.bind(py), &module_path, &name, call.bind(py));
    }
    host_answer(
        py,
        &local_name,
        &answer,
        Some(&module_path),
        Some(&name),
        Some(call.bind(py)),
    )
}

fn pending_run_module_call(py: Python<'_>, pending: &Bound<'_, PyAny>) -> PyResult<P> {
    let module_name: String = gs(pending, "module_name")?;
    let (params, problem) = pending_params(py, pending)?;
    if let Some(problem) = problem {
        return program_err(py, &format!("module '{module_name}': {problem}"));
    }
    let environ = g(pending, "environ")?;
    let mut nodes: Vec<P> = Vec::new();
    let mut kept: Vec<P> = Vec::new();
    let mut members: Vec<P> = Vec::new();
    for param in params.unwrap_or_default() {
        let tag = param.get_item(0)?;
        let tag_string = if tag.is_none() {
            None
        } else {
            Some(tag.extract::<String>()?)
        };
        let is_env: bool = param.get_item(2)?.extract()?;
        let index = param.get_item(3)?.extract::<Option<i64>>().unwrap_or(None);
        if is_env {
            let host = new_host(py, environ.clone().unbind())?;
            members.push(
                PyTuple::new(
                    py,
                    [
                        match &tag_string {
                            Some(tag) => py_string(py, tag),
                            None => py.None(),
                        },
                        host,
                    ],
                )?
                .into_any()
                .unbind(),
            );
            continue;
        }
        let index = match index {
            Some(index) => index,
            None => 0,
        };
        let value = pending_given_get(py, pending, index as usize)?;
        if value.is_none() {
            return program_err(py, &format!("module '{module_name}' is not ready to run"));
        }
        if !is_shim(py, &value, "_VibaData")? {
            return program_err(
                py,
                &format!(
                    "module '{module_name}': the {} argument is not viba data, so it cannot be \
                     part of {DEF_NAME}",
                    slot_name(index as usize, tag_string.as_deref())
                ),
            );
        }
        let data = g(&g(&value, "node")?, "data")?.unbind();
        let node = match &tag_string {
            Some(tag) => make_tagged(py, tag, data)?,
            None => data,
        };
        nodes.push(node);
        kept.push(value.unbind());
        members.push(
            PyTuple::new(
                py,
                [
                    match &tag_string {
                        Some(tag) => py_string(py, tag),
                        None => py.None(),
                    },
                    g(pending, "given")?.call_method1("get", (index,))?.unbind(),
                ],
            )?
            .into_any()
            .unbind(),
        );
    }
    let args: P = if nodes.is_empty() {
        let nil = viba_data(py, &py.None().into_bound(py))?;
        make_viba_data(py, nil)?
    } else if nodes.len() == 1 {
        let descriptor = if is_ast(py, nodes[0].bind(py), "Tagged")? {
            let tag: String = gs(nodes[0].bind(py), "tag")?;
            descriptor_of_tagged(
                py,
                &tag,
                &g(&g(kept[0].bind(py), "node")?, "descriptor")?,
                nodes[0].bind(py),
                &g(pending, "source_module")?,
            )?
        } else {
            g(&g(kept[0].bind(py), "node")?, "descriptor")?.unbind()
        };
        let object = make_vobject(py, descriptor.bind(py), nodes[0].bind(py), None, None)?;
        make_viba_data(py, object)?
    } else {
        let chain = ast_class(py, "ProductChain")?
            .call1((PyList::new(py, nodes)?,))?
            .unbind();
        let descriptor =
            descriptor_of_values(py, chain.bind(py), &g(pending, "source_module")?, kept)?;
        let object = make_vobject(py, descriptor.bind(py), chain.bind(py), None, None)?;
        make_viba_data(py, object)?
    };
    let members_list = pylist(py, members)?;
    let bindings = g(pending, "bindings")?;
    let bindings = if bindings.is_none() {
        pytuple(py, Vec::new())?
    } else {
        bindings.unbind()
    };
    let at: i64 = g(pending, "head")
        .ok()
        .and_then(|head| g(&head, "lineno").ok())
        .and_then(|lineno| lineno.extract::<i64>().ok())
        .unwrap_or(0);
    let answer = run_module(
        py,
        &g(pending, "runner")?,
        &g(pending, "module")?,
        &g(pending, "environ")?,
        &module_name,
        None,
        Some(args.bind(py)),
        Some(members_list.bind(py)),
        bindings.bind(py),
        g(pending, "appends_environ")?.extract::<bool>()?,
        at,
        false,
        false,
    )?;
    if is_stopped(py, answer.bind(py))? {
        return Ok(answer);
    }
    let node = ok_value(answer.bind(py))?;
    if is_a(py, &node, REFLECT_MODULE, "VObject")? {
        return make_ok(py, make_viba_data(py, node.unbind())?);
    }
    make_ok(py, new_host(py, node.unbind())?)
}

// ----------------------------------------------------------------------
// What travels: the pieces a call is serialized from
// ----------------------------------------------------------------------

fn given_value(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<P> {
    if is_shim(py, value, "_HostArgument")? {
        let node = g(value, "node")?;
        let module = g(value, "module")?;
        let descriptor = piece_descriptor(py, &node, &module)?;
        let object = make_vobject(py, descriptor.bind(py), &node, None, None)?;
        return make_viba_data(py, object);
    }
    if is_shim(py, value, "_VibaData")? {
        return Ok(value.clone().unbind());
    }
    let answer = host_answer(py, "a host argument", value, None, None, None)?;
    if is_type(py, answer.bind(py), "VibaProgramErr")? {
        return Err(runtime_error(&gs(answer.bind(py), "msg")?));
    }
    Ok(ok_value(answer.bind(py))?.unbind())
}

fn as_given_value(py: Python<'_>, value: &Bound<'_, PyAny>) -> P {
    if is_type(py, value, "Ok").unwrap_or(false) {
        return ok_value(value)
            .map(|value| value.unbind())
            .unwrap_or_else(|_| value.clone().unbind());
    }
    if is_type(py, value, "Err").unwrap_or(false) {
        return value.clone().unbind();
    }
    value.clone().unbind()
}

fn as_environ_value(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Option<P>> {
    if is_shim(py, value, "_HostArgument")? {
        return Ok(Some(value.clone().unbind()));
    }
    if is_environ_value(py, value)? {
        return Ok(Some(value.clone().unbind()));
    }
    if is_shim(py, value, "Environment")? {
        return Ok(Some(new_host(py, value.clone().unbind())?));
    }
    let answer = given_value(py, value)?;
    if is_environ_value(py, answer.bind(py))? || is_shim(py, answer.bind(py), "_VibaData")? {
        return Ok(Some(answer));
    }
    Ok(None)
}

fn travelling_piece(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    activation: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<P>> {
    if is_shim(py, value, "_VibaData")? {
        return Ok(Some(match activation {
            Some(activation) => act_travelling_data(py, activation, value)?,
            None => g(&g(value, "node")?, "data")?.unbind(),
        }));
    }
    if is_shim(py, value, "_HostArgument")? {
        return Ok(Some(match activation {
            Some(activation) => act_name_as_call(py, activation, &g(value, "node")?)?,
            None => g(value, "node")?.unbind(),
        }));
    }
    if is_shim(py, value, "_DynCall")? {
        let data = dyn_call_as_viba_data(py, value)?;
        return Ok(Some(g(data.bind(py), "data")?.unbind()));
    }
    if is_shim(py, value, "_Pending")? && gs(value, "kind")? == "func" {
        let data = pending_call_viba_data(py, value)?;
        return Ok(Some(g(data.bind(py), "data")?.unbind()));
    }
    Ok(None)
}

fn arguments_data(py: Python<'_>, pieces: &[(Option<String>, Option<P>)]) -> PyResult<Option<P>> {
    let mut nodes: Vec<P> = Vec::new();
    for (tag, piece) in pieces {
        let Some(piece) = piece else { continue };
        nodes.push(match tag {
            Some(tag) => make_tagged(py, tag, piece.clone_ref(py))?,
            None => piece.clone_ref(py),
        });
    }
    if nodes.is_empty() {
        return Ok(None);
    }
    if nodes.len() == 1 {
        return Ok(Some(nodes[0].clone_ref(py)));
    }
    Ok(Some(
        ast_class(py, "ProductChain")?
            .call1((PyList::new(py, nodes)?,))?
            .unbind(),
    ))
}

fn act_travelling_data(
    py: Python<'_>,
    act: &Bound<'_, PyAny>,
    value: &Bound<'_, PyAny>,
) -> PyResult<P> {
    let node = g(&g(value, "node")?, "data")?;
    let source_module = g(value, "source_module")?;
    if !source_module.is_none() && !source_module.is(&act_attr(act, "module")?) {
        return Ok(node.unbind());
    }
    if source_path(py, &node)?.is_some() {
        return act_name_as_call(py, act, &node);
    }
    let factors = viba_data_factors(py, &g(value, "node")?)?;
    if !factors
        .iter()
        .any(|factor| is_ast(py, factor, "Tagged").unwrap_or(false))
    {
        return Ok(node.unbind());
    }
    let mut pieces: Vec<P> = Vec::new();
    for factor in factors {
        if !is_ast(py, &factor, "Tagged")? {
            pieces.push(factor.unbind());
            continue;
        }
        let member = act_evaluate(py, act, &g(&factor, "type")?, &[], true)?;
        if is_stopped(py, member.bind(py))? {
            return Ok(node.unbind());
        }
        let piece = travelling_piece(py, &ok_value(member.bind(py))?, Some(act))?;
        let Some(piece) = piece else {
            return Ok(node.unbind());
        };
        pieces.push(make_tagged(py, &gs(&factor, "tag")?, piece)?);
    }
    if pieces.len() == 1 {
        return Ok(pieces.pop().unwrap());
    }
    Ok(ast_class(py, "ProductChain")?
        .call1((PyList::new(py, pieces)?,))?
        .unbind())
}

fn act_name_as_call(
    py: Python<'_>,
    act: &Bound<'_, PyAny>,
    node: &Bound<'_, PyAny>,
) -> PyResult<P> {
    let Some(name) = source_path(py, node)? else {
        return Ok(node.clone().unbind());
    };
    if !act_asks_host_by_name(py, act, node)? {
        return Ok(node.clone().unbind());
    }
    let reference = make_type_ref(py, DYN_CALL_NAME)?;
    make_partial(py, reference, make_constant_str(py, &name)?)
}

fn act_asks_host_by_name(
    py: Python<'_>,
    act: &Bound<'_, PyAny>,
    node: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    let target = act_call_target(py, act, node, &[])?;
    let Some(target) = target else {
        return Ok(false);
    };
    if !is_ok(py, target.bind(py))? {
        return Ok(false);
    }
    let value = ok_value(target.bind(py))?;
    Ok(is_shim(py, &value, "_Pending")? && gs(&value, "kind")? == "func")
}

fn act_members_as_values(
    py: Python<'_>,
    act: &Bound<'_, PyAny>,
    value: &Bound<'_, PyAny>,
) -> PyResult<(P, Option<P>)> {
    if !is_shim(py, value, "_VibaData")? {
        return Ok((value.clone().unbind(), None));
    }
    let source_module = g(value, "source_module")?;
    if !source_module.is_none() && !source_module.is(&act_attr(act, "module")?) {
        return Ok((value.clone().unbind(), None));
    }
    let factors = viba_data_factors(py, &g(value, "node")?)?;
    if !factors
        .iter()
        .any(|factor| is_ast(py, factor, "Tagged").unwrap_or(false))
    {
        return Ok((value.clone().unbind(), None));
    }
    let module = act_attr(act, "module")?;
    let mut pieces: Vec<P> = Vec::new();
    let mut members: Vec<P> = Vec::new();
    let mut pairs: Vec<(Option<String>, P)> = Vec::new();
    for factor in factors {
        if !is_ast(py, &factor, "Tagged")? {
            pieces.push(factor.clone().unbind());
            let descriptor = piece_descriptor(py, &factor, &module)?;
            let object = make_vobject(py, descriptor.bind(py), &factor, None, None)?;
            let own = make_viba_data(py, object)?;
            members.push(own.clone_ref(py));
            pairs.push((None, own));
            continue;
        }
        let member = act_evaluate(py, act, &g(&factor, "type")?, &[], true)?;
        if is_stopped(py, member.bind(py))? {
            return Ok((value.clone().unbind(), Some(member)));
        }
        let answered = ok_value(member.bind(py))?.unbind();
        if !is_shim(py, answered.bind(py), "_VibaData")? {
            let tag: String = gs(&factor, "tag")?;
            return Ok((
                value.clone().unbind(),
                Some(program_err(
                    py,
                    &format!(
                        "the {tag} member of an argument is not viba data, so it cannot be part \
                         of {DEF_NAME}"
                    ),
                )?),
            ));
        }
        let tag: String = gs(&factor, "tag")?;
        pieces.push(make_tagged(
            py,
            &tag,
            g(&g(answered.bind(py), "node")?, "data")?.unbind(),
        )?);
        members.push(answered.clone_ref(py));
        pairs.push((Some(tag), answered));
    }
    let chain = if pieces.len() == 1 {
        pieces.pop().unwrap()
    } else {
        ast_class(py, "ProductChain")?
            .call1((PyList::new(py, pieces)?,))?
            .unbind()
    };
    let descriptor = descriptor_of_values(py, chain.bind(py), &module, members)?;
    let object = make_vobject(py, descriptor.bind(py), chain.bind(py), None, None)?;
    let made = make_viba_data(py, object)?;
    made.bind(py).setattr(
        "members",
        PyList::new(
            py,
            pairs
                .into_iter()
                .map(|(tag, member)| {
                    Ok::<P, PyErr>(
                        PyTuple::new(
                            py,
                            [
                                match tag {
                                    Some(tag) => py_string(py, &tag),
                                    None => py.None(),
                                },
                                member,
                            ],
                        )?
                        .into_any()
                        .unbind(),
                    )
                })
                .collect::<PyResult<Vec<_>>>()?,
        )?,
    )?;
    Ok((made, None))
}

fn source_tags(py: Python<'_>, node: &Bound<'_, PyAny>, module: &Bound<'_, PyAny>) -> PyResult<P> {
    let node = source_tags_generic(py, node, module)?;
    if is_ast(py, node.bind(py), "TypeApp")? {
        let got = tagged_type_of(py, node.bind(py), module)?;
        if is_ok(py, got.bind(py))? {
            let value = ok_value(got.bind(py))?;
            if !value.is_none() {
                return Ok(value.unbind());
            }
        }
    }
    Ok(node)
}

fn source_tags_generic(
    py: Python<'_>,
    node: &Bound<'_, PyAny>,
    module: &Bound<'_, PyAny>,
) -> PyResult<P> {
    for field in g(node, "_fields")?.try_iter()? {
        let field: String = field?.extract()?;
        let value = g(node, &field)?;
        if is_ast_any(py, &value)? {
            let replaced = source_tags(py, &value, module)?;
            node.setattr(&field, replaced)?;
        } else if value.is_instance_of::<PyList>() {
            let mut items: Vec<P> = Vec::new();
            let mut changed = false;
            for item in value.try_iter()? {
                let item = item?;
                if is_ast_any(py, &item)? {
                    items.push(source_tags(py, &item, module)?);
                    changed = true;
                } else {
                    items.push(item.unbind());
                }
            }
            if changed {
                node.setattr(&field, PyList::new(py, items)?)?;
            }
        }
    }
    Ok(node.clone().unbind())
}

// ----------------------------------------------------------------------
// _DynCall: a call whose name is data
// ----------------------------------------------------------------------

fn dyn_call_give(
    py: Python<'_>,
    dyn_call: &Bound<'_, PyAny>,
    tag: Option<&str>,
    value: &Bound<'_, PyAny>,
) -> PyResult<P> {
    let environ = g(dyn_call, "environ")?;
    if environ.is_none() && (tag == Some(ENVIRON_TAG) || is_environ_value(py, value)?) {
        if !is_environ_value(py, value)? {
            let source: String = gs(dyn_call, "source")?;
            return program_err(py, &format!("{source} was not given an {ENVIRON_TYPE}"));
        }
        dyn_call.setattr("environ", g(value, "obj")?)?;
        return make_ok(py, dyn_call.clone().unbind());
    }
    if g(dyn_call, "name")?.is_none() {
        dyn_call.setattr("name", value)?;
        return make_ok(py, dyn_call.clone().unbind());
    }
    if gs(dyn_call, "kind")? == "method" && g(dyn_call, "value")?.is_none() {
        dyn_call.setattr("value", value)?;
        return make_ok(py, dyn_call.clone().unbind());
    }
    g(dyn_call, "given")?.call_method1("append", (value,))?;
    g(dyn_call, "given_tags")?.call_method1(
        "append",
        (match tag {
            Some(tag) => py_string(py, tag),
            None => py.None(),
        },),
    )?;
    make_ok(py, dyn_call.clone().unbind())
}

fn dyn_call_missing(py: Python<'_>, dyn_call: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
    let source: String = gs(dyn_call, "source")?;
    if g(dyn_call, "name")?.is_none() {
        return Ok(Some(format!("{source}: was given no name to call")));
    }
    if gs(dyn_call, "kind")? == "method" && g(dyn_call, "value")?.is_none() {
        return Ok(Some(format!(
            "{source}: was given no value to take the member from"
        )));
    }
    Ok(None)
}

fn dyn_call_fire(py: Python<'_>, dyn_call: &Bound<'_, PyAny>) -> PyResult<P> {
    if g(dyn_call, "environ")?.is_none() {
        let source: String = gs(dyn_call, "source")?;
        return program_err(
            py,
            &format!("{source}: was given no {ENVIRON_TYPE} to run in"),
        );
    }
    let name_value = g(dyn_call, "name")?;
    let Some(name) = symbol_text(py, &name_value)? else {
        let source: String = gs(dyn_call, "source")?;
        return program_err(
            py,
            &format!("{source}: the name is a str, and this is no string"),
        );
    };
    if gs(dyn_call, "kind")? == "call" {
        return dyn_call_the_name(py, dyn_call, &name);
    }
    dyn_call_the_member(py, dyn_call, &name)
}

fn dyn_call_as_viba_data(py: Python<'_>, dyn_call: &Bound<'_, PyAny>) -> PyResult<P> {
    make_viba_data(py, dyn_call_call_viba_data(py, dyn_call)?)
}

fn dyn_call_the_name(py: Python<'_>, dyn_call: &Bound<'_, PyAny>, name: &str) -> PyResult<P> {
    let environ = g(dyn_call, "environ")?;
    let compute = g(&environ, "compute")?;
    let source: String = gs(dyn_call, "source")?;
    if compute.is_none() {
        return program_err(
            py,
            &format!("{source}: the environment carries no compute side"),
        );
    }
    let (module, own) = match name.rsplit_once('.') {
        Some((module, own)) => (module.to_string(), own.to_string()),
        None => {
            let activation = g(dyn_call, "activation")?;
            let module: String = g(&g(&activation, "module")?, "name")?.extract()?;
            (module, name.to_string())
        }
    };
    let module_path = if module.is_empty() {
        String::new()
    } else {
        format!("/{}", module.replace('.', "/"))
    };
    let call = dyn_call_call_viba_data(py, dyn_call)?;
    let get_func = g(&compute, "get_func")?;
    let host = match get_func.call1((&module, &own)) {
        Ok(host) => host,
        Err(error) => {
            if error
                .value(py)
                .as_any()
                .is_instance(&type_class(py, "EnvironmentApiInvalidArgumentErr")?)?
            {
                return Ok(error.value(py).clone().into_any().unbind());
            }
            if is_py_exception(py, &error) {
                return under_op_err(
                    py,
                    &format!(
                        "{REASON_GET_FUNC_RAISED}: get_func('{module}', '{own}') raised {}",
                        format_exc(py, &error)
                    ),
                    &module_path,
                    name,
                    Some(call.bind(py)),
                    None,
                );
            }
            return Err(error);
        }
    };
    if let Some(failure) = handed_back_failure(py, &host)? {
        return completed(py, failure.bind(py), &module_path, name, call.bind(py));
    }
    if host.is_none() {
        return not_implemented_for(py, &module_path, name, call.bind(py));
    }
    let mut values: Vec<P> = vec![environ.unbind()];
    for value in g(dyn_call, "given")?.try_iter()? {
        values.push(argument_value(py, &value?)?);
    }
    let answer = match host.call1(PyTuple::new(py, values)?) {
        Ok(answer) => answer,
        Err(error) => {
            if let Some(stopped) = is_raised(py, &error)? {
                return Ok(stopped);
            }
            if is_py_exception(py, &error) {
                return under_op_err(
                    py,
                    &format!("{REASON_RAISED}: {name} raised {}", format_exc(py, &error)),
                    &module_path,
                    name,
                    Some(call.bind(py)),
                    None,
                );
            }
            return Err(error);
        }
    };
    if let Some(failure) = handed_back_failure(py, &answer)? {
        return completed(py, failure.bind(py), &module_path, name, call.bind(py));
    }
    host_answer(
        py,
        name,
        &answer,
        Some(&module_path),
        Some(name),
        Some(call.bind(py)),
    )
}

fn dyn_call_the_member(py: Python<'_>, dyn_call: &Bound<'_, PyAny>, name: &str) -> PyResult<P> {
    let Some(symbol) = symbol_of(name) else {
        return program_err(py, &symbol_problem(name));
    };
    let activation = g(dyn_call, "activation")?;
    let value = g(dyn_call, "value")?;
    let taken = act_take_member(
        py,
        &activation,
        new_member(py, &tag_of(&symbol))?.bind(py),
        &value,
    )?;
    if is_stopped(py, taken.bind(py))? {
        return Ok(taken);
    }
    let mut current = ok_value(taken.bind(py))?.unbind();
    let mut given_values: Vec<P> = Vec::new();
    if !is_shim(py, current.bind(py), "_HostFunction")? {
        let host = new_host(py, g(dyn_call, "environ")?.unbind())?;
        given_values.push(new_given(py, Some(ENVIRON_TAG), host)?);
        given_values.push(new_given(py, None, value.unbind())?);
    }
    let given = g(dyn_call, "given")?;
    let given_tags = g(dyn_call, "given_tags")?;
    for (value, tag) in given.try_iter()?.zip(given_tags.try_iter()?) {
        let value = value?;
        let tag = tag?;
        let tag = if tag.is_none() {
            None
        } else {
            Some(tag.extract::<String>()?)
        };
        given_values.push(new_given(py, tag.as_deref(), value.unbind())?);
    }
    let answered = act_give_all(py, &activation, current.bind(py), &given_values, &[], false)?;
    if is_stopped(py, answered.bind(py))? {
        return Ok(answered);
    }
    current = ok_value(answered.bind(py))?.unbind();
    if is_shim(py, current.bind(py), "_HostFunction")? {
        if !host_function_filled(py, current.bind(py))? {
            let name: String = gs(current.bind(py), "name")?;
            let given = g(current.bind(py), "given")?
                .call_method0("__len__")?
                .extract::<i64>()?;
            let slots: i64 = g(current.bind(py), "slots")?.extract()?;
            return program_err(
                py,
                &format!("the environment's {name} was given {given} of its {slots} arguments"),
            );
        }
        return host_function_run(py, current.bind(py));
    }
    if !is_shim(py, current.bind(py), "_Pending")? {
        return make_ok(py, current);
    }
    act_finish(py, current.bind(py))
}

fn dyn_call_call_viba_data(py: Python<'_>, dyn_call: &Bound<'_, PyAny>) -> PyResult<P> {
    let source: String = gs(dyn_call, "source")?;
    let node = make_type_ref(py, &source)?;
    let mut node = node;
    let mut leading: Vec<P> = vec![g(dyn_call, "name")?.unbind()];
    if gs(dyn_call, "kind")? == "method" {
        leading.push(g(dyn_call, "value")?.unbind());
    }
    let activation = g(dyn_call, "activation")?;
    for value in leading {
        let piece = travelling_piece(py, value.bind(py), Some(&activation))?;
        if let Some(piece) = piece {
            node = make_partial(py, node, piece)?;
        }
    }
    let given = g(dyn_call, "given")?;
    let given_tags = g(dyn_call, "given_tags")?;
    for (value, tag) in given.try_iter()?.zip(given_tags.try_iter()?) {
        let value = value?;
        let tag = tag?;
        let piece = travelling_piece(py, &value, Some(&activation))?;
        let Some(piece) = piece else { continue };
        node = if tag.is_none() {
            make_partial(py, node, piece)?
        } else {
            make_partial(py, node, make_tagged(py, &tag.extract::<String>()?, piece)?)?
        };
    }
    let module = no_module(py)?;
    let descriptor = piece_descriptor(py, node.bind(py), module.bind(py))?;
    make_vobject(py, descriptor.bind(py), node.bind(py), None, None)
}

// ----------------------------------------------------------------------
// _HostArgument, _HostFunction, _GetArgs, _IsSubType
// ----------------------------------------------------------------------

fn host_argument_closure(py: Python<'_>, host_argument: &Bound<'_, PyAny>) -> PyResult<P> {
    let closure = g(host_argument, "closure")?;
    let false_value = pyo3::types::PyBool::new(py, false);
    if !closure.is(false_value.as_any()) {
        return Ok(closure.unbind());
    }
    host_argument.setattr("closure", py.None())?;
    let pending = g(host_argument, "pending")?;
    let activation = g(&pending, "activation")?;
    if !activation.is_none() {
        let answer = taking_a_design(py, || {
            act_evaluate(
                py,
                &activation,
                &g(host_argument, "node")?,
                &scope_frames(py, &g(host_argument, "scope")?)?,
                true,
            )
        })?;
        if is_ok(py, answer.bind(py))? && is_a_source_call(py, &ok_value(answer.bind(py))?)? {
            let node = g(&ok_value(answer.bind(py))?, "node")?.unbind();
            host_argument.setattr("closure", node.bind(py))?;
        }
    }
    Ok(g(host_argument, "closure")?.unbind())
}

fn host_argument_call(
    py: Python<'_>,
    host_argument: &Bound<'_, PyAny>,
    environ: &Bound<'_, PyAny>,
    arguments: Vec<P>,
) -> PyResult<P> {
    if g(host_argument, "answer")?.is_none() {
        let computed = taking_a_design(py, || {
            host_argument_compute(
                py,
                host_argument,
                environ,
                arguments
                    .iter()
                    .map(|argument| argument.clone_ref(py))
                    .collect(),
            )
        })?;
        host_argument.setattr("answer", computed)?;
    }
    let answer = g(host_argument, "answer")?;
    if !is_ok(py, &answer)? {
        let raised = shim_class(py, "_Raised")?.call1((answer,))?.unbind();
        return Err(PyErr::from_value(raised.bind(py).clone()));
    }
    let value = ok_value(&answer)?.unbind();
    if is_ast(py, &g(host_argument, "node")?, "TypeRef")?
        && is_shim(py, value.bind(py), "_VibaData")?
        && is_ast(py, &g(&g(value.bind(py), "node")?, "data")?, "Partial")?
    {
        return Ok(g(&g(value.bind(py), "node")?, "data")?.unbind());
    }
    argument_value(py, as_given_value(py, value.bind(py)).bind(py))
}

fn host_argument_compute(
    py: Python<'_>,
    host_argument: &Bound<'_, PyAny>,
    environ: &Bound<'_, PyAny>,
    arguments: Vec<P>,
) -> PyResult<P> {
    let pending = g(host_argument, "pending")?;
    let source = g(&pending, "activation")?;
    let pending_source_module = g(&pending, "source_module")?;
    let source_module = if pending_source_module.is_none() {
        g(&pending, "module")?
    } else {
        pending_source_module
    };
    let given = as_environ_value(py, environ)?;
    let given = match given {
        Some(given) => given,
        None => {
            return program_err(
                py,
                "a function-typed argument is run with an environment, and this is none",
            )
        }
    };
    if g(host_argument, "environ")?.is_none() {
        host_argument.setattr("environ", given.clone_ref(py))?;
    }
    let module_in_source = {
        let found = g(host_argument, "source_module")?;
        if found.is_none() {
            source_module.clone()
        } else {
            found
        }
    };
    let origin = g(host_argument, "origin")?;
    let (given_args, given_members) =
        if !origin.is_none() && g(&origin, "module")?.is(&module_in_source) {
            (g(&origin, "args")?, g(&origin, "members")?)
        } else if g(&source, "module")?.is(&module_in_source) {
            (g(&source, "args")?, g(&source, "members")?)
        } else {
            (py.None().into_bound(py), py.None().into_bound(py))
        };
    let source_environ = if is_environ_value(py, given.bind(py))? {
        g(given.bind(py), "obj")?
    } else {
        py.None().into_bound(py)
    };
    let caller = new_activation(
        py,
        &g(&source, "runner")?,
        &module_in_source,
        &source_environ,
        &gs(host_argument, "name")?,
        &g(host_argument, "file")?,
        if given_args.is_none() {
            None
        } else {
            Some(&given_args)
        },
        if given_members.is_none() {
            None
        } else {
            Some(&given_members)
        },
        None,
    )?;
    if is_ast(py, &g(host_argument, "node")?, "Partial")? {
        let (head, _source) = call_parts(py, &g(host_argument, "node")?)?;
        if is_ast(py, head.bind(py), "Partial")? {
            let answer = act_evaluate(
                py,
                caller.bind(py),
                &g(host_argument, "node")?,
                &scope_frames(py, &g(host_argument, "scope")?)?,
                true,
            )?;
            if !is_ok(py, answer.bind(py))? {
                return Ok(answer);
            }
            return Ok(answer);
        }
    }
    act_apply_without_environ(
        py,
        caller.bind(py),
        &g(host_argument, "node")?,
        &scope_frames(py, &g(host_argument, "scope")?)?,
        given.bind(py),
        arguments,
    )
}

fn host_function_filled(py: Python<'_>, host_function: &Bound<'_, PyAny>) -> PyResult<bool> {
    let given = g(host_function, "given")?
        .call_method0("__len__")?
        .extract::<i64>()?;
    let slots: i64 = g(host_function, "slots")?.extract()?;
    Ok(given >= slots)
}

fn host_function_run(py: Python<'_>, host_function: &Bound<'_, PyAny>) -> PyResult<P> {
    let mut values: Vec<P> = Vec::new();
    for value in g(host_function, "given")?.try_iter()? {
        values.push(value?.unbind());
    }
    host_function_call(py, host_function, values, None)
}

fn pending_give_host_function(
    py: Python<'_>,
    host_function: &Bound<'_, PyAny>,
    item: &Bound<'_, PyAny>,
) -> PyResult<P> {
    let mut values: Vec<P> = Vec::new();
    for value in g(host_function, "given")?.try_iter()? {
        values.push(value?.unbind());
    }
    values.push(argument_value(py, &g(item, "value")?)?);
    let mut pieces: Vec<(Option<String>, Option<P>)> = Vec::new();
    for piece in g(host_function, "pieces")?.try_iter()? {
        let piece = piece?;
        let tag = piece.get_item(0)?;
        let tag = if tag.is_none() {
            None
        } else {
            Some(tag.extract::<String>()?)
        };
        let value = piece.get_item(1)?;
        pieces.push((
            tag,
            if value.is_none() {
                None
            } else {
                Some(value.unbind())
            },
        ));
    }
    let tag = g(item, "tag")?;
    let tag = if tag.is_none() {
        None
    } else {
        Some(tag.extract::<String>()?)
    };
    let value = g(item, "value")?;
    let piece = travelling_piece(py, &value, None)?;
    pieces.push((tag, piece));
    host_function_call(py, host_function, values, Some(pieces))
}

fn host_function_call(
    py: Python<'_>,
    host_function: &Bound<'_, PyAny>,
    values: Vec<P>,
    pieces: Option<Vec<(Option<String>, Option<P>)>>,
) -> PyResult<P> {
    let slots: i64 = g(host_function, "slots")?.extract()?;
    if (values.len() as i64) < slots {
        let name: String = gs(host_function, "name")?;
        let made = shim_class(py, "_HostFunction")?.call1((
            name,
            g(host_function, "func")?,
            slots,
            PyList::new(py, values)?,
            PyList::new(
                py,
                match &pieces {
                    Some(pieces) => pieces
                        .iter()
                        .map(|(tag, piece)| {
                            Ok::<P, PyErr>(
                                PyTuple::new(
                                    py,
                                    [
                                        match tag {
                                            Some(tag) => py_string(py, tag),
                                            None => py.None(),
                                        },
                                        match piece {
                                            Some(piece) => piece.clone_ref(py),
                                            None => py.None(),
                                        },
                                    ],
                                )?
                                .into_any()
                                .unbind(),
                            )
                        })
                        .collect::<PyResult<Vec<_>>>()?,
                    None => Vec::new(),
                },
            )?,
        ))?;
        return make_ok(py, made.unbind());
    }
    let pieces = match pieces {
        Some(pieces) => pieces,
        None => {
            let mut out: Vec<(Option<String>, Option<P>)> = Vec::new();
            for piece in g(host_function, "pieces")?.try_iter()? {
                let piece = piece?;
                let tag = piece.get_item(0)?;
                let tag = if tag.is_none() {
                    None
                } else {
                    Some(tag.extract::<String>()?)
                };
                let value = piece.get_item(1)?;
                out.push((
                    tag,
                    if value.is_none() {
                        None
                    } else {
                        Some(value.unbind())
                    },
                ));
            }
            out
        }
    };
    let api = format!("{ENVIRON_TYPE}.{}", gs(host_function, "name")?);
    let arguments = arguments_data(py, &pieces)?;
    let func = g(host_function, "func")?;
    let answer = match func.call1(PyTuple::new(py, values)?) {
        Ok(answer) => answer,
        Err(error) => {
            if let Some(stopped) = is_raised(py, &error)? {
                return Ok(stopped);
            }
            if is_py_exception(py, &error) {
                if error
                    .value(py)
                    .as_any()
                    .is_instance(&type_class(py, "InterpretError")?)?
                {
                    return Ok(error.value(py).clone().into_any().unbind());
                }
                return api_err(
                    py,
                    &format!("{REASON_RAISED}: {api} raised {}", format_exc(py, &error)),
                    &api,
                    arguments,
                );
            }
            return Err(error);
        }
    };
    if let Some(failure) = handed_back_failure(py, &answer)? {
        return Ok(failure);
    }
    host_answer(py, &api, &answer, None, None, None)
}

fn get_args_give(
    py: Python<'_>,
    get_args: &Bound<'_, PyAny>,
    _item: &Bound<'_, PyAny>,
) -> PyResult<P> {
    let activation = g(get_args, "activation")?;
    let args = g(&activation, "args")?;
    if args.is_none() {
        return program_err(
            py,
            &format!(
                "{GET_ARGS_NAME} is the arguments of a module call, and no module call is running \
                 here"
            ),
        );
    }
    make_ok(py, args.unbind())
}

fn is_sub_type_filled(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(g(value, "given")?
        .call_method0("__len__")?
        .extract::<i64>()?
        >= 2)
}

fn is_sub_type_run(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<P> {
    let mut values = Vec::new();
    for one in g(value, "given")?.try_iter()? {
        values.push(one?.unbind());
    }
    is_sub_type_answer(py, values)
}

fn is_sub_type_give(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    item: &Bound<'_, PyAny>,
) -> PyResult<P> {
    let mut values = Vec::new();
    for one in g(value, "given")?.try_iter()? {
        values.push(one?.unbind());
    }
    values.push(g(item, "value")?.unbind());
    if values.len() < 2 {
        let made = shim_class(py, "_IsSubType")?
            .call1((PyList::new(py, values)?,))?
            .unbind();
        return make_ok(py, made);
    }
    is_sub_type_answer(py, values)
}

fn is_sub_type_answer(py: Python<'_>, values: Vec<P>) -> PyResult<P> {
    let type_of = |value: &Bound<'_, PyAny>| -> PyResult<Option<P>> {
        let node = match g(value, "node") {
            Ok(node) => node,
            Err(_) => return Ok(None),
        };
        let resolvable = match g(&node, "descriptor") {
            Ok(descriptor) => g(&descriptor, "resolvable_type")?,
            Err(_) => py.None().into_bound(py),
        };
        if !resolvable.is_none() {
            return Ok(Some(resolvable.unbind()));
        }
        let data = match g(&node, "data") {
            Ok(data) => data,
            Err(_) => return Ok(None),
        };
        let source_module = match g(value, "source_module") {
            Ok(module) if !module.is_none() => module.unbind(),
            _ => no_module(py)?,
        };
        Ok(Some(make_ast_node_type(py, &data, source_module.bind(py))?))
    };
    let sub_type = type_of(values[0].bind(py))?;
    let sup_type = type_of(values[1].bind(py))?;
    let (Some(sub_type), Some(sup_type)) = (sub_type, sup_type) else {
        return program_err(
            py,
            &format!("{IS_SUB_TYPE_VALUE_NAME} takes two types: this argument is not one"),
        );
    };
    let judged = is_sub_type_value(py, sub_type.bind(py), sup_type.bind(py))?;
    if is_type(py, judged.bind(py), "Err")? {
        return Ok(judged);
    }
    let value = ok_value(judged.bind(py))?.unbind();
    let data = viba_data(py, value.bind(py))?;
    make_ok(py, make_viba_data(py, data)?)
}

// ----------------------------------------------------------------------
// interpret and exec: the two entry points
// ----------------------------------------------------------------------

fn run_problem(
    py: Python<'_>,
    entry: &str,
    environ: &Bound<'_, PyAny>,
    get_file: &Bound<'_, PyAny>,
    list_files: &Bound<'_, PyAny>,
) -> PyResult<Option<P>> {
    if !is_shim(py, environ, "Environment")? {
        return Ok(Some(program_err(
            py,
            &format!("{entry} needs an Environment"),
        )?));
    }
    let viba_path = g(environ, "viba_path")?;
    if !viba_path.is_none() {
        let path_like = module(py, "os")?.getattr("PathLike")?;
        if !viba_path.is_instance_of::<PyString>() && !viba_path.is_instance(&path_like)? {
            return Ok(Some(program_err(
                py,
                &format!(
                    "viba_path is a string of directories (or one path), not {}",
                    class_name(py, &viba_path)
                ),
            )?));
        }
    }
    if !get_file.is_none() && !get_file.is_callable() {
        return Ok(Some(program_err(
            py,
            &format!(
                "get_file is a function (or None), not {}",
                class_name(py, get_file)
            ),
        )?));
    }
    if !list_files.is_none() && !list_files.is_callable() {
        return Ok(Some(program_err(
            py,
            &format!(
                "list_files is a function (or None), not {}",
                class_name(py, list_files)
            ),
        )?));
    }
    Ok(None)
}

pub(crate) fn finished(py: Python<'_>, answer: &Bound<'_, PyAny>) -> PyResult<P> {
    if is_ok(py, answer)? {
        let value = ok_value(answer)?;
        let mut answer_object = None;
        if value.is_callable() {
            answer_object = Some(program_err(
                py,
                "the run answered a function that is still waiting for arguments",
            )?);
        } else if is_shim(py, &value, "Environment")? {
            answer_object = Some(program_err(
                py,
                &format!(
                    "the run answered the environment: {ENV_TYPE} is the call's rule rather than \
                     a value"
                ),
            )?);
        } else if !is_a(py, &value, REFLECT_MODULE, "VObject")? {
            answer_object = Some(program_err(
                py,
                &format!(
                    "the run answered a host value ({}), which is no viba data",
                    class_name(py, &value)
                ),
            )?);
        }
        if let Some(answer_object) = answer_object {
            return result_viba_data(py, answer_object.bind(py));
        }
        return result_viba_data(py, answer);
    }
    result_viba_data(py, answer)
}

#[pyfunction]
#[pyo3(name = "interpret_py", signature = (viba_main_file, environ, get_file = None, list_files = None))]
fn interpret_py(
    py: Python<'_>,
    viba_main_file: &Bound<'_, PyAny>,
    environ: &Bound<'_, PyAny>,
    get_file: Option<&Bound<'_, PyAny>>,
    list_files: Option<&Bound<'_, PyAny>>,
) -> PyResult<P> {
    let no_value = none(py);
    let get_file = get_file.unwrap_or(&no_value);
    let list_files = list_files.unwrap_or(&no_value);
    if let Some(problem) = run_problem(py, "interpret", environ, get_file, list_files)? {
        return result_viba_data(py, problem.bind(py));
    }
    if let Some(native) =
        crate::native_runtime::try_interpret(py, &viba_main_file, environ, get_file, list_files)?
    {
        return Ok(native);
    }
    let runner = new_runner(py, environ, get_file, list_files)?;
    let answer = runner_run_file(
        py,
        runner.bind(py),
        &viba_main_file.str()?.to_string(),
        environ,
    )?;
    finished(py, answer.bind(py))
}

#[pyfunction]
#[pyo3(name = "exec_py", signature = (viba_code, environ, get_file = None, list_files = None))]
fn exec_py(
    py: Python<'_>,
    viba_code: &Bound<'_, PyAny>,
    environ: &Bound<'_, PyAny>,
    get_file: Option<&Bound<'_, PyAny>>,
    list_files: Option<&Bound<'_, PyAny>>,
) -> PyResult<P> {
    let no_value = none(py);
    let get_file = get_file.unwrap_or(&no_value);
    let list_files = list_files.unwrap_or(&no_value);
    if !viba_code.is_instance_of::<PyString>() && !is_a(py, viba_code, REFLECT_MODULE, "VObject")? {
        return result_viba_data(
            py,
            program_err(
                py,
                &format!(
                    "exec needs a module — its text or a node — not {}",
                    class_name(py, viba_code)
                ),
            )?
            .bind(py),
        );
    }
    if let Some(problem) = run_problem(py, "exec", environ, get_file, list_files)? {
        return result_viba_data(py, problem.bind(py));
    }
    let runner = new_runner(py, environ, get_file, list_files)?;
    let answer = if viba_code.is_instance_of::<PyString>() {
        runner_run_code(
            py,
            runner.bind(py),
            &viba_code.extract::<String>()?,
            environ,
        )?
    } else {
        runner_run_node(py, runner.bind(py), viba_code, environ)?
    };
    finished(py, answer.bind(py))
}

fn new_runner(
    py: Python<'_>,
    environ: &Bound<'_, PyAny>,
    get_file: &Bound<'_, PyAny>,
    list_files: &Bound<'_, PyAny>,
) -> PyResult<P> {
    let kwargs = PyDict::new(py);
    let viba_path = g(environ, "viba_path")?;
    kwargs.set_item("viba_path", viba_path)?;
    kwargs.set_item("get_file", get_file)?;
    kwargs.set_item("list_files", list_files)?;
    Ok(shim_class(py, "_Runner")?.call((), Some(&kwargs))?.unbind())
}

#[pyfunction(name = "not_implemented_py")]
#[pyo3(signature = (msg = "", module_path = "", full_qualified_func_name = "", call = None))]
fn not_implemented_py(
    py: Python<'_>,
    msg: &str,
    module_path: &str,
    full_qualified_func_name: &str,
    call: Option<&Bound<'_, PyAny>>,
) -> PyResult<P> {
    let no_value = none(py);
    let call = call.unwrap_or(&no_value);
    failure_viba_data(
        py,
        NOT_IMPLEMENTED_TAG,
        msg,
        module_path,
        full_qualified_func_name,
        call,
    )
}

#[pyfunction(name = "is_interpret_result_py")]
fn is_interpret_result_py(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<bool> {
    let given = viba_data(py, value)?;
    let piece = g(given.bind(py), "data")?;
    if !is_ast(py, &piece, "Tagged")? {
        return Ok(false);
    }
    let tag: String = gs(&piece, "tag")?;
    if tag != OK_TAG && tag != ERR_TAG {
        return Ok(false);
    }
    let result_module = result_module(py)?;
    let node_type = make_ast_node_type(py, &piece, result_module.bind(py))?;
    let judged = is_sub_type_value(py, node_type.bind(py), interpret_result_type(py)?.bind(py))?;
    if !is_ok(py, judged.bind(py))? {
        return Ok(false);
    }
    let value = ok_value(judged.bind(py))?;
    let true_value = pyo3::types::PyBool::new(py, true);
    if value.is(true_value.as_any()) {
        return Ok(true);
    }
    Ok(value.extract::<bool>().unwrap_or(false))
}

#[pyfunction(name = "viba_data_py")]
pub(crate) fn viba_data_py(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<P> {
    viba_data(py, value)
}

#[pyfunction(name = "source_module_of_py")]
pub(crate) fn source_module_of_py(py: Python<'_>, node: &Bound<'_, PyAny>) -> PyResult<Option<P>> {
    source_module_of(py, node)
}

// ----------------------------------------------------------------------
// The environment's members, as the shim spells them
// ----------------------------------------------------------------------

#[pyfunction(name = "storage_sub_py")]
fn storage_sub_py(py: Python<'_>, storage: &Bound<'_, PyAny>, name: &str) -> PyResult<P> {
    storage_sub(py, storage, name)
}

#[pyfunction(name = "storage_tmp_py")]
fn storage_tmp_py(py: Python<'_>, storage: &Bound<'_, PyAny>) -> PyResult<P> {
    storage_tmp(py, storage)
}

#[pyfunction(name = "storage_get_text_py")]
fn storage_get_text_py(
    py: Python<'_>,
    storage: &Bound<'_, PyAny>,
    file_path: &str,
) -> PyResult<Option<String>> {
    storage_get_text(py, storage, file_path)
}

#[pyfunction(name = "storage_put_text_py")]
fn storage_put_text_py(
    py: Python<'_>,
    storage: &Bound<'_, PyAny>,
    file_path: &str,
    content: &str,
) -> PyResult<()> {
    storage_put_text(py, storage, file_path, content)
}

#[pyfunction(name = "storage_store_path_py")]
#[pyo3(signature = (storage, file_path, root_dir = None))]
fn storage_store_path_py(
    py: Python<'_>,
    storage: &Bound<'_, PyAny>,
    file_path: &str,
    root_dir: Option<&str>,
) -> PyResult<P> {
    store_path_of(py, storage, file_path, root_dir)
}

#[pyfunction(name = "sub_env_py")]
fn sub_env_py(py: Python<'_>, environ: &Bound<'_, PyAny>, name: &Bound<'_, PyAny>) -> PyResult<P> {
    sub_env(py, environ, name)
}

#[pyfunction(name = "tmp_env_py")]
fn tmp_env_py(py: Python<'_>, environ: &Bound<'_, PyAny>) -> PyResult<P> {
    tmp_env(py, environ)
}

#[pyfunction(name = "make_referenced_env_py")]
fn make_referenced_env_py(
    py: Python<'_>,
    environ: &Bound<'_, PyAny>,
    reference_store_root_dir: &Bound<'_, PyAny>,
) -> PyResult<P> {
    make_referenced_env(py, environ, reference_store_root_dir)
}

#[pyfunction(name = "get_root_py")]
fn get_root_py(py: Python<'_>, current: &Bound<'_, PyAny>) -> PyResult<Option<P>> {
    get_root(py, current)
}

#[pyfunction(name = "get_parent_py")]
fn get_parent_py(py: Python<'_>, environ: &Bound<'_, PyAny>) -> PyResult<P> {
    get_parent(py, environ)
}

#[pyfunction(name = "env_chain_length_py")]
fn env_chain_length_py(py: Python<'_>, environ: &Bound<'_, PyAny>) -> PyResult<i64> {
    env_chain_length(py, environ)
}

#[pyfunction(name = "get_relative_path_py")]
fn get_relative_path_py(
    py: Python<'_>,
    current: &Bound<'_, PyAny>,
    root: &Bound<'_, PyAny>,
) -> PyResult<String> {
    get_relative_path(py, current, root)
}

#[pyfunction(name = "find_by_relative_path_py")]
fn find_by_relative_path_py(
    py: Python<'_>,
    relative_path: &Bound<'_, PyAny>,
    root: &Bound<'_, PyAny>,
) -> PyResult<P> {
    find_by_relative_path(py, relative_path, root)
}

#[pyfunction(name = "find_by_relative_path_member_py")]
fn find_by_relative_path_member_py(
    py: Python<'_>,
    environ: &Bound<'_, PyAny>,
    relative_path: &Bound<'_, PyAny>,
    root: &Bound<'_, PyAny>,
) -> PyResult<P> {
    find_by_relative_path_member(py, environ, relative_path, root)
}

#[pyfunction(name = "compress_env_path_py")]
fn compress_env_path_py(
    py: Python<'_>,
    sup: &Bound<'_, PyAny>,
    sub: &Bound<'_, PyAny>,
) -> PyResult<P> {
    compress_env_path(py, sup, sub)
}

#[pyfunction(name = "resumable_stack_py")]
fn resumable_stack_py(
    py: Python<'_>,
    runner: &Bound<'_, PyAny>,
    environ: &Bound<'_, PyAny>,
) -> PyResult<P> {
    resumable_stack(py, runner, environ)
}

#[pyfunction(name = "try_compact_py")]
fn try_compact_py(
    py: Python<'_>,
    environ: &Bound<'_, PyAny>,
    env_chain_length_limit: &Bound<'_, PyAny>,
) -> PyResult<P> {
    try_compact(py, environ, env_chain_length_limit)
}

#[pyfunction(name = "snapshot_path_py")]
fn snapshot_path_py(py: Python<'_>, environ: &Bound<'_, PyAny>, name: &str) -> PyResult<String> {
    snapshot_path(py, environ, name)
}

#[pyfunction(name = "get_snapshot_py")]
fn get_snapshot_py(py: Python<'_>, environ: &Bound<'_, PyAny>, name: &str) -> PyResult<Option<P>> {
    get_snapshot(py, environ, name)
}

#[pyfunction(name = "get_snapshot_from_py")]
fn get_snapshot_from_py(
    py: Python<'_>,
    environ: &Bound<'_, PyAny>,
    name: &str,
    root: &str,
) -> PyResult<Option<P>> {
    get_snapshot_from(py, environ, name, root)
}

#[pyfunction(name = "put_snapshot_py")]
fn put_snapshot_py(
    py: Python<'_>,
    environ: &Bound<'_, PyAny>,
    value: &Bound<'_, PyAny>,
    name: &str,
) -> PyResult<()> {
    put_snapshot(py, environ, value, name)
}

#[pyfunction(name = "replayed_py")]
fn replayed_py(
    py: Python<'_>,
    environ: &Bound<'_, PyAny>,
    compute: &Bound<'_, PyAny>,
    name: &str,
) -> PyResult<P> {
    replayed(py, environ, compute, name)
}

#[pyfunction(name = "record_reenact_call_py")]
fn record_reenact_call_py(
    py: Python<'_>,
    environ: &Bound<'_, PyAny>,
    tag: &Bound<'_, PyAny>,
    f: &Bound<'_, PyAny>,
    args: &Bound<'_, PyAny>,
) -> PyResult<String> {
    record_reenact_call(py, environ, tag, f, args)
}

#[pyfunction(name = "reenact_reference_py")]
fn reenact_reference_py(
    py: Python<'_>,
    environ: &Bound<'_, PyAny>,
    tag: &Bound<'_, PyAny>,
) -> PyResult<P> {
    reenact_reference(py, environ, tag)
}

#[pyfunction(name = "store_reenact_reference_py")]
fn store_reenact_reference_py(
    py: Python<'_>,
    environ: &Bound<'_, PyAny>,
    tag: &Bound<'_, PyAny>,
    value: &Bound<'_, PyAny>,
) -> PyResult<P> {
    store_reenact_reference(py, environ, tag, value)
}

#[pyfunction(name = "run_entry_py")]
fn run_entry_py(
    py: Python<'_>,
    entry: &Bound<'_, PyAny>,
    environ: &Bound<'_, PyAny>,
    arguments: &Bound<'_, PyAny>,
) -> PyResult<P> {
    run_entry(py, entry, environ, arguments)
}

#[pyfunction(name = "reenact_calls_py")]
fn reenact_calls_py(
    py: Python<'_>,
    environ: &Bound<'_, PyAny>,
    tag: &Bound<'_, PyAny>,
    entry: &Bound<'_, PyAny>,
) -> PyResult<P> {
    reenact_calls(py, environ, tag, entry)
}

#[pyfunction(name = "storage_path_of_py")]
pub(crate) fn storage_path_of_py(py: Python<'_>, environ: &Bound<'_, PyAny>) -> PyResult<String> {
    storage_path(py, environ)
}

#[pyfunction(name = "call_signature_py")]
pub(crate) fn call_signature_py(py: Python<'_>, members: &Bound<'_, PyAny>) -> PyResult<P> {
    call_signature(py, members)
}

#[pyfunction(name = "runner_imported_py")]
#[pyo3(signature = (runner, name, near = None))]
fn runner_imported_py(
    py: Python<'_>,
    runner: &Bound<'_, PyAny>,
    name: &str,
    near: Option<&str>,
) -> PyResult<P> {
    runner_imported(py, runner, name, near)
}

#[pyfunction(name = "runner_builtin_module_py")]
fn runner_builtin_module_py(
    py: Python<'_>,
    runner: &Bound<'_, PyAny>,
    name: &str,
) -> PyResult<Option<P>> {
    runner_builtin_module(py, runner, name)
}

#[pyfunction(name = "runner_run_file_py")]
fn runner_run_file_py(
    py: Python<'_>,
    runner: &Bound<'_, PyAny>,
    file: &str,
    environ: &Bound<'_, PyAny>,
) -> PyResult<P> {
    runner_run_file(py, runner, file, environ)
}

#[pyfunction(name = "runner_run_code_py")]
fn runner_run_code_py(
    py: Python<'_>,
    runner: &Bound<'_, PyAny>,
    viba_code: &str,
    environ: &Bound<'_, PyAny>,
) -> PyResult<P> {
    runner_run_code(py, runner, viba_code, environ)
}

#[pyfunction(name = "runner_run_node_py")]
fn runner_run_node_py(
    py: Python<'_>,
    runner: &Bound<'_, PyAny>,
    node: &Bound<'_, PyAny>,
    environ: &Bound<'_, PyAny>,
) -> PyResult<P> {
    runner_run_node(py, runner, node, environ)
}

#[pyfunction(name = "resumable_current_env_path_py")]
fn resumable_current_env_path_py(py: Python<'_>, resumable: &Bound<'_, PyAny>) -> PyResult<String> {
    storage_path(py, &g(resumable, "environ")?)
}

#[pyfunction(name = "resumable_restored_py")]
fn resumable_restored_py(
    py: Python<'_>,
    resumable: &Bound<'_, PyAny>,
    root: &Bound<'_, PyAny>,
) -> PyResult<P> {
    let environ = g(resumable, "environ")?;
    let (root, relative) = match get_relative_path(py, &environ, root) {
        Ok(relative) => (root.clone().unbind(), relative),
        Err(_) => {
            let root = match get_root(py, &environ)? {
                Some(root) => root,
                None => {
                    return find_by_relative_path(py, &py_string(py, "").into_bound(py), &environ)
                }
            };
            let relative = get_relative_path(py, &environ, root.bind(py))?;
            (root, relative)
        }
    };
    find_by_relative_path(py, &py_string(py, &relative).into_bound(py), root.bind(py))
}

#[pyfunction(name = "resumable_run_py")]
fn resumable_run_py(
    py: Python<'_>,
    resumable: &Bound<'_, PyAny>,
    environ: &Bound<'_, PyAny>,
) -> PyResult<P> {
    let mut members: Vec<P> = Vec::new();
    for member in g(resumable, "members")?.try_iter()? {
        let member = member?;
        let tag = member.get_item(0)?;
        let value = member.get_item(1)?;
        let chosen = if is_environ_value(py, &value)? {
            new_host(py, environ.clone().unbind())?
        } else {
            value.unbind()
        };
        members.push(
            PyTuple::new(py, [tag.clone(), chosen.bind(py).clone()])?
                .into_any()
                .unbind(),
        );
    }
    let members = pylist(py, members)?;
    let runner_value = g(resumable, "runner")?;
    let module_value = g(resumable, "module")?;
    let name_value: String = gs(resumable, "name")?;
    let file_value = g(resumable, "file")?;
    let args_value = g(resumable, "args")?;
    let bindings_value = g(resumable, "bindings")?;
    run_module(
        py,
        &runner_value,
        &module_value,
        environ,
        &name_value,
        Some(&file_value),
        Some(&args_value),
        Some(&members.bind(py).clone()),
        &bindings_value,
        g(resumable, "passes_environ")?.extract()?,
        g(resumable, "at")?.extract()?,
        g(resumable, "give_environ")?.extract()?,
        true,
    )
}

#[pyfunction(name = "getter_call_py")]
pub(crate) fn getter_call_py(py: Python<'_>, getter: &Bound<'_, PyAny>) -> PyResult<P> {
    if g(getter, "answer")?.is_none() {
        let activation = g(getter, "activation")?;
        let node = g(getter, "node")?;
        let scope = scope_frames(py, &g(getter, "scope")?)?;
        let answer = taking_a_design(py, || act_evaluate(py, &activation, &node, &scope, true))?;
        getter.setattr("answer", answer)?;
    }
    let answer = g(getter, "answer")?;
    if !is_ok(py, &answer)? {
        let raised = shim_class(py, "_Raised")?.call1((answer,))?.unbind();
        return Err(PyErr::from_value(raised.bind(py).clone()));
    }
    argument_value(py, &ok_value(&answer)?)
}

#[pyfunction(name = "host_argument_call_py")]
pub(crate) fn host_argument_call_py(
    py: Python<'_>,
    host_argument: &Bound<'_, PyAny>,
    environ: &Bound<'_, PyAny>,
    arguments: &Bound<'_, PyAny>,
) -> PyResult<P> {
    let mut values = Vec::new();
    for argument in arguments.try_iter()? {
        values.push(argument?.unbind());
    }
    host_argument_call(py, host_argument, environ, values)
}

#[pyfunction(name = "host_argument_closure_py")]
pub(crate) fn host_argument_closure_py(
    py: Python<'_>,
    host_argument: &Bound<'_, PyAny>,
) -> PyResult<P> {
    host_argument_closure(py, host_argument)
}

#[pyfunction(name = "host_function_call_py")]
pub(crate) fn host_function_call_py(
    py: Python<'_>,
    host_function: &Bound<'_, PyAny>,
    arguments: &Bound<'_, PyAny>,
) -> PyResult<P> {
    let mut values: Vec<P> = Vec::new();
    for value in g(host_function, "given")?.try_iter()? {
        values.push(value?.unbind());
    }
    for argument in arguments.try_iter()? {
        values.push(argument_value(py, &argument?)?);
    }
    let slots: i64 = g(host_function, "slots")?.extract()?;
    if (values.len() as i64) < slots {
        let name: String = gs(host_function, "name")?;
        return Err(PyErr::from_value(
            program_err(
                py,
                &format!("{ENVIRON_TYPE}.{name} is still waiting for arguments"),
            )?
            .bind(py)
            .clone(),
        ));
    }
    let answer = host_function_call(py, host_function, values, None)?;
    if !is_ok(py, answer.bind(py))? {
        return Err(raise_result(py, answer.bind(py)));
    }
    Ok(ok_value(answer.bind(py))?.unbind())
}

fn raise_result(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyErr {
    if value.is_instance_of::<pyo3::exceptions::PyBaseException>() {
        return PyErr::from_value(value.clone());
    }
    pyo3::exceptions::PyTypeError::new_err("exceptions must derive from BaseException")
}

#[pyfunction(name = "dyn_call_as_viba_data_py")]
pub(crate) fn dyn_call_as_viba_data_py(py: Python<'_>, dyn_call: &Bound<'_, PyAny>) -> PyResult<P> {
    dyn_call_as_viba_data(py, dyn_call)
}

#[pyfunction(name = "pending_as_viba_data_py")]
pub(crate) fn pending_as_viba_data_py(py: Python<'_>, pending: &Bound<'_, PyAny>) -> PyResult<P> {
    pending_as_viba_data(py, pending)
}

#[pyfunction(name = "run_module_py")]
#[pyo3(signature = (runner, module, environ, name, file, args = None, members = None,
                    bindings = None, passes_environ = false, at = 0,
                    give_environ = false, driving = false))]
#[allow(clippy::too_many_arguments)]
fn run_module_py(
    py: Python<'_>,
    runner: &Bound<'_, PyAny>,
    module: &Bound<'_, PyAny>,
    environ: &Bound<'_, PyAny>,
    name: &str,
    file: &Bound<'_, PyAny>,
    args: Option<&Bound<'_, PyAny>>,
    members: Option<&Bound<'_, PyAny>>,
    bindings: Option<&Bound<'_, PyAny>>,
    passes_environ: bool,
    at: i64,
    give_environ: bool,
    driving: bool,
) -> PyResult<P> {
    let bindings = match bindings {
        Some(bindings) => bindings.clone().unbind(),
        None => pytuple(py, Vec::new())?,
    };
    let outcome = run_module_impl(
        py,
        runner,
        module,
        environ,
        name,
        if file.is_none() { None } else { Some(file) },
        args,
        members,
        bindings.bind(py),
        passes_environ,
        at,
        give_environ,
        driving,
    );
    outcome
}

#[pyfunction(name = "hold_py")]
fn hold_py(py: Python<'_>, runner: &Bound<'_, PyAny>, tasks: &Bound<'_, PyAny>) -> PyResult<()> {
    hold_impl(py, runner, tasks)
}

// ----------------------------------------------------------------------
// Registration
// ----------------------------------------------------------------------

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(interpret_py, m)?)?;
    m.add_function(wrap_pyfunction!(exec_py, m)?)?;
    m.add_function(wrap_pyfunction!(not_implemented_py, m)?)?;
    m.add_function(wrap_pyfunction!(is_interpret_result_py, m)?)?;
    m.add_function(wrap_pyfunction!(viba_data_py, m)?)?;
    m.add_function(wrap_pyfunction!(source_module_of_py, m)?)?;
    m.add_function(wrap_pyfunction!(storage_sub_py, m)?)?;
    m.add_function(wrap_pyfunction!(storage_tmp_py, m)?)?;
    m.add_function(wrap_pyfunction!(storage_get_text_py, m)?)?;
    m.add_function(wrap_pyfunction!(storage_put_text_py, m)?)?;
    m.add_function(wrap_pyfunction!(storage_store_path_py, m)?)?;
    m.add_function(wrap_pyfunction!(sub_env_py, m)?)?;
    m.add_function(wrap_pyfunction!(tmp_env_py, m)?)?;
    m.add_function(wrap_pyfunction!(make_referenced_env_py, m)?)?;
    m.add_function(wrap_pyfunction!(get_root_py, m)?)?;
    m.add_function(wrap_pyfunction!(get_parent_py, m)?)?;
    m.add_function(wrap_pyfunction!(env_chain_length_py, m)?)?;
    m.add_function(wrap_pyfunction!(get_relative_path_py, m)?)?;
    m.add_function(wrap_pyfunction!(find_by_relative_path_py, m)?)?;
    m.add_function(wrap_pyfunction!(find_by_relative_path_member_py, m)?)?;
    m.add_function(wrap_pyfunction!(compress_env_path_py, m)?)?;
    m.add_function(wrap_pyfunction!(resumable_stack_py, m)?)?;
    m.add_function(wrap_pyfunction!(try_compact_py, m)?)?;
    m.add_function(wrap_pyfunction!(snapshot_path_py, m)?)?;
    m.add_function(wrap_pyfunction!(get_snapshot_py, m)?)?;
    m.add_function(wrap_pyfunction!(get_snapshot_from_py, m)?)?;
    m.add_function(wrap_pyfunction!(put_snapshot_py, m)?)?;
    m.add_function(wrap_pyfunction!(replayed_py, m)?)?;
    m.add_function(wrap_pyfunction!(record_reenact_call_py, m)?)?;
    m.add_function(wrap_pyfunction!(reenact_reference_py, m)?)?;
    m.add_function(wrap_pyfunction!(store_reenact_reference_py, m)?)?;
    m.add_function(wrap_pyfunction!(run_entry_py, m)?)?;
    m.add_function(wrap_pyfunction!(reenact_calls_py, m)?)?;
    m.add_function(wrap_pyfunction!(storage_path_of_py, m)?)?;
    m.add_function(wrap_pyfunction!(call_signature_py, m)?)?;
    m.add_function(wrap_pyfunction!(runner_imported_py, m)?)?;
    m.add_function(wrap_pyfunction!(runner_builtin_module_py, m)?)?;
    m.add_function(wrap_pyfunction!(runner_run_file_py, m)?)?;
    m.add_function(wrap_pyfunction!(runner_run_code_py, m)?)?;
    m.add_function(wrap_pyfunction!(runner_run_node_py, m)?)?;
    m.add_function(wrap_pyfunction!(resumable_current_env_path_py, m)?)?;
    m.add_function(wrap_pyfunction!(resumable_restored_py, m)?)?;
    m.add_function(wrap_pyfunction!(resumable_run_py, m)?)?;
    m.add_function(wrap_pyfunction!(getter_call_py, m)?)?;
    m.add_function(wrap_pyfunction!(host_argument_call_py, m)?)?;
    m.add_function(wrap_pyfunction!(host_argument_closure_py, m)?)?;
    m.add_function(wrap_pyfunction!(host_function_call_py, m)?)?;
    m.add_function(wrap_pyfunction!(dyn_call_as_viba_data_py, m)?)?;
    m.add_function(wrap_pyfunction!(pending_as_viba_data_py, m)?)?;
    m.add_function(wrap_pyfunction!(run_module_py, m)?)?;
    m.add_function(wrap_pyfunction!(hold_py, m)?)?;
    Ok(())
}
