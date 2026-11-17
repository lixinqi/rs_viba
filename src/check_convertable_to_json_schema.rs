//! Whether a design can go to JSON and come back (was viba/check_convertable_to_json_schema.py).
//!
//! A viba value is written as JSON-looking data: a product is an object, a list
//! is an array, a scalar is a scalar. Reading a value back means picking an
//! option of a sum from what the JSON value is — and that only works where the
//! options can be told apart by what JSON has: null, a boolean, a number, a
//! string, an array, an object, or the tag an option carries. So the judgment
//! walks the type the entry names and asks of every sum it reaches that no two
//! options read as the same thing.
//!
//! The product side — one tag per product, inline chains that end — is
//! `check_tag_and_inline`, run over the same design after this walk.

use std::collections::HashSet;
use std::path::Path;

use pyo3::exceptions::{PyOSError, PySyntaxError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyModule, PySet, PyString, PyTuple};

use crate::pyast;
use crate::tagged::TAGGED_NAME;

/// The names a value of the two units reads as: null, or nothing at all.
pub(crate) const NULL_NAMES: [&str; 4] = ["nil", "Object", "void", "None"];
pub(crate) const NO_VALUE_NAMES: [&str; 2] = ["never", "Oneof"];

/// The fixed-width integers: their schema is JSON's one integer kind in a range.
pub(crate) const FIXED_INT_RANGES: [(&str, i128, i128); 8] = [
    ("int8_t", -(2i128.pow(7)), 2i128.pow(7) - 1),
    ("uint8_t", 0, 2i128.pow(8) - 1),
    ("int16_t", -(2i128.pow(15)), 2i128.pow(15) - 1),
    ("uint16_t", 0, 2i128.pow(16) - 1),
    ("int32_t", -(2i128.pow(31)), 2i128.pow(31) - 1),
    ("uint32_t", 0, 2i128.pow(32) - 1),
    ("int64_t", -(2i128.pow(63)), 2i128.pow(63) - 1),
    ("uint64_t", 0, 2i128.pow(64) - 1),
];

pub(crate) fn is_array_constructor(name: &str) -> bool {
    matches!(name, "list" | "set" | "ListLiteral" | "SetLiteral")
}

pub(crate) fn is_object_constructor(name: &str) -> bool {
    matches!(name, "dict" | "DictLiteral")
}

/// The JSON kind a scalar name reads as (`SCALAR_NAMES` in the judgment).
pub(crate) fn scalar_read_kind(name: &str) -> Option<&'static str> {
    match name {
        "int" | "float" | "int8_t" | "uint8_t" | "int16_t" | "uint16_t" | "int32_t"
        | "uint32_t" | "int64_t" | "uint64_t" => Some("number"),
        "str" => Some("string"),
        "bool" => Some("boolean"),
        _ => None,
    }
}

/// The schema a scalar name has, or None when it is no scalar.
pub(crate) fn scalar_schema(py: Python<'_>, name: &str) -> PyResult<Option<Py<PyAny>>> {
    let kind = match name {
        "int" => "integer",
        "float" => "number",
        "str" => "string",
        "bool" => "boolean",
        _ => {
            for (fixed, low, high) in FIXED_INT_RANGES {
                if fixed == name {
                    let schema = PyDict::new(py);
                    schema.set_item("type", "integer")?;
                    schema.set_item("minimum", low)?;
                    schema.set_item("maximum", high)?;
                    return Ok(Some(schema.into_any().unbind()));
                }
            }
            return Ok(None);
        }
    };
    Ok(Some(dict_obj(py, vec![("type", pyast::pystr(py, kind))])?))
}

/// The key a tag writes as: its name, the `$` left off.
pub(crate) fn tag_name(tag: &str) -> String {
    tag.strip_prefix('$').unwrap_or(tag).to_string()
}

/// A reference to a definition key, with the two characters JSON Pointer escapes.
pub(crate) fn ref_schema(py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
    let pointer = key.replace('~', "~0").replace('/', "~1");
    dict_obj(
        py,
        vec![("$ref", pyast::pystr(py, &format!("#/$defs/{pointer}")))],
    )
}

/// A Python str, as the `value!r` fragments of the error messages spell it.
pub(crate) fn repr_str(py: Python<'_>, text: &str) -> String {
    pyast::repr_of(py, PyString::new(py, text).as_any())
}

pub(crate) fn dict_obj(py: Python<'_>, entries: Vec<(&str, Py<PyAny>)>) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    for (key, value) in entries {
        dict.set_item(key, value)?;
    }
    Ok(dict.into_any().unbind())
}

pub(crate) fn list_obj(py: Python<'_>, items: Vec<Py<PyAny>>) -> PyResult<Py<PyAny>> {
    pyast::pylist(py, items)
}

/// Why a walk stopped: the judgment's mistake, the schema writer's, or Python's.
pub(crate) enum Failure {
    Check(String),
    Schema(String),
    Py(PyErr),
}

pub(crate) type WResult<T> = Result<T, Failure>;

impl From<PyErr> for Failure {
    fn from(error: PyErr) -> Self {
        Failure::Py(error)
    }
}

impl<'a, 'py> From<pyo3::CastError<'a, 'py>> for Failure {
    fn from(error: pyo3::CastError<'a, 'py>) -> Self {
        Failure::Py(error.into())
    }
}

impl Failure {
    pub fn message(self) -> String {
        match self {
            Failure::Check(message) | Failure::Schema(message) => message,
            Failure::Py(error) => error.to_string(),
        }
    }
}

fn check_error(message: String) -> Failure {
    Failure::Check(message)
}

fn schema_error(message: String) -> Failure {
    Failure::Schema(message)
}

/// The language's units (`reflect.language_config`): the names that stand for
/// `nil` and for `never`, and the two readings they answer to.
#[derive(Clone)]
pub(crate) struct Config {
    nil_eqv: HashSet<String>,
    never_eqv: HashSet<String>,
}

impl Config {
    pub fn in_nil_eqv(&self, name: &str) -> bool {
        self.nil_eqv.contains(name)
    }

    pub fn in_never_eqv(&self, name: &str) -> bool {
        self.never_eqv.contains(name)
    }

    /// A unit: `nil`/`Object` under any of its names, or `never`/`Oneof`.
    pub fn is_null(&self, name: &str) -> bool {
        self.in_nil_eqv(name) || NULL_NAMES.contains(&name)
    }

    /// The sum's unit: `never` itself, or a name standing for it.
    pub fn is_no_value(&self, name: &str) -> bool {
        self.in_never_eqv(name) || NO_VALUE_NAMES.contains(&name)
    }

    pub fn is_unit(&self, name: &str) -> bool {
        self.is_null(name) || self.is_no_value(name)
    }
}

pub(crate) fn language_config(rt: &Runtime<'_>) -> PyResult<Config> {
    let py = rt.py;
    let reflect = PyModule::import(py, "rs_viba.reflect")?;
    let config = reflect.getattr("language_config")?;
    Ok(Config {
        nil_eqv: sorted_names(py, &config.getattr("nil_eqv")?)?,
        never_eqv: sorted_names(py, &config.getattr("never_eqv")?)?,
    })
}

fn sorted_names(py: Python<'_>, values: &Bound<'_, PyAny>) -> PyResult<HashSet<String>> {
    let sorted = py.import("builtins")?.getattr("sorted")?.call1((values,))?;
    let mut names = HashSet::new();
    for item in sorted.try_iter()? {
        names.insert(item?.extract::<String>()?);
    }
    Ok(names)
}

/// A Python object a `Result` may be held as: an owned one, or one borrowed
/// from a call's answer. The runtime reads both the same way.
pub(crate) trait ResultValue<'py> {
    fn result_bound(&self, py: Python<'py>) -> Bound<'py, PyAny>;
}

impl<'py> ResultValue<'py> for Py<PyAny> {
    fn result_bound(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        self.bind(py).clone()
    }
}

impl<'py> ResultValue<'py> for Bound<'py, PyAny> {
    fn result_bound(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        let _ = py;
        self.clone()
    }
}

/// The Python modules and classes an entry is read through.
pub(crate) struct Runtime<'py> {
    pub py: Python<'py>,
    pub ast: Bound<'py, PyModule>,
    pub walk: Bound<'py, PyModule>,
    pub pattern: Bound<'py, PyModule>,
    pub types: Bound<'py, PyModule>,
    ok: Bound<'py, PyAny>,
    err: Bound<'py, PyAny>,
    ast_node_type: Bound<'py, PyAny>,
    custom_module: Bound<'py, PyAny>,
    generic_module: Bound<'py, PyAny>,
}

impl<'py> Runtime<'py> {
    pub fn new(py: Python<'py>) -> PyResult<Self> {
        let ast = PyModule::import(py, "rs_viba.viba_ast")?;
        let walk = PyModule::import(py, "rs_viba.design_walk")?;
        let pattern = PyModule::import(py, "rs_viba.pattern")?;
        let types = PyModule::import(py, "rs_viba.type")?;
        let ok = types.getattr("Ok")?;
        let err = types.getattr("Err")?;
        let ast_node_type = types.getattr("AstNodeType")?;
        let custom_module = types.getattr("CustomModuleType")?;
        let generic_module = pattern.getattr("GenericModuleType")?;
        Ok(Self {
            py,
            ast,
            walk,
            pattern,
            types,
            ok,
            err,
            ast_node_type,
            custom_module,
            generic_module,
        })
    }

    pub fn class_name(&self, node: &Bound<'py, PyAny>) -> PyResult<String> {
        pyast::class_name(node)
    }

    pub fn is_ast(&self, node: &Bound<'py, PyAny>, name: &str) -> PyResult<bool> {
        Ok(self.class_name(node)? == name)
    }

    pub fn is_err(&self, value: &(impl ResultValue<'py> + ?Sized)) -> PyResult<bool> {
        let value = value.result_bound(self.py);
        value.is_instance(&self.err)
    }

    pub fn is_custom_module(&self, value: &Bound<'py, PyAny>) -> PyResult<bool> {
        value.is_instance(&self.custom_module)
    }

    pub fn is_generic_module(&self, value: &Bound<'py, PyAny>) -> PyResult<bool> {
        value.is_instance(&self.generic_module)
    }

    pub fn is_ast_node_type(&self, value: &Bound<'py, PyAny>) -> PyResult<bool> {
        value.is_instance(&self.ast_node_type)
    }

    pub fn ok_value(&self, result: &(impl ResultValue<'py> + ?Sized)) -> PyResult<Py<PyAny>> {
        Ok(result.result_bound(self.py).getattr("ok_value")?.unbind())
    }

    pub fn err_msg(&self, result: &(impl ResultValue<'py> + ?Sized)) -> PyResult<String> {
        result.result_bound(self.py).getattr("msg")?.extract()
    }

    pub fn make_ok(&self, value: &(impl ResultValue<'py> + ?Sized)) -> PyResult<Py<PyAny>> {
        let value = value.result_bound(self.py);
        Ok(self.ok.call1((value,))?.unbind())
    }

    pub fn make_err(&self, message: &str) -> PyResult<Py<PyAny>> {
        Ok(self.err.call1((message,))?.unbind())
    }

    pub fn repr(&self, text: &str) -> String {
        repr_str(self.py, text)
    }

    pub fn type_name(&self, value: &Bound<'py, PyAny>) -> PyResult<String> {
        Ok(value.get_type().name()?.to_string())
    }

    pub fn chain_elements(
        &self,
        node: &Bound<'py, PyAny>,
    ) -> PyResult<Option<Vec<Bound<'py, PyAny>>>> {
        let value = self.walk.getattr("chain_elements")?.call1((node,))?;
        if value.is_none() {
            return Ok(None);
        }
        let mut elements = Vec::new();
        for item in value.try_iter()? {
            elements.push(item?);
        }
        Ok(Some(elements))
    }

    pub fn is_definition(&self, node: &Bound<'py, PyAny>) -> PyResult<bool> {
        self.walk
            .getattr("is_definition")?
            .call1((node,))?
            .extract()
    }

    pub fn is_function(&self, definition: &Bound<'py, PyAny>) -> PyResult<bool> {
        self.walk
            .getattr("is_function")?
            .call1((definition,))?
            .extract()
    }

    pub fn module_label(&self, module: &Bound<'py, PyAny>) -> PyResult<String> {
        let label = self.walk.getattr("module_label")?.call1((module,))?;
        label.str()?.extract()
    }

    pub fn under_builtin(&self, path: &str) -> PyResult<bool> {
        self.walk
            .getattr("under_builtin")?
            .call1((path,))?
            .extract()
    }

    pub fn split_type_name(&self, name: &str) -> PyResult<(String, String)> {
        self.walk
            .getattr("split_type_name")?
            .call1((name,))?
            .extract()
    }

    pub fn with_params(
        &self,
        definition: &Bound<'py, PyAny>,
        bindings: &Bound<'py, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        Ok(self
            .walk
            .getattr("with_params")?
            .call1((definition, bindings))?
            .unbind())
    }

    pub fn bindings_key(&self, bindings: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
        Ok(self
            .walk
            .getattr("bindings_key")?
            .call1((bindings,))?
            .unbind())
    }

    pub fn module_get_type(&self, module: &Bound<'py, PyAny>, name: &str) -> PyResult<Py<PyAny>> {
        Ok(self
            .types
            .getattr("module_get_type")?
            .call1((module, name))?
            .unbind())
    }

    pub fn generic_named(&self, module: &Bound<'py, PyAny>, name: &str) -> PyResult<Py<PyAny>> {
        Ok(self
            .pattern
            .getattr("generic_named")?
            .call1((module, name))?
            .unbind())
    }

    pub fn decide(
        &self,
        generic: &Bound<'py, PyAny>,
        arguments: &Bound<'py, PyAny>,
        module: &Bound<'py, PyAny>,
        argument_modules: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let kwargs = PyDict::new(self.py);
        match argument_modules {
            Some(value) => kwargs.set_item("argument_modules", value)?,
            None => kwargs.set_item("argument_modules", self.py.None())?,
        }
        Ok(self
            .pattern
            .getattr("decide")?
            .call((generic, arguments, module), Some(&kwargs))?
            .unbind())
    }

    /// A node's spelling, or its class name when it has none.
    pub fn spell(&self, node: &Bound<'py, PyAny>) -> String {
        match self
            .ast
            .getattr("unparse_type")
            .and_then(|function| function.call1((node,)))
            .and_then(|text| text.extract::<String>())
        {
            Ok(text) => text,
            Err(_) => self.class_name(node).unwrap_or_else(|_| "?".to_string()),
        }
    }

    pub fn dump(&self, node: &Bound<'py, PyAny>) -> PyResult<String> {
        self.ast.getattr("dump")?.call1((node,))?.extract()
    }

    pub fn source_path(&self, node: &Bound<'py, PyAny>) -> PyResult<Option<String>> {
        let value = self.ast.getattr("source_path")?.call1((node,))?;
        if value.is_none() {
            return Ok(None);
        }
        Ok(Some(value.extract()?))
    }

    pub fn ast_type_ref(&self, name: &str) -> PyResult<Py<PyAny>> {
        Ok(self.ast.getattr("TypeRef")?.call1((name,))?.unbind())
    }
}

/// The `generic_params` a definition declares, or none when it declares none.
pub(crate) fn generic_params(definition: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    let value = match definition.getattr("generic_params") {
        Ok(value) => value,
        Err(_) => return Ok(Vec::new()),
    };
    if value.is_none() {
        return Ok(Vec::new());
    }
    let mut params = Vec::new();
    for item in value.try_iter()? {
        params.push(item?.extract::<String>()?);
    }
    Ok(params)
}

/// The three parts of a binding: (the argument, its module, the bindings it was spelled under).
pub(crate) fn binding_parts<'py>(
    bound: &Bound<'py, PyAny>,
) -> PyResult<Option<(Py<PyAny>, Py<PyAny>, Py<PyAny>)>> {
    if !bound.is_instance_of::<PyTuple>() {
        return Ok(None);
    }
    let tuple = bound.cast::<PyTuple>()?;
    Ok(Some((
        tuple.get_item(0)?.unbind(),
        tuple.get_item(1)?.unbind(),
        tuple.get_item(2)?.unbind(),
    )))
}

fn collect_iterable<'py>(
    value: &Bound<'py, PyAny>,
) -> PyResult<Result<Vec<Bound<'py, PyAny>>, PyErr>> {
    match value.try_iter() {
        Ok(iterator) => {
            let mut items = Vec::new();
            for item in iterator {
                items.push(item?);
            }
            Ok(Ok(items))
        }
        Err(error) => Ok(Err(error)),
    }
}

/// `argument_problem`: why these arguments are no entry, or None when they are.
pub(crate) fn argument_problem(
    py: Python<'_>,
    viba_path: &Bound<'_, PyAny>,
    entry: &Bound<'_, PyAny>,
) -> PyResult<Option<String>> {
    if viba_path.is_none() {
        return Ok(Some("viba_path is a list of directories".to_string()));
    }
    if viba_path.is_instance_of::<PyString>() {
        return Ok(Some(
            "viba_path is a list of directories, not one directory".to_string(),
        ));
    }
    let paths = match collect_iterable(viba_path)? {
        Ok(paths) => paths,
        Err(error) => {
            if error.is_instance_of::<pyo3::exceptions::PyTypeError>(py) {
                return Ok(Some(format!(
                    "viba_path is a list of directories, not {}",
                    type_name_of(viba_path)?
                )));
            }
            return Err(error);
        }
    };
    for one in &paths {
        if !one.is_instance_of::<PyString>() {
            return Ok(Some(format!(
                "viba_path holds a directory, not {}",
                type_name_of(one)?
            )));
        }
    }
    let no_entry = "entry is (\"module_name\", name) or (\"type_names\", [names])".to_string();
    if !entry.is_instance_of::<PyTuple>() {
        return Ok(Some(no_entry));
    }
    let tuple = entry.cast::<PyTuple>()?;
    if tuple.len() != 2 || !tuple.get_item(0)?.is_instance_of::<PyString>() {
        return Ok(Some(no_entry));
    }
    let tag: String = tuple.get_item(0)?.extract()?;
    let tag = tag.trim_start_matches('$').to_string();
    let value = tuple.get_item(1)?;
    if tag == "module_name" {
        if !value.is_instance_of::<PyString>() {
            return Ok(Some(format!(
                "module_name is a string, not {}",
                type_name_of(&value)?
            )));
        }
        return Ok(None);
    }
    if tag == "type_names" {
        if value.is_instance_of::<PyString>() {
            return Ok(Some(
                "type_names is a list of names, not one name".to_string(),
            ));
        }
        let names = match collect_iterable(&value)? {
            Ok(names) => names,
            Err(error) => {
                if error.is_instance_of::<pyo3::exceptions::PyTypeError>(py) {
                    return Ok(Some(format!(
                        "type_names is a list of names, not {}",
                        type_name_of(&value)?
                    )));
                }
                return Err(error);
            }
        };
        for one in &names {
            if !one.is_instance_of::<PyString>() {
                return Ok(Some(format!(
                    "type_names holds a name, not {}",
                    type_name_of(one)?
                )));
            }
        }
        return Ok(None);
    }
    Ok(Some(format!(
        "{} is no branch of entry: module_name or type_names",
        repr_str(py, &tag)
    )))
}

fn type_name_of(value: &Bound<'_, PyAny>) -> PyResult<String> {
    Ok(value.get_type().name()?.to_string())
}

/// The module a name loads: a user module, never a generic.
fn load_module<'py>(
    rt: &Runtime<'py>,
    loader: &Bound<'py, PyAny>,
    name: &str,
    schema: bool,
) -> WResult<Py<PyAny>> {
    let py = rt.py;
    let loaded = loader.call_method1("imported", (name, py.None()))?;
    let fail = |message: String| {
        if schema {
            schema_error(message)
        } else {
            check_error(message)
        }
    };
    if rt.is_err(&loaded)? {
        let msg = rt.err_msg(&loaded)?;
        return Err(fail(format!(
            "cannot load the module {}: {msg}",
            rt.repr(name)
        )));
    }
    let module = rt.ok_value(&loaded)?;
    if !rt.is_custom_module(module.bind(py))? {
        return Err(fail(format!(
            "{} names a generic: a generic answers an application, not a bare name",
            rt.repr(name)
        )));
    }
    Ok(module)
}

/// The definitions an entry names: (the definition, the module it is in).
pub(crate) fn entry_definitions<'py>(
    rt: &Runtime<'py>,
    loader: &Bound<'py, PyAny>,
    tag: &str,
    value: &Bound<'py, PyAny>,
    strict: bool,
) -> WResult<Vec<(Py<PyAny>, Py<PyAny>)>> {
    let py = rt.py;
    let mut definitions: Vec<(Py<PyAny>, Py<PyAny>)> = Vec::new();
    if tag == "module_name" {
        let name: String = value.extract()?;
        let module = load_module(rt, loader, &name, false)?;
        let module = module.bind(py);
        let body = module.getattr("module")?.getattr("body")?;
        for statement in body.try_iter()? {
            let statement = statement?;
            if rt.is_definition(&statement)? {
                definitions.push((statement.unbind(), module.clone().unbind()));
            }
        }
        if definitions.is_empty() {
            return Err(check_error(format!(
                "the module {} defines no type",
                rt.repr(&name)
            )));
        }
        if !strict {
            let mut kept = Vec::new();
            for (definition, module) in definitions {
                if !rt.is_function(definition.bind(py))? {
                    kept.push((definition, module));
                }
            }
            definitions = kept;
            if definitions.is_empty() {
                return Err(check_error("the entry holds no type".to_string()));
            }
        }
        return Ok(definitions);
    }
    for item in value.try_iter()? {
        let item = item?;
        let name: String = item.extract()?;
        let (module_name, definition_name) = rt.split_type_name(&name)?;
        if module_name.is_empty() {
            return Err(check_error(format!(
                "{} is no type name: it is a module and a definition, like 'pkg.mod.Name'",
                rt.repr(&name)
            )));
        }
        let module = load_module(rt, loader, &module_name, false)?;
        let module = module.bind(py);
        let found = rt.module_get_type(module, &definition_name)?;
        if rt.is_err(&found)? {
            let msg = rt.err_msg(&found)?;
            return Err(check_error(format!("no type {}: {msg}", rt.repr(&name))));
        }
        let target = rt.ok_value(&found)?;
        let target = target.bind(py);
        if !rt.is_ast_node_type(target)? {
            return Err(check_error(format!("{} is no definition", rt.repr(&name))));
        }
        let ast_node = target.getattr("ast_node")?;
        if !rt.is_definition(&ast_node)? {
            return Err(check_error(format!("{} is no definition", rt.repr(&name))));
        }
        definitions.push((
            ast_node.unbind(),
            target.getattr("container_module")?.unbind(),
        ));
    }
    Ok(definitions)
}

pub(crate) fn joined_viba_path(viba_path: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
    let mut parts: Vec<String> = Vec::new();
    for item in viba_path.try_iter()? {
        parts.push(item?.extract()?);
    }
    if parts.is_empty() {
        return Ok(None);
    }
    Ok(Some(parts.join(":")))
}

pub(crate) fn make_runner(py: Python<'_>, joined: Option<&str>) -> PyResult<Py<PyAny>> {
    let interpret = PyModule::import(py, "rs_viba.interpret")?;
    let class = interpret.getattr("_Runner")?;
    match joined {
        Some(path) => Ok(class.call1((path,))?.unbind()),
        None => Ok(class.call1((py.None(),))?.unbind()),
    }
}

pub(crate) fn strict_flag(strict: Option<&Bound<'_, PyAny>>) -> PyResult<bool> {
    match strict {
        Some(value) if !value.is_none() => value.extract::<bool>(),
        _ => Ok(false),
    }
}

/// What a JSON value of an option is: a tag, a literal's kind and value, or a
/// kind with no value of its own.
enum Read {
    Tag(String),
    Literal { kind: String, value: Py<PyAny> },
    Kind(&'static str),
}

impl Read {
    fn kind_and_value(&self) -> (&str, Option<&Py<PyAny>>) {
        match self {
            Read::Tag(_) => ("tag", None),
            Read::Literal { kind, value } => (kind.as_str(), Some(value)),
            Read::Kind(kind) => (kind, None),
        }
    }
}

fn literal_kind(value: &Bound<'_, PyAny>) -> String {
    if value.is_instance_of::<PyBool>() {
        "boolean".to_string()
    } else if value.is_instance_of::<PyInt>() || value.is_instance_of::<PyFloat>() {
        "number".to_string()
    } else if value.is_instance_of::<PyString>() {
        "string".to_string()
    } else {
        "literal".to_string()
    }
}

fn read_name(py: Python<'_>, read: &Read) -> String {
    match read {
        Read::Tag(value) => format!("the tag {}", repr_str(py, value)),
        Read::Literal { value, .. } => format!("the value {}", pyast::repr_of(py, value.bind(py))),
        Read::Kind(kind) => match *kind {
            "number" => "a number".to_string(),
            "string" => "a string".to_string(),
            "boolean" => "a boolean".to_string(),
            "null" => "null".to_string(),
            "array" => "an array".to_string(),
            "object" => "an object".to_string(),
            other => other.to_string(),
        },
    }
}

fn clash(py: Python<'_>, one: &str, other: &str, read: &Read) -> String {
    if let Read::Tag(tag) = read {
        return format!(
            "the sum's options {one} and {other} carry the same tag {}",
            repr_str(py, tag)
        );
    }
    format!(
        "the sum's options {one} and {other} both read as {} in JSON: a tag tells them apart",
        read_name(py, read)
    )
}

/// Can one JSON value be both? Two tags answer yes when they are one tag.
fn same_read(py: Python<'_>, one: Option<&Read>, other: Option<&Read>) -> PyResult<bool> {
    let (Some(one), Some(other)) = (one, other) else {
        return Ok(false);
    };
    match (one, other) {
        (Read::Tag(left), Read::Tag(right)) => Ok(left == right),
        (Read::Tag(_), _) | (_, Read::Tag(_)) => Ok(false),
        _ => {
            let (one_kind, one_value) = one.kind_and_value();
            let (other_kind, other_value) = other.kind_and_value();
            if one_kind != other_kind {
                return Ok(false);
            }
            match (one_value, other_value) {
                (None, _) | (_, None) => Ok(true),
                (Some(left), Some(right)) => left.bind(py).eq(right.bind(py)),
            }
        }
    }
}

struct Checker<'a, 'py> {
    rt: &'a Runtime<'py>,
    loader: Bound<'py, PyAny>,
    config: Config,
    visited: Bound<'py, PySet>,
}

impl<'a, 'py> Checker<'a, 'py> {
    fn check(&mut self, definitions: &[(Py<PyAny>, Py<PyAny>)]) -> WResult<()> {
        let py = self.rt.py;
        for (definition, module) in definitions {
            self.definition(definition.bind(py), module.bind(py), None)?;
        }
        Ok(())
    }

    fn definition(
        &mut self,
        definition: &Bound<'py, PyAny>,
        module: &Bound<'py, PyAny>,
        bindings: Option<&Bound<'py, PyAny>>,
    ) -> WResult<()> {
        let py = self.rt.py;
        let base = match bindings {
            Some(value) => value.clone(),
            None => PyDict::new(py).into_any(),
        };
        let key = PyTuple::new(
            py,
            [
                module.getattr("file_path")?.unbind(),
                definition.getattr("name")?.unbind(),
                self.rt.bindings_key(&base)?,
            ],
        )?
        .into_any();
        if self.visited.contains(key.clone())? {
            return Ok(());
        }
        self.visited.add(key)?;
        let inner = self.rt.with_params(definition, &base)?;
        let body = definition.getattr("body")?;
        self.walk(&body, module, inner.bind(py))?;
        Ok(())
    }

    fn walk(
        &mut self,
        node: &Bound<'py, PyAny>,
        module: &Bound<'py, PyAny>,
        bindings: &Bound<'py, PyAny>,
    ) -> WResult<()> {
        if node.is_none() {
            return Ok(());
        }
        let class = self.rt.class_name(node)?;
        match class.as_str() {
            "Constant" | "Nil" | "Never" => return Ok(()),
            "Any" | "Ellipsis" => {
                return Err(check_error(format!(
                    "{} is no JSON value",
                    self.rt.spell(node)
                )))
            }
            "HintBlock" => return Ok(()),
            "Member" => {
                return Err(check_error(format!(
                    "{} is no JSON value",
                    self.rt.spell(node)
                )))
            }
            "TypeRef" => {
                let name: String = node.getattr("name")?.extract()?;
                return self.name(&name, module, bindings);
            }
            "MemberTaken" => return self.member(node, module, bindings),
            "Tagged" => {
                let body = node.getattr("type")?;
                return self.walk(&body, module, bindings);
            }
            "TypeApp" => return self.application(node, module, bindings),
            "Tuple" => {
                for element in pyast::field_list(node, "elements")? {
                    self.walk(&element, module, bindings)?;
                }
                return Ok(());
            }
            "Partial" => return Err(check_error("a call is no JSON value".to_string())),
            "Exponent" => return Err(check_error("a function is no JSON value".to_string())),
            _ => {}
        }
        let elements = match self.rt.chain_elements(node)? {
            Some(elements) => elements,
            None => {
                return Err(check_error(format!(
                    "{} is no JSON value",
                    self.rt.spell(node)
                )))
            }
        };
        if class == "Sum" || class == "SumChain" {
            return self.sum(node, &elements, module, bindings);
        }
        for element in &elements {
            self.walk(element, module, bindings)?;
        }
        Ok(())
    }

    fn name(
        &mut self,
        name: &str,
        module: &Bound<'py, PyAny>,
        bindings: &Bound<'py, PyAny>,
    ) -> WResult<()> {
        let py = self.rt.py;
        let dict = bindings.cast::<PyDict>()?;
        if let Some(bound) = dict.get_item(name)? {
            if bound.is_none() {
                return Err(check_error(format!(
                    "the generic parameter {} has no argument: no JSON value yet",
                    self.rt.repr(name)
                )));
            }
            let (node, target_module, target_bindings) =
                binding_parts(&bound)?.ok_or_else(|| {
                    check_error(format!(
                        "the generic parameter {} has no argument: no JSON value yet",
                        self.rt.repr(name)
                    ))
                })?;
            return self.walk(
                node.bind(py),
                target_module.bind(py),
                target_bindings.bind(py),
            );
        }
        if self.config.is_no_value(name) {
            return Ok(());
        }
        if self.config.is_null(name) {
            return Ok(());
        }
        if scalar_read_kind(name).is_some() {
            return Ok(());
        }
        let found = self.rt.module_get_type(module, name)?;
        if self.rt.is_err(&found)? {
            let msg = self.rt.err_msg(&found)?;
            return Err(check_error(format!(
                "cannot resolve {} in {}: {msg}",
                self.rt.repr(name),
                self.rt.module_label(module)?
            )));
        }
        let target = self.rt.ok_value(&found)?;
        let target = target.bind(py);
        if !self.rt.is_ast_node_type(target)? {
            return Ok(());
        }
        let ast_node = target.getattr("ast_node")?;
        if !self.rt.is_definition(&ast_node)? {
            return Ok(());
        }
        self.definition(&ast_node, &target.getattr("container_module")?, None)
    }

    fn application(
        &mut self,
        node: &Bound<'py, PyAny>,
        module: &Bound<'py, PyAny>,
        bindings: &Bound<'py, PyAny>,
    ) -> WResult<()> {
        let constructor: String = node.getattr("constructor")?.extract()?;
        let arguments = pyast::field_list(node, "args")?;
        if constructor == TAGGED_NAME {
            for argument in arguments.iter().skip(1) {
                self.walk(argument, module, bindings)?;
            }
            return Ok(());
        }
        if is_array_constructor(&constructor) || is_object_constructor(&constructor) {
            for argument in &arguments {
                self.walk(argument, module, bindings)?;
            }
            if is_object_constructor(&constructor) {
                let key = arguments.first().cloned().ok_or_else(|| {
                    check_error(format!(
                        "a JSON object's keys are strings: {} is not",
                        self.rt.spell(node)
                    ))
                })?;
                let (form, form_module, form_bindings) = self.stands_for(&key, module, bindings)?;
                let read = self.read_as(&form, &form_module, &form_bindings)?;
                let is_string = matches!(read, Some(Read::Kind("string")))
                    || matches!(&read, Some(Read::Literal { kind, .. }) if kind == "string");
                if !is_string {
                    return Err(check_error(format!(
                        "a JSON object's keys are strings: {} is not",
                        self.rt.spell(&key)
                    )));
                }
            }
            return Ok(());
        }
        if self.config.is_null(&constructor)
            || self.config.is_no_value(&constructor)
            || scalar_read_kind(&constructor).is_some()
        {
            return Ok(());
        }
        self.note_application(node, module, bindings)
    }

    fn note_application(
        &mut self,
        node: &Bound<'py, PyAny>,
        module: &Bound<'py, PyAny>,
        bindings: &Bound<'py, PyAny>,
    ) -> WResult<()> {
        let py = self.rt.py;
        let constructor: String = node.getattr("constructor")?.extract()?;
        let arguments = pyast::field_list(node, "args")?;
        let generic = self.rt.generic_named(module, &constructor)?;
        if self.rt.is_err(&generic)? {
            return Err(check_error(format!(
                "cannot take {} in {}: {}",
                self.rt.repr(&constructor),
                self.rt.module_label(module)?,
                self.rt.err_msg(&generic)?
            )));
        }
        let generic_value = self.rt.ok_value(&generic)?;
        if generic_value.bind(py).is_none() {
            let found = self.rt.module_get_type(module, &constructor)?;
            if self.rt.is_err(&found)? {
                return Err(check_error(format!(
                    "cannot resolve {} in {}: {}",
                    self.rt.repr(&constructor),
                    self.rt.module_label(module)?,
                    self.rt.err_msg(&found)?
                )));
            }
            let target = self.rt.ok_value(&found)?;
            let target = target.bind(py);
            if !self.rt.is_ast_node_type(target)? {
                return Err(check_error(format!(
                    "{} is no JSON value",
                    self.rt.spell(node)
                )));
            }
            let definition = target.getattr("ast_node")?;
            if !self.rt.is_definition(&definition)? {
                return Err(check_error(format!(
                    "{} is no JSON value",
                    self.rt.spell(node)
                )));
            }
            let params = generic_params(&definition)?;
            if params.len() != arguments.len() {
                return Err(check_error(format!(
                    "{}: {} takes {} parameters",
                    self.rt.spell(node),
                    self.rt
                        .repr(&definition.getattr("name")?.extract::<String>()?),
                    params.len()
                )));
            }
            let inner = bindings.cast::<PyDict>()?.copy()?;
            let container = target.getattr("container_module")?;
            for (param, argument) in params.iter().zip(arguments.iter()) {
                let parts = PyTuple::new(
                    py,
                    [
                        argument.clone().unbind(),
                        module.clone().unbind(),
                        bindings.clone().unbind(),
                    ],
                )?;
                inner.set_item(param.as_str(), parts)?;
            }
            return self.definition(&definition, &container, Some(inner.as_any()));
        }
        let args = node.getattr("args")?;
        let argument_modules = self.argument_modules(node, bindings)?;
        let got = self.rt.decide(
            generic_value.bind(py),
            &args,
            module,
            argument_modules.as_ref().map(|value| value.bind(py)),
        )?;
        if self.rt.is_err(&got)? {
            return Err(check_error(format!(
                "cannot decide {}: {}",
                self.rt.spell(node),
                self.rt.err_msg(&got)?
            )));
        }
        let choice = self.rt.ok_value(&got)?;
        let choice = choice.bind(py);
        if choice.is_none() || choice.getattr("body")?.is_none() {
            return Err(check_error(format!(
                "{}: the chosen file answers a member, not a type",
                self.rt.spell(node)
            )));
        }
        let inner = bindings.cast::<PyDict>()?.copy()?;
        bind_choice_bindings(py, &inner, choice, bindings)?;
        let body = choice.getattr("body")?;
        self.walk(&body, &choice.getattr("module")?, inner.as_any())
    }

    fn member(
        &mut self,
        node: &Bound<'py, PyAny>,
        module: &Bound<'py, PyAny>,
        bindings: &Bound<'py, PyAny>,
    ) -> WResult<()> {
        let py = self.rt.py;
        if let Some(source_path) = self.rt.source_path(node)? {
            return self.name(&source_path, module, bindings);
        }
        let owner = node.getattr("owner")?;
        if !self.rt.is_ast(&owner, "TypeApp")? {
            return Err(check_error(format!(
                "{} is no JSON value",
                self.rt.spell(node)
            )));
        }
        let constructor: String = owner.getattr("constructor")?.extract()?;
        let generic = self.rt.generic_named(module, &constructor)?;
        if self.rt.is_err(&generic)? {
            return Err(check_error(format!(
                "cannot take {} in {}: {}",
                self.rt.repr(&constructor),
                self.rt.module_label(module)?,
                self.rt.err_msg(&generic)?
            )));
        }
        let generic_value = self.rt.ok_value(&generic)?;
        if generic_value.bind(py).is_none() {
            return Err(check_error(format!(
                "{} has no JSON form of its own",
                self.rt.spell(node)
            )));
        }
        let args = owner.getattr("args")?;
        let argument_modules = self.argument_modules(&owner, bindings)?;
        let got = self.rt.decide(
            generic_value.bind(py),
            &args,
            module,
            argument_modules.as_ref().map(|value| value.bind(py)),
        )?;
        if self.rt.is_err(&got)? {
            return Err(check_error(format!(
                "cannot decide {}: {}",
                self.rt.spell(&owner),
                self.rt.err_msg(&got)?
            )));
        }
        let choice = self.rt.ok_value(&got)?;
        let choice = choice.bind(py);
        if choice.is_none() {
            return Err(check_error(format!(
                "{} has no JSON form of its own",
                self.rt.spell(node)
            )));
        }
        let inner = bindings.cast::<PyDict>()?.copy()?;
        bind_choice_bindings(py, &inner, choice, bindings)?;
        if !choice.getattr("body")?.is_none() {
            return Err(check_error(format!(
                "{} has no JSON form of its own",
                self.rt.spell(node)
            )));
        }
        let name: String = node.getattr("name")?.extract()?;
        let body = choice
            .getattr("module")?
            .getattr("module")?
            .getattr("body")?;
        for statement in body.try_iter()? {
            let statement = statement?;
            if self.rt.is_definition(&statement)?
                && statement.getattr("name")?.extract::<String>()? == name
            {
                return self.definition(
                    &statement,
                    &choice.getattr("module")?,
                    Some(inner.as_any()),
                );
            }
        }
        Err(check_error(format!(
            "no definition {} in {}",
            self.rt.repr(&name),
            self.rt.module_label(&choice.getattr("module")?)?
        )))
    }

    fn argument_modules(
        &mut self,
        node: &Bound<'py, PyAny>,
        bindings: &Bound<'py, PyAny>,
    ) -> PyResult<Option<Py<PyAny>>> {
        argument_modules(self.rt, node, bindings)
    }

    fn sum(
        &mut self,
        node: &Bound<'py, PyAny>,
        elements: &[Bound<'py, PyAny>],
        module: &Bound<'py, PyAny>,
        bindings: &Bound<'py, PyAny>,
    ) -> WResult<()> {
        let py = self.rt.py;
        let mut options: Vec<(Py<PyAny>, Py<PyAny>, Py<PyAny>, String)> = Vec::new();
        for element in elements {
            let (form, form_module, form_bindings) = self.stands_for(element, module, bindings)?;
            let mut seen: HashSet<usize> = HashSet::new();
            seen.insert(node.as_ptr() as usize);
            options.extend(self.flatten(
                &form,
                &form_module,
                &form_bindings,
                &seen,
                &self.rt.spell(element),
            )?);
        }
        let mut reads: Vec<(String, Option<Read>)> = Vec::new();
        for (form, form_module, form_bindings, spelling) in &options {
            let read = self.read_as(form.bind(py), form_module.bind(py), form_bindings.bind(py))?;
            for (other_spelling, other_read) in &reads {
                if same_read(py, read.as_ref(), other_read.as_ref())? {
                    let read = read.as_ref().ok_or_else(|| {
                        check_error(format!(
                            "the sum's options {other_spelling} and {spelling} both read as nothing"
                        ))
                    })?;
                    return Err(check_error(clash(py, other_spelling, spelling, read)));
                }
            }
            reads.push((spelling.clone(), read));
        }
        for element in elements {
            self.walk(element, module, bindings)?;
        }
        Ok(())
    }

    fn flatten(
        &mut self,
        form: &Bound<'py, PyAny>,
        module: &Bound<'py, PyAny>,
        bindings: &Bound<'py, PyAny>,
        seen: &HashSet<usize>,
        spelling: &str,
    ) -> WResult<Vec<(Py<PyAny>, Py<PyAny>, Py<PyAny>, String)>> {
        if !(self.rt.is_ast(form, "Sum")? || self.rt.is_ast(form, "SumChain")?) {
            return Ok(vec![(
                form.clone().unbind(),
                module.clone().unbind(),
                bindings.clone().unbind(),
                spelling.to_string(),
            )]);
        }
        let id = form.as_ptr() as usize;
        if seen.contains(&id) {
            return Ok(Vec::new());
        }
        let mut inner_seen = seen.clone();
        inner_seen.insert(id);
        let mut options = Vec::new();
        let elements = self.rt.chain_elements(form)?.unwrap_or_default();
        for element in &elements {
            let (inner, inner_module, inner_bindings) =
                self.stands_for(element, module, bindings)?;
            options.extend(self.flatten(
                &inner,
                &inner_module,
                &inner_bindings,
                &inner_seen,
                &self.rt.spell(element),
            )?);
        }
        Ok(options)
    }

    fn stands_for(
        &mut self,
        node: &Bound<'py, PyAny>,
        module: &Bound<'py, PyAny>,
        bindings: &Bound<'py, PyAny>,
    ) -> WResult<(Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        let py = self.rt.py;
        let mut seen: HashSet<usize> = HashSet::new();
        let mut node = node.clone();
        let mut module = module.clone();
        let mut bindings = bindings.clone();
        loop {
            let id = node.as_ptr() as usize;
            if seen.contains(&id) {
                return Ok((node, module, bindings));
            }
            seen.insert(id);
            let class = self.rt.class_name(&node)?;
            if class == "TypeRef" {
                let name: String = node.getattr("name")?.extract()?;
                let dict = bindings.cast::<PyDict>()?;
                if let Some(bound) = dict.get_item(name.as_str())? {
                    if let Some((inner, inner_module, inner_bindings)) = binding_parts(&bound)? {
                        node = inner.bind(py).clone();
                        module = inner_module.bind(py).clone();
                        bindings = inner_bindings.bind(py).clone();
                        continue;
                    }
                }
                if dict.contains(name.as_str())? {
                    return Ok((node, module, bindings));
                }
                if self.config.in_nil_eqv(&name) {
                    return Ok((node, module, bindings));
                }
                if self.config.in_never_eqv(&name) || scalar_read_kind(&name).is_some() {
                    return Ok((node, module, bindings));
                }
                let found = self.rt.module_get_type(&module, &name)?;
                if self.rt.is_err(&found)? {
                    return Err(check_error(format!(
                        "cannot resolve {} in {}: {}",
                        self.rt.repr(&name),
                        self.rt.module_label(&module)?,
                        self.rt.err_msg(&found)?
                    )));
                }
                let target = self.rt.ok_value(&found)?;
                let target = target.bind(py);
                if !self.rt.is_ast_node_type(target)? {
                    return Ok((node, module, bindings));
                }
                let definition = target.getattr("ast_node")?;
                if !self.rt.is_definition(&definition)? {
                    return Ok((node, module, bindings));
                }
                node = definition.getattr("body")?;
                module = target.getattr("container_module")?;
                let empty = PyDict::new(py);
                bindings = self
                    .rt
                    .with_params(&definition, empty.as_any())?
                    .bind(py)
                    .clone();
                continue;
            }
            if class == "MemberTaken" {
                if let Some(source_path) = self.rt.source_path(&node)? {
                    let type_ref = self.rt.ast_type_ref(&source_path)?;
                    node = type_ref.bind(py).clone();
                    continue;
                }
                return Ok((node, module, bindings));
            }
            if class == "TypeApp" {
                let constructor: String = node.getattr("constructor")?.extract()?;
                if constructor == TAGGED_NAME
                    || is_array_constructor(&constructor)
                    || is_object_constructor(&constructor)
                    || self.config.is_null(&constructor)
                    || self.config.is_no_value(&constructor)
                {
                    return Ok((node, module, bindings));
                }
                let generic = self.rt.generic_named(&module, &constructor)?;
                if self.rt.is_err(&generic)? {
                    return Err(check_error(format!(
                        "cannot take {} in {}: {}",
                        self.rt.repr(&constructor),
                        self.rt.module_label(&module)?,
                        self.rt.err_msg(&generic)?
                    )));
                }
                let generic_value = self.rt.ok_value(&generic)?;
                if !generic_value.bind(py).is_none() {
                    let args = node.getattr("args")?;
                    let argument_modules = self.argument_modules(&node, &bindings)?;
                    let got = self.rt.decide(
                        generic_value.bind(py),
                        &args,
                        &module,
                        argument_modules.as_ref().map(|value| value.bind(py)),
                    )?;
                    if self.rt.is_err(&got)? {
                        return Err(check_error(format!(
                            "cannot decide {}: {}",
                            self.rt.spell(&node),
                            self.rt.err_msg(&got)?
                        )));
                    }
                    let choice = self.rt.ok_value(&got)?;
                    let choice = choice.bind(py);
                    if choice.is_none() || choice.getattr("body")?.is_none() {
                        return Ok((node, module, bindings));
                    }
                    let inner = bindings.cast::<PyDict>()?.copy()?;
                    bind_choice_bindings(py, &inner, choice, &bindings)?;
                    node = choice.getattr("body")?;
                    module = choice.getattr("module")?;
                    bindings = inner.into_any();
                    continue;
                }
                let found = self.rt.module_get_type(&module, &constructor)?;
                if self.rt.is_err(&found)? {
                    return Err(check_error(format!(
                        "cannot resolve {} in {}: {}",
                        self.rt.repr(&constructor),
                        self.rt.module_label(&module)?,
                        self.rt.err_msg(&found)?
                    )));
                }
                let target = self.rt.ok_value(&found)?;
                let target = target.bind(py);
                if !self.rt.is_ast_node_type(target)? {
                    return Ok((node, module, bindings));
                }
                let definition = target.getattr("ast_node")?;
                if !self.rt.is_definition(&definition)? {
                    return Ok((node, module, bindings));
                }
                let params = generic_params(&definition)?;
                let arguments = pyast::field_list(&node, "args")?;
                if params.len() != arguments.len() {
                    return Ok((node, module, bindings));
                }
                let inner = bindings.cast::<PyDict>()?.copy()?;
                for (param, argument) in params.iter().zip(arguments.iter()) {
                    let parts = PyTuple::new(
                        py,
                        [
                            argument.clone().unbind(),
                            module.clone().unbind(),
                            bindings.clone().unbind(),
                        ],
                    )?;
                    inner.set_item(param.as_str(), parts)?;
                }
                let container = target.getattr("container_module")?;
                let with = self.rt.with_params(&definition, inner.as_any())?;
                node = definition.getattr("body")?;
                module = container;
                bindings = with.bind(py).clone();
                continue;
            }
            return Ok((node, module, bindings));
        }
    }

    fn read_as(
        &mut self,
        form: &Bound<'py, PyAny>,
        module: &Bound<'py, PyAny>,
        bindings: &Bound<'py, PyAny>,
    ) -> WResult<Option<Read>> {
        let py = self.rt.py;
        let class = self.rt.class_name(form)?;
        match class.as_str() {
            "Tagged" => {
                let tag: String = form.getattr("tag")?.extract()?;
                return Ok(Some(Read::Tag(tag)));
            }
            "Constant" => {
                let value = form.getattr("value")?;
                let kind = literal_kind(&value);
                return Ok(Some(Read::Literal {
                    kind,
                    value: value.unbind(),
                }));
            }
            "Nil" | "HintBlock" => return Ok(Some(Read::Kind("null"))),
            "Never" => return Ok(None),
            "Product" | "ProductChain" => {
                let elements = self.rt.chain_elements(form)?.unwrap_or_default();
                return self.read_product(&elements, module, bindings);
            }
            "Tuple" => return Ok(Some(Read::Kind("array"))),
            "TypeRef" => {
                let name: String = form.getattr("name")?.extract()?;
                let dict = bindings.cast::<PyDict>()?;
                if let Some(bound) = dict.get_item(name.as_str())? {
                    if bound.is_none() {
                        return Err(check_error(format!(
                            "the generic parameter {} has no argument: no JSON value yet",
                            self.rt.repr(&name)
                        )));
                    }
                    let (inner, inner_module, inner_bindings) =
                        binding_parts(&bound)?.ok_or_else(|| {
                            check_error(format!(
                                "the generic parameter {} has no argument: no JSON value yet",
                                self.rt.repr(&name)
                            ))
                        })?;
                    return self.read_as(
                        inner.bind(py),
                        inner_module.bind(py),
                        inner_bindings.bind(py),
                    );
                }
                if self.config.is_null(&name) {
                    return Ok(Some(Read::Kind("null")));
                }
                if self.config.is_no_value(&name) {
                    return Ok(None);
                }
                if let Some(kind) = scalar_read_kind(&name) {
                    return Ok(Some(Read::Kind(kind)));
                }
                return self.read_resolved(&name, module);
            }
            "TypeApp" => {
                let constructor: String = form.getattr("constructor")?.extract()?;
                if is_array_constructor(&constructor) {
                    return Ok(Some(Read::Kind("array")));
                }
                if is_object_constructor(&constructor) {
                    return Ok(Some(Read::Kind("object")));
                }
                if self.config.is_null(&constructor) {
                    return Ok(Some(Read::Kind("null")));
                }
                if self.config.is_no_value(&constructor) {
                    return Ok(None);
                }
                if constructor == TAGGED_NAME {
                    let arguments = pyast::field_list(form, "args")?;
                    let Some(first) = arguments.first() else {
                        return Ok(None);
                    };
                    let symbol = self.symbol(first, bindings)?;
                    return Ok(symbol.map(Read::Tag));
                }
                return Ok(None);
            }
            _ => return Ok(None),
        }
    }

    fn read_product(
        &mut self,
        elements: &[Bound<'py, PyAny>],
        module: &Bound<'py, PyAny>,
        bindings: &Bound<'py, PyAny>,
    ) -> WResult<Option<Read>> {
        let mut kept: Vec<Bound<'py, PyAny>> = Vec::new();
        for element in elements {
            if self.rt.is_ast(element, "Tagged")? {
                kept.push(element.clone());
                continue;
            }
            if self.is_unit(element)? {
                continue;
            }
            let (form, _form_module, _form_bindings) =
                self.stands_for(element, module, bindings)?;
            if self.is_unit(&form)? {
                continue;
            }
            kept.push(element.clone());
        }
        if kept.is_empty() {
            return Ok(Some(Read::Kind("null")));
        }
        if kept.len() == 1 && !self.rt.is_ast(&kept[0], "Tagged")? {
            let element = kept[0].clone();
            return self.read_as(&element, module, bindings);
        }
        Ok(Some(Read::Kind("object")))
    }

    fn is_unit(&mut self, form: &Bound<'py, PyAny>) -> WResult<bool> {
        let class = self.rt.class_name(form)?;
        if matches!(class.as_str(), "Nil" | "Never" | "HintBlock") {
            return Ok(true);
        }
        if class == "TypeRef" {
            let name: String = form.getattr("name")?.extract()?;
            return Ok(self.config.is_unit(&name));
        }
        if class == "TypeApp" {
            let name: String = form.getattr("constructor")?.extract()?;
            return Ok(self.config.is_unit(&name));
        }
        Ok(false)
    }

    fn read_resolved(&mut self, name: &str, module: &Bound<'py, PyAny>) -> WResult<Option<Read>> {
        let py = self.rt.py;
        let found = self.rt.module_get_type(module, name)?;
        if self.rt.is_err(&found)? {
            let msg = self.rt.err_msg(&found)?;
            return Err(check_error(format!(
                "cannot resolve {} in {}: {msg}",
                self.rt.repr(name),
                self.rt.module_label(module)?
            )));
        }
        let target = self.rt.ok_value(&found)?;
        let target = target.bind(py);
        if !self.rt.is_ast_node_type(target)? {
            return Ok(None);
        }
        let ast_node = target.getattr("ast_node")?;
        if !self.rt.is_definition(&ast_node)? {
            return Ok(None);
        }
        let type_ref = self.rt.ast_type_ref(name)?;
        let empty = PyDict::new(py);
        let (form, form_module, form_bindings) =
            self.stands_for(type_ref.bind(py), module, empty.as_any())?;
        self.read_as(&form, &form_module, &form_bindings)
    }

    fn symbol(
        &mut self,
        node: &Bound<'py, PyAny>,
        bindings: &Bound<'py, PyAny>,
    ) -> WResult<Option<String>> {
        let py = self.rt.py;
        if self.rt.is_ast(node, "Constant")? {
            let value = node.getattr("value")?;
            if value.is_instance_of::<PyString>() {
                return Ok(Some(value.extract()?));
            }
        }
        if self.rt.is_ast(node, "TypeRef")? {
            let name: String = node.getattr("name")?.extract()?;
            if let Some(bound) = bindings.cast::<PyDict>()?.get_item(name.as_str())? {
                if let Some((inner, _module, inner_bindings)) = binding_parts(&bound)? {
                    return self.symbol(inner.bind(py), inner_bindings.bind(py));
                }
            }
        }
        Ok(None)
    }
}

/// The module each argument was spelled in, where a decision bound it.
pub(crate) fn argument_modules<'py>(
    rt: &Runtime<'py>,
    node: &Bound<'py, PyAny>,
    bindings: &Bound<'py, PyAny>,
) -> PyResult<Option<Py<PyAny>>> {
    let py = rt.py;
    let dict = bindings.cast::<PyDict>()?;
    let mut kept: Vec<Py<PyAny>> = Vec::new();
    let mut any = false;
    for argument in pyast::field_list(node, "args")? {
        let mut where_module: Option<Py<PyAny>> = None;
        if rt.is_ast(&argument, "TypeRef")? {
            let name: String = argument.getattr("name")?.extract()?;
            if let Some(bound) = dict.get_item(name.as_str())? {
                if let Some((_node, module, _bindings)) = binding_parts(&bound)? {
                    where_module = Some(module);
                }
            }
        }
        any = any || where_module.is_some();
        kept.push(where_module.unwrap_or_else(|| py.None()));
    }
    if !any {
        return Ok(None);
    }
    Ok(Some(list_obj(py, kept)?))
}

/// The bindings a decision hands over, as the walk's inner bindings.
pub(crate) fn bind_choice_bindings(
    py: Python<'_>,
    inner: &Bound<'_, PyDict>,
    choice: &Bound<'_, PyAny>,
    outer: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let bindings = choice.getattr("bindings")?;
    let bindings = bindings.cast::<PyDict>()?;
    for (name, bound) in bindings.iter() {
        let ast_node = bound.getattr("ast_node")?;
        let container = bound.getattr("container_module")?;
        let parts = PyTuple::new(
            py,
            [
                ast_node.unbind(),
                container.unbind(),
                outer.clone().unbind(),
            ],
        )?;
        inner.set_item(name, parts)?;
    }
    Ok(())
}

// ----------------------------------------------------------------------
// The design: the modules the walk loaded, for the product check
// ----------------------------------------------------------------------

fn read_text(py: Python<'_>, path: &str) -> PyResult<String> {
    let pathlib = PyModule::import(py, "pathlib")?;
    let path_obj = pathlib.getattr("Path")?.call1((path,))?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("encoding", "utf-8")?;
    path_obj
        .call_method("read_text", (), Some(&kwargs))?
        .extract()
}

fn read_text_skipping(py: Python<'_>, path: &str) -> PyResult<Option<String>> {
    match read_text(py, path) {
        Ok(text) => Ok(Some(text)),
        Err(error) => {
            if error.is_instance_of::<PyOSError>(py) {
                Ok(None)
            } else {
                Err(error)
            }
        }
    }
}

fn module_files_skipping(
    py: Python<'_>,
    module_name: &str,
    source: &str,
) -> PyResult<Option<Vec<(String, String, String)>>> {
    let markdown = PyModule::import(py, "rs_viba.markdown")?;
    let sections = match markdown
        .getattr("module_files")?
        .call1((module_name, source))
    {
        Ok(sections) => sections,
        Err(error) => {
            if error.is_instance_of::<PyOSError>(py) || error.is_instance_of::<PySyntaxError>(py) {
                return Ok(None);
            }
            return Err(error);
        }
    };
    let mut out = Vec::new();
    for item in sections.try_iter()? {
        let item = item?;
        let tuple = item.cast::<PyTuple>()?;
        out.push((
            tuple.get_item(0)?.extract()?,
            tuple.get_item(1)?.extract()?,
            tuple.get_item(2)?.extract()?,
        ));
    }
    Ok(Some(out))
}

fn add_source<'py>(
    rt: &Runtime<'py>,
    pool: &Bound<'py, PyAny>,
    source: &str,
    file_path: &str,
    module_name: &str,
) -> PyResult<Option<Py<PyAny>>> {
    let py = rt.py;
    let descriptor = PyModule::import(py, "rs_viba.viba_type_descriptor")?;
    let parsed =
        descriptor
            .getattr("parse_viba_file")?
            .call1((pool, source, file_path, module_name))?;
    if rt.is_err(&parsed)? {
        return Ok(None);
    }
    let added = descriptor
        .getattr("pool_add_file")?
        .call1((pool, rt.ok_value(&parsed)?))?;
    if rt.is_err(&added)? {
        return Ok(None);
    }
    Ok(Some(rt.ok_value(&added)?))
}

/// A pool of the modules this walk loaded, for check_tag_and_inline.
fn design_pool<'py>(rt: &Runtime<'py>, loader: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
    let py = rt.py;
    let descriptor = PyModule::import(py, "rs_viba.viba_type_descriptor")?;
    let mut pool = descriptor.getattr("empty_pool")?.call0()?.unbind();
    let mut files: HashSet<String> = HashSet::new();
    let by_name = loader.getattr("by_name")?;
    let by_name = by_name.cast::<PyDict>()?;
    let mut items: Vec<(String, Py<PyAny>)> = Vec::new();
    for (key, value) in by_name.iter() {
        items.push((key.extract()?, value.unbind()));
    }
    for (name, module) in items {
        let module = module.bind(py);
        if rt.is_generic_module(module)? {
            let directory: String = module.getattr("directory")?.str()?.extract()?;
            if !Path::new(&directory).is_dir() {
                // A generic written as one Markdown document: its sections are
                // the files its directory would hold.
                if files.contains(&directory) || rt.under_builtin(&directory)? {
                    continue;
                }
                files.insert(directory.clone());
                let Some(source) = read_text_skipping(py, &directory)? else {
                    continue;
                };
                let Some(sections) = module_files_skipping(py, &name, &source)? else {
                    continue;
                };
                for (file_name, section_name, text) in sections {
                    if let Some(added) = add_source(
                        rt,
                        pool.bind(py),
                        &text,
                        &format!("{directory}#{file_name}"),
                        &section_name,
                    )? {
                        pool = added;
                    }
                }
                continue;
            }
            let marker = Path::new(&directory)
                .join("__generic__.viba")
                .to_string_lossy()
                .to_string();
            if files.contains(&marker) || rt.under_builtin(&marker)? {
                continue;
            }
            let source = read_text(py, &marker)?;
            if let Some(added) = add_source(
                rt,
                pool.bind(py),
                &source,
                &marker,
                &format!("{name}.__generic__"),
            )? {
                pool = added;
                files.insert(marker);
            }
            continue;
        }
        if !rt.is_custom_module(module)? {
            continue;
        }
        let file_path_value = module.getattr("file_path")?;
        if file_path_value.is_none() {
            continue;
        }
        let file_path: String = file_path_value.extract()?;
        if file_path.is_empty() || rt.under_builtin(&file_path)? {
            continue;
        }
        if file_path.ends_with(".viba.md") {
            if files.contains(&file_path) {
                continue;
            }
            files.insert(file_path.clone());
            let prefix = match name.rfind('.') {
                Some(index) => name[..index].to_string(),
                None => String::new(),
            };
            let Some(source) = read_text_skipping(py, &file_path)? else {
                continue;
            };
            let Some(sections) = module_files_skipping(py, &prefix, &source)? else {
                continue;
            };
            for (file_name, section_name, text) in sections {
                if let Some(added) = add_source(
                    rt,
                    pool.bind(py),
                    &text,
                    &format!("{file_path}#{file_name}"),
                    &section_name,
                )? {
                    pool = added;
                }
            }
            continue;
        }
        if files.contains(&file_path) {
            continue;
        }
        if !Path::new(&file_path).is_file() {
            continue;
        }
        let source = read_text(py, &file_path)?;
        if let Some(added) = add_source(rt, pool.bind(py), &source, &file_path, &name)? {
            pool = added;
            files.insert(file_path);
        }
    }
    rt.make_ok(&pool)
}

fn judge_walk<'py>(
    rt: &Runtime<'py>,
    config: &Config,
    loader: &Bound<'py, PyAny>,
    tag: &str,
    value: &Bound<'py, PyAny>,
    strict: bool,
) -> WResult<Py<PyAny>> {
    let py = rt.py;
    let mut checker = Checker {
        rt,
        loader: loader.clone(),
        config: config.clone(),
        visited: PySet::empty(py)?,
    };
    let mut first: Option<String> = None;
    let outcome = entry_definitions(rt, loader, tag, value, strict)
        .and_then(|definitions| checker.check(&definitions));
    match outcome {
        Ok(()) => {}
        Err(Failure::Check(message)) => first = Some(message),
        Err(other) => return Err(other),
    }
    let design = design_pool(rt, &checker.loader)?;
    if rt.is_err(&design)? {
        return Ok(design);
    }
    let pool = rt.ok_value(&design)?;
    let tag_check = PyModule::import(py, "rs_viba.check_tag_and_inline")?
        .getattr("check_tag_and_inline")?
        .call1((pool,))?;
    if rt.is_err(&tag_check)? {
        return Ok(tag_check.unbind());
    }
    if let Some(message) = first {
        return Ok(rt.make_err(&message)?);
    }
    let none = py.None();
    Ok(rt.make_ok(&none)?)
}

/// `check_convertable_to_json_schema`: Ok(None) or the first mistake, as a Result.
pub(crate) fn judge_entry<'py>(
    py: Python<'py>,
    viba_path: &Bound<'py, PyAny>,
    entry: &Bound<'py, PyAny>,
    strict: Option<&Bound<'py, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let rt = Runtime::new(py)?;
    if let Some(value) = strict {
        if !value.is_none() && !value.is_instance_of::<PyBool>() {
            let message = format!("strict is a bool or nil, not {}", rt.type_name(value)?);
            return rt.make_err(&message);
        }
    }
    if let Some(problem) = argument_problem(py, viba_path, entry)? {
        return rt.make_err(&problem);
    }
    let tuple = entry.cast::<PyTuple>()?;
    let tag: String = tuple.get_item(0)?.extract()?;
    let tag = tag.trim_start_matches('$').to_string();
    let value = tuple.get_item(1)?;
    let joined = joined_viba_path(viba_path)?;
    let loader = make_runner(py, joined.as_deref())?;
    let strict = strict_flag(strict)?;
    let config = language_config(&rt)?;
    match judge_walk(&rt, &config, loader.bind(py), &tag, &value, strict) {
        Ok(result) => Ok(result),
        Err(Failure::Py(error)) => Err(error),
        Err(other) => rt.make_err(&other.message()),
    }
}

#[pyfunction]
#[pyo3(signature = (viba_path, entry, strict = None))]
fn check_convertable_to_json_schema_py<'py>(
    py: Python<'py>,
    viba_path: &Bound<'py, PyAny>,
    entry: &Bound<'py, PyAny>,
    strict: Option<&Bound<'py, PyAny>>,
) -> PyResult<Py<PyAny>> {
    judge_entry(py, viba_path, entry, strict)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(check_convertable_to_json_schema_py, m)?)?;
    Ok(())
}
