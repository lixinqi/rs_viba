//! VibaTypeDescriptor — the descriptor side of the reflection protocol
//! (was `viba/viba_type_descriptor.py`).
//!
//! The protocol's data classes live in `rs_viba.viba_type_descriptor`; this
//! module keeps the algorithm: building a file's descriptors, rebuilding a
//! pool when one more file is added, and answering the descriptor queries.

use std::collections::HashSet;

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyModule, PyString, PyTuple};

use crate::ctx::Ctx;
use crate::errors::{VibaError, VibaResult};
use crate::partial::{self, JudgeFn, ResolveFn};
use crate::pyast;
use crate::tagged;
use crate::type_system;

// ----------------------------------------------------------------------
// Small helpers over the Python objects
// ----------------------------------------------------------------------

fn is_class(node: &Bound<'_, PyAny>, name: &str) -> bool {
    pyast::class_name(node)
        .map(|found| found == name)
        .unwrap_or(false)
}

fn optional_string(value: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
    if value.is_none() {
        Ok(None)
    } else {
        Ok(Some(value.extract()?))
    }
}

/// Python's `repr` of a string, for the error sentences the original spells
/// with `!r`.
fn repr_str(py: Python<'_>, text: &str) -> String {
    let value = PyString::new(py, text);
    pyast::repr_of(py, value.as_any())
}

fn type_module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    PyModule::import(py, "rs_viba.type")
}

fn type_class<'py>(py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    Ok(type_module(py)?.getattr(name)?)
}

fn ast_node_type<'py>(
    py: Python<'py>,
    node: &Bound<'py, PyAny>,
    module: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    Ok(type_class(py, "AstNodeType")?
        .call1((node, module))?
        .unbind())
}

fn builtin_module(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    type_class(py, "BUILTIN_MODULE")
}

fn custom_module_type<'py>(
    py: Python<'py>,
    tree: &Bound<'py, PyAny>,
    environment: &Bound<'py, PyAny>,
    imports: Py<PyAny>,
) -> PyResult<Py<PyAny>> {
    Ok(type_class(py, "CustomModuleType")?
        .call1((tree, environment, imports))?
        .unbind())
}

/// `value.ok_value` when `value` is an `Ok`, else `None`.
fn ok_value<'py>(value: &Bound<'py, PyAny>) -> PyResult<Option<Bound<'py, PyAny>>> {
    if type_system::is_err(value)? {
        Ok(None)
    } else {
        Ok(Some(value.getattr("ok_value")?))
    }
}

/// A design error as the exception the original raises: a `PartialError` for a
/// design mistake, a `TypeError` for a syntax node no branch covers.
fn design_error(py: Python<'_>, error: VibaError) -> PyErr {
    match error {
        VibaError::Program(message) => pyo3::exceptions::PyTypeError::new_err(message),
        other => other.into_pyerr(py),
    }
}

fn pattern_function<'py>(py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    Ok(PyModule::import(py, "rs_viba.pattern")?.getattr(name)?)
}

// ----------------------------------------------------------------------
// The contract: the Python classes, constants and units
// ----------------------------------------------------------------------

struct Contract<'py> {
    module: Bound<'py, PyModule>,
}

impl<'py> Contract<'py> {
    fn new(py: Python<'py>) -> PyResult<Self> {
        Ok(Self {
            module: PyModule::import(py, "rs_viba.viba_type_descriptor")?,
        })
    }

    fn py(&self) -> Python<'py> {
        self.module.py()
    }

    fn class(&self, name: &str) -> PyResult<Bound<'py, PyAny>> {
        Ok(self.module.getattr(name)?)
    }

    fn make(&self, name: &str, fields: &[(&str, Py<PyAny>)]) -> PyResult<Py<PyAny>> {
        let py = self.py();
        let kwargs = PyDict::new(py);
        for (key, value) in fields {
            kwargs.set_item(*key, value.clone_ref(py))?;
        }
        Ok(self.class(name)?.call((), Some(&kwargs))?.unbind())
    }

    fn type_descriptor(&self, kind: &str, payload: Py<PyAny>) -> PyResult<Py<PyAny>> {
        let py = self.py();
        self.make(
            "VibaTypeDescriptor",
            &[("kind", pyast::pystr(py, kind)), ("payload", payload)],
        )
    }
}

/// The branch names and chain-head units, as the Python shim spells them.
struct Kinds {
    type_ref: String,
    member_taken: String,
    type_app: String,
    tuple: String,
    tagged: String,
    sum: String,
    product: String,
    exponent: String,
    literal: String,
    nil: String,
    never: String,
    any: String,
    ellipsis: String,
    hint_block: String,
    product_unit: Vec<String>,
    sum_unit: Vec<String>,
    exponent_unit: Vec<String>,
}

impl Kinds {
    fn load(contract: &Contract<'_>) -> PyResult<Self> {
        let text = |name: &str| -> PyResult<String> { contract.class(name)?.extract() };
        let unit = |name: &str| -> PyResult<Vec<String>> {
            let mut names = Vec::new();
            for item in contract.class(name)?.try_iter()? {
                names.push(item?.extract()?);
            }
            Ok(names)
        };
        Ok(Self {
            type_ref: text("TYPE_REF")?,
            member_taken: text("MEMBER_TAKEN")?,
            type_app: text("TYPE_APP")?,
            tuple: text("TUPLE")?,
            tagged: text("TAGGED")?,
            sum: text("SUM")?,
            product: text("PRODUCT")?,
            exponent: text("EXPONENT")?,
            literal: text("LITERAL")?,
            nil: text("NIL")?,
            never: text("NEVER")?,
            any: text("ANY")?,
            ellipsis: text("ELLIPSIS")?,
            hint_block: text("HINT_BLOCK")?,
            product_unit: unit("PRODUCT_UNIT")?,
            sum_unit: unit("SUM_UNIT")?,
            exponent_unit: unit("EXPONENT_UNIT")?,
        })
    }
}

// ----------------------------------------------------------------------
// Syntax helpers: chains taken as element lists
// ----------------------------------------------------------------------

fn left_elements(
    node: &Bound<'_, PyAny>,
    kind: &str,
    chain_kind: &str,
) -> PyResult<Vec<Py<PyAny>>> {
    if is_class(node, chain_kind) {
        return Ok(pyast::field_list(node, "elements")?
            .into_iter()
            .map(|item| item.unbind())
            .collect());
    }
    let mut elements = Vec::new();
    let mut node = node.clone();
    while is_class(&node, kind) {
        elements.push(node.getattr("right")?.unbind());
        node = node.getattr("left")?;
    }
    elements.push(node.unbind());
    elements.reverse();
    Ok(elements)
}

fn exponent_elements(node: &Bound<'_, PyAny>) -> PyResult<Vec<Py<PyAny>>> {
    if is_class(node, "ExponentChain") {
        return Ok(pyast::field_list(node, "elements")?
            .into_iter()
            .map(|item| item.unbind())
            .collect());
    }
    let mut elements = Vec::new();
    let mut node = node.clone();
    while is_class(&node, "Exponent") {
        elements.push(node.getattr("argument")?.unbind());
        node = node.getattr("result")?;
    }
    elements.push(node.unbind());
    elements.reverse();
    Ok(elements)
}

/// `(chain head, argument list)`: a `<<` chain's arguments in source order.
fn partial_parts<'py>(
    node: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyAny>, Vec<Bound<'py, PyAny>>)> {
    let mut arguments = Vec::new();
    let mut node = node.clone();
    while is_class(&node, "Partial") {
        arguments.push(node.getattr("argument")?);
        node = node.getattr("function")?;
    }
    arguments.reverse();
    Ok((node, arguments))
}

fn source_path(py: Python<'_>, node: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
    let ast = PyModule::import(py, "rs_viba.viba_ast")?;
    let path = ast.getattr("source_path")?.call1((node,))?;
    optional_string(&path)
}

fn is_unit(node: &Bound<'_, PyAny>, names: &[String]) -> bool {
    if !is_class(node, "TypeRef") {
        return false;
    }
    node.getattr("name")
        .and_then(|value| value.extract::<String>())
        .map(|name| names.iter().any(|wanted| wanted == &name))
        .unwrap_or(false)
}

fn generic_params(py: Python<'_>, statement: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let mut items = Vec::new();
    if let Ok(value) = statement.getattr("generic_params") {
        if !value.is_none() {
            for item in value.try_iter()? {
                items.push(item?.unbind());
            }
        }
    }
    pyast::pylist(py, items)
}

fn import_locals_for(py: Python<'_>, tree: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let locals = PyDict::new(py);
    for statement in pyast::field_list(tree, "body")? {
        if !is_class(&statement, "Import") {
            continue;
        }
        let module: String = statement.getattr("module")?.extract()?;
        let alias = optional_string(&statement.getattr("alias")?)?;
        let local = alias.unwrap_or_else(|| module.clone());
        locals.set_item(local, module)?;
    }
    Ok(locals.into_any().unbind())
}

fn import_locals_of_file(file: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let py = file.py();
    let locals = PyDict::new(py);
    for import in file.getattr("imports")?.try_iter()? {
        let import = import?;
        let local: String = import.getattr("local_name")?.extract()?;
        let module: String = import.getattr("module_name")?.extract()?;
        locals.set_item(local, module)?;
    }
    Ok(locals.into_any().unbind())
}

fn normalize_source(source: &str) -> String {
    source.replace("\r\n", "\n").replace('\r', "\n")
}

fn sha256_hex(py: Python<'_>, text: &str) -> PyResult<String> {
    let hashlib = PyModule::import(py, "hashlib")?;
    let digest = hashlib
        .getattr("sha256")?
        .call1((PyBytes::new(py, text.as_bytes()),))?;
    digest.call_method0("hexdigest")?.extract()
}

fn parse_tree(py: Python<'_>, text: &str) -> PyResult<Py<PyAny>> {
    let ast = PyModule::import(py, "rs_viba.viba_ast")?;
    let parsed = ast.getattr("parse")?.call1((text,))?;
    Ok(ast.getattr("canonical")?.call1((parsed,))?.unbind())
}

// ----------------------------------------------------------------------
// The environment: a module name for a module, looked up in the pool
// ----------------------------------------------------------------------

/// The pool's module environment. The original closure captures the pool; this
/// class holds it, so a lookup always sees the pool as it is now.
///
/// The pool holds this environment, so the two form a cycle; `__traverse__` and
/// `__clear__` let Python's collector see it and break it.
#[pyclass(name = "_VibaPoolEnvironment")]
struct PoolEnvironment {
    pool: Option<Py<PyAny>>,
}

#[pymethods]
impl PoolEnvironment {
    fn __call__(&self, py: Python<'_>, module_name: String) -> PyResult<Py<PyAny>> {
        let pool = self.pool.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("the environment has no pool")
        })?;
        environment_lookup(py, pool.bind(py), &module_name)
    }

    fn __traverse__(&self, visit: pyo3::gc::PyVisit<'_>) -> Result<(), pyo3::gc::PyTraverseError> {
        visit.call(&self.pool)
    }

    fn __clear__(&mut self) {
        self.pool = None;
    }
}

fn pool_environment(py: Python<'_>, pool: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    Ok(Py::new(
        py,
        PoolEnvironment {
            pool: Some(pool.clone().unbind()),
        },
    )?
    .into_any())
}

fn new_pool(contract: &Contract<'_>) -> PyResult<Py<PyAny>> {
    let py = contract.py();
    contract.make(
        "VibaPool",
        &[
            ("files", PyList::empty(py).into_any().unbind()),
            ("file_name2file", PyDict::new(py).into_any().unbind()),
            ("full_name2definition", PyDict::new(py).into_any().unbind()),
            ("full_name2member", PyDict::new(py).into_any().unbind()),
            ("module_environment", py.None()),
        ],
    )
}

fn empty_pool(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let contract = Contract::new(py)?;
    let pool = new_pool(&contract)?;
    let environment = pool_environment(py, pool.bind(py))?;
    pool.bind(py).setattr("module_environment", environment)?;
    Ok(pool)
}

fn environment_lookup(
    py: Python<'_>,
    pool: &Bound<'_, PyAny>,
    module_name: &str,
) -> PyResult<Py<PyAny>> {
    let mut matches = Vec::new();
    for file in pyast::field_list(pool, "files")? {
        let name: String = file.getattr("module_name")?.extract()?;
        if name == module_name {
            matches.push(file);
        }
    }
    if matches.is_empty() {
        if let Some(generic) = generic_in_pool(py, pool, module_name)? {
            return Ok(generic);
        }
        if !module_name.contains('.') {
            let concept: String = PyModule::import(py, "rs_viba.type")?
                .getattr("BUILTIN_CONCEPT")?
                .extract()?;
            let builtin_name = format!("{}.{}", concept, module_name);
            if let Some(generic) = generic_in_pool(py, pool, &builtin_name)? {
                return Ok(generic);
            }
        }
        return type_system::err(
            py,
            &format!("no module named {} in pool", repr_str(py, module_name)),
        );
    }
    if matches.len() > 1 {
        return type_system::err(
            py,
            &format!(
                "module {} is served by {} files",
                repr_str(py, module_name),
                matches.len()
            ),
        );
    }
    let file = &matches[0];
    let tree = file.getattr("_tree")?;
    let environment = pool.getattr("module_environment")?;
    let imports = import_locals_of_file(file)?;
    type_system::ok(py, custom_module_type(py, &tree, &environment, imports)?)
}

/// The generic the pool holds under this module name, as a `Result`: `None`
/// when the pool holds none, an `Err` object when a file of it is misnamed.
fn generic_in_pool(
    py: Python<'_>,
    pool: &Bound<'_, PyAny>,
    module_name: &str,
) -> PyResult<Option<Py<PyAny>>> {
    let prefix = format!("{}.", module_name);
    let mut under = Vec::new();
    for file in pyast::field_list(pool, "files")? {
        let name: String = file.getattr("module_name")?.extract()?;
        if name.starts_with(&prefix) {
            under.push(file);
        }
    }
    let generic_file: String = PyModule::import(py, "rs_viba.pattern")?
        .getattr("GENERIC_FILE")?
        .extract()?;
    let marker_stem = generic_file.strip_suffix(".viba").unwrap_or(&generic_file);
    let marker_name = format!("{}{}", prefix, marker_stem);
    let mut marker = None;
    for file in &under {
        let name: String = file.getattr("module_name")?.extract()?;
        if name == marker_name {
            marker = Some(file.clone());
            break;
        }
    }
    let marker = match marker {
        Some(marker) => marker,
        None => return Ok(None),
    };
    let mut parts = Vec::new();
    for file in &under {
        if file.is(&marker) {
            continue;
        }
        let name: String = file.getattr("module_name")?.extract()?;
        let given = &name[prefix.len()..];
        let order = pattern_function(py, "order_of")?.call1((given,))?;
        if order.is_none() {
            let file_name: String = file.getattr("file_name")?.extract()?;
            return Ok(Some(type_system::err(
                py,
                &format!(
                    "{}: a file of the generic {} is named by its order, a number",
                    file_name,
                    repr_str(py, module_name)
                ),
            )?));
        }
        let declared = pattern_function(py, "declared_of")?.call1((given,))?;
        let file_name: String = file.getattr("file_name")?.extract()?;
        let file_module_name: String = file.getattr("module_name")?.extract()?;
        let tree = file.getattr("_tree")?;
        let environment = pool.getattr("module_environment")?;
        let imports = import_locals_of_file(file)?;
        let module = custom_module_type(py, &tree, &environment, imports)?;
        parts.push(
            PyTuple::new(
                py,
                [
                    order.unbind(),
                    pyast::pystr(py, &file_name),
                    pyast::pystr(py, &file_module_name),
                    declared.unbind(),
                    module,
                ],
            )?
            .into_any()
            .unbind(),
        );
    }
    let marker_tree = marker.getattr("_tree")?;
    let marker_file_name: String = marker.getattr("file_name")?.extract()?;
    let generic = pattern_function(py, "generic_of_entries")?.call1((
        module_name,
        marker_file_name,
        marker_tree,
        pyast::pylist(py, parts)?,
    ))?;
    Ok(Some(generic.unbind()))
}

// ----------------------------------------------------------------------
// Building the pool, building descriptors
// ----------------------------------------------------------------------

pub fn parse_viba_file<'py>(
    py: Python<'py>,
    pool: &Bound<'py, PyAny>,
    source: &str,
    file_name: &str,
    module_name: &str,
) -> PyResult<Py<PyAny>> {
    let text = normalize_source(source);
    let tree = match parse_tree(py, &text) {
        Ok(tree) => tree,
        Err(error) => {
            let message = format!(
                "cannot parse: {}",
                pyast::repr_of(py, error.value(py).as_any())
            );
            return type_system::err(py, &message);
        }
    };
    let last = module_name.rsplit('.').next().unwrap_or(module_name);
    let problem =
        pattern_function(py, "file_pattern_problem")?.call1((tree.clone_ref(py), last))?;
    if let Some(problem) = optional_string(&problem)? {
        return type_system::err(py, &problem);
    }
    let file_hash = sha256_hex(py, &text)?;
    let builder = Builder::new(py)?;
    match builder.build_file(pool, tree.bind(py), file_name, module_name, &file_hash) {
        Ok(file) => type_system::ok(py, file),
        Err(VibaError::Partial(message)) => type_system::err(py, &message),
        Err(other) => Err(design_error(py, other)),
    }
}

pub fn pool_add_file(
    py: Python<'_>,
    pool: &Bound<'_, PyAny>,
    file: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let file_pool = file.getattr("pool")?;
    if !file_pool.is(pool) {
        return type_system::err(py, "the file was not built into this pool");
    }
    let file_name: String = file.getattr("file_name")?.extract()?;
    let file_index = pool.getattr("file_name2file")?.cast_into::<PyDict>()?;
    if file_index.contains(file_name.clone())? {
        return type_system::err(py, &format!("duplicate file {}", repr_str(py, &file_name)));
    }
    let builder = Builder::new(py)?;
    let new_pool = new_pool(&builder.contract)?;
    let new_pool_bound = new_pool.bind(py);
    let environment = pool_environment(py, new_pool_bound)?;
    new_pool_bound.setattr("module_environment", environment)?;
    let new_files = new_pool_bound.getattr("files")?.cast_into::<PyList>()?;
    let new_file_index = new_pool_bound
        .getattr("file_name2file")?
        .cast_into::<PyDict>()?;

    let mut stored = pyast::field_list(pool, "files")?;
    stored.push(file.clone());
    let mut rebuilt = Vec::new();
    for old in stored {
        let tree = old.getattr("_tree")?;
        let old_file_name: String = old.getattr("file_name")?.extract()?;
        let old_module_name: String = old.getattr("module_name")?.extract()?;
        let old_hash: String = old.getattr("file_hash")?.extract()?;
        let fresh = builder
            .build_file(
                new_pool_bound,
                &tree,
                &old_file_name,
                &old_module_name,
                &old_hash,
            )
            .map_err(|error| design_error(py, error))?;
        let fresh_bound = fresh.bind(py);
        let fresh_name: String = fresh_bound.getattr("file_name")?.extract()?;
        if new_file_index.contains(fresh_name.clone())? {
            return type_system::err(py, &format!("duplicate file {}", repr_str(py, &fresh_name)));
        }
        new_files.append(fresh_bound)?;
        new_file_index.set_item(fresh_name, fresh_bound)?;
        rebuilt.push(fresh);
    }
    let definitions_index = new_pool_bound
        .getattr("full_name2definition")?
        .cast_into::<PyDict>()?;
    let members_index = new_pool_bound
        .getattr("full_name2member")?
        .cast_into::<PyDict>()?;
    for fresh in &rebuilt {
        for definition in pyast::field_list(fresh.bind(py), "definitions")? {
            let definition_full_name: String = definition.getattr("full_name")?.extract()?;
            if definitions_index.contains(definition_full_name.clone())? {
                return type_system::err(
                    py,
                    &format!(
                        "duplicate definition {}",
                        repr_str(py, &definition_full_name)
                    ),
                );
            }
            definitions_index.set_item(definition_full_name.clone(), &definition)?;
            for member in pyast::field_list(&definition, "members")? {
                let tag = member.getattr("tag")?;
                if tag.is_none() {
                    continue;
                }
                let tag: String = tag.extract()?;
                let full = format!("{}.{}", definition_full_name, tag);
                if members_index.contains(full.clone())? {
                    return type_system::err(
                        py,
                        &format!("duplicate member {}", repr_str(py, &full)),
                    );
                }
                members_index.set_item(full, &member)?;
            }
        }
    }
    type_system::ok(py, new_pool)
}

// ----------------------------------------------------------------------
// Queries on the pool / file / definition / member
// ----------------------------------------------------------------------

pub fn pool_find_file(
    py: Python<'_>,
    pool: &Bound<'_, PyAny>,
    file_name: &str,
) -> PyResult<Py<PyAny>> {
    let index = pool.getattr("file_name2file")?.cast_into::<PyDict>()?;
    match index.get_item(file_name)? {
        Some(found) => type_system::ok(py, found.unbind()),
        None => type_system::err(py, &format!("no file named {}", repr_str(py, file_name))),
    }
}

pub fn pool_find_definition(
    py: Python<'_>,
    pool: &Bound<'_, PyAny>,
    full_name: &str,
) -> PyResult<Py<PyAny>> {
    let index = pool
        .getattr("full_name2definition")?
        .cast_into::<PyDict>()?;
    match index.get_item(full_name)? {
        Some(found) => type_system::ok(py, found.unbind()),
        None => type_system::err(
            py,
            &format!("no definition named {}", repr_str(py, full_name)),
        ),
    }
}

pub fn pool_find_member(
    py: Python<'_>,
    pool: &Bound<'_, PyAny>,
    full_name: &str,
) -> PyResult<Py<PyAny>> {
    let index = pool.getattr("full_name2member")?.cast_into::<PyDict>()?;
    match index.get_item(full_name)? {
        Some(found) => type_system::ok(py, found.unbind()),
        None => type_system::err(py, &format!("no member named {}", repr_str(py, full_name))),
    }
}

pub fn file_find_import_by_local_name(
    py: Python<'_>,
    file: &Bound<'_, PyAny>,
    local_name: &str,
) -> PyResult<Py<PyAny>> {
    for import in pyast::field_list(file, "imports")? {
        let found: String = import.getattr("local_name")?.extract()?;
        if found == local_name {
            return type_system::ok(py, import.unbind());
        }
    }
    let file_name: String = file.getattr("file_name")?.extract()?;
    type_system::err(
        py,
        &format!(
            "file {} has no import named {}",
            repr_str(py, &file_name),
            repr_str(py, local_name)
        ),
    )
}

pub fn definition_members(py: Python<'_>, definition: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let mut members = Vec::new();
    for member in definition.getattr("members")?.try_iter()? {
        members.push(member?.unbind());
    }
    type_system::ok(py, pyast::pylist(py, members)?)
}

pub fn definition_find_member_by_tag(
    py: Python<'_>,
    definition: &Bound<'_, PyAny>,
    tag: &str,
) -> PyResult<Py<PyAny>> {
    for member in pyast::field_list(definition, "members")? {
        let member_tag = member.getattr("tag")?;
        if member_tag.is_none() {
            continue;
        }
        if member_tag.extract::<String>()? == tag {
            return type_system::ok(py, member.unbind());
        }
    }
    let full_name: String = definition.getattr("full_name")?.extract()?;
    type_system::err(
        py,
        &format!(
            "definition {} has no member tagged {}",
            repr_str(py, &full_name),
            repr_str(py, tag)
        ),
    )
}

pub fn definition_find_member_by_index(
    py: Python<'_>,
    definition: &Bound<'_, PyAny>,
    member_index: i64,
) -> PyResult<Py<PyAny>> {
    let members = pyast::field_list(definition, "members")?;
    if member_index >= 0 && (member_index as usize) < members.len() {
        return type_system::ok(py, members[member_index as usize].clone().unbind());
    }
    let full_name: String = definition.getattr("full_name")?.extract()?;
    type_system::err(
        py,
        &format!(
            "definition {} has no member at {}",
            repr_str(py, &full_name),
            member_index
        ),
    )
}

pub fn definition_file(py: Python<'_>, definition: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let pool = definition.getattr("pool")?;
    let file_name: String = definition.getattr("file_name")?.extract()?;
    pool_find_file(py, &pool, &file_name)
}

pub fn member_type_name(py: Python<'_>, member: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let contract = Contract::new(py)?;
    let kinds = Kinds::load(&contract)?;
    let member_type = member.getattr("member_type")?;
    let kind: String = member_type.getattr("kind")?.extract()?;
    if kind == kinds.type_ref {
        let type_name = member_type.getattr("payload")?.getattr("type_name")?;
        return type_system::ok(py, type_name.unbind());
    }
    let tag = member.getattr("tag")?;
    type_system::err(
        py,
        &format!(
            "member {} is not a name in the source",
            pyast::repr_of(py, &tag)
        ),
    )
}

pub fn member_resolved_definition(
    py: Python<'_>,
    member: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let name = member_type_name(py, member)?;
    if type_system::is_err(name.bind(py))? {
        return Ok(name);
    }
    let type_name: String = name.bind(py).getattr("ok_value")?.extract()?;
    let pool = member.getattr("pool")?;
    let containing_full_name: String = member.getattr("containing_full_name")?.extract()?;
    let containing = pool_find_definition(py, &pool, &containing_full_name)?;
    if type_system::is_err(containing.bind(py))? {
        return Ok(containing);
    }
    let containing_file_name: String = containing
        .bind(py)
        .getattr("ok_value")?
        .getattr("file_name")?
        .extract()?;
    let file = pool_find_file(py, &pool, &containing_file_name)?;
    if type_system::is_err(file.bind(py))? {
        return Ok(file);
    }
    let file = file.bind(py).getattr("ok_value")?;
    let parts: Vec<&str> = type_name.split('.').collect();
    let mut module_name: Option<String> = None;
    let mut target_name: Option<String> = None;
    for cut in (1..parts.len()).rev() {
        let prefix = parts[..cut].join(".");
        let import = file_find_import_by_local_name(py, &file, &prefix)?;
        if !type_system::is_err(import.bind(py))? {
            module_name = Some(
                import
                    .bind(py)
                    .getattr("ok_value")?
                    .getattr("module_name")?
                    .extract()?,
            );
            target_name = Some(parts[cut..].join("."));
            break;
        }
    }
    let (module_name, target_name) = match (module_name, target_name) {
        (Some(module_name), Some(target_name)) => (module_name, target_name),
        _ => {
            if parts.len() > 1 {
                return type_system::err(
                    py,
                    &format!(
                        "{} is neither an import nor a module of this file",
                        repr_str(py, parts[0])
                    ),
                );
            }
            let module_name: String = file.getattr("module_name")?.extract()?;
            (module_name, type_name.clone())
        }
    };
    let module = pool
        .getattr("module_environment")?
        .call1((module_name.as_str(),))?;
    if type_system::is_err(&module)? {
        return Ok(module.unbind());
    }
    let module_value = module.getattr("ok_value")?;
    let resolved = type_system::module_get_type(py, &module_value, &target_name, None)?;
    if type_system::is_err(resolved.bind(py))? {
        return Ok(resolved);
    }
    let resolved_value = resolved.bind(py).getattr("ok_value")?;
    let node = match resolved_value.getattr("ast_node") {
        Ok(node) => node,
        Err(_) => {
            return type_system::err(
                py,
                &format!("{} is not a definition", repr_str(py, &type_name)),
            )
        }
    };
    if !(is_class(&node, "TypeDefinition") || is_class(&node, "GenericDefinition")) {
        return type_system::err(
            py,
            &format!("{} is not a definition", repr_str(py, &type_name)),
        );
    }
    let node_name: String = node.getattr("name")?.extract()?;
    if node_name != target_name {
        return type_system::err(
            py,
            &format!(
                "{} does not name a definition of {}",
                repr_str(py, &type_name),
                repr_str(py, &module_name)
            ),
        );
    }
    pool_find_definition(py, &pool, &format!("{}.{}", module_name, node_name))
}

pub fn member_containing_definition(
    py: Python<'_>,
    member: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let pool = member.getattr("pool")?;
    let containing_full_name: String = member.getattr("containing_full_name")?.extract()?;
    pool_find_definition(py, &pool, &containing_full_name)
}

// ----------------------------------------------------------------------
// Descriptor building
// ----------------------------------------------------------------------

struct Builder<'py> {
    ctx: Ctx<'py>,
    contract: Contract<'py>,
    kinds: Kinds,
}

impl<'py> Builder<'py> {
    fn new(py: Python<'py>) -> PyResult<Self> {
        let contract = Contract::new(py)?;
        let kinds = Kinds::load(&contract)?;
        Ok(Self {
            ctx: Ctx::new(py)?,
            contract,
            kinds,
        })
    }

    fn py(&self) -> Python<'py> {
        self.ctx.py
    }

    fn build_file(
        &self,
        pool: &Bound<'py, PyAny>,
        tree: &Bound<'py, PyAny>,
        file_name: &str,
        module_name: &str,
        file_hash: &str,
    ) -> VibaResult<Py<PyAny>> {
        let py = self.py();
        let environment = pool.getattr("module_environment")?;
        let imports = import_locals_for(py, tree)?;
        let module = custom_module_type(py, tree, &environment, imports)?;
        let mut imported = Vec::new();
        let mut source_order = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        for statement in pyast::field_list(tree, "body")? {
            let class = pyast::class_name(&statement)?;
            if class == "Import" {
                let imported_module: String = statement.getattr("module")?.extract()?;
                let alias = optional_string(&statement.getattr("alias")?)?;
                let local = alias.unwrap_or_else(|| imported_module.clone());
                imported.push(self.contract.make(
                    "VibaImportDescriptor",
                    &[
                        ("pool", pool.clone().unbind()),
                        ("module_name", pyast::pystr(py, &imported_module)),
                        ("local_name", pyast::pystr(py, &local)),
                    ],
                )?);
            } else if class == "TypeDefinition" || class == "GenericDefinition" {
                let name: String = statement.getattr("name")?.extract()?;
                if !seen.insert(name.clone()) {
                    return Err(VibaError::Partial(format!(
                        "{} is defined twice: a module defines each name once",
                        repr_str(py, &name)
                    )));
                }
                source_order.push(statement);
            }
        }
        let mut definitions = Vec::new();
        for statement in &source_order {
            definitions.push(self.build_definition(
                pool,
                module.bind(py),
                statement,
                module_name,
                file_name,
                file_hash,
            )?);
        }
        Ok(self.contract.make(
            "VibaFileDescriptor",
            &[
                ("pool", pool.clone().unbind()),
                ("file_name", pyast::pystr(py, file_name)),
                ("file_hash", pyast::pystr(py, file_hash)),
                ("module_name", pyast::pystr(py, module_name)),
                ("imports", pyast::pylist(py, imported)?),
                ("definitions", pyast::pylist(py, definitions)?),
                ("tree", tree.clone().unbind()),
            ],
        )?)
    }

    fn build_definition(
        &self,
        pool: &Bound<'py, PyAny>,
        module: &Bound<'py, PyAny>,
        statement: &Bound<'py, PyAny>,
        module_name: &str,
        file_name: &str,
        file_hash: &str,
    ) -> VibaResult<Py<PyAny>> {
        let py = self.py();
        let def_name: String = statement.getattr("name")?.extract()?;
        let full_name = format!("{}.{}", module_name, def_name);
        let params = generic_params(py, statement)?;
        let body = statement.getattr("body")?;
        let body_type = self.build_type(pool, module, &body)?;
        let members = self.build_members(pool, module, &body, &full_name)?;
        Ok(self.contract.make(
            "VibaDefinitionDescriptor",
            &[
                ("pool", pool.clone().unbind()),
                ("def_name", pyast::pystr(py, &def_name)),
                ("module_name", pyast::pystr(py, module_name)),
                ("file_name", pyast::pystr(py, file_name)),
                ("file_hash", pyast::pystr(py, file_hash)),
                ("full_name", pyast::pystr(py, &full_name)),
                ("generic_params", params),
                ("body", body_type),
                ("members", members),
            ],
        )?)
    }

    /// A definition body the source has as a product/sum/exponent chain:
    /// `(elements, chain-head unit)`. The unit does not count as a member.
    fn body_elements(
        &self,
        body: &Bound<'py, PyAny>,
    ) -> PyResult<(Option<Vec<Py<PyAny>>>, Option<Vec<String>>)> {
        if is_class(body, "Product") || is_class(body, "ProductChain") {
            return Ok((
                Some(left_elements(body, "Product", "ProductChain")?),
                Some(self.kinds.product_unit.clone()),
            ));
        }
        if is_class(body, "Sum") || is_class(body, "SumChain") {
            return Ok((
                Some(left_elements(body, "Sum", "SumChain")?),
                Some(self.kinds.sum_unit.clone()),
            ));
        }
        if is_class(body, "Exponent") || is_class(body, "ExponentChain") {
            return Ok((
                Some(exponent_elements(body)?),
                Some(self.kinds.exponent_unit.clone()),
            ));
        }
        Ok((None, None))
    }

    fn build_members(
        &self,
        pool: &Bound<'py, PyAny>,
        module: &Bound<'py, PyAny>,
        body: &Bound<'py, PyAny>,
        full_name: &str,
    ) -> VibaResult<Py<PyAny>> {
        let py = self.py();
        let (elements, unit) = self.body_elements(body)?;
        let elements = match elements {
            Some(mut elements) => {
                if !elements.is_empty() {
                    if let Some(unit) = &unit {
                        if is_unit(elements[0].bind(py), unit) {
                            elements.remove(0);
                        }
                    }
                }
                elements
            }
            None => {
                if is_class(body, "Tagged") {
                    // The body is one whole tagged type: a product of one member.
                    vec![body.clone().unbind()]
                } else {
                    return Ok(pyast::pylist(py, Vec::new())?);
                }
            }
        };
        let mut members = Vec::new();
        for (index, element) in elements.iter().enumerate() {
            let element = element.bind(py);
            let (tag, node) = if is_class(element, "Tagged") {
                (
                    Some(element.getattr("tag")?.unbind()),
                    element.getattr("type")?,
                )
            } else {
                (None, element.clone())
            };
            let member_type = self.build_type(pool, module, &node)?;
            let tag = match tag {
                Some(tag) => tag,
                None => py.None(),
            };
            members.push(self.contract.make(
                "VibaMemberDescriptor",
                &[
                    ("pool", pool.clone().unbind()),
                    ("member_index", pyast::pyint(py, index as i64)),
                    ("tag", tag),
                    ("member_type", member_type),
                    ("containing_full_name", pyast::pystr(py, full_name)),
                ],
            )?);
        }
        Ok(pyast::pylist(py, members)?)
    }

    fn build_type(
        &self,
        pool: &Bound<'py, PyAny>,
        module: &Bound<'py, PyAny>,
        node: &Bound<'py, PyAny>,
    ) -> VibaResult<Py<PyAny>> {
        let py = self.py();
        if is_class(node, "Any") {
            return Ok(self.contract.type_descriptor(&self.kinds.any, py.None())?);
        }
        if is_class(node, "Partial") {
            if let Some((product, source_module)) = self.get_args_call(pool, node, module)? {
                let product = product.bind(py);
                let source_module = source_module.bind(py);
                return self.build_type(pool, source_module, product);
            }
            let resolver = DescriptorResolve {
                builder: self,
                pool,
            };
            let judge = Fits;
            let (reduced, source_module) =
                partial::reduce_partial(&self.ctx, node, module, &resolver, &judge, true)?;
            let reduced = reduced.bind(py);
            let source_module = source_module.bind(py);
            return self.build_type(pool, source_module, reduced);
        }
        let resolvable = ast_node_type(py, node, module)?;
        if is_class(node, "Product") || is_class(node, "ProductChain") {
            let mut elements = Vec::new();
            for element in left_elements(node, "Product", "ProductChain")? {
                elements.push(self.build_type(pool, module, element.bind(py))?);
            }
            let chain = self.contract.make(
                "VibaChainDescriptor",
                &[
                    ("pool", pool.clone().unbind()),
                    ("resolvable_type", resolvable),
                    ("elements", pyast::pylist(py, elements)?),
                ],
            )?;
            return Ok(self.contract.type_descriptor(&self.kinds.product, chain)?);
        }
        if is_class(node, "Sum") || is_class(node, "SumChain") {
            let mut elements = Vec::new();
            for element in left_elements(node, "Sum", "SumChain")? {
                elements.push(self.build_type(pool, module, element.bind(py))?);
            }
            let chain = self.contract.make(
                "VibaChainDescriptor",
                &[
                    ("pool", pool.clone().unbind()),
                    ("resolvable_type", resolvable),
                    ("elements", pyast::pylist(py, elements)?),
                ],
            )?;
            return Ok(self.contract.type_descriptor(&self.kinds.sum, chain)?);
        }
        if is_class(node, "Exponent") || is_class(node, "ExponentChain") {
            let mut elements = Vec::new();
            for element in exponent_elements(node)? {
                elements.push(self.build_type(pool, module, element.bind(py))?);
            }
            let chain = self.contract.make(
                "VibaChainDescriptor",
                &[
                    ("pool", pool.clone().unbind()),
                    ("resolvable_type", resolvable),
                    ("elements", pyast::pylist(py, elements)?),
                ],
            )?;
            return Ok(self.contract.type_descriptor(&self.kinds.exponent, chain)?);
        }
        if is_class(node, "TypeApp") {
            let mut args = Vec::new();
            for arg in pyast::field_list(node, "args")? {
                args.push(self.build_type(pool, module, &arg)?);
            }
            let constructor: String = node.getattr("constructor")?.extract()?;
            if let Some(folded) =
                self.tagged_descriptor(pool, &constructor, &args, resolvable.bind(py))?
            {
                return Ok(folded);
            }
            let application = self.contract.make(
                "VibaTypeAppDescriptor",
                &[
                    ("pool", pool.clone().unbind()),
                    ("resolvable_type", resolvable),
                    ("constructor_name", pyast::pystr(py, &constructor)),
                    ("args", pyast::pylist(py, args)?),
                ],
            )?;
            return Ok(self
                .contract
                .type_descriptor(&self.kinds.type_app, application)?);
        }
        if is_class(node, "Tuple") {
            let mut elements = Vec::new();
            for element in pyast::field_list(node, "elements")? {
                elements.push(self.build_type(pool, module, &element)?);
            }
            let tuple = self.contract.make(
                "VibaTupleDescriptor",
                &[
                    ("pool", pool.clone().unbind()),
                    ("resolvable_type", resolvable),
                    ("elements", pyast::pylist(py, elements)?),
                ],
            )?;
            return Ok(self.contract.type_descriptor(&self.kinds.tuple, tuple)?);
        }
        if is_class(node, "Tagged") {
            let tag = node.getattr("tag")?;
            let tagged_type = self.build_type(pool, module, &node.getattr("type")?)?;
            let tagged = self.contract.make(
                "VibaTaggedDescriptor",
                &[
                    ("pool", pool.clone().unbind()),
                    ("resolvable_type", resolvable),
                    ("tag", tag.unbind()),
                    ("tagged_type", tagged_type),
                ],
            )?;
            return Ok(self.contract.type_descriptor(&self.kinds.tagged, tagged)?);
        }
        if is_class(node, "TypeRef") {
            let type_name: String = node.getattr("name")?.extract()?;
            let reference = self.contract.make(
                "VibaTypeRefDescriptor",
                &[
                    ("pool", pool.clone().unbind()),
                    ("resolvable_type", resolvable),
                    ("type_name", pyast::pystr(py, &type_name)),
                ],
            )?;
            return Ok(self
                .contract
                .type_descriptor(&self.kinds.type_ref, reference)?);
        }
        if is_class(node, "MemberTaken") {
            // A member taken that the source has segment by segment counts as the
            // dotted name it was; with an application on the left it is left to
            // the decision layer.
            match source_path(py, node)? {
                Some(path) => {
                    let reference = self.contract.make(
                        "VibaTypeRefDescriptor",
                        &[
                            ("pool", pool.clone().unbind()),
                            ("resolvable_type", resolvable),
                            ("type_name", pyast::pystr(py, &path)),
                        ],
                    )?;
                    return Ok(self
                        .contract
                        .type_descriptor(&self.kinds.type_ref, reference)?);
                }
                None => {
                    let owner = self.build_type(pool, module, &node.getattr("owner")?)?;
                    let name = node.getattr("name")?;
                    let taken = self.contract.make(
                        "VibaMemberTakenDescriptor",
                        &[
                            ("pool", pool.clone().unbind()),
                            ("resolvable_type", resolvable),
                            ("owner", owner),
                            ("name", name.unbind()),
                        ],
                    )?;
                    return Ok(self
                        .contract
                        .type_descriptor(&self.kinds.member_taken, taken)?);
                }
            }
        }
        if is_class(node, "Member") {
            let tag: String = node.getattr("tag")?.extract()?;
            if tagged::INTERPRETER_MEMBER_TAGS.contains(&tag.as_str()) {
                // A member the interpreter answers, standing as a piece of its
                // own: that piece is the call the member's signature spells.
                let chain = tagged::interpreter_member_chain(&self.ctx, &tag)?
                    .ok_or_else(|| VibaError::Program("no descriptor for Member".to_string()))?;
                let builtin = builtin_module(py)?;
                return self.build_type(pool, &builtin, chain.bind(py));
            }
            return Err(VibaError::Program("no descriptor for Member".to_string()));
        }
        if is_class(node, "Constant") {
            let value = node.getattr("value")?.unbind();
            let literal = self.contract.make(
                "VibaLiteralDescriptor",
                &[
                    ("pool", pool.clone().unbind()),
                    ("resolvable_type", resolvable),
                    ("value", value),
                ],
            )?;
            return Ok(self
                .contract
                .type_descriptor(&self.kinds.literal, literal)?);
        }
        if is_class(node, "Nil") {
            return Ok(self.contract.type_descriptor(&self.kinds.nil, py.None())?);
        }
        if is_class(node, "Never") {
            return Ok(self
                .contract
                .type_descriptor(&self.kinds.never, py.None())?);
        }
        if is_class(node, "Ellipsis") {
            return Ok(self
                .contract
                .type_descriptor(&self.kinds.ellipsis, py.None())?);
        }
        if is_class(node, "HintBlock") {
            let code: String = node.getattr("code")?.extract()?;
            let hint = self.contract.make(
                "VibaHintBlockDescriptor",
                &[
                    ("pool", pool.clone().unbind()),
                    ("resolvable_type", resolvable),
                    ("code", pyast::pystr(py, &code)),
                ],
            )?;
            return Ok(self
                .contract
                .type_descriptor(&self.kinds.hint_block, hint)?);
        }
        Err(VibaError::Program(format!(
            "no descriptor for {}",
            pyast::class_name(node)?
        )))
    }

    fn tagged_descriptor(
        &self,
        pool: &Bound<'py, PyAny>,
        constructor_name: &str,
        args: &[Py<PyAny>],
        _resolvable: &Bound<'py, PyAny>,
    ) -> VibaResult<Option<Py<PyAny>>> {
        let py = self.py();
        if constructor_name != tagged::TAGGED_NAME || args.len() != 2 {
            return Ok(None);
        }
        let first = args[0].bind(py);
        let kind: String = first.getattr("kind")?.extract()?;
        if kind != self.kinds.literal {
            return Ok(None);
        }
        let value = first.getattr("payload")?.getattr("value")?;
        if !value.is_instance_of::<PyString>() {
            return Ok(None);
        }
        let symbol = match tagged::symbol_of(&value) {
            Some(symbol) => symbol,
            None => return Ok(None),
        };
        let inner = args[1].bind(py);
        let inner_resolvable = inner.getattr("resolvable_type")?;
        if inner_resolvable.is_none() {
            return Ok(None);
        }
        let tag = tagged::tag_of(&symbol);
        let tagged_node = self.ctx.make(
            "Tagged",
            &[
                ("tag", pyast::pystr(py, &tag)),
                ("type", inner_resolvable.getattr("ast_node")?.unbind()),
            ],
        )?;
        let resolvable = ast_node_type(
            py,
            tagged_node.bind(py),
            &inner_resolvable.getattr("container_module")?,
        )?;
        let tagged = self.contract.make(
            "VibaTaggedDescriptor",
            &[
                ("pool", pool.clone().unbind()),
                ("resolvable_type", resolvable),
                ("tag", pyast::pystr(py, &tag)),
                ("tagged_type", inner.clone().unbind()),
            ],
        )?;
        Ok(Some(
            self.contract.type_descriptor(&self.kinds.tagged, tagged)?,
        ))
    }

    fn descriptor_of_tagged(
        &self,
        tag: Py<PyAny>,
        inner: Py<PyAny>,
        node: &Bound<'py, PyAny>,
        source_module: &Bound<'py, PyAny>,
    ) -> VibaResult<Py<PyAny>> {
        let py = self.py();
        let tagged = self.contract.make(
            "VibaTaggedDescriptor",
            &[
                ("pool", empty_pool(py)?),
                ("resolvable_type", ast_node_type(py, node, source_module)?),
                ("tag", tag),
                ("tagged_type", inner),
            ],
        )?;
        Ok(self.contract.type_descriptor(&self.kinds.tagged, tagged)?)
    }

    fn descriptor_of_values(
        &self,
        node: &Bound<'py, PyAny>,
        source_module: &Bound<'py, PyAny>,
        members: &Bound<'py, PyAny>,
    ) -> VibaResult<Py<PyAny>> {
        let py = self.py();
        let resolvable = ast_node_type(py, node, source_module)?;
        let pieces = left_elements(node, "Product", "ProductChain")?;
        let mut elements = Vec::new();
        for (index, member) in members.try_iter()?.enumerate() {
            let member = member?;
            let descriptor = member.getattr("node")?.getattr("descriptor")?;
            match pieces.get(index) {
                Some(piece) if is_class(piece.bind(py), "Tagged") => {
                    let piece = piece.bind(py);
                    let own = match member.getattr("source_module") {
                        Ok(value) if !value.is_none() => value,
                        _ => source_module.clone(),
                    };
                    elements.push(self.descriptor_of_tagged(
                        piece.getattr("tag")?.unbind(),
                        descriptor.unbind(),
                        piece,
                        &own,
                    )?);
                }
                _ => elements.push(descriptor.unbind()),
            }
        }
        let chain = self.contract.make(
            "VibaChainDescriptor",
            &[
                ("pool", empty_pool(py)?),
                ("resolvable_type", resolvable),
                ("elements", pyast::pylist(py, elements)?),
            ],
        )?;
        Ok(self.contract.type_descriptor(&self.kinds.product, chain)?)
    }

    fn descriptor_of(&self, node: &Bound<'py, PyAny>) -> VibaResult<Py<PyAny>> {
        let py = self.py();
        let pool = empty_pool(py)?;
        let module = node.getattr("container_module")?;
        let ast_node = node.getattr("ast_node")?;
        self.build_type(pool.bind(py), &module, &ast_node)
    }

    /// `(body, source_module)` for the name a `<<` gives to, or None.
    fn partial_target(
        &self,
        pool: &Bound<'py, PyAny>,
        name: &str,
        module: &Bound<'_, PyAny>,
    ) -> VibaResult<Option<(Py<PyAny>, Py<PyAny>)>> {
        let py = self.py();
        let resolved = type_system::module_get_type(py, module, name, None)?;
        let resolved = resolved.bind(py);
        let value = match ok_value(resolved)? {
            Some(value) if is_class(&value, "AstNodeType") => Some(value),
            _ => None,
        };
        let value = match value {
            Some(value) => value,
            None => {
                if let Some(member) = self.product_member(pool, name, module)? {
                    return Ok(Some(member));
                }
                return partial::module_as_function(&self.ctx, module, name);
            }
        };
        let mut node = value.getattr("ast_node")?;
        let container = value.getattr("container_module")?;
        if is_class(&node, "GenericDefinition") {
            return Ok(None);
        }
        if is_class(&node, "TypeDefinition") {
            node = node.getattr("body")?;
        }
        let builtin = builtin_module(py)?;
        if !container.is(&builtin)
            && (is_class(&node, "Exponent") || is_class(&node, "ExponentChain"))
        {
            if let Some(problem) = partial::environment_result_problem(&self.ctx, &node, name)? {
                return Err(VibaError::Partial(problem));
            }
        }
        Ok(Some((node.unbind(), container.unbind())))
    }

    /// `(body, source_module)` for `args.a`: the member of the product a name
    /// stands for.
    fn product_member(
        &self,
        pool: &Bound<'py, PyAny>,
        name: &str,
        module: &Bound<'_, PyAny>,
    ) -> PyResult<Option<(Py<PyAny>, Py<PyAny>)>> {
        let py = self.py();
        let (head, dot, tag) = match name.rfind('.') {
            Some(index) => (&name[..index], &name[index..index + 1], &name[index + 1..]),
            None => ("", "", name),
        };
        if dot.is_empty() || head.is_empty() {
            return Ok(None);
        }
        let resolved = type_system::module_get_type(py, module, head, None)?;
        let value = match ok_value(resolved.bind(py))? {
            Some(value) if is_class(&value, "AstNodeType") => value,
            _ => return Ok(None),
        };
        let mut body = value.getattr("ast_node")?;
        let source_module = value.getattr("container_module")?;
        if is_class(&body, "TypeDefinition") {
            body = body.getattr("body")?;
        }
        let (body, source_module) = if is_class(&body, "Partial") {
            match self.get_args_call(pool, &body, &source_module)? {
                Some((product, module)) => (product.bind(py).clone(), module.bind(py).clone()),
                None => return Ok(None),
            }
        } else {
            (body, source_module)
        };
        let wanted = format!("${}", tag);
        for factor in left_elements(&body, "Product", "ProductChain")? {
            let factor = factor.bind(py);
            if !is_class(factor, "Tagged") {
                continue;
            }
            let factor_tag: String = factor.getattr("tag")?.extract()?;
            if factor_tag == wanted {
                return Ok(Some((
                    factor.getattr("type")?.unbind(),
                    source_module.unbind(),
                )));
            }
        }
        Ok(None)
    }

    /// `(product, source_module)` for `__get_args__ << <chain>`, or None.
    fn get_args_call(
        &self,
        _pool: &Bound<'py, PyAny>,
        node: &Bound<'py, PyAny>,
        module: &Bound<'py, PyAny>,
    ) -> PyResult<Option<(Py<PyAny>, Py<PyAny>)>> {
        let py = self.py();
        let (head, arguments) = partial_parts(node)?;
        if !is_class(&head, "TypeRef") {
            return Ok(None);
        }
        let head_name: String = head.getattr("name")?.extract()?;
        if head_name != partial::GET_ARGS_NAME || arguments.len() != 1 {
            return Ok(None);
        }
        let mut argument = arguments[0].clone();
        let mut source_module = module.clone();
        if is_class(&argument, "TypeRef") {
            let name: String = argument.getattr("name")?.extract()?;
            let resolved = type_system::module_get_type(py, module, &name, None)?;
            let value = match ok_value(resolved.bind(py))? {
                Some(value) if is_class(&value, "AstNodeType") => value,
                _ => return Ok(None),
            };
            argument = value.getattr("ast_node")?;
            source_module = value.getattr("container_module")?;
        }
        if is_class(&argument, "TypeDefinition") {
            argument = argument.getattr("body")?;
        }
        if !(is_class(&argument, "Exponent") || is_class(&argument, "ExponentChain")) {
            return Ok(None);
        }
        let product = partial::get_args_product(&self.ctx, &argument)?;
        Ok(Some((product, source_module.unbind())))
    }
}

/// The name a `<<` resolves through: `_partial_target`.
struct DescriptorResolve<'a, 'py> {
    builder: &'a Builder<'py>,
    pool: &'a Bound<'py, PyAny>,
}

impl ResolveFn for DescriptorResolve<'_, '_> {
    fn resolve(
        &self,
        _ctx: &Ctx<'_>,
        name: &str,
        module: &Bound<'_, PyAny>,
    ) -> VibaResult<Option<(Py<PyAny>, Py<PyAny>)>> {
        self.builder.partial_target(self.pool, name, module)
    }
}

/// Does the given type fit the slot the source has it in: `_fits`. The judgment
/// answers, and an error counts as "does not fit".
struct Fits;

impl JudgeFn for Fits {
    fn judge(
        &self,
        ctx: &Ctx<'_>,
        sub: &Bound<'_, PyAny>,
        sub_module: &Bound<'_, PyAny>,
        sup: &Bound<'_, PyAny>,
        sup_module: &Bound<'_, PyAny>,
    ) -> VibaResult<bool> {
        let sub = ast_node_type(ctx.py, sub, sub_module)?;
        let sup = ast_node_type(ctx.py, sup, sup_module)?;
        match crate::is_sub_type::judge(ctx.py, sub.bind(ctx.py), sup.bind(ctx.py), None) {
            Ok(result) => Ok(result),
            Err(_) => Ok(false),
        }
    }
}

// ----------------------------------------------------------------------
// Python-facing functions
// ----------------------------------------------------------------------

#[pyfunction]
fn empty_pool_py(py: Python<'_>) -> PyResult<Py<PyAny>> {
    empty_pool(py)
}

#[pyfunction]
fn parse_viba_file_py<'py>(
    py: Python<'py>,
    pool: &Bound<'py, PyAny>,
    source: &str,
    file_name: &str,
    module_name: &str,
) -> PyResult<Py<PyAny>> {
    parse_viba_file(py, pool, source, file_name, module_name)
}

#[pyfunction]
fn pool_add_file_py<'py>(
    py: Python<'py>,
    pool: &Bound<'py, PyAny>,
    file: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    pool_add_file(py, pool, file)
}

#[pyfunction]
fn pool_find_file_py<'py>(
    py: Python<'py>,
    pool: &Bound<'py, PyAny>,
    file_name: &str,
) -> PyResult<Py<PyAny>> {
    pool_find_file(py, pool, file_name)
}

#[pyfunction]
fn pool_find_definition_py<'py>(
    py: Python<'py>,
    pool: &Bound<'py, PyAny>,
    full_name: &str,
) -> PyResult<Py<PyAny>> {
    pool_find_definition(py, pool, full_name)
}

#[pyfunction]
fn pool_find_member_py<'py>(
    py: Python<'py>,
    pool: &Bound<'py, PyAny>,
    full_name: &str,
) -> PyResult<Py<PyAny>> {
    pool_find_member(py, pool, full_name)
}

#[pyfunction]
fn file_find_import_by_local_name_py<'py>(
    py: Python<'py>,
    file: &Bound<'py, PyAny>,
    local_name: &str,
) -> PyResult<Py<PyAny>> {
    file_find_import_by_local_name(py, file, local_name)
}

#[pyfunction]
fn definition_members_py<'py>(
    py: Python<'py>,
    definition: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    definition_members(py, definition)
}

#[pyfunction]
fn definition_find_member_by_tag_py<'py>(
    py: Python<'py>,
    definition: &Bound<'py, PyAny>,
    tag: &str,
) -> PyResult<Py<PyAny>> {
    definition_find_member_by_tag(py, definition, tag)
}

#[pyfunction]
fn definition_find_member_by_index_py<'py>(
    py: Python<'py>,
    definition: &Bound<'py, PyAny>,
    member_index: i64,
) -> PyResult<Py<PyAny>> {
    definition_find_member_by_index(py, definition, member_index)
}

#[pyfunction]
fn definition_file_py<'py>(py: Python<'py>, definition: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
    definition_file(py, definition)
}

#[pyfunction]
fn member_type_name_py<'py>(py: Python<'py>, member: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
    member_type_name(py, member)
}

#[pyfunction]
fn member_resolved_definition_py<'py>(
    py: Python<'py>,
    member: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    member_resolved_definition(py, member)
}

#[pyfunction]
fn member_containing_definition_py<'py>(
    py: Python<'py>,
    member: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    member_containing_definition(py, member)
}

#[pyfunction]
fn descriptor_of_py<'py>(py: Python<'py>, node: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
    let builder = Builder::new(py)?;
    builder
        .descriptor_of(node)
        .map_err(|error| design_error(py, error))
}

#[pyfunction]
fn descriptor_of_values_py<'py>(
    py: Python<'py>,
    node: &Bound<'py, PyAny>,
    source_module: &Bound<'py, PyAny>,
    members: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    let builder = Builder::new(py)?;
    builder
        .descriptor_of_values(node, source_module, members)
        .map_err(|error| design_error(py, error))
}

#[pyfunction]
fn descriptor_of_tagged_py<'py>(
    py: Python<'py>,
    tag: Py<PyAny>,
    inner: Py<PyAny>,
    node: &Bound<'py, PyAny>,
    source_module: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    let builder = Builder::new(py)?;
    builder
        .descriptor_of_tagged(tag, inner, node, source_module)
        .map_err(|error| design_error(py, error))
}

#[pyfunction]
fn tagged_descriptor_py<'py>(
    py: Python<'py>,
    pool: &Bound<'py, PyAny>,
    constructor_name: &str,
    args: &Bound<'py, PyAny>,
    resolvable: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    let builder = Builder::new(py)?;
    let mut items = Vec::new();
    for item in args.try_iter()? {
        items.push(item?.unbind());
    }
    match builder.tagged_descriptor(pool, constructor_name, &items, resolvable) {
        Ok(Some(folded)) => Ok(folded),
        Ok(None) => Ok(py.None()),
        Err(error) => Err(design_error(py, error)),
    }
}

#[pyfunction]
fn build_type_py<'py>(
    py: Python<'py>,
    pool: &Bound<'py, PyAny>,
    module: &Bound<'py, PyAny>,
    node: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    let builder = Builder::new(py)?;
    builder
        .build_type(pool, module, node)
        .map_err(|error| design_error(py, error))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PoolEnvironment>()?;
    m.add_function(wrap_pyfunction!(empty_pool_py, m)?)?;
    m.add_function(wrap_pyfunction!(parse_viba_file_py, m)?)?;
    m.add_function(wrap_pyfunction!(pool_add_file_py, m)?)?;
    m.add_function(wrap_pyfunction!(pool_find_file_py, m)?)?;
    m.add_function(wrap_pyfunction!(pool_find_definition_py, m)?)?;
    m.add_function(wrap_pyfunction!(pool_find_member_py, m)?)?;
    m.add_function(wrap_pyfunction!(file_find_import_by_local_name_py, m)?)?;
    m.add_function(wrap_pyfunction!(definition_members_py, m)?)?;
    m.add_function(wrap_pyfunction!(definition_find_member_by_tag_py, m)?)?;
    m.add_function(wrap_pyfunction!(definition_find_member_by_index_py, m)?)?;
    m.add_function(wrap_pyfunction!(definition_file_py, m)?)?;
    m.add_function(wrap_pyfunction!(member_type_name_py, m)?)?;
    m.add_function(wrap_pyfunction!(member_resolved_definition_py, m)?)?;
    m.add_function(wrap_pyfunction!(member_containing_definition_py, m)?)?;
    m.add_function(wrap_pyfunction!(descriptor_of_py, m)?)?;
    m.add_function(wrap_pyfunction!(descriptor_of_values_py, m)?)?;
    m.add_function(wrap_pyfunction!(descriptor_of_tagged_py, m)?)?;
    m.add_function(wrap_pyfunction!(tagged_descriptor_py, m)?)?;
    m.add_function(wrap_pyfunction!(build_type_py, m)?)?;
    Ok(())
}
