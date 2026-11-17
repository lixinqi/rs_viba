//! Pattern: a generic is a directory, and one of its files answers (was
//! viba/pattern.py).
//!
//! A generic has no body of its own. What it has is a *method*: a type argument
//! as the source has it, and the answer that argument asks for. Each answer is
//! one file, named by its decision order (`100.viba`), and the count of parts it
//! takes may be spelled in front of that number (`2_200.viba`). Inside, one
//! `pattern` line per parameter, in source order. Applying the generic takes the
//! files in order and keeps the first whose patterns fit the arguments as the
//! source has them: a name the file never defines is a parameter to extract,
//! while a known type restricts what that argument may be.
//!
//! The Python classes (`PatternFile`, `GenericModuleType`, `Choice`) live in
//! `rs_viba.pattern`; this module builds them and works with them, so the shim's
//! API is the one every other layer sees.

use std::collections::{BTreeSet, HashSet};
use std::sync::OnceLock;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule, PyString, PyTuple};

use crate::ctx::Ctx;
use crate::errors::{VibaError, VibaResult};
use crate::partial::{JudgeFn, ResolveFn, DEF_NAME};
use crate::pyast;
use crate::tagged;
use crate::type_system;
use crate::unparser;

/// The marker file that says a directory is a generic.
pub const GENERIC_FILE: &str = "__generic__.viba";

/// The most totals an application's parts are counted in; past that the walk is
/// not worth it, and no file is passed over.
const MOST_TOTALS: usize = 8;

// ----------------------------------------------------------------------
// Small helpers over the Python objects
// ----------------------------------------------------------------------

fn type_module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    PyModule::import(py, "rs_viba.type")
}

fn pattern_class<'py>(py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    Ok(PyModule::import(py, "rs_viba.pattern")?.getattr(name)?)
}

fn class_name(node: &Bound<'_, PyAny>) -> PyResult<String> {
    pyast::class_name(node)
}

fn is_class(node: &Bound<'_, PyAny>, name: &str) -> PyResult<bool> {
    Ok(class_name(node)? == name)
}

fn in_classes(ctx: &Ctx<'_>, node: &Bound<'_, PyAny>, names: &[&str]) -> PyResult<bool> {
    for name in names {
        if ctx.is(node, name)? {
            return Ok(true);
        }
    }
    Ok(false)
}

fn is_sum(ctx: &Ctx<'_>, node: &Bound<'_, PyAny>) -> PyResult<bool> {
    in_classes(ctx, node, &["Sum", "SumChain"])
}

fn is_product(ctx: &Ctx<'_>, node: &Bound<'_, PyAny>) -> PyResult<bool> {
    in_classes(ctx, node, &["Product", "ProductChain"])
}

fn is_exponent(ctx: &Ctx<'_>, node: &Bound<'_, PyAny>) -> PyResult<bool> {
    in_classes(ctx, node, &["Exponent", "ExponentChain"])
}

fn repr_text(py: Python<'_>, text: &str) -> String {
    let value = PyString::new(py, text);
    pyast::repr_of(py, value.as_any())
}

fn exception_text(py: Python<'_>, error: &PyErr) -> String {
    match error.value(py).as_any().str() {
        Ok(text) => text.to_string(),
        Err(_) => error.to_string(),
    }
}

fn as_list(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let list = PyList::empty(py);
    for item in value.try_iter()? {
        list.append(item?)?;
    }
    Ok(list.into_any().unbind())
}

fn optional_int(py: Python<'_>, value: Option<i64>) -> Py<PyAny> {
    match value {
        Some(value) => pyast::pyint(py, value),
        None => py.None(),
    }
}

fn err_message(result: &Bound<'_, PyAny>) -> PyResult<String> {
    result.getattr("msg")?.extract()
}

fn ast_node_type<'py>(
    py: Python<'py>,
    node: &Bound<'py, PyAny>,
    module: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    Ok(type_module(py)?
        .getattr("AstNodeType")?
        .call1((node, module))?
        .unbind())
}

fn type_of<'py>(
    ctx: &Ctx<'py>,
    node: &Bound<'py, PyAny>,
    module: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    ast_node_type(ctx.py, node, module)
}

/// `sub <: sup` where both sides are `AstNodeType`s, as the judgment layer says.
/// A judgment that cannot be made at all is no fit, not the end of the decision.
fn judge_ast(py: Python<'_>, sub: &Bound<'_, PyAny>, sup: &Bound<'_, PyAny>) -> bool {
    matches!(crate::is_sub_type::judge(py, sub, sup, None), Ok(true))
}

fn same_type(ctx: &Ctx<'_>, left: &Bound<'_, PyAny>, right: &Bound<'_, PyAny>) -> bool {
    judge_ast(ctx.py, left, right) && judge_ast(ctx.py, right, left)
}

/// The definition a body gives under this name: the last one.
fn definition<'py>(body: &Bound<'py, PyAny>, name: &str) -> PyResult<Option<Py<PyAny>>> {
    let mut found: Option<Py<PyAny>> = None;
    for statement in body.try_iter()? {
        let statement = statement?;
        if let Ok(statement_name) = statement.getattr("name") {
            if let Ok(statement_name) = statement_name.extract::<String>() {
                if statement_name == name {
                    found = Some(statement.unbind());
                }
            }
        }
    }
    Ok(found)
}

/// `getattr(getattr(module, "module", None), "body", None)`, as Option.
fn module_body<'py>(module: &Bound<'py, PyAny>) -> PyResult<Option<Bound<'py, PyAny>>> {
    let module_attr = match module.getattr("module") {
        Ok(value) => value,
        Err(_) => return Ok(None),
    };
    match module_attr.getattr("body") {
        Ok(body) if !body.is_none() => Ok(Some(body)),
        _ => Ok(None),
    }
}

// The pattern classes of the shim -------------------------------------------------

fn make_pattern_file(
    ctx: &Ctx<'_>,
    order: i64,
    path: &str,
    name: &str,
    declared: Option<i64>,
    load: Py<PyAny>,
) -> PyResult<Py<PyAny>> {
    let class = pattern_class(ctx.py, "PatternFile")?;
    Ok(class
        .call1((
            pyast::pyint(ctx.py, order),
            pyast::pystr(ctx.py, path),
            pyast::pystr(ctx.py, name),
            optional_int(ctx.py, declared),
            load,
        ))?
        .unbind())
}

fn make_generic<'py>(
    ctx: &Ctx<'py>,
    name: &str,
    directory: &str,
    entries: Vec<Py<PyAny>>,
    marker: Option<&Bound<'py, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let class = pattern_class(ctx.py, "GenericModuleType")?;
    let entries = ctx.list(entries)?;
    let marker: Py<PyAny> = match marker {
        Some(marker) => marker.clone().unbind(),
        None => ctx.py.None(),
    };
    Ok(class
        .call1((
            pyast::pystr(ctx.py, name),
            pyast::pystr(ctx.py, directory),
            entries,
            marker,
        ))?
        .unbind())
}

fn make_choice<'py>(
    ctx: &Ctx<'py>,
    entry: &Bound<'py, PyAny>,
    body: Py<PyAny>,
    module: &Bound<'py, PyAny>,
    bindings: Py<PyAny>,
) -> PyResult<Py<PyAny>> {
    let class = pattern_class(ctx.py, "Choice")?;
    Ok(class.call1((entry, body, module, bindings))?.unbind())
}

// ----------------------------------------------------------------------
// Loading a generic's directory
// ----------------------------------------------------------------------

/// The patterns a file's body gives, one per `pattern` line.
fn patterns_of(py: Python<'_>, body: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let list = PyList::empty(py);
    for statement in body.try_iter()? {
        let statement = statement?;
        if class_name(&statement)? == "Pattern" {
            list.append(statement.getattr("pattern")?)?;
        }
    }
    Ok(list.into_any().unbind())
}

/// Why this file may not spell `pattern`, or None when it may.
fn file_pattern_problem(
    _py: Python<'_>,
    tree: &Bound<'_, PyAny>,
    file_name: &str,
) -> PyResult<Option<String>> {
    let body = match tree.getattr("body") {
        Ok(body) => body,
        Err(_) => return Ok(None),
    };
    let mut spells_pattern = false;
    for statement in body.try_iter()? {
        if class_name(&statement?)? == "Pattern" {
            spells_pattern = true;
            break;
        }
    }
    if !spells_pattern {
        return Ok(None);
    }
    if order_of(file_name).is_some() {
        return Ok(None);
    }
    Ok(Some(format!(
        "{file_name} spells pattern, so it is a pattern file: one of the numbered files of its generic's directory"
    )))
}

/// (the count a file name declares, the order it gives), or None when it gives none.
fn name_parts(file_name: &str) -> Option<(Option<i64>, i64)> {
    let stem = match file_name.strip_suffix(".viba") {
        Some(stem) => stem,
        None => file_name,
    };
    if !stem.is_empty() && stem.bytes().all(|byte| byte.is_ascii_digit()) {
        return Some((None, stem.parse().ok()?));
    }
    let (count, order) = stem.split_once('_')?;
    if !count.is_empty()
        && count.bytes().all(|byte| byte.is_ascii_digit())
        && !order.is_empty()
        && order.bytes().all(|byte| byte.is_ascii_digit())
    {
        return Some((Some(count.parse().ok()?), order.parse().ok()?));
    }
    None
}

/// The decision order a file name spells, or None when it spells none.
fn order_of(file_name: &str) -> Option<i64> {
    name_parts(file_name).map(|(_, order)| order)
}

/// The count a file name declares, or None when it declares none.
fn declared_of(file_name: &str) -> Option<i64> {
    name_parts(file_name).and_then(|(declared, _)| declared)
}

/// `os.path.join` for the two parts this module joins.
fn join_path(directory: &str, file_name: &str) -> String {
    if directory.is_empty() {
        return file_name.to_string();
    }
    if directory.ends_with('/') {
        return format!("{}{}", directory, file_name);
    }
    format!("{}/{}", directory, file_name)
}

/// One pattern file, as its name alone says it (`generic_of_files`).
struct FilePart {
    order: i64,
    path: String,
    module_name: String,
    declared: Option<i64>,
}

/// One pattern file a layer already compiled (`generic_of_entries`).
struct EntryPart {
    order: i64,
    path: String,
    module_name: String,
    declared: Option<i64>,
    module: Py<PyAny>,
}

/// How one pattern file is loaded, for the file the decision reaches.
#[pyclass(name = "_PatternFileLoader")]
struct FileLoader {
    path: String,
    module_name: String,
    source_of: Py<PyAny>,
    module_of: Py<PyAny>,
}

#[pymethods]
impl FileLoader {
    fn __call__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let source = self.source_of.bind(py).call1((self.path.as_str(),))?;
        if source.is_none() {
            return type_system::err(py, &format!("cannot load {}", self.path));
        }
        let got = self.module_of.bind(py).call1((
            self.path.as_str(),
            self.module_name.as_str(),
            source,
        ))?;
        if type_system::is_err(&got)? {
            return Ok(got.unbind());
        }
        let module = got.getattr("ok_value")?;
        let body = match module_body(&module)? {
            Some(body) => body,
            None => {
                return type_system::err(
                    py,
                    &format!("{} is not a module of definitions", self.path),
                )
            }
        };
        let patterns = patterns_of(py, &body)?;
        let pair = PyTuple::new(py, [module.unbind(), patterns])?;
        type_system::ok(py, pair.into_any().unbind())
    }
}

/// Load a generic's whole directory: the `GenericModuleType`, or why not.
fn load_generic(
    ctx: &Ctx<'_>,
    directory: &str,
    name: &str,
    source_of: &Bound<'_, PyAny>,
    listing: &Bound<'_, PyAny>,
    module_of: &Bound<'_, PyAny>,
) -> VibaResult<Py<PyAny>> {
    let marker = join_path(directory, GENERIC_FILE);
    let marker_source = source_of.call1((marker.as_str(),))?;
    if marker_source.is_none() {
        return Err(VibaError::Program(format!(
            "{} is no generic: it has no {}",
            directory, GENERIC_FILE
        )));
    }
    let marker_text: String = marker_source.extract()?;
    let marker_tree = match crate::parser::parse(ctx.py, &marker_text) {
        Ok(tree) => tree.bind(ctx.py).clone(),
        Err(error) => {
            return Err(VibaError::Program(format!(
                "cannot parse {}: {}",
                marker,
                exception_text(ctx.py, &error)
            )))
        }
    };
    if let Some(problem) = file_pattern_problem(ctx.py, &marker_tree, GENERIC_FILE)? {
        return Err(VibaError::Program(format!("{}: {}", marker, problem)));
    }

    let listed = listing.call1((directory,))?;
    if type_system::is_err(&listed)? {
        return Err(VibaError::Program(err_message(&listed)?));
    }
    let names = listed.getattr("ok_value")?;
    let mut sorted: Vec<String> = Vec::new();
    for item in names.try_iter()? {
        sorted.push(item?.extract::<String>()?);
    }
    sorted.sort();

    let mut parts: Vec<FilePart> = Vec::new();
    for file_name in sorted {
        if file_name == GENERIC_FILE || !file_name.ends_with(".viba") {
            continue;
        }
        let path = join_path(directory, &file_name);
        let order = match order_of(&file_name) {
            Some(order) => order,
            None => {
                return Err(VibaError::Program(format!(
                    "{}: a file of the generic {} is named by its order, a number — {} is none",
                    path,
                    repr_text(ctx.py, name),
                    repr_text(ctx.py, &file_name)
                )))
            }
        };
        parts.push(FilePart {
            order,
            path,
            module_name: format!("{}.{}", name, order),
            declared: declared_of(&file_name),
        });
    }

    generic_of_files(
        ctx,
        name,
        directory,
        Some(&marker_tree),
        &parts,
        source_of,
        module_of,
    )
}

/// A generic built from names alone: one `(order, path, module name, declared)`
/// per pattern file, each given its text and its module when the decision
/// reaches it.
fn generic_of_files(
    ctx: &Ctx<'_>,
    name: &str,
    directory: &str,
    marker: Option<&Bound<'_, PyAny>>,
    parts: &[FilePart],
    source_of: &Bound<'_, PyAny>,
    module_of: &Bound<'_, PyAny>,
) -> VibaResult<Py<PyAny>> {
    let mut entries: Vec<(i64, Py<PyAny>)> = Vec::new();
    for part in parts {
        let loader = Py::new(
            ctx.py,
            FileLoader {
                path: part.path.clone(),
                module_name: part.module_name.clone(),
                source_of: source_of.clone().unbind(),
                module_of: module_of.clone().unbind(),
            },
        )?;
        let entry = make_pattern_file(
            ctx,
            part.order,
            &part.path,
            &part.module_name,
            part.declared,
            loader.into_any(),
        )?;
        entries.push((part.order, entry));
    }
    entries.sort_by_key(|(order, _)| *order);
    let entries: Vec<Py<PyAny>> = entries.into_iter().map(|(_, entry)| entry).collect();
    Ok(make_generic(ctx, name, directory, entries, marker)?)
}

/// A generic built from parts the caller already has: each file was compiled by
/// the layer that owns its files.
fn generic_of_entries(
    ctx: &Ctx<'_>,
    name: &str,
    directory: &str,
    marker: Option<&Bound<'_, PyAny>>,
    parts: &[EntryPart],
) -> VibaResult<Py<PyAny>> {
    let mut entries: Vec<(i64, Py<PyAny>)> = Vec::new();
    for part in parts {
        let module = part.module.bind(ctx.py);
        let body = match module_body(module)? {
            Some(body) => body,
            None => {
                return Err(VibaError::Program(format!(
                    "{} is not a module of definitions",
                    part.path
                )))
            }
        };
        let patterns = patterns_of(ctx.py, &body)?;
        let class = pattern_class(ctx.py, "PatternFile")?;
        let entry = class.clone().call_method1(
            "ready",
            (
                pyast::pyint(ctx.py, part.order),
                pyast::pystr(ctx.py, &part.path),
                pyast::pystr(ctx.py, &part.module_name),
                optional_int(ctx.py, part.declared),
                part.module.clone_ref(ctx.py),
                patterns,
            ),
        )?;
        entries.push((part.order, entry.unbind()));
    }
    entries.sort_by_key(|(order, _)| *order);
    let entries: Vec<Py<PyAny>> = entries.into_iter().map(|(_, entry)| entry).collect();
    Ok(make_generic(ctx, name, directory, entries, marker)?)
}

// ----------------------------------------------------------------------
// Reaching a generic from a name in the source
// ----------------------------------------------------------------------

/// The generic a name in the source names: an import, or the builtin library.
fn generic_named(
    ctx: &Ctx<'_>,
    module: &Bound<'_, PyAny>,
    name: &str,
) -> VibaResult<Option<Py<PyAny>>> {
    if !is_class(module, "CustomModuleType")? {
        return Ok(None);
    }
    let imports = module.getattr("imports")?;
    let mut prefixes: Vec<String> = Vec::new();
    for key in imports.try_iter()? {
        prefixes.push(key?.extract::<String>()?);
    }
    prefixes.sort_by(|left, right| right.len().cmp(&left.len()));
    for prefix in prefixes {
        if name == prefix {
            let module_name = imports.get_item(prefix.as_str())?;
            return generic_of(ctx, module, &module_name);
        }
        if name.starts_with(prefix.as_str()) && name[prefix.len()..].starts_with('.') {
            let rest = &name[prefix.len() + 1..];
            let module_name = imports.get_item(prefix.as_str())?;
            let handed = module.call_method1("module_environment", (module_name,))?;
            if type_system::is_err(&handed)? {
                return Err(VibaError::Program(err_message(&handed)?));
            }
            let handed = handed.getattr("ok_value")?;
            if is_class(&handed, "GenericModuleType")? {
                return Ok(None); // a member of a generic is not one
            }
            return generic_named(ctx, &handed, rest);
        }
    }
    let local = module.call_method1("lookup_local", (name,))?;
    if !type_system::is_err(&local)? {
        return Ok(None); // this module's own definition of that name
    }
    if let Some(builtin_name) = builtin_generic_name(ctx.py, name)? {
        let builtin_name = pyast::pystr(ctx.py, &builtin_name);
        return generic_of(ctx, module, builtin_name.bind(ctx.py));
    }
    Ok(None)
}

static BUILTIN_GENERIC_NAMES: OnceLock<HashSet<String>> = OnceLock::new();

/// The names under the builtin library's `builtin/` directory that are generics.
fn builtin_generic_names(py: Python<'_>) -> PyResult<&'static HashSet<String>> {
    if let Some(names) = BUILTIN_GENERIC_NAMES.get() {
        return Ok(names);
    }
    let types = type_module(py)?;
    let directory = types.getattr("BUILTIN_CONCEPT_DIR")?;
    let pattern = format!("*/{}", GENERIC_FILE);
    let found = directory.call_method1("glob", (pattern,))?;
    let mut names: HashSet<String> = HashSet::new();
    for one in found.try_iter()? {
        let one = one?;
        names.insert(
            one.getattr("parent")?
                .getattr("name")?
                .extract::<String>()?,
        );
    }
    Ok(BUILTIN_GENERIC_NAMES.get_or_init(|| names))
}

/// The builtin generic a name in the source stands for, or None.
fn builtin_generic_name(py: Python<'_>, name: &str) -> PyResult<Option<String>> {
    let types = type_module(py)?;
    let rest = types.getattr("builtin_directory_name")?.call1((name,))?;
    if rest.is_none() {
        return Ok(None);
    }
    let rest: String = rest.extract()?;
    let names = builtin_generic_names(py)?;
    Ok(if names.contains(&rest) {
        Some(rest)
    } else {
        None
    })
}

/// The generic this module name stands for, or None when it is something else.
fn generic_of(
    _ctx: &Ctx<'_>,
    module: &Bound<'_, PyAny>,
    module_name: &Bound<'_, PyAny>,
) -> VibaResult<Option<Py<PyAny>>> {
    let handed = module.call_method1("module_environment", (module_name,))?;
    if type_system::is_err(&handed)? {
        return Err(VibaError::Program(err_message(&handed)?));
    }
    let handed = handed.getattr("ok_value")?;
    if is_class(&handed, "GenericModuleType")? {
        Ok(Some(handed.unbind()))
    } else {
        Ok(None)
    }
}

// ----------------------------------------------------------------------
// `tagged[...]` taken where the decision's bindings are known
// ----------------------------------------------------------------------

/// The string a body in the source stands for here, or None.
fn string_of_definition(body: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
    let mut body = body.clone();
    while is_class(&body, "TypeDefinition")? {
        body = body.getattr("body")?;
    }
    if is_class(&body, "Constant")? {
        let value = body.getattr("value")?;
        if value.is_instance_of::<PyString>() {
            return Ok(Some(value.extract()?));
        }
        return Ok(None);
    }
    if body.is_instance_of::<PyString>() {
        return Ok(Some(body.extract()?));
    }
    Ok(None)
}

/// The string a name in the source stands for here: a definition of the module
/// holding a string, an alias followed to the end of its chain.
fn named_symbol(
    ctx: &Ctx<'_>,
    name: &str,
    module: &Bound<'_, PyAny>,
) -> VibaResult<Option<String>> {
    let reference = ctx.make("TypeRef", &[("name", ctx.pystr(name))])?;
    let (node, _source_module) = unfold(ctx, reference.bind(ctx.py), module)?;
    let node = node.bind(ctx.py);
    if ctx.is(node, "Constant")? {
        let value = node.getattr("value")?;
        if value.is_instance_of::<PyString>() {
            return Ok(Some(value.extract()?));
        }
    }
    Ok(None)
}

/// The tag a `tagged[...]` in the source stands for, or None when it is none.
///
/// `tagged[S, T]` is the tagged type `$S T` and `tagged[S]` is the member `$S`,
/// so a symbol can be spelled where only a tag would otherwise fit. A string
/// literal in the source is already folded where the source was parsed; what is
/// taken here is the symbol that is a *name* — a `pattern` line's parameter, or
/// one a decision bound.
///
/// `resolve` says what a name in the source stands for, for a layer that keeps
/// its own bindings (the judgment takes them through `AstNodeType.env_get`, the
/// other layers through their own). It answers `(body, source_module)`, and the
/// string is read off `body`: a `TypeDefinition` is unwrapped to the body it
/// holds, a `Constant` whose value is a string gives that string, and a plain
/// Python string is taken as it stands. `None` for a name this layer cannot take
/// there. Without `resolve` the name is unfolded in the module
/// (`module_get_type`), a definition's body included.
///
/// Ok(None) when this is no tagged application at all, or when its symbol is a
/// name that cannot be taken: the caller then takes the application its own way,
/// and reports it where it lands. Err when the application is no tag: the wrong
/// number of arguments, or a string that spells no symbol.
pub fn tagged_type_of<'py>(
    ctx: &Ctx<'py>,
    node: &Bound<'py, PyAny>,
    module: &Bound<'py, PyAny>,
    resolve: Option<&dyn ResolveFn>,
) -> VibaResult<Option<Py<PyAny>>> {
    if !(ctx.is(node, "TypeApp")?
        && node.getattr("constructor")?.extract::<String>()? == tagged::TAGGED_NAME)
    {
        return Ok(None);
    }
    let args = pyast::field_list(node, "args")?;
    if let Some(problem) = tagged::tagged_problem(ctx, &args)? {
        return Err(VibaError::Program(problem));
    }
    let mut symbol = tagged::literal_symbol(ctx, &args[0])?;
    if symbol.is_none() {
        let name = &args[0];
        if !ctx.is(name, "TypeRef")? {
            return Err(VibaError::Program(format!(
                "{} asks for a symbol spelled as a string, not {}",
                tagged::TAGGED_NAME,
                unparser::unparse_type(ctx.py, name, 2)?
            )));
        }
        let name_text: String = name.getattr("name")?.extract()?;
        let text = match resolve {
            Some(resolve) => match resolve.resolve(ctx, &name_text, module)? {
                Some((body, _source_module)) => string_of_definition(body.bind(ctx.py))?,
                None => None,
            },
            None => named_symbol(ctx, &name_text, module)?,
        };
        let text = match text {
            Some(text) => text,
            None => return Ok(None),
        };
        match tagged::symbol_of_text(&text) {
            Some(found) => symbol = Some(found),
            None => {
                return Err(VibaError::Program(format!(
                    "{} is no symbol for a tag",
                    repr_text(ctx.py, &text)
                )))
            }
        }
    }
    let symbol = symbol.unwrap();
    let owned: Vec<Py<PyAny>> = args.iter().map(|arg| arg.clone().unbind()).collect();
    Ok(Some(tagged::tagged_node(ctx, &symbol, &owned)?))
}

/// The Python-facing `tagged_type_of`'s `resolve` callback: a function of
/// `(name, module)` answering the string the name stands for, wrapped as the
/// one-step resolver `partial::ResolveFn` is.
struct TextResolve<'py> {
    callback: &'py Bound<'py, PyAny>,
}

impl ResolveFn for TextResolve<'_> {
    fn resolve(
        &self,
        _ctx: &Ctx<'_>,
        name: &str,
        module: &Bound<'_, PyAny>,
    ) -> VibaResult<Option<(Py<PyAny>, Py<PyAny>)>> {
        let text = self.callback.call1((name, module))?;
        if text.is_none() {
            return Ok(None);
        }
        Ok(Some((text.unbind(), module.clone().unbind())))
    }
}

// ----------------------------------------------------------------------
// Reaching a generic application
// ----------------------------------------------------------------------

/// The chosen body for a generic application, or None when it is none.
pub fn reduce_application<'py>(
    ctx: &Ctx<'py>,
    node: &Bound<'py, PyAny>,
    module: &Bound<'py, PyAny>,
    argument_modules: Option<&[Py<PyAny>]>,
    argument_parts: Option<&[Option<Vec<Py<PyAny>>>]>,
) -> VibaResult<Option<Py<PyAny>>> {
    if !ctx.is(node, "TypeApp")? {
        return Ok(None);
    }
    let constructor: String = node.getattr("constructor")?.extract()?;
    let generic = match generic_named(ctx, module, &constructor)? {
        Some(generic) => generic,
        None => return Ok(None),
    };
    let args = pyast::field_list(node, "args")?;
    Ok(Some(decide(
        ctx,
        generic.bind(ctx.py),
        &args,
        module,
        argument_modules,
        argument_parts,
    )?))
}

/// The first file whose patterns fit, or why none does.
fn decide<'py>(
    ctx: &Ctx<'py>,
    generic: &Bound<'py, PyAny>,
    arguments: &[Bound<'py, PyAny>],
    argument_module: &Bound<'py, PyAny>,
    argument_modules: Option<&[Py<PyAny>]>,
    argument_parts: Option<&[Option<Vec<Py<PyAny>>>]>,
) -> VibaResult<Py<PyAny>> {
    let where_modules: Vec<Py<PyAny>> = match argument_modules {
        Some(modules) if !modules.is_empty() => modules
            .iter()
            .map(|module| module.clone_ref(ctx.py))
            .collect(),
        _ => (0..arguments.len())
            .map(|_| argument_module.clone().unbind())
            .collect(),
    };
    let entries: Vec<Bound<'py, PyAny>> = generic
        .getattr("entries")?
        .try_iter()?
        .collect::<PyResult<Vec<_>>>()?;
    let mut has_declared = false;
    for entry in &entries {
        if !entry.getattr("declared")?.is_none() {
            has_declared = true;
            break;
        }
    }
    let offered = if has_declared {
        offered_parts(ctx, arguments, &where_modules)?
    } else {
        None
    };

    for entry in &entries {
        if let Some(offered) = &offered {
            if let Ok(declared) = entry.getattr("declared") {
                if !declared.is_none() {
                    let declared: i64 = declared.extract()?;
                    if !offered.contains(&declared) {
                        continue;
                    }
                }
            }
        }
        let got = entry.call_method0("take")?;
        if type_system::is_err(&got)? {
            return Err(VibaError::Program(err_message(&got)?));
        }
        let pair = got.getattr("ok_value")?;
        let module = pair.get_item(0)?;
        let patterns = pair.get_item(1)?;
        let patterns: Vec<Bound<'py, PyAny>> =
            patterns.try_iter()?.collect::<PyResult<Vec<_>>>()?;
        if patterns.len() != arguments.len() {
            continue;
        }
        let mut bindings: Py<PyAny> = PyDict::new(ctx.py).into_any().unbind();
        let members: Vec<Option<Vec<Option<Py<PyAny>>>>> = match argument_parts {
            Some(parts) if !parts.is_empty() => parts
                .iter()
                .map(|one| {
                    one.as_ref().map(|modules| {
                        modules
                            .iter()
                            .map(|module| Some(module.clone_ref(ctx.py)))
                            .collect()
                    })
                })
                .collect(),
            _ => (0..arguments.len()).map(|_| None).collect(),
        };
        let mut fits = true;
        let pairs = std::cmp::min(
            std::cmp::min(patterns.len(), arguments.len()),
            std::cmp::min(where_modules.len(), members.len()),
        );
        for index in 0..pairs {
            let source_module = where_modules[index].bind(ctx.py);
            let member_modules: Option<&[Option<Py<PyAny>>]> = members[index].as_deref();
            let matched = structural_pattern_match(
                ctx,
                &patterns[index],
                &module,
                &arguments[index],
                source_module,
                Some(bindings.bind(ctx.py)),
                member_modules,
            )?;
            match matched {
                Some(found) => bindings = found,
                None => {
                    fits = false;
                    break;
                }
            }
        }
        if !fits {
            continue;
        }
        // A file with no `__decl__` puts its answer in a named definition
        // (`value`, `impl`): it is its own module, and the caller spells which
        // definition it takes.
        let body = module.getattr("module")?;
        let body = body.getattr("body")?;
        let declared = definition(&body, DEF_NAME)?;
        let declared: Py<PyAny> = match declared {
            Some(declared) => declared.bind(ctx.py).getattr("body")?.unbind(),
            None => ctx.py.None(),
        };
        return Ok(make_choice(ctx, entry, declared, &module, bindings)?);
    }

    // Nothing fit: every file is taken, whatever its count said, so the answer
    // names what the directory holds and a file that cannot be taken is
    // reported here.
    let mut arities: BTreeSet<usize> = BTreeSet::new();
    for entry in &entries {
        let got = entry.call_method0("take")?;
        if type_system::is_err(&got)? {
            return Err(VibaError::Program(err_message(&got)?));
        }
        arities.insert(entry.getattr("patterns")?.len()?);
    }
    let mut spelled: Vec<String> = Vec::new();
    for argument in arguments {
        spelled.push(unparser::unparse_type(ctx.py, argument, 2)?);
    }
    let spelled = spelled.join(", ");
    let generic_name: String = generic.getattr("name")?.extract()?;
    if !arities.is_empty() && !arities.contains(&arguments.len()) {
        return Err(VibaError::Program(format!(
            "generic {} takes {} parameters, not {}: [{}]",
            repr_text(ctx.py, &generic_name),
            counted(&arities),
            arguments.len(),
            spelled
        )));
    }
    Err(VibaError::Program(format!(
        "no pattern of {} matches [{}]: the decision failed",
        repr_text(ctx.py, &generic_name),
        spelled
    )))
}

/// Every number of parts these arguments offer, or None when they offer no count.
fn offered_parts<'py>(
    ctx: &Ctx<'py>,
    arguments: &[Bound<'py, PyAny>],
    where_modules: &[Py<PyAny>],
) -> VibaResult<Option<BTreeSet<i64>>> {
    let mut totals: BTreeSet<i64> = BTreeSet::new();
    totals.insert(0);
    let pairs = std::cmp::min(arguments.len(), where_modules.len());
    for index in 0..pairs {
        let counts = parts_of_argument(ctx, &arguments[index], where_modules[index].bind(ctx.py))?;
        if counts.is_empty() {
            return Ok(None);
        }
        let mut next: BTreeSet<i64> = BTreeSet::new();
        for total in &totals {
            for count in &counts {
                next.insert(total + count);
            }
        }
        totals = next;
        if totals.len() > MOST_TOTALS {
            return Ok(None);
        }
    }
    Ok(Some(totals))
}

/// The numbers of parts this argument offers, in every way it may be taken.
fn parts_of_argument<'py>(
    ctx: &Ctx<'py>,
    argument: &Bound<'py, PyAny>,
    module: &Bound<'py, PyAny>,
) -> VibaResult<BTreeSet<i64>> {
    let mut counts: BTreeSet<i64> = BTreeSet::new();
    if ctx.is(argument, "Partial")? {
        let (_head, given) = application_parts(ctx, argument)?;
        counts.insert(given.len() as i64);
    }
    let node = match argument_as_type(ctx, argument, module) {
        Ok((node, _where)) => Some(node),
        Err(VibaError::Py(error)) => return Err(VibaError::Py(error)),
        Err(_) => None, // the matching reports it where it lands, if it does
    };
    if let Some(node) = node {
        if let Some(count) = parts_of_node(ctx, node.bind(ctx.py))? {
            counts.insert(count);
        }
    }
    Ok(counts)
}

/// How many parts this type is taken apart into, or None when it is not.
fn parts_of_node(ctx: &Ctx<'_>, node: &Bound<'_, PyAny>) -> PyResult<Option<i64>> {
    if is_sum(ctx, node)? {
        return Ok(None);
    }
    if is_product(ctx, node)? {
        return Ok(Some(
            crate::partial::product_elements(ctx, node)?.len() as i64
        ));
    }
    if is_exponent(ctx, node)? {
        return Ok(Some(exponent_elements(ctx, node)?.len() as i64));
    }
    if ctx.is(node, "Tuple")? {
        return Ok(Some(pyast::field_list(node, "elements")?.len() as i64));
    }
    if ctx.is(node, "Tagged")? {
        return Ok(Some(1));
    }
    if ctx.is(node, "TypeApp")? {
        return Ok(Some(pyast::field_list(node, "args")?.len() as i64));
    }
    Ok(None)
}

/// How many parts this `pattern` line takes, or None when it takes no count.
fn parts_of_pattern(
    ctx: &Ctx<'_>,
    pattern: &Bound<'_, PyAny>,
    module: &Bound<'_, PyAny>,
) -> PyResult<Option<i64>> {
    if ctx.is(pattern, "Ellipsis")? {
        return Ok(None);
    }
    if ctx.is(pattern, "TypeApp")?
        && pattern.getattr("constructor")?.extract::<String>()? == tagged::TAGGED_NAME
    {
        return Ok(Some(1));
    }
    if ctx.is(pattern, "Partial")? {
        let (_head, given) = application_parts(ctx, pattern)?;
        return Ok(Some(given.len() as i64));
    }
    if is_parameter(ctx, pattern, module)? {
        return Ok(None);
    }
    if !has_parameter(ctx, pattern, module)? {
        return Ok(None);
    }
    if is_sum(ctx, pattern)? {
        return Ok(None); // one part per branch the argument gave
    }
    if is_product(ctx, pattern)? {
        return Ok(Some(
            crate::partial::product_elements(ctx, pattern)?.len() as i64
        ));
    }
    if is_exponent(ctx, pattern)? {
        return Ok(Some(exponent_elements(ctx, pattern)?.len() as i64));
    }
    if ctx.is(pattern, "Tuple")? {
        return Ok(Some(pyast::field_list(pattern, "elements")?.len() as i64));
    }
    if ctx.is(pattern, "Tagged")? {
        return Ok(Some(1));
    }
    if ctx.is(pattern, "TypeApp")? {
        return Ok(Some(pyast::field_list(pattern, "args")?.len() as i64));
    }
    Ok(None)
}

/// Why the count in this file's name disagrees with its patterns, or None.
fn declared_problem(ctx: &Ctx<'_>, entry: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
    let declared = entry.getattr("declared")?;
    if declared.is_none() {
        return Ok(None);
    }
    let declared: i64 = declared.extract()?;
    let path: String = entry.getattr("path")?.extract()?;
    let patterns = entry.getattr("patterns")?;
    let module = entry.getattr("module")?;
    let mut taken: i64 = 0;
    for pattern in patterns.try_iter()? {
        let pattern = pattern?;
        match parts_of_pattern(ctx, &pattern, &module)? {
            Some(count) => taken += count,
            None => {
                return Ok(Some(format!(
                    "{}: the name says the file takes {} parts, and a `pattern` line in it takes no count",
                    path, declared
                )))
            }
        }
    }
    if taken != declared {
        return Ok(Some(format!(
            "{}: the name says the file takes {} parts, and its `pattern` lines take {}",
            path, declared, taken
        )));
    }
    Ok(None)
}

/// `1`, `1 or 2`, `1, 2 or 3` — how many parameters the files declare.
fn counted(arities: &BTreeSet<usize>) -> String {
    let spelled: Vec<String> = arities.iter().map(|number| number.to_string()).collect();
    if spelled.len() == 1 {
        return spelled[0].clone();
    }
    let last = spelled[spelled.len() - 1].clone();
    let head = spelled[..spelled.len() - 1].join(", ");
    format!("{} or {}", head, last)
}

// ----------------------------------------------------------------------
// Matching one pattern against one type in the source
// ----------------------------------------------------------------------

/// Bind this pattern's parameters to what the argument has in their place.
///
/// Ok(Some(bindings)) when the argument fits the pattern, Ok(None) when it does
/// not, Err when the pattern cannot be taken at all.
fn structural_pattern_match<'py>(
    ctx: &Ctx<'py>,
    pattern: &Bound<'py, PyAny>,
    pattern_module: &Bound<'py, PyAny>,
    argument: &Bound<'py, PyAny>,
    argument_module: &Bound<'py, PyAny>,
    bindings: Option<&Bound<'py, PyAny>>,
    parts: Option<&[Option<Py<PyAny>>]>,
) -> VibaResult<Option<Py<PyAny>>> {
    let bindings: Py<PyAny> = match bindings {
        Some(bindings) => bindings.clone().unbind(),
        None => PyDict::new(ctx.py).into_any().unbind(),
    };
    match match_node(
        ctx,
        pattern,
        pattern_module,
        argument,
        argument_module,
        bindings.bind(ctx.py),
        parts,
    ) {
        Ok(matched) => Ok(matched),
        Err(VibaError::Py(error)) => Err(VibaError::Py(error)),
        Err(error) => Err(VibaError::Program(error.message())),
    }
}

/// The bindings this part fits with, or None when it does not fit.
fn match_node<'py>(
    ctx: &Ctx<'py>,
    pattern: &Bound<'py, PyAny>,
    pattern_module: &Bound<'py, PyAny>,
    argument: &Bound<'py, PyAny>,
    argument_module: &Bound<'py, PyAny>,
    bindings: &Bound<'py, PyAny>,
    parts: Option<&[Option<Py<PyAny>>]>,
) -> VibaResult<Option<Py<PyAny>>> {
    if ctx.is(pattern, "Ellipsis")? {
        return Err(VibaError::Program(
            "ellipsis is not allowed in a `pattern` line: a pattern is one type".to_string(),
        ));
    }
    if ctx.is(pattern, "TypeApp")?
        && pattern.getattr("constructor")?.extract::<String>()? == tagged::TAGGED_NAME
    {
        return match_tagged(
            ctx,
            pattern,
            pattern_module,
            argument,
            argument_module,
            bindings,
            parts,
        );
    }
    if ctx.is(pattern, "Partial")? {
        return match_call(
            ctx,
            pattern,
            pattern_module,
            argument,
            argument_module,
            bindings,
        );
    }
    if is_parameter(ctx, pattern, pattern_module)? {
        let name: String = pattern.getattr("name")?.extract()?;
        if bindings.contains(name.as_str())? {
            let bound = bindings.get_item(name.as_str())?;
            let handed = type_of(ctx, argument, argument_module)?;
            if same_type(ctx, &bound, handed.bind(ctx.py)) {
                return Ok(Some(bindings.clone().unbind()));
            }
            return Ok(None);
        }
        let handed = type_of(ctx, argument, argument_module)?;
        bindings.set_item(name.as_str(), handed)?;
        return Ok(Some(bindings.clone().unbind()));
    }

    if !has_parameter(ctx, pattern, pattern_module)? {
        // Nothing to extract here: the type in the source is the whole question.
        let given = type_of(ctx, argument, argument_module)?;
        let declared = type_of(ctx, pattern, pattern_module)?;
        if judge_ast(ctx.py, given.bind(ctx.py), declared.bind(ctx.py)) {
            return Ok(Some(bindings.clone().unbind()));
        }
        return Ok(None);
    }

    let (unfolded, own_module) = argument_as_type(ctx, argument, argument_module)?;
    let unit_parts = product_parts(ctx, unfolded.bind(ctx.py), own_module.bind(ctx.py))?;
    let mut node = unfolded.clone_ref(ctx.py);
    if let Some(members) = &unit_parts {
        // A product's units carry no member, and a product of units alone is the
        // unit itself. What is left is what the pattern sees.
        if members.is_empty() {
            node = ctx.make0("Nil")?;
        } else if members.len() == 1 {
            node = members[0].clone_ref(ctx.py);
        }
    }
    let node_bound = node.bind(ctx.py);
    let module_bound = own_module.bind(ctx.py);

    if is_sum(ctx, pattern)? {
        return match_sum(
            ctx,
            pattern,
            pattern_module,
            node_bound,
            module_bound,
            bindings,
        );
    }
    if is_product(ctx, pattern)? {
        let pattern_parts = crate::partial::product_elements(ctx, pattern)?;
        let arguments: Vec<Py<PyAny>> = match &unit_parts {
            Some(members) => members
                .iter()
                .map(|member| member.clone_ref(ctx.py))
                .collect(),
            None => vec![node.clone_ref(ctx.py)],
        };
        return match_parts(
            ctx,
            &pattern_parts,
            pattern_module,
            &arguments,
            module_bound,
            bindings,
            parts,
        );
    }
    if is_exponent(ctx, pattern)? {
        let pattern_parts = exponent_elements(ctx, pattern)?;
        let arguments = if is_exponent(ctx, node_bound)? {
            exponent_elements(ctx, node_bound)?
        } else {
            vec![node.clone_ref(ctx.py)]
        };
        return match_parts(
            ctx,
            &pattern_parts,
            pattern_module,
            &arguments,
            module_bound,
            bindings,
            None,
        );
    }
    if ctx.is(pattern, "Tuple")? {
        if !ctx.is(node_bound, "Tuple")? {
            return Ok(None);
        }
        let pattern_elements: Vec<Py<PyAny>> = pyast::field_list(pattern, "elements")?
            .into_iter()
            .map(|element| element.unbind())
            .collect();
        let elements: Vec<Py<PyAny>> = pyast::field_list(node_bound, "elements")?
            .into_iter()
            .map(|element| element.unbind())
            .collect();
        return match_parts(
            ctx,
            &pattern_elements,
            pattern_module,
            &elements,
            module_bound,
            bindings,
            None,
        );
    }
    if ctx.is(pattern, "Tagged")? {
        if !ctx.is(node_bound, "Tagged")? {
            return Ok(None);
        }
        let pattern_tag: String = pattern.getattr("tag")?.extract()?;
        let node_tag: String = node_bound.getattr("tag")?.extract()?;
        if pattern_tag != node_tag {
            return Ok(None);
        }
        return match_node(
            ctx,
            &pattern.getattr("type")?,
            pattern_module,
            &node_bound.getattr("type")?,
            module_bound,
            bindings,
            None,
        );
    }
    if ctx.is(pattern, "TypeApp")? {
        if !ctx.is(node_bound, "TypeApp")? {
            return Ok(None);
        }
        let pattern_constructor: String = pattern.getattr("constructor")?.extract()?;
        let node_constructor: String = node_bound.getattr("constructor")?.extract()?;
        if constructor_key(ctx, &pattern_constructor, pattern_module)?
            != constructor_key(ctx, &node_constructor, module_bound)?
        {
            return Ok(None);
        }
        let pattern_args: Vec<Py<PyAny>> = pyast::field_list(pattern, "args")?
            .into_iter()
            .map(|argument| argument.unbind())
            .collect();
        let node_args: Vec<Py<PyAny>> = pyast::field_list(node_bound, "args")?
            .into_iter()
            .map(|argument| argument.unbind())
            .collect();
        return match_parts(
            ctx,
            &pattern_args,
            pattern_module,
            &node_args,
            module_bound,
            bindings,
            None,
        );
    }
    Ok(None)
}

/// `F << A`: the closure a call in the source is.
fn match_call<'py>(
    ctx: &Ctx<'py>,
    pattern: &Bound<'py, PyAny>,
    pattern_module: &Bound<'py, PyAny>,
    argument: &Bound<'py, PyAny>,
    argument_module: &Bound<'py, PyAny>,
    bindings: &Bound<'py, PyAny>,
) -> VibaResult<Option<Py<PyAny>>> {
    if !ctx.is(argument, "Partial")? {
        return Ok(None);
    }
    let (head, given) = application_parts(ctx, argument)?;
    for one in &given {
        if crate::partial::is_the_environment(
            ctx,
            one.bind(ctx.py),
            Some(argument_module),
            Some(&PartialJudge),
        )? {
            return Ok(None);
        }
    }
    let (wanted_head, wanted) = application_parts(ctx, pattern)?;
    if given.len() != wanted.len() {
        return Ok(None); // the pattern spells exactly this many
    }
    let mut matched = match match_node(
        ctx,
        wanted_head.bind(ctx.py),
        pattern_module,
        head.bind(ctx.py),
        argument_module,
        bindings,
        None,
    )? {
        Some(matched) => matched,
        None => return Ok(None),
    };
    for index in 0..wanted.len() {
        let found = match_node(
            ctx,
            wanted[index].bind(ctx.py),
            pattern_module,
            given[index].bind(ctx.py),
            argument_module,
            matched.bind(ctx.py),
            None,
        )?;
        matched = match found {
            Some(found) => found,
            None => return Ok(None),
        };
    }
    Ok(Some(matched))
}

/// A call in the source taken apart: (the head, the arguments in source order).
fn application_parts<'py>(
    ctx: &Ctx<'py>,
    node: &Bound<'py, PyAny>,
) -> PyResult<(Py<PyAny>, Vec<Py<PyAny>>)> {
    let mut given: Vec<Py<PyAny>> = Vec::new();
    let mut current = node.clone().unbind();
    while ctx.is(current.bind(ctx.py), "Partial")? {
        let bound = current.bind(ctx.py);
        given.push(bound.getattr("argument")?.unbind());
        current = bound.getattr("function")?.unbind();
    }
    given.reverse();
    Ok((current, given))
}

/// `tagged[S, T]`: the argument has to be the tag S spells.
fn match_tagged<'py>(
    ctx: &Ctx<'py>,
    pattern: &Bound<'py, PyAny>,
    pattern_module: &Bound<'py, PyAny>,
    argument: &Bound<'py, PyAny>,
    argument_module: &Bound<'py, PyAny>,
    bindings: &Bound<'py, PyAny>,
    parts: Option<&[Option<Py<PyAny>>]>,
) -> VibaResult<Option<Py<PyAny>>> {
    let args = pyast::field_list(pattern, "args")?;
    if let Some(problem) = tagged::tagged_problem(ctx, &args)? {
        return Err(VibaError::Program(problem));
    }
    let (node, module) = argument_as_type(ctx, argument, argument_module)?;
    let node_bound = node.bind(ctx.py);
    let wanted = tagged::literal_symbol(ctx, &args[0])?;
    let bindings: Py<PyAny> = bindings.clone().unbind();
    let bindings = match wanted {
        Some(wanted) => {
            if !(ctx.is(node_bound, "Tagged")? || ctx.is(node_bound, "Member")?) {
                return Ok(None);
            }
            let tag: String = node_bound.getattr("tag")?.extract()?;
            if tag != tagged::tag_of(&wanted) {
                return Ok(None);
            }
            bindings
        }
        None => {
            if !(ctx.is(node_bound, "Tagged")? || ctx.is(node_bound, "Member")?) {
                return Ok(None);
            }
            let tag: String = node_bound.getattr("tag")?.extract()?;
            let text: String = tag.chars().skip(1).collect();
            let constant = ctx.make("Constant", &[("value", ctx.pystr(&text))])?;
            let taken = match_node(
                ctx,
                &args[0],
                pattern_module,
                constant.bind(ctx.py),
                pattern_module,
                bindings.bind(ctx.py),
                None,
            )?;
            match taken {
                Some(taken) => taken,
                None => return Ok(None),
            }
        }
    };
    if args.len() == 1 {
        return Ok(Some(bindings));
    }
    if !ctx.is(node_bound, "Tagged")? {
        return Ok(None);
    }
    let mut module = module;
    if let Some(parts) = parts {
        if parts.len() == 1 {
            if let Some(own) = &parts[0] {
                module = own.clone_ref(ctx.py);
            }
        }
    }
    match_node(
        ctx,
        &args[1],
        pattern_module,
        &node_bound.getattr("type")?,
        module.bind(ctx.py),
        bindings.bind(ctx.py),
        None,
    )
}

/// A product's members with its units gone; None when it is no product.
fn product_parts<'py>(
    ctx: &Ctx<'py>,
    node: &Bound<'py, PyAny>,
    module: &Bound<'py, PyAny>,
) -> VibaResult<Option<Vec<Py<PyAny>>>> {
    if !is_product(ctx, node)? {
        return Ok(None);
    }
    let unit_node = ctx.make0("Nil")?;
    let unit = type_of(ctx, unit_node.bind(ctx.py), module)?;
    let mut members: Vec<Py<PyAny>> = Vec::new();
    for part in crate::partial::product_elements(ctx, node)? {
        let part_type = type_of(ctx, part.bind(ctx.py), module)?;
        if !judge_ast(ctx.py, part_type.bind(ctx.py), unit.bind(ctx.py)) {
            members.push(part);
        }
    }
    Ok(Some(members))
}

/// Parts against parts, in order: every pair has to fit, sharing bindings.
fn match_parts<'py>(
    ctx: &Ctx<'py>,
    patterns: &[Py<PyAny>],
    pattern_module: &Bound<'py, PyAny>,
    arguments: &[Py<PyAny>],
    argument_module: &Bound<'py, PyAny>,
    bindings: &Bound<'py, PyAny>,
    parts: Option<&[Option<Py<PyAny>>]>,
) -> VibaResult<Option<Py<PyAny>>> {
    if patterns.len() != arguments.len() {
        return Ok(None);
    }
    let mut bindings: Py<PyAny> = bindings.clone().unbind();
    for index in 0..patterns.len() {
        let own: &Bound<'py, PyAny> = match parts {
            Some(parts) if index < parts.len() => match &parts[index] {
                Some(own) => own.bind(ctx.py),
                None => argument_module,
            },
            _ => argument_module,
        };
        let found = match_node(
            ctx,
            patterns[index].bind(ctx.py),
            pattern_module,
            arguments[index].bind(ctx.py),
            own,
            bindings.bind(ctx.py),
            None,
        )?;
        bindings = match found {
            Some(found) => found,
            None => return Ok(None),
        };
    }
    Ok(Some(bindings))
}

/// A sum pattern: every branch of the argument fits some branch of it.
fn match_sum<'py>(
    ctx: &Ctx<'py>,
    pattern: &Bound<'py, PyAny>,
    pattern_module: &Bound<'py, PyAny>,
    node: &Bound<'py, PyAny>,
    module: &Bound<'py, PyAny>,
    bindings: &Bound<'py, PyAny>,
) -> VibaResult<Option<Py<PyAny>>> {
    let branches: Vec<Py<PyAny>> = if is_sum(ctx, node)? {
        sum_elements(ctx, node)?
    } else {
        vec![node.clone().unbind()]
    };
    let alternatives = sum_elements(ctx, pattern)?;
    let mut bindings: Py<PyAny> = bindings.clone().unbind();
    for branch in &branches {
        let mut found: Option<Py<PyAny>> = None;
        for alternative in &alternatives {
            let attempt_bindings = dict_copy(ctx, bindings.bind(ctx.py))?;
            let attempt = match_node(
                ctx,
                alternative.bind(ctx.py),
                pattern_module,
                branch.bind(ctx.py),
                module,
                attempt_bindings.bind(ctx.py),
                None,
            )?;
            if let Some(attempt) = attempt {
                found = Some(attempt);
                break;
            }
        }
        match found {
            Some(found) => bindings = found,
            None => return Ok(None),
        }
    }
    Ok(Some(bindings))
}

fn dict_copy<'py>(ctx: &Ctx<'py>, bindings: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
    if let Ok(dict) = bindings.cast::<PyDict>() {
        return Ok(dict.copy()?.into_any().unbind());
    }
    let builtins = PyModule::import(ctx.py, "builtins")?;
    Ok(builtins.getattr("dict")?.call1((bindings,))?.unbind())
}

// ----------------------------------------------------------------------
// Names, structure, and what fits
// ----------------------------------------------------------------------

/// The branches of a sum in the source, flattened in source order.
fn sum_elements<'py>(ctx: &Ctx<'py>, node: &Bound<'py, PyAny>) -> PyResult<Vec<Py<PyAny>>> {
    if ctx.is(node, "Sum")? {
        let mut elements = sum_elements(ctx, &node.getattr("left")?)?;
        elements.extend(sum_elements(ctx, &node.getattr("right")?)?);
        return Ok(elements);
    }
    if ctx.is(node, "SumChain")? {
        return Ok(pyast::field_list(node, "elements")?
            .into_iter()
            .map(|element| element.unbind())
            .collect());
    }
    Ok(vec![node.clone().unbind()])
}

/// A function in the source taken apart: the result first, the arguments after.
fn exponent_elements<'py>(ctx: &Ctx<'py>, node: &Bound<'py, PyAny>) -> PyResult<Vec<Py<PyAny>>> {
    let mut elements = raw_exponent_elements(ctx, node)?;
    if elements.len() > 1 {
        let last = elements[elements.len() - 1].bind(ctx.py);
        if ctx.is(last, "HintBlock")? || ctx.is(last, "Partial")? {
            elements.pop();
        }
    }
    Ok(elements)
}

/// The elements of an exponent chain in the source, body and all.
fn raw_exponent_elements<'py>(
    ctx: &Ctx<'py>,
    node: &Bound<'py, PyAny>,
) -> PyResult<Vec<Py<PyAny>>> {
    if ctx.is(node, "Exponent")? {
        let mut elements = raw_exponent_elements(ctx, &node.getattr("result")?)?;
        elements.push(node.getattr("argument")?.unbind());
        return Ok(elements);
    }
    if ctx.is(node, "ExponentChain")? {
        return Ok(pyast::field_list(node, "elements")?
            .into_iter()
            .map(|element| element.unbind())
            .collect());
    }
    Ok(vec![node.clone().unbind()])
}

/// A name the pattern's module never defines: a parameter to extract.
fn is_parameter(
    ctx: &Ctx<'_>,
    node: &Bound<'_, PyAny>,
    module: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    if !ctx.is(node, "TypeRef")? {
        return Ok(false);
    }
    let name: String = node.getattr("name")?.extract()?;
    let resolved = type_system::module_get_type(ctx.py, module, &name, None)?;
    type_system::is_err(resolved.bind(ctx.py))
}

/// Whether a parameter sits anywhere in this pattern.
fn has_parameter(
    ctx: &Ctx<'_>,
    node: &Bound<'_, PyAny>,
    module: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    for part in crate::partial::walk(ctx, node)? {
        if is_parameter(ctx, part.bind(ctx.py), module)? {
            return Ok(true);
        }
    }
    Ok(false)
}

/// The argument as a type: its names unfolded, its `<<` in the source given.
fn argument_as_type<'py>(
    ctx: &Ctx<'py>,
    node: &Bound<'py, PyAny>,
    module: &Bound<'py, PyAny>,
) -> VibaResult<(Py<PyAny>, Py<PyAny>)> {
    let (node, module) = unfold(ctx, node, module)?;
    let tagged = tagged_type_of(ctx, node.bind(ctx.py), module.bind(ctx.py), None)?;
    let node = match tagged {
        Some(tagged) => tagged,
        None => node,
    };
    if !ctx.is(node.bind(ctx.py), "Partial")? {
        return Ok((node, module));
    }
    crate::partial::reduce_partial(
        ctx,
        node.bind(ctx.py),
        module.bind(ctx.py),
        &PartialTarget,
        &PartialJudge,
        true,
    )
}

/// (body, source module) for the name a `<<` in the source gives to, or None.
fn partial_target(
    ctx: &Ctx<'_>,
    name: &str,
    module: &Bound<'_, PyAny>,
) -> VibaResult<Option<(Py<PyAny>, Py<PyAny>)>> {
    let resolved = type_system::module_get_type(ctx.py, module, name, None)?;
    let resolved = resolved.bind(ctx.py);
    if !type_system::is_err(resolved)? {
        let target = resolved.getattr("ok_value")?;
        if ctx.is(&target, "AstNodeType")? {
            let body = target.getattr("ast_node")?;
            if ctx.is(&body, "GenericDefinition")? {
                return Ok(None); // a bare generic name has no body
            }
            let body = if ctx.is(&body, "TypeDefinition")? {
                body.getattr("body")?
            } else {
                body
            };
            return Ok(Some((
                body.unbind(),
                target.getattr("container_module")?.unbind(),
            )));
        }
    }
    crate::partial::module_as_function(ctx, module, name)
}

/// A name is transparent: unfold it to the structure it stands for.
fn unfold<'py>(
    ctx: &Ctx<'py>,
    node: &Bound<'py, PyAny>,
    module: &Bound<'py, PyAny>,
) -> VibaResult<(Py<PyAny>, Py<PyAny>)> {
    let mut node = node.clone().unbind();
    let mut module = module.clone().unbind();
    let mut seen: HashSet<String> = HashSet::new();
    loop {
        let current = node.bind(ctx.py);
        if !ctx.is(current, "TypeRef")? {
            return Ok((node, module));
        }
        let name: String = current.getattr("name")?.extract()?;
        if seen.contains(&name) {
            return Ok((node, module));
        }
        seen.insert(name.clone());
        let resolved = type_system::module_get_type(ctx.py, module.bind(ctx.py), &name, None)?;
        let resolved = resolved.bind(ctx.py);
        let target = if !type_system::is_err(resolved)? {
            let value = resolved.getattr("ok_value")?;
            if ctx.is(&value, "AstNodeType")? {
                Some(value)
            } else {
                None
            }
        } else {
            None
        };
        let target = match target {
            Some(target) => target,
            None => {
                // A module name: what it is as a type is the `__decl__` it gave.
                return match module_def_body(ctx, module.bind(ctx.py), &name)? {
                    Some((body, source_module)) => Ok((body, source_module)),
                    None => Ok((node, module)),
                };
            }
        };
        let body = target.getattr("ast_node")?;
        if ctx.is(&body, "GenericDefinition")? {
            return Ok((node, module));
        }
        let body = if ctx.is(&body, "TypeDefinition")? {
            body.getattr("body")?
        } else {
            body
        };
        node = body.unbind();
        module = target.getattr("container_module")?.unbind();
    }
}

/// (the `__decl__` chain of the module this name binds, that module) or None.
fn module_def_body<'py>(
    ctx: &Ctx<'py>,
    module: &Bound<'py, PyAny>,
    name: &str,
) -> VibaResult<Option<(Py<PyAny>, Py<PyAny>)>> {
    let imports = match module.getattr("imports") {
        Ok(imports) if !imports.is_none() => imports,
        _ => return Ok(None),
    };
    if !imports.contains(name)? {
        return Ok(None);
    }
    let module_name = imports.get_item(name)?;
    let handed = module.call_method1("module_environment", (module_name,))?;
    if type_system::is_err(&handed)? {
        return Ok(None);
    }
    let handed = handed.getattr("ok_value")?;
    if ctx.is(&handed, "GenericModuleType")? {
        return Ok(None);
    }
    let body = match module_body(&handed)? {
        Some(body) => body,
        None => return Ok(None),
    };
    let definition = definition(&body, DEF_NAME)?;
    match definition {
        Some(definition) => Ok(Some((
            definition.bind(ctx.py).getattr("body")?.unbind(),
            handed.unbind(),
        ))),
        None => Ok(None),
    }
}

/// What a constructor names, as something two sides compare by.
fn constructor_key(
    ctx: &Ctx<'_>,
    constructor: &str,
    module: &Bound<'_, PyAny>,
) -> VibaResult<String> {
    let reference = ctx.make("TypeRef", &[("name", ctx.pystr(constructor))])?;
    let (node, _module) = unfold(ctx, reference.bind(ctx.py), module)?;
    let node = node.bind(ctx.py);
    if ctx.is(node, "TypeRef")? {
        return Ok(node.getattr("name")?.extract()?);
    }
    if ctx.is(node, "TypeApp")? {
        return Ok(node.getattr("constructor")?.extract()?);
    }
    Ok(constructor.to_string())
}

// The two callbacks `partial::reduce_partial` takes, as this module takes them.

struct PartialTarget;

impl ResolveFn for PartialTarget {
    fn resolve(
        &self,
        ctx: &Ctx<'_>,
        name: &str,
        module: &Bound<'_, PyAny>,
    ) -> VibaResult<Option<(Py<PyAny>, Py<PyAny>)>> {
        partial_target(ctx, name, module)
    }
}

struct PartialJudge;

impl JudgeFn for PartialJudge {
    fn judge(
        &self,
        ctx: &Ctx<'_>,
        sub: &Bound<'_, PyAny>,
        sub_module: &Bound<'_, PyAny>,
        sup: &Bound<'_, PyAny>,
        sup_module: &Bound<'_, PyAny>,
    ) -> VibaResult<bool> {
        let sub = type_of(ctx, sub, sub_module)?;
        let sup = type_of(ctx, sup, sup_module)?;
        Ok(judge_ast(ctx.py, sub.bind(ctx.py), sup.bind(ctx.py)))
    }
}

// ----------------------------------------------------------------------
// Python-facing functions
// ----------------------------------------------------------------------

fn result_object(py: Python<'_>, outcome: VibaResult<Option<Py<PyAny>>>) -> PyResult<Py<PyAny>> {
    match outcome {
        Ok(Some(value)) => type_system::ok(py, value),
        Ok(None) => type_system::ok(py, py.None()),
        Err(VibaError::Py(error)) => Err(error),
        Err(error) => type_system::err(py, &error.message()),
    }
}

fn required_object(py: Python<'_>, outcome: VibaResult<Py<PyAny>>) -> PyResult<Py<PyAny>> {
    match outcome {
        Ok(value) => type_system::ok(py, value),
        Err(VibaError::Py(error)) => Err(error),
        Err(error) => type_system::err(py, &error.message()),
    }
}

fn parse_modules(value: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Vec<Py<PyAny>>>> {
    match value {
        Some(value) if !value.is_none() => {
            let mut modules: Vec<Py<PyAny>> = Vec::new();
            for item in value.try_iter()? {
                modules.push(item?.unbind());
            }
            Ok(Some(modules))
        }
        _ => Ok(None),
    }
}

fn parse_parts(value: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Vec<Option<Vec<Py<PyAny>>>>>> {
    match value {
        Some(value) if !value.is_none() => {
            let mut parts: Vec<Option<Vec<Py<PyAny>>>> = Vec::new();
            for item in value.try_iter()? {
                let item = item?;
                if !item.is_truthy()? {
                    parts.push(None);
                    continue;
                }
                let mut modules: Vec<Py<PyAny>> = Vec::new();
                for module in item.try_iter()? {
                    modules.push(module?.unbind());
                }
                parts.push(Some(modules));
            }
            Ok(Some(parts))
        }
        _ => Ok(None),
    }
}

fn parse_member_modules(
    value: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<Vec<Option<Py<PyAny>>>>> {
    match value {
        Some(value) if !value.is_none() => {
            let mut parts: Vec<Option<Py<PyAny>>> = Vec::new();
            for item in value.try_iter()? {
                let item = item?;
                if item.is_truthy()? {
                    parts.push(Some(item.unbind()));
                } else {
                    parts.push(None);
                }
            }
            Ok(Some(parts))
        }
        _ => Ok(None),
    }
}

#[pyfunction]
fn pattern_file_take_py(py: Python<'_>, entry: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let ctx = Ctx::new(py)?;
    let problem = entry.getattr("_problem")?;
    if !problem.is_none() {
        return Ok(problem.unbind());
    }
    let taken: bool = entry.getattr("_taken")?.extract()?;
    if !taken {
        let loader = entry.getattr("_load")?;
        let got = loader.call0()?;
        if type_system::is_err(&got)? {
            entry.setattr("_problem", &got)?;
            return Ok(got.unbind());
        }
        let pair = got.getattr("ok_value")?;
        let module = pair.get_item(0)?;
        let patterns = pair.get_item(1)?;
        entry.setattr("module", &module)?;
        entry.setattr("patterns", as_list(py, &patterns)?)?;
        entry.setattr("_taken", true)?;
    }
    let checked: bool = entry.getattr("_checked")?.extract()?;
    if !checked {
        entry.setattr("_checked", true)?;
        if let Some(problem) = declared_problem(&ctx, entry)? {
            entry.setattr("_problem", type_system::err(py, &problem)?)?;
        }
    }
    let problem = entry.getattr("_problem")?;
    if !problem.is_none() {
        return Ok(problem.unbind());
    }
    let module = entry.getattr("module")?;
    let patterns = entry.getattr("patterns")?;
    let pair = PyTuple::new(py, [module, patterns])?;
    type_system::ok(py, pair.into_any().unbind())
}

#[pyfunction]
fn patterns_of_py(py: Python<'_>, body: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    patterns_of(py, body)
}

#[pyfunction]
fn file_pattern_problem_py(
    py: Python<'_>,
    tree: &Bound<'_, PyAny>,
    file_name: &str,
) -> PyResult<Py<PyAny>> {
    match file_pattern_problem(py, tree, file_name)? {
        Some(problem) => Ok(pyast::pystr(py, &problem)),
        None => Ok(py.None()),
    }
}

#[pyfunction]
fn order_of_py(py: Python<'_>, file_name: &str) -> PyResult<Py<PyAny>> {
    Ok(match order_of(file_name) {
        Some(order) => pyast::pyint(py, order),
        None => py.None(),
    })
}

#[pyfunction]
fn declared_of_py(py: Python<'_>, file_name: &str) -> PyResult<Py<PyAny>> {
    Ok(match declared_of(file_name) {
        Some(declared) => pyast::pyint(py, declared),
        None => py.None(),
    })
}

#[pyfunction]
fn load_generic_py<'py>(
    py: Python<'py>,
    directory: &str,
    name: &str,
    source_of: &Bound<'py, PyAny>,
    listing: &Bound<'py, PyAny>,
    module_of: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    let ctx = Ctx::new(py)?;
    required_object(
        py,
        load_generic(&ctx, directory, name, source_of, listing, module_of),
    )
}

#[pyfunction]
fn generic_of_files_py<'py>(
    py: Python<'py>,
    name: &str,
    directory: &str,
    marker: &Bound<'py, PyAny>,
    parts: &Bound<'py, PyAny>,
    source_of: &Bound<'py, PyAny>,
    module_of: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    let ctx = Ctx::new(py)?;
    let mut parsed: Vec<FilePart> = Vec::new();
    for part in parts.try_iter()? {
        let part = part?;
        let order: i64 = part.get_item(0)?.extract()?;
        let path: String = part.get_item(1)?.extract()?;
        let module_name: String = part.get_item(2)?.extract()?;
        let declared = part.get_item(3)?;
        let declared: Option<i64> = if declared.is_none() {
            None
        } else {
            Some(declared.extract()?)
        };
        parsed.push(FilePart {
            order,
            path,
            module_name,
            declared,
        });
    }
    let marker: Option<&Bound<'py, PyAny>> = if marker.is_none() { None } else { Some(marker) };
    required_object(
        py,
        generic_of_files(&ctx, name, directory, marker, &parsed, source_of, module_of),
    )
}

#[pyfunction]
fn generic_of_entries_py<'py>(
    py: Python<'py>,
    name: &str,
    directory: &str,
    marker: &Bound<'py, PyAny>,
    parts: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    let ctx = Ctx::new(py)?;
    let mut parsed: Vec<EntryPart> = Vec::new();
    for part in parts.try_iter()? {
        let part = part?;
        let order: i64 = part.get_item(0)?.extract()?;
        let path: String = part.get_item(1)?.extract()?;
        let module_name: String = part.get_item(2)?.extract()?;
        let declared = part.get_item(3)?;
        let declared: Option<i64> = if declared.is_none() {
            None
        } else {
            Some(declared.extract()?)
        };
        parsed.push(EntryPart {
            order,
            path,
            module_name,
            declared,
            module: part.get_item(4)?.unbind(),
        });
    }
    let marker: Option<&Bound<'py, PyAny>> = if marker.is_none() { None } else { Some(marker) };
    required_object(
        py,
        generic_of_entries(&ctx, name, directory, marker, &parsed),
    )
}

#[pyfunction]
fn generic_named_py(py: Python<'_>, module: &Bound<'_, PyAny>, name: &str) -> PyResult<Py<PyAny>> {
    let ctx = Ctx::new(py)?;
    result_object(py, generic_named(&ctx, module, name))
}

#[pyfunction(signature = (node, module, resolve = None))]
fn tagged_type_of_py<'py>(
    py: Python<'py>,
    node: &Bound<'py, PyAny>,
    module: &Bound<'py, PyAny>,
    resolve: Option<&Bound<'py, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let ctx = Ctx::new(py)?;
    let callback = match resolve {
        Some(callback) if !callback.is_none() => Some(TextResolve { callback }),
        _ => None,
    };
    let resolver: Option<&dyn ResolveFn> = match &callback {
        Some(callback) => Some(callback as &dyn ResolveFn),
        None => None,
    };
    result_object(py, tagged_type_of(&ctx, node, module, resolver))
}

#[pyfunction(signature = (node, module, argument_modules = None, argument_parts = None))]
fn reduce_application_py<'py>(
    py: Python<'py>,
    node: &Bound<'py, PyAny>,
    module: &Bound<'py, PyAny>,
    argument_modules: Option<&Bound<'py, PyAny>>,
    argument_parts: Option<&Bound<'py, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let ctx = Ctx::new(py)?;
    let modules = parse_modules(argument_modules)?;
    let parts = parse_parts(argument_parts)?;
    result_object(
        py,
        reduce_application(&ctx, node, module, modules.as_deref(), parts.as_deref()),
    )
}

#[pyfunction(signature = (generic, arguments, argument_module, argument_modules = None, argument_parts = None))]
fn decide_py<'py>(
    py: Python<'py>,
    generic: &Bound<'py, PyAny>,
    arguments: &Bound<'py, PyAny>,
    argument_module: &Bound<'py, PyAny>,
    argument_modules: Option<&Bound<'py, PyAny>>,
    argument_parts: Option<&Bound<'py, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let ctx = Ctx::new(py)?;
    let arguments: Vec<Bound<'py, PyAny>> = arguments.try_iter()?.collect::<PyResult<Vec<_>>>()?;
    let modules = parse_modules(argument_modules)?;
    let parts = parse_parts(argument_parts)?;
    required_object(
        py,
        decide(
            &ctx,
            generic,
            &arguments,
            argument_module,
            modules.as_deref(),
            parts.as_deref(),
        ),
    )
}

#[pyfunction(signature = (pattern, pattern_module, argument, argument_module, bindings = None, parts = None))]
fn structural_pattern_match_py<'py>(
    py: Python<'py>,
    pattern: &Bound<'py, PyAny>,
    pattern_module: &Bound<'py, PyAny>,
    argument: &Bound<'py, PyAny>,
    argument_module: &Bound<'py, PyAny>,
    bindings: Option<&Bound<'py, PyAny>>,
    parts: Option<&Bound<'py, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let ctx = Ctx::new(py)?;
    let parts = parse_member_modules(parts)?;
    result_object(
        py,
        structural_pattern_match(
            &ctx,
            pattern,
            pattern_module,
            argument,
            argument_module,
            bindings,
            parts.as_deref(),
        ),
    )
}

#[pyfunction]
fn sum_elements_py(py: Python<'_>, node: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let ctx = Ctx::new(py)?;
    let elements = sum_elements(&ctx, node)?;
    ctx.list(elements)
}

#[pyfunction]
fn exponent_elements_py(py: Python<'_>, node: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let ctx = Ctx::new(py)?;
    let elements = exponent_elements(&ctx, node)?;
    ctx.list(elements)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pattern_file_take_py, m)?)?;
    m.add_function(wrap_pyfunction!(patterns_of_py, m)?)?;
    m.add_function(wrap_pyfunction!(file_pattern_problem_py, m)?)?;
    m.add_function(wrap_pyfunction!(order_of_py, m)?)?;
    m.add_function(wrap_pyfunction!(declared_of_py, m)?)?;
    m.add_function(wrap_pyfunction!(load_generic_py, m)?)?;
    m.add_function(wrap_pyfunction!(generic_of_files_py, m)?)?;
    m.add_function(wrap_pyfunction!(generic_of_entries_py, m)?)?;
    m.add_function(wrap_pyfunction!(generic_named_py, m)?)?;
    m.add_function(wrap_pyfunction!(tagged_type_of_py, m)?)?;
    m.add_function(wrap_pyfunction!(reduce_application_py, m)?)?;
    m.add_function(wrap_pyfunction!(decide_py, m)?)?;
    m.add_function(wrap_pyfunction!(structural_pattern_match_py, m)?)?;
    m.add_function(wrap_pyfunction!(sum_elements_py, m)?)?;
    m.add_function(wrap_pyfunction!(exponent_elements_py, m)?)?;
    Ok(())
}
