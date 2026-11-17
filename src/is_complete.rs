//! is_complete: does every address a design can list walk to a leaf?
//!
//! A port of `viba/is_complete.py`: the dependency pool is built here, and the
//! checker walks the design in Rust. The layers that were ported in parallel
//! (the descriptor pool, patterns, the AST) are called through their Python
//! bindings, which are thin bindings over Rust.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::rc::Rc;

use pyo3::exceptions::{PyOSError, PySyntaxError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyCFunction, PyDict, PyFloat, PyInt, PyModule, PyString};

use crate::pyast;
use crate::tagged::TAGGED_NAME;

const ENTRY_FILE_NAME: &str = "entry.viba";
const ENTRY_MODULE_NAME: &str = "entry";
const BUILTIN_MODULE_NAME: &str = "viba.builtin";

const LEAF_NAMES: [&str; 12] = [
    "bool", "int", "float", "str", "int8_t", "uint8_t", "int16_t", "uint16_t", "int32_t",
    "uint32_t", "int64_t", "uint64_t",
];
const CONTAINER_NAMES: [&str; 3] = ["list", "set", "dict"];
const UNIT_NAMES: [&str; 4] = ["Object", "nil", "Oneof", "never"];

type Obj = Py<PyAny>;

// ----------------------------------------------------------------------
// Bindings: what a definition's parameter names stand for in this walk
// ----------------------------------------------------------------------

struct Binding {
    node: Obj,
    module: String,
    bindings: Rc<Bindings>,
}

#[derive(Clone)]
enum BindingValue {
    Placeholder,
    Bound(Rc<Binding>),
}

#[derive(Default)]
struct Bindings {
    map: HashMap<String, BindingValue>,
}

impl Bindings {
    fn with(mut self) -> Rc<Bindings> {
        Rc::new(Bindings {
            map: std::mem::take(&mut self.map),
        })
    }

    fn copy_of(bindings: &Rc<Bindings>) -> Bindings {
        Bindings {
            map: bindings.map.clone(),
        }
    }
}

#[derive(Clone)]
enum TagKind {
    None,
    Text(String),
    Unknown,
}

struct OptionItem {
    tag: TagKind,
    name: Option<String>,
    known: bool,
}

struct FormRef {
    node: Obj,
    known: bool,
    module: String,
    bindings: Rc<Bindings>,
}

struct FoundDefinition {
    definition: Obj,
    module: String,
    bindings: Rc<Bindings>,
}

enum Decision {
    Failed,
    NoGeneric,
    Body(Obj, String, Rc<Bindings>),
}

// ----------------------------------------------------------------------
// Compile files, gather dependencies
// ----------------------------------------------------------------------

fn module_name_of(file_path: &str) -> String {
    let mut text = file_path.replace('\\', "/");
    text = text.trim_matches('/').to_string();
    for suffix in [".viba", ".md"] {
        if text.ends_with(suffix) {
            text.truncate(text.len() - suffix.len());
            break;
        }
    }
    text.replace('/', ".")
}

fn add_file(
    descriptor: &Bound<'_, PyModule>,
    err_class: &Bound<'_, PyAny>,
    pool: &Bound<'_, PyAny>,
    file_name: &str,
    module_name: &str,
    source: &str,
) -> PyResult<Option<Obj>> {
    let parsed =
        descriptor
            .getattr("parse_viba_file")?
            .call1((pool, source, file_name, module_name))?;
    if is_instance_of(&parsed, err_class)? {
        return Ok(None);
    }
    let added = descriptor
        .getattr("pool_add_file")?
        .call1((pool, parsed.getattr("ok_value")?))?;
    if is_instance_of(&added, err_class)? {
        return Ok(None);
    }
    Ok(Some(added.getattr("ok_value")?.unbind()))
}

fn is_instance_of(value: &Bound<'_, PyAny>, class: &Bound<'_, PyAny>) -> PyResult<bool> {
    value.is_instance(class)
}

/// The sort key Python's `sorted()` uses on `Path` objects: the parts.
fn path_key(path: &Path) -> Vec<String> {
    path.components()
        .map(|part| part.as_os_str().to_string_lossy().into_owned())
        .collect()
}

fn collect_matching(root: &Path, suffix: &str, out: &mut Vec<PathBuf>) {
    let entries = match std::fs::read_dir(root) {
        Ok(entries) => entries,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let is_match = entry.file_name().to_string_lossy().ends_with(suffix);
        if is_match {
            out.push(path.clone());
        }
        let metadata = match std::fs::symlink_metadata(&path) {
            Ok(metadata) => metadata,
            Err(_) => continue,
        };
        if metadata.file_type().is_dir() && !metadata.file_type().is_symlink() {
            collect_matching(&path, suffix, out);
        }
    }
}

fn written_as_document(path: &Path) -> bool {
    let parent = match path.parent() {
        Some(parent) => parent,
        None => return false,
    };
    let document = format!("{}.viba.md", parent.to_string_lossy());
    std::fs::metadata(document)
        .map(|metadata| metadata.is_file())
        .unwrap_or(false)
}

/// (file_name, module_name, source): the library first, then the caller's paths.
fn dependencies(
    py: Python<'_>,
    library: Option<&Bound<'_, PyAny>>,
    viba_paths: Option<&Bound<'_, PyAny>>,
) -> PyResult<Vec<(String, String, String)>> {
    let mut found: HashMap<String, (String, String)> = HashMap::new();
    let mut order: Vec<String> = Vec::new();

    if let Some(library) = library {
        if !library.is_none() {
            for item in library.try_iter()? {
                let item = item?;
                let file_path: String = item.get_item(0)?.str()?.extract()?;
                let file_content: String = item.get_item(1)?.extract()?;
                let module_name = module_name_of(&file_path);
                if !module_name.is_empty() && !found.contains_key(&module_name) {
                    found.insert(module_name.clone(), (file_path, file_content));
                    order.push(module_name);
                }
            }
        }
    }

    let markdown = PyModule::import(py, "rs_viba.markdown")?;
    if let Some(viba_paths) = viba_paths {
        if !viba_paths.is_none() {
            for directory in viba_paths.try_iter()? {
                let directory = directory?;
                let root = PathBuf::from(directory.str()?.extract::<String>()?);
                let root_dir = match std::fs::metadata(&root) {
                    Ok(metadata) if metadata.is_dir() => root.clone(),
                    _ => continue,
                };

                let mut documents: Vec<PathBuf> = Vec::new();
                collect_matching(&root_dir, ".viba.md", &mut documents);
                documents.sort_by_key(|path| path_key(path));
                for path in documents {
                    let relative = match path.strip_prefix(&root_dir) {
                        Ok(relative) => relative.to_string_lossy().into_owned(),
                        Err(_) => continue,
                    };
                    let mut stem = relative;
                    stem.truncate(stem.len().saturating_sub(".viba.md".len()));
                    let module_name = module_name_of(&stem);
                    if module_name.is_empty() {
                        continue;
                    }
                    let source = match std::fs::read_to_string(&path) {
                        Ok(source) => source,
                        Err(_) => continue,
                    };
                    let files = match markdown
                        .getattr("module_files")?
                        .call1((&module_name, source))
                    {
                        Ok(files) => files,
                        Err(error) => {
                            if error.is_instance_of::<PyOSError>(py)
                                || error.is_instance_of::<PySyntaxError>(py)
                            {
                                continue;
                            }
                            return Err(error);
                        }
                    };
                    for file in files.try_iter()? {
                        let file = file?;
                        let file_name: String = file.get_item(0)?.extract()?;
                        let name: String = file.get_item(1)?.extract()?;
                        let file_source: String = file.get_item(2)?.extract()?;
                        if !found.contains_key(&name) {
                            found.insert(
                                name.clone(),
                                (
                                    format!("{}#{}", path.to_string_lossy(), file_name),
                                    file_source,
                                ),
                            );
                            order.push(name);
                        }
                    }
                }

                let mut sources: Vec<PathBuf> = Vec::new();
                collect_matching(&root_dir, ".viba", &mut sources);
                sources.sort_by_key(|path| path_key(path));
                for path in sources {
                    let relative = match path.strip_prefix(&root_dir) {
                        Ok(relative) => relative,
                        Err(_) => continue,
                    };
                    let module_name = module_name_of(&relative.to_string_lossy());
                    if module_name.is_empty() || found.contains_key(&module_name) {
                        continue;
                    }
                    if written_as_document(&path) {
                        continue;
                    }
                    let source = match std::fs::read_to_string(&path) {
                        Ok(source) => source,
                        Err(_) => continue,
                    };
                    found.insert(
                        module_name.clone(),
                        (path.to_string_lossy().into_owned(), source),
                    );
                    order.push(module_name);
                }
            }
        }
    }

    let mut out = Vec::new();
    for module_name in order {
        if let Some((file_name, source)) = found.remove(&module_name) {
            out.push((file_name, module_name, source));
        }
    }
    Ok(out)
}

// ----------------------------------------------------------------------
// The walk
// ----------------------------------------------------------------------

struct Checker<'py> {
    py: Python<'py>,
    type_module: Bound<'py, PyModule>,
    descriptor_module: Bound<'py, PyModule>,
    pattern_module: Bound<'py, PyModule>,
    viba_ast: Bound<'py, PyModule>,
    design_walk: Bound<'py, PyModule>,
    partial_module: Bound<'py, PyModule>,
    ok_class: Bound<'py, PyAny>,
    err_class: Bound<'py, PyAny>,
    ast_node_type_class: Bound<'py, PyAny>,
    pool: Obj,
    stops: HashSet<String>,
    files: HashMap<String, Obj>,
    definitions: HashMap<String, Rc<Vec<(String, Obj)>>>,
    visiting: HashSet<String>,
}

impl<'py> Checker<'py> {
    fn new(py: Python<'py>, pool: Obj, stops: HashSet<String>) -> PyResult<Self> {
        let type_module = PyModule::import(py, "rs_viba.type")?;
        let descriptor_module = PyModule::import(py, "rs_viba.viba_type_descriptor")?;
        let pattern_module = PyModule::import(py, "rs_viba.pattern")?;
        let viba_ast = PyModule::import(py, "rs_viba.viba_ast")?;
        let design_walk = PyModule::import(py, "rs_viba.design_walk")?;
        let partial_module = PyModule::import(py, "rs_viba.partial")?;
        let ok_class = type_module.getattr("Ok")?;
        let err_class = type_module.getattr("Err")?;
        let ast_node_type_class = type_module.getattr("AstNodeType")?;
        let mut files = HashMap::new();
        for file in pool.bind(py).getattr("files")?.try_iter()? {
            let file = file?;
            let module_name: String = file.getattr("module_name")?.extract()?;
            files.insert(module_name, file.unbind());
        }
        Ok(Self {
            py,
            type_module,
            descriptor_module,
            pattern_module,
            viba_ast,
            design_walk,
            partial_module,
            ok_class,
            err_class,
            ast_node_type_class,
            pool,
            stops,
            files,
            definitions: HashMap::new(),
            visiting: HashSet::new(),
        })
    }

    fn complete(&mut self) -> PyResult<bool> {
        let definitions = self.definitions_of(ENTRY_MODULE_NAME)?;
        for (_name, definition) in definitions.iter() {
            let bindings = Rc::new(Bindings::default());
            if !self.definition(definition.bind(self.py), ENTRY_MODULE_NAME, Some(&bindings))? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    // ---- resolution ----

    fn definitions_of(&mut self, module_name: &str) -> PyResult<Rc<Vec<(String, Obj)>>> {
        if let Some(cached) = self.definitions.get(module_name) {
            return Ok(cached.clone());
        }
        let taken = self.take_definitions(module_name)?;
        let taken = Rc::new(taken);
        self.definitions
            .insert(module_name.to_string(), taken.clone());
        Ok(taken)
    }

    fn take_definitions(&mut self, module_name: &str) -> PyResult<Vec<(String, Obj)>> {
        if module_name == BUILTIN_MODULE_NAME {
            return Ok(Vec::new());
        }
        let module = self
            .pool
            .bind(self.py)
            .call_method1("module_environment", (module_name,))?;
        if self.is_err(&module)? {
            return Ok(Vec::new());
        }
        let body = module
            .getattr("ok_value")?
            .getattr("module")?
            .getattr("body")?;
        let mut out = Vec::new();
        for statement in body.try_iter()? {
            let statement = statement?;
            if is_definition_node(&statement)? {
                let name: String = statement.getattr("name")?.extract()?;
                out.push((name, statement.unbind()));
            }
        }
        Ok(out)
    }

    fn definition_named(&mut self, module_name: &str, name: &str) -> PyResult<Option<Obj>> {
        let definitions = self.definitions_of(module_name)?;
        for (candidate, node) in definitions.iter() {
            if candidate == name {
                return Ok(Some(node.clone_ref(self.py)));
            }
        }
        Ok(None)
    }

    fn resolve(&mut self, name: &str, module_name: &str) -> PyResult<Option<(Obj, String)>> {
        let parts: Vec<&str> = name.split('.').collect();
        let file = self
            .files
            .get(module_name)
            .map(|file| file.clone_ref(self.py));
        if let Some(file) = file {
            let file_handle = file.bind(self.py);
            for cut in (1..parts.len()).rev() {
                let local = parts[..cut].join(".");
                let imported = self
                    .descriptor_module
                    .getattr("file_find_import_by_local_name")?
                    .call1((&file_handle, local))?;
                if self.is_err(&imported)? {
                    continue;
                }
                let import = imported.getattr("ok_value")?;
                let mut target: String = import.getattr("module_name")?.extract()?;
                let rest = &parts[cut..];
                if rest.len() > 1 {
                    target = format!("{}.{}", target, rest[..rest.len() - 1].join("."));
                }
                let last = rest[rest.len() - 1];
                if let Some(node) = self.definition_named(&target, last)? {
                    return Ok(Some((node, target)));
                }
                break;
            }
        }
        if let Some(node) = self.definition_named(module_name, name)? {
            return Ok(Some((node, module_name.to_string())));
        }
        if let Some(value) = self.builtin_lookup_value(name)? {
            if value.is_instance(&self.ast_node_type_class)? {
                let ast_node = value.getattr("ast_node")?.unbind();
                return Ok(Some((ast_node, BUILTIN_MODULE_NAME.to_string())));
            }
        }
        if let Some(piece) = self.builtin_member_piece(name)? {
            return Ok(Some((piece, BUILTIN_MODULE_NAME.to_string())));
        }
        Ok(None)
    }

    /// What the builtin vocabulary says this name is, if anything.
    ///
    /// The layer this asks answers either a `Result` (the spec's shape) or the
    /// value itself (the shape the current binding hands out); both are taken.
    fn builtin_lookup_value(&self, name: &str) -> PyResult<Option<Bound<'py, PyAny>>> {
        let found = self
            .type_module
            .getattr("BUILTIN_MODULE")?
            .call_method1("lookup", (name,))?;
        if self.is_err(&found)? {
            return Ok(None);
        }
        if found.is_instance(&self.ok_class)? {
            let value = found.getattr("ok_value")?;
            if value.is_none() {
                return Ok(None);
            }
            return Ok(Some(value));
        }
        Ok(Some(found))
    }

    fn builtin_member_piece(&self, name: &str) -> PyResult<Option<Obj>> {
        let (head, tag) = match name.rfind('.') {
            Some(index) => (&name[..index], &name[index + 1..]),
            None => return Ok(None),
        };
        let value = match self.builtin_lookup_value(head)? {
            Some(value) => value,
            None => return Ok(None),
        };
        if !value.is_instance(&self.ast_node_type_class)? {
            return Ok(None);
        }
        let ast_node = value.getattr("ast_node")?;
        let body = if is_class(&ast_node, "TypeDefinition")? {
            ast_node.getattr("body")?
        } else {
            ast_node.clone()
        };
        let factors = self
            .partial_module
            .getattr("product_elements")?
            .call1((&body,))?;
        for factor in factors.try_iter()? {
            let factor = factor?;
            if is_class(&factor, "Tagged")? {
                let factor_tag: String = factor.getattr("tag")?.extract()?;
                if factor_tag == format!("${}", tag) {
                    return Ok(Some(factor.getattr("type")?.unbind()));
                }
            }
        }
        Ok(None)
    }

    // ---- the walk itself ----

    fn walk(
        &mut self,
        node: &Bound<'py, PyAny>,
        module_name: &str,
        bindings: &Rc<Bindings>,
    ) -> PyResult<bool> {
        let class = class_name(node)?;
        match class.as_str() {
            "Constant" | "Nil" | "Never" => Ok(true),
            "Ellipsis" => Ok(self.stops.contains("...")),
            "HintBlock" => Ok(false),
            "TypeRef" => {
                let name: String = node.getattr("name")?.extract()?;
                self.name(&name, module_name, bindings)
            }
            "MemberTaken" => self.member(node, module_name, bindings),
            "TypeApp" => {
                if let Some(answer) = self.tagged(node, module_name, bindings)? {
                    return Ok(answer);
                }
                self.application(node, module_name, bindings)
            }
            "Tagged" => {
                let inner = node.getattr("type")?;
                self.walk(&inner, module_name, bindings)
            }
            "Tuple" => {
                for element in field_list(node, "elements")? {
                    if !self.walk(&element, module_name, bindings)? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            "Partial" => self.call(node, module_name, bindings),
            "Sum" | "SumChain" => self.sum(node, module_name, bindings),
            _ => {
                let mut elements = match self.chain_elements(node)? {
                    Some(elements) => elements,
                    None => return Ok(false),
                };
                if let Some(first) = elements.first() {
                    let first = first.bind(self.py);
                    if is_unit_head(&first)? {
                        elements.remove(0);
                    }
                }
                for element in &elements {
                    if !self.walk(element.bind(self.py), module_name, bindings)? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
        }
    }

    fn sum(
        &mut self,
        node: &Bound<'py, PyAny>,
        module_name: &str,
        bindings: &Rc<Bindings>,
    ) -> PyResult<bool> {
        let mut seen: HashSet<usize> = HashSet::new();
        let options = self.options(node, module_name, bindings, &mut seen)?;
        if !told_apart(&options) {
            return Ok(false);
        }
        for element in self.chain_elements(node)?.unwrap_or_default() {
            if !self.walk(element.bind(self.py), module_name, bindings)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn options(
        &mut self,
        node: &Bound<'py, PyAny>,
        module_name: &str,
        bindings: &Rc<Bindings>,
        seen: &mut HashSet<usize>,
    ) -> PyResult<Vec<OptionItem>> {
        if !seen.insert(node.as_ptr() as usize) {
            return Ok(Vec::new());
        }
        let mut out = Vec::new();
        for element in self.chain_elements(node)?.unwrap_or_default() {
            let form = self.stands_for(element.bind(self.py), module_name, bindings)?;
            let form_node = form.node.clone_ref(self.py);
            {
                let form_bound = form_node.bind(self.py);
                if is_unit_head(&form_bound)? {
                    continue;
                }
                let class = class_name(&form_bound)?;
                if class == "Sum" || class == "SumChain" {
                    let nested = self.options(&form_bound, &form.module, &form.bindings, seen)?;
                    out.extend(nested);
                    continue;
                }
                let tag = option_tag(&form_bound)?;
                let name = option_name(&form_bound)?;
                out.push(OptionItem {
                    tag,
                    name,
                    known: form.known,
                });
            }
        }
        Ok(out)
    }

    fn stands_for(
        &mut self,
        node: &Bound<'py, PyAny>,
        module_name: &str,
        bindings: &Rc<Bindings>,
    ) -> PyResult<FormRef> {
        let mut current = node.clone().unbind();
        let mut module = module_name.to_string();
        let mut binds = bindings.clone();
        let mut seen: HashSet<usize> = HashSet::new();
        loop {
            if !seen.insert(current.as_ptr() as usize) {
                return Ok(FormRef {
                    node: current,
                    known: false,
                    module,
                    bindings: binds,
                });
            }
            let bound = current.bind(self.py).clone();
            let class = class_name(&bound)?;
            match class.as_str() {
                "TypeRef" => {
                    let name: String = bound.getattr("name")?.extract()?;
                    if is_named_form(&name) || self.stops.contains(&name) {
                        return Ok(FormRef {
                            node: current.clone_ref(self.py),
                            known: true,
                            module,
                            bindings: binds,
                        });
                    }
                    let binding = binds.map.get(&name).cloned();
                    match binding {
                        Some(BindingValue::Bound(binding)) => {
                            current = binding.node.clone_ref(self.py);
                            module = binding.module.clone();
                            binds = binding.bindings.clone();
                            continue;
                        }
                        Some(BindingValue::Placeholder) => {
                            return Ok(FormRef {
                                node: current.clone_ref(self.py),
                                known: false,
                                module,
                                bindings: binds,
                            });
                        }
                        None => {}
                    }
                    match self.resolve(&name, &module)? {
                        None => {
                            return Ok(FormRef {
                                node: current.clone_ref(self.py),
                                known: false,
                                module,
                                bindings: binds,
                            });
                        }
                        Some((definition, target)) => {
                            let next = getattr_default(
                                definition.bind(self.py),
                                "body",
                                definition.bind(self.py),
                            )?;
                            let next = if is_class(&next, "TypeDefinition")? {
                                next.getattr("body")?
                            } else {
                                next
                            };
                            current = next.unbind();
                            module = target;
                            continue;
                        }
                    }
                }
                "TypeApp" => {
                    let constructor: String = bound.getattr("constructor")?.extract()?;
                    if is_named_form(&constructor) || self.stops.contains(&constructor) {
                        return Ok(FormRef {
                            node: current.clone_ref(self.py),
                            known: true,
                            module,
                            bindings: binds,
                        });
                    }
                    match self.decision(&bound, &module, &binds)? {
                        Decision::Body(body, target, inner) => {
                            current = body;
                            module = target;
                            binds = inner;
                            continue;
                        }
                        _ => {}
                    }
                    match self.generic_body(&bound, &module, &binds)? {
                        None => {
                            return Ok(FormRef {
                                node: current.clone_ref(self.py),
                                known: false,
                                module,
                                bindings: binds,
                            });
                        }
                        Some((definition, target, inner)) => {
                            let next = getattr_default(
                                definition.bind(self.py),
                                "body",
                                definition.bind(self.py),
                            )?;
                            current = next.unbind();
                            module = target;
                            binds = inner;
                            continue;
                        }
                    }
                }
                "MemberTaken" => match self.member_definition(&bound, &module, &binds)? {
                    None => {
                        return Ok(FormRef {
                            node: current.clone_ref(self.py),
                            known: false,
                            module,
                            bindings: binds,
                        });
                    }
                    Some(found) => {
                        let next = getattr_default(
                            found.definition.bind(self.py),
                            "body",
                            found.definition.bind(self.py),
                        )?;
                        current = next.unbind();
                        module = found.module;
                        binds = found.bindings;
                        continue;
                    }
                },
                _ => {
                    return Ok(FormRef {
                        node: current.clone_ref(self.py),
                        known: true,
                        module,
                        bindings: binds,
                    });
                }
            }
        }
    }

    fn call(
        &mut self,
        node: &Bound<'py, PyAny>,
        module_name: &str,
        bindings: &Rc<Bindings>,
    ) -> PyResult<bool> {
        let (head, arguments) = self.call_parts(node)?;
        let head_bound = head.bind(self.py);
        if matches!(
            class_name(&head_bound)?.as_str(),
            "Constant" | "Nil" | "Never"
        ) {
            return Ok(false);
        }
        if !self.walk(&head_bound, module_name, bindings)? {
            return Ok(false);
        }
        for argument in arguments {
            let argument = argument.bind(self.py);
            if is_class(&argument, "HintBlock")? {
                continue;
            }
            let piece = if is_class(&argument, "Tagged")? {
                argument.getattr("type")?
            } else {
                argument.clone()
            };
            if !self.walk(&piece, module_name, bindings)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn member(
        &mut self,
        node: &Bound<'py, PyAny>,
        module_name: &str,
        bindings: &Rc<Bindings>,
    ) -> PyResult<bool> {
        match self.member_definition(node, module_name, bindings)? {
            None => Ok(false),
            Some(found) => {
                let definition = found.definition.bind(self.py);
                self.definition(&definition, &found.module, Some(&found.bindings))
            }
        }
    }

    fn member_definition(
        &mut self,
        node: &Bound<'py, PyAny>,
        module_name: &str,
        bindings: &Rc<Bindings>,
    ) -> PyResult<Option<FoundDefinition>> {
        let owner = node.getattr("owner")?;
        if !is_class(&owner, "TypeApp")? {
            return Ok(None);
        }
        let module = match self.module_environment(module_name)? {
            Some(module) => module,
            None => return Ok(None),
        };
        let argument_modules = self.argument_modules(&owner, module_name, bindings)?;
        let decision = self.pattern_module.getattr("reduce_application")?.call1((
            &owner,
            module,
            argument_modules,
        ))?;
        if self.is_err(&decision)? {
            return Ok(None);
        }
        let chosen = decision.getattr("ok_value")?;
        if chosen.is_none() {
            return Ok(None);
        }
        let inner = self.bindings_of_choice(&chosen, module_name, bindings)?;
        let chosen_entry = chosen.getattr("entry")?;
        let chosen_module: String = chosen_entry.getattr("name")?.extract()?;
        let name: String = node.getattr("name")?.extract()?;
        let body = chosen_entry
            .getattr("module")
            .ok()
            .and_then(|module| module.getattr("module").ok())
            .and_then(|module| module.getattr("body").ok());
        let body = match body {
            Some(body) => body,
            None => return Ok(None),
        };
        for statement in body.try_iter()? {
            let statement = statement?;
            if is_definition_node(&statement)? {
                let statement_name: String = statement.getattr("name")?.extract()?;
                if statement_name == name {
                    return Ok(Some(FoundDefinition {
                        definition: statement.unbind(),
                        module: chosen_module,
                        bindings: inner,
                    }));
                }
            }
        }
        Ok(None)
    }

    fn bindings_of_choice(
        &mut self,
        chosen: &Bound<'py, PyAny>,
        module_name: &str,
        bindings: &Rc<Bindings>,
    ) -> PyResult<Rc<Bindings>> {
        let mut inner = Bindings::copy_of(bindings);
        let chosen_bindings = chosen.getattr("bindings")?.cast_into::<PyDict>()?;
        for (name, value) in chosen_bindings.iter() {
            let name: String = name.extract()?;
            let node = value.getattr("ast_node")?.unbind();
            inner.map.insert(
                name,
                BindingValue::Bound(Rc::new(Binding {
                    node,
                    module: module_name.to_string(),
                    bindings: bindings.clone(),
                })),
            );
        }
        Ok(inner.with())
    }

    fn tagged(
        &mut self,
        node: &Bound<'py, PyAny>,
        module_name: &str,
        bindings: &Rc<Bindings>,
    ) -> PyResult<Option<bool>> {
        let module = match self.module_environment(module_name)? {
            Some(module) => module,
            None => return Ok(None),
        };
        let args = field_list(node, "args")?;
        let mut symbol_text: Option<String> = None;
        if let Some(first) = args.first() {
            if is_class(first, "TypeRef")? {
                let name: String = first.getattr("name")?.extract()?;
                symbol_text = self.symbol_text(&name, module_name, bindings)?;
            }
        }
        let resolver = make_resolver(self.py, symbol_text)?;
        let got = self
            .pattern_module
            .getattr("tagged_type_of")?
            .call1((node, module, resolver))?;
        if self.is_err(&got)? {
            return Ok(Some(false));
        }
        let value = got.getattr("ok_value")?;
        if !value.is_none() {
            return Ok(Some(self.walk(&value, module_name, bindings)?));
        }
        if self.unbound_symbol(node, bindings)? {
            if args.len() == 2 {
                return Ok(Some(self.walk(&args[1], module_name, bindings)?));
            }
            return Ok(Some(true));
        }
        Ok(None)
    }

    fn symbol_text(
        &mut self,
        name: &str,
        module_name: &str,
        bindings: &Rc<Bindings>,
    ) -> PyResult<Option<String>> {
        match bindings.map.get(name).cloned() {
            Some(BindingValue::Placeholder) => Ok(None),
            Some(BindingValue::Bound(binding)) => {
                let source_form = binding.node.bind(self.py);
                constant_string(&source_form)
            }
            None => {
                let found = match self.resolve(name, module_name)? {
                    Some(found) => found,
                    None => return Ok(None),
                };
                let node = found.0.bind(self.py);
                if is_class(&node, "TypeDefinition")? {
                    let body = node.getattr("body")?;
                    constant_string(&body)
                } else {
                    constant_string(&node)
                }
            }
        }
    }

    fn name(&mut self, name: &str, module_name: &str, bindings: &Rc<Bindings>) -> PyResult<bool> {
        match bindings.map.get(name).cloned() {
            Some(BindingValue::Placeholder) => return Ok(true),
            Some(BindingValue::Bound(binding)) => {
                let node = binding.node.bind(self.py);
                return self.walk(&node, &binding.module, &binding.bindings);
            }
            None => {}
        }
        if self.stops.contains(name) || is_named_form(name) {
            return Ok(true);
        }
        match self.resolve(name, module_name)? {
            None => Ok(false),
            Some((node, target)) => {
                let node = node.bind(self.py);
                self.definition(&node, &target, None)
            }
        }
    }

    fn application(
        &mut self,
        node: &Bound<'py, PyAny>,
        module_name: &str,
        bindings: &Rc<Bindings>,
    ) -> PyResult<bool> {
        let constructor: String = node.getattr("constructor")?.extract()?;
        if self.stops.contains(&constructor) {
            return Ok(true);
        }
        if CONTAINER_NAMES.contains(&constructor.as_str()) {
            for argument in field_list(node, "args")? {
                if !self.walk(&argument, module_name, bindings)? {
                    return Ok(false);
                }
            }
            return Ok(true);
        }
        if LEAF_NAMES.contains(&constructor.as_str()) || UNIT_NAMES.contains(&constructor.as_str())
        {
            return Ok(true);
        }
        if let Some(answer) = self.decide(node, module_name, bindings)? {
            return Ok(answer);
        }
        match self.generic_body(node, module_name, bindings)? {
            None => Ok(false),
            Some((definition, target, inner)) => {
                let definition = definition.bind(self.py);
                self.definition(&definition, &target, Some(&inner))
            }
        }
    }

    fn generic_body(
        &mut self,
        node: &Bound<'py, PyAny>,
        module_name: &str,
        bindings: &Rc<Bindings>,
    ) -> PyResult<Option<(Obj, String, Rc<Bindings>)>> {
        let caller_module = module_name;
        let constructor: String = node.getattr("constructor")?.extract()?;
        let found = match self.resolve(&constructor, caller_module)? {
            Some(found) => found,
            None => return Ok(None),
        };
        let (definition, target_module) = found;
        if !is_class(&definition.bind(self.py), "GenericDefinition")? {
            return Ok(None);
        }
        let params = optional_field_list(&definition.bind(self.py), "generic_params")?;
        let args = field_list(node, "args")?;
        if params.len() != args.len() {
            return Ok(None);
        }
        let mut inner = Bindings::copy_of(bindings);
        for (param, arg) in params.iter().zip(args.iter()) {
            let param: String = param.extract()?;
            inner.map.insert(
                param,
                BindingValue::Bound(Rc::new(Binding {
                    node: arg.clone().unbind(),
                    module: caller_module.to_string(),
                    bindings: bindings.clone(),
                })),
            );
        }
        Ok(Some((definition, target_module, inner.with())))
    }

    fn decide(
        &mut self,
        node: &Bound<'py, PyAny>,
        module_name: &str,
        bindings: &Rc<Bindings>,
    ) -> PyResult<Option<bool>> {
        match self.decision(node, module_name, bindings)? {
            Decision::Failed => Ok(Some(false)),
            Decision::NoGeneric => Ok(None),
            Decision::Body(body, target, inner) => {
                Ok(Some(self.walk(&body.bind(self.py), &target, &inner)?))
            }
        }
    }

    fn decision(
        &mut self,
        node: &Bound<'py, PyAny>,
        module_name: &str,
        bindings: &Rc<Bindings>,
    ) -> PyResult<Decision> {
        let module = match self.module_environment(module_name)? {
            Some(module) => module,
            None => return Ok(Decision::NoGeneric),
        };
        let argument_modules = self.argument_modules(node, module_name, bindings)?;
        let decision = self.pattern_module.getattr("reduce_application")?.call1((
            node,
            module,
            argument_modules,
        ))?;
        if self.is_err(&decision)? {
            return Ok(Decision::Failed);
        }
        let chosen = decision.getattr("ok_value")?;
        if chosen.is_none() {
            return Ok(Decision::NoGeneric);
        }
        let inner = self.bindings_of_choice(&chosen, module_name, bindings)?;
        let body = chosen.getattr("body")?.unbind();
        let entry: String = chosen.getattr("entry")?.getattr("name")?.extract()?;
        Ok(Decision::Body(body, entry, inner))
    }

    fn argument_modules(
        &mut self,
        node: &Bound<'py, PyAny>,
        module_name: &str,
        bindings: &Rc<Bindings>,
    ) -> PyResult<Obj> {
        let _ = module_name;
        let mut kept: Vec<Obj> = Vec::new();
        let mut any = false;
        for argument in field_list(node, "args")? {
            let mut where_: Option<Obj> = None;
            if is_class(&argument, "TypeRef")? {
                let name: String = argument.getattr("name")?.extract()?;
                if let Some(BindingValue::Bound(binding)) = bindings.map.get(&name).cloned() {
                    if let Some(module) = self.module_environment(&binding.module)? {
                        where_ = Some(module);
                        any = true;
                    }
                }
            }
            kept.push(where_.unwrap_or_else(|| self.py.None()));
        }
        if !any {
            return Ok(self.py.None());
        }
        crate::pyast::pylist(self.py, kept)
    }

    fn module_environment(&mut self, module_name: &str) -> PyResult<Option<Obj>> {
        let module = self
            .pool
            .bind(self.py)
            .call_method1("module_environment", (module_name,))?;
        if self.is_err(&module)? {
            return Ok(None);
        }
        Ok(Some(module.getattr("ok_value")?.unbind()))
    }

    fn definition(
        &mut self,
        node: &Bound<'py, PyAny>,
        module_name: &str,
        bindings: Option<&Rc<Bindings>>,
    ) -> PyResult<bool> {
        let mut inner = match bindings {
            Some(bindings) => Bindings::copy_of(bindings),
            None => Bindings::default(),
        };
        for param in optional_field_list(node, "generic_params")? {
            let param: String = param.extract()?;
            inner.map.entry(param).or_insert(BindingValue::Placeholder);
        }
        let inner = inner.with();
        let name: Option<String> = getattr_opt_string(node, "name")?;
        let key = format!(
            "{}:{}|{}:{}|{}",
            module_name.len(),
            module_name,
            name.as_ref().map(|value| value.len()).unwrap_or(0),
            name.clone().unwrap_or_else(|| "None".to_string()),
            self.bindings_key(&inner)?
        );
        if self.visiting.contains(&key) {
            return Ok(true);
        }
        self.visiting.insert(key.clone());
        let body = getattr_default(node, "body", node)?;
        let result = self.walk(&body, module_name, &inner);
        self.visiting.remove(&key);
        result
    }

    fn bindings_key(&self, bindings: &Rc<Bindings>) -> PyResult<String> {
        let mut parts: Vec<(String, String)> = Vec::new();
        for (name, value) in bindings.map.iter() {
            let text = match value {
                BindingValue::Placeholder => "placeholder".to_string(),
                BindingValue::Bound(binding) => {
                    let dump: String = self
                        .viba_ast
                        .getattr("dump")?
                        .call1((binding.node.bind(self.py),))?
                        .extract()?;
                    format!("{}:{}:{}", binding.module.len(), binding.module, dump)
                }
            };
            parts.push((name.clone(), text));
        }
        parts.sort();
        let mut out = String::new();
        for (name, text) in parts {
            out.push_str(&format!("{}:{};{}:{}|", name.len(), name, text.len(), text));
        }
        Ok(out)
    }

    fn unbound_symbol(
        &mut self,
        node: &Bound<'py, PyAny>,
        bindings: &Rc<Bindings>,
    ) -> PyResult<bool> {
        if !(is_class(node, "TypeApp")?
            && node.getattr("constructor")?.extract::<String>()? == TAGGED_NAME)
        {
            return Ok(false);
        }
        let args = field_list(node, "args")?;
        let mut symbol = match args.first() {
            Some(first) => first.clone().unbind(),
            None => return Ok(false),
        };
        let mut env = bindings.clone();
        let mut seen: HashSet<usize> = HashSet::new();
        loop {
            let symbol_bound = symbol.bind(self.py);
            if !is_class(&symbol_bound, "TypeRef")? {
                return Ok(false);
            }
            let name: String = symbol_bound.getattr("name")?.extract()?;
            if !env.map.contains_key(&name) {
                return Ok(false);
            }
            if !seen.insert(Rc::as_ptr(&env) as usize) {
                return Ok(false);
            }
            match env.map.get(&name).cloned() {
                Some(BindingValue::Placeholder) | None => return Ok(true),
                Some(BindingValue::Bound(binding)) => {
                    symbol = binding.node.clone_ref(self.py);
                    env = binding.bindings.clone();
                }
            }
        }
    }

    // ---- small helpers ----

    fn is_err(&self, value: &Bound<'py, PyAny>) -> PyResult<bool> {
        value.is_instance(&self.err_class)
    }

    fn chain_elements(&self, node: &Bound<'py, PyAny>) -> PyResult<Option<Vec<Obj>>> {
        let elements = self.design_walk.getattr("chain_elements")?.call1((node,))?;
        if elements.is_none() {
            return Ok(None);
        }
        let mut out = Vec::new();
        for element in elements.try_iter()? {
            out.push(element?.unbind());
        }
        Ok(Some(out))
    }

    fn call_parts(&self, node: &Bound<'py, PyAny>) -> PyResult<(Obj, Vec<Obj>)> {
        let parts = self.design_walk.getattr("call_parts")?.call1((node,))?;
        let head = parts.get_item(0)?.unbind();
        let mut arguments = Vec::new();
        for argument in parts.get_item(1)?.try_iter()? {
            arguments.push(argument?.unbind());
        }
        Ok((head, arguments))
    }
}

// ----------------------------------------------------------------------
// Small helpers
// ----------------------------------------------------------------------

fn class_name(node: &Bound<'_, PyAny>) -> PyResult<String> {
    pyast::class_name(node)
}

fn is_class(node: &Bound<'_, PyAny>, name: &str) -> PyResult<bool> {
    Ok(class_name(node)? == name)
}

fn is_definition_node(node: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(matches!(
        class_name(node)?.as_str(),
        "TypeDefinition" | "GenericDefinition"
    ))
}

fn field_list<'py>(node: &Bound<'py, PyAny>, name: &str) -> PyResult<Vec<Bound<'py, PyAny>>> {
    pyast::field_list(node, name)
}

fn optional_field_list<'py>(
    node: &Bound<'py, PyAny>,
    name: &str,
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    match node.getattr(name) {
        Ok(value) if !value.is_none() => {
            let mut out = Vec::new();
            for item in value.try_iter()? {
                out.push(item?);
            }
            Ok(out)
        }
        Ok(_) => Ok(Vec::new()),
        Err(error) if error.is_instance_of::<pyo3::exceptions::PyAttributeError>(node.py()) => {
            Ok(Vec::new())
        }
        Err(error) => Err(error),
    }
}

fn getattr_default<'py>(
    node: &Bound<'py, PyAny>,
    name: &str,
    default: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    match node.getattr(name) {
        Ok(value) => Ok(value),
        Err(error) if error.is_instance_of::<pyo3::exceptions::PyAttributeError>(node.py()) => {
            Ok(default.clone())
        }
        Err(error) => Err(error),
    }
}

fn getattr_opt_string(node: &Bound<'_, PyAny>, name: &str) -> PyResult<Option<String>> {
    match node.getattr(name) {
        Ok(value) if !value.is_none() => Ok(Some(value.extract()?)),
        Ok(_) => Ok(None),
        Err(error) if error.is_instance_of::<pyo3::exceptions::PyAttributeError>(node.py()) => {
            Ok(None)
        }
        Err(error) => Err(error),
    }
}

fn constant_string(node: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
    if is_class(node, "Constant")? {
        let value = node.getattr("value")?;
        if value.is_instance_of::<PyString>() {
            return Ok(Some(value.extract()?));
        }
    }
    Ok(None)
}

fn is_unit_head(node: &Bound<'_, PyAny>) -> PyResult<bool> {
    let class = class_name(node)?;
    if class == "Nil" || class == "Never" {
        return Ok(true);
    }
    if class == "TypeRef" {
        let name: String = node.getattr("name")?.extract()?;
        return Ok(UNIT_NAMES.contains(&name.as_str()));
    }
    Ok(false)
}

fn is_named_form(name: &str) -> bool {
    LEAF_NAMES.contains(&name) || CONTAINER_NAMES.contains(&name) || UNIT_NAMES.contains(&name)
}

fn option_tag(form: &Bound<'_, PyAny>) -> PyResult<TagKind> {
    let class = class_name(form)?;
    if class == "Tagged" {
        return Ok(TagKind::Text(form.getattr("tag")?.extract()?));
    }
    if class == "TypeApp" {
        let constructor: String = form.getattr("constructor")?.extract()?;
        if constructor == TAGGED_NAME {
            return Ok(TagKind::Unknown);
        }
    }
    Ok(TagKind::None)
}

fn option_name(form: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
    let inner = if is_class(form, "Tagged")? {
        form.getattr("type")?
    } else {
        form.clone()
    };
    let class = class_name(&inner)?;
    if class == "Constant" {
        let value = inner.getattr("value")?;
        if value.is_instance_of::<PyBool>() {
            let truth: bool = value.extract()?;
            return Ok(Some(format!("literal\u{0}bool\u{0}{}", truth)));
        }
        if value.is_instance_of::<PyInt>() {
            let text: String = value.str()?.extract()?;
            return Ok(Some(format!("literal\u{0}int\u{0}{}", text)));
        }
        if value.is_instance_of::<PyFloat>() {
            let text: String = value.repr()?.extract()?;
            return Ok(Some(format!("literal\u{0}float\u{0}{}", text)));
        }
        if value.is_instance_of::<PyString>() {
            let text: String = value.extract()?;
            return Ok(Some(format!("literal\u{0}str\u{0}{}", text)));
        }
        return Ok(None);
    }
    if class == "TypeRef" {
        let name: String = inner.getattr("name")?.extract()?;
        return Ok(Some(format!("type\u{0}{}", name)));
    }
    if class == "TypeApp" {
        let constructor: String = inner.getattr("constructor")?.extract()?;
        return Ok(Some(format!("type\u{0}{}", constructor)));
    }
    Ok(None)
}

fn told_apart(options: &[OptionItem]) -> bool {
    let mut tags: HashSet<String> = HashSet::new();
    for option in options {
        if let TagKind::Text(tag) = &option.tag {
            if !tags.insert(tag.clone()) {
                return false;
            }
        }
    }
    let mut names: HashMap<&str, usize> = HashMap::new();
    for option in options {
        if let Some(name) = &option.name {
            *names.entry(name.as_str()).or_insert(0) += 1;
        }
    }
    for option in options {
        if !matches!(option.tag, TagKind::None) {
            continue;
        }
        if !option.known {
            return false;
        }
        if let Some(name) = &option.name {
            if names.get(name.as_str()).copied().unwrap_or(0) > 1 {
                return false;
            }
        }
    }
    true
}

fn make_resolver<'py>(py: Python<'py>, text: Option<String>) -> PyResult<Bound<'py, PyCFunction>> {
    PyCFunction::new_closure(py, None, None, move |_args, _kwargs| {
        Python::attach(|py| match &text {
            Some(text) => Ok::<Obj, PyErr>(PyString::new(py, text).into_any().unbind()),
            None => Ok::<Obj, PyErr>(py.None()),
        })
    })
}

// ----------------------------------------------------------------------
// Entry points
// ----------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (entry_file_content, library = None, viba_paths = None, terminators = None))]
fn is_complete_py(
    py: Python<'_>,
    entry_file_content: &str,
    library: Option<&Bound<'_, PyAny>>,
    viba_paths: Option<&Bound<'_, PyAny>>,
    terminators: Option<&Bound<'_, PyAny>>,
) -> PyResult<bool> {
    let descriptor = PyModule::import(py, "rs_viba.viba_type_descriptor")?;
    let type_module = PyModule::import(py, "rs_viba.type")?;
    let err_class = type_module.getattr("Err")?;
    let initial_pool = descriptor.getattr("empty_pool")?.call0()?;
    let mut pool = match add_file(
        &descriptor,
        &err_class,
        &initial_pool,
        ENTRY_FILE_NAME,
        ENTRY_MODULE_NAME,
        entry_file_content,
    )? {
        Some(pool) => pool,
        None => return Ok(false),
    };
    for (file_name, module_name, source) in dependencies(py, library, viba_paths)? {
        if let Some(added) = add_file(
            &descriptor,
            &err_class,
            pool.bind(py),
            &file_name,
            &module_name,
            &source,
        )? {
            pool = added;
        }
    }
    let mut stops: HashSet<String> = HashSet::new();
    if let Some(terminators) = terminators {
        if !terminators.is_none() {
            for name in terminators.try_iter()? {
                stops.insert(name?.extract()?);
            }
        }
    }
    let mut checker = Checker::new(py, pool, stops)?;
    checker.complete()
}

/// The chain helper, exposed for the checks that read a binary chain directly.
#[pyfunction]
fn is_complete_chain_elements<'py>(
    py: Python<'py>,
    node: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    let design_walk = PyModule::import(py, "rs_viba.design_walk")?;
    Ok(design_walk
        .getattr("chain_elements")?
        .call1((node,))?
        .unbind())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(is_complete_py, m)?)?;
    m.add_function(wrap_pyfunction!(is_complete_chain_elements, m)?)?;
    Ok(())
}
