//! extract_sub_repo: a sub-repo as one Markdown document, with line maps (was
//! viba/extract_sub_repo.py).
//!
//! Two entries. `extract_sub_repo_into_viba_markdown` takes a repo — its
//! `.viba` files, and the `.viba.md` documents inside it — and answers the
//! sub-repo as one document (one section per file, its path the section's name)
//! plus a map from every code line to the module and the line it came from.
//! `extract_sub_viba_markdown` takes a document and answers a document, mapping
//! to the source document's lines.
//!
//! What the walk reads is the source, never a run: the modules come from the
//! interpreter's own loader (`rs_viba.interpret._Runner`, asked the way an
//! import asks it), and the readings of the syntax come from the layers that
//! already own them (`rs_viba.type`, `rs_viba.pattern`, `rs_viba.design_walk`,
//! `rs_viba.viba_ast`, `rs_viba.markdown`). This module is the one layer that
//! decides what a sub-repo keeps.

use std::collections::{HashMap, HashSet};
use std::path::{Component, Path, PathBuf};

use pyo3::exceptions::{PyOSError, PySyntaxError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList, PyModule, PyString, PyTuple};

use crate::pattern::GENERIC_FILE;
use crate::pyast;
use crate::tagged::TAGGED_NAME;

/// The language's own units (`reflect.language_config`): a name that stands for
/// the sum's unit or the product's unit.
const DEFAULT_NEVER_EQV: &[&str] = &["Oneof"];
const DEFAULT_NIL_EQV: &[&str] = &["Object", "Assert", "Appendix", "Hint"];

const CONTAINER_NAMES: &[&str] = &["list", "set", "dict"];
const UNIT_NAMES: &[&str] = &["Object", "nil", "void", "None", "never", "Oneof"];

// ----------------------------------------------------------------------
// The walk's own refusal
// ----------------------------------------------------------------------

pyo3::create_exception!(
    rs_viba_native,
    SliceError,
    pyo3::exceptions::PyException,
    "The first thing that does not hold; the walk stops here."
);

/// The first thing that does not hold, as the error the walk stops with.
///
/// It is an exception rather than a Rust error type so that every Python call
/// inside the walk converts with `?` the same way; the two entries turn it back
/// into the `Err` they answer.
fn slice_error(message: String) -> PyErr {
    SliceError::new_err(message)
}

// ----------------------------------------------------------------------
// The small readings the walk makes of Python objects
// ----------------------------------------------------------------------

/// Python's `repr` of a string, for the `!r` of an error message.
fn repr_str(py: Python<'_>, text: &str) -> String {
    let value = PyString::new(py, text);
    pyast::repr_of(py, value.as_any())
}

/// Python's `str(value)`: a file path, a module name, a message.
fn text_of(value: &Bound<'_, PyAny>) -> PyResult<String> {
    value.str()?.extract()
}

/// `str(exc)`: the sentence an exception carries.
fn pyerr_text(py: Python<'_>, error: &PyErr) -> PyResult<String> {
    error.value(py).str()?.extract()
}

/// Python's `name.rsplit(".", 1)[-1]`: the last segment, or the whole name.
fn last_segment(name: &str) -> &str {
    match name.rsplit_once('.') {
        Some((_before, after)) => after,
        None => name,
    }
}

/// Python's `os.path.join(directory, GENERIC_FILE)`, on the paths a generic's
/// directory is.
fn os_join(directory: &str, name: &str) -> String {
    if directory.is_empty() {
        name.to_string()
    } else if directory.ends_with('/') {
        format!("{directory}{name}")
    } else {
        format!("{directory}/{name}")
    }
}

/// `_path_of_module`: the file a module is — its name's dots become directories,
/// `.viba` at the end.
fn path_of_module(module_name: &str) -> String {
    format!("{}.viba", module_name.replace('.', "/"))
}

/// `_module_of_path`: the module a file is — the path without `.viba`, its
/// separators dots.
fn module_of_path(path: &str) -> String {
    let name = path.strip_suffix(".viba").unwrap_or(path);
    name.replace('/', ".")
}

/// Python's `Path.resolve()` with `strict=False`: the absolute path, symlinks
/// resolved as far as the filesystem has them, and `.`/`..` folded.
fn resolve_path(path: &str) -> PathBuf {
    let original = Path::new(path);
    if let Ok(canonical) = original.canonicalize() {
        return canonical;
    }
    let mut prefix = original.to_path_buf();
    let mut rest: Vec<std::ffi::OsString> = Vec::new();
    loop {
        if let Ok(canonical) = prefix.canonicalize() {
            let mut out = canonical;
            for part in rest.iter().rev() {
                out.push(part);
            }
            return lexical_normalize(&out);
        }
        match prefix.file_name() {
            Some(name) => {
                rest.push(name.to_os_string());
                if !prefix.pop() {
                    break;
                }
            }
            None => break,
        }
    }
    lexical_normalize(original)
}

fn lexical_normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// Python's `Path.read_text(encoding="utf-8")`: text mode folds `\r\n` and `\r`
/// into `\n`, and decoding is strict.
fn read_text(path: &str) -> std::io::Result<String> {
    let bytes = std::fs::read(path)?;
    let text = String::from_utf8(bytes)
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
    Ok(translate_newlines(&text))
}

fn translate_newlines(text: &str) -> String {
    if !text.contains('\r') {
        return text.to_string();
    }
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\r' {
            if chars.peek() == Some(&'\n') {
                chars.next();
            }
            out.push('\n');
        } else {
            out.push(ch);
        }
    }
    out
}

/// Python's `str.splitlines()`, as `rs_viba.markdown` spells it.
fn split_lines(text: &str) -> Vec<String> {
    crate::markdown::split_lines(text)
        .into_iter()
        .map(str::to_string)
        .collect()
}

/// `module_label`: what an error says a module is.
fn module_label(module: &Bound<'_, PyAny>) -> PyResult<String> {
    if let Ok(file_path) = module.getattr("file_path") {
        if file_path.is_truthy()? {
            return text_of(&file_path);
        }
    }
    if let Ok(name) = module.getattr("name") {
        if name.is_truthy()? {
            return text_of(&name);
        }
    }
    Ok("?".to_string())
}

/// A Result the walk just asked for: is it the error branch?
fn result_is_err(value: &Bound<'_, PyAny>, err_class: &Bound<'_, PyAny>) -> PyResult<bool> {
    value.is_instance(err_class)
}

/// The message of a Result's error branch.
fn result_msg(value: &Bound<'_, PyAny>) -> PyResult<String> {
    text_of(&value.getattr("msg")?)
}

// ----------------------------------------------------------------------
// The Slicer
// ----------------------------------------------------------------------

/// One file of the sub-repo that keeps some of its definitions.
struct KeptFile {
    /// The file this entry is keyed by: the module's `file_path`.
    file_path: String,
    /// The module the file was first reached as; its parse tree is the one
    /// `_kept_lines` reads (the loader answers one module per file).
    module: Py<PyAny>,
    names: HashSet<String>,
    imports: HashSet<String>,
}

/// One extraction: the loader, what was reached, and where the sub-repo goes.
struct Slicer<'py> {
    py: Python<'py>,
    type_module: Bound<'py, PyModule>,
    pattern_module: Bound<'py, PyModule>,
    walk_module: Bound<'py, PyModule>,
    markdown_module: Bound<'py, PyModule>,
    ast_module: Bound<'py, PyModule>,
    err_class: Bound<'py, PyAny>,
    custom_module_class: Bound<'py, PyAny>,
    ast_node_type_class: Bound<'py, PyAny>,

    repo: String,
    repo_root: PathBuf,
    extract_roots: Vec<PathBuf>,
    type_names: Vec<String>,
    never_eqv: HashSet<String>,
    nil_eqv: HashSet<String>,
    loader: Py<PyAny>,

    /// The files with some of their definitions kept, in the order reached.
    kept: Vec<KeptFile>,
    kept_index: HashMap<String, usize>,
    /// The files kept whole (a generic's marker and pattern files):
    /// module name -> (the file, or the section of a document).
    whole: Vec<(String, (String, Option<String>))>,
    whole_index: HashMap<String, usize>,
    whole_walked: HashSet<String>,
    /// (file, definition, bindings) already walked.
    visited: HashSet<(String, String, Vec<(String, String)>)>,
}

impl<'py> Slicer<'py> {
    fn new(
        py: Python<'py>,
        repo: String,
        viba_path: Vec<String>,
        extract_source: Vec<String>,
        type_names: Vec<String>,
        never_eqv: Option<Vec<String>>,
        nil_eqv: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let type_module = PyModule::import(py, "rs_viba.type")?;
        let pattern_module = PyModule::import(py, "rs_viba.pattern")?;
        let walk_module = PyModule::import(py, "rs_viba.design_walk")?;
        let markdown_module = PyModule::import(py, "rs_viba.markdown")?;
        let ast_module = PyModule::import(py, "rs_viba.viba_ast")?;
        let err_class = type_module.getattr("Err")?;
        let custom_module_class = type_module.getattr("CustomModuleType")?;
        let ast_node_type_class = type_module.getattr("AstNodeType")?;

        // The loader is the one a run would use, so a name resolves where it
        // would resolve when the design runs. Nothing is run through it.
        let interpret = PyModule::import(py, "rs_viba.interpret")?;
        let runner_class = interpret.getattr("_Runner")?;
        let search: Py<PyAny> = if viba_path.is_empty() {
            py.None()
        } else {
            PyString::new(py, &viba_path.join(":")).into_any().unbind()
        };
        let loader = runner_class.call1((search,))?.unbind();

        let repo_root = resolve_path(&repo);
        let extract_roots = extract_source.iter().map(|one| resolve_path(one)).collect();
        let never: HashSet<String> = match never_eqv {
            None => DEFAULT_NEVER_EQV
                .iter()
                .map(|one| one.to_string())
                .collect(),
            Some(list) => list.into_iter().collect(),
        };
        let nil: HashSet<String> = match nil_eqv {
            None => DEFAULT_NIL_EQV.iter().map(|one| one.to_string()).collect(),
            Some(list) => list.into_iter().collect(),
        };
        Ok(Self {
            py,
            type_module,
            pattern_module,
            walk_module,
            markdown_module,
            ast_module,
            err_class,
            custom_module_class,
            ast_node_type_class,
            repo,
            repo_root,
            extract_roots,
            type_names,
            never_eqv: never,
            nil_eqv: nil,
            loader,
            kept: Vec::new(),
            kept_index: HashMap::new(),
            whole: Vec::new(),
            whole_index: HashMap::new(),
            whole_walked: HashSet::new(),
            visited: HashSet::new(),
        })
    }

    // ---- the entry ----

    /// The sub-repo as one document, and where each of its code lines came from.
    fn markdown(&mut self) -> PyResult<(String, Vec<(usize, String, usize)>)> {
        if !Path::new(&self.repo).is_dir() {
            return Err(slice_error(format!(
                "no such repo directory: {}",
                self.repo
            )));
        }
        for name in self.type_names.clone() {
            self.start(&name)?;
        }
        self.render()
    }

    fn start(&mut self, name: &str) -> PyResult<()> {
        let (module_name, definition_name) = split_type_name(name);
        if module_name.is_empty() {
            return Err(slice_error(format!(
                "{} is no type name: it is a module and a definition, like \
                 'pkg.mod.Name'",
                repr_str(self.py, name)
            )));
        }
        let loader = self.loader.clone_ref(self.py);
        let loaded = loader
            .bind(self.py)
            .call_method1("imported", (module_name.as_str(), self.py.None()))?;
        if result_is_err(&loaded, &self.err_class)? {
            return Err(slice_error(format!(
                "cannot load the module {} of {}: {}",
                repr_str(self.py, &module_name),
                repr_str(self.py, name),
                result_msg(&loaded)?
            )));
        }
        let module = loaded.getattr("ok_value")?;
        if !module.is_instance(&self.custom_module_class)? {
            return Err(slice_error(format!(
                "{} names a generic: a generic answers an application, not a bare \
                 name",
                repr_str(self.py, name)
            )));
        }
        let file_path = self.file_path_of(&module)?;
        if !self.is_source_file(file_path.as_deref()) {
            return Err(slice_error(format!(
                "{} is not in the extraction source: {}",
                repr_str(self.py, name),
                module_label(&module)?
            )));
        }
        if self.is_pattern_file(&module)? {
            return Err(slice_error(format!(
                "{} is a definition in a pattern file: a generic's files are kept \
                 whole by an application of it, not by a name",
                repr_str(self.py, name)
            )));
        }
        let found = self
            .type_module
            .getattr("module_get_type")?
            .call1((module.clone(), definition_name.as_str()))?;
        if result_is_err(&found, &self.err_class)? {
            return Err(slice_error(format!(
                "no type {}: {}",
                repr_str(self.py, name),
                result_msg(&found)?
            )));
        }
        let target = found.getattr("ok_value")?;
        if !target.is_instance(&self.ast_node_type_class)?
            || !self.is_definition(&target.getattr("ast_node")?)?
        {
            return Err(slice_error(format!(
                "{} is no definition",
                repr_str(self.py, name)
            )));
        }
        let ast_node = target.getattr("ast_node")?;
        let container = target.getattr("container_module")?;
        let empty = PyDict::new(self.py);
        self.keep(&ast_node, &container, empty.as_any())?;
        Ok(())
    }

    // ---- keeping ----

    /// Keep this definition, and walk what it reaches.
    fn keep(
        &mut self,
        definition: &Bound<'py, PyAny>,
        module: &Bound<'py, PyAny>,
        bindings: &Bound<'py, PyAny>,
    ) -> PyResult<()> {
        if !module.is_instance(&self.custom_module_class)? {
            return Ok(()); // the builtin vocabulary: the interpreter's
        }
        let file_path = match self.file_path_of(module)? {
            Some(path) if !path.is_empty() => path,
            _ => return Ok(()),
        };
        if self.under_builtin(&file_path)? {
            return Ok(());
        }
        if !self.is_source_file(Some(&file_path)) {
            return Ok(());
        }
        let definition_name: String = definition.getattr("name")?.extract()?;
        let key = (
            file_path.clone(),
            definition_name.clone(),
            self.bindings_key(bindings)?,
        );
        if self.visited.contains(&key) {
            return Ok(());
        }
        self.visited.insert(key);
        let body = definition.getattr("body")?;
        if self.is_generic_file(&file_path) {
            // The file is kept whole, so every definition in it is walked; this
            // one also stands in the bindings the decision gave it.
            self.keep_whole(module, None)?;
            let inner = self.with_params(definition, bindings)?;
            return self.walk(&body, module, inner.bind(self.py), false);
        }
        let index = self.kept_file_index(module, &file_path)?;
        self.kept[index].names.insert(definition_name);
        let inner = self.with_params(definition, bindings)?;
        self.walk(&body, module, inner.bind(self.py), false)
    }

    /// Keep the whole file, and walk every definition it has.
    ///
    /// A generic's pattern file is kept whole: which definitions the file needs
    /// is the file's own business. What the file's own definitions mention is
    /// walked with no bindings — a name no decision bound there is a pattern
    /// parameter, and a pattern parameter is a placeholder.
    fn keep_whole(&mut self, module: &Bound<'py, PyAny>, document: Option<String>) -> PyResult<()> {
        let file_path = self.file_path_of(module)?;
        let name = self.whole_name(module, document.as_deref())?;
        let real_path = match file_path {
            Some(path) if !path.is_empty() => Some(path),
            _ => None,
        };
        if real_path.is_none() && document.is_none() {
            return Ok(());
        }
        if self.whole_walked.contains(&name) {
            return Ok(());
        }
        let mut path = real_path;
        let mut section = None;
        if path.is_none() {
            // A section of a document: the document is the file, the section
            // its text.
            let document = document.unwrap_or_default();
            section = Some(format!("{}.viba", last_segment(&name)));
            path = Some(document);
        }
        let path = path.unwrap_or_default();
        self.whole_walked.insert(name.clone());
        self.set_whole(name, (path.clone(), section));
        let module_tree = module.getattr("module")?;
        let body = module_tree.getattr("body")?;
        for statement in body.try_iter()? {
            let statement = statement?;
            if !self.is_definition(&statement)? {
                continue;
            }
            let statement_name: String = statement.getattr("name")?.extract()?;
            let empty = PyDict::new(self.py);
            let key = (
                path.clone(),
                statement_name,
                self.bindings_key(empty.as_any())?,
            );
            if self.visited.contains(&key) {
                continue;
            }
            self.visited.insert(key);
            let statement_body = statement.getattr("body")?;
            let empty = PyDict::new(self.py);
            let inner = self.with_params(&statement, empty.as_any())?;
            self.walk(&statement_body, module, inner.bind(self.py), true)?;
        }
        Ok(())
    }

    /// Keep the marker and the pattern files the application can take.
    fn keep_generic(
        &mut self,
        generic: &Bound<'py, PyAny>,
        arguments: usize,
        choice: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<()> {
        let directory = text_of(&generic.getattr("directory")?)?;
        let marker = os_join(&directory, GENERIC_FILE);
        if !self.is_source_file(Some(&marker)) {
            return Ok(());
        }
        let is_document = directory.ends_with(".viba.md");
        let generic_name = text_of(&generic.getattr("name")?)?;
        let value = if is_document {
            (directory, Some(GENERIC_FILE.to_string()))
        } else {
            (marker, None)
        };
        self.set_whole(format!("{generic_name}.__generic__"), value);
        if let Some(choice) = choice {
            let entry = choice.getattr("entry")?;
            return self.keep_pattern(&entry);
        }
        let entries = generic.getattr("entries")?;
        for entry in entries.try_iter()? {
            let entry = entry?;
            let got = entry.call_method0("take")?;
            if result_is_err(&got, &self.err_class)? {
                return Err(slice_error(format!(
                    "cannot take {}: {}",
                    text_of(&entry.getattr("path")?)?,
                    result_msg(&got)?
                )));
            }
            let module_and_patterns = got.getattr("ok_value")?;
            let patterns = module_and_patterns.get_item(1)?;
            if patterns.len()? == arguments {
                self.keep_pattern(&entry)?;
            }
        }
        Ok(())
    }

    fn keep_pattern(&mut self, entry: &Bound<'py, PyAny>) -> PyResult<()> {
        let got = entry.call_method0("take")?;
        if result_is_err(&got, &self.err_class)? {
            return Err(slice_error(format!(
                "cannot take {}: {}",
                text_of(&entry.getattr("path")?)?,
                result_msg(&got)?
            )));
        }
        let module_and_patterns = got.getattr("ok_value")?;
        let module = module_and_patterns.get_item(0)?;
        if !module.is_instance(&self.custom_module_class)? {
            return Ok(());
        }
        let entry_path = text_of(&entry.getattr("path")?)?;
        if !self.is_source_file(Some(&entry_path)) {
            return Ok(());
        }
        let parent = Path::new(&entry_path)
            .parent()
            .map(|path| path.to_string_lossy().to_string())
            .unwrap_or_default();
        let document = if parent.ends_with(".viba.md") {
            Some(parent)
        } else {
            None
        };
        self.keep_whole(&module, document)
    }

    fn kept_file_index(&mut self, module: &Bound<'py, PyAny>, file_path: &str) -> PyResult<usize> {
        if let Some(index) = self.kept_index.get(file_path) {
            return Ok(*index);
        }
        let index = self.kept.len();
        self.kept_index.insert(file_path.to_string(), index);
        self.kept.push(KeptFile {
            file_path: file_path.to_string(),
            module: module.clone().unbind(),
            names: HashSet::new(),
            imports: HashSet::new(),
        });
        Ok(index)
    }

    fn set_whole(&mut self, name: String, value: (String, Option<String>)) {
        if let Some(index) = self.whole_index.get(&name) {
            self.whole[*index].1 = value;
        } else {
            self.whole_index.insert(name.clone(), self.whole.len());
            self.whole.push((name, value));
        }
    }

    // ---- the walk ----

    fn walk(
        &mut self,
        node: &Bound<'py, PyAny>,
        module: &Bound<'py, PyAny>,
        bindings: &Bound<'py, PyAny>,
        loose: bool,
    ) -> PyResult<()> {
        if node.is_none() {
            return Ok(());
        }
        let class = pyast::class_name(node)?;
        match class.as_str() {
            "Constant" | "Nil" | "Never" | "Any" | "Ellipsis" | "HintBlock" | "Member" => {
                return Ok(())
            }
            "TypeRef" => {
                let name: String = node.getattr("name")?.extract()?;
                return self.name(&name, module, bindings, loose);
            }
            "MemberTaken" => return self.member(node, module, bindings, loose),
            "Tagged" => {
                let inner = node.getattr("type")?;
                return self.walk(&inner, module, bindings, loose);
            }
            "TypeApp" => return self.application(node, module, bindings, loose),
            "Tuple" => {
                for element in node.getattr("elements")?.try_iter()? {
                    let element = element?;
                    self.walk(&element, module, bindings, loose)?;
                }
                return Ok(());
            }
            "Partial" => return self.call(node, module, bindings, loose),
            _ => {}
        }
        if let Some(elements) = self.chain_elements(node)? {
            for element in elements {
                self.walk(&element, module, bindings, loose)?;
            }
        }
        Ok(())
    }

    fn name(
        &mut self,
        name: &str,
        module: &Bound<'py, PyAny>,
        bindings: &Bound<'py, PyAny>,
        loose: bool,
    ) -> PyResult<()> {
        let dict = bindings.cast::<PyDict>()?;
        if let Some(bound) = dict.get_item(name)? {
            if bound.is_none() {
                return Ok(()); // a generic parameter, no argument yet
            }
            let node = bound.get_item(0)?;
            let container = bound.get_item(1)?;
            let inner = bound.get_item(2)?;
            return self.walk(&node, &container, &inner, false);
        }
        if self.never_eqv.contains(name) || self.nil_eqv.contains(name) {
            return Ok(()); // a unit: no definition behind it
        }
        if UNIT_NAMES.contains(&name) {
            return Ok(());
        }
        let found = self
            .type_module
            .getattr("module_get_type")?
            .call1((module.clone(), name))?;
        if result_is_err(&found, &self.err_class)? {
            if loose {
                return Ok(()); // a pattern parameter of this file
            }
            return Err(slice_error(format!(
                "cannot resolve {} in {}: {}",
                repr_str(self.py, name),
                module_label(module)?,
                result_msg(&found)?
            )));
        }
        let target = found.getattr("ok_value")?;
        if !target.is_instance(&self.ast_node_type_class)? {
            return Ok(()); // a builtin scalar, container or form
        }
        let ast_node = target.getattr("ast_node")?;
        if !self.is_definition(&ast_node)? {
            return Ok(());
        }
        self.note_import(module, name)?;
        let container = target.getattr("container_module")?;
        let empty = PyDict::new(self.py);
        self.keep(&ast_node, &container, empty.as_any())?;
        Ok(())
    }

    /// A kept definition of this file names something through an import.
    fn note_import(&mut self, module: &Bound<'py, PyAny>, name: &str) -> PyResult<()> {
        if !module.is_instance(&self.custom_module_class)? {
            return Ok(());
        }
        let file_path = match self.file_path_of(module)? {
            Some(path) if !path.is_empty() => path,
            _ => return Ok(()),
        };
        if !self.is_source_file(Some(&file_path)) {
            return Ok(());
        }
        if self.is_generic_file(&file_path) {
            return Ok(()); // whole file: every import stays
        }
        let imports = module.getattr("imports")?;
        if let Some(local) = self.longest_import(&imports, name)? {
            let index = self.kept_file_index(module, &file_path)?;
            self.kept[index].imports.insert(local);
        }
        Ok(())
    }

    fn application(
        &mut self,
        node: &Bound<'py, PyAny>,
        module: &Bound<'py, PyAny>,
        bindings: &Bound<'py, PyAny>,
        loose: bool,
    ) -> PyResult<()> {
        let constructor: String = text_of(&node.getattr("constructor")?)?;
        let args = node.getattr("args")?;
        let arguments: Vec<Bound<'py, PyAny>> = args.try_iter()?.collect::<PyResult<Vec<_>>>()?;
        if constructor == TAGGED_NAME {
            for argument in arguments.iter().skip(1) {
                self.walk(argument, module, bindings, loose)?;
            }
            return Ok(());
        }
        for argument in &arguments {
            self.walk(argument, module, bindings, loose)?;
        }
        if CONTAINER_NAMES.contains(&constructor.as_str())
            || UNIT_NAMES.contains(&constructor.as_str())
        {
            return Ok(());
        }
        if self.never_eqv.contains(&constructor) || self.nil_eqv.contains(&constructor) {
            return Ok(());
        }
        self.note_import(module, &constructor)?;
        let generic = self
            .pattern_module
            .getattr("generic_named")?
            .call1((module.clone(), constructor.as_str()))?;
        if result_is_err(&generic, &self.err_class)? {
            return Err(slice_error(format!(
                "cannot take {} in {}: {}",
                repr_str(self.py, &constructor),
                module_label(module)?,
                result_msg(&generic)?
            )));
        }
        let generic = generic.getattr("ok_value")?;
        if generic.is_none() {
            return self.name(&constructor, module, bindings, loose);
        }
        let argument_modules = self.argument_modules(&arguments, bindings)?;
        let got = self.decide(&generic, &args, module, argument_modules.as_ref())?;
        let got = got.bind(self.py);
        let choice = if result_is_err(got, &self.err_class)? {
            None
        } else {
            Some(got.getattr("ok_value")?)
        };
        self.keep_generic(&generic, arguments.len(), choice.as_ref())?;
        let Some(choice) = choice else {
            return Ok(());
        };
        let body = choice.getattr("body")?;
        if body.is_none() {
            return Ok(());
        }
        let inner = self.bind_choice(bindings, &choice)?;
        let choice_module = choice.getattr("module")?;
        self.walk(&body, &choice_module, inner.bind(self.py), false)
    }

    /// The module each argument was spelled in, where a decision bound it.
    fn argument_modules(
        &self,
        arguments: &[Bound<'py, PyAny>],
        bindings: &Bound<'py, PyAny>,
    ) -> PyResult<Option<Py<PyAny>>> {
        let dict = bindings.cast::<PyDict>()?;
        let mut kept: Vec<Option<Bound<'py, PyAny>>> = Vec::new();
        for argument in arguments {
            let mut where_ = None;
            if pyast::class_name(argument)? == "TypeRef" {
                let name: String = argument.getattr("name")?.extract()?;
                if let Some(bound) = dict.get_item(name.as_str())? {
                    if bound.is_instance_of::<PyTuple>() {
                        where_ = Some(bound.get_item(1)?);
                    }
                }
            }
            kept.push(where_);
        }
        if kept.iter().all(|one| one.is_none()) {
            return Ok(None);
        }
        let list = PyList::empty(self.py);
        for one in kept {
            match one {
                Some(value) => list.append(value)?,
                None => list.append(self.py.None())?,
            }
        }
        Ok(Some(list.into_any().unbind()))
    }

    fn member(
        &mut self,
        node: &Bound<'py, PyAny>,
        module: &Bound<'py, PyAny>,
        bindings: &Bound<'py, PyAny>,
        loose: bool,
    ) -> PyResult<()> {
        let source_path = self
            .ast_module
            .getattr("source_path")?
            .call1((node.clone(),))?;
        if !source_path.is_none() {
            let path: String = source_path.extract()?;
            return self.name(&path, module, bindings, loose);
        }
        let owner = node.getattr("owner")?;
        if pyast::class_name(&owner)? != "TypeApp" {
            return Ok(());
        }
        let constructor: String = text_of(&owner.getattr("constructor")?)?;
        self.note_import(module, &constructor)?;
        let generic = self
            .pattern_module
            .getattr("generic_named")?
            .call1((module.clone(), constructor.as_str()))?;
        if result_is_err(&generic, &self.err_class)? {
            return Err(slice_error(format!(
                "cannot take {} in {}: {}",
                repr_str(self.py, &constructor),
                module_label(module)?,
                result_msg(&generic)?
            )));
        }
        let generic = generic.getattr("ok_value")?;
        if generic.is_none() {
            // Not a generic directory: the application is taken as it always is,
            // and the member is a member of what it answers.
            return self.application(&owner, module, bindings, loose);
        }
        let args = owner.getattr("args")?;
        let arguments: Vec<Bound<'py, PyAny>> = args.try_iter()?.collect::<PyResult<Vec<_>>>()?;
        for argument in &arguments {
            self.walk(argument, module, bindings, loose)?;
        }
        let argument_modules = self.argument_modules(&arguments, bindings)?;
        let got = self.decide(&generic, &args, module, argument_modules.as_ref())?;
        let got = got.bind(self.py);
        let choice = if result_is_err(got, &self.err_class)? {
            None
        } else {
            Some(got.getattr("ok_value")?)
        };
        self.keep_generic(&generic, arguments.len(), choice.as_ref())?;
        let Some(choice) = choice else {
            return Ok(());
        };
        let inner = self.bind_choice(bindings, &choice)?;
        let body = choice.getattr("body")?;
        if body.is_none() {
            // The file is a module: the member names one of its definitions.
            let member_name: String = node.getattr("name")?.extract()?;
            let choice_module = choice.getattr("module")?;
            return self.keep_named(&choice_module, &member_name, inner.bind(self.py));
        }
        let choice_module = choice.getattr("module")?;
        self.walk(&body, &choice_module, inner.bind(self.py), false)
    }

    fn keep_named(
        &mut self,
        module: &Bound<'py, PyAny>,
        name: &str,
        bindings: &Bound<'py, PyAny>,
    ) -> PyResult<()> {
        if !module.is_instance(&self.custom_module_class)? {
            return Ok(());
        }
        let module_tree = module.getattr("module")?;
        let body = module_tree.getattr("body")?;
        for statement in body.try_iter()? {
            let statement = statement?;
            if self.is_definition(&statement)? {
                let statement_name: String = statement.getattr("name")?.extract()?;
                if statement_name == name {
                    return self.keep(&statement, module, bindings);
                }
            }
        }
        Err(slice_error(format!(
            "no definition {} in {}",
            repr_str(self.py, name),
            module_label(module)?
        )))
    }

    fn call(
        &mut self,
        node: &Bound<'py, PyAny>,
        module: &Bound<'py, PyAny>,
        bindings: &Bound<'py, PyAny>,
        loose: bool,
    ) -> PyResult<()> {
        let (head, arguments) = self.call_parts(node)?;
        self.walk(&head, module, bindings, loose)?;
        for argument in arguments {
            let class = pyast::class_name(&argument)?;
            if class == "HintBlock" {
                continue; // documentation is no argument
            }
            if class == "Tagged" {
                let inner = argument.getattr("type")?;
                self.walk(&inner, module, bindings, loose)?;
            } else {
                self.walk(&argument, module, bindings, loose)?;
            }
        }
        Ok(())
    }

    // ---- the files ----

    fn is_source_file(&self, file_path: Option<&str>) -> bool {
        let Some(file_path) = file_path else {
            return false;
        };
        if file_path.is_empty() {
            return false;
        }
        let path = resolve_path(file_path);
        if !path.starts_with(&self.repo_root) {
            return false;
        }
        self.extract_roots.iter().any(|root| path.starts_with(root))
    }

    /// Is this file one of a generic's: the marker, or a numbered file beside it?
    fn is_generic_file(&self, file_path: &str) -> bool {
        let path = Path::new(file_path);
        if path
            .file_name()
            .map(|name| name == GENERIC_FILE)
            .unwrap_or(false)
        {
            return true;
        }
        match path.parent() {
            Some(parent) => parent.join(GENERIC_FILE).is_file(),
            None => false,
        }
    }

    /// Is this module one pattern file: a file beside a marker, or a section of a
    /// document that holds a marker?
    fn is_pattern_file(&self, module: &Bound<'py, PyAny>) -> PyResult<bool> {
        let file_path = match self.file_path_of(module)? {
            Some(path) if !path.is_empty() => path,
            _ => return Ok(false),
        };
        let path = Path::new(&file_path);
        let name = path
            .file_name()
            .map(|name| name.to_string_lossy().to_string())
            .unwrap_or_default();
        if name == GENERIC_FILE {
            return Ok(false);
        }
        if name.ends_with(".viba") && self.is_generic_file(&file_path) {
            return Ok(true);
        }
        if !name.ends_with(".viba.md") {
            return Ok(false);
        }
        let text = match read_text(&file_path) {
            Ok(text) => text,
            Err(error) if error.kind() == std::io::ErrorKind::InvalidData => {
                // Python's text mode raises a UnicodeDecodeError, which is no
                // OSError: it is not caught here either.
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "the document's bytes are not utf-8",
                ));
            }
            Err(_error) => return Ok(false), // OSError
        };
        let sections = match self.files_of(&text) {
            Ok(sections) => sections,
            Err(error) => {
                if error.is_instance_of::<PySyntaxError>(self.py)
                    || error.is_instance_of::<PyOSError>(self.py)
                {
                    return Ok(false);
                }
                return Err(error);
            }
        };
        let module_name = match module.getattr("name") {
            Ok(value) if value.is_truthy()? => text_of(&value)?,
            _ => String::new(),
        };
        let section = format!("{}.viba", last_segment(&module_name));
        Ok(sections.iter().any(|(one, _)| one == GENERIC_FILE)
            && sections.iter().any(|(one, _)| one == &section)
            && section != GENERIC_FILE)
    }

    // ---- the answer ----

    /// The kept files as one document's sections, and the line map.
    fn render(&self) -> PyResult<(String, Vec<(usize, String, usize)>)> {
        let mut entries: Vec<(String, Vec<(String, String, usize)>)> = Vec::new();
        for index in 0..self.kept.len() {
            let file_path = self.kept[index].file_path.clone();
            let module_name = self.module_name(index, &file_path)?;
            let lines = self.kept_lines(index, &module_name)?;
            entries.push((path_of_module(&module_name), lines));
        }
        for (module_name, (path, section)) in &self.whole {
            let lines = self.lines_at(path, section.as_deref())?;
            let parts = lines
                .into_iter()
                .enumerate()
                .map(|(offset, text)| (text, module_name.clone(), offset + 1))
                .collect();
            entries.push((path_of_module(module_name), parts));
        }
        entries.sort_by(|left, right| left.0.cmp(&right.0));
        let mut lines: Vec<String> = Vec::new();
        let mut origin: Vec<(usize, String, usize)> = Vec::new();
        for (path, parts) in entries {
            lines.push(format!("## `{path}`"));
            lines.push(String::new());
            lines.push(format!("```viba {path}"));
            for (text, module, module_line) in parts {
                lines.push(text);
                origin.push((lines.len(), module, module_line));
            }
            lines.push("```".to_string());
            lines.push(String::new());
        }
        let document = if lines.is_empty() {
            String::new()
        } else {
            format!("{}\n", lines.join("\n").trim_end_matches('\n'))
        };
        Ok((document, origin))
    }

    /// The name a file kept whole answers to: its module's, or its path taken as
    /// one.
    fn whole_name(&self, module: &Bound<'py, PyAny>, document: Option<&str>) -> PyResult<String> {
        let name = match module.getattr("name") {
            Ok(value) if value.is_truthy()? => text_of(&value)?,
            _ => String::new(),
        };
        if !name.is_empty() {
            return Ok(name);
        }
        let file_path = match document {
            Some(path) if !path.is_empty() => Some(path.to_string()),
            _ => match self.file_path_of(module)? {
                Some(path) if !path.is_empty() => Some(path),
                _ => None,
            },
        };
        match file_path {
            Some(path) => Ok(module_of_path(&self.relative(&path)?)),
            None => Err(PyTypeError::new_err(
                "a file kept whole needs a path: the module records none",
            )),
        }
    }

    /// The module's own name: what it was loaded as, or its path taken as one.
    fn module_name(&self, index: usize, file_path: &str) -> PyResult<String> {
        let module = self.kept[index].module.bind(self.py);
        let name = match module.getattr("name") {
            Ok(value) if value.is_truthy()? => text_of(&value)?,
            _ => String::new(),
        };
        if !name.is_empty() {
            return Ok(name);
        }
        Ok(module_of_path(&self.relative(file_path)?))
    }

    /// The file's kept statements, line by line, with where each came from.
    fn kept_lines(
        &self,
        index: usize,
        module_name: &str,
    ) -> PyResult<Vec<(String, String, usize)>> {
        let source = self.module_lines(index, module_name)?;
        let entry = &self.kept[index];
        let module_tree = entry.module.bind(self.py).getattr("module")?;
        let body = module_tree.getattr("body")?;
        let mut statements: Vec<(Bound<'py, PyAny>, i64)> = Vec::new();
        for statement in body.try_iter()? {
            let statement = statement?;
            if let Some(line) = lineno_of(&statement)? {
                if line != 0 {
                    statements.push((statement, line));
                }
            }
        }
        let mut out: Vec<(String, String, usize)> = Vec::new();
        for (statement, start) in &statements {
            let class = pyast::class_name(statement)?;
            if class == "Import" {
                let alias = statement.getattr("alias")?;
                let chosen = if alias.is_truthy()? {
                    alias
                } else {
                    statement.getattr("module")?
                };
                let text: String = chosen.extract()?;
                if !entry.imports.contains(&text) {
                    continue;
                }
            } else {
                let is_definition = self.is_definition(statement)?;
                if !is_definition {
                    continue;
                }
                let name: String = statement.getattr("name")?.extract()?;
                if !entry.names.contains(&name) {
                    continue;
                }
            }
            let after = lineno_after(&statements, statement)?;
            let end = after.unwrap_or(source.len() as i64 + 1);
            let from = (*start as usize).saturating_sub(1);
            let to = (end.max(0) as usize).saturating_sub(1).min(source.len());
            if from < to {
                for offset in from..to {
                    out.push((source[offset].clone(), module_name.to_string(), offset + 1));
                }
            }
        }
        while out
            .last()
            .map(|(text, _, _)| text.trim().is_empty())
            .unwrap_or(false)
        {
            out.pop(); // the file ends at its last line of text
        }
        Ok(out)
    }

    /// The module's own lines: the file's, or the section's of a document.
    fn module_lines(&self, index: usize, module_name: &str) -> PyResult<Vec<String>> {
        let module = self.kept[index].module.bind(self.py);
        self.module_lines_of(&module, module_name)
    }

    fn module_lines_of(
        &self,
        module: &Bound<'py, PyAny>,
        module_name: &str,
    ) -> PyResult<Vec<String>> {
        let file_path = self.file_path_of(module)?.unwrap_or_default();
        if file_path.ends_with(".viba.md") {
            let section = format!("{}.viba", last_segment(module_name));
            let text = self.read_text_or_py(&file_path)?;
            for (name, lines) in self.lines_of(&text)? {
                if name == section {
                    return Ok(lines.into_iter().map(|(text, _at)| text).collect());
                }
            }
            return Ok(Vec::new());
        }
        let text = self.read_text_or_py(&file_path)?;
        Ok(split_lines(&text))
    }

    fn read_text_or_py(&self, path: &str) -> PyResult<String> {
        read_text(path).map_err(|error| {
            if error.kind() == std::io::ErrorKind::InvalidData {
                pyo3::exceptions::PyValueError::new_err(format!("{path}: the bytes are not utf-8"))
            } else {
                PyOSError::new_err(error.to_string())
            }
        })
    }

    fn relative(&self, file_path: &str) -> PyResult<String> {
        let resolved = resolve_path(file_path);
        let relpath = self.py.import("os")?.getattr("path")?.getattr("relpath")?;
        let value = relpath.call1((
            resolved.to_string_lossy().to_string(),
            self.repo_root.to_string_lossy().to_string(),
        ))?;
        value.extract()
    }

    fn lines_at(&self, path: &str, section: Option<&str>) -> PyResult<Vec<String>> {
        let text = self.read_text_or_py(path)?;
        let Some(section) = section else {
            return Ok(split_lines(&text));
        };
        for (name, lines) in self.lines_of(&text)? {
            if name == section {
                return Ok(lines.into_iter().map(|(text, _at)| text).collect());
            }
        }
        Ok(Vec::new())
    }

    // ---- the readings this layer shares ----

    fn file_path_of(&self, module: &Bound<'py, PyAny>) -> PyResult<Option<String>> {
        match module.getattr("file_path") {
            Ok(value) if !value.is_none() => Ok(Some(text_of(&value)?)),
            Ok(_) => Ok(None),
            Err(error) if error.is_instance_of::<pyo3::exceptions::PyAttributeError>(self.py) => {
                Ok(None)
            }
            Err(error) => Err(error),
        }
    }

    fn is_definition(&self, node: &Bound<'py, PyAny>) -> PyResult<bool> {
        self.walk_module
            .getattr("is_definition")?
            .call1((node.clone(),))?
            .extract()
    }

    fn chain_elements(&self, node: &Bound<'py, PyAny>) -> PyResult<Option<Vec<Bound<'py, PyAny>>>> {
        let got = self
            .walk_module
            .getattr("chain_elements")?
            .call1((node.clone(),))?;
        if got.is_none() {
            return Ok(None);
        }
        Ok(Some(got.try_iter()?.collect::<PyResult<Vec<_>>>()?))
    }

    fn call_parts(
        &self,
        node: &Bound<'py, PyAny>,
    ) -> PyResult<(Bound<'py, PyAny>, Vec<Bound<'py, PyAny>>)> {
        let got = self
            .walk_module
            .getattr("call_parts")?
            .call1((node.clone(),))?;
        let head = got.get_item(0)?;
        let arguments = got.get_item(1)?.try_iter()?.collect::<PyResult<Vec<_>>>()?;
        Ok((head, arguments))
    }

    fn with_params(
        &self,
        definition: &Bound<'py, PyAny>,
        bindings: &Bound<'py, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        Ok(self
            .walk_module
            .getattr("with_params")?
            .call1((definition.clone(), bindings.clone()))?
            .unbind())
    }

    fn bindings_key(&self, bindings: &Bound<'py, PyAny>) -> PyResult<Vec<(String, String)>> {
        self.walk_module
            .getattr("bindings_key")?
            .call1((bindings.clone(),))?
            .extract()
    }

    fn longest_import(&self, imports: &Bound<'py, PyAny>, name: &str) -> PyResult<Option<String>> {
        let got = self
            .walk_module
            .getattr("longest_import")?
            .call1((imports.clone(), name))?;
        if got.is_none() {
            return Ok(None);
        }
        Ok(Some(got.extract()?))
    }

    fn under_builtin(&self, file_path: &str) -> PyResult<bool> {
        self.walk_module
            .getattr("under_builtin")?
            .call1((file_path,))?
            .extract()
    }

    fn decide(
        &self,
        generic: &Bound<'py, PyAny>,
        arguments: &Bound<'py, PyAny>,
        module: &Bound<'py, PyAny>,
        argument_modules: Option<&Py<PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let kwargs = PyDict::new(self.py);
        if let Some(modules) = argument_modules {
            kwargs.set_item("argument_modules", modules)?;
        }
        Ok(self
            .pattern_module
            .getattr("decide")?
            .call(
                (generic.clone(), arguments.clone(), module.clone()),
                Some(&kwargs),
            )?
            .unbind())
    }

    fn bind_choice(
        &self,
        bindings: &Bound<'py, PyAny>,
        choice: &Bound<'py, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let inner = bindings.cast::<PyDict>()?.copy()?;
        let bound = choice.getattr("bindings")?;
        for (pattern_name, value) in bound.cast::<PyDict>()?.iter() {
            let node = value.getattr("ast_node")?;
            let container = value.getattr("container_module")?;
            let triple = PyTuple::new(
                self.py,
                [node.unbind(), container.unbind(), bindings.clone().unbind()],
            )?;
            inner.set_item(pattern_name, triple)?;
        }
        Ok(inner.into_any().unbind())
    }

    fn files_of(&self, text: &str) -> PyResult<Vec<(String, String)>> {
        self.markdown_module
            .getattr("files_of")?
            .call1((text,))?
            .extract()
    }

    fn lines_of(&self, text: &str) -> PyResult<Vec<(String, Vec<(String, usize)>)>> {
        self.markdown_module
            .getattr("lines_of")?
            .call1((text,))?
            .extract()
    }
}

/// `statement.lineno`, or None when it has none (the original filters on the
/// attribute's truth, so 0 counts as none).
fn lineno_of(statement: &Bound<'_, PyAny>) -> PyResult<Option<i64>> {
    match statement.getattr("lineno") {
        Ok(value) if !value.is_none() => Ok(Some(value.extract()?)),
        Ok(_) => Ok(None),
        Err(error)
            if error.is_instance_of::<pyo3::exceptions::PyAttributeError>(statement.py()) =>
        {
            Ok(None)
        }
        Err(error) => Err(error),
    }
}

/// The line the next statement starts on, or None when this is the last.
fn lineno_after(
    statements: &[(Bound<'_, PyAny>, i64)],
    statement: &Bound<'_, PyAny>,
) -> PyResult<Option<i64>> {
    for (index, (other, _line)) in statements.iter().enumerate() {
        if other.is(statement) {
            return Ok(statements.get(index + 1).map(|(_next, line)| *line));
        }
    }
    Ok(None)
}

/// `split_type_name`: `pkg.mod.Name` -> (`pkg.mod`, `Name`); the name before the
/// last dot is the module.
fn split_type_name(name: &str) -> (String, String) {
    match name.rsplit_once('.') {
        Some((module_name, definition)) if !module_name.is_empty() && !definition.is_empty() => {
            (module_name.to_string(), definition.to_string())
        }
        _ => (String::new(), name.to_string()),
    }
}

// ----------------------------------------------------------------------
// The arguments
// ----------------------------------------------------------------------

fn argument_problem(
    py: Python<'_>,
    repo: &Bound<'_, PyAny>,
    viba_path: &Bound<'_, PyAny>,
    extract_source_viba_path: &Bound<'_, PyAny>,
    type_names: &Bound<'_, PyAny>,
    never_eqv: &Bound<'_, PyAny>,
    nil_eqv: &Bound<'_, PyAny>,
) -> PyResult<Option<String>> {
    let pathlike = py.import("os")?.getattr("PathLike")?;
    let is_path = repo.is_instance_of::<PyString>() || repo.is_instance(&pathlike)?;
    if !is_path {
        return Ok(Some(format!(
            "repo is a path, not {}",
            pyast::class_name(repo)?
        )));
    }
    let groups: [(&str, &Bound<'_, PyAny>); 5] = [
        ("viba_path", viba_path),
        ("extract_source_viba_path", extract_source_viba_path),
        ("type_names", type_names),
        ("never_eqv", never_eqv),
        ("nil_eqv", nil_eqv),
    ];
    for (label, given) in groups {
        if given.is_none() {
            if label == "never_eqv" || label == "nil_eqv" {
                continue;
            }
            return Ok(Some(format!("{label} is a list of strings")));
        }
        if given.is_instance_of::<PyString>() {
            return Ok(Some(format!(
                "{label} is a list of strings, not one string"
            )));
        }
        let mut parts: Vec<Bound<'_, PyAny>> = Vec::new();
        let iterator = match given.try_iter() {
            Ok(iterator) => iterator,
            Err(error) => {
                if error.is_instance_of::<pyo3::exceptions::PyTypeError>(py) {
                    return Ok(Some(format!(
                        "{label} is a list of strings, not {}",
                        pyast::class_name(given)?
                    )));
                }
                return Err(error);
            }
        };
        for item in iterator {
            match item {
                Ok(item) => parts.push(item),
                Err(error) => {
                    if error.is_instance_of::<pyo3::exceptions::PyTypeError>(py) {
                        return Ok(Some(format!(
                            "{label} is a list of strings, not {}",
                            pyast::class_name(given)?
                        )));
                    }
                    return Err(error);
                }
            }
        }
        for one in parts {
            if !one.is_instance_of::<PyString>() {
                return Ok(Some(format!(
                    "{label} holds a string, not {}",
                    pyast::class_name(&one)?
                )));
            }
        }
    }
    Ok(None)
}

/// Why the sections cannot be the files of one directory, or None when they can.
fn sections_problem(py: Python<'_>, names: &[String]) -> Option<String> {
    let known: HashSet<&str> = names.iter().map(|name| name.as_str()).collect();
    for name in names {
        let parts: Vec<&str> = name.split('/').collect();
        let outside = name.is_empty()
            || name.starts_with('/')
            || name.contains('\0')
            || parts
                .iter()
                .any(|part| part.is_empty() || *part == "." || *part == "..");
        if outside {
            return Some(format!(
                "a section names a file outside the document's directory: {}",
                repr_str(py, name)
            ));
        }
        for cut in 1..parts.len() {
            let prefix = parts[..cut].join("/");
            if known.contains(prefix.as_str()) {
                return Some(format!(
                    "a section names the file {}, and another section needs it to be \
                     a directory",
                    repr_str(py, &prefix)
                ));
            }
        }
    }
    None
}

/// The strings an iterable of strings holds, `[str(one) for one in value]`.
fn strings(value: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    let mut out = Vec::new();
    for item in value.try_iter()? {
        out.push(item?.str()?.extract()?);
    }
    Ok(out)
}

/// Python's `os.path.normpath`.
fn normpath(py: Python<'_>, path: &str) -> PyResult<String> {
    py.import("os")?
        .getattr("path")?
        .getattr("normpath")?
        .call1((path,))?
        .extract()
}

// ----------------------------------------------------------------------
// The two entries
// ----------------------------------------------------------------------

/// The refusal as the `Err` an entry answers; any other error stays the
/// exception it is.
fn fail(py: Python<'_>, error: PyErr) -> PyResult<Py<PyAny>> {
    if error.is_instance_of::<SliceError>(py) {
        let message = pyerr_text(py, &error)?;
        return make_err(py, &message);
    }
    Err(error)
}

fn make_err(py: Python<'_>, message: &str) -> PyResult<Py<PyAny>> {
    let class = PyModule::import(py, "rs_viba.type")?.getattr("Err")?;
    Ok(class.call1((message,))?.unbind())
}

fn make_ok(py: Python<'_>, value: Py<PyAny>) -> PyResult<Py<PyAny>> {
    let class = PyModule::import(py, "rs_viba.type")?.getattr("Ok")?;
    Ok(class.call1((value,))?.unbind())
}

/// `extract_sub_repo_into_viba_markdown`: the sub-repo as one document.
fn repo_entry(
    py: Python<'_>,
    repo: &Bound<'_, PyAny>,
    viba_path: &Bound<'_, PyAny>,
    extract_source_viba_path: &Bound<'_, PyAny>,
    type_names: &Bound<'_, PyAny>,
    never_eqv: &Bound<'_, PyAny>,
    nil_eqv: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    if let Some(problem) = argument_problem(
        py,
        repo,
        viba_path,
        extract_source_viba_path,
        type_names,
        never_eqv,
        nil_eqv,
    )? {
        return make_err(py, &problem);
    }
    let repo = normpath(py, &text_of(repo)?)?;
    let viba_paths = strings(viba_path)?;
    let extract_source = strings(extract_source_viba_path)?;
    let names = strings(type_names)?;
    let never = if never_eqv.is_none() {
        None
    } else {
        Some(strings(never_eqv)?)
    };
    let nil = if nil_eqv.is_none() {
        None
    } else {
        Some(strings(nil_eqv)?)
    };
    let mut slicer = Slicer::new(py, repo, viba_paths, extract_source, names, never, nil)?;
    match slicer.markdown() {
        Ok((document, origin)) => {
            let map = PyDict::new(py);
            for (at, module, line) in origin {
                let value = PyTuple::new(
                    py,
                    [pyast::pystr(py, &module), pyast::pyint(py, line as i64)],
                )?;
                map.set_item(pyast::pyint(py, at as i64), value)?;
            }
            let value = PyTuple::new(py, [pyast::pystr(py, &document), map.into_any().unbind()])?;
            make_ok(py, value.into_any().unbind())
        }
        Err(error) => fail(py, error),
    }
}

/// `extract_sub_viba_markdown`: a document in, a document out.
fn document_entry(
    py: Python<'_>,
    source_viba_md: &Bound<'_, PyAny>,
    type_names: &Bound<'_, PyAny>,
    never_eqv: &Bound<'_, PyAny>,
    nil_eqv: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    if !source_viba_md.is_instance_of::<PyString>() {
        return make_err(
            py,
            &format!(
                "source_viba_md is a document's text, not {}",
                pyast::class_name(source_viba_md)?
            ),
        );
    }
    let text: String = source_viba_md.extract()?;
    let markdown_module = PyModule::import(py, "rs_viba.markdown")?;
    let files: Vec<(String, String)> =
        match markdown_module.getattr("files_of")?.call1((text.as_str(),)) {
            Ok(value) => value.extract()?,
            Err(error) => {
                if error.is_instance_of::<PySyntaxError>(py) {
                    return make_err(py, &pyerr_text(py, &error)?);
                }
                return Err(error);
            }
        };
    let names: Vec<String> = files.iter().map(|(name, _body)| name.clone()).collect();
    if let Some(problem) = sections_problem(py, &names) {
        return make_err(py, &problem);
    }
    let tempfile = py.import("tempfile")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("prefix", "viba-extract-")?;
    let scratch = tempfile
        .getattr("TemporaryDirectory")?
        .call((), Some(&kwargs))?;
    let root: String = text_of(&scratch.getattr("name")?)?;

    // Write the sections out: the document is the repo the walk takes apart.
    let pathlib = py.import("pathlib")?;
    let root_path = pathlib.getattr("Path")?.call1((root.as_str(),))?;
    for (name, body) in &files {
        let path = root_path.call_method1("joinpath", (name.as_str(),))?;
        let result = (|| -> PyResult<()> {
            let mkdir = PyDict::new(py);
            mkdir.set_item("parents", true)?;
            mkdir.set_item("exist_ok", true)?;
            path.getattr("parent")?
                .call_method("mkdir", (), Some(&mkdir))?;
            let write = PyDict::new(py);
            write.set_item("encoding", "utf-8")?;
            path.call_method("write_text", (body.as_str(),), Some(&write))?;
            Ok(())
        })();
        if let Err(error) = result {
            scratch.call_method0("cleanup").ok();
            if error.is_instance_of::<PyOSError>(py) {
                return make_err(
                    py,
                    &format!(
                        "cannot write the section {}: {}",
                        repr_str(py, name),
                        pyerr_text(py, &error)?
                    ),
                );
            }
            return Err(error);
        }
    }

    let root_string = PyString::new(py, &root);
    let one_root = PyList::new(py, [root.as_str()])?;
    let problem = argument_problem(
        py,
        root_string.as_any(),
        one_root.as_any(),
        one_root.as_any(),
        type_names,
        never_eqv,
        nil_eqv,
    )?;
    if let Some(problem) = problem {
        scratch.call_method0("cleanup").ok();
        return make_err(py, &problem);
    }
    let names = strings(type_names)?;
    let never = if never_eqv.is_none() {
        None
    } else {
        Some(strings(never_eqv)?)
    };
    let nil = if nil_eqv.is_none() {
        None
    } else {
        Some(strings(nil_eqv)?)
    };
    let built = Slicer::new(
        py,
        root.clone(),
        vec![root.clone()],
        vec![root.clone()],
        names,
        never,
        nil,
    );
    let mut slicer = match built {
        Ok(slicer) => slicer,
        Err(error) => {
            scratch.call_method0("cleanup").ok();
            return Err(error);
        }
    };
    let outcome = slicer.markdown();
    scratch.call_method0("cleanup").ok();
    let (document, origin) = match outcome {
        Ok(answer) => answer,
        Err(error) => return fail(py, error),
    };

    // where a line of a module is in the source document: its section's lines
    let mut places: HashMap<(String, usize), usize> = HashMap::new();
    let sections: Vec<(String, Vec<(String, usize)>)> = markdown_module
        .getattr("lines_of")?
        .call1((text.as_str(),))?
        .extract()?;
    for (name, lines) in &sections {
        for (index, (_text, at)) in lines.iter().enumerate() {
            places.insert((name.clone(), index + 1), *at);
        }
    }
    let texts: HashMap<String, String> = files.into_iter().collect();

    let answer = PyDict::new(py);
    for (document_line, module, module_line) in origin {
        let at = source_line_of(py, &markdown_module, &places, &texts, &module, module_line)?;
        match at {
            Some(at) => {
                answer.set_item(
                    pyast::pyint(py, document_line as i64),
                    pyast::pyint(py, at as i64),
                )?;
            }
            None => {
                return make_err(
                    py,
                    &format!(
                        "line {document_line} of the answer is from {}, and the \
                         source document has no section for it",
                        repr_str(py, &module)
                    ),
                )
            }
        }
    }
    let value = PyTuple::new(
        py,
        [pyast::pystr(py, &document), answer.into_any().unbind()],
    )?;
    make_ok(py, value.into_any().unbind())
}

/// The source document's line a module line is, or None when it is not one.
///
/// The places a module can come from are the loader's, in its order: the
/// document beside its directory first (a section that is a document of its
/// own), then the file at its dotted path, then one file named with its dots.
fn source_line_of(
    py: Python<'_>,
    markdown_module: &Bound<'_, PyModule>,
    places: &HashMap<(String, usize), usize>,
    texts: &HashMap<String, String>,
    module: &str,
    module_line: usize,
) -> PyResult<Option<usize>> {
    let parts: Vec<&str> = module.split('.').collect();
    if parts.len() > 1 {
        let outer = format!("{}.viba.md", parts[..parts.len() - 1].join("/"));
        if let Some(body) = texts.get(&outer) {
            let inner: Vec<(String, Vec<(String, usize)>)> =
                match markdown_module.getattr("lines_of")?.call1((body.as_str(),)) {
                    Ok(value) => value.extract()?,
                    Err(error) => {
                        if error.is_instance_of::<PySyntaxError>(py) {
                            return Ok(None);
                        }
                        return Err(error);
                    }
                };
            let section = format!("{}.viba", parts[parts.len() - 1]);
            let Some((_name, lines)) = inner.iter().find(|(name, _)| name == &section) else {
                return Ok(None);
            };
            if module_line < 1 || module_line > lines.len() {
                return Ok(None);
            }
            return Ok(places.get(&(outer, lines[module_line - 1].1)).copied());
        }
    }
    for section in [
        format!("{}.viba", parts.join("/")),
        format!("{module}.viba"),
    ] {
        if let Some(at) = places.get(&(section, module_line)) {
            return Ok(Some(*at));
        }
    }
    Ok(None)
}

// ----------------------------------------------------------------------
// The Python entry points
// ----------------------------------------------------------------------

#[pyfunction]
fn extract_sub_repo_into_viba_markdown_py(
    py: Python<'_>,
    repo: &Bound<'_, PyAny>,
    viba_path: &Bound<'_, PyAny>,
    extract_source_viba_path: &Bound<'_, PyAny>,
    type_names: &Bound<'_, PyAny>,
    never_eqv: &Bound<'_, PyAny>,
    nil_eqv: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    repo_entry(
        py,
        repo,
        viba_path,
        extract_source_viba_path,
        type_names,
        never_eqv,
        nil_eqv,
    )
}

#[pyfunction]
fn extract_sub_viba_markdown_py(
    py: Python<'_>,
    source_viba_md: &Bound<'_, PyAny>,
    type_names: &Bound<'_, PyAny>,
    never_eqv: &Bound<'_, PyAny>,
    nil_eqv: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    document_entry(py, source_viba_md, type_names, never_eqv, nil_eqv)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(extract_sub_repo_into_viba_markdown_py, m)?)?;
    m.add_function(wrap_pyfunction!(extract_sub_viba_markdown_py, m)?)?;
    Ok(())
}
