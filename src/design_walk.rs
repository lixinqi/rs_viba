//! Small readings of a design's source shared by the walking layers (was viba/design_walk.py).

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::ctx::Ctx;
use crate::pyast;

fn chain_elements(ctx: &Ctx<'_>, node: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let class = ctx.class_name(node)?;
    if matches!(
        class.as_str(),
        "SumChain" | "ProductChain" | "ExponentChain"
    ) {
        let items: Vec<Py<PyAny>> = crate::pyast::field_list(node, "elements")?
            .into_iter()
            .map(|item| item.unbind())
            .collect();
        return ctx.list(items);
    }
    let mut elements: Vec<Py<PyAny>> = Vec::new();
    if class == "Sum" || class == "Product" {
        let mut current = node.clone().unbind();
        loop {
            let bound = current.bind(ctx.py);
            if ctx.class_name(bound)? != class {
                elements.push(current.clone_ref(ctx.py));
                break;
            }
            elements.push(bound.getattr("right")?.unbind());
            current = bound.getattr("left")?.unbind();
        }
        elements.reverse();
        return ctx.list(elements);
    }
    if class == "Exponent" {
        let mut current = node.clone().unbind();
        loop {
            let bound = current.bind(ctx.py);
            if !ctx.is(bound, "Exponent")? {
                elements.push(current.clone_ref(ctx.py));
                break;
            }
            elements.push(bound.getattr("argument")?.unbind());
            current = bound.getattr("result")?.unbind();
        }
        elements.reverse();
        return ctx.list(elements);
    }
    Ok(ctx.py.None())
}

fn call_parts(ctx: &Ctx<'_>, node: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let mut arguments: Vec<Py<PyAny>> = Vec::new();
    let mut current = node.clone().unbind();
    while ctx.is(current.bind(ctx.py), "Partial")? {
        let bound = current.bind(ctx.py);
        arguments.push(bound.getattr("argument")?.unbind());
        current = bound.getattr("function")?.unbind();
    }
    arguments.reverse();
    let arguments = ctx.list(arguments)?;
    let tuple = pyo3::types::PyTuple::new(ctx.py, [current, arguments])?;
    Ok(tuple.into_any().unbind())
}

fn with_params(
    ctx: &Ctx<'_>,
    definition: &Bound<'_, PyAny>,
    bindings: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let dict = bindings.cast::<PyDict>()?.copy()?;
    if let Ok(params) = definition.getattr("generic_params") {
        if !params.is_none() {
            for param in params.try_iter()? {
                let param = param?;
                if !dict.contains(param.clone())? {
                    dict.set_item(param, ctx.py.None())?;
                }
            }
        }
    }
    Ok(dict.into_any().unbind())
}

fn bindings_key(py: Python<'_>, bindings: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let mut parts: Vec<(String, String)> = Vec::new();
    let dict = bindings.cast::<PyDict>()?;
    for (key, value) in dict.iter() {
        let name: String = key.extract()?;
        let text = if value.is_none() {
            "placeholder".to_string()
        } else {
            // A bound value is (node, module): the module's file_path and the node's dump.
            let module = value.get_item(1)?;
            let file_path: String = module
                .getattr("file_path")
                .ok()
                .and_then(|p| p.extract().ok())
                .unwrap_or_default();
            let node = value.get_item(0)?;
            let dump = py
                .import("rs_viba.viba_ast")?
                .getattr("dump")?
                .call1((node,))?
                .extract::<String>()?;
            format!("{},{}", file_path, dump)
        };
        parts.push((name, text));
    }
    parts.sort();
    let out = pyo3::types::PyList::empty(py);
    for (name, text) in parts {
        let tuple = pyo3::types::PyTuple::new(
            py,
            [
                crate::pyast::pystr(py, &name),
                crate::pyast::pystr(py, &text),
            ],
        )?;
        out.append(tuple)?;
    }
    let tuple = pyo3::types::PyTuple::new(py, out.iter())?;
    Ok(tuple.into_any().unbind())
}

#[pyfunction]
fn chain_elements_py<'py>(py: Python<'py>, node: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
    let ctx = Ctx::new(py)?;
    chain_elements(&ctx, node)
}

#[pyfunction]
fn call_parts_py<'py>(py: Python<'py>, node: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
    let ctx = Ctx::new(py)?;
    call_parts(&ctx, node)
}

#[pyfunction]
fn split_type_name(name: &str) -> (String, String) {
    match name.rfind('.') {
        Some(index) if index > 0 && index + 1 < name.len() => {
            (name[..index].to_string(), name[index + 1..].to_string())
        }
        _ => (String::new(), name.to_string()),
    }
}

#[pyfunction]
fn is_function<'py>(definition: &Bound<'py, PyAny>) -> PyResult<bool> {
    let body = definition.getattr("body")?;
    let name = pyast::class_name(&body)?;
    Ok(matches!(
        name.as_str(),
        "Partial" | "Exponent" | "ExponentChain"
    ))
}

#[pyfunction]
fn is_definition<'py>(node: &Bound<'py, PyAny>) -> PyResult<bool> {
    let name = pyast::class_name(node)?;
    Ok(matches!(
        name.as_str(),
        "TypeDefinition" | "GenericDefinition"
    ))
}

#[pyfunction]
fn with_params_py<'py>(
    py: Python<'py>,
    definition: &Bound<'py, PyAny>,
    bindings: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    let ctx = Ctx::new(py)?;
    with_params(&ctx, definition, bindings)
}

#[pyfunction]
fn bindings_key_py<'py>(py: Python<'py>, bindings: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
    bindings_key(py, bindings)
}

#[pyfunction]
fn longest_import(imports: &Bound<'_, PyAny>, name: &str) -> PyResult<Py<PyAny>> {
    let dict = imports.cast::<PyDict>()?;
    let mut locals: Vec<String> = Vec::new();
    for (key, _) in dict.iter() {
        locals.push(key.extract()?);
    }
    locals.sort_by_key(|local| std::cmp::Reverse(local.len()));
    for local in locals {
        if name == local || name.starts_with(&(local.clone() + ".")) {
            return Ok(pyo3::types::PyString::new(imports.py(), &local)
                .into_any()
                .unbind());
        }
    }
    Ok(imports.py().None())
}

fn is_under(path: &str, root: &str) -> bool {
    let path = std::path::Path::new(path);
    let root = std::path::Path::new(root);
    path.canonicalize()
        .ok()
        .map(|p| p.starts_with(root))
        .unwrap_or_else(|| path.starts_with(root))
}

#[pyfunction]
fn is_under_py(path: &Bound<'_, PyAny>, root: &Bound<'_, PyAny>) -> PyResult<bool> {
    let path: String = path.str()?.extract()?;
    let root: String = root.str()?.extract()?;
    let root_canonical = std::path::Path::new(&root)
        .canonicalize()
        .unwrap_or_else(|_| std::path::PathBuf::from(&root));
    Ok(is_under(&path, root_canonical.to_string_lossy().as_ref()))
}

#[pyfunction]
fn under_builtin<'py>(py: Python<'py>, file_path: &Bound<'py, PyAny>) -> PyResult<bool> {
    let path: String = file_path.str()?.extract()?;
    let builtin: String = PyModule::import(py, "rs_viba.type")?
        .getattr("BUILTIN_DIR")?
        .str()?
        .extract()?;
    let root = std::path::Path::new(&builtin)
        .canonicalize()
        .unwrap_or_else(|_| std::path::PathBuf::from(&builtin));
    Ok(is_under(&path, root.to_string_lossy().as_ref()))
}

#[pyfunction]
fn module_label<'py>(module: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
    if let Ok(file_path) = module.getattr("file_path") {
        if !file_path.is_none() {
            return Ok(file_path.unbind());
        }
    }
    if let Ok(name) = module.getattr("name") {
        if !name.is_none() {
            return Ok(name.unbind());
        }
    }
    Ok(pyo3::types::PyString::new(module.py(), "?")
        .into_any()
        .unbind())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(chain_elements_py, m)?)?;
    m.add_function(wrap_pyfunction!(call_parts_py, m)?)?;
    m.add_function(wrap_pyfunction!(split_type_name, m)?)?;
    m.add_function(wrap_pyfunction!(is_function, m)?)?;
    m.add_function(wrap_pyfunction!(is_definition, m)?)?;
    m.add_function(wrap_pyfunction!(with_params_py, m)?)?;
    m.add_function(wrap_pyfunction!(bindings_key_py, m)?)?;
    m.add_function(wrap_pyfunction!(longest_import, m)?)?;
    m.add_function(wrap_pyfunction!(is_under_py, m)?)?;
    m.add_function(wrap_pyfunction!(under_builtin, m)?)?;
    m.add_function(wrap_pyfunction!(module_label, m)?)?;
    Ok(())
}
