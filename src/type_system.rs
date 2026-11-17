//! type: Result values, the Type model and module lookup (was viba/type.py).

use std::collections::HashSet;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule, PyString};

use crate::parser;
use crate::pyast;

fn type_module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    PyModule::import(py, "rs_viba.type")
}

fn class<'py>(module: &Bound<'py, PyModule>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    module.getattr(name)
}

fn class_name(value: &Bound<'_, PyAny>) -> PyResult<String> {
    pyast::class_name(value)
}

fn is_class(value: &Bound<'_, PyAny>, name: &str) -> PyResult<bool> {
    Ok(class_name(value)? == name)
}

fn pyrepr(py: Python<'_>, value: &Bound<'_, PyAny>) -> String {
    pyast::repr_of(py, value)
}

pub fn ok(py: Python<'_>, value: Py<PyAny>) -> PyResult<Py<PyAny>> {
    let module = type_module(py)?;
    Ok(class(&module, "Ok")?.call1((value,))?.unbind())
}

pub fn err(py: Python<'_>, message: &str) -> PyResult<Py<PyAny>> {
    let module = type_module(py)?;
    Ok(class(&module, "Err")?
        .call1((PyString::new(py, message),))?
        .unbind())
}

pub fn is_err(value: &Bound<'_, PyAny>) -> PyResult<bool> {
    let py = value.py();
    let module = type_module(py)?;
    value.is_instance(&class(&module, "Err")?)
}

fn ok_value<'py>(value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    value.getattr("ok_value")
}

fn err_message(value: &Bound<'_, PyAny>) -> PyResult<String> {
    value.getattr("msg")?.extract()
}

fn make_ast_node_type(
    py: Python<'_>,
    node: &Bound<'_, PyAny>,
    module: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let types = type_module(py)?;
    Ok(class(&types, "AstNodeType")?
        .call1((node, module))?
        .unbind())
}

pub fn builtin_module_py(py: Python<'_>) -> PyResult<Py<PyAny>> {
    Ok(builtin_module(py)?.unbind())
}

fn builtin_module<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
    type_module(py)?.getattr("BUILTIN_MODULE")
}

fn classes_of_basic(py: Python<'_>, name: &str) -> Option<&'static str> {
    Some(match name {
        "bool" => "BoolType",
        "int" => "IntType",
        "float" => "FloatType",
        "str" => "StrType",
        "int8_t" => "Int8Type",
        "uint8_t" => "UInt8Type",
        "int16_t" => "Int16Type",
        "uint16_t" => "UInt16Type",
        "int32_t" => "Int32Type",
        "uint32_t" => "UInt32Type",
        "int64_t" => "Int64Type",
        "uint64_t" => "UInt64Type",
        "nil" | "void" | "None" | "Object" => "NilType",
        "never" | "Oneof" => "NeverType",
        _ => return None,
    })
}

fn generic_names(name: &str) -> bool {
    matches!(
        name,
        "List" | "list" | "set" | "dict" | "ListLiteral" | "SetLiteral" | "DictLiteral" | "tagged"
    )
}

fn load_library(py: Python<'_>, module: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let cached = module.getattr("_library")?;
    if !cached.is_none() {
        return Ok(cached.unbind());
    }
    let file_path: String = module.getattr("file_path")?.extract()?;
    let source = std::fs::read_to_string(&file_path)?;
    let tree = parser::parse(py, &source)?;
    let tree_bound = tree.bind(py);
    let body = pyast::field_list(tree_bound, "body")?;
    let library = PyDict::new(py);
    for item in body {
        let name = class_name(&item)?;
        if name == "TypeDefinition" || name == "GenericDefinition" {
            let definition_name: String = item.getattr("name")?.extract()?;
            library.set_item(definition_name, item)?;
        }
    }
    module.setattr("_library", &library)?;
    Ok(library.into_any().unbind())
}

fn load_operators(py: Python<'_>, module: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let cached = module.getattr("_operators")?;
    if !cached.is_none() {
        return Ok(cached.unbind());
    }
    let library = load_library(py, module)?.into_bound(py);
    let library = library.cast::<PyDict>()?;
    let operators = PyDict::new(py);
    if let Some(definition) = library.get_item("builtin")? {
        let body = definition.getattr("body")?;
        let mut factors = Vec::new();
        product_factors(&body, &mut factors)?;
        for factor in factors {
            let factor = factor.bind(py);
            if class_name(factor)? == "Tagged" {
                let tag: String = factor.getattr("tag")?.extract()?;
                let name = tag.strip_prefix('$').unwrap_or(&tag);
                let piece = factor.getattr("type")?;
                operators.set_item(name, piece)?;
            }
        }
    }
    module.setattr("_operators", &operators)?;
    Ok(operators.into_any().unbind())
}

fn product_factors(node: &Bound<'_, PyAny>, out: &mut Vec<Py<PyAny>>) -> PyResult<()> {
    let name = class_name(node)?;
    if name == "Product" {
        let left = node.getattr("left")?;
        let right = node.getattr("right")?;
        product_factors(&left, out)?;
        product_factors(&right, out)?;
    } else if name == "ProductChain" {
        for item in pyast::field_list(node, "elements")? {
            out.push(item.unbind());
        }
    } else {
        out.push(node.clone().unbind());
    }
    Ok(())
}

pub fn builtin_lookup(
    py: Python<'_>,
    module: &Bound<'_, PyAny>,
    type_name: &str,
) -> PyResult<Py<PyAny>> {
    if let Some(class_name) = classes_of_basic(py, type_name) {
        let types = type_module(py)?;
        let value = class(&types, class_name)?.call0()?.unbind();
        return ok(py, value);
    }
    if generic_names(type_name) {
        let types = type_module(py)?;
        let value = PyString::new(py, type_name);
        let built = class(&types, "BuiltinGenericType")?
            .call1((value,))?
            .unbind();
        return ok(py, built);
    }
    let library_py = load_library(py, module)?.into_bound(py);
    let library = library_py.cast::<PyDict>()?;
    if let Some(definition) = library.get_item(type_name)? {
        let builtin = builtin_module(py)?;
        let value = make_ast_node_type(py, &definition, &builtin)?;
        return ok(py, value);
    }
    let operators_py = load_operators(py, module)?.into_bound(py);
    let operators = operators_py.cast::<PyDict>()?;
    if let Some(piece) = operators.get_item(type_name)? {
        let builtin = builtin_module(py)?;
        let value = make_ast_node_type(py, &piece, &builtin)?;
        return ok(py, value);
    }
    err(
        py,
        &format!(
            "no built-in type named {}",
            pyrepr(py, &PyString::new(py, type_name).as_any())
        ),
    )
}

pub fn builtin_operator(
    py: Python<'_>,
    module: &Bound<'_, PyAny>,
    name: &str,
) -> PyResult<Py<PyAny>> {
    let operators_py = load_operators(py, module)?.into_bound(py);
    let operators = operators_py.cast::<PyDict>()?;
    match operators.get_item(name)? {
        Some(value) => Ok(value.unbind()),
        None => Ok(py.None()),
    }
}

pub(crate) fn custom_lookup_local_pub<'py>(
    py: Python<'py>,
    module: &Bound<'py, PyAny>,
    type_name: &str,
) -> PyResult<Py<PyAny>> {
    custom_lookup_local(py, module, type_name)
}

fn custom_lookup_local(
    py: Python<'_>,
    module: &Bound<'_, PyAny>,
    type_name: &str,
) -> PyResult<Py<PyAny>> {
    let tree = module.getattr("module")?;
    for node in pyast::field_list(&tree, "body")? {
        let name = class_name(&node)?;
        if name == "TypeDefinition" || name == "GenericDefinition" {
            let definition_name: String = node.getattr("name")?.extract()?;
            if definition_name == type_name {
                let value = make_ast_node_type(py, &node, module)?;
                return ok(py, value);
            }
        }
    }
    lookup_imported(py, module, type_name)
}

fn lookup_imported(
    py: Python<'_>,
    module: &Bound<'_, PyAny>,
    type_name: &str,
) -> PyResult<Py<PyAny>> {
    let imports = module.getattr("imports")?;
    let imports = imports.cast::<PyDict>()?;
    let parts: Vec<&str> = type_name.split('.').collect();
    for cut in (1..parts.len()).rev() {
        let prefix = parts[..cut].join(".");
        let module_name = match imports.get_item(prefix.as_str())? {
            Some(value) => value,
            None => continue,
        };
        let imported = module
            .getattr("module_environment")?
            .call1((&module_name,))?;
        if is_err(&imported)? {
            return err(
                py,
                &format!(
                    "{} names {}: {}",
                    pyrepr(py, &PyString::new(py, &prefix).as_any()),
                    pyrepr(py, &module_name),
                    err_message(&imported)?
                ),
            );
        }
        let rest = parts[cut..].join(".");
        let imported_module = ok_value(&imported)?;
        if !is_class(&imported_module, "CustomModuleType")? {
            return err(
                py,
                &format!(
                    "{} is a generic: it answers an application ({rest}[T, ...]), not the bare name {type_name}",
                    pyrepr(py, &module_name)
                ),
            );
        }
        return custom_lookup_local(py, &imported_module, &rest);
    }
    if parts.len() == 1 {
        if let Some(module_name) = imports.get_item(parts[0])? {
            let imported = module
                .getattr("module_environment")?
                .call1((&module_name,))?;
            if is_err(&imported)? {
                return err(
                    py,
                    &format!(
                        "{} names {}: {}",
                        pyrepr(py, &PyString::new(py, parts[0]).as_any()),
                        pyrepr(py, &module_name),
                        err_message(&imported)?
                    ),
                );
            }
            let imported_module = ok_value(&imported)?;
            if !is_class(&imported_module, "CustomModuleType")? {
                return err(
                    py,
                    &format!(
                        "{} is a generic: it answers an application ({}[T, ...]), not the bare name {type_name}",
                        pyrepr(py, &module_name),
                        parts[0]
                    ),
                );
            }
            return err(
                py,
                &format!(
                    "{} names the module {}: a module is no type",
                    pyrepr(py, &PyString::new(py, type_name).as_any()),
                    pyrepr(py, &module_name)
                ),
            );
        }
    }
    err(
        py,
        &format!(
            "no type named {} in module",
            pyrepr(py, &PyString::new(py, type_name).as_any())
        ),
    )
}

fn module_get_type_seen(
    py: Python<'_>,
    module: &Bound<'_, PyAny>,
    type_name: &str,
    seen: &mut HashSet<usize>,
) -> PyResult<Py<PyAny>> {
    let module_name = class_name(module)?;
    if module_name == "BuiltinModuleType" {
        return Ok(module.getattr("lookup")?.call1((type_name,))?.unbind());
    }
    if module_name != "CustomModuleType" {
        return err(
            py,
            &format!(
                "no type named {}: {} is a generic, and a generic is applied with [T, ...]",
                pyrepr(py, &PyString::new(py, type_name).as_any()),
                pyrepr(py, module)
            ),
        );
    }
    let local = custom_lookup_local(py, module, type_name)?;
    if !is_err(local.bind(py))? {
        return Ok(local);
    }
    let via_env = module.getattr("module_environment")?.call1((type_name,))?;
    if !is_err(&via_env)? {
        let candidate = ok_value(&via_env)?;
        let tree = candidate.getattr("module")?;
        let key = tree.as_ptr() as usize;
        if !seen.contains(&key) {
            let current = module.getattr("module")?;
            seen.insert(current.as_ptr() as usize);
            return module_get_type_seen(py, &candidate, type_name, seen);
        }
    }
    let builtin = builtin_module(py)?;
    let builtin_answer = builtin.getattr("lookup")?.call1((type_name,))?;
    if !is_err(&builtin_answer)? {
        return Ok(builtin_answer.unbind());
    }
    let note = if is_err(&via_env)? {
        err_message(&via_env)?
    } else {
        "it names a module already met".to_string()
    };
    err(
        py,
        &format!(
            "type {} unresolved: {}; {}",
            pyrepr(py, &PyString::new(py, type_name).as_any()),
            err_message(local.bind(py))?,
            note
        ),
    )
}

pub fn module_get_type(
    py: Python<'_>,
    module: &Bound<'_, PyAny>,
    type_name: &str,
    _seen: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let mut seen = HashSet::new();
    module_get_type_seen(py, module, type_name, &mut seen)
}

pub fn custom_module(
    py: Python<'_>,
    source: &str,
    environment: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let tree = parser::parse(py, source)?;
    let types = type_module(py)?;
    let cls = class(&types, "CustomModuleType")?;
    match environment {
        Some(environment) => Ok(cls.call1((tree, environment))?.unbind()),
        None => Ok(cls.call1((tree,))?.unbind()),
    }
}

pub fn entry_type(
    py: Python<'_>,
    source: &str,
    module: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let types = type_module(py)?;
    let module = match module {
        Some(module) => module.clone(),
        None => {
            let empty = parser::parse(py, "")?;
            class(&types, "CustomModuleType")?.call1((empty,))?
        }
    };
    let tree = parser::parse(py, &format!("__entry__ = {}", source))?;
    let body = pyast::field_list(tree.bind(py), "body")?;
    let definition = &body[0];
    let node = definition.getattr("body")?;
    make_ast_node_type(py, &node, &module)
}

pub fn takes_a_product(spelled: &Bound<'_, PyAny>) -> PyResult<bool> {
    let name = class_name(spelled)?;
    if matches!(
        name.as_str(),
        "Any" | "Nil" | "Never" | "Ellipsis" | "Product" | "ProductChain"
    ) {
        return Ok(true);
    }
    if name == "TypeRef" {
        let type_name: String = spelled.getattr("name")?.extract()?;
        return Ok(matches!(type_name.as_str(), "Any" | "Object" | "nil"));
    }
    Ok(false)
}

#[pyfunction]
fn builtin_lookup_py(
    py: Python<'_>,
    module: &Bound<'_, PyAny>,
    type_name: &str,
) -> PyResult<Py<PyAny>> {
    builtin_lookup(py, module, type_name)
}

#[pyfunction]
fn builtin_operator_py(
    py: Python<'_>,
    module: &Bound<'_, PyAny>,
    name: &str,
) -> PyResult<Py<PyAny>> {
    builtin_operator(py, module, name)
}

#[pyfunction]
fn custom_lookup_local_py(
    py: Python<'_>,
    module: &Bound<'_, PyAny>,
    type_name: &str,
) -> PyResult<Py<PyAny>> {
    custom_lookup_local(py, module, type_name)
}

#[pyfunction(signature = (module, type_name, _seen = None))]
fn module_get_type_py(
    py: Python<'_>,
    module: &Bound<'_, PyAny>,
    type_name: &str,
    _seen: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    module_get_type(py, module, type_name, _seen)
}

#[pyfunction(signature = (source, environment = None))]
fn custom_module_py(
    py: Python<'_>,
    source: &str,
    environment: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    custom_module(py, source, environment)
}

#[pyfunction(signature = (source, module = None))]
fn entry_type_py(
    py: Python<'_>,
    source: &str,
    module: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    entry_type(py, source, module)
}

#[pyfunction]
fn takes_a_product_py(spelled: &Bound<'_, PyAny>) -> PyResult<bool> {
    takes_a_product(spelled)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(builtin_lookup_py, m)?)?;
    m.add_function(wrap_pyfunction!(builtin_operator_py, m)?)?;
    m.add_function(wrap_pyfunction!(custom_lookup_local_py, m)?)?;
    m.add_function(wrap_pyfunction!(module_get_type_py, m)?)?;
    m.add_function(wrap_pyfunction!(custom_module_py, m)?)?;
    m.add_function(wrap_pyfunction!(entry_type_py, m)?)?;
    m.add_function(wrap_pyfunction!(takes_a_product_py, m)?)?;
    Ok(())
}
