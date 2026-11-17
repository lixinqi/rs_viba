//! Shared helpers for working with the Python-visible AST classes.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString};

pub struct Nodes<'py> {
    pub module: Bound<'py, PyModule>,
}

impl<'py> Nodes<'py> {
    pub fn new(py: Python<'py>) -> PyResult<Self> {
        Ok(Self {
            module: PyModule::import(py, "rs_viba.viba_ast.nodes")?,
        })
    }

    pub fn class(&self, name: &str) -> PyResult<Bound<'py, PyAny>> {
        self.module.getattr(name)
    }

    pub fn make(&self, name: &str, fields: &[(&str, Py<PyAny>)]) -> PyResult<Py<PyAny>> {
        let py = self.module.py();
        let cls = self.class(name)?;
        let kwargs = PyDict::new(py);
        for (key, value) in fields {
            kwargs.set_item(key, value)?;
        }
        Ok(cls.call((), Some(&kwargs))?.unbind())
    }

    pub fn make0(&self, name: &str) -> PyResult<Py<PyAny>> {
        Ok(self.class(name)?.call0()?.unbind())
    }
}

pub fn class_name(obj: &Bound<'_, PyAny>) -> PyResult<String> {
    Ok(obj.get_type().name()?.to_string())
}

pub fn field<'py>(node: &Bound<'py, PyAny>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    node.getattr(name)
}

pub fn field_string(node: &Bound<'_, PyAny>, name: &str) -> PyResult<String> {
    node.getattr(name)?.extract()
}

pub fn field_opt_string(node: &Bound<'_, PyAny>, name: &str) -> PyResult<Option<String>> {
    let value = node.getattr(name)?;
    if value.is_none() {
        Ok(None)
    } else {
        Ok(Some(value.extract()?))
    }
}

pub fn field_list<'py>(node: &Bound<'py, PyAny>, name: &str) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let value = node.getattr(name)?;
    if value.is_none() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for item in value.try_iter()? {
        out.push(item?);
    }
    Ok(out)
}

pub fn pystr(py: Python<'_>, text: &str) -> Py<PyAny> {
    PyString::new(py, text).into_any().unbind()
}

pub fn pybool(py: Python<'_>, value: bool) -> Py<PyAny> {
    value
        .into_pyobject(py)
        .unwrap()
        .to_owned()
        .into_any()
        .unbind()
}

pub fn pyint(py: Python<'_>, value: i64) -> Py<PyAny> {
    value
        .into_pyobject(py)
        .unwrap()
        .to_owned()
        .into_any()
        .unbind()
}

pub fn pynone(py: Python<'_>) -> Py<PyAny> {
    py.None()
}

pub fn pylist(py: Python<'_>, items: Vec<Py<PyAny>>) -> PyResult<Py<PyAny>> {
    let list = PyList::empty(py);
    for item in items {
        list.append(item)?;
    }
    Ok(list.into_any().unbind())
}

pub fn set_lineno(py: Python<'_>, node: &Bound<'_, PyAny>, line: usize) -> PyResult<()> {
    node.setattr("lineno", pyint(py, line as i64))?;
    Ok(())
}

pub fn repr_of(py: Python<'_>, value: &Bound<'_, PyAny>) -> String {
    match py
        .import("builtins")
        .and_then(|b| b.getattr("repr"))
        .and_then(|r| r.call1((value,)))
    {
        Ok(text) => text.extract().unwrap_or_else(|_| "<unrepr>".to_string()),
        Err(_) => "<unrepr>".to_string(),
    }
}
