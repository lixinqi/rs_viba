//! Native pattern data classes, exported through rs_viba.pattern.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule, PyString, PyType};

use crate::pyast;

#[pyclass(name = "PatternFile", module = "rs_viba.pattern")]
pub struct PatternFile {
    #[pyo3(get, set)]
    order: i64,
    #[pyo3(get, set)]
    path: String,
    #[pyo3(get, set)]
    name: String,
    #[pyo3(get, set)]
    declared: Option<i64>,
    #[pyo3(get, set)]
    module: Py<PyAny>,
    #[pyo3(get, set)]
    patterns: Py<PyAny>,
    #[pyo3(get, set)]
    _load: Py<PyAny>,
    #[pyo3(get, set)]
    _problem: Py<PyAny>,
    #[pyo3(get, set)]
    _taken: bool,
    #[pyo3(get, set)]
    _checked: bool,
}

#[pymethods]
impl PatternFile {
    #[new]
    #[pyo3(signature = (order, path, name, declared, load))]
    fn new(
        py: Python<'_>,
        order: i64,
        path: String,
        name: String,
        declared: Option<i64>,
        load: Py<PyAny>,
    ) -> Self {
        PatternFile {
            order,
            path,
            name,
            declared,
            module: py.None(),
            patterns: py.None(),
            _load: load,
            _problem: py.None(),
            _taken: false,
            _checked: false,
        }
    }

    #[classmethod]
    #[pyo3(signature = (order, path, name, declared, module, patterns))]
    fn ready(
        cls: &Bound<'_, PyType>,
        py: Python<'_>,
        order: i64,
        path: String,
        name: String,
        declared: Option<i64>,
        module: Py<PyAny>,
        patterns: Py<PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let made = cls.call1((order, path, name, declared, py.None()))?;
        made.setattr("module", module)?;
        let values: Vec<Bound<'_, PyAny>> = patterns
            .bind(py)
            .try_iter()?
            .collect::<PyResult<Vec<_>>>()?;
        made.setattr("patterns", PyList::new(py, values)?)?;
        made.setattr("_taken", true)?;
        Ok(made.unbind())
    }

    fn take(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let object: Py<PatternFile> = slf.into();
        Ok(py
            .import("rs_viba.pattern")?
            .getattr("_pattern_file_take")?
            .call1((object.bind(py),))?
            .unbind())
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let state = if self.patterns.bind(py).is_none() {
            "not taken".to_string()
        } else {
            format!("{} params", self.patterns.bind(py).len()?)
        };
        Ok(format!(
            "PatternFile({}, {}, {})",
            self.order,
            pyast::repr_of(py, &PyString::new(py, &self.path).into_any()),
            state
        ))
    }
}

#[pyclass(name = "Choice", module = "rs_viba.pattern")]
pub struct Choice {
    #[pyo3(get, set)]
    entry: Py<PyAny>,
    #[pyo3(get, set)]
    body: Py<PyAny>,
    #[pyo3(get, set)]
    module: Py<PyAny>,
    #[pyo3(get, set)]
    bindings: Py<PyAny>,
}

#[pymethods]
impl Choice {
    #[new]
    fn new(
        py: Python<'_>,
        entry: Py<PyAny>,
        body: Py<PyAny>,
        module: Py<PyAny>,
        bindings: Py<PyAny>,
    ) -> PyResult<Self> {
        let copied = PyDict::new(py);
        copied.call_method1("update", (bindings.bind(py),))?;
        Ok(Choice {
            entry,
            body,
            module,
            bindings: copied.into_any().unbind(),
        })
    }

    fn env_get(&self, py: Python<'_>, name: &str) -> PyResult<Py<PyAny>> {
        let type_module = py.import("rs_viba.type")?;
        if self.bindings.bind(py).contains(name)? {
            let bound = self.bindings.bind(py).get_item(name)?;
            Ok(type_module.getattr("Ok")?.call1((bound,))?.unbind())
        } else {
            Ok(type_module
                .getattr("Err")?
                .call1((format!(
                    "unbound parameter {}",
                    pyast::repr_of(py, &PyString::new(py, name).into_any())
                ),))?
                .unbind())
        }
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let path = self.entry.bind(py).getattr("path")?;
        let sorted = py
            .import("builtins")?
            .getattr("sorted")?
            .call1((self.bindings.bind(py),))?;
        Ok(format!(
            "Choice({}, {})",
            pyast::repr_of(py, &path),
            pyast::repr_of(py, &sorted)
        ))
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PatternFile>()?;
    m.add_class::<Choice>()?;
    Ok(())
}
