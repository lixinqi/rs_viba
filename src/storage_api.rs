//! Native storage and compute classes, exported through rs_viba.interpret.

use pyo3::prelude::*;
use pyo3::types::PyModule;

#[pyclass(name = "EnvironmentStorage", module = "rs_viba.interpret", subclass)]
pub struct EnvironmentStorage {
    #[pyo3(get, set)]
    cur_storage_path: String,
    #[pyo3(get, set)]
    sub_storage: Py<PyAny>,
    #[pyo3(get, set)]
    store_root_dir: Py<PyAny>,
    #[pyo3(get, set)]
    parent: Py<PyAny>,
}

#[pymethods]
impl EnvironmentStorage {
    #[new]
    #[pyo3(signature = (cur_storage_path, sub_storage = None, store_root_dir = None, parent = None))]
    fn new(
        py: Python<'_>,
        cur_storage_path: String,
        sub_storage: Option<Py<PyAny>>,
        store_root_dir: Option<Py<PyAny>>,
        parent: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        let dict = pyo3::types::PyDict::new(py);
        if let Some(value) = &sub_storage {
            if !value.bind(py).is_none() {
                dict.call_method1("update", (value.bind(py),))?;
            }
        }
        let store_root_dir = match store_root_dir {
            Some(value) if !value.bind(py).is_none() => value,
            _ => py
                .import("rs_viba.interpret")?
                .getattr("DEFAULT_STORE_ROOT")?
                .unbind(),
        };
        Ok(EnvironmentStorage {
            cur_storage_path,
            sub_storage: dict.into_any().unbind(),
            store_root_dir,
            parent: parent.unwrap_or_else(|| py.None()),
        })
    }

    fn sub(slf: PyRef<'_, Self>, py: Python<'_>, name: &str) -> PyResult<Py<PyAny>> {
        let object: Py<EnvironmentStorage> = slf.into();
        Ok(py
            .import("rs_viba._native")?
            .getattr("storage_sub_py")?
            .call1((object.bind(py), name))?
            .unbind())
    }

    fn tmp(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let object: Py<EnvironmentStorage> = slf.into();
        Ok(py
            .import("rs_viba._native")?
            .getattr("storage_tmp_py")?
            .call1((object.bind(py),))?
            .unbind())
    }

    fn get_text(slf: PyRef<'_, Self>, py: Python<'_>, file_path: &str) -> PyResult<Py<PyAny>> {
        let object: Py<EnvironmentStorage> = slf.into();
        Ok(py
            .import("rs_viba._native")?
            .getattr("storage_get_text_py")?
            .call1((object.bind(py), file_path))?
            .unbind())
    }

    fn put_text(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        file_path: &str,
        content: &str,
    ) -> PyResult<()> {
        let object: Py<EnvironmentStorage> = slf.into();
        py.import("rs_viba._native")?
            .getattr("storage_put_text_py")?
            .call1((object.bind(py), file_path, content))?;
        Ok(())
    }

    fn _store_path(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        file_path: &str,
        root_dir: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let object: Py<EnvironmentStorage> = slf.into();
        let root_dir = root_dir
            .map(|value| value.clone().unbind())
            .unwrap_or_else(|| py.None());
        Ok(py
            .import("rs_viba._native")?
            .getattr("storage_store_path_py")?
            .call1((object.bind(py), file_path, root_dir))?
            .unbind())
    }
}

#[pyclass(name = "EnvironmentCompute", module = "rs_viba.interpret")]
pub struct EnvironmentCompute {
    #[pyo3(get, set)]
    get_func: Py<PyAny>,
}

#[pymethods]
impl EnvironmentCompute {
    #[new]
    fn new(get_func: Py<PyAny>) -> Self {
        EnvironmentCompute { get_func }
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<EnvironmentStorage>()?;
    m.add_class::<EnvironmentCompute>()?;
    Ok(())
}
