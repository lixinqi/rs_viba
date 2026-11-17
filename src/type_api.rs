//! Native Result and Frame classes, exported through rs_viba.type.

use pyo3::prelude::*;
use pyo3::types::PyModule;

use crate::pyast;

#[pyclass(name = "Ok", module = "rs_viba.type")]
pub struct Ok {
    #[pyo3(get, set)]
    ok_value: Py<PyAny>,
}

#[pymethods]
impl Ok {
    #[new]
    fn new(ok_value: Py<PyAny>) -> Self {
        Ok { ok_value }
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        format!("Ok({})", pyast::repr_of(py, self.ok_value.bind(py)))
    }
}

#[pyclass(name = "Err", module = "rs_viba.type", subclass, dict)]
pub struct Err {
    #[pyo3(get, set)]
    msg: String,
}

#[pymethods]
impl Err {
    #[new]
    #[pyo3(signature = (*args, **kwargs))]
    fn new(
        args: &Bound<'_, pyo3::types::PyTuple>,
        kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<Self> {
        let mut msg = String::new();
        if let Some(first) = args.get_item(0).ok() {
            if let Ok(text) = first.extract::<String>() {
                msg = text;
            }
        }
        if msg.is_empty() {
            if let Some(kwargs) = kwargs {
                if let Some(value) = kwargs.get_item("msg")? {
                    msg = value.extract()?;
                }
            }
        }
        Ok(Err { msg })
    }

    #[pyo3(signature = (*args, **kwargs))]
    fn __init__(
        &mut self,
        args: &Bound<'_, pyo3::types::PyTuple>,
        kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<()> {
        if let Ok(first) = args.get_item(0) {
            if let Ok(msg) = first.extract::<String>() {
                self.msg = msg;
                return Ok(());
            }
        }
        if let Some(kwargs) = kwargs {
            if let Some(msg) = kwargs.get_item("msg")? {
                self.msg = msg.extract()?;
            }
        }
        Ok(())
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        format!(
            "Err({})",
            pyast::repr_of(py, &pyo3::types::PyString::new(py, &self.msg).into_any())
        )
    }
}

#[pyclass(name = "Frame", module = "rs_viba.type")]
pub struct Frame {
    #[pyo3(get, set)]
    file_path: String,
    #[pyo3(get, set)]
    lineno: i64,
}

#[pymethods]
impl Frame {
    #[new]
    fn new(file_path: String, lineno: i64) -> Self {
        Frame { file_path, lineno }
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        format!(
            "Frame({}, {})",
            pyast::repr_of(
                py,
                &pyo3::types::PyString::new(py, &self.file_path).into_any()
            ),
            self.lineno
        )
    }

    fn __eq__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        if pyast::class_name(other)? != "Frame" {
            return Ok(false);
        }
        let file_path: String = other.getattr("file_path")?.extract()?;
        let lineno: i64 = other.getattr("lineno")?.extract()?;
        Ok(self.file_path == file_path && self.lineno == lineno)
    }

    fn __hash__(&self, py: Python<'_>) -> PyResult<isize> {
        let key = pyo3::types::PyTuple::new(
            py,
            [
                pyo3::types::PyString::new(py, &self.file_path).into_any(),
                self.lineno.into_pyobject(py)?.into_any(),
            ],
        )?;
        py.import("builtins")?
            .getattr("hash")?
            .call1((key,))?
            .extract()
    }
}

#[pyclass(extends=Err, subclass, dict, name = "InterpretError", module = "rs_viba.type")]
pub struct InterpretError;

#[pymethods]
impl InterpretError {
    #[new]
    #[pyo3(signature = (msg = String::new()))]
    fn new(msg: String) -> PyClassInitializer<Self> {
        PyClassInitializer::from(Err { msg }).add_subclass(InterpretError)
    }
}

#[pyclass(extends=InterpretError, name = "VibaProgramErr", module = "rs_viba.type")]
pub struct VibaProgramErr {
    #[pyo3(get, set)]
    stack: Py<PyAny>,
}

#[pymethods]
impl VibaProgramErr {
    #[new]
    #[pyo3(signature = (msg, stack = None))]
    fn new(py: Python<'_>, msg: String, stack: Option<Py<PyAny>>) -> PyClassInitializer<Self> {
        let stack = match stack {
            Some(stack) => stack,
            None => pyo3::types::PyList::empty(py).into_any().unbind(),
        };
        PyClassInitializer::from(Err { msg })
            .add_subclass(InterpretError)
            .add_subclass(VibaProgramErr { stack })
    }

    fn __repr__(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<String> {
        let object: Py<VibaProgramErr> = slf.into();
        let msg: String = object.bind(py).getattr("msg")?.extract()?;
        Ok(format!(
            "VibaProgramErr({})",
            crate::pyast::repr_of(py, &pyo3::types::PyString::new(py, &msg).into_any())
        ))
    }
}

#[pyclass(extends=InterpretError, name = "UnderlyingOpErr", module = "rs_viba.type")]
pub struct UnderlyingOpErr {
    #[pyo3(get, set)]
    module_path: String,
    #[pyo3(get, set)]
    full_qualified_func_name: String,
    #[pyo3(get, set)]
    call: Py<PyAny>,
    #[pyo3(get, set)]
    tag: String,
}

#[pymethods]
impl UnderlyingOpErr {
    #[new]
    #[pyo3(signature = (msg = String::new(), module_path = String::new(), full_qualified_func_name = String::new(), call = None, tag = None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        msg: String,
        module_path: String,
        full_qualified_func_name: String,
        call: Option<Py<PyAny>>,
        tag: Option<String>,
    ) -> PyClassInitializer<Self> {
        PyClassInitializer::from(Err { msg })
            .add_subclass(InterpretError)
            .add_subclass(UnderlyingOpErr {
                module_path,
                full_qualified_func_name,
                call: call.unwrap_or_else(|| py.None()),
                tag: tag.unwrap_or_else(|| "$underlying_viba_op_err".to_string()),
            })
    }

    fn __repr__(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<String> {
        let module_path = slf.module_path.clone();
        let full_qualified_func_name = slf.full_qualified_func_name.clone();
        let tag = slf.tag.clone();
        let object: Py<UnderlyingOpErr> = slf.into();
        let message: String = object.bind(py).getattr("msg")?.extract()?;
        Ok(format!(
            "UnderlyingOpErr({}, {}, {}, tag={})",
            crate::pyast::repr_of(py, &pyo3::types::PyString::new(py, &message).into_any()),
            crate::pyast::repr_of(py, &pyo3::types::PyString::new(py, &module_path).into_any()),
            crate::pyast::repr_of(
                py,
                &pyo3::types::PyString::new(py, &full_qualified_func_name).into_any()
            ),
            crate::pyast::repr_of(py, &pyo3::types::PyString::new(py, &tag).into_any())
        ))
    }
}

#[pyclass(extends=InterpretError, name = "EnvironmentApiInvalidArgumentErr", module = "rs_viba.type")]
pub struct EnvironmentApiInvalidArgumentErr {
    #[pyo3(get, set)]
    api_name: String,
    #[pyo3(get, set)]
    arguments: Py<PyAny>,
}

#[pymethods]
impl EnvironmentApiInvalidArgumentErr {
    #[new]
    fn new(msg: String, api_name: String, arguments: Py<PyAny>) -> PyClassInitializer<Self> {
        PyClassInitializer::from(Err { msg })
            .add_subclass(InterpretError)
            .add_subclass(EnvironmentApiInvalidArgumentErr {
                api_name,
                arguments,
            })
    }

    fn __str__(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<String> {
        slf.into_pyobject(py)?.getattr("msg")?.extract()
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Ok>()?;
    m.add_class::<Err>()?;
    m.add_class::<Frame>()?;
    m.add_class::<InterpretError>()?;
    m.add_class::<VibaProgramErr>()?;
    m.add_class::<UnderlyingOpErr>()?;
    m.add_class::<EnvironmentApiInvalidArgumentErr>()?;
    Ok(())
}
