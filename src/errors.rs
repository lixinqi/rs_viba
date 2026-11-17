//! Shared error type for the design-side Rust modules.

use pyo3::prelude::*;

#[derive(Debug)]
pub enum VibaError {
    Partial(String),
    Unresolved(String),
    DuplicateTag(String),
    InlineCycle(String),
    Program(String),
    Py(PyErr),
}

pub type VibaResult<T> = Result<T, VibaError>;

impl From<PyErr> for VibaError {
    fn from(error: PyErr) -> Self {
        VibaError::Py(error)
    }
}

impl VibaError {
    pub fn message(&self) -> String {
        match self {
            VibaError::Partial(message)
            | VibaError::Unresolved(message)
            | VibaError::DuplicateTag(message)
            | VibaError::InlineCycle(message)
            | VibaError::Program(message) => message.clone(),
            VibaError::Py(error) => error.to_string(),
        }
    }

    pub fn into_pyerr(self, py: Python<'_>) -> PyErr {
        match self {
            VibaError::Py(error) => error,
            _ => {
                let message = self.message();
                match py.import("rs_viba.type") {
                    Ok(module) => {
                        let name = match self {
                            VibaError::Partial(_) => "PartialError",
                            VibaError::Unresolved(_) => "UnresolvedTypeError",
                            VibaError::DuplicateTag(_) => "DuplicateTagError",
                            VibaError::InlineCycle(_) => "InlineCycleError",
                            _ => "Exception",
                        };
                        match module.getattr(name) {
                            Ok(class) => match class.call1((message,)) {
                                Ok(instance) => match instance.extract::<Py<PyAny>>() {
                                    Ok(instance) => {
                                        PyErr::from_value(instance.bind(py).cast().unwrap().clone())
                                    }
                                    Err(_) => {
                                        pyo3::exceptions::PyRuntimeError::new_err(self.message())
                                    }
                                },
                                Err(_) => pyo3::exceptions::PyRuntimeError::new_err(self.message()),
                            },
                            Err(_) => pyo3::exceptions::PyRuntimeError::new_err(self.message()),
                        }
                    }
                    Err(_) => pyo3::exceptions::PyRuntimeError::new_err(message),
                }
            }
        }
    }
}
