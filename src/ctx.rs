//! A parse/design context: the Python interpreter's GIL and the AST classes.

use pyo3::prelude::*;

use crate::pyast::Nodes;

pub struct Ctx<'py> {
    pub py: Python<'py>,
    pub nodes: Nodes<'py>,
}

impl<'py> Ctx<'py> {
    pub fn new(py: Python<'py>) -> PyResult<Self> {
        Ok(Self {
            py,
            nodes: Nodes::new(py)?,
        })
    }

    pub fn class_name(&self, node: &Bound<'_, PyAny>) -> PyResult<String> {
        crate::pyast::class_name(node)
    }

    pub fn is(&self, node: &Bound<'_, PyAny>, name: &str) -> PyResult<bool> {
        Ok(self.class_name(node)? == name)
    }

    pub fn field<'a>(
        &self,
        node: &'a Bound<'py, PyAny>,
        name: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        node.getattr(name)
    }

    pub fn make(&self, name: &str, fields: &[(&str, Py<PyAny>)]) -> PyResult<Py<PyAny>> {
        self.nodes.make(name, fields)
    }

    pub fn make0(&self, name: &str) -> PyResult<Py<PyAny>> {
        self.nodes.make0(name)
    }

    pub fn pystr(&self, text: &str) -> Py<PyAny> {
        crate::pyast::pystr(self.py, text)
    }

    pub fn list(&self, items: Vec<Py<PyAny>>) -> PyResult<Py<PyAny>> {
        crate::pyast::pylist(self.py, items)
    }
}
