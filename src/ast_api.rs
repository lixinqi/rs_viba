//! Native AST node classes, exported through rs_viba.viba_ast.nodes.
//!
//! A node is one Python object with two pieces: its concrete class, which
//! carries `_fields`, and a map of field values.  The classes are Rust; the
//! values are Python objects, so the parser and the Python API keep the same
//! constructor and attribute behaviour.

use std::collections::HashMap;

use pyo3::exceptions::{PyAttributeError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule, PyString, PyTuple};

#[pyclass(subclass, name = "AST", module = "rs_viba.viba_ast.nodes")]
pub struct AST {
    fields: HashMap<String, Py<PyAny>>,
}

fn field_tuple(py: Python<'_>, names: &[&str]) -> PyResult<Py<PyAny>> {
    Ok(
        PyTuple::new(py, names.iter().map(|name| PyString::new(py, name)))?
            .into_any()
            .unbind(),
    )
}

fn base_new(
    py: Python<'_>,
    class_name: &str,
    names: &[&str],
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<AST> {
    if args.len() > names.len() {
        return Err(PyTypeError::new_err(format!(
            "{} expected at most {} positional arguments, got {}",
            class_name,
            names.len(),
            args.len()
        )));
    }
    let mut fields = HashMap::new();
    for (index, name) in names.iter().enumerate() {
        let value = if index < args.len() {
            Some(args.get_item(index)?)
        } else {
            match kwargs {
                Some(kwargs) => kwargs.get_item(*name)?,
                None => None,
            }
        };
        fields.insert(
            (*name).to_string(),
            value
                .map(|value| value.unbind())
                .unwrap_or_else(|| py.None()),
        );
    }
    if let Some(kwargs) = kwargs {
        let mut unexpected = Vec::new();
        for (key, _) in kwargs.iter() {
            let key: String = key.extract()?;
            if !names.contains(&key.as_str()) {
                unexpected.push(key);
            }
        }
        if !unexpected.is_empty() {
            return Err(PyTypeError::new_err(format!(
                "{} got unexpected keyword argument(s): {}",
                class_name,
                unexpected.join(", ")
            )));
        }
    }
    Ok(AST { fields })
}

#[pymethods]
impl AST {
    #[new]
    #[pyo3(signature = (*args, **kwargs))]
    fn new(
        py: Python<'_>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        base_new(py, "AST", &[], args, kwargs)
    }

    #[classattr]
    fn _fields(py: Python<'_>) -> PyResult<Py<PyAny>> {
        field_tuple(py, &[])
    }

    fn __getattr__(&self, py: Python<'_>, name: &str) -> PyResult<Py<PyAny>> {
        match self.fields.get(name) {
            Some(value) => Ok(value.clone_ref(py)),
            None => Err(PyAttributeError::new_err(name.to_string())),
        }
    }

    fn __setattr__(mut slf: PyRefMut<'_, Self>, name: &str, value: Py<PyAny>) -> PyResult<()> {
        slf.fields.insert(name.to_string(), value);
        Ok(())
    }

    fn __deepcopy__(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        memo: &Bound<'_, PyDict>,
    ) -> PyResult<Py<PyAny>> {
        let fields: Vec<(String, Py<PyAny>)> = slf
            .fields
            .iter()
            .map(|(name, value)| (name.clone(), value.clone_ref(py)))
            .collect();
        let object: Py<AST> = slf.into();
        let bound = object.bind(py);
        let key = bound.as_ptr() as usize;
        let class = bound.get_type();
        let fresh = class.call0()?;
        memo.set_item(key, &fresh)?;
        let copy_module = py.import("copy")?;
        for (name, value) in fields {
            let copied = copy_module
                .getattr("deepcopy")?
                .call1((value.bind(py), memo))?;
            fresh.setattr(name.as_str(), copied)?;
        }
        Ok(fresh.unbind())
    }
}

macro_rules! ast_node {
    ($name:ident, $py:literal, [$($field:literal),* $(,)?]) => {
        #[pyclass(extends=AST, name=$py, module="rs_viba.viba_ast.nodes")]
        pub struct $name;

        #[pymethods]
        impl $name {
            #[new]
            #[pyo3(signature = (*args, **kwargs))]
            fn new(
                py: Python<'_>,
                args: &Bound<'_, PyTuple>,
                kwargs: Option<&Bound<'_, PyDict>>,
            ) -> PyResult<PyClassInitializer<Self>> {
                let base = base_new(py, $py, &[$($field),*], args, kwargs)?;
                Ok(PyClassInitializer::from(base).add_subclass($name))
            }

            #[classattr]
            fn _fields(py: Python<'_>) -> PyResult<Py<PyAny>> {
                field_tuple(py, &[$($field),*])
            }
        }
    };
}

ast_node!(Module, "Module", ["body"]);
ast_node!(TypeDefinition, "TypeDefinition", ["name", "body"]);
ast_node!(
    GenericDefinition,
    "GenericDefinition",
    ["name", "generic_params", "body"]
);
ast_node!(Sum, "Sum", ["left", "right"]);
ast_node!(Product, "Product", ["left", "right"]);
ast_node!(Exponent, "Exponent", ["result", "argument"]);
ast_node!(Partial, "Partial", ["function", "argument"]);
ast_node!(Tagged, "Tagged", ["tag", "type"]);
ast_node!(Member, "Member", ["tag"]);
ast_node!(MemberTaken, "MemberTaken", ["owner", "name"]);
ast_node!(TypeApp, "TypeApp", ["constructor", "args"]);
ast_node!(Tuple, "Tuple", ["elements"]);
ast_node!(TypeRef, "TypeRef", ["name"]);
ast_node!(Constant, "Constant", ["value"]);
ast_node!(Nil, "Nil", []);
ast_node!(Never, "Never", []);
ast_node!(Any, "Any", []);
ast_node!(Ellipsis, "Ellipsis", []);
ast_node!(HintBlock, "HintBlock", ["code"]);
ast_node!(Import, "Import", ["module", "alias"]);
ast_node!(Pattern, "Pattern", ["pattern"]);
ast_node!(SumChain, "SumChain", ["elements"]);
ast_node!(ProductChain, "ProductChain", ["elements"]);
ast_node!(ExponentChain, "ExponentChain", ["elements"]);

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<AST>()?;
    m.add_class::<Module>()?;
    m.add_class::<TypeDefinition>()?;
    m.add_class::<GenericDefinition>()?;
    m.add_class::<Sum>()?;
    m.add_class::<Product>()?;
    m.add_class::<Exponent>()?;
    m.add_class::<Partial>()?;
    m.add_class::<Tagged>()?;
    m.add_class::<Member>()?;
    m.add_class::<MemberTaken>()?;
    m.add_class::<TypeApp>()?;
    m.add_class::<Tuple>()?;
    m.add_class::<TypeRef>()?;
    m.add_class::<Constant>()?;
    m.add_class::<Nil>()?;
    m.add_class::<Never>()?;
    m.add_class::<Any>()?;
    m.add_class::<Ellipsis>()?;
    m.add_class::<HintBlock>()?;
    m.add_class::<Import>()?;
    m.add_class::<Pattern>()?;
    m.add_class::<SumChain>()?;
    m.add_class::<ProductChain>()?;
    m.add_class::<ExponentChain>()?;
    Ok(())
}
