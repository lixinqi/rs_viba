//! Native descriptor classes, exported through rs_viba.viba_type_descriptor.
//!
//! These are the data classes of the descriptor layer.  The builders and the
//! queries already live in Rust; this module makes the objects they hand out
//! Rust objects too, so the Python package is the boundary rather than the
//! implementation.

use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::types::PyString;

use crate::pyast;

fn none(py: Python<'_>) -> Py<PyAny> {
    py.None()
}

fn maybe<'py>(value: &Bound<'py, PyAny>, name: &str) -> PyResult<Py<PyAny>> {
    match value.getattr(name) {
        Ok(found) => Ok(found.unbind()),
        Err(_) => Ok(value.py().None()),
    }
}

#[pyclass(name = "VibaTypeDescriptor", module = "rs_viba.viba_type_descriptor")]
pub struct VibaTypeDescriptor {
    #[pyo3(get, set)]
    kind: String,
    #[pyo3(get, set)]
    payload: Py<PyAny>,
}

#[pymethods]
impl VibaTypeDescriptor {
    #[new]
    #[pyo3(signature = (kind, payload = None))]
    fn new(py: Python<'_>, kind: String, payload: Option<Py<PyAny>>) -> Self {
        VibaTypeDescriptor {
            kind,
            payload: payload.unwrap_or_else(|| none(py)),
        }
    }

    #[getter]
    fn pool(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        maybe(self.payload.bind(py), "pool")
    }

    #[getter]
    fn resolvable_type(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        maybe(self.payload.bind(py), "resolvable_type")
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        format!(
            "VibaTypeDescriptor({}, {})",
            self.kind,
            crate::pyast::repr_of(py, self.payload.bind(py))
        )
    }
}

macro_rules! descriptor_class {
    ($name:ident, $py:literal, { $( $field:ident : $ty:ty ),* $(,)? }) => {
        #[pyclass(name = $py, module = "rs_viba.viba_type_descriptor")]
        pub struct $name {
            $( #[pyo3(get, set)] $field: $ty, )*
        }

        #[pymethods]
        impl $name {
            #[new]
            #[pyo3(signature = ( $( $field ),* ))]
            fn new( $( $field: $ty ),* ) -> Self {
                $name { $( $field ),* }
            }
        }
    };
}

descriptor_class!(VibaTypeRefDescriptor, "VibaTypeRefDescriptor", {
    pool: Py<PyAny>,
    resolvable_type: Py<PyAny>,
    type_name: String,
});

descriptor_class!(VibaTypeAppDescriptor, "VibaTypeAppDescriptor", {
    pool: Py<PyAny>,
    resolvable_type: Py<PyAny>,
    constructor_name: String,
    args: Py<PyAny>,
});

descriptor_class!(VibaTupleDescriptor, "VibaTupleDescriptor", {
    pool: Py<PyAny>,
    resolvable_type: Py<PyAny>,
    elements: Py<PyAny>,
});

descriptor_class!(VibaTaggedDescriptor, "VibaTaggedDescriptor", {
    pool: Py<PyAny>,
    resolvable_type: Py<PyAny>,
    tag: String,
    tagged_type: Py<PyAny>,
});

descriptor_class!(VibaChainDescriptor, "VibaChainDescriptor", {
    pool: Py<PyAny>,
    resolvable_type: Py<PyAny>,
    elements: Py<PyAny>,
});

descriptor_class!(VibaLiteralDescriptor, "VibaLiteralDescriptor", {
    pool: Py<PyAny>,
    resolvable_type: Py<PyAny>,
    value: Py<PyAny>,
});

descriptor_class!(VibaHintBlockDescriptor, "VibaHintBlockDescriptor", {
    pool: Py<PyAny>,
    resolvable_type: Py<PyAny>,
    code: String,
});

descriptor_class!(VibaMemberTakenDescriptor, "VibaMemberTakenDescriptor", {
    pool: Py<PyAny>,
    resolvable_type: Py<PyAny>,
    owner: Py<PyAny>,
    name: String,
});

descriptor_class!(VibaMemberDescriptor, "VibaMemberDescriptor", {
    pool: Py<PyAny>,
    member_index: usize,
    tag: Py<PyAny>,
    member_type: Py<PyAny>,
    containing_full_name: String,
});

descriptor_class!(VibaImportDescriptor, "VibaImportDescriptor", {
    pool: Py<PyAny>,
    module_name: String,
    local_name: String,
});

#[pyclass(
    name = "VibaDefinitionDescriptor",
    module = "rs_viba.viba_type_descriptor"
)]
pub struct VibaDefinitionDescriptor {
    #[pyo3(get, set)]
    pool: Py<PyAny>,
    #[pyo3(get, set)]
    def_name: String,
    #[pyo3(get, set)]
    module_name: String,
    #[pyo3(get, set)]
    file_name: String,
    #[pyo3(get, set)]
    file_hash: String,
    #[pyo3(get, set)]
    full_name: String,
    #[pyo3(get, set)]
    generic_params: Py<PyAny>,
    #[pyo3(get, set)]
    body: Py<PyAny>,
    #[pyo3(get, set)]
    members: Py<PyAny>,
}

#[pymethods]
impl VibaDefinitionDescriptor {
    #[new]
    #[pyo3(signature = (pool, def_name, module_name, file_name, file_hash, full_name, generic_params, body, members))]
    fn new(
        pool: Py<PyAny>,
        def_name: String,
        module_name: String,
        file_name: String,
        file_hash: String,
        full_name: String,
        generic_params: Py<PyAny>,
        body: Py<PyAny>,
        members: Py<PyAny>,
    ) -> Self {
        VibaDefinitionDescriptor {
            pool,
            def_name,
            module_name,
            file_name,
            file_hash,
            full_name,
            generic_params,
            body,
            members,
        }
    }
}

#[pyclass(name = "VibaFileDescriptor", module = "rs_viba.viba_type_descriptor")]
pub struct VibaFileDescriptor {
    #[pyo3(get, set)]
    pool: Py<PyAny>,
    #[pyo3(get, set)]
    file_name: String,
    #[pyo3(get, set)]
    file_hash: String,
    #[pyo3(get, set)]
    module_name: String,
    #[pyo3(get, set)]
    imports: Py<PyAny>,
    #[pyo3(get, set)]
    definitions: Py<PyAny>,
    #[pyo3(get, set)]
    _tree: Py<PyAny>,
}

#[pymethods]
impl VibaFileDescriptor {
    #[new]
    #[pyo3(signature = (pool, file_name, file_hash, module_name, imports, definitions, tree))]
    fn new(
        pool: Py<PyAny>,
        file_name: String,
        file_hash: String,
        module_name: String,
        imports: Py<PyAny>,
        definitions: Py<PyAny>,
        tree: Py<PyAny>,
    ) -> Self {
        VibaFileDescriptor {
            pool,
            file_name,
            file_hash,
            module_name,
            imports,
            definitions,
            _tree: tree,
        }
    }
}

#[pyclass(name = "VibaPool", module = "rs_viba.viba_type_descriptor")]
pub struct VibaPool {
    #[pyo3(get, set)]
    files: Py<PyAny>,
    #[pyo3(get, set)]
    file_name2file: Py<PyAny>,
    #[pyo3(get, set)]
    full_name2definition: Py<PyAny>,
    #[pyo3(get, set)]
    full_name2member: Py<PyAny>,
    #[pyo3(get, set)]
    module_environment: Py<PyAny>,
}

#[pymethods]
impl VibaPool {
    #[new]
    fn new(
        files: Py<PyAny>,
        file_name2file: Py<PyAny>,
        full_name2definition: Py<PyAny>,
        full_name2member: Py<PyAny>,
        module_environment: Py<PyAny>,
    ) -> Self {
        VibaPool {
            files,
            file_name2file,
            full_name2definition,
            full_name2member,
            module_environment,
        }
    }
}

#[pymethods]
impl VibaTypeRefDescriptor {
    fn __repr__(&self, py: Python<'_>) -> String {
        format!(
            "VibaTypeRefDescriptor({})",
            crate::pyast::repr_of(py, &PyString::new(py, &self.type_name).into_any())
        )
    }
}

#[pymethods]
impl VibaMemberTakenDescriptor {
    fn __repr__(&self, py: Python<'_>) -> String {
        format!(
            "VibaMemberTakenDescriptor({}, {})",
            crate::pyast::repr_of(py, self.owner.bind(py)),
            crate::pyast::repr_of(py, &PyString::new(py, &self.name).into_any())
        )
    }
}

#[pymethods]
impl VibaTypeAppDescriptor {
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let count = self.args.bind(py).len()?;
        Ok(format!(
            "VibaTypeAppDescriptor({}, {} args)",
            crate::pyast::repr_of(py, &PyString::new(py, &self.constructor_name).into_any()),
            count
        ))
    }
}

#[pymethods]
impl VibaTupleDescriptor {
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "VibaTupleDescriptor({} elements)",
            self.elements.bind(py).len()?
        ))
    }
}

#[pymethods]
impl VibaChainDescriptor {
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "VibaChainDescriptor({} elements)",
            self.elements.bind(py).len()?
        ))
    }
}

#[pymethods]
impl VibaLiteralDescriptor {
    fn __repr__(&self, py: Python<'_>) -> String {
        format!(
            "VibaLiteralDescriptor({})",
            crate::pyast::repr_of(py, self.value.bind(py))
        )
    }
}

#[pymethods]
impl VibaHintBlockDescriptor {
    fn __repr__(&self) -> &'static str {
        "VibaHintBlockDescriptor(...)"
    }
}

#[pymethods]
impl VibaImportDescriptor {
    fn __repr__(&self, py: Python<'_>) -> String {
        format!(
            "VibaImportDescriptor({} as {})",
            crate::pyast::repr_of(py, &PyString::new(py, &self.module_name).into_any()),
            crate::pyast::repr_of(py, &PyString::new(py, &self.local_name).into_any())
        )
    }
}

#[pymethods]
impl VibaMemberDescriptor {
    fn __repr__(&self, py: Python<'_>) -> String {
        format!(
            "VibaMemberDescriptor({}, {})",
            self.member_index,
            crate::pyast::repr_of(py, self.tag.bind(py))
        )
    }
}

#[pymethods]
impl VibaDefinitionDescriptor {
    fn __repr__(&self, py: Python<'_>) -> String {
        format!(
            "VibaDefinitionDescriptor({})",
            crate::pyast::repr_of(py, &PyString::new(py, &self.full_name).into_any())
        )
    }
}

#[pymethods]
impl VibaFileDescriptor {
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "VibaFileDescriptor({}, {} defs)",
            crate::pyast::repr_of(py, &PyString::new(py, &self.module_name).into_any()),
            self.definitions.bind(py).len()?
        ))
    }
}

#[pymethods]
impl VibaPool {
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!("VibaPool({} files)", self.files.bind(py).len()?))
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<VibaTypeDescriptor>()?;
    m.add_class::<VibaMemberTakenDescriptor>()?;
    m.add_class::<VibaTypeRefDescriptor>()?;
    m.add_class::<VibaTypeAppDescriptor>()?;
    m.add_class::<VibaTupleDescriptor>()?;
    m.add_class::<VibaTaggedDescriptor>()?;
    m.add_class::<VibaChainDescriptor>()?;
    m.add_class::<VibaLiteralDescriptor>()?;
    m.add_class::<VibaHintBlockDescriptor>()?;
    m.add_class::<VibaMemberDescriptor>()?;
    m.add_class::<VibaDefinitionDescriptor>()?;
    m.add_class::<VibaImportDescriptor>()?;
    m.add_class::<VibaFileDescriptor>()?;
    m.add_class::<VibaPool>()?;
    Ok(())
}
