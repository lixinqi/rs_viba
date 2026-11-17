//! Native Type leaves, exported through rs_viba.type.
//!
//! The module classes (`ModuleType`, `BuiltinModuleType`, `CustomModuleType`,
//! `AstNodeType`) and `FixedIntType` remain Python classes for now; the leaves
//! they sit beside are Rust so the judgment does not cross into Python for
//! every scalar type.

use pyo3::prelude::*;
use pyo3::types::PyModule;

use crate::pyast;

#[pyclass(subclass, dict, name = "Type", module = "rs_viba.type")]
pub struct Type;

#[pymethods]
impl Type {
    #[new]
    #[pyo3(signature = (*_args, **_kwargs))]
    fn new(
        _args: &Bound<'_, pyo3::types::PyTuple>,
        _kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> Self {
        Type
    }
}

macro_rules! plain_type {
    ($name:ident, $py:literal) => {
        #[pyclass(extends=Type, name=$py, module="rs_viba.type")]
        pub struct $name;

        #[pymethods]
        impl $name {
            #[new]
            fn new() -> PyClassInitializer<Self> {
                PyClassInitializer::from(Type).add_subclass($name)
            }
        }
    };
}

plain_type!(NeverType, "NeverType");
plain_type!(AnyType, "AnyType");
plain_type!(NilType, "NilType");
plain_type!(BoolType, "BoolType");
plain_type!(IntType, "IntType");
plain_type!(FloatType, "FloatType");
plain_type!(StrType, "StrType");

#[pyclass(extends=Type, subclass, name = "FixedIntType", module = "rs_viba.type")]
pub struct FixedIntType;

#[pymethods]
impl FixedIntType {
    #[new]
    fn new() -> PyClassInitializer<Self> {
        PyClassInitializer::from(Type).add_subclass(FixedIntType)
    }
}

macro_rules! fixed_int {
    ($name:ident, $py:literal, $text:literal, $min:expr, $max:expr) => {
        #[pyclass(extends=FixedIntType, name=$py, module="rs_viba.type")]
        pub struct $name;

        #[pymethods]
        impl $name {
            #[new]
            fn new() -> PyClassInitializer<Self> {
                PyClassInitializer::from(Type)
                    .add_subclass(FixedIntType)
                    .add_subclass($name)
            }

            #[classattr]
            fn name() -> &'static str {
                $text
            }

            #[classattr]
            fn minimum() -> i128 {
                $min
            }

            #[classattr]
            fn maximum() -> i128 {
                $max
            }
        }
    };
}

fixed_int!(
    Int8Type,
    "Int8Type",
    "int8_t",
    -(2i128.pow(7)),
    2i128.pow(7) - 1
);
fixed_int!(UInt8Type, "UInt8Type", "uint8_t", 0, 2i128.pow(8) - 1);
fixed_int!(
    Int16Type,
    "Int16Type",
    "int16_t",
    -(2i128.pow(15)),
    2i128.pow(15) - 1
);
fixed_int!(UInt16Type, "UInt16Type", "uint16_t", 0, 2i128.pow(16) - 1);
fixed_int!(
    Int32Type,
    "Int32Type",
    "int32_t",
    -(2i128.pow(31)),
    2i128.pow(31) - 1
);
fixed_int!(UInt32Type, "UInt32Type", "uint32_t", 0, 2i128.pow(32) - 1);
fixed_int!(
    Int64Type,
    "Int64Type",
    "int64_t",
    -(2i128.pow(63)),
    2i128.pow(63) - 1
);
fixed_int!(UInt64Type, "UInt64Type", "uint64_t", 0, 2i128.pow(64) - 1);

macro_rules! literal_type {
    ($name:ident, $py:literal) => {
        #[pyclass(extends=Type, name=$py, module="rs_viba.type")]
        pub struct $name {
            #[pyo3(get, set)]
            value: Py<PyAny>,
        }

        #[pymethods]
        impl $name {
            #[new]
            fn new(value: Py<PyAny>) -> PyClassInitializer<Self> {
                PyClassInitializer::from(Type).add_subclass($name { value })
            }
        }
    };
}

literal_type!(BoolLiteralType, "BoolLiteralType");
literal_type!(IntLiteralType, "IntLiteralType");
literal_type!(FloatLiteralType, "FloatLiteralType");
literal_type!(StrLiteralType, "StrLiteralType");

#[pyclass(extends=Type, name = "BuiltinGenericType", module = "rs_viba.type")]
pub struct BuiltinGenericType {
    #[pyo3(get, set)]
    name: String,
}

#[pymethods]
impl BuiltinGenericType {
    #[new]
    fn new(name: String) -> PyClassInitializer<Self> {
        PyClassInitializer::from(Type).add_subclass(BuiltinGenericType { name })
    }
}

#[pyclass(extends=Type, subclass, name = "ModuleType", module = "rs_viba.type")]
pub struct ModuleType;

#[pymethods]
impl ModuleType {
    #[new]
    #[pyo3(signature = (*_args, **_kwargs))]
    fn new(
        _args: &Bound<'_, pyo3::types::PyTuple>,
        _kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyClassInitializer<Self> {
        PyClassInitializer::from(Type).add_subclass(ModuleType)
    }
}

#[pyclass(extends=ModuleType, name = "BuiltinModuleType", module = "rs_viba.type")]
pub struct BuiltinModuleType {
    #[pyo3(get, set)]
    _library: Py<PyAny>,
    #[pyo3(get, set)]
    _operators: Py<PyAny>,
}

#[pymethods]
impl BuiltinModuleType {
    #[new]
    fn new(py: Python<'_>) -> PyClassInitializer<Self> {
        PyClassInitializer::from(Type)
            .add_subclass(ModuleType)
            .add_subclass(BuiltinModuleType {
                _library: py.None(),
                _operators: py.None(),
            })
    }

    #[getter]
    fn name(&self) -> &'static str {
        "builtin"
    }

    #[getter]
    fn file_path(&self, py: Python<'_>) -> PyResult<String> {
        let builtin: String = py
            .import("rs_viba.type")?
            .getattr("BUILTIN_DIR")?
            .str()?
            .extract()?;
        Ok(format!("{}/builtin.viba", builtin))
    }

    #[getter]
    fn root(&self, py: Python<'_>) -> PyResult<String> {
        py.import("rs_viba.type")?
            .getattr("BUILTIN_DIR")?
            .str()?
            .extract()
    }

    #[getter]
    fn module_path(&self) -> &'static str {
        "/builtin"
    }

    fn lookup(slf: PyRef<'_, Self>, py: Python<'_>, type_name: &str) -> PyResult<Py<PyAny>> {
        let object: Py<BuiltinModuleType> = slf.into();
        crate::type_system::builtin_lookup(py, object.bind(py).as_any(), type_name)
    }

    fn operator(slf: PyRef<'_, Self>, py: Python<'_>, name: &str) -> PyResult<Py<PyAny>> {
        let object: Py<BuiltinModuleType> = slf.into();
        crate::type_system::builtin_operator(py, object.bind(py).as_any(), name)
    }
}

#[pyclass(extends=ModuleType, name = "GenericModuleType", module = "rs_viba.pattern")]
pub struct GenericModuleType {
    #[pyo3(get, set)]
    name: String,
    #[pyo3(get, set)]
    directory: String,
    #[pyo3(get, set)]
    entries: Py<PyAny>,
    #[pyo3(get, set)]
    module: Py<PyAny>,
}

#[pymethods]
impl GenericModuleType {
    #[new]
    #[pyo3(signature = (name, directory, entries, module = None))]
    fn new(
        py: Python<'_>,
        name: String,
        directory: String,
        entries: Py<PyAny>,
        module: Option<Py<PyAny>>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let module = match module {
            Some(module) => module,
            None => py
                .import("rs_viba.viba_ast")?
                .getattr("Module")?
                .call0()?
                .unbind(),
        };
        Ok(PyClassInitializer::from(Type)
            .add_subclass(ModuleType)
            .add_subclass(GenericModuleType {
                name,
                directory,
                entries,
                module,
            }))
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "GenericModuleType({}, {} patterns)",
            pyast::repr_of(py, &pyo3::types::PyString::new(py, &self.name).into_any()),
            self.entries.bind(py).len()?
        ))
    }
}

#[pyclass(name = "_MissingModule")]
pub struct MissingModule;

#[pymethods]
impl MissingModule {
    fn __call__(&self, py: Python<'_>, name: &str) -> PyResult<Py<PyAny>> {
        let type_module = py.import("rs_viba.type")?;
        let message = format!(
            "module {} not found",
            pyast::repr_of(py, &pyo3::types::PyString::new(py, name).into_any())
        );
        Ok(type_module.getattr("Err")?.call1((message,))?.unbind())
    }
}

#[pyclass(extends=ModuleType, name = "CustomModuleType", module = "rs_viba.type")]
pub struct CustomModuleType {
    #[pyo3(get, set)]
    module: Py<PyAny>,
    #[pyo3(get, set)]
    module_environment: Py<PyAny>,
    #[pyo3(get, set)]
    imports: Py<PyAny>,
    #[pyo3(get, set)]
    name: String,
    #[pyo3(get, set)]
    file_path: Py<PyAny>,
    #[pyo3(get, set)]
    root: Py<PyAny>,
}

#[pymethods]
impl CustomModuleType {
    #[new]
    #[pyo3(signature = (module, module_environment = None, imports = None, name = String::new(), file_path = None, root = None))]
    fn new(
        py: Python<'_>,
        module: Py<PyAny>,
        module_environment: Option<Py<PyAny>>,
        imports: Option<Py<PyAny>>,
        name: String,
        file_path: Option<Py<PyAny>>,
        root: Option<Py<PyAny>>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let module_environment = module_environment.unwrap_or_else(|| {
            Py::new(py, MissingModule)
                .map(|value| value.into_any())
                .unwrap_or_else(|_| py.None())
        });
        let imports = match imports {
            Some(imports) => imports,
            None => pyo3::types::PyDict::new(py).into_any().unbind(),
        };
        Ok(PyClassInitializer::from(Type)
            .add_subclass(ModuleType)
            .add_subclass(CustomModuleType {
                module,
                module_environment,
                imports,
                name,
                file_path: file_path.unwrap_or_else(|| py.None()),
                root: root.unwrap_or_else(|| py.None()),
            }))
    }

    #[getter]
    fn module_path(&self, py: Python<'_>) -> PyResult<String> {
        if !self.name.is_empty() {
            return Ok(format!("/{}", self.name.replace('.', "/")));
        }
        let file_path = self.file_path.bind(py);
        if file_path.is_none() {
            return Ok(String::new());
        }
        let mut file: String = file_path.extract()?;
        let root = self.root.bind(py);
        if !root.is_none() {
            let root: String = root.extract()?;
            if let Ok(relative) = py
                .import("os.path")?
                .getattr("relpath")?
                .call1((file.as_str(), root.as_str()))?
                .extract::<String>()
            {
                file = relative;
            }
        }
        if let Some(stripped) = file.strip_suffix(".viba") {
            file = stripped.to_string();
        }
        Ok(if file.starts_with('/') {
            file
        } else {
            format!("/{}", file)
        })
    }

    fn lookup_local(slf: PyRef<'_, Self>, py: Python<'_>, type_name: &str) -> PyResult<Py<PyAny>> {
        let object: Py<CustomModuleType> = slf.into();
        crate::type_system::custom_lookup_local_pub(py, object.bind(py).as_any(), type_name)
    }
}

fn source_path(py: Python<'_>, node: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
    if pyast::class_name(node)? == "TypeRef" {
        return Ok(Some(node.getattr("name")?.extract()?));
    }
    if pyast::class_name(node)? == "MemberTaken" {
        let owner = source_path(py, &node.getattr("owner")?)?;
        return Ok(owner.map(|owner| {
            format!(
                "{}.{}",
                owner,
                node.getattr("name").unwrap().extract::<String>().unwrap()
            )
        }));
    }
    Ok(None)
}

#[pyclass(extends=Type, name = "AstNodeType", module = "rs_viba.type")]
pub struct AstNodeType {
    #[pyo3(get, set)]
    ast_node: Py<PyAny>,
    #[pyo3(get, set)]
    container_module: Py<PyAny>,
    #[pyo3(get, set)]
    env_get: Py<PyAny>,
}

#[pymethods]
impl AstNodeType {
    #[new]
    #[pyo3(signature = (ast_node, container_module, env_get = None))]
    fn new(
        py: Python<'_>,
        ast_node: Py<PyAny>,
        container_module: Py<PyAny>,
        env_get: Option<Py<PyAny>>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let mut ast_node = ast_node;
        {
            let bound = ast_node.bind(py);
            if pyast::class_name(bound)? == "MemberTaken" {
                if let Some(path) = source_path(py, bound)? {
                    ast_node = py
                        .import("rs_viba.viba_ast")?
                        .getattr("TypeRef")?
                        .call1((path,))?
                        .unbind();
                }
            }
        }
        Ok(PyClassInitializer::from(Type).add_subclass(AstNodeType {
            ast_node,
            container_module,
            env_get: env_get.unwrap_or_else(|| py.None()),
        }))
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Type>()?;
    m.add_class::<NeverType>()?;
    m.add_class::<AnyType>()?;
    m.add_class::<NilType>()?;
    m.add_class::<BoolType>()?;
    m.add_class::<IntType>()?;
    m.add_class::<FloatType>()?;
    m.add_class::<StrType>()?;
    m.add_class::<FixedIntType>()?;
    m.add_class::<Int8Type>()?;
    m.add_class::<UInt8Type>()?;
    m.add_class::<Int16Type>()?;
    m.add_class::<UInt16Type>()?;
    m.add_class::<Int32Type>()?;
    m.add_class::<UInt32Type>()?;
    m.add_class::<Int64Type>()?;
    m.add_class::<UInt64Type>()?;
    m.add_class::<BoolLiteralType>()?;
    m.add_class::<IntLiteralType>()?;
    m.add_class::<FloatLiteralType>()?;
    m.add_class::<StrLiteralType>()?;
    m.add_class::<BuiltinGenericType>()?;
    m.add_class::<ModuleType>()?;
    m.add_class::<BuiltinModuleType>()?;
    m.add_class::<GenericModuleType>()?;
    m.add_class::<CustomModuleType>()?;
    m.add_class::<AstNodeType>()?;
    Ok(())
}
