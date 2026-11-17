//! The interpreter's small runtime data classes, exported through
//! rs_viba.interpret: the values a run threads and the records it keeps.

use pyo3::prelude::*;
use pyo3::types::PyModule;

use crate::pyast;

#[pyclass(name = "_VibaData", module = "rs_viba.interpret")]
pub struct VibaData {
    #[pyo3(get, set)]
    node: Py<PyAny>,
    #[pyo3(get, set)]
    source_module: Py<PyAny>,
    #[pyo3(get, set)]
    given: Py<PyAny>,
    #[pyo3(get, set)]
    bindings: Py<PyAny>,
    #[pyo3(get, set)]
    members: Py<PyAny>,
}

#[pymethods]
impl VibaData {
    #[new]
    #[pyo3(signature = (node, source_module = None, given = None, bindings = None, members = None))]
    fn new(
        py: Python<'_>,
        node: Py<PyAny>,
        source_module: Option<Py<PyAny>>,
        given: Option<Py<PyAny>>,
        bindings: Option<Py<PyAny>>,
        members: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        let source_module = match source_module {
            Some(source_module) => source_module,
            None => match crate::interpret::source_module_of_py(py, node.bind(py))? {
                Some(source_module) => source_module,
                None => py.None(),
            },
        };
        Ok(VibaData {
            node,
            source_module,
            given: given.unwrap_or_else(|| py.None()),
            bindings: bindings.unwrap_or_else(|| py.None()),
            members: members.unwrap_or_else(|| py.None()),
        })
    }

    fn __repr__(&self) -> &'static str {
        "_VibaData(...)"
    }
}

#[pyclass(name = "_Host", module = "rs_viba.interpret")]
pub struct Host {
    #[pyo3(get, set)]
    obj: Py<PyAny>,
}

#[pymethods]
impl Host {
    #[new]
    fn new(obj: Py<PyAny>) -> Self {
        Host { obj }
    }
}

#[pyclass(name = "_Member", module = "rs_viba.interpret")]
pub struct Member {
    #[pyo3(get, set)]
    tag: String,
}

#[pymethods]
impl Member {
    #[new]
    fn new(tag: String) -> Self {
        Member { tag }
    }
}

#[pyclass(name = "_Given", module = "rs_viba.interpret")]
pub struct Given {
    #[pyo3(get, set)]
    tag: Py<PyAny>,
    #[pyo3(get, set)]
    value: Py<PyAny>,
}

#[pymethods]
impl Given {
    #[new]
    fn new(tag: Py<PyAny>, value: Py<PyAny>) -> Self {
        Given { tag, value }
    }
}

#[pyclass(name = "_GetArgs", module = "rs_viba.interpret")]
pub struct GetArgs {
    #[pyo3(get, set)]
    activation: Py<PyAny>,
}

#[pymethods]
impl GetArgs {
    #[new]
    fn new(activation: Py<PyAny>) -> Self {
        GetArgs { activation }
    }
}

#[pyclass(name = "_IsSubType", module = "rs_viba.interpret")]
pub struct IsSubType {
    #[pyo3(get, set)]
    given: Py<PyAny>,
}

#[pymethods]
impl IsSubType {
    #[new]
    #[pyo3(signature = (given = None))]
    fn new(py: Python<'_>, given: Option<Py<PyAny>>) -> PyResult<Self> {
        let given = match given {
            Some(given) => given,
            None => pyo3::types::PyList::empty(py).into_any().unbind(),
        };
        Ok(IsSubType { given })
    }
}

#[pyclass(name = "_HostFunction", module = "rs_viba.interpret")]
pub struct HostFunction {
    #[pyo3(get, set)]
    name: String,
    #[pyo3(get, set)]
    func: Py<PyAny>,
    #[pyo3(get, set)]
    slots: i64,
    #[pyo3(get, set)]
    given: Py<PyAny>,
    #[pyo3(get, set)]
    pieces: Py<PyAny>,
}

#[pymethods]
impl HostFunction {
    #[new]
    #[pyo3(signature = (name, func, slots = 1, given = None, pieces = None))]
    fn new(
        py: Python<'_>,
        name: String,
        func: Py<PyAny>,
        slots: i64,
        given: Option<Py<PyAny>>,
        pieces: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        let list = |value: Option<Py<PyAny>>| -> PyResult<Py<PyAny>> {
            match value {
                Some(value) => Ok(value),
                None => Ok(pyo3::types::PyList::empty(py).into_any().unbind()),
            }
        };
        Ok(HostFunction {
            name,
            func,
            slots,
            given: list(given)?,
            pieces: list(pieces)?,
        })
    }

    fn as_callable(slf: PyRef<'_, Self>, py: Python<'_>) -> Py<PyAny> {
        let object: Py<HostFunction> = slf.into();
        object.into_any()
    }

    #[pyo3(signature = (*arguments))]
    fn __call__(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        arguments: &Bound<'_, pyo3::types::PyTuple>,
    ) -> PyResult<Py<PyAny>> {
        let object: Py<HostFunction> = slf.into();
        crate::interpret::host_function_call_py(py, object.bind(py).as_any(), arguments.as_any())
    }
}

#[pyclass(name = "_DynCall", module = "rs_viba.interpret")]
pub struct DynCall {
    #[pyo3(get, set)]
    activation: Py<PyAny>,
    #[pyo3(get, set)]
    kind: String,
    #[pyo3(get, set)]
    head: Py<PyAny>,
    #[pyo3(get, set)]
    source: String,
    #[pyo3(get, set)]
    environ: Py<PyAny>,
    #[pyo3(get, set)]
    name: Py<PyAny>,
    #[pyo3(get, set)]
    value: Py<PyAny>,
    #[pyo3(get, set)]
    given: Py<PyAny>,
    #[pyo3(get, set)]
    given_tags: Py<PyAny>,
}

#[pymethods]
impl DynCall {
    #[new]
    #[pyo3(signature = (activation, kind, head, source))]
    fn new(
        py: Python<'_>,
        activation: Py<PyAny>,
        kind: String,
        head: Py<PyAny>,
        source: String,
    ) -> Self {
        DynCall {
            activation,
            kind,
            head,
            source,
            environ: py.None(),
            name: py.None(),
            value: py.None(),
            given: pyo3::types::PyList::empty(py).into_any().unbind(),
            given_tags: pyo3::types::PyList::empty(py).into_any().unbind(),
        }
    }

    fn as_viba_data(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let object: Py<DynCall> = slf.into();
        crate::interpret::dyn_call_as_viba_data_py(py, object.bind(py).as_any())
    }
}

#[pyclass(name = "_Activation", module = "rs_viba.interpret")]
pub struct Activation {
    #[pyo3(get, set)]
    runner: Py<PyAny>,
    #[pyo3(get, set)]
    module: Py<PyAny>,
    #[pyo3(get, set)]
    environ: Py<PyAny>,
    #[pyo3(get, set)]
    name: String,
    #[pyo3(get, set)]
    file: Py<PyAny>,
    #[pyo3(get, set)]
    args: Py<PyAny>,
    #[pyo3(get, set)]
    path: String,
    #[pyo3(get, set)]
    signature: Py<PyAny>,
    #[pyo3(get, set)]
    members: Py<PyAny>,
    #[pyo3(get, set)]
    bindings: Py<PyAny>,
    #[pyo3(get, set)]
    defined: Py<PyAny>,
}

#[pymethods]
impl Activation {
    #[new]
    #[pyo3(signature = (runner, module, environ, name, file, args = None, members = None, bindings = None))]
    fn new(
        py: Python<'_>,
        runner: Py<PyAny>,
        module: Py<PyAny>,
        environ: Py<PyAny>,
        name: String,
        file: Py<PyAny>,
        args: Option<Py<PyAny>>,
        members: Option<Py<PyAny>>,
        bindings: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        let native = PyModule::import(py, "rs_viba._native")?;
        let args = match args {
            Some(args) => args,
            None => {
                let data = crate::interpret::viba_data_py(py, py.None().into_bound(py).as_any())?;
                native.getattr("_VibaData")?.call1((data,))?.unbind()
            }
        };
        let path = crate::interpret::storage_path_of_py(py, environ.bind(py))?;
        let members_ref = match &members {
            Some(members) => members.bind(py).clone(),
            None => py.None().into_bound(py),
        };
        let signature = crate::interpret::call_signature_py(py, &members_ref)?;
        let members = match members {
            Some(members) => members,
            None => {
                let host = native.getattr("_Host")?.call1((environ.clone_ref(py),))?;
                let item = pyo3::types::PyTuple::new(
                    py,
                    [pyo3::types::PyString::new(py, "$env").into_any(), host],
                )?;
                pyo3::types::PyList::new(py, [item])?.into_any().unbind()
            }
        };
        let bindings = bindings.unwrap_or_else(|| pyo3::types::PyDict::new(py).into_any().unbind());
        Ok(Activation {
            runner,
            module,
            environ,
            name,
            file,
            args,
            path,
            signature,
            members,
            bindings,
            defined: pyo3::types::PyDict::new(py).into_any().unbind(),
        })
    }
}

#[pyclass(name = "_Pending", module = "rs_viba.interpret")]
pub struct Pending {
    #[pyo3(get, set)]
    kind: String,
    #[pyo3(get, set)]
    activation: Py<PyAny>,
    #[pyo3(get, set)]
    head: Py<PyAny>,
    #[pyo3(get, set)]
    source: String,
    #[pyo3(get, set)]
    name: String,
    #[pyo3(get, set)]
    module: Py<PyAny>,
    #[pyo3(get, set)]
    source_module: Py<PyAny>,
    #[pyo3(get, set)]
    elements: Py<PyAny>,
    #[pyo3(get, set)]
    runner: Py<PyAny>,
    #[pyo3(get, set)]
    module_name: Py<PyAny>,
    #[pyo3(get, set)]
    environ: Py<PyAny>,
    #[pyo3(get, set)]
    given: Py<PyAny>,
    #[pyo3(get, set)]
    given_tags: Py<PyAny>,
    #[pyo3(get, set)]
    bindings: Py<PyAny>,
    #[pyo3(get, set)]
    pattern_env: Py<PyAny>,
    #[pyo3(get, set)]
    params_override: Py<PyAny>,
    #[pyo3(get, set)]
    source_bindings: Py<PyAny>,
    #[pyo3(get, set)]
    appends_environ: bool,
    #[pyo3(get, set)]
    member_of: Py<PyAny>,
    #[pyo3(get, set)]
    member_tag: Py<PyAny>,
}

#[pymethods]
impl Pending {
    #[new]
    #[pyo3(signature = (kind, activation = None, head = None, source = String::new(), name = String::new(), module = None, source_module = None, elements = None, runner = None, module_name = None, bindings = None, pattern_env = None, slots = None, source_bindings = None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        kind: String,
        activation: Option<Py<PyAny>>,
        head: Option<Py<PyAny>>,
        source: String,
        name: String,
        module: Option<Py<PyAny>>,
        source_module: Option<Py<PyAny>>,
        elements: Option<Py<PyAny>>,
        runner: Option<Py<PyAny>>,
        module_name: Option<Py<PyAny>>,
        bindings: Option<Py<PyAny>>,
        pattern_env: Option<Py<PyAny>>,
        slots: Option<Py<PyAny>>,
        source_bindings: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        let list = |value: Option<Py<PyAny>>| -> PyResult<Py<PyAny>> {
            match value {
                Some(value) => Ok(value),
                None => Ok(pyo3::types::PyList::empty(py).into_any().unbind()),
            }
        };
        let dict = || pyo3::types::PyDict::new(py).into_any().unbind();
        let module = module.unwrap_or_else(|| py.None());
        let source_module = match source_module {
            Some(source_module) => source_module,
            None => module.clone_ref(py),
        };
        Ok(Pending {
            kind,
            activation: activation.unwrap_or_else(|| py.None()),
            head: head.unwrap_or_else(|| py.None()),
            name: if name.is_empty() {
                source.clone()
            } else {
                name
            },
            source,
            module,
            source_module,
            elements: list(elements)?,
            runner: runner.unwrap_or_else(|| py.None()),
            module_name: module_name.unwrap_or_else(|| py.None()),
            environ: py.None(),
            given: dict(),
            given_tags: dict(),
            bindings: bindings.unwrap_or_else(|| py.None()),
            pattern_env: pattern_env.unwrap_or_else(|| py.None()),
            params_override: slots.unwrap_or_else(|| py.None()),
            source_bindings: source_bindings.unwrap_or_else(|| py.None()),
            appends_environ: false,
            member_of: py.None(),
            member_tag: py.None(),
        })
    }

    fn as_viba_data(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let object: Py<Pending> = slf.into();
        crate::interpret::pending_as_viba_data_py(py, object.bind(py).as_any())
    }
}

#[pyclass(name = "_HostArgument", module = "rs_viba.interpret")]
pub struct HostArgument {
    #[pyo3(get, set)]
    node: Py<PyAny>,
    #[pyo3(get, set)]
    scope: Py<PyAny>,
    #[pyo3(get, set)]
    pending: Py<PyAny>,
    #[pyo3(get, set)]
    module: Py<PyAny>,
    #[pyo3(get, set)]
    owner: Py<PyAny>,
    #[pyo3(get, set)]
    file: Py<PyAny>,
    #[pyo3(get, set)]
    source_module: Py<PyAny>,
    #[pyo3(get, set)]
    origin: Py<PyAny>,
    #[pyo3(get, set)]
    name: String,
    #[pyo3(get, set)]
    answer: Py<PyAny>,
    #[pyo3(get, set)]
    environ: Py<PyAny>,
    #[pyo3(get, set)]
    closure: Py<PyAny>,
}

#[pymethods]
impl HostArgument {
    #[new]
    #[pyo3(signature = (node, scope, pending, module, owner, file = None, source_module = None, name = String::new(), origin = None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        node: Py<PyAny>,
        scope: Py<PyAny>,
        pending: Py<PyAny>,
        module: Py<PyAny>,
        owner: Py<PyAny>,
        file: Option<Py<PyAny>>,
        source_module: Option<Py<PyAny>>,
        name: String,
        origin: Option<Py<PyAny>>,
    ) -> Self {
        HostArgument {
            node,
            scope,
            pending,
            module,
            owner,
            file: file.unwrap_or_else(|| py.None()),
            source_module: source_module.unwrap_or_else(|| py.None()),
            origin: origin.unwrap_or_else(|| py.None()),
            name,
            answer: py.None(),
            environ: py.None(),
            closure: pyo3::types::PyBool::new(py, false)
                .to_owned()
                .into_any()
                .unbind(),
        }
    }

    fn closure_of(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let object: Py<HostArgument> = slf.into();
        crate::interpret::host_argument_closure_py(py, object.bind(py).as_any())
    }

    #[pyo3(signature = (environ, *arguments))]
    fn __call__(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        environ: &Bound<'_, PyAny>,
        arguments: &Bound<'_, pyo3::types::PyTuple>,
    ) -> PyResult<Py<PyAny>> {
        let object: Py<HostArgument> = slf.into();
        crate::interpret::host_argument_call_py(
            py,
            object.bind(py).as_any(),
            environ,
            arguments.as_any(),
        )
    }
}

#[pyclass(name = "_Getter", module = "rs_viba.interpret")]
pub struct Getter {
    #[pyo3(get, set)]
    activation: Py<PyAny>,
    #[pyo3(get, set)]
    node: Py<PyAny>,
    #[pyo3(get, set)]
    scope: Py<PyAny>,
    #[pyo3(get, set)]
    answer: Py<PyAny>,
    #[pyo3(get, set)]
    file: Py<PyAny>,
    #[pyo3(get, set)]
    source_module: Py<PyAny>,
    #[pyo3(get, set)]
    name: Py<PyAny>,
    #[pyo3(get, set)]
    origin: Py<PyAny>,
}

#[pymethods]
impl Getter {
    #[new]
    #[pyo3(signature = (activation, node, scope = None))]
    fn new(
        py: Python<'_>,
        activation: Py<PyAny>,
        node: Py<PyAny>,
        scope: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        let scope = scope.unwrap_or_else(|| pyo3::types::PyTuple::empty(py).into_any().unbind());
        let file = activation.bind(py).getattr("file")?.unbind();
        let source_module = activation.bind(py).getattr("module")?.unbind();
        let name = activation.bind(py).getattr("name")?.unbind();
        Ok(Getter {
            file,
            source_module,
            origin: activation.clone_ref(py),
            activation,
            node,
            scope,
            answer: py.None(),
            name,
        })
    }

    fn __call__(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let object: Py<Getter> = slf.into();
        crate::interpret::getter_call_py(py, object.bind(py).as_any())
    }
}

#[pyclass(name = "_Runner", module = "rs_viba.interpret")]
pub struct Runner {
    #[pyo3(get, set)]
    paths: Py<PyAny>,
    #[pyo3(get, set)]
    get_file: Py<PyAny>,
    #[pyo3(get, set)]
    list_files: Py<PyAny>,
    #[pyo3(get, set)]
    by_path: Py<PyAny>,
    #[pyo3(get, set)]
    documents: Py<PyAny>,
    #[pyo3(get, set)]
    by_name: Py<PyAny>,
    #[pyo3(get, set)]
    path_of: Py<PyAny>,
    #[pyo3(get, set)]
    running: Py<PyAny>,
    #[pyo3(get, set)]
    done: Py<PyAny>,
    #[pyo3(get, set)]
    computing: Py<PyAny>,
    #[pyo3(get, set)]
    running_calls: Py<PyAny>,
    #[pyo3(get, set)]
    global_resumable_stack: Py<PyAny>,
    #[pyo3(get, set)]
    held_paths: Py<PyAny>,
    #[pyo3(get, set)]
    compacted_chain_length: i64,
    #[pyo3(get, set)]
    viba_path: Py<PyAny>,
}

#[pymethods]
impl Runner {
    #[new]
    #[pyo3(signature = (viba_path = None, get_file = None, list_files = None))]
    fn new(
        py: Python<'_>,
        viba_path: Option<Py<PyAny>>,
        get_file: Option<Py<PyAny>>,
        list_files: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        let viba_path = viba_path.unwrap_or_else(|| py.None());
        let paths = pyo3::types::PyList::empty(py);
        if !viba_path.bind(py).is_none() {
            let text: String = py
                .import("os")?
                .getattr("fspath")?
                .call1((viba_path.bind(py),))?
                .extract()?;
            for part in text.split(':') {
                if part.is_empty() {
                    continue;
                }
                paths.append(py.import("pathlib")?.getattr("Path")?.call1((part,))?)?;
            }
        }
        let dict = || pyo3::types::PyDict::new(py).into_any().unbind();
        let list = || pyo3::types::PyList::empty(py).into_any().unbind();
        Ok(Runner {
            paths: paths.into_any().unbind(),
            get_file: get_file.unwrap_or_else(|| py.None()),
            list_files: list_files.unwrap_or_else(|| py.None()),
            by_path: dict(),
            documents: dict(),
            by_name: dict(),
            path_of: dict(),
            running: list(),
            done: dict(),
            computing: list(),
            running_calls: list(),
            global_resumable_stack: list(),
            held_paths: dict(),
            compacted_chain_length: 0,
            viba_path,
        })
    }

    fn imported(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        name: &str,
        near: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let object: Py<Runner> = slf.into();
        let near = near
            .map(|value| value.clone().unbind())
            .unwrap_or_else(|| py.None());
        Ok(py
            .import("rs_viba._native")?
            .getattr("runner_imported_py")?
            .call1((object.bind(py), name, near))?
            .unbind())
    }

    fn load(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        name: &str,
        near: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        Runner::imported(slf, py, name, near)
    }

    fn module(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        name: &str,
        near: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        Runner::imported(slf, py, name, near)
    }

    fn builtin_module(slf: PyRef<'_, Self>, py: Python<'_>, name: &str) -> PyResult<Py<PyAny>> {
        let object: Py<Runner> = slf.into();
        Ok(py
            .import("rs_viba._native")?
            .getattr("runner_builtin_module_py")?
            .call1((object.bind(py), name))?
            .unbind())
    }

    fn run_file(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        file: &Bound<'_, PyAny>,
        environ: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let object: Py<Runner> = slf.into();
        Ok(py
            .import("rs_viba._native")?
            .getattr("runner_run_file_py")?
            .call1((object.bind(py), file, environ))?
            .unbind())
    }

    fn run_code(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        viba_code: &Bound<'_, PyAny>,
        environ: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let object: Py<Runner> = slf.into();
        Ok(py
            .import("rs_viba._native")?
            .getattr("runner_run_code_py")?
            .call1((object.bind(py), viba_code, environ))?
            .unbind())
    }

    fn run_node(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        node: &Bound<'_, PyAny>,
        environ: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let object: Py<Runner> = slf.into();
        Ok(py
            .import("rs_viba._native")?
            .getattr("runner_run_node_py")?
            .call1((object.bind(py), node, environ))?
            .unbind())
    }
}

#[pyclass(name = "_ResumableCall", module = "rs_viba.interpret")]
pub struct ResumableCall {
    #[pyo3(get, set)]
    runner: Py<PyAny>,
    #[pyo3(get, set)]
    module: Py<PyAny>,
    #[pyo3(get, set)]
    environ: Py<PyAny>,
    #[pyo3(get, set)]
    name: String,
    #[pyo3(get, set)]
    file: Py<PyAny>,
    #[pyo3(get, set)]
    args: Py<PyAny>,
    #[pyo3(get, set)]
    members: Py<PyAny>,
    #[pyo3(get, set)]
    bindings: Py<PyAny>,
    #[pyo3(get, set)]
    passes_environ: bool,
    #[pyo3(get, set)]
    give_environ: bool,
    #[pyo3(get, set)]
    at: i64,
}

#[pymethods]
impl ResumableCall {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        runner: Py<PyAny>,
        module: Py<PyAny>,
        environ: Py<PyAny>,
        name: String,
        file: Py<PyAny>,
        args: Py<PyAny>,
        members: Py<PyAny>,
        bindings: Py<PyAny>,
        passes_environ: bool,
        give_environ: bool,
        at: i64,
    ) -> Self {
        ResumableCall {
            runner,
            module,
            environ,
            name,
            file,
            args,
            members,
            bindings,
            passes_environ,
            give_environ,
            at,
        }
    }

    #[getter]
    fn current_env_path(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let object: Py<ResumableCall> = slf.into();
        Ok(py
            .import("rs_viba._native")?
            .getattr("resumable_current_env_path_py")?
            .call1((object.bind(py),))?
            .unbind())
    }

    fn restored(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        root: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let object: Py<ResumableCall> = slf.into();
        Ok(py
            .import("rs_viba._native")?
            .getattr("resumable_restored_py")?
            .call1((object.bind(py), root))?
            .unbind())
    }

    fn run(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        environ: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let object: Py<ResumableCall> = slf.into();
        Ok(py
            .import("rs_viba._native")?
            .getattr("resumable_run_py")?
            .call1((object.bind(py), environ))?
            .unbind())
    }
}

#[pyclass(extends=pyo3::exceptions::PyException, name = "_Raised", module = "rs_viba.interpret")]
pub struct Raised {
    #[pyo3(get, set)]
    result: Py<PyAny>,
}

#[pymethods]
impl Raised {
    #[new]
    fn new(result: Py<PyAny>) -> Self {
        Raised { result }
    }

    fn __str__(&self, py: Python<'_>) -> String {
        crate::pyast::repr_of(py, self.result.bind(py))
    }
}

#[pyclass(extends=pyo3::exceptions::PyBaseException, name = "_Compact", module = "rs_viba.interpret")]
pub struct Compact {
    #[pyo3(get, set)]
    tasks: Py<PyAny>,
    #[pyo3(get, set)]
    environ: Py<PyAny>,
}

#[pymethods]
impl Compact {
    #[new]
    fn new(tasks: Py<PyAny>, environ: Py<PyAny>) -> Self {
        Compact { tasks, environ }
    }

    fn __str__(&self) -> &'static str {
        "the chain of environments got too long"
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let _ = pyast::class_name; // keep the module's helper linked for parity
    m.add_class::<VibaData>()?;
    m.add_class::<Host>()?;
    m.add_class::<Member>()?;
    m.add_class::<Given>()?;
    m.add_class::<GetArgs>()?;
    m.add_class::<IsSubType>()?;
    m.add_class::<HostFunction>()?;
    m.add_class::<DynCall>()?;
    m.add_class::<Activation>()?;
    m.add_class::<Pending>()?;
    m.add_class::<HostArgument>()?;
    m.add_class::<Getter>()?;
    m.add_class::<Runner>()?;
    m.add_class::<ResumableCall>()?;
    m.add_class::<Raised>()?;
    m.add_class::<Compact>()?;
    Ok(())
}
