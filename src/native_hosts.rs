//! Direct Rust implementations of the host steps used by the Y and compaction
//! exercises.  This module is opt-in through EnvironmentCompute.native_hosts;
//! the Python host remains the default and the reference.

use pyo3::prelude::*;

fn int_of(value: &Bound<'_, PyAny>) -> PyResult<i64> {
    value.getattr("value")?.extract()
}

fn bool_of(value: &Bound<'_, PyAny>) -> PyResult<bool> {
    value.getattr("value")?.extract()
}

fn never(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let ast = py.import("rs_viba.viba_ast")?;
    let node = ast.getattr("Never")?.call0()?;
    let type_module = py.import("rs_viba.type")?;
    let no_module = type_module.getattr("custom_module")?.call1(("",))?;
    let ast_node_type = type_module
        .getattr("AstNodeType")?
        .call1((node.clone(), no_module))?;
    let descriptor = py
        .import("rs_viba.viba_type_descriptor")?
        .getattr("descriptor_of")?
        .call1((ast_node_type,))?;
    let reflect = py.import("rs_viba.reflect")?;
    Ok(reflect
        .getattr("VObject")?
        .call1((reflect.getattr("access")?, descriptor, node))?
        .unbind())
}

fn sub_env(py: Python<'_>, environ: &Bound<'_, PyAny>, name: &str) -> PyResult<Py<PyAny>> {
    Ok(py
        .import("rs_viba._native")?
        .getattr("sub_env_py")?
        .call1((environ, name))?
        .unbind())
}

pub(crate) fn call(
    py: Python<'_>,
    _module: &str,
    name: &str,
    args: &[Py<PyAny>],
) -> Option<PyResult<Py<PyAny>>> {
    let at = |index: usize| args.get(index).map(|value| value.bind(py));
    match name {
        "echo" => at(1).map(|value| Ok(value.clone().unbind())),
        "lt" | "sub" | "add" => Some((|| {
            let x = at(1).ok_or_else(|| pyo3::exceptions::PyTypeError::new_err("missing x"))?;
            let y = at(2).ok_or_else(|| pyo3::exceptions::PyTypeError::new_err("missing y"))?;
            match name {
                "lt" => Ok(pyo3::types::PyBool::new(py, int_of(&x)? < int_of(&y)?)
                    .to_owned()
                    .into_any()
                    .unbind()),
                "sub" => Ok((int_of(&x)? - int_of(&y)?)
                    .into_pyobject(py)?
                    .into_any()
                    .unbind()),
                _ => Ok((int_of(&x)? + int_of(&y)?)
                    .into_pyobject(py)?
                    .into_any()
                    .unbind()),
            }
        })()),
        "note" => Some((|| Ok(1i64.into_pyobject(py)?.into_any().unbind()))()),
        "limit" => Some((|| Ok(0i64.into_pyobject(py)?.into_any().unbind()))()),
        "echo_or_never" | "never_or_echo" => Some((|| {
            let environ =
                at(0).ok_or_else(|| pyo3::exceptions::PyTypeError::new_err("missing env"))?;
            let condition =
                at(1).ok_or_else(|| pyo3::exceptions::PyTypeError::new_err("missing condition"))?;
            let get_v =
                at(2).ok_or_else(|| pyo3::exceptions::PyTypeError::new_err("missing get_v"))?;
            let take = bool_of(&condition)?;
            let take = if name == "echo_or_never" { take } else { !take };
            if take {
                let child = sub_env(py, &environ.clone(), name)?;
                Ok(get_v.call1((child,))?.unbind())
            } else {
                never(py)
            }
        })()),
        _ => None,
    }
}
