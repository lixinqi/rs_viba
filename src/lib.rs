//! Rust implementation of viba, exported to Python through `rs_viba._native`.

use pyo3::prelude::*;

mod ast_api;
mod builder;
mod chain;
mod check_convertable_to_json_schema;
mod check_tag_and_inline;
mod convert_to_json_schema;
mod ctx;
mod descriptor_api;
mod design_walk;
mod errors;
mod extract_sub_repo;
mod generated_host;
mod interpret;
mod is_complete;
mod is_sub_type;
mod markdown;
mod native_hosts;
mod native_runtime;
mod parser;
mod partial;
mod pattern;
mod pattern_api;
mod pyast;
mod reflect;
mod repomix;
mod runtime_data;
mod serialize;
mod storage_api;
mod tagged;
mod type_api;
mod type_model;
mod type_system;
mod unparser;
mod viba_type_descriptor;

#[pyfunction]
fn parse(py: Python<'_>, source: &str) -> PyResult<Py<PyAny>> {
    parser::parse(py, source)
}

#[pyfunction(name = "unparse")]
#[pyo3(signature = (tree, indent = 2))]
fn py_unparse(py: Python<'_>, tree: &Bound<'_, PyAny>, indent: usize) -> PyResult<String> {
    unparser::unparse_module(py, tree, indent)
}

#[pyfunction]
#[pyo3(signature = (node, indent = 2))]
fn unparse_type(py: Python<'_>, node: &Bound<'_, PyAny>, indent: usize) -> PyResult<String> {
    unparser::unparse_type(py, node, indent)
}

#[pyfunction]
fn canonical(py: Python<'_>, tree: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    chain::canonical(py, tree)
}

#[pyfunction]
fn convert_to_chain_style(py: Python<'_>, node: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let nodes = pyast::Nodes::new(py)?;
    chain::convert_to_chain_style(&nodes, node)
}

#[pyfunction]
fn convert_from_chain_style(py: Python<'_>, node: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let nodes = pyast::Nodes::new(py)?;
    chain::convert_from_chain_style(&nodes, node)
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parse, m)?)?;
    m.add_function(wrap_pyfunction!(py_unparse, m)?)?;
    m.add_function(wrap_pyfunction!(unparse_type, m)?)?;
    m.add_function(wrap_pyfunction!(canonical, m)?)?;
    m.add_function(wrap_pyfunction!(convert_to_chain_style, m)?)?;
    m.add_function(wrap_pyfunction!(convert_from_chain_style, m)?)?;
    ast_api::register(m)?;
    builder::register(m)?;
    markdown::register(m)?;
    reflect::register(m)?;
    serialize::register(m)?;
    storage_api::register(m)?;
    repomix::register(m)?;
    runtime_data::register(m)?;
    type_api::register(m)?;
    type_model::register(m)?;
    type_system::register(m)?;
    viba_type_descriptor::register(m)?;
    partial::register(m)?;
    pattern_api::register(m)?;
    pattern::register(m)?;
    extract_sub_repo::register(m)?;
    check_convertable_to_json_schema::register(m)?;
    convert_to_json_schema::register(m)?;
    descriptor_api::register(m)?;
    design_walk::register(m)?;
    is_complete::register(m)?;
    check_tag_and_inline::register(m)?;
    interpret::register(m)?;
    is_sub_type::register(m)?;
    Ok(())
}
