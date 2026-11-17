//! A design's tags, and the chains that spread them.
//!
//! A port of `viba/check_tag_and_inline.py`: the walk is here; the map it walks
//! (`VibaAccess`, the descriptor kinds) is called through the Python bindings
//! of the layers ported in parallel.

use std::collections::HashSet;

use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule};

type Obj = Py<PyAny>;

enum Failure {
    DuplicateTag(String),
    InlineCycle(String),
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct Key {
    node: usize,
    arguments: Vec<Option<usize>>,
    frame: Option<usize>,
}

struct Checker<'py> {
    py: Python<'py>,
    access: Obj,
    kinds: Kinds,
    visited: HashSet<Key>,
}

struct Kinds {
    type_app: String,
    product: String,
    sum: String,
    exponent: String,
    tagged: String,
    tuple: String,
}

impl<'py> Checker<'py> {
    fn check(&mut self, design: &Bound<'py, PyAny>) -> PyResult<Option<Failure>> {
        for file in design.getattr("files")?.try_iter()? {
            let file = file?;
            for definition in file.getattr("definitions")?.try_iter()? {
                let definition = definition?;
                let body = definition.getattr("body")?;
                if let Some(failure) = self.check_piece(&body, None)? {
                    return Ok(Some(failure));
                }
            }
        }
        Ok(None)
    }

    /// Look at this piece, and at everything under it.
    fn check_piece(
        &mut self,
        descriptor: &Bound<'py, PyAny>,
        frame: Option<usize>,
    ) -> PyResult<Option<Failure>> {
        if let Some(key) = self.key(descriptor, frame)? {
            if !self.visited.insert(key) {
                return Ok(None);
            }
        }
        if self.kind(descriptor)? == self.kinds.type_app {
            return self.check_application(descriptor, frame);
        }
        let unfolded = self
            .access
            .bind(self.py)
            .getattr("unfold")?
            .call1((descriptor,))?;
        if self.kind(&unfolded)? == self.kinds.type_app {
            return self.check_piece(&unfolded, frame);
        }
        self.dispatch(&unfolded, frame)
    }

    /// An application: its arguments as they stand, then the body they make,
    /// taken under this application.
    fn check_application(
        &mut self,
        descriptor: &Bound<'py, PyAny>,
        frame: Option<usize>,
    ) -> PyResult<Option<Failure>> {
        let payload = descriptor.getattr("payload")?;
        for argument in payload.getattr("args")?.try_iter()? {
            let argument = argument?;
            if let Some(failure) = self.check_piece(&argument, frame)? {
                return Ok(Some(failure));
            }
        }
        let body = self
            .access
            .bind(self.py)
            .getattr("unfold")?
            .call1((descriptor,))?;
        if self.kind(&body)? == self.kinds.type_app {
            return Ok(None);
        }
        let inner = self.node_id(descriptor)?;
        self.check_piece(&body, inner)
    }

    fn dispatch(
        &mut self,
        unfolded: &Bound<'py, PyAny>,
        frame: Option<usize>,
    ) -> PyResult<Option<Failure>> {
        let kind = self.kind(unfolded)?;
        if kind == self.kinds.product {
            return self.check_product(unfolded, frame);
        }
        if kind == self.kinds.sum || kind == self.kinds.exponent {
            let payload = unfolded.getattr("payload")?;
            for element in payload.getattr("elements")?.try_iter()? {
                let element = element?;
                if let Some(failure) = self.check_piece(&element, frame)? {
                    return Ok(Some(failure));
                }
            }
            return Ok(None);
        }
        if kind == self.kinds.tagged {
            let payload = unfolded.getattr("payload")?;
            let tagged_type = payload.getattr("tagged_type")?;
            return self.check_piece(&tagged_type, frame);
        }
        if kind == self.kinds.tuple {
            let payload = unfolded.getattr("payload")?;
            for element in payload.getattr("elements")?.try_iter()? {
                let element = element?;
                if let Some(failure) = self.check_piece(&element, frame)? {
                    return Ok(Some(failure));
                }
            }
        }
        Ok(None)
    }

    /// One product: its chain has to end, and its tags — the inlined ones
    /// included — have to be all different.
    fn check_product(
        &mut self,
        descriptor: &Bound<'py, PyAny>,
        frame: Option<usize>,
    ) -> PyResult<Option<Failure>> {
        let cycle = self
            .access
            .bind(self.py)
            .getattr("inline_cycle")?
            .call1((descriptor,))?;
        if !cycle.is_none() {
            let name = crate::pyast::repr_of(self.py, &cycle);
            return Ok(Some(Failure::InlineCycle(format!(
                "the inline chain comes back to {}",
                name
            ))));
        }
        let members = self
            .access
            .bind(self.py)
            .getattr("members_of")?
            .call1((descriptor,))?;
        let mut tags: HashSet<String> = HashSet::new();
        if !members.is_none() {
            for member in members.try_iter()? {
                let member = member?;
                let tag = member.get_item(0)?;
                let piece = member.get_item(1)?;
                if !tag.is_none() {
                    let tag: String = tag.extract()?;
                    if !tags.insert(tag.clone()) {
                        return Ok(Some(Failure::DuplicateTag(format!(
                            "the tag {} appears twice in one product",
                            tag
                        ))));
                    }
                }
                if let Some(failure) = self.check_piece(&piece, frame)? {
                    return Ok(Some(failure));
                }
            }
        }
        Ok(None)
    }

    /// What makes this piece the same question: the node it stands as, the
    /// nodes its arguments stand as (for an application), and the application
    /// it sits under.
    fn key(
        &mut self,
        descriptor: &Bound<'py, PyAny>,
        frame: Option<usize>,
    ) -> PyResult<Option<Key>> {
        let node = match self.node_id(descriptor)? {
            Some(node) => node,
            None => return Ok(None),
        };
        let mut arguments = Vec::new();
        if self.kind(descriptor)? == self.kinds.type_app {
            let payload = descriptor.getattr("payload")?;
            for argument in payload.getattr("args")?.try_iter()? {
                let argument = argument?;
                arguments.push(self.node_id(&argument)?);
            }
        }
        Ok(Some(Key {
            node,
            arguments,
            frame,
        }))
    }

    fn node_id(&mut self, descriptor: &Bound<'py, PyAny>) -> PyResult<Option<usize>> {
        let payload = match descriptor.getattr("payload") {
            Ok(payload) => payload,
            Err(_) => return Ok(None),
        };
        if payload.is_none() {
            return Ok(None);
        }
        let resolvable = match payload.getattr("resolvable_type") {
            Ok(resolvable) => resolvable,
            Err(error) if error.is_instance_of::<pyo3::exceptions::PyAttributeError>(self.py) => {
                return Ok(None);
            }
            Err(error) => return Err(error),
        };
        if resolvable.is_none() {
            return Ok(None);
        }
        let ast_node = match resolvable.getattr("ast_node") {
            Ok(ast_node) => ast_node,
            Err(error) if error.is_instance_of::<pyo3::exceptions::PyAttributeError>(self.py) => {
                return Ok(None);
            }
            Err(error) => return Err(error),
        };
        if ast_node.is_none() {
            return Ok(None);
        }
        Ok(Some(ast_node.as_ptr() as usize))
    }

    fn kind(&mut self, descriptor: &Bound<'py, PyAny>) -> PyResult<String> {
        descriptor.getattr("kind")?.extract()
    }
}

fn kinds(py: Python<'_>) -> PyResult<(Kinds, Bound<'_, PyModule>)> {
    let descriptor = PyModule::import(py, "rs_viba.viba_type_descriptor")?;
    let kind = |name: &str| -> PyResult<String> { descriptor.getattr(name)?.extract() };
    Ok((
        Kinds {
            type_app: kind("TYPE_APP")?,
            product: kind("PRODUCT")?,
            sum: kind("SUM")?,
            exponent: kind("EXPONENT")?,
            tagged: kind("TAGGED")?,
            tuple: kind("TUPLE")?,
        },
        descriptor,
    ))
}

#[pyfunction]
#[pyo3(signature = (design, config = None))]
fn check_tag_and_inline_py(
    py: Python<'_>,
    design: &Bound<'_, PyAny>,
    config: Option<&Bound<'_, PyAny>>,
) -> PyResult<Obj> {
    let reflect = PyModule::import(py, "rs_viba.reflect")?;
    let config = match config {
        Some(config) => config.clone(),
        None => reflect.getattr("language_config")?,
    };
    let access = reflect.getattr("VibaAccess")?.call1((config,))?.unbind();
    let (kinds, _descriptor) = kinds(py)?;
    let mut checker = Checker {
        py,
        access,
        kinds,
        visited: HashSet::new(),
    };
    match checker.check(design) {
        Ok(None) => crate::type_system::ok(py, py.None()),
        Ok(Some(failure)) => {
            let message = match failure {
                Failure::DuplicateTag(message) | Failure::InlineCycle(message) => message,
            };
            crate::type_system::err(py, &message)
        }
        Err(error) => {
            // The checker raises nothing itself; a mistake the accessor reports
            // as one of the design's own errors is still answered as `Err`.
            let type_module = PyModule::import(py, "rs_viba.type")?;
            let mut mistake = None;
            for class_name in ["DuplicateTagError", "InlineCycleError"] {
                if let Ok(class) = type_module.getattr(class_name) {
                    if error.is_instance(py, &class) {
                        mistake = Some(error.value(py).str()?.extract::<String>()?);
                        break;
                    }
                }
            }
            match mistake {
                Some(message) => crate::type_system::err(py, &message),
                None => Err(error),
            }
        }
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(check_tag_and_inline_py, m)?)?;
    Ok(())
}
