//! The JSON Schema a design's types answer to
//! (was viba/convert_to_json_schema.py).
//!
//! The entry is the one `check_convertable_to_json_schema` takes: a module name,
//! or a group of type names. The judgment runs first — a design that cannot go to
//! JSON and come back has no schema worth writing — and its error is the answer.
//!
//! Every definition the entry reaches gets a `$defs` entry, keyed by its whole
//! name, and is referenced by `$ref` — that is what keeps a recursive type
//! finite. The root `$ref` is the first entry.

use std::collections::HashSet;

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDict, PyModule, PyString, PyTuple};

use crate::check_convertable_to_json_schema::{
    argument_modules, argument_problem, bind_choice_bindings, binding_parts, dict_obj,
    entry_definitions, generic_params, is_array_constructor, is_object_constructor,
    joined_viba_path, judge_entry, language_config, list_obj, make_runner, ref_schema,
    scalar_schema, strict_flag, tag_name, Config, Failure, Runtime, WResult,
};
use crate::pyast;
use crate::tagged::TAGGED_NAME;

const DIALECT: &str = "https://json-schema.org/draft/2020-12/schema";

/// The name a module is written under, `""` when it has none.
fn module_name(module: &Bound<'_, PyAny>) -> PyResult<String> {
    match module.getattr("name") {
        Ok(value) => {
            if value.is_none() {
                return Ok(String::new());
            }
            value.str()?.extract()
        }
        Err(_) => Ok(String::new()),
    }
}

fn null_schema(py: Python<'_>) -> PyResult<Py<PyAny>> {
    dict_obj(py, vec![("type", pyast::pystr(py, "null"))])
}

fn schema_error(message: String) -> Failure {
    Failure::Schema(message)
}

struct Builder<'a, 'py> {
    rt: &'a Runtime<'py>,
    defs: Bound<'py, PyDict>,
    config: Config,
}

impl<'a, 'py> Builder<'a, 'py> {
    fn new(rt: &'a Runtime<'py>) -> PyResult<Self> {
        Ok(Self {
            rt,
            defs: PyDict::new(rt.py),
            config: language_config(rt)?,
        })
    }

    // ---- the document ----

    fn document(&mut self, definitions: &[(String, Py<PyAny>, Py<PyAny>)]) -> WResult<Py<PyAny>> {
        let py = self.rt.py;
        let mut refs: Vec<Py<PyAny>> = Vec::new();
        for (_full_name, definition, module) in definitions {
            refs.push(self.schema_of(definition.bind(py), module.bind(py), None)?);
        }
        let document = PyDict::new(py);
        document.set_item("$schema", DIALECT)?;
        if let Some(first) = refs.first() {
            document.set_item("$ref", first.bind(py).get_item("$ref")?)?;
        }
        document.set_item("$defs", self.defs.clone())?;
        Ok(document.into_any().unbind())
    }

    fn full_name(
        &self,
        module: &Bound<'py, PyAny>,
        definition: &Bound<'py, PyAny>,
    ) -> PyResult<String> {
        let name = module_name(module)?;
        let name = if name.is_empty() {
            "?".to_string()
        } else {
            name
        };
        Ok(format!(
            "{name}.{}",
            definition.getattr("name")?.extract::<String>()?
        ))
    }

    fn key_of(
        &self,
        definition: &Bound<'py, PyAny>,
        module: &Bound<'py, PyAny>,
        bindings: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<String> {
        let full_name = self.full_name(module, definition)?;
        let Some(bindings) = bindings else {
            return Ok(full_name);
        };
        let bindings = bindings.cast::<PyDict>()?;
        if bindings.is_empty() {
            return Ok(full_name);
        }
        Ok(format!("{full_name}@{}", self.bindings_digest(bindings)?))
    }

    /// A short, stable key for the arguments a definition stands in.
    fn bindings_digest(&self, bindings: &Bound<'py, PyDict>) -> PyResult<String> {
        let py = self.rt.py;
        let mut names: Vec<String> = Vec::new();
        for (key, _value) in bindings.iter() {
            names.push(key.extract()?);
        }
        names.sort();
        let mut parts: Vec<String> = Vec::new();
        for name in names {
            let bound = bindings
                .get_item(name.as_str())?
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(name.clone()))?;
            if bound.is_none() {
                parts.push(format!("{name}=?"));
                continue;
            }
            let (node, module, _outer) = binding_parts(&bound)?.ok_or_else(|| {
                PyTypeError::new_err("a binding is no (the type, its module) pair")
            })?;
            let dump = self.rt.dump(node.bind(py))?;
            let module = module_name(module.bind(py))?;
            parts.push(format!("{name}={dump}@{module}"));
        }
        let joined = parts.join("|");
        let hashlib = PyModule::import(py, "hashlib")?;
        let digest = hashlib
            .getattr("sha1")?
            .call1((PyBytes::new(py, joined.as_bytes()),))?;
        let hex: String = digest.call_method0("hexdigest")?.extract()?;
        Ok(hex.chars().take(8).collect())
    }

    // ---- definitions ----

    fn schema_of(
        &mut self,
        definition: &Bound<'py, PyAny>,
        module: &Bound<'py, PyAny>,
        bindings: Option<&Bound<'py, PyAny>>,
    ) -> WResult<Py<PyAny>> {
        let py = self.rt.py;
        let key = self.key_of(definition, module, bindings)?;
        if self.defs.contains(key.as_str())? {
            return Ok(ref_schema(py, &key)?);
        }
        self.defs.set_item(key.as_str(), PyDict::new(py))?;
        let empty = PyDict::new(py);
        let base: &Bound<'py, PyAny> = match bindings {
            Some(value) => value,
            None => empty.as_any(),
        };
        let with = self.rt.with_params(definition, base)?;
        let body = definition.getattr("body")?;
        let schema = self.schema(&body, module, with.bind(py))?;
        let slot = self
            .defs
            .get_item(key.as_str())?
            .ok_or_else(|| schema_error(format!("the definition {key} stands nowhere")))?;
        let slot = slot.cast::<PyDict>()?;
        let schema_bound = schema.bind(py);
        match schema_bound.cast::<PyDict>() {
            Ok(mapping) => {
                for (field, value) in mapping.iter() {
                    slot.set_item(field, value)?;
                }
            }
            Err(_) => {
                return Err(Failure::Py(PyTypeError::new_err(
                    "'bool' object is not iterable",
                )))
            }
        }
        Ok(ref_schema(py, &key)?)
    }

    // ---- the pieces ----

    fn schema(
        &mut self,
        node: &Bound<'py, PyAny>,
        module: &Bound<'py, PyAny>,
        bindings: &Bound<'py, PyAny>,
    ) -> WResult<Py<PyAny>> {
        let py = self.rt.py;
        let class = self.rt.class_name(node)?;
        match class.as_str() {
            "Constant" => {
                let value = node.getattr("value")?.unbind();
                Ok(dict_obj(py, vec![("const", value)])?)
            }
            "Nil" => Ok(null_schema(py)?),
            "Never" => Ok(pyast::pybool(py, false)),
            "HintBlock" => Ok(null_schema(py)?),
            "Any" | "Ellipsis" => Err(schema_error(format!(
                "{} has no JSON Schema",
                self.rt.spell(node)
            ))),
            "TypeRef" => {
                let name: String = node.getattr("name")?.extract()?;
                self.named(&name, module, bindings)
            }
            "Tagged" => {
                let tag: String = node.getattr("tag")?.extract()?;
                let body = node.getattr("type")?;
                self.tagged(&tag, &body, module, bindings)
            }
            "MemberTaken" => self.member(node, module, bindings),
            "TypeApp" => self.application(node, module, bindings),
            "Tuple" => {
                let elements = pyast::field_list(node, "elements")?;
                let mut prefix: Vec<Py<PyAny>> = Vec::new();
                for element in &elements {
                    prefix.push(self.schema(element, module, bindings)?);
                }
                let out = PyDict::new(py);
                out.set_item("type", "array")?;
                out.set_item("prefixItems", list_obj(py, prefix)?)?;
                out.set_item("items", pyast::pybool(py, false))?;
                out.set_item("minItems", pyast::pyint(py, elements.len() as i64))?;
                Ok(out.into_any().unbind())
            }
            "Partial" | "Exponent" => Err(schema_error(format!(
                "{} is no JSON value",
                self.rt.spell(node)
            ))),
            _ => {
                let Some(elements) = self.rt.chain_elements(node)? else {
                    return Err(schema_error(format!(
                        "{} has no JSON Schema",
                        self.rt.spell(node)
                    )));
                };
                if class == "Sum" || class == "SumChain" {
                    let options = self.sum_options(&elements, module, bindings)?;
                    let out = PyDict::new(py);
                    out.set_item("oneOf", list_obj(py, options)?)?;
                    return Ok(out.into_any().unbind());
                }
                if class == "Product" || class == "ProductChain" {
                    return self.product(&elements, module, bindings);
                }
                Err(schema_error(format!(
                    "{} has no JSON Schema",
                    self.rt.spell(node)
                )))
            }
        }
    }

    /// One schema per option, a sum that stands in a sum taken together with it.
    fn sum_options(
        &mut self,
        elements: &[Bound<'py, PyAny>],
        module: &Bound<'py, PyAny>,
        bindings: &Bound<'py, PyAny>,
    ) -> WResult<Vec<Py<PyAny>>> {
        let mut options: Vec<Py<PyAny>> = Vec::new();
        for element in elements {
            if self.is_no_value(element)? {
                continue;
            }
            let (form, _form_module, _form_bindings) =
                self.stands_for(element, module, bindings)?;
            if self.is_no_value(&form)? {
                continue;
            }
            if self.rt.is_ast(element, "Sum")? || self.rt.is_ast(element, "SumChain")? {
                let inner = self.rt.chain_elements(element)?.unwrap_or_default();
                options.extend(self.sum_options(&inner, module, bindings)?);
                continue;
            }
            options.push(self.schema(element, module, bindings)?);
        }
        Ok(options)
    }

    fn named(
        &mut self,
        name: &str,
        module: &Bound<'py, PyAny>,
        bindings: &Bound<'py, PyAny>,
    ) -> WResult<Py<PyAny>> {
        let py = self.rt.py;
        let dict = bindings.cast::<PyDict>()?;
        if let Some(bound) = dict.get_item(name)? {
            if bound.is_none() {
                return Err(schema_error(format!(
                    "the generic parameter {} has no argument",
                    self.rt.repr(name)
                )));
            }
            let (node, target_module, target_bindings) =
                binding_parts(&bound)?.ok_or_else(|| {
                    schema_error(format!(
                        "the generic parameter {} has no argument",
                        self.rt.repr(name)
                    ))
                })?;
            return self.schema(
                node.bind(py),
                target_module.bind(py),
                target_bindings.bind(py),
            );
        }
        if self.config.is_null(name) {
            return Ok(null_schema(py)?);
        }
        if self.config.is_no_value(name) {
            return Ok(pyast::pybool(py, false));
        }
        if let Some(scalar) = scalar_schema(py, name)? {
            return Ok(scalar);
        }
        let found = self.rt.module_get_type(module, name)?;
        if self.rt.is_err(&found)? {
            return Err(schema_error(format!(
                "cannot resolve {} in {}: {}",
                self.rt.repr(name),
                self.rt.module_label(module)?,
                self.rt.err_msg(&found)?
            )));
        }
        let target = self.rt.ok_value(&found)?;
        let target = target.bind(py);
        if !self.rt.is_ast_node_type(target)? {
            return Err(schema_error(format!(
                "{} has no JSON Schema",
                self.rt.repr(name)
            )));
        }
        let ast_node = target.getattr("ast_node")?;
        if !self.rt.is_definition(&ast_node)? {
            return Err(schema_error(format!(
                "{} has no JSON Schema",
                self.rt.repr(name)
            )));
        }
        self.schema_of(&ast_node, &target.getattr("container_module")?, None)
    }

    fn tagged(
        &mut self,
        tag: &str,
        body: &Bound<'py, PyAny>,
        module: &Bound<'py, PyAny>,
        bindings: &Bound<'py, PyAny>,
    ) -> WResult<Py<PyAny>> {
        let py = self.rt.py;
        let key = tag_name(tag);
        let properties = PyDict::new(py);
        properties.set_item(key.as_str(), self.schema(body, module, bindings)?)?;
        let out = PyDict::new(py);
        out.set_item("type", "object")?;
        out.set_item("properties", properties)?;
        out.set_item("required", list_obj(py, vec![pyast::pystr(py, &key)])?)?;
        out.set_item("additionalProperties", pyast::pybool(py, false))?;
        Ok(out.into_any().unbind())
    }

    fn member(
        &mut self,
        node: &Bound<'py, PyAny>,
        module: &Bound<'py, PyAny>,
        bindings: &Bound<'py, PyAny>,
    ) -> WResult<Py<PyAny>> {
        let py = self.rt.py;
        if let Some(source_path) = self.rt.source_path(node)? {
            return self.named(&source_path, module, bindings);
        }
        let owner = node.getattr("owner")?;
        if !self.rt.is_ast(&owner, "TypeApp")? {
            return Err(schema_error(format!(
                "{} has no JSON Schema",
                self.rt.spell(node)
            )));
        }
        let constructor: String = owner.getattr("constructor")?.extract()?;
        let generic = self.rt.generic_named(module, &constructor)?;
        if self.rt.is_err(&generic)? {
            return Err(schema_error(format!(
                "cannot take {} in {}: {}",
                self.rt.repr(&constructor),
                self.rt.module_label(module)?,
                self.rt.err_msg(&generic)?
            )));
        }
        let generic_value = self.rt.ok_value(&generic)?;
        if generic_value.bind(py).is_none() {
            return Err(schema_error(format!(
                "{} has no JSON Schema",
                self.rt.spell(node)
            )));
        }
        let args = owner.getattr("args")?;
        let argument_modules = argument_modules(self.rt, &owner, bindings)?;
        let got = self.rt.decide(
            generic_value.bind(py),
            &args,
            module,
            argument_modules.as_ref().map(|value| value.bind(py)),
        )?;
        if self.rt.is_err(&got)? {
            return Err(schema_error(format!(
                "cannot decide {}",
                self.rt.spell(&owner)
            )));
        }
        let choice = self.rt.ok_value(&got)?;
        let choice = choice.bind(py);
        if choice.is_none() {
            return Err(schema_error(format!(
                "cannot decide {}",
                self.rt.spell(&owner)
            )));
        }
        let choice_module = choice.getattr("module")?;
        if !choice.getattr("body")?.is_none() {
            return Err(schema_error(format!(
                "{} has no JSON Schema",
                self.rt.spell(node)
            )));
        }
        let inner = bindings.cast::<PyDict>()?.copy()?;
        bind_choice_bindings(py, &inner, choice, bindings)?;
        let name: String = node.getattr("name")?.extract()?;
        let body = choice_module.getattr("module")?.getattr("body")?;
        for statement in body.try_iter()? {
            let statement = statement?;
            if self.rt.is_definition(&statement)?
                && statement.getattr("name")?.extract::<String>()? == name
            {
                return self.schema_of(&statement, &choice_module, Some(inner.as_any()));
            }
        }
        Err(schema_error(format!(
            "no definition {} in {}",
            self.rt.repr(&name),
            self.rt.module_label(&choice_module)?
        )))
    }

    fn application(
        &mut self,
        node: &Bound<'py, PyAny>,
        module: &Bound<'py, PyAny>,
        bindings: &Bound<'py, PyAny>,
    ) -> WResult<Py<PyAny>> {
        let py = self.rt.py;
        let constructor: String = node.getattr("constructor")?.extract()?;
        let args = pyast::field_list(node, "args")?;
        if constructor == TAGGED_NAME {
            let (Some(symbol_node), Some(body)) = (args.first(), args.get(1)) else {
                return Err(schema_error(format!(
                    "{} has no tag to write",
                    self.rt.spell(node)
                )));
            };
            let Some(symbol) = self.symbol(symbol_node, bindings)? else {
                return Err(schema_error(format!(
                    "{} has no tag to write",
                    self.rt.spell(node)
                )));
            };
            return self.tagged(&symbol, body, module, bindings);
        }
        if is_array_constructor(&constructor) {
            let Some(first) = args.first() else {
                return Err(schema_error(format!(
                    "{} has no JSON Schema",
                    self.rt.spell(node)
                )));
            };
            let out = PyDict::new(py);
            out.set_item("type", "array")?;
            out.set_item("items", self.schema(first, module, bindings)?)?;
            if matches!(constructor.as_str(), "set" | "SetLiteral") {
                out.set_item("uniqueItems", pyast::pybool(py, true))?;
            }
            return Ok(out.into_any().unbind());
        }
        if is_object_constructor(&constructor) {
            let Some(value) = args.get(1) else {
                return Err(schema_error(format!(
                    "{} has no JSON Schema",
                    self.rt.spell(node)
                )));
            };
            let out = PyDict::new(py);
            out.set_item("type", "object")?;
            out.set_item(
                "additionalProperties",
                self.schema(value, module, bindings)?,
            )?;
            return Ok(out.into_any().unbind());
        }
        if self.config.is_null(&constructor) {
            return Ok(null_schema(py)?);
        }
        if let Some(scalar) = scalar_schema(py, &constructor)? {
            return Ok(scalar);
        }
        let generic = self.rt.generic_named(module, &constructor)?;
        if self.rt.is_err(&generic)? {
            return Err(schema_error(format!(
                "cannot take {} in {}: {}",
                self.rt.repr(&constructor),
                self.rt.module_label(module)?,
                self.rt.err_msg(&generic)?
            )));
        }
        let generic_value = self.rt.ok_value(&generic)?;
        if generic_value.bind(py).is_none() {
            let found = self.rt.module_get_type(module, &constructor)?;
            if self.rt.is_err(&found)? {
                return Err(schema_error(format!(
                    "cannot resolve {} in {}: {}",
                    self.rt.repr(&constructor),
                    self.rt.module_label(module)?,
                    self.rt.err_msg(&found)?
                )));
            }
            let target = self.rt.ok_value(&found)?;
            let target = target.bind(py);
            if !self.rt.is_ast_node_type(target)? {
                return Err(schema_error(format!(
                    "{} has no JSON Schema",
                    self.rt.spell(node)
                )));
            }
            let definition = target.getattr("ast_node")?;
            if !self.rt.is_definition(&definition)? {
                return Err(schema_error(format!(
                    "{} has no JSON Schema",
                    self.rt.spell(node)
                )));
            }
            let params = generic_params(&definition)?;
            if params.len() != args.len() {
                return Err(schema_error(format!(
                    "{}: {} takes {} parameters",
                    self.rt.spell(node),
                    self.rt
                        .repr(&definition.getattr("name")?.extract::<String>()?),
                    params.len()
                )));
            }
            let inner = bindings.cast::<PyDict>()?.copy()?;
            for (param, argument) in params.iter().zip(args.iter()) {
                let parts = PyTuple::new(
                    py,
                    [
                        argument.clone().unbind(),
                        module.clone().unbind(),
                        bindings.clone().unbind(),
                    ],
                )?;
                inner.set_item(param.as_str(), parts)?;
            }
            return self.schema_of(
                &definition,
                &target.getattr("container_module")?,
                Some(inner.as_any()),
            );
        }
        let got = self.rt.decide(
            generic_value.bind(py),
            &node.getattr("args")?,
            module,
            argument_modules(self.rt, node, bindings)?
                .as_ref()
                .map(|value| value.bind(py)),
        )?;
        if self.rt.is_err(&got)? {
            return Err(schema_error(format!(
                "cannot decide {}: {}",
                self.rt.spell(node),
                self.rt.err_msg(&got)?
            )));
        }
        let choice = self.rt.ok_value(&got)?;
        let choice = choice.bind(py);
        if choice.is_none() || choice.getattr("body")?.is_none() {
            return Err(schema_error(format!(
                "{}: the chosen file answers a member",
                self.rt.spell(node)
            )));
        }
        let inner = bindings.cast::<PyDict>()?.copy()?;
        bind_choice_bindings(py, &inner, choice, bindings)?;
        self.schema(
            &choice.getattr("body")?,
            &choice.getattr("module")?,
            inner.as_any(),
        )
    }

    /// The product's schema: its tagged members, or the one member it keeps.
    fn product(
        &mut self,
        elements: &[Bound<'py, PyAny>],
        module: &Bound<'py, PyAny>,
        bindings: &Bound<'py, PyAny>,
    ) -> WResult<Py<PyAny>> {
        let py = self.rt.py;
        let mut kept: Vec<Bound<'py, PyAny>> = Vec::new();
        for element in elements {
            if self.rt.is_ast(element, "Tagged")? {
                kept.push(element.clone());
                continue;
            }
            let (form, _form_module, _form_bindings) =
                self.stands_for(element, module, bindings)?;
            if self.is_unit(&form)? {
                continue;
            }
            kept.push(element.clone());
        }
        if kept.is_empty() {
            return Ok(null_schema(py)?);
        }
        if kept.len() == 1 && !self.rt.is_ast(&kept[0], "Tagged")? {
            let element = kept[0].clone();
            return self.schema(&element, module, bindings);
        }
        let properties = PyDict::new(py);
        let mut required: Vec<Py<PyAny>> = Vec::new();
        self.members(&kept, module, bindings, &properties, &mut required)?;
        let out = PyDict::new(py);
        out.set_item("type", "object")?;
        out.set_item("properties", properties)?;
        out.set_item("required", list_obj(py, required)?)?;
        out.set_item("additionalProperties", pyast::pybool(py, false))?;
        Ok(out.into_any().unbind())
    }

    fn members(
        &mut self,
        elements: &[Bound<'py, PyAny>],
        module: &Bound<'py, PyAny>,
        bindings: &Bound<'py, PyAny>,
        properties: &Bound<'py, PyDict>,
        required: &mut Vec<Py<PyAny>>,
    ) -> WResult<()> {
        let py = self.rt.py;
        for element in elements {
            if self.rt.is_ast(element, "Tagged")? {
                let key = tag_name(&element.getattr("tag")?.extract::<String>()?);
                let body = element.getattr("type")?;
                properties.set_item(key.as_str(), self.schema(&body, module, bindings)?)?;
                required.push(pyast::pystr(py, &key));
                continue;
            }
            let (form, form_module, form_bindings) = self.stands_for(element, module, bindings)?;
            if self.is_unit(&form)? {
                continue; // a unit is no member
            }
            let inner = self.rt.chain_elements(&form)?;
            if let Some(mut inner) = inner {
                if self.rt.is_ast(&form, "Product")? || self.rt.is_ast(&form, "ProductChain")? {
                    if let Some(head) = inner.first().cloned() {
                        let rest = inner[1..].to_vec();
                        if self.is_unit(&head)? {
                            inner = rest;
                        }
                    }
                    self.members(&inner, &form_module, &form_bindings, properties, required)?;
                    continue;
                }
            }
            return Err(schema_error(format!(
                "the product's member {} has no tag: a JSON object needs a key",
                self.rt.spell(element)
            )));
        }
        Ok(())
    }

    /// The form an inline slot stands for: (the form, the module, the bindings).
    fn stands_for(
        &mut self,
        node: &Bound<'py, PyAny>,
        module: &Bound<'py, PyAny>,
        bindings: &Bound<'py, PyAny>,
    ) -> WResult<(Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        let py = self.rt.py;
        let mut seen: HashSet<usize> = HashSet::new();
        let mut node = node.clone();
        let mut module = module.clone();
        let mut bindings = bindings.clone();
        loop {
            let id = node.as_ptr() as usize;
            if seen.contains(&id) {
                return Ok((node, module, bindings));
            }
            seen.insert(id);
            let class = self.rt.class_name(&node)?;
            if class == "TypeRef" {
                let name: String = node.getattr("name")?.extract()?;
                let dict = bindings.cast::<PyDict>()?;
                if let Some(bound) = dict.get_item(name.as_str())? {
                    if let Some((inner, inner_module, inner_bindings)) = binding_parts(&bound)? {
                        node = inner.bind(py).clone();
                        module = inner_module.bind(py).clone();
                        bindings = inner_bindings.bind(py).clone();
                        continue;
                    }
                }
                if dict.contains(name.as_str())? {
                    return Ok((node, module, bindings));
                }
                let found = self.rt.module_get_type(&module, &name)?;
                if self.rt.is_err(&found)? {
                    return Err(schema_error(format!(
                        "cannot resolve {} in {}: {}",
                        self.rt.repr(&name),
                        self.rt.module_label(&module)?,
                        self.rt.err_msg(&found)?
                    )));
                }
                let target = self.rt.ok_value(&found)?;
                let target = target.bind(py);
                if !self.rt.is_ast_node_type(target)? {
                    return Ok((node, module, bindings));
                }
                let definition = target.getattr("ast_node")?;
                if !self.rt.is_definition(&definition)? {
                    return Ok((node, module, bindings));
                }
                node = definition.getattr("body")?;
                module = target.getattr("container_module")?;
                let empty = PyDict::new(py);
                bindings = self
                    .rt
                    .with_params(&definition, empty.as_any())?
                    .bind(py)
                    .clone();
                continue;
            }
            if class == "TypeApp" {
                let constructor: String = node.getattr("constructor")?.extract()?;
                if constructor == TAGGED_NAME
                    || is_array_constructor(&constructor)
                    || is_object_constructor(&constructor)
                {
                    return Ok((node, module, bindings));
                }
                let generic = self.rt.generic_named(&module, &constructor)?;
                if self.rt.is_err(&generic)? {
                    return Err(schema_error(format!(
                        "cannot take {} in {}: {}",
                        self.rt.repr(&constructor),
                        self.rt.module_label(&module)?,
                        self.rt.err_msg(&generic)?
                    )));
                }
                let generic_value = self.rt.ok_value(&generic)?;
                if generic_value.bind(py).is_none() {
                    return Ok((node, module, bindings));
                }
                let args = node.getattr("args")?;
                let argument_modules = argument_modules(self.rt, &node, &bindings)?;
                let got = self.rt.decide(
                    generic_value.bind(py),
                    &args,
                    &module,
                    argument_modules.as_ref().map(|value| value.bind(py)),
                )?;
                if self.rt.is_err(&got)? {
                    return Ok((node, module, bindings));
                }
                let choice = self.rt.ok_value(&got)?;
                let choice = choice.bind(py);
                if choice.is_none() || choice.getattr("body")?.is_none() {
                    return Ok((node, module, bindings));
                }
                let inner = bindings.cast::<PyDict>()?.copy()?;
                bind_choice_bindings(py, &inner, choice, &bindings)?;
                node = choice.getattr("body")?;
                module = choice.getattr("module")?;
                bindings = inner.into_any();
                continue;
            }
            return Ok((node, module, bindings));
        }
    }

    /// A unit: `nil`/`Object` under any of its names, or `never`/`Oneof`.
    fn is_unit(&mut self, form: &Bound<'py, PyAny>) -> WResult<bool> {
        let class = self.rt.class_name(form)?;
        if matches!(class.as_str(), "Nil" | "Never" | "HintBlock") {
            return Ok(true);
        }
        if class == "TypeRef" {
            let name: String = form.getattr("name")?.extract()?;
            return Ok(self.config.is_unit(&name));
        }
        if class == "TypeApp" {
            let name: String = form.getattr("constructor")?.extract()?;
            return Ok(self.config.is_unit(&name));
        }
        Ok(false)
    }

    /// The sum's unit: `never` itself, or a name standing for it.
    fn is_no_value(&mut self, form: &Bound<'py, PyAny>) -> WResult<bool> {
        let class = self.rt.class_name(form)?;
        if class == "Never" {
            return Ok(true);
        }
        if class == "TypeRef" {
            let name: String = form.getattr("name")?.extract()?;
            return Ok(self.config.is_no_value(&name));
        }
        if class == "TypeApp" {
            let name: String = form.getattr("constructor")?.extract()?;
            return Ok(self.config.is_no_value(&name));
        }
        Ok(false)
    }

    fn symbol(
        &mut self,
        node: &Bound<'py, PyAny>,
        bindings: &Bound<'py, PyAny>,
    ) -> WResult<Option<String>> {
        let py = self.rt.py;
        if self.rt.is_ast(node, "Constant")? {
            let value = node.getattr("value")?;
            if value.is_instance_of::<PyString>() {
                return Ok(Some(value.extract()?));
            }
        }
        if self.rt.is_ast(node, "TypeRef")? {
            let name: String = node.getattr("name")?.extract()?;
            if let Some(bound) = bindings.cast::<PyDict>()?.get_item(name.as_str())? {
                if let Some((inner, _module, inner_bindings)) = binding_parts(&bound)? {
                    return self.symbol(inner.bind(py), inner_bindings.bind(py));
                }
            }
        }
        Ok(None)
    }
}

fn json_text(rt: &Runtime<'_>, document: &Py<PyAny>) -> PyResult<String> {
    let py = rt.py;
    let json = PyModule::import(py, "json")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("ensure_ascii", false)?;
    kwargs.set_item("indent", 2)?;
    json.getattr("dumps")?
        .call((document.bind(py),), Some(&kwargs))?
        .extract()
}

fn convert_walk<'py>(
    rt: &Runtime<'py>,
    loader: &Bound<'py, PyAny>,
    viba_path: &Bound<'py, PyAny>,
    entry: &Bound<'py, PyAny>,
    strict: Option<&Bound<'py, PyAny>>,
    strict_value: bool,
    tag: &str,
    value: &Bound<'py, PyAny>,
) -> WResult<Py<PyAny>> {
    let py = rt.py;
    let mut builder = Builder::new(rt)?;
    let definitions = entry_definitions(rt, loader, tag, value, strict_value)?;
    let checked = judge_entry(py, viba_path, entry, strict)?;
    if rt.is_err(&checked)? {
        return Ok(checked);
    }
    let mut entries: Vec<(String, Py<PyAny>, Py<PyAny>)> = Vec::new();
    for (definition, module) in &definitions {
        let full_name = builder.full_name(module.bind(py), definition.bind(py))?;
        entries.push((full_name, definition.clone_ref(py), module.clone_ref(py)));
    }
    let document = builder.document(&entries)?;
    let text = json_text(rt, &document)?;
    let answer = pyast::pystr(py, &format!("{text}\n"));
    Ok(rt.make_ok(&answer)?)
}

/// `convert_to_json_schema`: the schema as JSON text, or why there is none.
pub(crate) fn convert_entry<'py>(
    py: Python<'py>,
    viba_path: &Bound<'py, PyAny>,
    entry: &Bound<'py, PyAny>,
    strict: Option<&Bound<'py, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let rt = Runtime::new(py)?;
    if let Some(value) = strict {
        if !value.is_none() && !value.is_instance_of::<PyBool>() {
            let message = format!("strict is a bool or nil, not {}", rt.type_name(value)?);
            return rt.make_err(&message);
        }
    }
    if let Some(problem) = argument_problem(py, viba_path, entry)? {
        return rt.make_err(&problem);
    }
    let tuple = entry.cast::<PyTuple>()?;
    let tag: String = tuple.get_item(0)?.extract()?;
    let tag = tag.trim_start_matches('$').to_string();
    let value = tuple.get_item(1)?;
    let joined = joined_viba_path(viba_path)?;
    let loader = make_runner(py, joined.as_deref())?;
    let strict_value = strict_flag(strict)?;
    match convert_walk(
        &rt,
        loader.bind(py),
        viba_path,
        entry,
        strict,
        strict_value,
        &tag,
        &value,
    ) {
        Ok(result) => Ok(result),
        Err(Failure::Py(error)) => Err(error),
        Err(other) => rt.make_err(&other.message()),
    }
}

#[pyfunction]
#[pyo3(signature = (viba_path, entry, strict = None))]
fn convert_to_json_schema_py<'py>(
    py: Python<'py>,
    viba_path: &Bound<'py, PyAny>,
    entry: &Bound<'py, PyAny>,
    strict: Option<&Bound<'py, PyAny>>,
) -> PyResult<Py<PyAny>> {
    convert_entry(py, viba_path, entry, strict)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(convert_to_json_schema_py, m)?)?;
    Ok(())
}
