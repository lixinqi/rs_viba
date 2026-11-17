//! `T << X`: partial application, as the source has it (was viba/partial.py).

use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

use crate::ctx::Ctx;
use crate::errors::{VibaError, VibaResult};
use crate::tagged::{self, GET_ATTR_TAG, GET_ITEM_TAG, INTERPRETER_MEMBER_TAGS};
use crate::type_system;

pub const RET_NAME: &str = "__impl__";
pub const DEF_NAME: &str = "__decl__";
pub const GET_ARGS_NAME: &str = "__get_args__";
pub const ENVIRON_TAG: &str = "$env";
pub const ENV_TYPE: &str = "Env";
pub const ENVIRONMENT_TYPE: &str = "Environment";
pub const DYN_CALL_NAME: &str = "__dyn_call__";
pub const DYN_METHOD_NAME: &str = "__dyn_method__";

pub trait ResolveFn {
    fn resolve(
        &self,
        ctx: &Ctx<'_>,
        name: &str,
        module: &Bound<'_, PyAny>,
    ) -> VibaResult<Option<(Py<PyAny>, Py<PyAny>)>>;
}

pub trait JudgeFn {
    fn judge(
        &self,
        ctx: &Ctx<'_>,
        sub: &Bound<'_, PyAny>,
        sub_module: &Bound<'_, PyAny>,
        sup: &Bound<'_, PyAny>,
        sup_module: &Bound<'_, PyAny>,
    ) -> VibaResult<bool>;
}

#[pyclass(name = "_NamedMember")]
pub struct NamedMember {
    #[pyo3(get)]
    pub owner: Py<PyAny>,
    #[pyo3(get)]
    pub module: Py<PyAny>,
}

fn is(ctx: &Ctx<'_>, node: &Bound<'_, PyAny>, name: &str) -> bool {
    ctx.class_name(node)
        .map(|value| value == name)
        .unwrap_or(false)
}

fn attrs(ctx: &Ctx<'_>, node: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    let mut names = Vec::new();
    for field in node.getattr("_fields")?.try_iter()? {
        names.push(field?.extract()?);
    }
    Ok(names)
}

pub fn elements(ctx: &Ctx<'_>, node: &Bound<'_, PyAny>) -> PyResult<Vec<Py<PyAny>>> {
    if ctx.is(node, "ExponentChain")? {
        return Ok(crate::pyast::field_list(node, "elements")?
            .into_iter()
            .map(|item| item.unbind())
            .collect());
    }
    if ctx.is(node, "Exponent")? {
        let mut result = elements(ctx, &node.getattr("result")?)?;
        result.push(node.getattr("argument")?.unbind());
        return Ok(result);
    }
    Ok(vec![node.clone().unbind()])
}

pub struct Param {
    pub tag: Option<String>,
    pub source_form: Py<PyAny>,
    pub is_env: bool,
    pub slot: Option<usize>,
}

pub fn parameters_of(ctx: &Ctx<'_>, chain: &Bound<'_, PyAny>) -> PyResult<Vec<Param>> {
    let mut items = elements(ctx, chain)?;
    let is_body = |node: &Bound<'_, PyAny>| -> PyResult<bool> {
        Ok(ctx.is(node, "HintBlock")? || ctx.is(node, "Partial")?)
    };
    if !items.is_empty() {
        let last = items.last().unwrap().bind(ctx.py);
        if is_body(last)? {
            items.pop();
        }
    }
    let mut params = Vec::new();
    let mut slot = 0usize;
    for item in &items[1..] {
        let node = item.bind(ctx.py);
        if ctx.is(node, "HintBlock")? {
            continue;
        }
        let (tag, source_form) = if ctx.is(node, "Tagged")? {
            (
                Some(node.getattr("tag")?.extract::<String>()?),
                node.getattr("type")?,
            )
        } else {
            (None, node.clone())
        };
        if tag.as_deref() == Some(ENVIRON_TAG) {
            params.push(Param {
                tag,
                source_form: source_form.unbind(),
                is_env: true,
                slot: None,
            });
        } else {
            params.push(Param {
                tag,
                source_form: source_form.unbind(),
                is_env: false,
                slot: Some(slot),
            });
            slot += 1;
        }
    }
    Ok(params)
}

pub fn result_of(ctx: &Ctx<'_>, chain: &Bound<'_, PyAny>) -> PyResult<Option<Py<PyAny>>> {
    Ok(elements(ctx, chain)?.into_iter().next())
}

pub fn answers_the_environment(ctx: &Ctx<'_>, chain: &Bound<'_, PyAny>) -> PyResult<bool> {
    match result_of(ctx, chain)? {
        Some(result) => names_the_environment(ctx, result.bind(ctx.py)),
        None => Ok(false),
    }
}

pub fn names_the_environment(ctx: &Ctx<'_>, node: &Bound<'_, PyAny>) -> PyResult<bool> {
    let mut node = node.clone();
    if ctx.is(&node, "Tagged")? {
        if node.getattr("tag")?.extract::<String>()? == ENVIRON_TAG {
            return Ok(true);
        }
        node = node.getattr("type")?;
    }
    if ctx.is(&node, "TypeRef")? {
        let name: String = node.getattr("name")?.extract()?;
        let last = name.split('.').next_back().unwrap_or("");
        return Ok(last == ENV_TYPE || last == ENVIRONMENT_TYPE);
    }
    Ok(false)
}

pub fn environment_result_problem(
    ctx: &Ctx<'_>,
    chain: &Bound<'_, PyAny>,
    name: &str,
) -> PyResult<Option<String>> {
    if !answers_the_environment(ctx, chain)? {
        return Ok(None);
    }
    Ok(Some(format!(
        "{} answers the environment: {} is the call's rule rather than a value, and only a builtin function may answer one",
        name, ENV_TYPE
    )))
}

pub fn file_environment_result_problem(
    ctx: &Ctx<'_>,
    tree: &Bound<'_, PyAny>,
) -> PyResult<Option<String>> {
    let body = match tree.getattr("body") {
        Ok(body) => body,
        Err(_) => return Ok(None),
    };
    for statement in body.try_iter()? {
        let statement = statement?;
        let class = ctx.class_name(&statement)?;
        if class != "TypeDefinition" && class != "GenericDefinition" {
            continue;
        }
        let name: String = statement.getattr("name")?.extract()?;
        let body = statement.getattr("body")?;
        for node in walk(ctx, &body)? {
            let node = node.bind(ctx.py);
            if !(ctx.is(node, "Exponent")? || ctx.is(node, "ExponentChain")?) {
                continue;
            }
            if answers_the_environment(ctx, node)? {
                return Ok(environment_result_problem(ctx, node, &name)?);
            }
        }
    }
    Ok(None)
}

pub fn parameter_node(
    ctx: &Ctx<'_>,
    tag: Option<&str>,
    source_form: &Py<PyAny>,
) -> PyResult<Py<PyAny>> {
    match tag {
        Some(tag) => ctx.make(
            "Tagged",
            &[
                ("tag", ctx.pystr(tag)),
                ("type", source_form.clone_ref(ctx.py)),
            ],
        ),
        None => Ok(source_form.clone_ref(ctx.py)),
    }
}

pub fn get_args_product(ctx: &Ctx<'_>, chain: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let params = parameters_of(ctx, chain)?;
    let mut members = Vec::new();
    for param in params {
        members.push(parameter_node(
            ctx,
            param.tag.as_deref(),
            &param.source_form,
        )?);
    }
    if members.is_empty() {
        return ctx.make0("Nil");
    }
    if members.len() == 1 {
        return Ok(members.remove(0));
    }
    let members = ctx.list(members)?;
    ctx.make("ProductChain", &[("elements", members)])
}

pub fn product_elements(ctx: &Ctx<'_>, node: &Bound<'_, PyAny>) -> PyResult<Vec<Py<PyAny>>> {
    if ctx.is(node, "Product")? {
        let mut left = product_elements(ctx, &node.getattr("left")?)?;
        left.extend(product_elements(ctx, &node.getattr("right")?)?);
        return Ok(left);
    }
    if ctx.is(node, "ProductChain")? {
        return Ok(crate::pyast::field_list(node, "elements")?
            .into_iter()
            .map(|item| item.unbind())
            .collect());
    }
    Ok(vec![node.clone().unbind()])
}

pub fn definition(
    ctx: &Ctx<'_>,
    module: &Bound<'_, PyAny>,
    name: &str,
) -> PyResult<Option<Py<PyAny>>> {
    let tree = module.getattr("module").unwrap_or_else(|_| module.clone());
    let body = match tree.getattr("body") {
        Ok(body) => body,
        Err(_) => return Ok(None),
    };
    let mut found: Option<Py<PyAny>> = None;
    for node in body.try_iter()? {
        let node = node?;
        if let Ok(node_name) = node.getattr("name") {
            if let Ok(node_name) = node_name.extract::<String>() {
                if node_name == name {
                    found = Some(node.unbind());
                }
            }
        }
    }
    Ok(found)
}

fn dynamic_call(ctx: &Ctx<'_>, node: &Bound<'_, PyAny>) -> PyResult<bool> {
    let mut node = node.clone();
    while ctx.is(&node, "Partial")? {
        node = node.getattr("function")?;
    }
    if ctx.is(&node, "TypeRef")? {
        let name: String = node.getattr("name")?.extract()?;
        return Ok(name == DYN_CALL_NAME || name == DYN_METHOD_NAME);
    }
    Ok(false)
}

pub fn module_as_function(
    ctx: &Ctx<'_>,
    module: &Bound<'_, PyAny>,
    name: &str,
) -> VibaResult<Option<(Py<PyAny>, Py<PyAny>)>> {
    let imports = match module.getattr("imports") {
        Ok(imports) => imports,
        Err(_) => return Ok(None),
    };
    let imports = match imports.cast::<pyo3::types::PyDict>() {
        Ok(imports) => imports,
        Err(_) => return Ok(None),
    };
    let import_name = match imports.get_item(name) {
        Ok(Some(value)) => value,
        _ => return Ok(None),
    };
    let imported = module
        .getattr("module_environment")?
        .call1((import_name,))?;
    if type_system::is_err(&imported)? {
        return Ok(None);
    }
    let imported = imported.getattr("ok_value")?;
    if definition(ctx, &imported, RET_NAME)?.is_none() {
        return Ok(None);
    }
    let declared = match definition(ctx, &imported, DEF_NAME)? {
        Some(declared) => declared,
        None => return Ok(None),
    };
    let declared = declared.bind(ctx.py);
    let chain = declared.getattr("body")?;
    if !(ctx.is(&chain, "Exponent")? || ctx.is(&chain, "ExponentChain")?) {
        let text = crate::unparser::unparse_type(ctx.py, &chain, 2)?
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        return Err(VibaError::Partial(format!(
            "__decl__ is not a function type: {}",
            text
        )));
    }
    let result = match result_of(ctx, &chain)? {
        Some(result) => result,
        None => return Ok(None),
    };
    let problem = environment_result_problem(
        ctx,
        &chain,
        &format!(
            "module {}",
            crate::pyast::repr_of(ctx.py, &ctx.pystr(name).bind(ctx.py).clone())
        ),
    )?;
    if let Some(problem) = problem {
        return Err(VibaError::Partial(problem));
    }
    let params = parameters_of(ctx, &chain)?;
    let mut slots = Vec::new();
    for param in params {
        if param.is_env {
            continue;
        }
        slots.push(parameter_node(
            ctx,
            param.tag.as_deref(),
            &param.source_form,
        )?);
    }
    if slots.is_empty() {
        return Ok(Some((result, imported.unbind())));
    }
    let mut chain_elements = vec![result];
    chain_elements.extend(slots);
    let elements = ctx.list(chain_elements)?;
    let chain = ctx.make("ExponentChain", &[("elements", elements)])?;
    Ok(Some((chain, imported.unbind())))
}

fn unfold<'py>(
    ctx: &Ctx<'py>,
    node: &Bound<'py, PyAny>,
    module: &Bound<'py, PyAny>,
    resolve: &dyn ResolveFn,
) -> VibaResult<(Py<PyAny>, Py<PyAny>)> {
    let mut node = node.clone().unbind();
    let mut module = module.clone().unbind();
    let mut seen = std::collections::HashSet::new();
    loop {
        let bound = node.bind(ctx.py);
        if ctx.is(bound, "MemberTaken")? {
            let path = source_path(ctx, bound)?;
            match path {
                Some(path) => node = ctx.make("TypeRef", &[("name", ctx.pystr(&path))])?,
                None => return Ok((node.clone_ref(ctx.py), module)),
            }
            continue;
        }
        if !ctx.is(bound, "TypeRef")? {
            return Ok((node.clone_ref(ctx.py), module));
        }
        let name: String = bound.getattr("name")?.extract()?;
        if seen.contains(&name) {
            return Ok((node.clone_ref(ctx.py), module));
        }
        seen.insert(name.clone());
        match resolve.resolve(ctx, &name, &module.bind(ctx.py))? {
            Some((new_node, new_module)) => {
                node = new_node;
                module = new_module;
            }
            None => return Ok((node, module)),
        }
    }
}

fn source_path(ctx: &Ctx<'_>, node: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
    if ctx.is(node, "TypeRef")? {
        return Ok(Some(node.getattr("name")?.extract()?));
    }
    if ctx.is(node, "MemberTaken")? {
        let owner = source_path(ctx, &node.getattr("owner")?)?;
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

fn source_form(ctx: &Ctx<'_>, node: &Bound<'_, PyAny>) -> PyResult<String> {
    let text = crate::unparser::unparse_type(ctx.py, node, 2)?;
    Ok(text.split_whitespace().collect::<Vec<_>>().join(" "))
}

fn is_documentation(ctx: &Ctx<'_>, node: &Bound<'_, PyAny>) -> PyResult<bool> {
    for item in walk(ctx, node)? {
        if ctx.is(&item.bind(ctx.py), "HintBlock")? {
            return Ok(true);
        }
    }
    Ok(false)
}

pub fn walk<'py>(ctx: &Ctx<'py>, node: &Bound<'py, PyAny>) -> PyResult<Vec<Py<PyAny>>> {
    let mut queue = std::collections::VecDeque::new();
    queue.push_back(node.clone().unbind());
    let mut out = Vec::new();
    while let Some(current) = queue.pop_front() {
        let current_bound = current.bind(ctx.py);
        out.push(current.clone_ref(ctx.py));
        for name in attrs(ctx, current_bound)? {
            let value = current_bound.getattr(name.as_str())?;
            if value.is_instance_of::<PyList>() {
                for item in value.try_iter()? {
                    let item = item?;
                    if item.get_type().hasattr("_fields").unwrap_or(false) {
                        queue.push_back(item.unbind());
                    }
                }
            } else if value.get_type().hasattr("_fields").unwrap_or(false) {
                queue.push_back(value.unbind());
            }
        }
    }
    Ok(out)
}

fn is_the_empty_product(ctx: &Ctx<'_>, node: &Bound<'_, PyAny>) -> PyResult<bool> {
    if !ctx.is(node, "Tuple")? {
        return Ok(false);
    }
    let elements = crate::pyast::field_list(node, "elements")?;
    Ok(elements.is_empty())
}

fn judged(
    ctx: &Ctx<'_>,
    judge: &dyn JudgeFn,
    given: &Bound<'_, PyAny>,
    given_module: &Bound<'_, PyAny>,
    declared: &Bound<'_, PyAny>,
    declared_module: &Bound<'_, PyAny>,
) -> bool {
    judge
        .judge(ctx, given, given_module, declared, declared_module)
        .unwrap_or(false)
}

pub fn is_the_environment(
    ctx: &Ctx<'_>,
    node: &Bound<'_, PyAny>,
    module: Option<&Bound<'_, PyAny>>,
    judge: Option<&dyn JudgeFn>,
) -> PyResult<bool> {
    let mut node = node.clone();
    if ctx.is(&node, "Tagged")? {
        if node.getattr("tag")?.extract::<String>()? == ENVIRON_TAG {
            return Ok(true);
        }
        node = node.getattr("type")?;
    }
    let (module, judge) = match (module, judge) {
        (Some(module), Some(judge)) => (module, judge),
        _ => return Ok(false),
    };
    let environment = ctx.make("TypeRef", &[("name", ctx.pystr(ENV_TYPE))])?;
    let builtin = crate::type_system::builtin_module_py(ctx.py)?;
    Ok(judged(
        ctx,
        judge,
        &node,
        module,
        environment.bind(ctx.py),
        builtin.bind(ctx.py),
    ) && judged(
        ctx,
        judge,
        environment.bind(ctx.py),
        builtin.bind(ctx.py),
        &node,
        module,
    ))
}

fn takes_the_rest(ctx: &Ctx<'_>, declared: &Bound<'_, PyAny>) -> PyResult<bool> {
    if !ctx.is(declared, "Tagged")? {
        return Ok(false);
    }
    let body = declared.getattr("type")?;
    ctx.is(&body, "Ellipsis")
}

fn matches(
    ctx: &Ctx<'_>,
    declared: &Bound<'_, PyAny>,
    given: &Bound<'_, PyAny>,
    module: &Bound<'_, PyAny>,
    given_module: &Bound<'_, PyAny>,
    judge: &dyn JudgeFn,
) -> VibaResult<bool> {
    let declared_tagged = ctx.is(declared, "Tagged")?;
    let given_tagged = ctx.is(given, "Tagged")?;
    if declared_tagged && given_tagged {
        let declared_tag: String = declared.getattr("tag")?.extract()?;
        let given_tag: String = given.getattr("tag")?.extract()?;
        let declared_type = declared.getattr("type")?;
        let given_type = given.getattr("type")?;
        if declared_tag != given_tag {
            let takes = type_system::takes_a_product(&declared_type)?;
            return Ok(
                takes && judge.judge(ctx, &given_type, given_module, &declared_type, module)?
            );
        }
        if judge.judge(ctx, &given_type, given_module, &declared_type, module)? {
            return Ok(true);
        }
        return Err(VibaError::Partial(format!(
            "{} does not fit {}: {} <: {} does not hold",
            source_form(ctx, given)?,
            source_form(ctx, declared)?,
            source_form(ctx, &given_type)?,
            source_form(ctx, &declared_type)?
        )));
    }
    if declared_tagged {
        let declared_type = declared.getattr("type")?;
        if judge.judge(ctx, given, given_module, &declared_type, module)? {
            return Ok(true);
        }
        return Err(VibaError::Partial(format!(
            "{} does not fit {}: {} <: {} does not hold",
            source_form(ctx, given)?,
            source_form(ctx, declared)?,
            source_form(ctx, given)?,
            source_form(ctx, &declared_type)?
        )));
    }
    Ok(crate::unparser::unparse_type(ctx.py, declared, 2)?
        == crate::unparser::unparse_type(ctx.py, given, 2)?)
}

fn nothing_left_to_give(
    ctx: &Ctx<'_>,
    node: &Bound<'_, PyAny>,
    module: &Bound<'_, PyAny>,
    judge: &dyn JudgeFn,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    if !(ctx.is(node, "Exponent")? || ctx.is(node, "ExponentChain")?) {
        return Ok((node.clone().unbind(), module.clone().unbind()));
    }
    let elements = elements(ctx, node)?;
    if elements.len() > 1 {
        let mut all = true;
        for element in &elements[1..] {
            let element = element.bind(ctx.py);
            if !(is_documentation(ctx, element)?
                || is_the_empty_product(ctx, element)?
                || is_the_environment(ctx, element, Some(module), Some(judge))?)
            {
                all = false;
                break;
            }
        }
        if all {
            return Ok((elements[0].clone_ref(ctx.py), module.clone().unbind()));
        }
    }
    Ok((node.clone().unbind(), module.clone().unbind()))
}

pub fn reduce_partial(
    ctx: &Ctx<'_>,
    node: &Bound<'_, PyAny>,
    module: &Bound<'_, PyAny>,
    resolve: &dyn ResolveFn,
    judge: &dyn JudgeFn,
    outermost: bool,
) -> VibaResult<(Py<PyAny>, Py<PyAny>)> {
    if dynamic_call(ctx, node)? {
        return Ok((ctx.make0("Any")?, module.clone().unbind()));
    }
    let mut node = node.clone().unbind();
    let mut module = module.clone().unbind();
    while ctx.is(node.bind(ctx.py), "Partial")? {
        let bound = node.bind(ctx.py);
        let (base, base_module) = reduce_partial(
            ctx,
            &bound.getattr("function")?,
            &module.bind(ctx.py),
            resolve,
            judge,
            false,
        )?;
        let argument = bound.getattr("argument")?;
        let (new_node, new_module) = give(
            ctx,
            base.bind(ctx.py),
            base_module.bind(ctx.py),
            &argument,
            module.bind(ctx.py),
            resolve,
            judge,
        )?;
        node = new_node;
        module = new_module;
    }
    if outermost {
        return Ok(nothing_left_to_give(
            ctx,
            node.bind(ctx.py),
            module.bind(ctx.py),
            judge,
        )?);
    }
    Ok((node, module))
}

fn give(
    ctx: &Ctx<'_>,
    base: &Bound<'_, PyAny>,
    module: &Bound<'_, PyAny>,
    argument: &Bound<'_, PyAny>,
    argument_module: &Bound<'_, PyAny>,
    resolve: &dyn ResolveFn,
    judge: &dyn JudgeFn,
) -> VibaResult<(Py<PyAny>, Py<PyAny>)> {
    let base = tagged_base(ctx, base, module, resolve)?;
    let base_bound = base.bind(ctx.py);
    if ctx.is(base_bound, "_NamedMember")? {
        let owner = base_bound.getattr("owner")?;
        let base_module = base_bound.getattr("module")?;
        return named_member(
            ctx,
            &owner,
            &base_module,
            argument,
            argument_module,
            resolve,
        );
    }
    if ctx.is(base_bound, "Member")? {
        let tag: String = base_bound.getattr("tag")?.extract()?;
        if tag == GET_ATTR_TAG {
            if is_the_environment(ctx, argument, Some(argument_module), Some(judge))? {
                return Ok((base, module.clone().unbind()));
            }
            let owner = argument.clone().unbind();
            let owner_module = argument_module.clone().unbind();
            let named = Py::new(
                ctx.py,
                NamedMember {
                    owner,
                    module: owner_module,
                },
            )?;
            return Ok((named.into_any(), module.clone().unbind()));
        }
        if INTERPRETER_MEMBER_TAGS.contains(&tag.as_str()) {
            let chain = member_design_chain(ctx, &tag)?;
            if is_the_environment(ctx, argument, Some(argument_module), Some(judge))? {
                let builtin = crate::type_system::builtin_module_py(ctx.py)?;
                return Ok((chain, builtin));
            }
            let builtin = crate::type_system::builtin_module_py(ctx.py)?;
            return give(
                ctx,
                chain.bind(ctx.py),
                &builtin.bind(ctx.py),
                argument,
                argument_module,
                resolve,
                judge,
            );
        }
        return member_of(ctx, &tag, argument, argument_module, resolve, judge);
    }
    if is_the_environment(ctx, argument, Some(argument_module), Some(judge))? {
        let (base, module) = unfold(ctx, base_bound, module, resolve)?;
        let elements = if ctx.is(base.bind(ctx.py), "Exponent")?
            || ctx.is(base.bind(ctx.py), "ExponentChain")?
        {
            elements(ctx, base.bind(ctx.py))?
        } else {
            vec![base]
        };
        let mut rest = vec![elements[0].clone_ref(ctx.py)];
        for one in &elements[1..] {
            if !is_the_environment(
                ctx,
                one.bind(ctx.py),
                Some(module.bind(ctx.py)),
                Some(judge),
            )? {
                rest.push(one.clone_ref(ctx.py));
            }
        }
        if rest.len() == 1 {
            return Ok((rest.remove(0), module));
        }
        let rest = ctx.list(rest)?;
        let chain = ctx.make("ExponentChain", &[("elements", rest)])?;
        return Ok((chain, module));
    }
    let (base, module) = unfold(ctx, base_bound, module, resolve)?;
    if !(ctx.is(base.bind(ctx.py), "Exponent")? || ctx.is(base.bind(ctx.py), "ExponentChain")?) {
        return Err(VibaError::Partial(format!(
            "only a function has arguments to give, not {}",
            source_form(ctx, base.bind(ctx.py))?
        )));
    }
    let elements = elements(ctx, base.bind(ctx.py))?;
    for (index, declared) in elements[1..].iter().enumerate() {
        let index = index + 1;
        let declared_bound = declared.bind(ctx.py);
        if takes_the_rest(ctx, declared_bound)? {
            return Ok((elements[0].clone_ref(ctx.py), module));
        }
        if matches(
            ctx,
            declared_bound,
            argument,
            module.bind(ctx.py),
            argument_module,
            judge,
        )? {
            let mut rest: Vec<Py<PyAny>> = elements[..index]
                .iter()
                .map(|item| item.clone_ref(ctx.py))
                .collect();
            rest.extend(
                elements[index + 1..]
                    .iter()
                    .map(|item| item.clone_ref(ctx.py)),
            );
            let mut documentation_only = true;
            for one in &rest[1..] {
                if !is_documentation(ctx, one.bind(ctx.py))? {
                    documentation_only = false;
                    break;
                }
            }
            if documentation_only {
                return Ok((rest.remove(0), module));
            }
            let rest = ctx.list(rest)?;
            let chain = ctx.make("ExponentChain", &[("elements", rest)])?;
            return Ok((chain, module));
        }
    }
    Err(VibaError::Partial(format!(
        "the function has no such argument: {}",
        source_form(ctx, argument)?
    )))
}

fn member_design_chain(ctx: &Ctx<'_>, tag: &str) -> PyResult<Py<PyAny>> {
    let signature = tagged::interpreter_member_chain(ctx, tag)?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(tag.to_string()))?;
    let signature = signature.bind(ctx.py);
    let elements = elements(ctx, signature)?;
    let mut kept = Vec::new();
    for element in &elements[1..] {
        let element_bound = element.bind(ctx.py);
        let is_env = ctx.is(element_bound, "Tagged")?
            && element_bound.getattr("tag")?.extract::<String>()? == ENVIRON_TAG;
        if !is_env {
            kept.push(element.clone_ref(ctx.py));
        }
    }
    if tag == GET_ITEM_TAG {
        let ellipsis = ctx.make0("Ellipsis")?;
        kept.push(ctx.make("Tagged", &[("tag", ctx.pystr("$args")), ("type", ellipsis)])?);
    }
    let mut all = vec![elements[0].clone_ref(ctx.py)];
    all.extend(kept);
    let all = ctx.list(all)?;
    ctx.make("ExponentChain", &[("elements", all)])
}

fn member_of(
    ctx: &Ctx<'_>,
    tag: &str,
    owner: &Bound<'_, PyAny>,
    owner_module: &Bound<'_, PyAny>,
    resolve: &dyn ResolveFn,
    judge: &dyn JudgeFn,
) -> VibaResult<(Py<PyAny>, Py<PyAny>)> {
    let owner = if ctx.is(owner, "Tagged")? {
        owner.getattr("type")?
    } else {
        owner.clone()
    };
    let given_owner = owner.clone();
    let given_owner_module = owner_module.clone();
    let (owner, owner_module) = unfold(ctx, &owner, owner_module, resolve)?;
    let owner_bound = owner.bind(ctx.py);
    if ctx.is(owner_bound, "Product")? || ctx.is(owner_bound, "ProductChain")? {
        for factor in product_elements(ctx, owner_bound)? {
            let factor = factor.bind(ctx.py);
            if ctx.is(factor, "Tagged")? && factor.getattr("tag")?.extract::<String>()? == tag {
                let (base, source_module) = unfold(
                    ctx,
                    &factor.getattr("type")?,
                    owner_module.bind(ctx.py),
                    resolve,
                )?;
                if !(ctx.is(base.bind(ctx.py), "Exponent")?
                    || ctx.is(base.bind(ctx.py), "ExponentChain")?)
                {
                    return Ok((base, source_module));
                }
                return give(
                    ctx,
                    base.bind(ctx.py),
                    source_module.bind(ctx.py),
                    &given_owner,
                    &given_owner_module,
                    resolve,
                    judge,
                );
            }
        }
        return Err(VibaError::Partial(format!(
            "no member tagged {} to take from {}",
            crate::pyast::repr_of(ctx.py, &ctx.pystr(tag).bind(ctx.py).clone()),
            source_form(ctx, owner_bound)?
        )));
    }
    Err(VibaError::Partial(format!(
        "{} is no value to take the member {} from",
        source_form(ctx, owner_bound)?,
        crate::pyast::repr_of(ctx.py, &ctx.pystr(tag).bind(ctx.py).clone())
    )))
}

fn tagged_base(
    ctx: &Ctx<'_>,
    node: &Bound<'_, PyAny>,
    module: &Bound<'_, PyAny>,
    resolve: &dyn ResolveFn,
) -> VibaResult<Py<PyAny>> {
    if !(ctx.is(node, "TypeApp")?
        && node.getattr("constructor")?.extract::<String>()? == tagged::TAGGED_NAME)
    {
        return Ok(node.clone().unbind());
    }
    let args = crate::pyast::field_list(node, "args")?;
    if let Some(problem) = tagged::tagged_problem(ctx, &args)? {
        return Err(VibaError::Partial(problem));
    }
    let symbol = tagged::literal_symbol(ctx, &args[0])?;
    let symbol = match symbol {
        Some(symbol) => symbol,
        None => {
            let text = symbol_text_of(ctx, &args[0], module, resolve)?;
            match text {
                None => return Ok(node.clone().unbind()),
                Some(text) => match tagged::symbol_of_text(&text) {
                    Some(symbol) => symbol,
                    None => {
                        return Err(VibaError::Partial(format!(
                            "{} is no symbol for a tag",
                            crate::pyast::repr_of(ctx.py, &ctx.pystr(&text).bind(ctx.py).clone())
                        )))
                    }
                },
            }
        }
    };
    let owned_args: Vec<Py<PyAny>> = args.iter().map(|arg| arg.clone().unbind()).collect();
    Ok(tagged::tagged_node(ctx, &symbol, &owned_args)?)
}

fn symbol_text_of(
    ctx: &Ctx<'_>,
    node: &Bound<'_, PyAny>,
    module: &Bound<'_, PyAny>,
    resolve: &dyn ResolveFn,
) -> VibaResult<Option<String>> {
    if ctx.is(node, "Constant")? {
        let value = node.getattr("value")?;
        if value.is_instance_of::<pyo3::types::PyString>() {
            return Ok(Some(value.extract()?));
        }
        return Ok(None);
    }
    let mut node = node.clone();
    if ctx.is(&node, "MemberTaken")? {
        match source_path(ctx, &node)? {
            Some(path) => {
                node = ctx
                    .make("TypeRef", &[("name", ctx.pystr(&path))])?
                    .into_bound(ctx.py)
            }
            None => return Ok(None),
        }
    }
    if !ctx.is(&node, "TypeRef")? {
        return Ok(None);
    }
    let name: String = node.getattr("name")?.extract()?;
    let target = resolve.resolve(ctx, &name, module)?;
    let (body, _) = match target {
        Some(target) => target,
        None => return Ok(None),
    };
    let mut body = body.bind(ctx.py).clone();
    while ctx.is(&body, "TypeDefinition")? {
        body = body.getattr("body")?;
    }
    if ctx.is(&body, "Constant")? {
        let value = body.getattr("value")?;
        if value.is_instance_of::<pyo3::types::PyString>() {
            return Ok(Some(value.extract()?));
        }
    }
    Ok(None)
}

fn named_member(
    ctx: &Ctx<'_>,
    owner: &Bound<'_, PyAny>,
    module: &Bound<'_, PyAny>,
    name: &Bound<'_, PyAny>,
    name_module: &Bound<'_, PyAny>,
    resolve: &dyn ResolveFn,
) -> VibaResult<(Py<PyAny>, Py<PyAny>)> {
    let text = symbol_text_of(ctx, name, name_module, resolve)?;
    let text = match text {
        Some(text) => text,
        None => return Ok((ctx.make0("Any")?, module.clone().unbind())),
    };
    let symbol = match tagged::symbol_of_text(&text) {
        Some(symbol) => symbol,
        None => {
            return Err(VibaError::Partial(format!(
                "{} is no symbol for a tag",
                crate::pyast::repr_of(ctx.py, &ctx.pystr(&text).bind(ctx.py).clone())
            )))
        }
    };
    let tag = tagged::tag_of(&symbol);
    let (owner, owner_module) = unfold(ctx, owner, module, resolve)?;
    let owner_bound = owner.bind(ctx.py);
    if ctx.is(owner_bound, "Product")? || ctx.is(owner_bound, "ProductChain")? {
        for factor in product_elements(ctx, owner_bound)? {
            let factor = factor.bind(ctx.py);
            if ctx.is(factor, "Tagged")? && factor.getattr("tag")?.extract::<String>()? == tag {
                return Ok((factor.getattr("type")?.unbind(), owner_module));
            }
        }
    }
    Err(VibaError::Partial(format!(
        "no member tagged {} to take from {}",
        crate::pyast::repr_of(ctx.py, &ctx.pystr(&tag).bind(ctx.py).clone()),
        source_form(ctx, owner.bind(ctx.py))?
    )))
}

// ----------------------------------------------------------------------
// Python-facing functions
// ----------------------------------------------------------------------

fn params_to_py(ctx: &Ctx<'_>, params: &[Param]) -> PyResult<Py<PyAny>> {
    let list = PyList::empty(ctx.py);
    for param in params {
        let tag: Py<PyAny> = match &param.tag {
            Some(tag) => ctx.pystr(tag),
            None => ctx.py.None(),
        };
        let is_env = crate::pyast::pybool(ctx.py, param.is_env);
        let slot: Py<PyAny> = match param.slot {
            Some(slot) => crate::pyast::pyint(ctx.py, slot as i64),
            None => ctx.py.None(),
        };
        let tuple = PyTuple::new(
            ctx.py,
            [tag, param.source_form.clone_ref(ctx.py), is_env, slot],
        )?;
        list.append(tuple)?;
    }
    Ok(list.into_any().unbind())
}

#[pyfunction]
fn parameters_of_py<'py>(py: Python<'py>, chain: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
    let ctx = Ctx::new(py)?;
    let params = parameters_of(&ctx, chain)?;
    params_to_py(&ctx, &params)
}

#[pyfunction]
fn result_of_py<'py>(py: Python<'py>, chain: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
    let ctx = Ctx::new(py)?;
    match result_of(&ctx, chain)? {
        Some(result) => Ok(result),
        None => Ok(py.None()),
    }
}

#[pyfunction]
fn product_elements_py<'py>(py: Python<'py>, node: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
    let ctx = Ctx::new(py)?;
    let items = product_elements(&ctx, node)?;
    ctx.list(items)
}

#[pyfunction]
fn get_args_product_py<'py>(py: Python<'py>, chain: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
    let ctx = Ctx::new(py)?;
    get_args_product(&ctx, chain)
}

#[pyfunction]
fn parameter_node_py<'py>(
    py: Python<'py>,
    tag: Option<&Bound<'py, PyAny>>,
    source_form: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    let ctx = Ctx::new(py)?;
    let tag = match tag {
        Some(tag) if !tag.is_none() => Some(tag.extract::<String>()?),
        _ => None,
    };
    parameter_node(&ctx, tag.as_deref(), &source_form.clone().unbind())
}

#[pyfunction]
fn answers_the_environment_py<'py>(py: Python<'py>, chain: &Bound<'py, PyAny>) -> PyResult<bool> {
    let ctx = Ctx::new(py)?;
    answers_the_environment(&ctx, chain)
}

#[pyfunction]
fn names_the_environment_py<'py>(py: Python<'py>, node: &Bound<'py, PyAny>) -> PyResult<bool> {
    let ctx = Ctx::new(py)?;
    names_the_environment(&ctx, node)
}

#[pyfunction]
fn environment_result_problem_py<'py>(
    py: Python<'py>,
    chain: &Bound<'py, PyAny>,
    name: &str,
) -> PyResult<Py<PyAny>> {
    let ctx = Ctx::new(py)?;
    match environment_result_problem(&ctx, chain, name)? {
        Some(problem) => Ok(ctx.pystr(&problem)),
        None => Ok(py.None()),
    }
}

#[pyfunction]
fn file_environment_result_problem_py<'py>(
    py: Python<'py>,
    tree: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    let ctx = Ctx::new(py)?;
    match file_environment_result_problem(&ctx, tree)? {
        Some(problem) => Ok(ctx.pystr(&problem)),
        None => Ok(py.None()),
    }
}

#[pyfunction]
fn module_as_function_py<'py>(
    py: Python<'py>,
    module: &Bound<'py, PyAny>,
    name: &str,
) -> PyResult<Py<PyAny>> {
    let ctx = Ctx::new(py)?;
    match module_as_function(&ctx, module, name) {
        Ok(Some((body, source_module))) => {
            let tuple = PyTuple::new(py, [body, source_module])?;
            Ok(tuple.into_any().unbind())
        }
        Ok(None) => Ok(py.None()),
        Err(error) => Err(error.into_pyerr(py)),
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NamedMember>()?;
    m.add_function(wrap_pyfunction!(parameters_of_py, m)?)?;
    m.add_function(wrap_pyfunction!(result_of_py, m)?)?;
    m.add_function(wrap_pyfunction!(product_elements_py, m)?)?;
    m.add_function(wrap_pyfunction!(get_args_product_py, m)?)?;
    m.add_function(wrap_pyfunction!(parameter_node_py, m)?)?;
    m.add_function(wrap_pyfunction!(answers_the_environment_py, m)?)?;
    m.add_function(wrap_pyfunction!(names_the_environment_py, m)?)?;
    m.add_function(wrap_pyfunction!(environment_result_problem_py, m)?)?;
    m.add_function(wrap_pyfunction!(file_environment_result_problem_py, m)?)?;
    m.add_function(wrap_pyfunction!(module_as_function_py, m)?)?;
    Ok(())
}
