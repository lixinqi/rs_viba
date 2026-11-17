//! The judgment `sub <: sup` (was viba/is_sub_type.py).
//!
//! Consumes Type values only (rs_viba.type); nothing above the type level is
//! visible here. `Ok(True/False)` is the judgment; `Err(message)` reports
//! malformed input, never a judgment.
//!
//! The checker walks the AST structure under the Types: names unfold to their
//! bodies, products inline their untagged members, functions compare
//! co-/contravariantly, and cycles are taken coinductively. All the mutable
//! state (memo tables, env stacks) lives behind `RefCell`, so the judge adapter
//! handed to `partial::reduce_partial` can hold `&Checker` and still memoize;
//! no `RefCell` borrow is held across a recursive call.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyFloat, PyInt, PyModule, PyString};

use crate::ctx::Ctx;
use crate::errors::{VibaError, VibaResult};
use crate::partial::{self, JudgeFn, ResolveFn};
use crate::pattern;
use crate::pyast;
use crate::type_system;
use crate::unparser;

const ENVIRON_TAG: &str = "$env";
const GET_ARGS_NAME: &str = "__get_args__";
const SUM_NODES: &[&str] = &["Sum", "SumChain"];
const PROD_NODES: &[&str] = &["Product", "ProductChain"];
const EXP_NODES: &[&str] = &["Exponent", "ExponentChain"];
const BASE_TYPES: &[&str] = &["BoolType", "IntType", "FloatType", "StrType"];
const LITERAL_TYPES: &[&str] = &[
    "BoolLiteralType",
    "IntLiteralType",
    "FloatLiteralType",
    "StrLiteralType",
];

fn base_of_literal(name: &str) -> Option<&'static str> {
    match name {
        "BoolLiteralType" => Some("BoolType"),
        "IntLiteralType" => Some("IntType"),
        "FloatLiteralType" => Some("FloatType"),
        "StrLiteralType" => Some("StrType"),
        _ => None,
    }
}

fn literal_of_container(name: &str) -> Option<&'static str> {
    match name {
        "list" => Some("ListLiteral"),
        "set" => Some("SetLiteral"),
        "dict" => Some("DictLiteral"),
        _ => None,
    }
}

fn is_literal_type(name: &str) -> bool {
    LITERAL_TYPES.contains(&name)
}

fn last_segment(name: &str) -> &str {
    name.rsplit('.').next().unwrap_or("")
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Side {
    Sub,
    Sup,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Unit {
    Nil,
    Never,
}

// ----------------------------------------------------------------------
// Identity keys
// ----------------------------------------------------------------------

/// A Python object kept alive by the key it is part of, compared by identity.
///
/// The Python checker keys its memos by `id(obj)`; holding the object here does
/// the same and keeps the id from being reused once the local reference goes.
struct ObjId(Py<PyAny>);

impl ObjId {
    fn of(value: &Bound<'_, PyAny>) -> Self {
        Self(value.clone().unbind())
    }
}

impl Clone for ObjId {
    fn clone(&self) -> Self {
        Python::attach(|py| Self(self.0.clone_ref(py)))
    }
}

impl PartialEq for ObjId {
    fn eq(&self, other: &Self) -> bool {
        self.0.as_ptr() == other.0.as_ptr()
    }
}

impl Eq for ObjId {}

impl Hash for ObjId {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (self.0.as_ptr() as usize).hash(state);
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
enum NodeKey {
    Defn(String),
    Inline(ObjId),
}

#[derive(Clone, PartialEq, Eq, Hash)]
enum TypeKey {
    Ast {
        module: ObjId,
        node: NodeKey,
        env: Vec<ObjId>,
    },
    Generic(String),
    Literal(String, String),
    Plain(String),
    ErrMarker,
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct WalkKey {
    sn: ObjId,
    sp: ObjId,
    s_mod: ObjId,
    p_mod: ObjId,
    senv: Vec<ObjId>,
    penv: Vec<ObjId>,
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct AppKey {
    node: ObjId,
    side: Side,
    module: ObjId,
    parts: Vec<TypeKey>,
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct DecisionKey {
    node: ObjId,
    module: ObjId,
    modules: Option<Vec<ObjId>>,
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct PartialKey {
    node: ObjId,
    module: ObjId,
    side: Side,
    env: Vec<ObjId>,
}

/// The identity of a definition an inline chain is expanding.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum InlineKey {
    Inline(usize),
    App(usize),
}

struct Member {
    node: Py<PyAny>,
    module: Py<PyAny>,
    env: Option<Py<PyAny>>,
}

type Branch = (String, Py<PyAny>, Py<PyAny>);

enum Decision {
    None,
    Chosen(Py<PyAny>),
    Failed(String),
}

// ----------------------------------------------------------------------
// The binder: an AstNodeType env_get that binds parameters to their meanings
// ----------------------------------------------------------------------

/// `env_get` for a generic definition's parameters: each name the call site's
/// argument stands for, answered as the `Ok`/`Err` result the judgment reads.
#[pyclass(name = "_SubtypeBinder", module = "rs_viba._native")]
struct Binder {
    meanings: HashMap<String, Py<PyAny>>,
}

#[pymethods]
impl Binder {
    fn __call__(&self, py: Python<'_>, name: &str) -> PyResult<Py<PyAny>> {
        match self.meanings.get(name) {
            // The stored value is already the Ok/Err result.
            Some(value) => Ok(value.clone_ref(py)),
            None => {
                let spelled = pyast::repr_of(py, &PyString::new(py, name).into_any());
                type_system::err(py, &format!("unbound parameter {}", spelled))
            }
        }
    }
}

// ----------------------------------------------------------------------
// Adapters the partial layer talks to
// ----------------------------------------------------------------------

struct PartialResolver<'a, 'py> {
    checker: &'a Checker<'py>,
    side: Side,
}

impl ResolveFn for PartialResolver<'_, '_> {
    fn resolve(
        &self,
        _ctx: &Ctx<'_>,
        name: &str,
        module: &Bound<'_, PyAny>,
    ) -> VibaResult<Option<(Py<PyAny>, Py<PyAny>)>> {
        self.checker.partial_target(name, module, self.side)
    }
}

struct PartialJudge<'a, 'py> {
    checker: &'a Checker<'py>,
}

impl JudgeFn for PartialJudge<'_, '_> {
    fn judge(
        &self,
        _ctx: &Ctx<'_>,
        sub: &Bound<'_, PyAny>,
        sub_module: &Bound<'_, PyAny>,
        sup: &Bound<'_, PyAny>,
        sup_module: &Bound<'_, PyAny>,
    ) -> VibaResult<bool> {
        self.checker.walk(sub, sub_module, sup, sup_module)
    }
}

/// The symbol a name in the source stands for, for `pattern::tagged_type_of`.
///
/// A name means a string here in three ways: it is one in the source, a plain
/// definition holds one, or it is a generic parameter an application bound to a
/// string argument (the env_get channel). The name means whatever the judgment
/// resolves it to; the string is what that meaning spells.
struct SymbolResolver<'a, 'py> {
    checker: &'a Checker<'py>,
    side: Side,
}

impl ResolveFn for SymbolResolver<'_, '_> {
    fn resolve(
        &self,
        _ctx: &Ctx<'_>,
        name: &str,
        module: &Bound<'_, PyAny>,
    ) -> VibaResult<Option<(Py<PyAny>, Py<PyAny>)>> {
        let checker = self.checker;
        let py = checker.py();
        let resolved = checker.resolve_name(name, module, self.side)?;
        let resolved = resolved.bind(py);
        if type_system::is_err(resolved)? {
            return Ok(None);
        }
        let meaning = resolved.getattr("ok_value")?;
        if checker.instance_of(&meaning, "StrLiteralType")? {
            let text = meaning.getattr("value")?.unbind();
            return Ok(Some((text, module.clone().unbind())));
        }
        if !meaning.hasattr("ast_node")? {
            return Ok(None);
        }
        let mut source_form = meaning.getattr("ast_node")?;
        if checker.is(&source_form, "TypeDefinition")? {
            source_form = source_form.getattr("body")?;
        }
        if checker.is(&source_form, "Constant")? {
            let value = source_form.getattr("value")?;
            if value.is_instance_of::<PyString>() {
                return Ok(Some((value.unbind(), module.clone().unbind())));
            }
        }
        Ok(None)
    }
}

// ----------------------------------------------------------------------
// Mutable state
// ----------------------------------------------------------------------

struct EnvStacks {
    sub: Vec<Py<PyAny>>,
    sup: Vec<Py<PyAny>>,
}

impl EnvStacks {
    fn new() -> Self {
        Self {
            sub: Vec::new(),
            sup: Vec::new(),
        }
    }

    fn get(&self, side: Side) -> &Vec<Py<PyAny>> {
        match side {
            Side::Sub => &self.sub,
            Side::Sup => &self.sup,
        }
    }

    fn get_mut(&mut self, side: Side) -> &mut Vec<Py<PyAny>> {
        match side {
            Side::Sub => &mut self.sub,
            Side::Sup => &mut self.sup,
        }
    }
}

// ----------------------------------------------------------------------
// The checker
// ----------------------------------------------------------------------

pub struct Checker<'py> {
    ctx: Ctx<'py>,
    types: Bound<'py, PyModule>,
    terminators: HashSet<String>,
    nil_eqv: HashSet<String>,
    never_eqv: HashSet<String>,
    builtin_module: Py<PyAny>,
    unit_nil: Py<PyAny>,
    unit_never: Py<PyAny>,
    memo: RefCell<HashMap<(TypeKey, TypeKey), bool>>,
    walk_memo: RefCell<HashMap<WalkKey, bool>>,
    walking: RefCell<HashSet<WalkKey>>,
    env_stacks: RefCell<EnvStacks>,
    unfolding: RefCell<HashSet<AppKey>>,
    unfolded: RefCell<HashMap<AppKey, Py<PyAny>>>,
    decisions: RefCell<HashMap<DecisionKey, Decision>>,
    partial_memo: RefCell<HashMap<PartialKey, (Py<PyAny>, Py<PyAny>)>>,
}

fn named_set(config: Option<&Bound<'_, PyAny>>, field: &str) -> PyResult<HashSet<String>> {
    let mut names = HashSet::new();
    if let Some(config) = config {
        if config.hasattr(field)? {
            let value = config.getattr(field)?;
            if !value.is_none() {
                for item in value.try_iter()? {
                    names.insert(item?.extract()?);
                }
            }
        }
    }
    Ok(names)
}

fn terminator_set(value: Option<&Bound<'_, PyAny>>) -> PyResult<HashSet<String>> {
    let mut names = HashSet::new();
    if let Some(value) = value {
        if !value.is_none() {
            for item in value.try_iter()? {
                names.insert(item?.extract()?);
            }
        }
    }
    Ok(names)
}

pub fn judge<'py>(
    py: Python<'py>,
    sub: &Bound<'py, PyAny>,
    sup: &Bound<'py, PyAny>,
    config: Option<&Bound<'py, PyAny>>,
) -> VibaResult<bool> {
    let checker = Checker::new(py, &HashSet::new(), config)?;
    checker.check(sub, sup)
}

impl<'py> Checker<'py> {
    fn new(
        py: Python<'py>,
        terminators: &HashSet<String>,
        config: Option<&Bound<'py, PyAny>>,
    ) -> VibaResult<Self> {
        let ctx = Ctx::new(py)?;
        let types = PyModule::import(py, "rs_viba.type")?;
        let nil_eqv = named_set(config, "nil_eqv")?;
        let never_eqv = named_set(config, "never_eqv")?;
        let builtin_module = types.getattr("BUILTIN_MODULE")?.unbind();
        let unit_nil = ctx.nodes.make0("Nil")?;
        let unit_never = ctx.nodes.make0("Never")?;
        Ok(Self {
            ctx,
            types,
            terminators: terminators.clone(),
            nil_eqv,
            never_eqv,
            builtin_module,
            unit_nil,
            unit_never,
            memo: RefCell::new(HashMap::new()),
            walk_memo: RefCell::new(HashMap::new()),
            walking: RefCell::new(HashSet::new()),
            env_stacks: RefCell::new(EnvStacks::new()),
            unfolding: RefCell::new(HashSet::new()),
            unfolded: RefCell::new(HashMap::new()),
            decisions: RefCell::new(HashMap::new()),
            partial_memo: RefCell::new(HashMap::new()),
        })
    }

    fn py(&self) -> Python<'py> {
        self.ctx.py
    }

    // ------------------------------------------------------------------
    // Small helpers over the Python values
    // ------------------------------------------------------------------

    fn class(&self, node: &Bound<'_, PyAny>) -> VibaResult<String> {
        Ok(pyast::class_name(node)?)
    }

    fn is(&self, node: &Bound<'_, PyAny>, name: &str) -> VibaResult<bool> {
        Ok(self.class(node)? == name)
    }

    fn is_in(&self, node: &Bound<'_, PyAny>, names: &[&str]) -> VibaResult<bool> {
        let class = self.class(node)?;
        Ok(names.contains(&class.as_str()))
    }

    fn instance_of(&self, node: &Bound<'_, PyAny>, name: &str) -> VibaResult<bool> {
        let class = self.types.getattr(name)?;
        Ok(node.is_instance(&class)?)
    }

    fn field_string(&self, node: &Bound<'_, PyAny>, name: &str) -> VibaResult<String> {
        Ok(node.getattr(name)?.extract()?)
    }

    fn field_list(
        &self,
        node: &Bound<'py, PyAny>,
        name: &str,
    ) -> VibaResult<Vec<Bound<'py, PyAny>>> {
        Ok(pyast::field_list(node, name)?)
    }

    fn repr_str(&self, text: &str) -> VibaResult<String> {
        let py = self.py();
        Ok(pyast::repr_of(py, &PyString::new(py, text).into_any()))
    }

    fn repr_node(&self, node: &Bound<'_, PyAny>) -> String {
        pyast::repr_of(self.py(), node)
    }

    fn make_type(&self, name: &str) -> VibaResult<Py<PyAny>> {
        Ok(self.types.getattr(name)?.call0()?.unbind())
    }

    fn ast_type(
        &self,
        node: &Bound<'_, PyAny>,
        module: &Bound<'_, PyAny>,
    ) -> VibaResult<Py<PyAny>> {
        Ok(self
            .types
            .getattr("AstNodeType")?
            .call1((node, module))?
            .unbind())
    }

    fn ast_type_env(
        &self,
        node: &Bound<'_, PyAny>,
        module: &Bound<'_, PyAny>,
        env: &Bound<'_, PyAny>,
    ) -> VibaResult<Py<PyAny>> {
        Ok(self
            .types
            .getattr("AstNodeType")?
            .call1((node, module, env))?
            .unbind())
    }

    fn one_line(&self, node: &Bound<'_, PyAny>) -> VibaResult<String> {
        let text = unparser::unparse_type(self.py(), node, 2)?;
        Ok(text.split_whitespace().collect::<Vec<_>>().join(" "))
    }

    fn result_is_err(&self, result: &Py<PyAny>) -> VibaResult<bool> {
        Ok(type_system::is_err(result.bind(self.py()))?)
    }

    fn err_message(&self, result: &Bound<'_, PyAny>) -> VibaResult<String> {
        Ok(result.getattr("msg")?.extract()?)
    }

    fn ok_value(&self, result: &Bound<'py, PyAny>) -> VibaResult<Bound<'py, PyAny>> {
        Ok(result.getattr("ok_value")?)
    }

    fn generic_params(&self, definition: &Bound<'_, PyAny>) -> VibaResult<Vec<String>> {
        if !definition.hasattr("generic_params")? {
            return Ok(Vec::new());
        }
        let params = definition.getattr("generic_params")?;
        if params.is_none() {
            return Ok(Vec::new());
        }
        Ok(params.extract()?)
    }

    /// `_definition_named`: a module's own top-level definition under this name.
    fn definition_named(
        &self,
        module: &Bound<'_, PyAny>,
        name: &str,
    ) -> VibaResult<Option<Py<PyAny>>> {
        let tree = match module.getattr("module") {
            Ok(tree) => tree,
            Err(_) => return Ok(None),
        };
        let body = match tree.getattr("body") {
            Ok(body) => body,
            Err(_) => return Ok(None),
        };
        for statement in body.try_iter()? {
            let statement = statement?;
            if self.as_definition(&statement)? == Some(name.to_string()) {
                return Ok(Some(statement.unbind()));
            }
        }
        Ok(None)
    }

    /// `_as_definition`: the definition an AST node is, by name; else None.
    fn as_definition(&self, node: &Bound<'_, PyAny>) -> VibaResult<Option<String>> {
        if self.is(node, "TypeDefinition")? || self.is(node, "GenericDefinition")? {
            return Ok(Some(self.field_string(node, "name")?));
        }
        Ok(None)
    }

    /// `_as_leaf`: an AstNodeType wrapping a leaf node IS that leaf.
    fn as_leaf(&self, value: &Bound<'_, PyAny>) -> VibaResult<Py<PyAny>> {
        if !self.instance_of(value, "AstNodeType")? {
            return Ok(value.clone().unbind());
        }
        let node = value.getattr("ast_node")?;
        if self.is(&node, "Nil")? {
            return self.make_type("NilType");
        }
        if self.is(&node, "Never")? {
            return self.make_type("NeverType");
        }
        if self.is(&node, "Constant")? {
            return self.literal_type(&node.getattr("value")?);
        }
        Ok(value.clone().unbind())
    }

    /// `_literal_type`: bool before int (bool is an int in Python).
    fn literal_type(&self, value: &Bound<'_, PyAny>) -> VibaResult<Py<PyAny>> {
        let py = self.py();
        let (class, holds) = if value.is_instance_of::<PyBool>() {
            ("BoolLiteralType", true)
        } else if value.is_instance_of::<PyInt>() {
            ("IntLiteralType", true)
        } else if value.is_instance_of::<PyFloat>() {
            ("FloatLiteralType", true)
        } else if value.is_instance_of::<PyString>() {
            ("StrLiteralType", true)
        } else {
            ("", false)
        };
        if !holds {
            return Err(VibaError::Py(PyTypeError::new_err(format!(
                "unsupported literal: {}",
                pyast::repr_of(py, value)
            ))));
        }
        Ok(self.types.getattr(class)?.call1((value,))?.unbind())
    }

    /// `_unwrap_definition`: plain TypeDefinitions are transparent.
    fn unwrap_definition(&self, entry: &Bound<'_, PyAny>) -> VibaResult<(Py<PyAny>, Py<PyAny>)> {
        let node = entry.getattr("ast_node")?;
        let module = entry.getattr("container_module")?;
        if self.is(&node, "TypeDefinition")? {
            return Ok((node.getattr("body")?.unbind(), module.unbind()));
        }
        Ok((node.unbind(), module.unbind()))
    }

    fn flatten_sum(&self, node: &Bound<'_, PyAny>) -> VibaResult<Vec<Py<PyAny>>> {
        if self.is(node, "Sum")? {
            let mut left = self.flatten_sum(&node.getattr("left")?)?;
            left.extend(self.flatten_sum(&node.getattr("right")?)?);
            return Ok(left);
        }
        if self.is(node, "SumChain")? {
            return Ok(self
                .field_list(node, "elements")?
                .into_iter()
                .map(|item| item.unbind())
                .collect());
        }
        Ok(vec![node.clone().unbind()])
    }

    fn exponent_elements(&self, node: &Bound<'_, PyAny>) -> VibaResult<Vec<Py<PyAny>>> {
        Ok(partial::elements(&self.ctx, node)?)
    }

    fn exponent_parts(&self, node: &Bound<'_, PyAny>) -> VibaResult<(Py<PyAny>, Vec<Py<PyAny>>)> {
        if !self.is_in(node, EXP_NODES)? {
            return Err(VibaError::Program(format!(
                "not an exponent: {}",
                self.repr_node(node)
            )));
        }
        let mut elements = self.exponent_elements(node)?;
        if elements.is_empty() {
            return Err(VibaError::Program("not an exponent".to_string()));
        }
        let result = elements.remove(0);
        Ok((result, elements))
    }

    fn param_index(&self, node: &Bound<'_, PyAny>, params: &[String]) -> VibaResult<Option<usize>> {
        let node = if self.is(node, "Tagged")? {
            node.getattr("type")?
        } else {
            node.clone()
        };
        if !self.is(&node, "TypeRef")? {
            return Ok(None);
        }
        let name = self.field_string(&node, "name")?;
        Ok(params.iter().position(|param| param == &name))
    }

    fn partial_parts(&self, node: &Bound<'_, PyAny>) -> VibaResult<(Py<PyAny>, Vec<Py<PyAny>>)> {
        let mut arguments = Vec::new();
        let mut current = node.clone().unbind();
        loop {
            let bound = current.bind(self.py());
            if !self.is(bound, "Partial")? {
                break;
            }
            arguments.push(bound.getattr("argument")?.unbind());
            current = bound.getattr("function")?.unbind();
        }
        arguments.reverse();
        Ok((current, arguments))
    }

    fn has_code_block(&self, node: &Bound<'_, PyAny>) -> VibaResult<bool> {
        for part in partial::walk(&self.ctx, node)? {
            if self.is(part.bind(self.py()), "HintBlock")? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    // ------------------------------------------------------------------
    // Type-level dispatch
    // ------------------------------------------------------------------

    fn check(&self, sub: &Bound<'py, PyAny>, sup: &Bound<'py, PyAny>) -> VibaResult<bool> {
        let sub_key = self.type_key(sub, &self.env_ids(Side::Sub))?;
        let sup_key = self.type_key(sup, &self.env_ids(Side::Sup))?;
        let key = (sub_key, sup_key);
        if let Some(result) = self.memo.borrow().get(&key) {
            return Ok(*result);
        }
        let pushed = self.push_envs(sub, sup)?;
        let outcome = self.check_uncached(sub, sup);
        self.pop_envs(&pushed);
        let result = outcome?;
        self.memo.borrow_mut().insert(key, result);
        Ok(result)
    }

    /// An AstNodeType carrying env_get scopes its own free names.
    fn push_envs(&self, sub: &Bound<'_, PyAny>, sup: &Bound<'_, PyAny>) -> VibaResult<Vec<Side>> {
        let mut pushed = Vec::new();
        if self.push_one(Side::Sub, sub)? {
            pushed.push(Side::Sub);
        }
        if self.push_one(Side::Sup, sup)? {
            pushed.push(Side::Sup);
        }
        Ok(pushed)
    }

    fn push_one(&self, side: Side, value: &Bound<'_, PyAny>) -> VibaResult<bool> {
        if !value.hasattr("env_get")? {
            return Ok(false);
        }
        let env = value.getattr("env_get")?;
        if env.is_none() {
            return Ok(false);
        }
        self.env_stacks
            .borrow_mut()
            .get_mut(side)
            .push(env.unbind());
        Ok(true)
    }

    fn pop_envs(&self, pushed: &[Side]) {
        for side in pushed {
            self.pop_env(*side);
        }
    }

    fn push_env(&self, side: Side, env: &Py<PyAny>) {
        self.env_stacks
            .borrow_mut()
            .get_mut(side)
            .push(env.clone_ref(self.py()));
    }

    fn pop_env(&self, side: Side) {
        self.env_stacks.borrow_mut().get_mut(side).pop();
    }

    fn env_ids(&self, side: Side) -> Vec<ObjId> {
        let mut out = Vec::new();
        for env in self.env_stacks.borrow().get(side) {
            out.push(ObjId(env.clone_ref(self.py())));
        }
        out
    }

    fn swap_envs(&self) {
        let mut stacks = self.env_stacks.borrow_mut();
        let EnvStacks { sub, sup } = &mut *stacks;
        std::mem::swap(sub, sup);
    }

    /// Module definitions win; a product member comes next; the side's env_get
    /// supplements (innermost binding first) — the free-variable channel.
    fn resolve_name(
        &self,
        name: &str,
        module: &Bound<'_, PyAny>,
        side: Side,
    ) -> VibaResult<Py<PyAny>> {
        let py = self.py();
        let resolved = type_system::module_get_type(py, module, name, None)?;
        if !self.result_is_err(&resolved)? {
            return Ok(resolved);
        }
        if let Some(member) = self.product_member_type(name, module, side)? {
            return Ok(member);
        }
        let envs: Vec<Py<PyAny>> = self
            .env_stacks
            .borrow()
            .get(side)
            .iter()
            .map(|env| env.clone_ref(py))
            .collect();
        for env in envs.iter().rev() {
            let got = env.bind(py).call1((name,))?;
            if !type_system::is_err(&got)? {
                return Ok(got.unbind());
            }
        }
        Ok(resolved)
    }

    /// `args.a` taken as a type: the `$a` member's declared type.
    fn product_member_type(
        &self,
        name: &str,
        module: &Bound<'_, PyAny>,
        side: Side,
    ) -> VibaResult<Option<Py<PyAny>>> {
        let py = self.py();
        let (head, tag) = match name.rfind('.') {
            Some(index) => (&name[..index], &name[index + 1..]),
            None => return Ok(None),
        };
        if head.is_empty() {
            return Ok(None);
        }
        let (body, container_module) = match self.head_body(head, module, side)? {
            Some(found) => found,
            None => return Ok(None),
        };
        let mut body = body;
        let mut container_module = container_module;
        if self.is(body.bind(py), "Partial")? {
            let (reduced, reduced_module) =
                self.partial(body.bind(py), container_module.bind(py), side)?;
            body = reduced;
            container_module = reduced_module;
        }
        for factor in partial::product_elements(&self.ctx, body.bind(py))? {
            let factor = factor.bind(py);
            if self.is(factor, "Tagged")? {
                let factor_tag = self.field_string(factor, "tag")?;
                if factor_tag == format!("${}", tag) {
                    let entry =
                        self.ast_type(&factor.getattr("type")?, container_module.bind(py))?;
                    return Ok(Some(type_system::ok(py, entry)?));
                }
            }
        }
        Ok(None)
    }

    /// (body, container_module) for the name, following aliases to the end.
    fn head_body(
        &self,
        name: &str,
        module: &Bound<'_, PyAny>,
        side: Side,
    ) -> VibaResult<Option<(Py<PyAny>, Py<PyAny>)>> {
        let py = self.py();
        let mut name = name.to_string();
        let mut module = module.clone().unbind();
        let mut seen = HashSet::new();
        while !seen.contains(&name) {
            seen.insert(name.clone());
            let resolved = self.resolve_name(&name, module.bind(py), side)?;
            if self.result_is_err(&resolved)? {
                return Ok(None);
            }
            let entry = resolved.bind(py).getattr("ok_value")?;
            if !self.instance_of(&entry, "AstNodeType")? {
                return Ok(None);
            }
            let mut node = entry.getattr("ast_node")?;
            if self.is(&node, "TypeDefinition")? {
                node = node.getattr("body")?;
            }
            if !self.is(&node, "TypeRef")? {
                return Ok(Some((
                    node.unbind(),
                    entry.getattr("container_module")?.unbind(),
                )));
            }
            name = self.field_string(&node, "name")?;
            module = entry.getattr("container_module")?.unbind();
        }
        Ok(None)
    }

    /// Take `tagged[...]` as the tag it spells, on either side; else None.
    fn tagged_side(
        &self,
        sn: &Bound<'py, PyAny>,
        s_mod: &Bound<'py, PyAny>,
        sp: &Bound<'py, PyAny>,
        p_mod: &Bound<'py, PyAny>,
    ) -> VibaResult<Option<bool>> {
        for (node, module, side) in [(sn, s_mod, Side::Sub), (sp, p_mod, Side::Sup)] {
            let resolver = SymbolResolver {
                checker: self,
                side,
            };
            let got = pattern::tagged_type_of(&self.ctx, node, module, Some(&resolver));
            let got = match got {
                Ok(value) => value,
                Err(VibaError::Py(error)) => return Err(VibaError::Py(error)),
                Err(error) => return Err(VibaError::Unresolved(error.message())),
            };
            let got = match got {
                Some(value) => value,
                None => continue,
            };
            let got = got.bind(self.py());
            return match side {
                Side::Sub => Ok(Some(self.walk(got, module, sp, p_mod)?)),
                Side::Sup => Ok(Some(self.walk(sn, s_mod, got, module)?)),
            };
        }
        Ok(None)
    }

    /// Lift a Constant, TypeRef, Nil or Never to a Type; else None.
    /// An unresolvable TypeRef is malformed.
    fn lift(
        &self,
        node: &Bound<'_, PyAny>,
        module: &Bound<'_, PyAny>,
        side: Side,
    ) -> VibaResult<Option<Py<PyAny>>> {
        if self.is(node, "Constant")? {
            return Ok(Some(self.literal_type(&node.getattr("value")?)?));
        }
        if self.is(node, "Nil")? {
            return Ok(Some(self.make_type("NilType")?));
        }
        if self.is(node, "Never")? {
            return Ok(Some(self.make_type("NeverType")?));
        }
        if self.is(node, "TypeRef")? {
            return Ok(Some(self.lift_ref(node, module, side)?));
        }
        Ok(None)
    }

    fn lift_ref(
        &self,
        node: &Bound<'_, PyAny>,
        module: &Bound<'_, PyAny>,
        side: Side,
    ) -> VibaResult<Py<PyAny>> {
        let name = self.field_string(node, "name")?;
        let resolved = self.resolve_name(&name, module, side)?;
        if self.result_is_err(&resolved)? {
            // Why it is unresolved is more useful than "unresolved".
            return Err(VibaError::Unresolved(
                self.err_message(&resolved.bind(self.py()))?,
            ));
        }
        Ok(self.ok_value(&resolved.bind(self.py()))?.unbind())
    }

    fn check_uncached(&self, sub: &Bound<'py, PyAny>, sup: &Bound<'py, PyAny>) -> VibaResult<bool> {
        if self.instance_of(sub, "NeverType")? {
            return Ok(true);
        }
        if self.instance_of(sup, "AnyType")? {
            return Ok(true); // Any is the top: everything fits
        }
        if self.instance_of(sub, "AnyType")? {
            return self.instance_of(sup, "AnyType"); // Any is only below Any
        }
        if self.instance_of(sup, "NeverType")? || self.instance_of(sup, "NilType")? {
            return Ok(self.class(sub)? == self.class(sup)?);
        }
        if let Some(result) = self.probe_leaves(sub, sup)? {
            return Ok(result);
        }
        if self.instance_of(sup, "AstNodeType")? && self.instance_of(sub, "AstNodeType")? {
            return self.check_ast_pair(sub, sup);
        }
        if self.instance_of(sup, "AstNodeType")?
            && self.is(&sup.getattr("ast_node")?, "HintBlock")?
        {
            return self.instance_of(sub, "NilType"); // hint block: the unit is its resident
        }
        Ok(false)
    }

    fn probe_leaves(
        &self,
        sub: &Bound<'_, PyAny>,
        sup: &Bound<'_, PyAny>,
    ) -> VibaResult<Option<bool>> {
        if let Some(result) = self.check_base(sub, sup)? {
            return Ok(Some(result));
        }
        if let Some(result) = self.check_literal(sub, sup)? {
            return Ok(Some(result));
        }
        self.check_builtin_name(sub, sup)
    }

    fn check_base(
        &self,
        sub: &Bound<'_, PyAny>,
        sup: &Bound<'_, PyAny>,
    ) -> VibaResult<Option<bool>> {
        if self.instance_of(sup, "FixedIntType")? {
            // A width is the set of integers it holds.
            if self.instance_of(sub, "IntLiteralType")? {
                let value = sub.getattr("value")?;
                let minimum = sup.getattr("minimum")?;
                let maximum = sup.getattr("maximum")?;
                return Ok(Some(minimum.le(&value)? && value.le(&maximum)?));
            }
            if self.instance_of(sub, "FixedIntType")? {
                let sup_minimum = sup.getattr("minimum")?;
                let sup_maximum = sup.getattr("maximum")?;
                let sub_minimum = sub.getattr("minimum")?;
                let sub_maximum = sub.getattr("maximum")?;
                return Ok(Some(
                    sup_minimum.le(&sub_minimum)? && sub_maximum.le(&sup_maximum)?,
                ));
            }
            return Ok(Some(false));
        }
        if self.instance_of(sub, "FixedIntType")? {
            return Ok(Some(self.instance_of(sup, "IntType")?));
        }
        let sup_class = self.class(sup)?;
        if !BASE_TYPES.contains(&sup_class.as_str()) {
            return Ok(None);
        }
        let sub_class = self.class(sub)?;
        Ok(Some(
            sub_class == sup_class || base_of_literal(&sub_class) == Some(sup_class.as_str()),
        ))
    }

    fn check_literal(
        &self,
        sub: &Bound<'_, PyAny>,
        sup: &Bound<'_, PyAny>,
    ) -> VibaResult<Option<bool>> {
        let sup_class = self.class(sup)?;
        if !is_literal_type(&sup_class) {
            return Ok(None);
        }
        if self.class(sub)? != sup_class {
            return Ok(Some(false));
        }
        let sub_value = sub.getattr("value")?;
        let sup_value = sup.getattr("value")?;
        Ok(Some(sub_value.eq(&sup_value)?))
    }

    /// A builtin container with no arguments is its name.
    fn check_builtin_name(
        &self,
        sub: &Bound<'_, PyAny>,
        sup: &Bound<'_, PyAny>,
    ) -> VibaResult<Option<bool>> {
        if !self.instance_of(sup, "BuiltinGenericType")? {
            return Ok(None);
        }
        if !self.instance_of(sub, "BuiltinGenericType")? {
            return Ok(Some(false));
        }
        Ok(Some(
            self.field_string(sup, "name")? == self.field_string(sub, "name")?,
        ))
    }

    // ------------------------------------------------------------------
    // AstNodeType pairs
    // ------------------------------------------------------------------

    fn check_ast_pair(&self, sub: &Bound<'py, PyAny>, sup: &Bound<'py, PyAny>) -> VibaResult<bool> {
        let py = self.py();
        let (sn, s_mod) = self.unwrap_definition(sub)?;
        let (sp, p_mod) = self.unwrap_definition(sup)?;
        let sn = sn.bind(py);
        let sp = sp.bind(py);
        if self.is(sn, "GenericDefinition")? && self.is(sp, "GenericDefinition")? {
            // Two bare generic names: the same name, and nothing else.
            return Ok(self.field_string(sn, "name")? == self.field_string(sp, "name")?);
        }
        self.walk(sn, s_mod.bind(py), sp, p_mod.bind(py))
    }

    // ------------------------------------------------------------------
    // Structural walk over viba_ast nodes
    // ------------------------------------------------------------------

    fn walk(
        &self,
        sn: &Bound<'py, PyAny>,
        s_mod: &Bound<'py, PyAny>,
        sp: &Bound<'py, PyAny>,
        p_mod: &Bound<'py, PyAny>,
    ) -> VibaResult<bool> {
        let py = self.py();
        // A unit a config names is bypassed, not resolved.
        let sn = self.config_unit_node(sn)?.into_bound(py);
        let sp = self.config_unit_node(sp)?.into_bound(py);
        // Names unfold, `<<` is given, and both can hand the other work.
        let (sn, s_mod) = self.normalize(&sn, s_mod, Side::Sub)?;
        let (sp, p_mod) = self.normalize(&sp, p_mod, Side::Sup)?;
        let sn = self.config_unit_node(sn.bind(py))?.into_bound(py);
        let sp = self.config_unit_node(sp.bind(py))?.into_bound(py);
        let key = WalkKey {
            sn: ObjId::of(&sn),
            sp: ObjId::of(&sp),
            s_mod: ObjId::of(s_mod.bind(py)),
            p_mod: ObjId::of(p_mod.bind(py)),
            senv: self.env_ids(Side::Sub),
            penv: self.env_ids(Side::Sup),
        };
        if self.walking.borrow().contains(&key) {
            return Ok(true); // coinductive assumption
        }
        if let Some(result) = self.walk_memo.borrow().get(&key) {
            return Ok(*result);
        }
        self.walking.borrow_mut().insert(key.clone());
        let outcome = self.walk_inner(&sn, s_mod.bind(py), &sp, p_mod.bind(py));
        self.walking.borrow_mut().remove(&key);
        let result = outcome?;
        self.walk_memo.borrow_mut().insert(key, result);
        Ok(result)
    }

    /// A TypeRef is transparent unless it names a generic: it unfolds, name
    /// after name, to the bound body.
    fn unfold_ref(
        &self,
        node: &Bound<'_, PyAny>,
        module: &Bound<'_, PyAny>,
        side: Side,
    ) -> VibaResult<(Py<PyAny>, Py<PyAny>)> {
        let py = self.py();
        let mut node = node.clone().unbind();
        let mut module = module.clone().unbind();
        let mut seen = HashSet::new();
        loop {
            let bound = node.bind(py);
            if !self.is(bound, "TypeRef")? {
                return Ok((node, module));
            }
            let name = self.field_string(bound, "name")?;
            if seen.contains(&name) {
                return Ok((node, module));
            }
            seen.insert(name.clone());
            let resolved = self.resolve_name(&name, module.bind(py), side)?;
            if self.result_is_err(&resolved)? {
                return Ok((node, module));
            }
            let target = resolved.bind(py).getattr("ok_value")?;
            if !self.instance_of(&target, "AstNodeType")? {
                return Ok((node, module));
            }
            let mut body = target.getattr("ast_node")?;
            if self.is(&body, "GenericDefinition")? {
                return Ok((node, module));
            }
            if self.is(&body, "TypeDefinition")? {
                body = body.getattr("body")?;
            }
            node = body.unbind();
            module = target.getattr("container_module")?.unbind();
        }
    }

    fn walk_inner(
        &self,
        sn: &Bound<'py, PyAny>,
        s_mod: &Bound<'py, PyAny>,
        sp: &Bound<'py, PyAny>,
        p_mod: &Bound<'py, PyAny>,
    ) -> VibaResult<bool> {
        let py = self.py();
        // A product nothing but one member is left of is that member; one with
        // no member left is the unit itself.
        if let Some((node, module, env)) = self.reduce_product(sn, s_mod, Side::Sub)? {
            let pushed = env.is_some();
            if let Some(env) = &env {
                self.push_env(Side::Sub, env);
            }
            let outcome = self.walk(node.bind(py), module.bind(py), sp, p_mod);
            if pushed {
                self.pop_env(Side::Sub);
            }
            return outcome;
        }
        if let Some((node, module, env)) = self.reduce_product(sp, p_mod, Side::Sup)? {
            let pushed = env.is_some();
            if let Some(env) = &env {
                self.push_env(Side::Sup, env);
            }
            let outcome = self.walk(sn, s_mod, node.bind(py), module.bind(py));
            if pushed {
                self.pop_env(Side::Sup);
            }
            return outcome;
        }
        // A `tagged[...]` in the source is the tag it spells.
        if let Some(tagged) = self.tagged_side(sn, s_mod, sp, p_mod)? {
            return Ok(tagged);
        }
        // `g[T].value` / `g[T].type`: the member of the file the decision chose.
        if let Some(member) = self.member_meaning(sn, s_mod, Side::Sub)? {
            return self.unfold_member(member.bind(py), Side::Sub, sp, p_mod);
        }
        if let Some(member) = self.member_meaning(sp, p_mod, Side::Sup)? {
            return self.unfold_member(member.bind(py), Side::Sup, sn, s_mod);
        }
        // A generic application is decided before anything else takes it.
        if self.is(sn, "TypeApp")? && self.decided(sn, s_mod)? {
            return self.unfold_typeapp(sn, s_mod, Side::Sub, sp, p_mod);
        }
        if self.is(sp, "TypeApp")? && self.decided(sp, p_mod)? {
            return self.unfold_typeapp(sp, p_mod, Side::Sup, sn, s_mod);
        }
        if self.is(sn, "Never")? {
            return Ok(true); // bottom fits anywhere
        }
        if self.is(sp, "Any")? {
            // Any is the top; the other direction is no special case.
            return Ok(true);
        }
        // `...` is an open type.
        if self.is(sn, "Ellipsis")? && !(self.is_in(sp, SUM_NODES)? || self.is(sp, "Tagged")?) {
            return self.is(sp, "Ellipsis");
        }
        if self.is(sp, "Ellipsis")? {
            return self.walk_rest(sn, s_mod);
        }
        if self.is_in(sp, &["Nil", "Never"])? {
            // Leaf on leaf: a name in the source must be recognized.
            let sub_leaf = self.lift(sn, s_mod, Side::Sub)?;
            let sup_leaf = self.lift(sp, p_mod, Side::Sup)?;
            if let (Some(sub_leaf), Some(sup_leaf)) = (&sub_leaf, &sup_leaf) {
                return Ok(self.class(sub_leaf.bind(py))? == self.class(sup_leaf.bind(py))?);
            }
            if self.is(sn, "TypeApp")? {
                // An application on the sub side unfolds too.
                return self.unfold_typeapp(sn, s_mod, Side::Sub, sp, p_mod);
            }
            if self.is(sn, "HintBlock")? && self.is(sp, "Nil")? {
                return Ok(true); // a hint block is a unit, and nil takes it
            }
            return Ok(self.class(sn)? == self.class(sp)?);
        }
        if !self.terminators.is_empty() {
            if let Some(operand) = self.never_head_operand(sp, p_mod)? {
                return self.walk_never_head(sn, s_mod, sp, p_mod, &operand);
            }
        }
        if let Some(mixed) = self.walk_mixed_typeapp(sn, s_mod, sp, p_mod)? {
            return Ok(mixed);
        }
        if let Some(lifted) = self.walk_lifted(sn, s_mod, sp, p_mod)? {
            return Ok(lifted);
        }
        self.walk_structural(sn, s_mod, sp, p_mod)
    }

    // ------------------------------------------------------------------
    // never-headed chains
    // ------------------------------------------------------------------

    fn walk_never_head(
        &self,
        sn: &Bound<'py, PyAny>,
        s_mod: &Bound<'py, PyAny>,
        sp: &Bound<'py, PyAny>,
        p_mod: &Bound<'py, PyAny>,
        operand: &(Py<PyAny>, Py<PyAny>),
    ) -> VibaResult<bool> {
        let py = self.py();
        if let Some(sub_operand) = self.never_head_operand(sn, s_mod)? {
            if self.same_branches_settled(&sub_operand, operand)? {
                return Ok(true);
            }
        }
        if self.branches_settled(sn, s_mod, operand)? {
            return Ok(true);
        }
        let (node, _module) = self.unfold_ref(sn, s_mod, Side::Sub)?;
        if self.is(sp, "TypeApp")? {
            if self.is(node.bind(py), "TypeApp")? {
                // Both sides are applications in the source: a copy settles
                // nothing, and it does not count as an exponent here.
                return Ok(false);
            }
            return self.unfold_typeapp(sp, p_mod, Side::Sup, sn, s_mod);
        }
        if let Some(mixed) = self.walk_mixed_typeapp(sn, s_mod, sp, p_mod)? {
            return Ok(mixed);
        }
        self.walk_exponent(sn, s_mod, sp, p_mod)
    }

    fn never_head_operand(
        &self,
        node: &Bound<'_, PyAny>,
        module: &Bound<'_, PyAny>,
    ) -> VibaResult<Option<(Py<PyAny>, Py<PyAny>)>> {
        if let Some(operand) = self.never_head_application(node, module)? {
            return Ok(Some(operand));
        }
        self.never_head_exponent(node, module)
    }

    fn never_head_application(
        &self,
        node: &Bound<'_, PyAny>,
        module: &Bound<'_, PyAny>,
    ) -> VibaResult<Option<(Py<PyAny>, Py<PyAny>)>> {
        let py = self.py();
        if !self.is(node, "TypeApp")? {
            return Ok(None);
        }
        let constructor = self.field_string(node, "constructor")?;
        let resolved = self.resolve_name(&constructor, module, Side::Sup)?;
        if self.result_is_err(&resolved)? {
            return Ok(None);
        }
        let target = resolved.bind(py).getattr("ok_value")?;
        if !self.instance_of(&target, "AstNodeType")? {
            return Ok(None);
        }
        let definition = target.getattr("ast_node")?;
        if !self.is(&definition, "GenericDefinition")? {
            return Ok(None);
        }
        let params = self.generic_params(&definition)?;
        let args = self.field_list(node, "args")?;
        if params.len() != args.len() {
            return Ok(None);
        }
        let body =
            crate::chain::convert_to_chain_style(&self.ctx.nodes, &definition.getattr("body")?)?;
        if !self.is_in(&body.bind(py), EXP_NODES)? {
            return Ok(None);
        }
        let elements = self.exponent_elements(body.bind(py))?;
        if elements.len() != 2 || !self.is(elements[0].bind(py), "Never")? {
            return Ok(None);
        }
        let index = match self.param_index(elements[1].bind(py), &params)? {
            Some(index) => index,
            None => return Ok(None),
        };
        Ok(Some((
            args[index].clone().unbind(),
            module.clone().unbind(),
        )))
    }

    fn never_head_exponent(
        &self,
        node: &Bound<'_, PyAny>,
        module: &Bound<'_, PyAny>,
    ) -> VibaResult<Option<(Py<PyAny>, Py<PyAny>)>> {
        let py = self.py();
        if !self.is_in(node, EXP_NODES)? {
            return Ok(None);
        }
        let elements = self.exponent_elements(node)?;
        if elements.len() != 2 || !self.is(elements[0].bind(py), "Never")? {
            return Ok(None);
        }
        let operand = elements[1].bind(py);
        let operand = if self.is(operand, "Tagged")? {
            operand.getattr("type")?.unbind()
        } else {
            operand.clone().unbind()
        };
        Ok(Some((operand, module.clone().unbind())))
    }

    /// Both sides never-headed: same tags, every sub branch settled.
    fn same_branches_settled(
        &self,
        sub_operand: &(Py<PyAny>, Py<PyAny>),
        sup_operand: &(Py<PyAny>, Py<PyAny>),
    ) -> VibaResult<bool> {
        let py = self.py();
        let sub_branches =
            match self.sum_branches(sub_operand.0.bind(py), sub_operand.1.bind(py))? {
                Some(branches) => branches,
                None => return Ok(false),
            };
        let sup_branches =
            match self.sum_branches(sup_operand.0.bind(py), sup_operand.1.bind(py))? {
                Some(branches) => branches,
                None => return Ok(false),
            };
        let mut sub_tags: Vec<String> =
            sub_branches.iter().map(|(tag, _, _)| tag.clone()).collect();
        let mut sup_tags: Vec<String> =
            sup_branches.iter().map(|(tag, _, _)| tag.clone()).collect();
        sub_tags.sort();
        sup_tags.sort();
        if sub_tags != sup_tags {
            return Ok(false);
        }
        for (_, branch, module) in &sub_branches {
            if !self.is_terminator(branch.bind(py), module.bind(py))? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn is_terminator(
        &self,
        node: &Bound<'_, PyAny>,
        module: &Bound<'_, PyAny>,
    ) -> VibaResult<bool> {
        let py = self.py();
        let (node, _module) = self.unfold_ref(node, module, Side::Sub)?;
        let node = node.bind(py);
        if self.is(node, "Never")? {
            return Ok(true);
        }
        if !self.is(node, "TypeApp")? {
            return Ok(false);
        }
        Ok(self
            .terminators
            .contains(&self.field_string(node, "constructor")?))
    }

    fn branches_settled(
        &self,
        sn: &Bound<'_, PyAny>,
        s_mod: &Bound<'_, PyAny>,
        sup_operand: &(Py<PyAny>, Py<PyAny>),
    ) -> VibaResult<bool> {
        let py = self.py();
        let branches = match self.sum_branches(sup_operand.0.bind(py), sup_operand.1.bind(py))? {
            Some(branches) => branches,
            None => return Ok(false),
        };
        let fields = self.tagged_fields(sn, s_mod, &HashSet::new())?;
        for (tag, branch_type, branch_mod) in &branches {
            if !self.branch_settled(fields.get(tag), branch_type.bind(py), branch_mod.bind(py))? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// One branch is settled by a terminator at that tag, or by a field that
    /// counts as `never <- branch`, whose argument is a supertype of it.
    fn branch_settled(
        &self,
        field: Option<&Member>,
        branch_type: &Bound<'_, PyAny>,
        branch_mod: &Bound<'_, PyAny>,
    ) -> VibaResult<bool> {
        let py = self.py();
        let field = match field {
            Some(field) => field,
            None => return Ok(false),
        };
        let (node, module) =
            self.unfold_ref(field.node.bind(py), field.module.bind(py), Side::Sub)?;
        if self.is_terminator(node.bind(py), module.bind(py))? {
            return Ok(true);
        }
        let meaning = match self.never_head_operand(node.bind(py), module.bind(py))? {
            Some(meaning) => meaning,
            None => return Ok(false),
        };
        self.swap_envs();
        let outcome = self.walk(
            branch_type,
            branch_mod,
            meaning.0.bind(py),
            meaning.1.bind(py),
        );
        self.swap_envs();
        outcome
    }

    /// [(tag, body, module)] for the operand of a never-headed chain; None
    /// when a branch is untagged, or when the sum is a cycle.
    fn sum_branches(
        &self,
        node: &Bound<'_, PyAny>,
        module: &Bound<'_, PyAny>,
    ) -> VibaResult<Option<Vec<Branch>>> {
        let py = self.py();
        let mut out = Vec::new();
        let mut seen: HashSet<InlineKey> = HashSet::new();
        let mut stack: Vec<(Py<PyAny>, Py<PyAny>)> =
            vec![(node.clone().unbind(), module.clone().unbind())];
        while let Some((current, current_mod)) = stack.pop() {
            let current_bound = current.bind(py);
            let current_mod_bound = current_mod.bind(py);
            if let Some((key, _name)) =
                self.inlining_key(current_bound, current_mod_bound, Side::Sup)?
            {
                if seen.contains(&key) {
                    return Ok(None);
                }
                seen.insert(key);
            }
            let (current, current_mod) =
                self.unfold_ref(current_bound, current_mod_bound, Side::Sup)?;
            let bound = current.bind(py);
            if self.is(bound, "Sum")? {
                stack.push((bound.getattr("right")?.unbind(), current_mod.clone_ref(py)));
                stack.push((bound.getattr("left")?.unbind(), current_mod.clone_ref(py)));
            } else if self.is(bound, "SumChain")? {
                let elements = self.field_list(bound, "elements")?;
                for element in elements.into_iter().rev() {
                    stack.push((element.unbind(), current_mod.clone_ref(py)));
                }
            } else if self.is(bound, "Tagged")? {
                out.push((
                    self.field_string(bound, "tag")?,
                    bound.getattr("type")?.unbind(),
                    current_mod,
                ));
            } else {
                return Ok(None);
            }
        }
        Ok(Some(out))
    }

    /// {tag: (body, module)} over a product of tags, unfolding named
    /// definitions and named products along the way.
    fn tagged_fields(
        &self,
        node: &Bound<'_, PyAny>,
        module: &Bound<'_, PyAny>,
        seen: &HashSet<InlineKey>,
    ) -> VibaResult<HashMap<String, Member>> {
        let py = self.py();
        let mut seen = seen.clone();
        if let Some((key, name)) = self.inlining_key(node, module, Side::Sub)? {
            if seen.contains(&key) {
                return Err(VibaError::InlineCycle(format!(
                    "the inline chain comes back to {}",
                    self.repr_str(&name)?
                )));
            }
            seen.insert(key);
        }
        let (node, module) = self.unfold_ref(node, module, Side::Sub)?;
        let node = node.bind(py);
        let module = module.bind(py);
        if self.is(node, "Tagged")? {
            let mut fields = HashMap::new();
            fields.insert(
                self.field_string(node, "tag")?,
                Member {
                    node: node.getattr("type")?.unbind(),
                    module: module.clone().unbind(),
                    env: None,
                },
            );
            return Ok(fields);
        }
        if self.is_in(node, PROD_NODES)? {
            let mut fields: HashMap<String, Member> = HashMap::new();
            for element in partial::product_elements(&self.ctx, node)? {
                let element = element.bind(py);
                for (tag, field) in self.tagged_fields(element, module, &seen)? {
                    if fields.contains_key(&tag) {
                        return Err(VibaError::DuplicateTag(format!(
                            "the tag {} appears twice in one product",
                            tag
                        )));
                    }
                    fields.insert(tag, field);
                }
            }
            return Ok(fields);
        }
        Ok(HashMap::new())
    }

    // ------------------------------------------------------------------
    // Applied generics: unfold one side, params bound via env_get
    // ------------------------------------------------------------------

    fn walk_mixed_typeapp(
        &self,
        sn: &Bound<'py, PyAny>,
        s_mod: &Bound<'py, PyAny>,
        sp: &Bound<'py, PyAny>,
        p_mod: &Bound<'py, PyAny>,
    ) -> VibaResult<Option<bool>> {
        let sn_app = self.is(sn, "TypeApp")?;
        if sn_app == self.is(sp, "TypeApp")? {
            return Ok(None);
        }
        let (node, module, side, other, other_mod) = if sn_app {
            (sn, s_mod, Side::Sub, sp, p_mod)
        } else {
            (sp, p_mod, Side::Sup, sn, s_mod)
        };
        if !self.is_generic_application(node, module, side)? {
            return Ok(None);
        }
        Ok(Some(
            self.unfold_typeapp(node, module, side, other, other_mod)?,
        ))
    }

    /// True only when the constructor is a GenericDefinition.
    fn is_generic_application(
        &self,
        node: &Bound<'_, PyAny>,
        module: &Bound<'_, PyAny>,
        side: Side,
    ) -> VibaResult<bool> {
        let py = self.py();
        let constructor = self.field_string(node, "constructor")?;
        let resolved = self.resolve_name(&constructor, module, side)?;
        if self.result_is_err(&resolved)? {
            return Err(VibaError::Unresolved(format!(
                "unresolvable constructor {}",
                self.repr_str(&constructor)?
            )));
        }
        let target = self.ok_value(&resolved.bind(py))?;
        if !self.instance_of(&target, "AstNodeType")? {
            return Ok(false);
        }
        self.is(&target.getattr("ast_node")?, "GenericDefinition")
    }

    /// `g[T].name` -> the chosen file's definition `name`, or None.
    fn member_meaning(
        &self,
        node: &Bound<'_, PyAny>,
        module: &Bound<'_, PyAny>,
        side: Side,
    ) -> VibaResult<Option<Py<PyAny>>> {
        let py = self.py();
        if !self.is(node, "MemberTaken")? {
            return Ok(None);
        }
        let owner = node.getattr("owner")?;
        if !self.is(&owner, "TypeApp")? {
            return Ok(None);
        }
        let (owner, modules) = self.bound_arguments(&owner, module, side)?;
        let decision = self.decision(owner.bind(py), module, Some(&modules))?;
        let chosen = match decision {
            Some(chosen) => chosen,
            None => return Ok(None),
        };
        let chosen = chosen.bind(py);
        let name = self.field_string(node, "name")?;
        let entry = chosen.getattr("entry")?;
        let definition = self.definition_named(&entry.getattr("module")?, &name)?;
        let definition = match definition {
            Some(definition) => definition,
            None => {
                let path: String = entry.getattr("path")?.extract()?;
                return Err(VibaError::Unresolved(format!(
                    "{}: no definition named {}",
                    path,
                    self.repr_str(&name)?
                )));
            }
        };
        let definition = definition.bind(py);
        Ok(Some(self.ast_type_env(
            &definition.getattr("body")?,
            &chosen.getattr("module")?,
            &chosen.getattr("env_get")?,
        )?))
    }

    /// The member's body takes, in the walk, the side the member take had.
    fn unfold_member(
        &self,
        entry: &Bound<'py, PyAny>,
        side: Side,
        other: &Bound<'py, PyAny>,
        other_mod: &Bound<'py, PyAny>,
    ) -> VibaResult<bool> {
        self.push_env(side, &entry.getattr("env_get")?.unbind());
        let outcome = self.walk_unfolded(entry, side, other, other_mod);
        self.pop_env(side);
        outcome
    }

    /// Whether this application is a generic's, decided.
    fn decided(&self, node: &Bound<'_, PyAny>, module: &Bound<'_, PyAny>) -> VibaResult<bool> {
        Ok(self.decision(node, module, None)?.is_some())
    }

    /// The decision this application makes, worked out once per node.
    fn decision(
        &self,
        node: &Bound<'_, PyAny>,
        module: &Bound<'_, PyAny>,
        modules: Option<&[Py<PyAny>]>,
    ) -> VibaResult<Option<Py<PyAny>>> {
        let py = self.py();
        let key = DecisionKey {
            node: ObjId::of(node),
            module: ObjId::of(module),
            modules: modules.map(|list| list.iter().map(|one| ObjId(one.clone_ref(py))).collect()),
        };
        if let Some(found) = self.decisions.borrow().get(&key) {
            match found {
                Decision::None => return Ok(None),
                Decision::Chosen(choice) => return Ok(Some(choice.clone_ref(py))),
                Decision::Failed(message) => return Err(VibaError::Unresolved(message.clone())),
            }
        }
        let got = pattern::reduce_application(&self.ctx, node, module, modules, None);
        let decision = match got {
            Ok(Some(choice)) => Decision::Chosen(choice),
            Ok(None) => Decision::None,
            Err(VibaError::Py(error)) => return Err(VibaError::Py(error)),
            Err(error) => Decision::Failed(error.message()),
        };
        let answer = match &decision {
            Decision::None => Ok(None),
            Decision::Chosen(choice) => Ok(Some(choice.clone_ref(py))),
            Decision::Failed(message) => Err(VibaError::Unresolved(message.clone())),
        };
        self.decisions.borrow_mut().insert(key, decision);
        answer
    }

    /// (the application with the names this walk bound put in, the modules).
    fn bound_arguments(
        &self,
        owner: &Bound<'_, PyAny>,
        module: &Bound<'_, PyAny>,
        side: Side,
    ) -> VibaResult<(Py<PyAny>, Vec<Py<PyAny>>)> {
        let py = self.py();
        let mut arguments: Vec<Py<PyAny>> = Vec::new();
        let mut modules: Vec<Py<PyAny>> = Vec::new();
        for argument in self.field_list(owner, "args")? {
            let mut bound: Option<Py<PyAny>> = None;
            if self.is(&argument, "TypeRef")? {
                let name = self.field_string(&argument, "name")?;
                bound = self.walk_binding(&name, side)?;
            }
            if let Some(bound) = bound {
                let entry = bound.bind(py);
                if self.instance_of(entry, "AstNodeType")? {
                    arguments.push(entry.getattr("ast_node")?.unbind());
                    modules.push(entry.getattr("container_module")?.unbind());
                    continue;
                }
            }
            arguments.push(argument.clone().unbind());
            modules.push(module.clone().unbind());
        }
        let application = self.ctx.make(
            "TypeApp",
            &[
                ("constructor", owner.getattr("constructor")?.unbind()),
                ("args", self.ctx.list(arguments)?),
            ],
        )?;
        Ok((application, modules))
    }

    /// The part this walk bound that name to, or None.
    fn walk_binding(&self, name: &str, side: Side) -> VibaResult<Option<Py<PyAny>>> {
        let py = self.py();
        let envs: Vec<Py<PyAny>> = self
            .env_stacks
            .borrow()
            .get(side)
            .iter()
            .map(|env| env.clone_ref(py))
            .collect();
        for env in envs.iter().rev() {
            let got = env.bind(py).call1((name,))?;
            if !type_system::is_err(&got)? {
                return Ok(Some(got.getattr("ok_value")?.unbind()));
            }
        }
        Ok(None)
    }

    fn unfold_typeapp(
        &self,
        node: &Bound<'py, PyAny>,
        module: &Bound<'py, PyAny>,
        side: Side,
        other: &Bound<'py, PyAny>,
        other_mod: &Bound<'py, PyAny>,
    ) -> VibaResult<bool> {
        let py = self.py();
        let app_key = self.app_key(node, module, side)?;
        if self.unfolding.borrow().contains(&app_key) {
            return Ok(true); // coinductive assumption (fallback when unfolding)
        }
        let entry = match self.applied_meaning(&app_key, node, module, side)? {
            Some(entry) => entry,
            None => return Ok(false),
        };
        self.unfolding.borrow_mut().insert(app_key.clone());
        self.push_env(side, &entry.bind(py).getattr("env_get")?.unbind());
        let outcome = self.walk_unfolded(entry.bind(py), side, other, other_mod);
        self.pop_env(side);
        self.unfolding.borrow_mut().remove(&app_key);
        outcome
    }

    /// The generic's body takes the TypeApp's side in the walk.
    fn walk_unfolded(
        &self,
        entry: &Bound<'_, PyAny>,
        side: Side,
        other: &Bound<'py, PyAny>,
        other_mod: &Bound<'py, PyAny>,
    ) -> VibaResult<bool> {
        let body = entry.getattr("ast_node")?;
        let container_module = entry.getattr("container_module")?;
        if side == Side::Sub {
            self.walk(&body, &container_module, other, other_mod)
        } else {
            self.walk(other, other_mod, &body, &container_module)
        }
    }

    /// TypeApp -> the generic's body as an AstNodeType whose env_get binds
    /// formal parameters to the actual arguments.
    fn applied_meaning(
        &self,
        app_key: &AppKey,
        node: &Bound<'_, PyAny>,
        module: &Bound<'_, PyAny>,
        side: Side,
    ) -> VibaResult<Option<Py<PyAny>>> {
        let py = self.py();
        if let Some(cached) = self.unfolded.borrow().get(app_key) {
            return Ok(Some(cached.clone_ref(py)));
        }
        if let Some(chosen) = self.decision(node, module, None)? {
            let chosen = chosen.bind(py);
            let body = chosen.getattr("body")?;
            if body.is_none() {
                // This file gives no `__decl__`: it answers its own module.
                return Ok(None);
            }
            let entry = self.ast_type_env(
                &body,
                &chosen.getattr("module")?,
                &chosen.getattr("env_get")?,
            )?;
            self.unfolded
                .borrow_mut()
                .insert(app_key.clone(), entry.clone_ref(py));
            return Ok(Some(entry));
        }
        let target = self.constructor_target(node, module, side)?;
        let target_bound = target.bind(py);
        if !self.instance_of(target_bound, "AstNodeType")? {
            return Ok(None);
        }
        let definition = target_bound.getattr("ast_node")?;
        let params = self.generic_params(&definition)?;
        let args = self.field_list(node, "args")?;
        if params.is_empty() || params.len() != args.len() {
            return Ok(None);
        }
        let env_get = self.binder(&params, &args, module, side)?;
        let entry = self.ast_type_env(
            &definition.getattr("body")?,
            &target_bound.getattr("container_module")?,
            env_get.bind(py),
        )?;
        self.unfolded
            .borrow_mut()
            .insert(app_key.clone(), entry.clone_ref(py));
        Ok(Some(entry))
    }

    fn constructor_target(
        &self,
        node: &Bound<'_, PyAny>,
        module: &Bound<'_, PyAny>,
        side: Side,
    ) -> VibaResult<Py<PyAny>> {
        let constructor = self.field_string(node, "constructor")?;
        let resolved = self.resolve_name(&constructor, module, side)?;
        if self.result_is_err(&resolved)? {
            return Err(VibaError::Unresolved(format!(
                "unresolvable constructor {}",
                self.repr_str(&constructor)?
            )));
        }
        Ok(self.ok_value(&resolved.bind(self.py()))?.unbind())
    }

    fn binder(
        &self,
        params: &[String],
        args: &[Bound<'py, PyAny>],
        module: &Bound<'_, PyAny>,
        side: Side,
    ) -> VibaResult<Py<PyAny>> {
        let mut meanings = HashMap::new();
        for (name, arg) in params.iter().zip(args.iter()) {
            meanings.insert(name.clone(), self.meaning(arg, module, side)?);
        }
        Ok(Py::new(self.py(), Binder { meanings })?.into_any())
    }

    /// Eager meaning of an actual argument, as the Ok/Err result env_get gives.
    fn meaning(
        &self,
        arg: &Bound<'_, PyAny>,
        module: &Bound<'_, PyAny>,
        side: Side,
    ) -> VibaResult<Py<PyAny>> {
        if self.is(arg, "TypeRef")? {
            let name = self.field_string(arg, "name")?;
            return self.resolve_name(&name, module, side);
        }
        let entry = self.ast_type(arg, module)?;
        let leaf = self.as_leaf(entry.bind(self.py()))?;
        Ok(type_system::ok(self.py(), leaf)?)
    }

    /// Stable identity for an unfolding: same node, same resolved actuals.
    fn app_key(
        &self,
        node: &Bound<'_, PyAny>,
        module: &Bound<'_, PyAny>,
        side: Side,
    ) -> VibaResult<AppKey> {
        let py = self.py();
        let mut parts = Vec::new();
        for arg in self.field_list(node, "args")? {
            let meaning = self.meaning(&arg, module, side)?;
            let meaning = meaning.bind(py);
            if type_system::is_err(meaning)? {
                parts.push(TypeKey::ErrMarker);
            } else {
                parts.push(self.type_key(&meaning.getattr("ok_value")?, &[])?);
            }
        }
        Ok(AppKey {
            node: ObjId::of(node),
            side,
            module: ObjId::of(module),
            parts,
        })
    }

    fn walk_lifted(
        &self,
        sn: &Bound<'py, PyAny>,
        s_mod: &Bound<'py, PyAny>,
        sp: &Bound<'py, PyAny>,
        p_mod: &Bound<'py, PyAny>,
    ) -> VibaResult<Option<bool>> {
        let py = self.py();
        let sub_t = self.lift(sn, s_mod, Side::Sub)?;
        let sup_t = self.lift(sp, p_mod, Side::Sup)?;
        if let (Some(sub_t), Some(sup_t)) = (&sub_t, &sup_t) {
            return Ok(Some(self.check(sub_t.bind(py), sup_t.bind(py))?));
        }
        if let Some(sub_t) = sub_t {
            return Ok(Some(self.leaf_vs_struct(sn, s_mod, sp, p_mod, &sub_t)?));
        }
        if let Some(sup_t) = sup_t {
            return Ok(Some(self.struct_vs_leaf(sn, s_mod, sp, p_mod, &sup_t)?));
        }
        Ok(None)
    }

    fn leaf_vs_struct(
        &self,
        sn: &Bound<'py, PyAny>,
        s_mod: &Bound<'py, PyAny>,
        sp: &Bound<'py, PyAny>,
        p_mod: &Bound<'py, PyAny>,
        sub_t: &Py<PyAny>,
    ) -> VibaResult<bool> {
        let py = self.py();
        if self.is_in(sp, SUM_NODES)? {
            return self.any_branch(sn, s_mod, sp, p_mod);
        }
        if self.is(sp, "Tagged")? {
            return self.walk(sn, s_mod, &sp.getattr("type")?, p_mod);
        }
        let other = self.ast_type(sp, p_mod)?;
        self.check(sub_t.bind(py), other.bind(py))
    }

    fn struct_vs_leaf(
        &self,
        sn: &Bound<'py, PyAny>,
        s_mod: &Bound<'py, PyAny>,
        sp: &Bound<'py, PyAny>,
        p_mod: &Bound<'py, PyAny>,
        sup_t: &Py<PyAny>,
    ) -> VibaResult<bool> {
        let py = self.py();
        if self.is_in(sn, SUM_NODES)? {
            return self.all_branches(sn, s_mod, sp, p_mod);
        }
        if self.is(sn, "Tagged")? {
            return self.walk(&sn.getattr("type")?, s_mod, sp, p_mod);
        }
        let other = self.ast_type(sn, s_mod)?;
        self.check(other.bind(py), sup_t.bind(py))
    }

    fn any_branch(
        &self,
        sn: &Bound<'py, PyAny>,
        s_mod: &Bound<'py, PyAny>,
        sp: &Bound<'py, PyAny>,
        p_mod: &Bound<'py, PyAny>,
    ) -> VibaResult<bool> {
        let py = self.py();
        for branch in self.flatten_sum(sp)? {
            if self.walk(sn, s_mod, branch.bind(py), p_mod)? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn all_branches(
        &self,
        sn: &Bound<'py, PyAny>,
        s_mod: &Bound<'py, PyAny>,
        sp: &Bound<'py, PyAny>,
        p_mod: &Bound<'py, PyAny>,
    ) -> VibaResult<bool> {
        let py = self.py();
        for branch in self.flatten_sum(sn)? {
            if !self.walk(branch.bind(py), s_mod, sp, p_mod)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    // ------------------------------------------------------------------
    // Structural cases (neither side lifts)
    // ------------------------------------------------------------------

    fn walk_structural(
        &self,
        sn: &Bound<'py, PyAny>,
        s_mod: &Bound<'py, PyAny>,
        sp: &Bound<'py, PyAny>,
        p_mod: &Bound<'py, PyAny>,
    ) -> VibaResult<bool> {
        if self.is_in(sp, SUM_NODES)? {
            return self.walk_sum(sn, s_mod, sp, p_mod);
        }
        if self.is_in(sn, SUM_NODES)? {
            return self.all_branches(sn, s_mod, sp, p_mod);
        }
        if self.is(sp, "Tagged")? {
            return self.walk_tagged(sn, s_mod, sp, p_mod);
        }
        if self.is_in(sp, PROD_NODES)? {
            return self.walk_products(sn, s_mod, sp, p_mod);
        }
        if self.is(sp, "Tuple")? {
            return self.walk_tuple(sn, s_mod, sp, p_mod);
        }
        if self.is_in(sp, EXP_NODES)? {
            return self.walk_exponent(sn, s_mod, sp, p_mod);
        }
        if self.is(sp, "TypeApp")? && self.is(sn, "TypeApp")? {
            return self.walk_typeapps(sn, s_mod, sp, p_mod);
        }
        if self.is(sp, "HintBlock")? {
            // A hint block has no members: the unit is its only resident.
            if self.is(sn, "HintBlock")? {
                return Ok(true);
            }
            if let Some(lifted) = self.lift(sn, s_mod, Side::Sub)? {
                return self.instance_of(&lifted.bind(self.py()), "NilType");
            }
            return Ok(false);
        }
        Ok(false)
    }

    fn walk_sum(
        &self,
        sn: &Bound<'py, PyAny>,
        s_mod: &Bound<'py, PyAny>,
        sp: &Bound<'py, PyAny>,
        p_mod: &Bound<'py, PyAny>,
    ) -> VibaResult<bool> {
        if self.is_in(sn, SUM_NODES)? {
            return self.all_branches(sn, s_mod, sp, p_mod);
        }
        self.any_branch(sn, s_mod, sp, p_mod)
    }

    /// `...`, the rest of the arguments, taken as a supertype.
    fn walk_rest(&self, sn: &Bound<'_, PyAny>, s_mod: &Bound<'_, PyAny>) -> VibaResult<bool> {
        let py = self.py();
        if self.is(sn, "Ellipsis")?
            || self.is(sn, "Tagged")?
            || self.is(sn, "Product")?
            || self.is(sn, "ProductChain")?
        {
            return Ok(true);
        }
        if self.is_in(sn, SUM_NODES)? {
            for branch in self.flatten_sum(sn)? {
                if !self.walk_rest(branch.bind(py), s_mod)? {
                    return Ok(false);
                }
            }
            return Ok(true);
        }
        Ok(false)
    }

    fn walk_tagged(
        &self,
        sn: &Bound<'py, PyAny>,
        s_mod: &Bound<'py, PyAny>,
        sp: &Bound<'py, PyAny>,
        p_mod: &Bound<'py, PyAny>,
    ) -> VibaResult<bool> {
        if self.is_in(sn, PROD_NODES)? {
            let (tagged, _bare) =
                self.split_product(sn, s_mod, Side::Sub, &HashSet::new(), None)?;
            let tag = self.field_string(sp, "tag")?;
            let member = match tagged.get(&tag) {
                Some(member) => member,
                None => return Ok(false),
            };
            let other = Member {
                node: sp.getattr("type")?.unbind(),
                module: p_mod.clone().unbind(),
                env: None,
            };
            return self.walk_bound(member, &other);
        }
        if !self.is(sn, "Tagged")? {
            return self.walk(sn, s_mod, &sp.getattr("type")?, p_mod);
        }
        if self.field_string(sn, "tag")? != self.field_string(sp, "tag")? {
            return Ok(false);
        }
        self.walk(&sn.getattr("type")?, s_mod, &sp.getattr("type")?, p_mod)
    }

    /// Tagged products match by tag (commutative); sup's tags must all be
    /// present in sub (width). Untagged members pair in order.
    fn walk_products(
        &self,
        sn: &Bound<'py, PyAny>,
        s_mod: &Bound<'py, PyAny>,
        sp: &Bound<'py, PyAny>,
        p_mod: &Bound<'py, PyAny>,
    ) -> VibaResult<bool> {
        let (sub_tagged, sub_bare) =
            self.split_product(sn, s_mod, Side::Sub, &HashSet::new(), None)?;
        let (sup_tagged, sup_bare) =
            self.split_product(sp, p_mod, Side::Sup, &HashSet::new(), None)?;
        for (tag, sup_member) in &sup_tagged {
            let sub_member = match sub_tagged.get(tag) {
                Some(member) => member,
                None => return Ok(false),
            };
            if !self.walk_bound(sub_member, sup_member)? {
                return Ok(false);
            }
        }
        if sub_bare.len() != sup_bare.len() {
            return Ok(false);
        }
        for (sub_member, sup_member) in sub_bare.iter().zip(sup_bare.iter()) {
            if !self.walk_bound(sub_member, sup_member)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Compare two member bodies, each under its own bindings in the source.
    fn walk_bound(&self, sub: &Member, sup: &Member) -> VibaResult<bool> {
        let py = self.py();
        let mut pushed = Vec::new();
        if let Some(env) = &sub.env {
            self.push_env(Side::Sub, env);
            pushed.push(Side::Sub);
        }
        if let Some(env) = &sup.env {
            self.push_env(Side::Sup, env);
            pushed.push(Side::Sup);
        }
        let outcome = self.walk(
            sub.node.bind(py),
            sub.module.bind(py),
            sup.node.bind(py),
            sup.module.bind(py),
        );
        for side in pushed {
            self.pop_env(side);
        }
        outcome
    }

    /// Flatten a Product/ProductChain into ({tag: member}, [member]).
    fn split_product(
        &self,
        node: &Bound<'_, PyAny>,
        module: &Bound<'_, PyAny>,
        side: Side,
        seen: &HashSet<InlineKey>,
        env: Option<&Py<PyAny>>,
    ) -> VibaResult<(HashMap<String, Member>, Vec<Member>)> {
        let py = self.py();
        let mut tagged: HashMap<String, Member> = HashMap::new();
        let mut bare: Vec<Member> = Vec::new();
        for elem in partial::product_elements(&self.ctx, node)? {
            let elem = elem.bind(py);
            let (element_tagged, element_bare) =
                self.split_element(elem, module, side, seen, env)?;
            for (tag, member) in element_tagged {
                if tagged.contains_key(&tag) {
                    return Err(VibaError::DuplicateTag(format!(
                        "the tag {} appears twice in one product",
                        tag
                    )));
                }
                tagged.insert(tag, member);
            }
            bare.extend(element_bare);
        }
        Ok((tagged, bare))
    }

    /// A product with one member left, as that member: (node, module, env).
    fn reduce_product(
        &self,
        node: &Bound<'_, PyAny>,
        module: &Bound<'_, PyAny>,
        side: Side,
    ) -> VibaResult<Option<(Py<PyAny>, Py<PyAny>, Option<Py<PyAny>>)>> {
        if !self.is_in(node, PROD_NODES)? {
            return Ok(None);
        }
        let (tagged, bare) = self.split_product(node, module, side, &HashSet::new(), None)?;
        if tagged.is_empty() && bare.is_empty() {
            return Ok(Some((
                self.ctx.nodes.make0("Nil")?,
                module.clone().unbind(),
                None,
            )));
        }
        if tagged.is_empty() && bare.len() == 1 {
            let member = bare.into_iter().next().unwrap();
            return Ok(Some((member.node, member.module, member.env)));
        }
        if bare.is_empty() && tagged.len() == 1 {
            let (tag, member) = tagged.into_iter().next().unwrap();
            let tagged_node = self.ctx.make(
                "Tagged",
                &[("tag", self.ctx.pystr(&tag)), ("type", member.node)],
            )?;
            return Ok(Some((tagged_node, member.module, member.env)));
        }
        Ok(None)
    }

    /// One untagged member: what it contributes to ({tag: member}, [member]).
    fn split_element(
        &self,
        elem: &Bound<'_, PyAny>,
        module: &Bound<'_, PyAny>,
        side: Side,
        seen: &HashSet<InlineKey>,
        env: Option<&Py<PyAny>>,
    ) -> VibaResult<(HashMap<String, Member>, Vec<Member>)> {
        let py = self.py();
        let mut pushed = 0usize;
        let mut ambient: Option<Py<PyAny>> = env.map(|value| value.clone_ref(py));
        let outcome = (|| -> VibaResult<(HashMap<String, Member>, Vec<Member>)> {
            let mut node = elem.clone().unbind();
            let mut node_mod = module.clone().unbind();
            let mut seen = seen.clone();
            loop {
                let (next_node, next_mod) = self.partial(node.bind(py), node_mod.bind(py), side)?;
                node = next_node;
                node_mod = next_mod;
                if let Some(unit) = self.config_unit(node.bind(py))? {
                    if unit == Unit::Nil {
                        // A unit a config names is no member.
                        return Ok((HashMap::new(), Vec::new()));
                    }
                }
                if let Some((key, name)) =
                    self.inlining_key(node.bind(py), node_mod.bind(py), side)?
                {
                    if seen.contains(&key) {
                        return Err(VibaError::InlineCycle(format!(
                            "the inline chain comes back to {}",
                            self.repr_str(&name)?
                        )));
                    }
                    seen.insert(key);
                }
                let (next_node, next_mod) =
                    self.unfold_ref(node.bind(py), node_mod.bind(py), side)?;
                node = next_node;
                node_mod = next_mod;
                if !self.is(node.bind(py), "TypeApp")? {
                    break;
                }
                let parts = self.application_parts(node.bind(py), node_mod.bind(py), side)?;
                let (body, container_module, app_key, binder) = match parts {
                    Some(parts) => parts,
                    None => break,
                };
                if seen.contains(&app_key) {
                    let constructor = self.field_string(node.bind(py), "constructor")?;
                    return Err(VibaError::InlineCycle(format!(
                        "the inline chain comes back to {}",
                        self.repr_str(&constructor)?
                    )));
                }
                seen.insert(app_key);
                self.push_env(side, &binder);
                pushed += 1;
                ambient = Some(binder);
                node = body;
                node_mod = container_module;
            }
            let node_bound = node.bind(py);
            if self.is_in(node_bound, PROD_NODES)? {
                let (tagged, bare) = self.split_product(
                    node_bound,
                    node_mod.bind(py),
                    side,
                    &seen,
                    ambient.as_ref(),
                )?;
                return Ok((tagged, bare));
            }
            if self.is(node_bound, "Tagged")? {
                let mut fields = HashMap::new();
                fields.insert(
                    self.field_string(node_bound, "tag")?,
                    Member {
                        node: node_bound.getattr("type")?.unbind(),
                        module: node_mod.clone_ref(py),
                        env: ambient,
                    },
                );
                return Ok((fields, Vec::new()));
            }
            if self.is_product_unit(node_bound, node_mod.bind(py), side)? {
                return Ok((HashMap::new(), Vec::new()));
            }
            Ok((
                HashMap::new(),
                vec![Member {
                    node: elem.clone().unbind(),
                    module: module.clone().unbind(),
                    env: env.map(|value| value.clone_ref(py)),
                }],
            ))
        })();
        for _ in 0..pushed {
            self.pop_env(side);
        }
        outcome
    }

    /// (body, module, key, binder) for a generic application with a body.
    fn application_parts(
        &self,
        node: &Bound<'_, PyAny>,
        module: &Bound<'_, PyAny>,
        side: Side,
    ) -> VibaResult<Option<(Py<PyAny>, Py<PyAny>, InlineKey, Py<PyAny>)>> {
        let py = self.py();
        let constructor = self.field_string(node, "constructor")?;
        let resolved = self.resolve_name(&constructor, module, side)?;
        if self.result_is_err(&resolved)? {
            return Ok(None);
        }
        let target = resolved.bind(py).getattr("ok_value")?;
        if !self.instance_of(&target, "AstNodeType")? {
            return Ok(None);
        }
        let definition = target.getattr("ast_node")?;
        if !self.is(&definition, "GenericDefinition")? {
            return Ok(None);
        }
        let params = self.generic_params(&definition)?;
        let args = self.field_list(node, "args")?;
        if params.len() != args.len() {
            return Ok(None);
        }
        let binder = self.binder(&params, &args, module, side)?;
        Ok(Some((
            definition.getattr("body")?.unbind(),
            target.getattr("container_module")?.unbind(),
            InlineKey::App(definition.as_ptr() as usize),
            binder,
        )))
    }

    /// (key, the name in the source) of the definition an untagged member names.
    fn inlining_key(
        &self,
        node: &Bound<'_, PyAny>,
        module: &Bound<'_, PyAny>,
        side: Side,
    ) -> VibaResult<Option<(InlineKey, String)>> {
        let py = self.py();
        if !self.is(node, "TypeRef")? {
            return Ok(None);
        }
        let name = self.field_string(node, "name")?;
        let resolved = self.resolve_name(&name, module, side)?;
        if self.result_is_err(&resolved)? {
            return Ok(None);
        }
        let target = resolved.bind(py).getattr("ok_value")?;
        if !self.instance_of(&target, "AstNodeType")? {
            return Ok(None);
        }
        let definition = target.getattr("ast_node")?;
        if !self.is(&definition, "TypeDefinition")? {
            return Ok(None);
        }
        Ok(Some((
            InlineKey::Inline(definition.as_ptr() as usize),
            name,
        )))
    }

    /// Unfold names and give `<<`: until neither has anything left.
    fn normalize(
        &self,
        node: &Bound<'_, PyAny>,
        module: &Bound<'_, PyAny>,
        side: Side,
    ) -> VibaResult<(Py<PyAny>, Py<PyAny>)> {
        let py = self.py();
        let mut node = node.clone().unbind();
        let mut module = module.clone().unbind();
        loop {
            let before = (node.as_ptr(), module.as_ptr());
            let (next_node, next_module) = self.unfold_ref(node.bind(py), module.bind(py), side)?;
            node = next_node;
            module = next_module;
            let (next_node, next_module) = self.partial(node.bind(py), module.bind(py), side)?;
            node = next_node;
            module = next_module;
            if (node.as_ptr(), module.as_ptr()) == before {
                return Ok((node, module));
            }
        }
    }

    /// A design's `<<` reduced: the function with that argument given.
    fn partial(
        &self,
        node: &Bound<'_, PyAny>,
        module: &Bound<'_, PyAny>,
        side: Side,
    ) -> VibaResult<(Py<PyAny>, Py<PyAny>)> {
        let py = self.py();
        if !self.is(node, "Partial")? {
            return Ok((node.clone().unbind(), module.clone().unbind()));
        }
        let key = PartialKey {
            node: ObjId::of(node),
            module: ObjId::of(module),
            side,
            env: self.env_ids(side),
        };
        if let Some(found) = self.partial_memo.borrow().get(&key) {
            return Ok((found.0.clone_ref(py), found.1.clone_ref(py)));
        }
        if let Some(product) = self.get_args_call(node, module, side)? {
            self.partial_memo
                .borrow_mut()
                .insert(key, (product.0.clone_ref(py), product.1.clone_ref(py)));
            return Ok(product);
        }
        let resolve = PartialResolver {
            checker: self,
            side,
        };
        let judge = PartialJudge { checker: self };
        let out = partial::reduce_partial(&self.ctx, node, module, &resolve, &judge, true)?;
        self.partial_memo
            .borrow_mut()
            .insert(key, (out.0.clone_ref(py), out.1.clone_ref(py)));
        Ok(out)
    }

    /// (product, module) for `__get_args__ << <chain>`, or None.
    fn get_args_call(
        &self,
        node: &Bound<'_, PyAny>,
        module: &Bound<'_, PyAny>,
        side: Side,
    ) -> VibaResult<Option<(Py<PyAny>, Py<PyAny>)>> {
        let py = self.py();
        let (head, arguments) = self.partial_parts(node)?;
        let head = head.bind(py);
        if !self.is(head, "TypeRef")? {
            return Ok(None);
        }
        if self.field_string(head, "name")? != GET_ARGS_NAME {
            return Ok(None);
        }
        if arguments.len() != 1 {
            return Err(VibaError::Partial(format!(
                "{} takes one argument: the function type",
                GET_ARGS_NAME
            )));
        }
        let (chain, container_module) = self.normalize(arguments[0].bind(py), module, side)?;
        let chain_bound = chain.bind(py);
        if !self.is_in(chain_bound, EXP_NODES)? {
            return Err(VibaError::Partial(format!(
                "{} asks for a function type, not {}",
                GET_ARGS_NAME,
                self.one_line(chain_bound)?
            )));
        }
        let product = partial::get_args_product(&self.ctx, chain_bound)?;
        Ok(Some((product, container_module)))
    }

    /// (body, module) for the name a `<<` gives to, or None.
    fn partial_target(
        &self,
        name: &str,
        module: &Bound<'_, PyAny>,
        side: Side,
    ) -> VibaResult<Option<(Py<PyAny>, Py<PyAny>)>> {
        let py = self.py();
        let resolved = self.resolve_name(name, module, side)?;
        if self.result_is_err(&resolved)? {
            return partial::module_as_function(&self.ctx, module, name);
        }
        let entry = resolved.bind(py).getattr("ok_value")?;
        if !self.instance_of(&entry, "AstNodeType")? {
            return partial::module_as_function(&self.ctx, module, name);
        }
        let mut node = entry.getattr("ast_node")?;
        if self.is(&node, "GenericDefinition")? {
            return Ok(None); // a bare generic name has no body
        }
        if self.is(&node, "TypeDefinition")? {
            node = node.getattr("body")?;
        }
        let container_module = entry.getattr("container_module")?;
        let builtin = self.builtin_module.bind(py);
        if container_module.as_ptr() != builtin.as_ptr()
            && (self.is(&node, "Exponent")? || self.is(&node, "ExponentChain")?)
        {
            if let Some(problem) = partial::environment_result_problem(&self.ctx, &node, name)? {
                return Err(VibaError::Partial(problem));
            }
        }
        Ok(Some((node.unbind(), container_module.unbind())))
    }

    /// The unit the config calls this piece, or None.
    fn config_unit(&self, node: &Bound<'_, PyAny>) -> VibaResult<Option<Unit>> {
        let name = if self.is(node, "TypeRef")? {
            self.field_string(node, "name")?
        } else if self.is(node, "TypeApp")? {
            self.field_string(node, "constructor")?
        } else {
            return Ok(None);
        };
        let name = last_segment(&name);
        if self.nil_eqv.contains(name) {
            return Ok(Some(Unit::Nil));
        }
        if self.never_eqv.contains(name) {
            return Ok(Some(Unit::Never));
        }
        Ok(None)
    }

    /// The unit in the source that a config names, so the rest of the walk
    /// takes it like any other nil or never; the node itself when it names none.
    fn config_unit_node(&self, node: &Bound<'_, PyAny>) -> VibaResult<Py<PyAny>> {
        match self.config_unit(node)? {
            Some(Unit::Nil) => Ok(self.unit_nil.clone_ref(self.py())),
            Some(Unit::Never) => Ok(self.unit_never.clone_ref(self.py())),
            None => Ok(node.clone().unbind()),
        }
    }

    /// The arguments that carry documentation drop out of the chain.
    fn drop_unit_args(&self, args: &[Py<PyAny>]) -> VibaResult<Vec<Py<PyAny>>> {
        let py = self.py();
        let mut kept = Vec::new();
        for arg in args {
            if !self.carries_documentation(arg.bind(py))? {
                kept.push(arg.clone_ref(py));
            }
        }
        Ok(kept)
    }

    /// The environment is no argument of the design.
    fn drop_env_argument(&self, args: Vec<Py<PyAny>>) -> VibaResult<Vec<Py<PyAny>>> {
        let py = self.py();
        let mut kept = Vec::new();
        for arg in args {
            let bound = arg.bind(py);
            let is_env =
                self.is(bound, "Tagged")? && self.field_string(bound, "tag")? == ENVIRON_TAG;
            if !is_env {
                kept.push(arg.clone_ref(py));
            }
        }
        Ok(kept)
    }

    /// A hint block, or a unit a config names applied to one.
    fn carries_documentation(&self, node: &Bound<'_, PyAny>) -> VibaResult<bool> {
        if self.is(node, "HintBlock")? {
            return Ok(true);
        }
        if !self.is(node, "TypeApp")? {
            return Ok(false);
        }
        if self.config_unit(node)? != Some(Unit::Nil) {
            return Ok(false);
        }
        for arg in self.field_list(node, "args")? {
            if self.has_code_block(&arg)? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// The product's unit: nil by form, a hint block, a name the config calls
    /// nil, or a name (a generic parameter among them) bound to nil.
    fn is_product_unit(
        &self,
        node: &Bound<'_, PyAny>,
        module: &Bound<'_, PyAny>,
        side: Side,
    ) -> VibaResult<bool> {
        let py = self.py();
        if self.is(node, "Nil")? || self.is(node, "HintBlock")? {
            return Ok(true);
        }
        if let Some(unit) = self.config_unit(node)? {
            return Ok(unit == Unit::Nil);
        }
        if !self.is(node, "TypeRef")? {
            return Ok(false);
        }
        let name = self.field_string(node, "name")?;
        let resolved = self.resolve_name(&name, module, side)?;
        if self.result_is_err(&resolved)? {
            return Ok(false);
        }
        Ok(self.instance_of(&self.ok_value(&resolved.bind(py))?, "NilType")?)
    }

    fn walk_tuple(
        &self,
        sn: &Bound<'_, PyAny>,
        s_mod: &Bound<'_, PyAny>,
        sp: &Bound<'_, PyAny>,
        p_mod: &Bound<'_, PyAny>,
    ) -> VibaResult<bool> {
        if !self.is(sn, "Tuple")? {
            return Ok(false);
        }
        let sub_elements = self.field_list(sn, "elements")?;
        let sup_elements = self.field_list(sp, "elements")?;
        if sub_elements.len() != sup_elements.len() {
            return Ok(false);
        }
        for (sub, sup) in sub_elements.iter().zip(sup_elements.iter()) {
            if !self.walk(sub, s_mod, sup, p_mod)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Two functions: the result covariantly, the arguments contravariantly.
    fn walk_exponent(
        &self,
        sn: &Bound<'_, PyAny>,
        s_mod: &Bound<'_, PyAny>,
        sp: &Bound<'_, PyAny>,
        p_mod: &Bound<'_, PyAny>,
    ) -> VibaResult<bool> {
        let py = self.py();
        if !(self.is_in(sn, EXP_NODES)? && self.is_in(sp, EXP_NODES)?) {
            return Ok(false);
        }
        let (sub_res, sub_args) = self.exponent_parts(sn)?;
        let (sup_res, sup_args) = self.exponent_parts(sp)?;
        let mut sub_args = self.drop_env_argument(self.drop_unit_args(&sub_args)?)?;
        let sup_args = self.drop_env_argument(self.drop_unit_args(&sup_args)?)?;
        if sub_args.len() < sup_args.len() {
            for _ in 0..(sup_args.len() - sub_args.len()) {
                sub_args.push(self.ctx.nodes.make0("Never")?);
            }
        }
        if !self.walk(sub_res.bind(py), s_mod, sup_res.bind(py), p_mod)? {
            return Ok(false);
        }
        self.swap_envs();
        let outcome = (|| -> VibaResult<bool> {
            // the sub is cut to the sup's length
            for (sup_arg, sub_arg) in sup_args.iter().zip(sub_args.iter()) {
                if !self.walk(sup_arg.bind(py), p_mod, sub_arg.bind(py), s_mod)? {
                    return Ok(false);
                }
            }
            Ok(true)
        })();
        self.swap_envs();
        outcome
    }

    // ------------------------------------------------------------------
    // Applied generics at the TypeApp level
    // ------------------------------------------------------------------

    fn walk_typeapps(
        &self,
        sn: &Bound<'_, PyAny>,
        s_mod: &Bound<'_, PyAny>,
        sp: &Bound<'_, PyAny>,
        p_mod: &Bound<'_, PyAny>,
    ) -> VibaResult<bool> {
        let py = self.py();
        let sub_constructor = self.field_string(sn, "constructor")?;
        let sup_constructor = self.field_string(sp, "constructor")?;
        let sub_c = self.resolve_constructor(&sub_constructor, s_mod, Side::Sub)?;
        let sup_c = self.resolve_constructor(&sup_constructor, p_mod, Side::Sup)?;
        let sub_c_bound = sub_c.bind(py);
        let sup_c_bound = sup_c.bind(py);
        if self.literal_resident(sn, s_mod, sub_c_bound, sp, p_mod, sup_c_bound)? {
            return Ok(true);
        }
        if self.instance_of(sub_c_bound, "BuiltinGenericType")?
            && self.instance_of(sup_c_bound, "BuiltinGenericType")?
        {
            return self.builtin_against_builtin(sn, s_mod, sub_c_bound, sp, p_mod, sup_c_bound);
        }
        // Every other constructor names a definition, and a definition is an
        // alias of its body: unfold (one side or both) and compare structurally.
        if self.unfold_typeapp(sn, s_mod, Side::Sub, sp, p_mod)? {
            return Ok(true);
        }
        if self.unfold_typeapp(sp, p_mod, Side::Sup, sn, s_mod)? {
            return Ok(true);
        }
        if self.unfolds(sn, s_mod, Side::Sub)? || self.unfolds(sp, p_mod, Side::Sup)? {
            return Ok(false);
        }
        // Neither side has a body to unfold: the actuals do not fit.
        if self.constructor_name(sub_c_bound)? != self.constructor_name(sup_c_bound)? {
            return Ok(false);
        }
        let sub_args = self.field_list(sn, "args")?;
        let sup_args = self.field_list(sp, "args")?;
        if sub_args.len() != sup_args.len() {
            return Ok(false);
        }
        for (sub, sup) in sub_args.iter().zip(sup_args.iter()) {
            if !self.walk(sub, s_mod, sup, p_mod)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn builtin_against_builtin(
        &self,
        sn: &Bound<'_, PyAny>,
        s_mod: &Bound<'_, PyAny>,
        sub_c: &Bound<'_, PyAny>,
        sp: &Bound<'_, PyAny>,
        p_mod: &Bound<'_, PyAny>,
        sup_c: &Bound<'_, PyAny>,
    ) -> VibaResult<bool> {
        if self.field_string(sub_c, "name")? != self.field_string(sup_c, "name")? {
            return Ok(false);
        }
        let sub_args = self.field_list(sn, "args")?;
        let sup_args = self.field_list(sp, "args")?;
        if sub_args.len() != sup_args.len() {
            return Ok(false);
        }
        for (sub, sup) in sub_args.iter().zip(sup_args.iter()) {
            if !self.walk(sub, s_mod, sup, p_mod)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// A *Literal constructor against its container: every literal element
    /// must fit the container's element union.
    fn literal_resident(
        &self,
        sn: &Bound<'_, PyAny>,
        s_mod: &Bound<'_, PyAny>,
        sub_c: &Bound<'_, PyAny>,
        sp: &Bound<'_, PyAny>,
        p_mod: &Bound<'_, PyAny>,
        sup_c: &Bound<'_, PyAny>,
    ) -> VibaResult<bool> {
        let py = self.py();
        if !self.instance_of(sub_c, "BuiltinGenericType")?
            || !self.instance_of(sup_c, "BuiltinGenericType")?
        {
            return Ok(false);
        }
        let sub_name = self.field_string(sub_c, "name")?;
        let sup_name = self.field_string(sup_c, "name")?;
        if literal_of_container(&sup_name) != Some(sub_name.as_str()) {
            return Ok(false);
        }
        if sup_name == "dict" {
            return self.dict_literal_resident(sn, s_mod, sp, p_mod);
        }
        let sup_args = self.field_list(sp, "args")?;
        if sup_args.len() != 1 {
            return Ok(false);
        }
        for arg in self.field_list(sn, "args")? {
            if !self.walk(&arg, s_mod, &sup_args[0], p_mod)? {
                return Ok(false);
            }
        }
        let _ = py;
        Ok(true)
    }

    fn dict_literal_resident(
        &self,
        sn: &Bound<'_, PyAny>,
        s_mod: &Bound<'_, PyAny>,
        sp: &Bound<'_, PyAny>,
        p_mod: &Bound<'_, PyAny>,
    ) -> VibaResult<bool> {
        let sup_args = self.field_list(sp, "args")?;
        if sup_args.len() != 2 {
            return Ok(false);
        }
        for pair in self.field_list(sn, "args")? {
            if !self.dict_pair(&pair, s_mod, &sup_args[0], &sup_args[1], p_mod)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn dict_pair(
        &self,
        pair: &Bound<'_, PyAny>,
        s_mod: &Bound<'_, PyAny>,
        key_t: &Bound<'_, PyAny>,
        val_t: &Bound<'_, PyAny>,
        p_mod: &Bound<'_, PyAny>,
    ) -> VibaResult<bool> {
        if !self.is(pair, "Tuple")? {
            return Ok(false);
        }
        let elements = self.field_list(pair, "elements")?;
        if elements.len() != 2 {
            return Ok(false);
        }
        let keys_ok = self.walk(&elements[0], s_mod, key_t, p_mod)?;
        let vals_ok = self.walk(&elements[1], s_mod, val_t, p_mod)?;
        Ok(keys_ok && vals_ok)
    }

    fn resolve_constructor(
        &self,
        name: &str,
        module: &Bound<'_, PyAny>,
        side: Side,
    ) -> VibaResult<Py<PyAny>> {
        let resolved = self.resolve_name(name, module, side)?;
        if self.result_is_err(&resolved)? {
            return Err(VibaError::Unresolved(format!(
                "unresolvable constructor {}",
                self.repr_str(name)?
            )));
        }
        Ok(self.ok_value(&resolved.bind(self.py()))?.unbind())
    }

    /// Whether this application has a body to unfold.
    fn unfolds(
        &self,
        node: &Bound<'_, PyAny>,
        module: &Bound<'_, PyAny>,
        side: Side,
    ) -> VibaResult<bool> {
        let py = self.py();
        let constructor = self.field_string(node, "constructor")?;
        let resolved = self.resolve_name(&constructor, module, side)?;
        if self.result_is_err(&resolved)? {
            return Ok(false);
        }
        let target = resolved.bind(py).getattr("ok_value")?;
        if !self.instance_of(&target, "AstNodeType")? {
            return Ok(false);
        }
        let definition = target.getattr("ast_node")?;
        if !self.is(&definition, "GenericDefinition")? {
            return Ok(false);
        }
        let params = self.generic_params(&definition)?;
        Ok(params.len() == self.field_list(node, "args")?.len())
    }

    /// The name a constructor has in the source, builtin or definition.
    fn constructor_name(&self, constructor: &Bound<'_, PyAny>) -> VibaResult<String> {
        if self.instance_of(constructor, "BuiltinGenericType")? {
            return self.field_string(constructor, "name");
        }
        if !constructor.hasattr("ast_node")? {
            return Ok(String::new());
        }
        let node = constructor.getattr("ast_node")?;
        Ok(self.as_definition(&node)?.unwrap_or_default())
    }

    // ------------------------------------------------------------------
    // The memo key
    // ------------------------------------------------------------------

    /// Canonical comparison key for the memo table.
    fn type_key(&self, value: &Bound<'_, PyAny>, env: &[ObjId]) -> VibaResult<TypeKey> {
        if self.instance_of(value, "AstNodeType")? {
            let node = value.getattr("ast_node")?;
            let node_key = match self.as_definition(&node)? {
                Some(name) => NodeKey::Defn(name),
                None => NodeKey::Inline(ObjId::of(&node)),
            };
            return Ok(TypeKey::Ast {
                module: ObjId::of(&value.getattr("container_module")?),
                node: node_key,
                env: env.to_vec(),
            });
        }
        if self.instance_of(value, "BuiltinGenericType")? {
            return Ok(TypeKey::Generic(self.field_string(value, "name")?));
        }
        let class = self.class(value)?;
        if is_literal_type(&class) {
            let repr = pyast::repr_of(self.py(), &value.getattr("value")?);
            return Ok(TypeKey::Literal(class, repr));
        }
        Ok(TypeKey::Plain(class))
    }
}

// ----------------------------------------------------------------------
// Python entry points
// ----------------------------------------------------------------------

#[pyfunction(signature = (sub, sup, terminators = None, config = None))]
fn is_sub_type_py<'py>(
    py: Python<'py>,
    sub: &Bound<'py, PyAny>,
    sup: &Bound<'py, PyAny>,
    terminators: Option<&Bound<'py, PyAny>>,
    config: Option<&Bound<'py, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let terminators = terminator_set(terminators)?;
    let outcome =
        Checker::new(py, &terminators, config).and_then(|checker| checker.check(sub, sup));
    match outcome {
        Ok(result) => type_system::ok(py, pyast::pybool(py, result)),
        Err(VibaError::Py(error)) => Err(error),
        Err(error) => type_system::err(py, &error.message()),
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(is_sub_type_py, m)?)?;
    Ok(())
}
