//! A native execution path for straight-line viba modules.
//!
//! This is the first slice of the native runtime: source is compiled into a
//! Rust expression tree, evaluated over Rust values, and only the host steps
//! and the final answer cross into Python.  Anything outside the straight-line
//! shape (imports, generics, patterns, products, snapshots) is left to the
//! existing interpreter, which is the behavioural reference.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use pyo3::prelude::*;
use pyo3::types::{PyList, PyModule, PyString};

use crate::ctx::Ctx;
use crate::pyast;

const ENV_TAG: &str = "$env";

#[derive(Debug)]
enum Unsupported {
    Shape,
}

type Compile<T> = Result<T, Unsupported>;

enum Value {
    Nil,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Env(Py<PyAny>),
    Product(Vec<(Option<String>, Value)>),
}

impl Value {
    fn clone_in(&self, py: Python<'_>) -> Value {
        match self {
            Value::Nil => Value::Nil,
            Value::Bool(value) => Value::Bool(*value),
            Value::Int(value) => Value::Int(*value),
            Value::Float(value) => Value::Float(*value),
            Value::Str(value) => Value::Str(value.clone()),
            Value::Env(value) => Value::Env(value.clone_ref(py)),
            Value::Product(items) => Value::Product(
                items
                    .iter()
                    .map(|(tag, value)| (tag.clone(), value.clone_in(py)))
                    .collect(),
            ),
        }
    }
}

struct Arg {
    tag: Option<String>,
    value: Expr,
}

enum Expr {
    Const(Value),
    Ref(String),
    Member(Box<Expr>, String),
    Call { head: String, args: Vec<Arg> },
    Product(Vec<(Option<String>, Expr)>),
}

struct Program {
    module_name: String,
    steps: HashMap<String, usize>,
    expressions: HashMap<String, Expr>,
    implementation: Expr,
}

static PROGRAM_CACHE: Mutex<Option<HashMap<String, (u64, Arc<Program>)>>> = Mutex::new(None);

static DESCRIPTOR_CACHE: Mutex<Option<HashMap<String, Py<PyAny>>>> = Mutex::new(None);

fn value_key(value: &Value) -> String {
    match value {
        Value::Nil => "nil".to_string(),
        Value::Bool(value) => format!("bool:{value}"),
        Value::Int(value) => format!("int:{value}"),
        Value::Float(value) => format!("float:{:?}", value.to_bits()),
        Value::Str(value) => format!("str:{}:{value}", value.len()),
        Value::Env(_) => "env".to_string(),
        Value::Product(items) => {
            let mut key = String::from("product(");
            for (tag, item) in items {
                key.push_str(tag.as_deref().unwrap_or(""));
                key.push(':');
                key.push_str(&value_key(item));
                key.push(';');
            }
            key.push(')');
            key
        }
    }
}

fn cache_key(path: &str, source: &str) -> (String, u64) {
    let stamp = std::fs::metadata(path)
        .and_then(|metadata| metadata.modified())
        .ok()
        .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|duration| duration.as_nanos() as u64)
        .unwrap_or(0)
        ^ (source.len() as u64).rotate_left(17);
    (path.to_string(), stamp)
}

fn unsupported<T>() -> Compile<T> {
    Err(Unsupported::Shape)
}

fn elements_of(py: Python<'_>, node: &Bound<'_, PyAny>) -> PyResult<Vec<Py<PyAny>>> {
    let ctx = Ctx::new(py)?;
    crate::partial::elements(&ctx, node)
}

fn call_parts(py: Python<'_>, node: &Bound<'_, PyAny>) -> PyResult<(Py<PyAny>, Vec<Py<PyAny>>)> {
    let mut arguments = Vec::new();
    let mut current = node.clone().unbind();
    while pyast::class_name(current.bind(py))? == "Partial" {
        let bound = current.bind(py);
        arguments.push(bound.getattr("argument")?.unbind());
        current = bound.getattr("function")?.unbind();
    }
    arguments.reverse();
    Ok((current, arguments))
}

fn scalar_value(py: Python<'_>, node: &Bound<'_, PyAny>) -> PyResult<Option<Value>> {
    if !pyast::class_name(node)?.eq("Constant") {
        return Ok(None);
    }
    let value = node.getattr("value")?;
    Ok(Some(if value.is_none() {
        Value::Nil
    } else if value.is_instance_of::<pyo3::types::PyBool>() {
        Value::Bool(value.extract()?)
    } else if value.is_instance_of::<pyo3::types::PyInt>() {
        match value.extract::<i64>() {
            Ok(value) => Value::Int(value),
            Err(_) => return Ok(None),
        }
    } else if value.is_instance_of::<pyo3::types::PyFloat>() {
        Value::Float(value.extract()?)
    } else if value.is_instance_of::<PyString>() {
        Value::Str(value.extract()?)
    } else {
        return Ok(None);
    }))
}

fn scalar_type(node: &Bound<'_, PyAny>) -> PyResult<bool> {
    if pyast::class_name(node)? != "TypeRef" {
        return Ok(false);
    }
    let name: String = node.getattr("name")?.extract()?;
    Ok(matches!(
        name.as_str(),
        "bool"
            | "int"
            | "float"
            | "str"
            | "int8_t"
            | "uint8_t"
            | "int16_t"
            | "uint16_t"
            | "int32_t"
            | "uint32_t"
            | "int64_t"
            | "uint64_t"
    ))
}

fn compile_product_part(
    py: Python<'_>,
    node: &Bound<'_, PyAny>,
    items: &mut Vec<(Option<String>, Expr)>,
) -> PyResult<Compile<()>> {
    let class = pyast::class_name(node)?;
    match class.as_str() {
        "Product" => {
            let left = node.getattr("left")?;
            match compile_product_part(py, &left, items)? {
                Ok(()) => {}
                Err(unsupported) => return Ok(Err(unsupported)),
            }
            let right = node.getattr("right")?;
            compile_product_part(py, &right, items)
        }
        "Tagged" => {
            let tag: String = node.getattr("tag")?.extract()?;
            let inner = node.getattr("type")?;
            match compile_expr(py, &inner)? {
                Ok(value) => {
                    items.push((Some(tag), value));
                    Ok(Ok(()))
                }
                Err(unsupported) => Ok(Err(unsupported)),
            }
        }
        "TypeRef" => {
            let name: String = node.getattr("name")?.extract()?;
            if name == "Object" {
                Ok(Ok(()))
            } else {
                match compile_expr(py, node)? {
                    Ok(value) => {
                        items.push((None, value));
                        Ok(Ok(()))
                    }
                    Err(unsupported) => Ok(Err(unsupported)),
                }
            }
        }
        _ => match compile_expr(py, node)? {
            Ok(value) => {
                items.push((None, value));
                Ok(Ok(()))
            }
            Err(unsupported) => Ok(Err(unsupported)),
        },
    }
}

fn compile_expr(py: Python<'_>, node: &Bound<'_, PyAny>) -> PyResult<Compile<Expr>> {
    let class = pyast::class_name(node)?;
    match class.as_str() {
        "Constant" => match scalar_value(py, node)? {
            Some(value) => Ok(Ok(Expr::Const(value))),
            None => Ok(unsupported()),
        },
        "TypeRef" => {
            let name: String = node.getattr("name")?.extract()?;
            if let Some((head, member)) = name.rsplit_once('.') {
                if !head.is_empty() && !member.is_empty() {
                    return Ok(Ok(Expr::Member(
                        Box::new(Expr::Ref(head.to_string())),
                        member.to_string(),
                    )));
                }
            }
            Ok(Ok(Expr::Ref(name)))
        }
        "MemberTaken" => {
            let owner = node.getattr("owner")?;
            let name: String = node.getattr("name")?.extract()?;
            if pyast::class_name(&owner)? != "TypeRef" {
                return Ok(unsupported());
            }
            let base = Expr::Ref(owner.getattr("name")?.extract()?);
            Ok(Ok(Expr::Member(Box::new(base), name)))
        }
        "Product" => {
            let mut items = Vec::new();
            match compile_product_part(py, node, &mut items)? {
                Ok(()) => Ok(Ok(Expr::Product(items))),
                Err(unsupported) => Ok(Err(unsupported)),
            }
        }
        "ProductChain" => {
            let mut items = Vec::new();
            for element in node.getattr("elements")?.try_iter()? {
                let element = element?;
                let class = pyast::class_name(&element)?;
                if class == "TypeRef" {
                    let name: String = element.getattr("name")?.extract()?;
                    if name == "Object" {
                        continue;
                    }
                }
                if class == "Tagged" {
                    let tag: String = element.getattr("tag")?.extract()?;
                    let inner = element.getattr("type")?;
                    match compile_expr(py, &inner)? {
                        Ok(value) => items.push((Some(tag), value)),
                        Err(unsupported) => return Ok(Err(unsupported)),
                    }
                } else {
                    match compile_expr(py, &element)? {
                        Ok(value) => items.push((None, value)),
                        Err(unsupported) => return Ok(Err(unsupported)),
                    }
                }
            }
            Ok(Ok(Expr::Product(items)))
        }
        "Tagged" => {
            let tag: String = node.getattr("tag")?.extract()?;
            let inner = node.getattr("type")?;
            match compile_expr(py, &inner)? {
                Ok(value) => Ok(Ok(Expr::Product(vec![(Some(tag), value)]))),
                Err(unsupported) => Ok(Err(unsupported)),
            }
        }
        "Partial" => {
            let (head, arguments) = call_parts(py, node)?;
            let head = head.bind(py);
            if pyast::class_name(head)? != "TypeRef" {
                return Ok(unsupported());
            }
            let name: String = head.getattr("name")?.extract()?;
            let mut args = Vec::new();
            for argument in arguments {
                let argument = argument.bind(py);
                if pyast::class_name(argument)? == "Tagged" {
                    let tag: String = argument.getattr("tag")?.extract()?;
                    let inner = argument.getattr("type")?;
                    match compile_expr(py, &inner)? {
                        Ok(value) => args.push(Arg {
                            tag: Some(tag),
                            value,
                        }),
                        Err(unsupported) => return Ok(Err(unsupported)),
                    }
                } else {
                    match compile_expr(py, argument)? {
                        Ok(value) => args.push(Arg { tag: None, value }),
                        Err(unsupported) => return Ok(Err(unsupported)),
                    }
                }
            }
            Ok(Ok(Expr::Call { head: name, args }))
        }
        _ => Ok(unsupported()),
    }
}

fn compile(
    py: Python<'_>,
    tree: &Bound<'_, PyAny>,
    module_name: &str,
) -> PyResult<Compile<Program>> {
    let body = match tree.getattr("body") {
        Ok(body) => body,
        Err(_) => return Ok(unsupported()),
    };
    let mut steps = HashMap::new();
    let mut expressions = HashMap::new();
    let mut declaration_env = false;
    let mut implementation = None;
    for node in body.try_iter()? {
        let node = node?;
        match pyast::class_name(&node)?.as_str() {
            "Import" | "Pattern" => return Ok(unsupported()),
            "GenericDefinition" => return Ok(unsupported()),
            "TypeDefinition" => {
                let name: String = node.getattr("name")?.extract()?;
                let definition_body = node.getattr("body")?;
                let definition_class = pyast::class_name(&definition_body)?;
                if definition_class == "Exponent" || definition_class == "ExponentChain" {
                    let elements = elements_of(py, &definition_body)?;
                    if elements.len() >= 2 {
                        let last = elements.last().unwrap().bind(py);
                        if pyast::class_name(last)? == "HintBlock" {
                            let mut declared = elements[1..elements.len() - 1].iter();
                            let Some(first) = declared.next() else {
                                return Ok(unsupported());
                            };
                            let first = first.bind(py);
                            if pyast::class_name(first)? != "Tagged"
                                || first.getattr("tag")?.extract::<String>()? != ENV_TAG
                            {
                                return Ok(unsupported());
                            }
                            for parameter in declared {
                                let parameter = parameter.bind(py);
                                if pyast::class_name(parameter)? != "Tagged" {
                                    return Ok(unsupported());
                                }
                                if !scalar_type(&parameter.getattr("type")?)? {
                                    return Ok(unsupported());
                                }
                            }
                            if !scalar_type(&elements[0].bind(py))? {
                                return Ok(unsupported());
                            }
                            steps.insert(name, elements.len() - 2);
                            continue;
                        }
                    }
                    if name == "__decl__" {
                        if elements.len() == 2 {
                            let arg = elements[1].bind(py);
                            if pyast::class_name(arg)? == "Tagged"
                                && arg.getattr("tag")?.extract::<String>()? == ENV_TAG
                            {
                                declaration_env = true;
                                continue;
                            }
                        }
                        return Ok(unsupported());
                    }
                }
                match compile_expr(py, &definition_body)? {
                    Ok(expression) => {
                        if name == "__impl__" {
                            implementation = Some(expression);
                        } else {
                            expressions.insert(name, expression);
                        }
                    }
                    Err(unsupported) => return Ok(Err(unsupported)),
                }
            }
            _ => return Ok(unsupported()),
        }
    }
    if !declaration_env || implementation.is_none() {
        return Ok(unsupported());
    }
    let program = Program {
        module_name: module_name.to_string(),
        steps,
        expressions,
        implementation: implementation.unwrap(),
    };
    match validate(&program) {
        Ok(()) => Ok(Ok(program)),
        Err(unsupported) => Ok(Err(unsupported)),
    }
}

fn validate_expression(
    program: &Program,
    expression: &Expr,
    visiting: &mut HashSet<String>,
    seen: &mut HashSet<String>,
) -> Compile<()> {
    match expression {
        Expr::Const(_) => Ok(()),
        Expr::Ref(name) => {
            if !program.expressions.contains_key(name) {
                return Err(Unsupported::Shape);
            }
            validate_name(program, name, visiting, seen)
        }
        Expr::Member(base, _) => validate_expression(program, base, visiting, seen),
        Expr::Product(items) => {
            for (_, item) in items {
                validate_expression(program, item, visiting, seen)?;
            }
            Ok(())
        }
        Expr::Call { head, args } => {
            if head == "__get_args__" {
                if args.len() != 1
                    || !matches!(&args[0].value, Expr::Ref(name) if name == "__decl__")
                {
                    return Err(Unsupported::Shape);
                }
                return Ok(());
            } else {
                let Some(arity) = program.steps.get(head) else {
                    return Err(Unsupported::Shape);
                };
                if args.len() != *arity || *arity == 0 {
                    return Err(Unsupported::Shape);
                }
                let first = &args[0];
                if first.tag.as_deref() != Some(ENV_TAG) {
                    return Err(Unsupported::Shape);
                }
                let passes_the_environment = matches!(
                    &first.value,
                    Expr::Member(base, name)
                        if name == "env" && matches!(base.as_ref(), Expr::Ref(root) if root == "args")
                );
                if !passes_the_environment {
                    return Err(Unsupported::Shape);
                }
            }
            for arg in args {
                validate_expression(program, &arg.value, visiting, seen)?;
            }
            Ok(())
        }
    }
}

fn validate_name(
    program: &Program,
    name: &str,
    visiting: &mut HashSet<String>,
    seen: &mut HashSet<String>,
) -> Compile<()> {
    if seen.contains(name) {
        return Ok(());
    }
    if !visiting.insert(name.to_string()) {
        return Err(Unsupported::Shape);
    }
    if let Some(expression) = program.expressions.get(name) {
        validate_expression(program, expression, visiting, seen)?;
    }
    visiting.remove(name);
    seen.insert(name.to_string());
    Ok(())
}

fn validate(program: &Program) -> Compile<()> {
    let mut visiting = HashSet::new();
    let mut seen = HashSet::new();
    validate_expression(program, &program.implementation, &mut visiting, &mut seen)
}

fn python_scalar(py: Python<'_>, value: &Value) -> PyResult<Py<PyAny>> {
    match value {
        Value::Nil => Ok(py.None()),
        Value::Bool(value) => Ok(value.into_pyobject(py)?.to_owned().into_any().unbind()),
        Value::Int(value) => Ok(value.into_pyobject(py)?.to_owned().into_any().unbind()),
        Value::Float(value) => Ok(value.into_pyobject(py)?.to_owned().into_any().unbind()),
        Value::Str(value) => Ok(PyString::new(py, value).into_any().unbind()),
        Value::Env(value) => Ok(value.clone_ref(py)),
        Value::Product(_) => Err(pyo3::exceptions::PyTypeError::new_err(
            "a product is not a scalar",
        )),
    }
}

fn value_ast(py: Python<'_>, value: &Value) -> PyResult<Py<PyAny>> {
    let ast = PyModule::import(py, "rs_viba.viba_ast")?;
    match value {
        Value::Product(items) => {
            let mut elements: Vec<Py<PyAny>> =
                vec![ast.getattr("TypeRef")?.call1(("Object",))?.unbind()];
            for (tag, item) in items {
                let child = value_ast(py, item)?;
                let element = match tag {
                    Some(tag) => ast.getattr("Tagged")?.call1((tag, child))?.unbind(),
                    None => child,
                };
                elements.push(element);
            }
            Ok(ast.getattr("ProductChain")?.call1((elements,))?.unbind())
        }
        _ => {
            let scalar = python_scalar(py, value)?;
            Ok(ast.getattr("Constant")?.call1((scalar,))?.unbind())
        }
    }
}

fn value_object(py: Python<'_>, value: &Value) -> PyResult<Py<PyAny>> {
    let ast_class = value_ast(py, value)?;
    let key = value_key(value);
    let cached = DESCRIPTOR_CACHE.lock().ok().and_then(|cache| {
        cache
            .as_ref()
            .and_then(|cache| cache.get(&key).map(|value| value.clone_ref(py)))
    });
    let descriptor = match cached {
        Some(descriptor) => descriptor,
        None => {
            let type_module = PyModule::import(py, "rs_viba.type")?;
            let no_module = type_module.getattr("custom_module")?.call1(("",))?;
            let ast_node_type = type_module
                .getattr("AstNodeType")?
                .call1((ast_class.clone_ref(py), no_module))?;
            let made = PyModule::import(py, "rs_viba.viba_type_descriptor")?
                .getattr("descriptor_of")?
                .call1((ast_node_type,))?
                .unbind();
            if let Ok(mut cache) = DESCRIPTOR_CACHE.lock() {
                cache
                    .get_or_insert_with(HashMap::new)
                    .insert(key, made.clone_ref(py));
            }
            made
        }
    };
    let reflect = PyModule::import(py, "rs_viba.reflect")?;
    Ok(reflect
        .getattr("VObject")?
        .call1((reflect.getattr("access")?, descriptor, ast_class))?
        .unbind())
}

fn value_of_python(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Option<Value>> {
    if value.is_none() {
        return Ok(Some(Value::Nil));
    }
    if value.is_instance_of::<pyo3::types::PyBool>() {
        return Ok(Some(Value::Bool(value.extract()?)));
    }
    if value.is_instance_of::<pyo3::types::PyInt>() {
        return Ok(value.extract::<i64>().ok().map(Value::Int));
    }
    if value.is_instance_of::<pyo3::types::PyFloat>() {
        return Ok(Some(Value::Float(value.extract()?)));
    }
    if value.is_instance_of::<PyString>() {
        return Ok(Some(Value::Str(value.extract()?)));
    }
    if pyast::class_name(value)? == "VObject" {
        return match value.getattr("value") {
            Ok(leaf) => value_of_python(py, &leaf),
            Err(_) => Ok(None),
        };
    }
    Ok(None)
}

fn member(value: &Value, tag: &str, py: Python<'_>) -> Option<Value> {
    let wanted = format!("${}", tag);
    match value {
        Value::Product(items) => items
            .iter()
            .find(|(item_tag, _)| item_tag.as_deref() == Some(wanted.as_str()))
            .map(|(_, value)| value.clone_in(py)),
        _ => None,
    }
}

fn evaluate(
    py: Python<'_>,
    program: &Program,
    environ: &Bound<'_, PyAny>,
    expression: &Expr,
    memo: &mut HashMap<String, Value>,
) -> PyResult<Option<Value>> {
    match expression {
        Expr::Const(value) => Ok(Some(value.clone_in(py))),
        Expr::Ref(name) => {
            if let Some(value) = memo.get(name) {
                return Ok(Some(value.clone_in(py)));
            }
            if let Some(expression) = program.expressions.get(name) {
                let value = evaluate(py, program, environ, expression, memo)?;
                if let Some(value) = &value {
                    memo.insert(name.clone(), value.clone_in(py));
                }
                return Ok(value);
            }
            Ok(None)
        }
        Expr::Member(base, tag) => {
            let Some(base) = evaluate(py, program, environ, base, memo)? else {
                return Ok(None);
            };
            let got = member(&base, tag, py);
            Ok(got)
        }
        Expr::Product(items) => {
            let mut values = Vec::new();
            for (tag, item) in items {
                let Some(value) = evaluate(py, program, environ, item, memo)? else {
                    return Ok(None);
                };
                values.push((tag.clone(), value));
            }
            Ok(Some(Value::Product(values)))
        }
        Expr::Product(items) => {
            let mut values = Vec::new();
            for (tag, item) in items {
                let Some(value) = evaluate(py, program, environ, item, memo)? else {
                    return Ok(None);
                };
                values.push((tag.clone(), value));
            }
            Ok(Some(Value::Product(values)))
        }
        Expr::Call { head, args } => {
            if head == "__get_args__" {
                return Ok(Some(Value::Product(vec![(
                    Some(ENV_TAG.to_string()),
                    Value::Env(environ.clone().unbind()),
                )])));
            }
            if !program.steps.contains_key(head) {
                return Ok(None);
            }
            let mut values = Vec::new();
            for arg in args {
                let Some(value) = evaluate(py, program, environ, &arg.value, memo)? else {
                    return Ok(None);
                };
                values.push(value);
            }
            let get_func = environ.getattr("compute")?.getattr("get_func")?;
            let host = match get_func.call1((program.module_name.as_str(), head.as_str())) {
                Ok(host) => host,
                Err(_) => return Ok(None),
            };
            if host.is_none() {
                return Ok(None);
            }
            let mut call_args: Vec<Py<PyAny>> = Vec::new();
            for value in values {
                match value {
                    Value::Env(env) => call_args.push(env),
                    Value::Product(_) => return Ok(None),
                    value => call_args.push(value_object(py, &value).map_err(|error| error)?),
                }
            }
            let tuple = pyo3::types::PyTuple::new(py, call_args)?;
            let answer = match host.call1(tuple) {
                Ok(answer) => answer,
                Err(_) => return Ok(None),
            };
            let value = value_of_python(py, &answer)?;
            Ok(value)
        }
    }
}

fn read_source(path: &str) -> Option<String> {
    std::fs::read_to_string(path).ok()
}

fn module_name_of(path: &str) -> String {
    std::path::Path::new(path)
        .file_stem()
        .map(|stem| stem.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.to_string())
}

pub(crate) fn try_interpret(
    py: Python<'_>,
    viba_main_file: &Bound<'_, PyAny>,
    environ: &Bound<'_, PyAny>,
    get_file: &Bound<'_, PyAny>,
    list_files: &Bound<'_, PyAny>,
) -> PyResult<Option<Py<PyAny>>> {
    if !get_file.is_none() || !list_files.is_none() {
        return Ok(None);
    }
    if pyast::class_name(environ)? != "Environment" {
        return Ok(None);
    }
    if let Ok(viba_path) = environ.getattr("viba_path") {
        if !viba_path.is_none() {
            return Ok(None);
        }
    }
    let compute = environ.getattr("compute").ok();
    let Some(compute) = compute else {
        return Ok(None);
    };
    if compute.is_none()
        || !compute
            .getattr("get_func")
            .map(|f| f.is_callable())
            .unwrap_or(false)
    {
        return Ok(None);
    }
    let path: String = match viba_main_file.extract() {
        Ok(path) => path,
        Err(_) => return Ok(None),
    };
    let Some(source) = read_source(&path) else {
        return Ok(None);
    };
    let (key, stamp) = cache_key(&path, &source);
    let cached = {
        let cache = PROGRAM_CACHE.lock().ok();
        cache
            .as_ref()
            .and_then(|cache| cache.as_ref())
            .and_then(|cache| cache.get(&key))
            .filter(|(seen, _)| *seen == stamp)
            .map(|(_, program)| Arc::clone(program))
    };
    let program = match cached {
        Some(program) => program,
        None => {
            let tree = match crate::parser::parse(py, &source) {
                Ok(tree) => tree,
                Err(_) => return Ok(None),
            };
            let module_name = module_name_of(&path);
            let program = match compile(py, tree.bind(py), &module_name)? {
                Ok(program) => program,
                Err(_) => return Ok(None),
            };
            let program = Arc::new(program);
            if let Ok(mut cache) = PROGRAM_CACHE.lock() {
                cache
                    .get_or_insert_with(HashMap::new)
                    .insert(key, (stamp, Arc::clone(&program)));
            }
            program
        }
    };
    let mut memo = HashMap::new();
    let Some(value) = evaluate(py, &program, environ, &program.implementation, &mut memo)? else {
        return Ok(None);
    };
    if matches!(value, Value::Env(_)) {
        return Ok(None);
    }
    let payload = value_object(py, &value)?;
    let type_module = PyModule::import(py, "rs_viba.type")?;
    let answer = type_module.getattr("Ok")?.call1((payload,))?;
    let finished = crate::interpret::finished(py, &answer)?;
    Ok(Some(finished))
}
