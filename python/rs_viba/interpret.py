"""viba.interpret — run a viba module (Rust implementation).

The classes here are the runtime's data: the environment the host hands over,
the values a run threads, and the records a run keeps. What each of them does
lives in Rust (`src/interpret.rs`); the classes below are the Python-visible
spelling of that data, and the functions are the members the design hangs off
an environment (`args.env.sub_env`, `$try_compact << args.env << 32`).

`interpret` and `exec` answer the declared `InterpretResult` as one node of
viba data: `$ok` carries the `__impl__` that came out, `$err` why the run
stopped. A host function is called with the arguments already evaluated and
answers a `VObject`, a scalar, or `None`.
"""

import contextvars
import os
import tempfile
from pathlib import Path

from rs_viba import markdown, viba_ast
from rs_viba import _native
from rs_viba.type import (  # noqa: F401  (the names this module has always carried)
    BUILTIN_CONCEPT_DIR, BUILTIN_DIR, BUILTIN_MODULE, ENVIRONMENT_API_TAG,
    ENVIRONMENT_TYPE, ENV_TYPE, ENVIRON_TAG, Err, EnvironmentApiInvalidArgumentErr,
    ERR_TAG, FAILURE_TAG, Frame, InterpretError, ModuleType, NOT_IMPLEMENTED_TAG,
    NeverType, NilType, OK_TAG, Ok, PROGRAM_ERR_TAG, PartialError,
    REASON_GET_FUNC_RAISED, REASON_NO_IMPLEMENTATION, REASON_NO_LEAF, REASON_RAISED,
    Stack, UnderlyingOpErr, VibaProgramErr, AstNodeType, CustomModuleType,
    builtin_directory_name, custom_module, module_get_type,
    takes_a_product as _takes_a_product,
)

# A scalar a host answers belongs to no file: its leaf gets an empty module.
_NO_MODULE = custom_module("")

TMP_PREFIX = "tmp_"
RET_NAME = "__impl__"
DEF_NAME = "__decl__"
GET_ARGS_NAME = "__get_args__"
IS_SUB_TYPE_VALUE_NAME = "__is_sub_type_value__"
ENVIRON_TAG = "$env"
ENVIRON_TYPE = "Environment"

# A call whose name travels as data: the interpreter answers both names.
DYN_CALL_NAME = "__dyn_call__"
DYN_METHOD_NAME = "__dyn_method__"

CODE_LABEL = "<viba_code>"
ENV_TYPE = "Env"

DEFAULT_STORE_ROOT = os.path.join(tempfile.gettempdir(), "viba-store")
SNAPSHOT_NAME = "value"
SNAPSHOT_SUFFIX = ".viba"

RESULT_MODULE_NAME = "interpret_result"
RESULT_TYPE_NAME = "InterpretResult"

# The run the interpreter is in; `try_compact` takes the run from here.
_CURRENT_RUN = contextvars.ContextVar("viba_run", default=None)


# ----------------------------------------------------------------------
# The host side of an environment
# ----------------------------------------------------------------------


class EnvironmentStorage:
    """Where a module's files, sub-modules and snapshots live."""

    __slots__ = ("cur_storage_path", "sub_storage", "store_root_dir", "parent")

    def __init__(self, cur_storage_path: str, sub_storage=None,
                 store_root_dir=None, parent=None):
        self.cur_storage_path = cur_storage_path
        self.sub_storage = dict(sub_storage or {})
        self.store_root_dir = store_root_dir or DEFAULT_STORE_ROOT
        self.parent = parent

    def sub(self, name: str) -> "EnvironmentStorage":
        """The child storage for `name`: `<cur>/<name>`, made on demand."""
        return _native.storage_sub_py(self, name)

    def tmp(self) -> "EnvironmentStorage":
        """A child under a name nobody chose, fresh on every call."""
        return _native.storage_tmp_py(self)

    def get_text(self, file_path: str):
        """The text stored at `file_path`, or None when nothing is there."""
        return _native.storage_get_text_py(self, file_path)

    def put_text(self, file_path: str, content: str) -> None:
        """Store `content` at `file_path`, making the directories on the way."""
        _native.storage_put_text_py(self, file_path, content)

    def _store_path(self, file_path: str, root_dir=None):
        return _native.storage_store_path_py(self, file_path, root_dir)


class EnvironmentCompute:
    """The implementations: `get_func(module_path, func_name) -> callable|None`."""

    __slots__ = ("get_func",)

    def __init__(self, get_func):
        self.get_func = get_func


class Environment:
    """Storage, compute and the module search path, and the children under it."""

    __slots__ = ("storage", "compute", "viba_path", "parent",
                 "uncompress_relative_path",
                 "sub_env", "tmp_env", "get_root", "get_parent",
                 "get_relative_path", "find_by_relative_path", "compress_env_path",
                 "try_compact", "make_referenced_env", "record_reenact_call",
                 "reenact_reference", "store_reenact_reference", "reenact_calls")

    def __init__(self, storage, compute, viba_path=None, parent=None,
                 uncompress_relative_path=None):
        self.storage = storage
        self.compute = compute
        self.parent = parent
        self.viba_path = viba_path
        self.uncompress_relative_path = uncompress_relative_path
        self.sub_env = sub_env
        self.tmp_env = tmp_env
        self.get_root = get_root
        self.get_parent = get_parent
        self.get_relative_path = get_relative_path
        self.find_by_relative_path = _BoundFindByRelativePath(self)
        self.compress_env_path = compress_env_path
        self.try_compact = try_compact
        self.make_referenced_env = make_referenced_env
        self.record_reenact_call = record_reenact_call
        self.reenact_reference = reenact_reference
        self.store_reenact_reference = store_reenact_reference
        self.reenact_calls = reenact_calls


class ReferencedEnvironment(Environment):
    """The replay side of a run: an environment that also points at the recording."""

    __slots__ = ("reference_store_root_dir",)

    def __init__(self, storage, compute, viba_path=None, parent=None,
                 uncompress_relative_path=None, reference_store_root_dir=None):
        super().__init__(storage, compute, viba_path, parent,
                         uncompress_relative_path)
        self.reference_store_root_dir = reference_store_root_dir


class _BoundFindByRelativePath:
    """`args.env.find_by_relative_path`: the member, with its own environment."""

    __slots__ = ("environ",)

    def __init__(self, environ):
        self.environ = environ

    def __call__(self, relative_path, root):
        return _native.find_by_relative_path_member_py(self.environ, relative_path,
                                                       root)


# ----------------------------------------------------------------------
# The values a run threads, and the records it keeps
# ----------------------------------------------------------------------


class _VibaData:
    """A piece of viba data with its design, the module it came from, and what
    a decision bound for it."""

    __slots__ = ("node", "source_module", "given", "bindings", "members")

    def __init__(self, node, source_module=None, given=None, bindings=None,
                 members=None):
        self.node = node
        self.source_module = (source_module if source_module is not None
                              else _native.source_module_of_py(node))
        self.given = given
        self.bindings = bindings
        self.members = members


class _Host:
    """Anything the host handed over: the environment, a host value."""

    __slots__ = ("obj",)

    def __init__(self, obj):
        self.obj = obj


class _Member:
    """`$tag` at the head of a chain: the member `$tag` of the value given first."""

    __slots__ = ("tag",)

    def __init__(self, tag):
        self.tag = tag


class _Given:
    """An argument on its way to a call: its tag and its value."""

    __slots__ = ("tag", "value")

    def __init__(self, tag, value):
        self.tag = tag
        self.value = value


class _Raised(Exception):
    """A `Result` a getter has to cross a host call with."""

    def __init__(self, result):
        super().__init__(repr(result))
        self.result = result


class _Getter:
    """One argument in the source of a lazy call, computed when the host wants it."""

    __slots__ = ("activation", "node", "scope", "answer",
                 "file", "source_module", "name", "origin")

    def __init__(self, activation, node, scope=()):
        self.activation = activation
        self.node = node
        self.scope = scope
        self.answer = None
        self.file = activation.file
        self.source_module = activation.module
        self.origin = activation
        self.name = activation.name

    def __call__(self):
        return _native.getter_call_py(self)


class _Activation:
    """One module running with one environment."""

    __slots__ = ("runner", "module", "environ", "name", "file", "args", "path",
                 "signature", "members", "bindings", "defined")

    def __init__(self, runner, module, environ, name, file, args=None,
                 members=None, bindings=None):
        self.runner = runner
        self.module = module
        self.environ = environ
        self.name = name
        self.file = file
        if args is None:
            args = _VibaData(_native.viba_data_py(None))
        self.args = args
        self.path = _native.storage_path_of_py(environ)
        self.signature = _native.call_signature_py(members)
        if members is None:
            members = [(ENVIRON_TAG, _Host(environ))]
        self.members = members
        self.bindings = bindings or {}
        self.defined = {}


class _Pending:
    """A call not yet given an environment: a function or a module, and what it
    has been given."""

    __slots__ = ("kind", "activation", "head", "source", "name", "module",
                 "source_module", "elements", "runner", "module_name", "environ",
                 "given", "given_tags", "bindings", "pattern_env",
                 "params_override", "source_bindings", "appends_environ",
                 "member_of", "member_tag")

    def __init__(self, kind, activation=None, head=None, source="", name="",
                 module=None, source_module=None, elements=(), runner=None,
                 module_name=None, bindings=(), pattern_env=None, slots=None,
                 source_bindings=None):
        self.kind = kind
        self.activation = activation
        self.head = head
        self.source = source
        self.name = name or source
        self.module = module
        self.source_module = source_module or module
        self.elements = list(elements)
        self.runner = runner
        self.module_name = module_name
        self.environ = None
        self.given = {}
        self.given_tags = {}
        self.bindings = bindings
        self.pattern_env = pattern_env
        self.params_override = slots
        self.source_bindings = source_bindings
        self.appends_environ = False
        self.member_of = None
        self.member_tag = None

    def as_viba_data(self):
        return _native.pending_as_viba_data_py(self)


class _HostArgument:
    """An argument in the source of a function-typed slot, while the host holds it."""

    __slots__ = ("node", "scope", "pending", "module", "owner", "file",
                 "source_module", "origin", "name", "answer", "environ", "closure")

    def __init__(self, node, scope, pending, module, owner, file=None,
                 source_module=None, name="", origin=None):
        self.node = node
        self.scope = scope
        self.pending = pending
        self.module = module
        self.owner = owner
        self.file = file
        self.source_module = source_module
        self.origin = origin
        self.name = name
        self.answer = None
        self.environ = None
        self.closure = False

    def closure_of(self):
        return _native.host_argument_closure_py(self)

    def __call__(self, environ, *arguments):
        return _native.host_argument_call_py(self, environ, arguments)


class _GetArgs:
    """`__get_args__ << __decl__`: the arguments this call was handed."""

    __slots__ = ("activation",)

    def __init__(self, activation):
        self.activation = activation


class _IsSubType:
    """`__is_sub_type_value__ << sub << sup`: the judgment, with no environment."""

    __slots__ = ("given",)

    def __init__(self, given=None):
        self.given = list(given or [])


class _HostFunction:
    """A function the host hangs off the environment: `args.env.sub_env`."""

    __slots__ = ("name", "func", "slots", "given", "pieces")

    def __init__(self, name, func, slots=1, given=None, pieces=None):
        self.name = name
        self.func = func
        self.slots = slots
        self.given = list(given or [])
        self.pieces = list(pieces or [])

    def as_callable(self):
        def call(*arguments):
            return _native.host_function_call_py(self, arguments)
        return call


class _DynCall:
    """A call whose name is data: `__dyn_call__` and `__dyn_method__`."""

    __slots__ = ("activation", "kind", "head", "source", "environ", "name",
                 "value", "given", "given_tags")

    def __init__(self, activation, kind, head, source):
        self.activation = activation
        self.kind = kind
        self.head = head
        self.source = source
        self.environ = None
        self.name = None
        self.value = None
        self.given = []
        self.given_tags = []

    def as_viba_data(self):
        return _native.dyn_call_as_viba_data_py(self)


class _Runner:
    """One run: the files it has loaded, and where it looks for more."""

    __slots__ = ("paths", "get_file", "list_files", "by_path", "documents",
                 "by_name", "path_of", "running", "done", "computing",
                 "running_calls", "global_resumable_stack", "held_paths",
                 "compacted_chain_length", "viba_path")

    def __init__(self, viba_path=None, get_file=None, list_files=None):
        self.viba_path = viba_path
        self.paths = [Path(part) for part in
                      ("" if viba_path is None else os.fspath(viba_path)).split(":")
                      if part]
        self.get_file = get_file
        self.list_files = list_files
        self.by_path = {}
        self.documents = {}
        self.by_name = {}
        self.path_of = {}
        self.running = []
        self.done = {}
        self.computing = []
        self.running_calls = []
        self.global_resumable_stack = []
        self.held_paths = {}
        self.compacted_chain_length = 0

    # ---- the loader other layers use (json-schema, extract-sub-repo) ----

    def imported(self, name, near=None):
        """The module `name`: loaded, or found next to `near`, or on the path."""
        return _native.runner_imported_py(self, name, near)

    def load(self, name, near=None):
        """The module `name`, the way an import finds it."""
        return self.imported(name, near)

    def module(self, name, near=None):
        return self.imported(name, near)

    def builtin_module(self, name):
        return _native.runner_builtin_module_py(self, name)

    def run_file(self, file, environ):
        return _native.runner_run_file_py(self, file, environ)

    def run_code(self, viba_code, environ):
        return _native.runner_run_code_py(self, viba_code, environ)

    def run_node(self, node, environ):
        return _native.runner_run_node_py(self, node, environ)


class _Compact(BaseException):
    """Stop here: the calls running now have to be made from the work list."""

    __slots__ = ("tasks", "environ")

    def __init__(self, tasks, environ):
        super().__init__("the chain of environments got too long")
        self.tasks = tasks
        self.environ = environ


class _ResumableCall:
    """One module call of a run, in the form that can make it again."""

    __slots__ = ("runner", "module", "environ", "name", "file", "args",
                 "members", "bindings", "passes_environ", "give_environ", "at")

    def __init__(self, runner, module, environ, name, file, args, members,
                 bindings, passes_environ, give_environ, at):
        self.runner = runner
        self.module = module
        self.environ = environ
        self.name = name
        self.file = file
        self.args = args
        self.members = members
        self.bindings = bindings
        self.passes_environ = passes_environ
        self.give_environ = give_environ
        self.at = at

    @property
    def current_env_path(self):
        return _native.resumable_current_env_path_py(self)

    def restored(self, root):
        return _native.resumable_restored_py(self, root)

    def run(self, environ):
        return _native.resumable_run_py(self, environ)


# ----------------------------------------------------------------------
# The environment's members, as plain functions of the environment
# ----------------------------------------------------------------------


def sub_env(environ, name):
    """A child environment: its own storage, the parent's compute."""
    return _native.sub_env_py(environ, name)


def tmp_env(environ):
    """A child environment under a name of its own, fresh every time."""
    return _native.tmp_env_py(environ)


def make_referenced_env(environ, reference_store_root_dir):
    """The same environment, with the store a previous recording wrote as its reference."""
    return _native.make_referenced_env_py(environ, reference_store_root_dir)


def get_root(current):
    """The root of the chain this environment belongs to."""
    return _native.get_root_py(current)


def get_parent(environ):
    """The environment this one was made from, or None for a chain's root."""
    return _native.get_parent_py(environ)


def env_chain_length(environ):
    """How many environments the chain from this one up to its root has."""
    return _native.env_chain_length_py(environ)


def get_relative_path(current, root):
    """`current`'s storage path seen from `root` down: `""` for the root itself."""
    return _native.get_relative_path_py(current, root)


def find_by_relative_path(relative_path, root):
    """The environment `root` reaches by that relative path."""
    return _native.find_by_relative_path_py(relative_path, root)


def compress_env_path(sup, sub):
    """The storage path `sub` stands for, compressed into a sibling of `sup`."""
    return _native.compress_env_path_py(sup, sub)


def resumable_stack(runner, environ):
    """The calls running now, outermost first, as `_ResumableCall`s."""
    return _native.resumable_stack_py(runner, environ)


def try_compact(environ, env_chain_length_limit):
    """Compact the run where the chain of environments is longer than the limit."""
    return _native.try_compact_py(environ, env_chain_length_limit)


# ----------------------------------------------------------------------
# Snapshots: what makes a run replayable
# ----------------------------------------------------------------------


def snapshot_path(environ, name=SNAPSHOT_NAME):
    """Where this call's snapshot lives: `<cur_storage_path>/<name>.viba`."""
    return _native.snapshot_path_py(environ, name)


def get_snapshot(environ, name=SNAPSHOT_NAME):
    """The value stored for this call, or None when nothing is stored yet."""
    return _native.get_snapshot_py(environ, name)


def get_snapshot_from(environ, name, root):
    """The snapshot `name` stored under `root`, at this environment's data path."""
    return _native.get_snapshot_from_py(environ, name, root)


def put_snapshot(environ, value, name=SNAPSHOT_NAME):
    """Store `value` as this call's snapshot."""
    return _native.put_snapshot_py(environ, value, name)


def replayed(environ, compute, name=SNAPSHOT_NAME):
    """The snapshot of this call, or `compute()` — and then the snapshot."""
    return _native.replayed_py(environ, compute, name)


# ----------------------------------------------------------------------
# Recording and replay
# ----------------------------------------------------------------------


def record_reenact_call(environ, tag, f, args):
    """Record this call of an entry; answer the name its invocation runs under."""
    return _native.record_reenact_call_py(environ, tag, f, args)


def reenact_reference(environ, tag):
    """Whether a reference is there, and what it is: `$found bool * $value Any`."""
    return _native.reenact_reference_py(environ, tag)


def store_reenact_reference(environ, tag, value):
    """Store a reference's answer; a replay hands back what it read, unwritten."""
    return _native.store_reenact_reference_py(environ, tag, value)


def reenact_calls(environ, tag, entry):
    """Run every call recorded under `tag` again, and answer their answers."""
    return _native.reenact_calls_py(environ, tag, entry)


def _run_entry(entry, environ, arguments):
    """Run the entry a call was recorded through, at `environ`, with `arguments`."""
    return _native.run_entry_py(entry, environ, arguments)


# ----------------------------------------------------------------------
# What a run answers, as viba data
# ----------------------------------------------------------------------


def viba_data(value):
    """A host value as viba data."""
    return _native.viba_data_py(value)


def not_implemented(msg="", module_path="", full_qualified_func_name="", call=None):
    """What a step answers with when it has no implementation: the failure data."""
    return _native.not_implemented_py(msg, module_path, full_qualified_func_name,
                                       call)


def is_interpret_result(value):
    """Is this piece of viba data a resident of the declared `InterpretResult`?"""
    return _native.is_interpret_result_py(value)


def interpret(viba_main_file, environ, get_file=None, list_files=None):
    """Run `viba_main_file` with `environ`; its `__impl__` is the answer."""
    return _native.interpret_py(viba_main_file, environ, get_file, list_files)


def exec(viba_code, environ, get_file=None, list_files=None):
    """Run the module `viba_code` gives with `environ`."""
    return _native.exec_py(viba_code, environ, get_file, list_files)


# ----------------------------------------------------------------------
# The runtime's own indirections (patched by the suites)
# ----------------------------------------------------------------------


def _run_module(runner, module, environ, name, file, args=None, members=None,
                bindings=(), passes_environ=False, at=0, give_environ=False,
                driving=False):
    """The module as a function: the environment in, `__impl__` out."""
    return _native.run_module_py(runner, module, environ, name, file, args,
                                 members, bindings, passes_environ, at,
                                 give_environ, driving)


def _hold(runner, tasks):
    """Put a chain of calls back on the work list, outermost first, once each."""
    return _native.hold_py(runner, tasks)


__all__ = ["interpret", "exec", "Environment", "EnvironmentStorage",
           "EnvironmentCompute", "ReferencedEnvironment", "viba_data",
           "snapshot_path", "get_snapshot", "put_snapshot", "replayed"]
