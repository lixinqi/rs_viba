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
from rs_viba._native import (
    EnvironmentStorage, EnvironmentCompute,
    _HostArgument, _Getter, _Runner, _ResumableCall, _Raised, _Compact,
    _VibaData, _Host, _Member, _Given, _GetArgs, _IsSubType, _HostFunction, _DynCall, _Activation, _Pending,
)
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
