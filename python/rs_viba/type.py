"""The Type model consumed by is_sub_type.

This is the metalanguage layer: Viba types being compared are reified
as values of the `Type` classes below. Structure stays at the viba.viba_ast
layer (wrapped in AstNodeType); this module models leaves, references
and the module machinery they need for lexical resolution.

Decoupling contract: this layer knows how to build Type values from
viba.viba_ast nodes, and nothing about how a module runs.
"""

import os
from pathlib import Path
from typing import Callable, List, Optional, Union

from rs_viba import viba_ast
from rs_viba._native import Ok, Err, Frame, InterpretError, VibaProgramErr, UnderlyingOpErr, EnvironmentApiInvalidArgumentErr
from rs_viba._native import (
    Type, NeverType, AnyType, NilType, BoolType, IntType, FloatType, StrType,
    FixedIntType, Int8Type, UInt8Type, Int16Type, UInt16Type, Int32Type,
    UInt32Type, Int64Type, UInt64Type, BoolLiteralType, IntLiteralType,
    FloatLiteralType, StrLiteralType, BuiltinGenericType, AstNodeType,
    ModuleType, BuiltinModuleType, CustomModuleType,
)



# ----------------------------------------------------------------------
# Result (cf. Result[T] = Oneof | $ok T | $err Err, where Err carries one sentence)
# ----------------------------------------------------------------------

Result = Union[Ok, Err]


Stack = List[Frame]


# The two tags a `Failure` answers under: one payload, told apart by the tag.
FAILURE_TAG = "$underlying_viba_op_err"   # the implementation, or `get_func`, broke
NOT_IMPLEMENTED_TAG = "$not_implemented_err"     # nothing implements that step

# The two branches a run answers under, and the tags the other two stops carry
# (`InterpretResult` below). A host that hands the failure back gives the same
# tag it would be answered under.
OK_TAG = "$ok"                                   # the answer
ERR_TAG = "$err"                                 # why it stopped
PROGRAM_ERR_TAG = "$viba_program_err"            # the program, or the environment
ENVIRONMENT_API_TAG = "$environment_api_invalid_argument_err"   # one api of it


# The words a failure's `msg` opens with: short, stable, the same ones whoever
# looks at the message switches on.
ENVIRON_TAG = "$env"
ENVIRONMENT_TYPE = "Environment"
ENV_TYPE = "Env"

REASON_NO_IMPLEMENTATION = "no implementation"   # get_func answered None
REASON_GET_FUNC_RAISED = "get_func raised"       # get_func broke
REASON_RAISED = "raised"                         # the implementation broke
REASON_NO_LEAF = "no leaf"                       # it answered something with no leaf


# The declared result, its branches serialized:
#
#     InterpretResult =
#         Oneof
#       | $ok Any
#       | $err InterpretError
#
# The error side is `InterpretError` (`$viba_program_err`, an
# `EnvironmentApiInvalidArgumentErr` under `$environment_api_invalid_argument_err`,
# or a `Failure` under one of its two tags).
#
# `interpret` and `exec` answer that declaration as **one node of viba data**:
# `$ok (<the answer>)` or `$err (<why it stopped>)`, with the members of each
# stop addressable by tag (`viba/interpret.py`). So the answer serializes as it
# stands, and nothing on this side has to be taken apart to carry it anywhere.


# ----------------------------------------------------------------------
# Type leaves
# ----------------------------------------------------------------------


# The concept in the builtin library that holds the builtin operators: its
# members are spelled `builtin.add`, and count as `add` on their own.
BUILTIN_CONCEPT = "builtin"


def takes_a_product(spelled) -> bool:
    from rs_viba import _native
    return _native.takes_a_product_py(spelled)


def builtin_directory_name(name: str) -> Optional[str]:
    """The bare name a name in the source stands for in the builtin directory, or None.

    The built-in vocabulary is the package's own: beside `builtin.viba` sit the
    modules (`Y.viba`, `apply.viba`, `sub_env_run.viba`), and under `builtin/`
    the generics (`builtin/is_closure/`, `builtin/unclosure/`). Those names are
    taken from every module, after its own definitions and its imports.
    `sub_env_run` and `builtin.sub_env_run` are the same one, and so are
    `is_closure` and `builtin.is_closure`; a dotted name that carries no
    `builtin.` prefix (`demo.print`) is no builtin name, and neither is `builtin`
    itself. Whether the directory really holds that file is the loader's to say.
    """
    rest = name[len(BUILTIN_CONCEPT) + 1:] if name.startswith(BUILTIN_CONCEPT + ".") else name
    if not rest or "." in rest or rest == BUILTIN_CONCEPT:
        return None
    return rest


def _is_definition(node) -> bool:
    return isinstance(node, (viba_ast.TypeDefinition, viba_ast.GenericDefinition))


# ----------------------------------------------------------------------
# AstNodeType — the escape hatch for the structure layer
# ----------------------------------------------------------------------



# The concept in the builtin library that holds the builtin operators: its
# members are spelled `builtin.add`, and count as `add` on their own.
BUILTIN_CONCEPT = "builtin"


def takes_a_product(spelled) -> bool:
    from rs_viba import _native
    return _native.takes_a_product_py(spelled)


def builtin_directory_name(name: str) -> Optional[str]:
    """The bare name a name in the source stands for in the builtin directory, or None."""
    rest = name[len(BUILTIN_CONCEPT) + 1:] if name.startswith(BUILTIN_CONCEPT + ".") else name
    if not rest or "." in rest or rest == BUILTIN_CONCEPT:
        return None
    return rest


# The single shared built-in module instance.
BUILTIN_MODULE = BuiltinModuleType()

# The directory the built-in vocabulary lives in.
BUILTIN_DIR = Path(__file__).resolve().parent
BUILTIN_CONCEPT_DIR = BUILTIN_DIR / BUILTIN_CONCEPT
VIBA_SUFFIX = ".viba"


class UnresolvedTypeError(Exception):
    """A TypeRef could not be resolved to a Type."""


class DuplicateTagError(Exception):
    """One product gives the same tag twice, inlined members counted."""


class PartialError(Exception):
    """A `<<` that cannot be given: the left side is no function, or it has no
    such argument."""


class InlineCycleError(Exception):
    """An inline chain comes back to a definition it is already expanding."""


def module_get_type(module: ModuleType, type_name: str, _seen=None) -> Result:
    from rs_viba import _native
    return _native.module_get_type_py(module, type_name, _seen)


# ----------------------------------------------------------------------
# Convenience constructors (test-oriented; not tied to how a module runs)
# ----------------------------------------------------------------------


def custom_module(source: str, environment: Callable[[str], Result] = None) -> CustomModuleType:
    from rs_viba import _native
    return _native.custom_module_py(source, environment)


def entry_type(source: str, module: ModuleType = None) -> AstNodeType:
    from rs_viba import _native
    return _native.entry_type_py(source, module)
