"""viba.reflect — treat a design as a map and take a viba data through it.

This is the access side of ``viba/core/data/viba-reflect.md``: the map (the descriptors) comes
from the descriptor side walking the design, the data (Data) is one viba data's
type expression, and access walks the map's addresses one step at a time. Names
follow the rule ``viba_type_descriptor.py`` keeps for
``viba_type_descriptor.viba``: the protocol's class names stay as they are, its
functions land as methods in snake_case, this layer's own helpers take an
underscore.

    protocol (viba/core/data/viba-reflect.md)         Python
    ------------------------------    ------------------------------------
    VibaStep                          VibaStep
    the four VibaStep branches         by_tag / by_field_index / at_index / at_key
    VibaPath                          VibaPath (= list[VibaStep])
    VObject[Data]                     VObject
    VibaReflectConfig                 Config
    Result[T]                         Ok / Err (viba.type)
    VibaConstant                      the naked value (bool / int / float / str / None)
    VibaAccess[Data]                  VibaAccess, built with a Config
    VibaRoot / VibaHas / VibaGet      VibaAccess.root / has / get / leaf /
    VibaLeaf / VibaLength / VibaKeys  length / keys
    VibaResolve / VibaGetByPath       VibaAccess.resolve / get_by_path /
    VibaListFields                    list_fields

What the protocol does not name is this layer's binding or helping, not a protocol concept:

    the Data parameter    -> VibaData: one viba data's type expression
    section 5.4 "throw"   -> VibaReflectError
    walking the map       -> VibaAccess's underscore methods and this module's

Two rules from the protocol:

* ``root`` only pairs (descriptor, data). Versions are none of its business:
  which revision a viba data was built against is the caller's bookkeeping.
* ``has`` answers true or false only: no such address in the design and no
  such piece in the viba data are both ``False``. In ``get``, "the viba data has
  no such piece" is ``Ok(nil)`` and only "the map has no such address" is
  ``Err``; ``leaf`` / ``length`` / ``keys`` give ``Err`` when they cannot.

The config is the one thing an implementation is told and this module is not:
which names in the source stand for the units ``nil`` and ``never``. Every accessor
carries one, and this module's own ``access`` carries the language layer's
default.

The section 5.4 Python landing lives on `VObject`: ``get_{name}()`` / ``has_{name}()`` /
``in`` / ``try_get_{name}()`` / ``node[i]`` / ``node.value`` / ``len(node)`` /
iteration / ``keys()`` / ``values()`` / ``items()``. The taking ones throw; the asking
ones never throw.

The data classes below are the protocol's, copied as they are; the accessor and
every walk it takes live in Rust (``src/reflect.rs``), behind
``rs_viba._native.VibaAccess``.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from rs_viba.type import AstNodeType, Ok

from rs_viba._native import (
    VibaAccess, VObject, VibaData, Config, VibaStep, VibaFunction,
    reflect_dir_py as _dir_names,
    reflect_key_text_py as _key_text,
    reflect_scalar_name_py as _scalar_name,
    reflect_tag_of_py as _tag_of,
    reflect_type_name_py as _type_name,
)

# The builtin leaves a sum can name: the scalars, and the fixed-width
# integers, whose value is a host int. Then the containers, and the literal
# forms implementations use.
FIXED_INT_NAMES = ("int8_t", "uint8_t", "int16_t", "uint16_t",
                   "int32_t", "uint32_t", "int64_t", "uint64_t")
SCALAR_NAMES = ("bool", "int", "float", "str") + FIXED_INT_NAMES
CONTAINERS = ("list", "set", "dict")
LITERAL_CTORS = ("ListLiteral", "SetLiteral", "DictLiteral")
# A literal spells the container it is a resident of (`viba/type.viba`:
# `ListLiteral[a, b, c]` is a resident of `list[a | b | c]`), so a piece whose data
# has that source form is taken the way any other piece of that container is.
LITERAL_KINDS = {"ListLiteral": "list", "SetLiteral": "set", "DictLiteral": "dict"}


def _is_literal(descriptor) -> bool:
    """Whether this design piece is one of the three literal constructors."""
    return (descriptor.kind == "type_app"
            and descriptor.payload.constructor_name in LITERAL_KINDS)


class VibaReflectError(Exception):
    """Thrown when the taking path fails; it carries the protocol's ``Err`` text.

    Section 5.4 says "throw" without naming the exception; the name is this
    layer's.
    """


# ----------------------------------------------------------------------
# Binding the config (VibaReflectConfig)
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# VibaStep and VibaPath
# ----------------------------------------------------------------------


def by_tag(name: str) -> VibaStep:
    return VibaStep("by_tag", _tag_of(name))


def by_field_index(index: int) -> VibaStep:
    return VibaStep("by_field_index", index)


def at_index(index: int) -> VibaStep:
    return VibaStep("at_index", index)


def at_key(key: str) -> VibaStep:
    return VibaStep("at_key", key)


VibaPath = List[VibaStep]


# ----------------------------------------------------------------------
# Binding Data
# ----------------------------------------------------------------------


# This layer's default accessor
# ----------------------------------------------------------------------

# The language layer's units: what viba/type.viba and the descriptor's own
# docs give as the source for the sum unit and the product unit. The config is a
# value a caller can hand around; the accessor built from it is not.
language_config = Config(never_eqv={"Oneof"},
                         nil_eqv={"Object", "Assert", "Appendix", "Hint"})

access = VibaAccess(language_config)


# The protocol names (viba/core/data/viba-reflect.md sections 4 and 5), and only these.
__all__ = [
    "access", "Config", "language_config",
    "VibaAccess", "VObject", "VibaStep", "VibaPath",
    "by_tag", "by_field_index", "at_index", "at_key",
]

# Bound by this layer, imported by name, not protocol concepts:
#   VibaData         what the Data parameter binds to here
#   VibaReflectError the section 5.4 "taking path throws" exception
