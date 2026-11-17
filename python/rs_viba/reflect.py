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
    VibaAccess,
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


class Config:
    """Which names in the source stand for the units ``never`` and ``nil``.

    ``never_eqv`` holds the names equivalent to ``never``, the sum's unit, and
    ``nil_eqv`` the names equivalent to ``nil``, the product's unit. The units
    themselves are units by kind, so naming them again changes nothing.
    Whoever builds an accessor fills these in.

    Either a name or a descriptor/type node may be given; only its name in the
    source is kept.
    """

    def __init__(self, never_eqv=(), nil_eqv=()):
        self.never_eqv = frozenset(_type_name(name) for name in never_eqv)
        self.nil_eqv = frozenset(_type_name(name) for name in nil_eqv)

    def __repr__(self):
        return f"Config(never_eqv={sorted(self.never_eqv)!r}, nil_eqv={sorted(self.nil_eqv)!r})"

    def is_unit_name(self, name: str) -> bool:
        return name in self.never_eqv or name in self.nil_eqv


# ----------------------------------------------------------------------
# VibaStep and VibaPath
# ----------------------------------------------------------------------


class VibaStep:
    """One step of an address: one of the four branches."""

    __slots__ = ("kind", "value")

    def __init__(self, kind: str, value):
        self.kind = kind
        self.value = value

    def __repr__(self):
        return f"{self.kind}({self.value!r})"

    def __eq__(self, other):
        return isinstance(other, VibaStep) and (self.kind, self.value) == (other.kind, other.value)

    def __hash__(self):
        return hash((self.kind, self.value))


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


class VibaData:
    """This layer's binding of the protocol's Data parameter: one viba data's
    type expression.

    The protocol leaves Data to the implementation; on this side a viba data has
    the design's own language as its source form, so it lands as an AST node.
    Versions and the like are carried by the caller, not by the protocol.
    """

    __slots__ = ("node", "module")

    def __init__(self, node, module=None):
        if isinstance(node, AstNodeType):
            module = module or node.container_module
            node = node.ast_node
        self.node = node
        self.module = module


# ----------------------------------------------------------------------
# VObject[Data]
# ----------------------------------------------------------------------


class VibaFunction:
    """A function or a closure as a value: its name, and the module that defines it.

    A function value is not built, it is *defined*: the piece that stands for it is
    a name in some file, and a module that takes that value has to run the function
    the file defines, not a name of its own that happens to be spelled the same.
    So the value carries where it is defined:

    * ``name`` — the name the module defines it under;
    * ``module`` — the module itself, and ``module_name`` the name it was loaded as
      (``demo.show``);
    * ``relative_path`` — the path that module is known by (``/demo/show``);
    * ``absolute_path`` — the file that module was taken from, when it has one.

    Only a function or a closure carries this. A constructed piece — a product,
    a leaf, a container — is built rather than defined, so it has no module of
    its own; the pieces it is built from each carry their own.
    """

    __slots__ = ("name", "module")

    def __init__(self, name: str, module):
        self.name = name
        self.module = module

    @property
    def module_name(self) -> str:
        """The name the module was loaded as, or "" when it records none."""
        return getattr(self.module, "name", "")

    @property
    def relative_path(self) -> str:
        """The path the module is known by (`/demo/show`), or "" when it records none."""
        return getattr(self.module, "module_path", "")

    @property
    def absolute_path(self) -> str:
        """The file that module was taken from, or "" for a module with no file."""
        return getattr(self.module, "file_path", "") or ""

    def __repr__(self):
        return f"VibaFunction({self.module_name}.{self.name})"

    def __eq__(self, other):
        return (isinstance(other, VibaFunction) and self.name == other.name
                and self.module is other.module)

    def __hash__(self):
        return hash((self.name, id(self.module)))


class VObject:
    """One object: a piece of the design's descriptor plus a piece of the viba data.

    ``path`` is the VibaPath walked from the root (the protocol's `VObject` has
    no such field; this layer keeps it so an address can be stored, printed and
    compared). ``function`` is the function or closure this object *is*, for a
    value that stands for one, and None for every other value.
    """

    __slots__ = ("_access", "descriptor", "data", "path", "data_module", "function")

    def __init__(self, access: "VibaAccess", descriptor, data,
                 path: Sequence[VibaStep] = (), data_module=None, function=None):
        self._access = access
        self.descriptor = descriptor
        self.data = data
        self.path = tuple(path)
        # The module the viba data's source is in: a name on the data side is
        # resolved there, not where the design piece came from.
        self.data_module = data_module
        # The function or closure this object is, for a value that stands for one:
        # a function is defined somewhere, so it says where, and every other value
        # is built and says None.
        self.function = function

    def __repr__(self):
        where = ".".join(repr(step) for step in self.path) or "root"
        return f"VObject({where})"

    # ---- section 5.4 accessors on a `VObject`: the taking ones throw ----

    def by_tag(self, name: str) -> "VObject":
        return self._access._unwrap(self._access.get(self, by_tag(name)))

    def by_field_index(self, index: int) -> "VObject":
        return self._access._unwrap(self._access.get(self, by_field_index(index)))

    def at_index(self, index: int) -> "VObject":
        return self._access._unwrap(self._access.get(self, at_index(index)))

    def at_key(self, key: str) -> "VObject":
        return self._access._unwrap(self._access.get(self, at_key(key)))

    @property
    def leaf(self):
        """The node's $leaf cell: the literal this piece carries."""
        return self._access._unwrap(self._access.leaf(self))

    @property
    def value(self):
        """The section 5.4 Python landing: ``leaf`` lands on ``node.value``, the
        naked value (bool / int / float / str / None)."""
        return self.leaf

    # ---- the chain head as the source has it, no unfolding ----

    @property
    def is_list(self) -> bool:
        return self._access.container_kind(self.descriptor) == "list"

    @property
    def is_set(self) -> bool:
        return self._access.container_kind(self.descriptor) == "set"

    @property
    def is_dict(self) -> bool:
        return self._access.container_kind(self.descriptor) == "dict"

    def __len__(self) -> int:
        return self._access._unwrap(self._access.length(self))

    def keys(self) -> List[str]:
        return list(self._access._unwrap(self._access.keys(self)))

    def values(self):
        return [self[key] for key in self.keys()]

    def items(self):
        return [(key, self[key]) for key in self.keys()]

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def __getitem__(self, key):
        if isinstance(key, int) and not isinstance(key, bool):
            return self._access._unwrap(self._access.get(self, at_index(key)))
        return self._access._unwrap(self._access.get(self, at_key(str(key))))

    # ---- magic methods of the Python landing ----

    def _has_tag(self, name: str) -> bool:
        given = self._access.has(self, by_tag(name))
        return bool(given.ok_value) if isinstance(given, Ok) else False

    def _has_field(self, index: int) -> bool:
        given = self._access.has(self, by_field_index(index))
        return bool(given.ok_value) if isinstance(given, Ok) else False

    def __contains__(self, name: str) -> bool:
        return self._has_tag(name)

    def __getattr__(self, name: str):
        if name.startswith("get_field_"):
            return lambda: self.by_field_index(int(name[len("get_field_"):]))
        if name.startswith("has_field_"):
            return lambda: self._has_field(int(name[len("has_field_"):]))
        if name.startswith("try_get_field_"):
            return lambda: self._access.get(
                self, by_field_index(int(name[len("try_get_field_"):])))
        if name.startswith("get_"):
            return lambda: self.by_tag(name[len("get_"):])
        if name.startswith("has_"):
            return lambda: self._has_tag(name[len("has_"):])
        if name.startswith("try_get_"):
            return lambda: self._access.get(self, by_tag(name[len("try_get_"):]))
        raise AttributeError(name)

    def __dir__(self):
        return _dir_names(self)


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
