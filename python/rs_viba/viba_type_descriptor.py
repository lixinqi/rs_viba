"""VibaTypeDescriptor — the descriptor side of the reflection protocol.

The spec is ``viba/viba_type_descriptor.viba``; this module implements it.
Spec names are capitalized (``ParseVibaFile``), the Python here is
snake_case (``parse_viba_file``), as in ``viba/type.py``.

The classes below are the protocol's data classes, copied as they are; every
function is a thin wrapper over ``rs_viba._native``, where the algorithm lives
(``src/viba_type_descriptor.rs``).

Two things the spec says, and how they land here:

* Every descriptor holds its pool, so any descriptor can answer its own
  questions; type expressions additionally hold the ``viba.type`` value
  they stand for (``resolvable_type``).
* Fields and branches are the same thing (a tag plus a type), so there is
  one ``VibaMemberDescriptor`` and the pool has one member table.

What the spec leaves open, and how it lands here:

* ``VibaPool`` has no mutable state, so adding a file rebuilds every
  descriptor in the pool to point at the new pool.  Settled: pools are
  built offline in one go, never added to while in use, so the cost is
  accepted.
"""

from __future__ import annotations

from typing import List, Optional

from rs_viba._native import (
    build_type_py as _build_type,
    definition_file_py as definition_file,
    definition_find_member_by_index_py as definition_find_member_by_index,
    definition_find_member_by_tag_py as definition_find_member_by_tag,
    definition_members_py as definition_members,
    descriptor_of_py as descriptor_of,
    descriptor_of_tagged_py as descriptor_of_tagged,
    descriptor_of_values_py as descriptor_of_values,
    empty_pool_py as empty_pool,
    file_find_import_by_local_name_py as file_find_import_by_local_name,
    member_containing_definition_py as member_containing_definition,
    member_resolved_definition_py as member_resolved_definition,
    member_type_name_py as member_type_name,
    parse_viba_file_py as parse_viba_file,
    pool_add_file_py as pool_add_file,
    pool_find_definition_py as pool_find_definition,
    pool_find_file_py as pool_find_file,
    pool_find_member_py as pool_find_member,
    tagged_descriptor_py as tagged_descriptor,
)
from rs_viba.type import AstNodeType

# The builtin name that takes a call's arguments back as a product.
GET_ARGS_NAME = "__get_args__"

# Branch names of the spec's VibaTypeDescriptor sum.
TYPE_REF = "type_ref"
MEMBER_TAKEN = "member_taken"
TYPE_APP = "type_app"
TUPLE = "tuple"
TAGGED = "tagged"
SUM = "sum"
PRODUCT = "product"
EXPONENT = "exponent"
LITERAL = "literal"
NIL = "nil"
NEVER = "never"
ANY = "any"
ELLIPSIS = "ellipsis"
HINT_BLOCK = "code_block"

PRODUCT_UNIT = ("Object", "nil")  # the unit of a product chain head
SUM_UNIT = ("Oneof", "never")  # the unit of a sum chain head
EXPONENT_UNIT = ("never",)  # the unit of an exponent chain head (the result element)


# ----------------------------------------------------------------------
# Type expressions: the twelve branches of the spec's sum
# ----------------------------------------------------------------------


class VibaTypeDescriptor:
    """A type expression in the source: one branch of the spec's sum."""

    __slots__ = ("kind", "payload")

    def __init__(self, kind: str, payload=None):
        self.kind = kind
        self.payload = payload

    @property
    def pool(self) -> Optional["VibaPool"]:
        return getattr(self.payload, "pool", None)

    @property
    def resolvable_type(self) -> Optional[AstNodeType]:
        return getattr(self.payload, "resolvable_type", None)

    def __repr__(self):
        return f"VibaTypeDescriptor({self.kind}, {self.payload!r})"


class VibaMemberTakenDescriptor:
    """`g[T].name` — a member of what an application answers.

    Which member that is a definition of is known once the application is
    decided, so the descriptor keeps the owner's own descriptor and the name;
    the layer that takes designs apart decides, and takes the member there
    (viba/core/viba-pattern.md).
    """

    __slots__ = ("pool", "resolvable_type", "owner", "name")

    def __init__(self, pool, resolvable_type, owner, name: str):
        self.pool = pool
        self.resolvable_type = resolvable_type
        self.owner = owner
        self.name = name

    def __repr__(self):
        return f"VibaMemberTakenDescriptor({self.owner!r}, {self.name!r})"


class VibaTypeRefDescriptor:
    """name — a reference, kept as the source has it."""

    __slots__ = ("pool", "resolvable_type", "type_name")

    def __init__(self, pool, resolvable_type, type_name: str):
        self.pool = pool
        self.resolvable_type = resolvable_type
        self.type_name = type_name

    def __repr__(self):
        return f"VibaTypeRefDescriptor({self.type_name!r})"


class VibaTypeAppDescriptor:
    """Constructor[Arg, ...]"""

    __slots__ = ("pool", "resolvable_type", "constructor_name", "args")

    def __init__(self, pool, resolvable_type, constructor_name: str, args: List[VibaTypeDescriptor]):
        self.pool = pool
        self.resolvable_type = resolvable_type
        self.constructor_name = constructor_name
        self.args = args

    def __repr__(self):
        return f"VibaTypeAppDescriptor({self.constructor_name!r}, {len(self.args)} args)"


class VibaTupleDescriptor:
    """(A, B, C) — positional product."""

    __slots__ = ("pool", "resolvable_type", "elements")

    def __init__(self, pool, resolvable_type, elements: List[VibaTypeDescriptor]):
        self.pool = pool
        self.resolvable_type = resolvable_type
        self.elements = elements

    def __repr__(self):
        return f"VibaTupleDescriptor({len(self.elements)} elements)"


class VibaTaggedDescriptor:
    """$tag Type"""

    __slots__ = ("pool", "resolvable_type", "tag", "tagged_type")

    def __init__(self, pool, resolvable_type, tag: str, tagged_type: VibaTypeDescriptor):
        self.pool = pool
        self.resolvable_type = resolvable_type
        self.tag = tag
        self.tagged_type = tagged_type

    def __repr__(self):
        return f"VibaTaggedDescriptor({self.tag!r})"


class VibaChainDescriptor:
    """Canonical sum chain or product chain."""

    __slots__ = ("pool", "resolvable_type", "elements")

    def __init__(self, pool, resolvable_type, elements: List[VibaTypeDescriptor]):
        self.pool = pool
        self.resolvable_type = resolvable_type
        self.elements = elements

    def __repr__(self):
        return f"VibaChainDescriptor({len(self.elements)} elements)"


class VibaLiteralDescriptor:
    """A literal: the value itself (bool / int / float / str / nil)."""

    __slots__ = ("pool", "resolvable_type", "value")

    def __init__(self, pool, resolvable_type, value):
        self.pool = pool
        self.resolvable_type = resolvable_type
        self.value = value

    def __repr__(self):
        return f"VibaLiteralDescriptor({self.value!r})"


class VibaHintBlockDescriptor:
    """{...} — kept verbatim."""

    __slots__ = ("pool", "resolvable_type", "code")

    def __init__(self, pool, resolvable_type, code: str):
        self.pool = pool
        self.resolvable_type = resolvable_type
        self.code = code

    def __repr__(self):
        return "VibaHintBlockDescriptor(...)"


# ----------------------------------------------------------------------
# Members / definitions / imports / files
# ----------------------------------------------------------------------


class VibaMemberDescriptor:
    """A field of a product or a branch of a sum; the same thing."""

    __slots__ = ("pool", "member_index", "tag", "member_type", "containing_full_name")

    def __init__(self, pool, member_index: int, tag, member_type, containing_full_name: str):
        self.pool = pool
        self.member_index = member_index
        self.tag = tag
        self.member_type = member_type
        self.containing_full_name = containing_full_name

    def __repr__(self):
        return f"VibaMemberDescriptor({self.member_index}, {self.tag!r})"


class VibaDefinitionDescriptor:
    __slots__ = (
        "pool", "def_name", "module_name", "file_name", "file_hash",
        "full_name", "generic_params", "body", "members",
    )

    def __init__(self, pool, def_name, module_name, file_name, file_hash,
                 full_name, generic_params, body, members):
        self.pool = pool
        self.def_name = def_name
        self.module_name = module_name
        self.file_name = file_name
        self.file_hash = file_hash
        self.full_name = full_name
        self.generic_params = generic_params
        self.body = body
        self.members = members

    def __repr__(self):
        return f"VibaDefinitionDescriptor({self.full_name!r})"


class VibaImportDescriptor:
    __slots__ = ("pool", "module_name", "local_name")

    def __init__(self, pool, module_name: str, local_name: str):
        self.pool = pool
        self.module_name = module_name
        self.local_name = local_name

    def __repr__(self):
        return f"VibaImportDescriptor({self.module_name!r} as {self.local_name!r})"


class VibaFileDescriptor:
    __slots__ = ("pool", "file_name", "file_hash", "module_name",
                 "imports", "definitions", "_tree")

    def __init__(self, pool, file_name, file_hash, module_name, imports, definitions, tree):
        self.pool = pool
        self.file_name = file_name
        self.file_hash = file_hash
        self.module_name = module_name
        self.imports = imports
        self.definitions = definitions
        self._tree = tree  # private: lets the pool rebind descriptors to itself

    def __repr__(self):
        return f"VibaFileDescriptor({self.module_name!r}, {len(self.definitions)} defs)"


class VibaPool:
    __slots__ = ("files", "file_name2file", "full_name2definition",
                 "full_name2member", "module_environment")

    def __init__(self, files, file_name2file, full_name2definition,
                 full_name2member, module_environment):
        self.files = files
        self.file_name2file = file_name2file
        self.full_name2definition = full_name2definition
        self.full_name2member = full_name2member
        self.module_environment = module_environment

    def __repr__(self):
        return f"VibaPool({len(self.files)} files)"


__all__ = [
    "VibaTypeDescriptor",
    "VibaTypeRefDescriptor", "VibaTypeAppDescriptor", "VibaTupleDescriptor",
    "VibaTaggedDescriptor", "VibaChainDescriptor",
    "VibaLiteralDescriptor", "VibaHintBlockDescriptor",
    "VibaMemberDescriptor", "VibaDefinitionDescriptor",
    "VibaImportDescriptor", "VibaFileDescriptor", "VibaPool",
    "empty_pool", "descriptor_of",
    "parse_viba_file", "pool_add_file", "pool_find_file",
    "pool_find_definition", "pool_find_member",
    "file_find_import_by_local_name",
    "definition_members", "definition_find_member_by_tag",
    "definition_find_member_by_index", "definition_file",
    "member_type_name", "member_resolved_definition",
    "member_containing_definition",
]
