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

from rs_viba._native import (
    VibaTypeDescriptor, VibaMemberTakenDescriptor, VibaTypeRefDescriptor,
    VibaTypeAppDescriptor, VibaTupleDescriptor, VibaTaggedDescriptor,
    VibaChainDescriptor, VibaLiteralDescriptor, VibaHintBlockDescriptor,
    VibaMemberDescriptor, VibaDefinitionDescriptor, VibaImportDescriptor,
    VibaFileDescriptor, VibaPool,
)

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
