"""viba.serialize — serialize a viba data into viba source (the mirror of reflect).

    from viba.reflect import access, VibaData
    from viba import serialize

    root = access.root(definition, VibaData(viba_data))
    serialize.serialize("entry", root.ok_value)
    # -> Ok('entry =\n  Object\n  * $left 1\n  * $right 2\n')

Every value is asked for through the protocol's cells — the same ones that take
values — so this knows nothing about a design beyond what the source has, and nothing
about how Data is bound. What comes out is canonical viba source (it goes through
viba.builder), and `builder.check` parses it back before it is handed over.

A piece the design has no spelling for gives Err rather than a wrong spelling:
that is a viba data the design cannot carry, and the caller decides what to do
about it.

Two things about the layer a piece is on:

- A product leads with `Object` whether or not the design headed its chain with a
  unit, so the block a product comes out as always says on its first line which
  layer it is (viba/core/viba-style.md §6).
- A sum gives the branch the piece is on and nothing else: the value does not
  carry the other branches, and one branch is no sum (viba/core/viba-style.md §7).
- A piece whose type says nothing about it (`Any`, the top type) is serialized
  from its own source form: the type gives no member to walk, so the only thing
  that says what the piece is is the piece itself. A declaration that has `Any`
  in a slot (`InterpretResult` has `$ok Any`) stays round-trippable this way.

The walking lives in Rust (``src/serialize.rs``); `SerializeGap` is this module's
own name for a piece the design has no spelling for.
"""

from __future__ import annotations

from rs_viba._native import serialize_py as _serialize

__all__ = ["serialize", "SerializeGap"]


class SerializeGap(Exception):
    """No viba spelling for this part of the design."""


def serialize(name: str, node):
    """A piece of the design in hand (a VObject) -> viba source: the definition
    named `name` whose body is that piece's spelling.

    The node carries the accessor it was taken through, so this takes the piece
    exactly as the caller did, and asks for no accessor of its own.
    """
    return _serialize(name, node)
