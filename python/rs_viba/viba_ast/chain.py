"""Chain-style canonicalization (implemented in Rust)."""

from rs_viba._native import convert_from_chain_style, convert_to_chain_style
from rs_viba.viba_ast.nodes import ExponentChain, ProductChain, SumChain

__all__ = ["convert_to_chain_style", "convert_from_chain_style", "is_chain_type"]


def is_chain_type(node) -> bool:
    return isinstance(node, (SumChain, ProductChain, ExponentChain))
