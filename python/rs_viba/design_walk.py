"""Small readings of a design's source shared by the walking layers."""

from rs_viba._native import (
    bindings_key_py as bindings_key,
    call_parts_py as call_parts,
    chain_elements_py as chain_elements,
    is_definition,
    is_function,
    is_under_py as is_under,
    longest_import,
    module_label,
    split_type_name,
    under_builtin,
    with_params_py as with_params,
)

__all__ = [
    "chain_elements", "call_parts", "split_type_name", "is_definition",
    "is_function", "with_params", "bindings_key", "longest_import",
    "is_under", "under_builtin", "module_label",
]
