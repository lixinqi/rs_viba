"""is_complete: does every address a design can list walk to a leaf?"""

from rs_viba._native import (
    is_complete_chain_elements as _chain_elements,
    is_complete_py as is_complete,
)

__all__ = ["is_complete"]
