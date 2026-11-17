"""markdown: a directory of .viba files read as one document (implemented in Rust)."""

from rs_viba._native import files_of, lines_of, module_files

__all__ = ["files_of", "lines_of", "module_files"]
