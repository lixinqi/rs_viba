"""Pattern: a generic is a directory, and one of its files answers (implemented in Rust)."""

from typing import Callable, Dict, List, Optional

from rs_viba import viba_ast
from rs_viba._native import (
    PatternFile, Choice,
    GenericModuleType,
    decide_py as decide,
    file_pattern_problem_py as file_pattern_problem,
    generic_named_py as generic_named,
    generic_of_entries_py as generic_of_entries,
    generic_of_files_py as generic_of_files,
    load_generic_py as load_generic,
    order_of_py as order_of,
    declared_of_py as declared_of,
    pattern_file_take_py as _pattern_file_take,
    patterns_of_py as patterns_of,
    reduce_application_py as reduce_application,
    structural_pattern_match_py as structural_pattern_match,
    sum_elements_py as sum_elements,
    exponent_elements_py as exponent_elements,
    tagged_type_of_py as tagged_type_of,
)
from rs_viba.type import AstNodeType, Err, Ok, ModuleType

GENERIC_FILE = "__generic__.viba"


__all__ = [
    "GENERIC_FILE", "Choice", "GenericModuleType", "PatternFile", "declared_of",
    "decide", "exponent_elements", "file_pattern_problem", "generic_named",
    "generic_of_entries", "generic_of_files", "load_generic", "order_of",
    "patterns_of", "reduce_application", "structural_pattern_match",
    "sum_elements", "tagged_type_of",
]
