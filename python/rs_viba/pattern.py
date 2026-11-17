"""Pattern: a generic is a directory, and one of its files answers (implemented in Rust)."""

from typing import Callable, Dict, List, Optional

from rs_viba import viba_ast
from rs_viba._native import (
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


class PatternFile:
    """One file of a generic: the count its name spells, its order, and its text."""

    __slots__ = ("order", "path", "name", "declared", "module", "patterns",
                 "_load", "_problem", "_taken", "_checked")

    def __init__(self, order: int, path: str, name: str,
                 declared: Optional[int], load: Optional[Callable[[], object]]):
        self.order = order
        self.path = path
        self.name = name
        self.declared = declared
        self.module = None
        self.patterns = None
        self._load = load
        self._problem = None
        self._taken = False
        self._checked = False

    @classmethod
    def ready(cls, order: int, path: str, name: str, declared: Optional[int],
              module, patterns) -> "PatternFile":
        entry = cls(order, path, name, declared, None)
        entry.module = module
        entry.patterns = list(patterns)
        entry._taken = True
        return entry

    def take(self):
        return _pattern_file_take(self)

    def __repr__(self):
        state = ("not taken" if self.patterns is None
                 else f"{len(self.patterns)} params")
        return f"PatternFile({self.order}, {self.path!r}, {state})"


class GenericModuleType(ModuleType):
    """A generic: the directory whose files are its patterns."""

    def __init__(self, name: str, directory: str, entries: List[PatternFile],
                 module: Optional[viba_ast.Module] = None):
        self.name = name
        self.directory = directory
        self.entries = list(entries)
        self.module = module if module is not None else viba_ast.Module([])

    def __repr__(self):
        return f"GenericModuleType({self.name!r}, {len(self.entries)} patterns)"


class Choice:
    """What the decision picked: the body to take, and the names it binds."""

    __slots__ = ("entry", "body", "module", "bindings")

    def __init__(self, entry: PatternFile, body, module, bindings: Dict[str, AstNodeType]):
        self.entry = entry
        self.body = body
        self.module = module
        self.bindings = dict(bindings)

    def env_get(self, name: str):
        bound = self.bindings.get(name)
        if bound is None:
            return Err(f"unbound parameter {name!r}")
        return Ok(bound)

    def __repr__(self):
        return f"Choice({self.entry.path!r}, {sorted(self.bindings)})"


__all__ = [
    "GENERIC_FILE", "Choice", "GenericModuleType", "PatternFile", "declared_of",
    "decide", "exponent_elements", "file_pattern_problem", "generic_named",
    "generic_of_entries", "generic_of_files", "load_generic", "order_of",
    "patterns_of", "reduce_application", "structural_pattern_match",
    "sum_elements", "tagged_type_of",
]
