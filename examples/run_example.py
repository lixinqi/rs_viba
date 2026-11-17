#!/usr/bin/env python3
"""Run examples/score.viba with both viba implementations and compare.

    python3 examples/run_example.py

`viba` is the original Python project beside this one
(/Users/bytedance/workspace/viba); `rs_viba` is this checkout's Rust
implementation.  Each is imported under its own package name, so both run in
one process against the same source file and the same host functions.
"""

import importlib
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ORIGINAL = ROOT.parent / "viba"
SOURCE = HERE / "score.viba"


def host(module_path, func_name):
    """The steps this program leaves to its host."""
    if func_name == "a_value":
        return lambda env: 3
    if func_name == "b_value":
        return lambda env: 4
    if func_name == "add":
        return lambda env, x, y: x.value + y.value
    if func_name == "double":
        return lambda env, x: x.value * 2
    return None


def run_program(package_name, store):
    """Run the program once, under one implementation, and return its value."""
    interpret_mod = importlib.import_module(f"{package_name}.interpret")
    reflect_mod = importlib.import_module(f"{package_name}.reflect")
    type_mod = importlib.import_module(f"{package_name}.type")

    environ = interpret_mod.Environment(
        interpret_mod.EnvironmentStorage("root", None, str(store)),
        interpret_mod.EnvironmentCompute(host),
    )
    result = interpret_mod.interpret(str(SOURCE), environ)
    try:
        ok = result.by_tag("$ok")
    except Exception as error:
        raise AssertionError(f"{package_name} stopped: {error}") from error
    answered = reflect_mod.access.leaf(ok)
    if not isinstance(answered, type_mod.Ok):
        raise AssertionError(f"{package_name} returned no value: {answered!r}")
    ret = answered.ok_value
    return ret


def main():
    # The original project has to come first: the repo's `viba/` directory is
    # only the test data and documentation this checkout carries.
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ORIGINAL))

    returns = {}
    for package_name in ("viba", "rs_viba"):
        with tempfile.TemporaryDirectory(prefix=f"{package_name}-score-") as tmp:
            returns[package_name] = run_program(package_name, tmp)
            print(f"{package_name:8} ret={returns[package_name]!r}")

    print(f"same:    {returns['viba'] == returns['rs_viba']}")
    return 0 if returns["viba"] == returns["rs_viba"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
