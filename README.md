# rs_viba

The viba implementation, rewritten in Rust and exported to Python as `rs_viba`.

viba is a language for design files: a `.viba` file defines types with `|`
(sum), `*` (product), `<-` (exponent) and `<<` (partial application), and may
leave the logic of a step to the host through a hint block. The Python project
this crate replaces lives beside it; every behaviour here is exercised by the
same test suites.

## Layout

- `src/` — the implementation. One Rust module per former Python module:
  `parser.rs`, `unparser.rs`, `chain.rs` (AST), `type_system.rs`
  (`rs_viba.type`), `viba_type_descriptor.rs`, `partial.rs`, `pattern.rs`,
  `is_sub_type.rs`, `is_complete.rs`, `reflect.rs`, `serialize.rs`,
  `builder.rs`, `interpret.rs`, `markdown.rs`, `repomix.rs`,
  `extract_sub_repo.rs`, `convert_to_json_schema.rs`,
  `check_convertable_to_json_schema.rs`, `check_tag_and_inline.rs`,
  `design_walk.rs`.
- `python/rs_viba/` — the exported Python package. It holds the public classes
  and thin function bindings; the algorithms live in `src/`. The compiled
  extension is `rs_viba._native` (PyO3).
- `rs_viba` — a symlink to `python/rs_viba`, so the tests find the package from
  the checkout root.
- `tests/` — the viba test suites, with only the `viba` imports replaced by
  `rs_viba`. `viba/` beside it holds the test data and documentation those
  suites read.

## Build and run

```bash
python3 -m venv .venv
.venv/bin/pip install maturin
PATH="$HOME/.cargo/bin:$PATH" .venv/bin/maturin develop --release
python3 run_tests.py
```

`run_tests.py` runs every `tests/test_*.py` and prints one line per suite; a
subset can be selected by name, for example `python3 run_tests.py pattern`.

## The Python boundary

The exported package is what `import rs_viba` reaches. Its modules are named
after the Python project's modules (`rs_viba.interpret`, `rs_viba.pattern`, and
so on), so replacing `viba` with `rs_viba` in an import is the whole migration
for a caller. The classes the suites construct or isinstance-check (AST nodes,
`Ok`/`Err`, descriptors, `VObject`, environments) are Python classes in
`python/rs_viba/`; every operation on them is a Rust function in `src/`.

## Status

All 59 suites under `tests/` pass against this implementation:

```
$ python3 run_tests.py
...
59/59 suites passed
```

That is the same suite count as the Python project. `tests/` differs from the
project's test directory in one way only: every `viba` import was replaced by
`rs_viba` (verified by reversing the replacement and diffing against the
original files).
