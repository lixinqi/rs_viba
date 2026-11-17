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

## A runnable example

[`examples/score.viba`](examples/score.viba) is one program: four steps with
their types and hints, whose logic the host gives in
[`examples/run_example.py`](examples/run_example.py). The runner executes the
same source once with the original Python package (`../viba`) and once with
`rs_viba`, then compares the return values:

```bash
python3 examples/run_example.py
# viba     ret=14
# rs_viba  ret=14
# same:    True
```

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

## Native execution path

`src/native_runtime.rs` is the first slice of the native runtime. A module
with no imports, no generics and no patterns, whose host steps take an
environment followed by scalar arguments, is parsed once into a Rust
expression tree and evaluated over `Value` values. Final tagged products are
also built natively; their synthetic type descriptors are cached by value
structure, so repeated runs do not rebuild them. The compiled tree is cached
by file path, modification time and source length, so repeated runs of the same
module skip parsing and compilation. Every other shape falls back to the full
interpreter, which remains the behavioural reference.

The AST node classes are Rust classes (`src/ast_api.rs`); the module
`rs_viba.viba_ast.nodes` imports them, and their `_fields`, constructors,
field access and `copy.deepcopy` behave as before. `Ok`, `Err`, `Frame`, `InterpretError`, `VibaProgramErr`,
`UnderlyingOpErr` and `EnvironmentApiInvalidArgumentErr` are Rust classes in
`src/type_api.rs`; the last one is an `InterpretError` subclass too. The `Type` leaves and `AstNodeType` are Rust classes in
`src/type_model.rs`; `ModuleType`, `CustomModuleType`, `BuiltinModuleType` and
`GenericModuleType` are Rust classes too. The interpreter's small runtime
records (`_VibaData`, `_Host`, `_Member`, `_Given`, `_Activation`, `_Pending`,
`_HostArgument`, `_GetArgs`, `_IsSubType`, `_HostFunction`, `_DynCall`,
`_Getter`, `_Runner`, `_ResumableCall`) and the control-flow exceptions
`_Compact` and `_Raised` are Rust classes in `src/runtime_data.rs`.
`EnvironmentStorage` and `EnvironmentCompute` are Rust classes in
`src/storage_api.rs`; the remaining Python classes in the interpreter module
are the host-facing `Environment`, `ReferencedEnvironment` and the helper
`_BoundFindByRelativePath`. `PatternFile` and `Choice` are Rust classes in
`src/pattern_api.rs`. `Config`, `VibaStep`, `VibaFunction`, `VObject`
and `VibaData` are Rust classes (`src/reflect.rs`) exported through
`rs_viba.reflect`; `VibaAccess` was already Rust. The descriptor classes
(`VibaTypeDescriptor`, the payload descriptors, `VibaMemberDescriptor`,
`VibaDefinitionDescriptor`, `VibaFileDescriptor`, `VibaPool`) are Rust classes
in `src/descriptor_api.rs`, imported by `rs_viba.viba_type_descriptor`. Walking the same design and data through `viba.reflect` and
`rs_viba.reflect` gives the same addresses, members and leaves. The script
`tools/compare_reflect.py` compares three designs and 24 nodes; each node's
path, repr and leaf match between the two implementations. It also checks the
error paths for a missing tag, a wrong index, a wrong key and a non-leaf root;
both implementations answer the same result or the same exception text.

Warm measurements on the same machine (Apple M4 Pro, Python 3.12.13, release
build), compared with the Python project:

| program | viba | rs_viba | speedup |
|---|---:|---:|---:|
| `examples/score.viba` | 487.0 us | 56.4 us | 8.63x |
| a chain of 4 host steps | 499.8 us | 64.7 us | 7.73x |
| a chain of 16 host steps | 1720.8 us | 89.2 us | 19.29x |
| a chain of 32 host steps | 3387.7 us | 123.7 us | 27.38x |

These are warm runs of a module already compiled in the same process. Cold
startup is not covered by these numbers and was excluded from the target.

All 59 suites pass with this path enabled by default.
