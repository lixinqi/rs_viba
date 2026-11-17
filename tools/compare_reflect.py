"""Compare the Python and Rust reflect implementations over sample designs.

Run from the repository root:

    python3 tools/compare_reflect.py

The script parses the same source through both packages, builds the same
VibaData through both ASTs, walks every address returned by `_walk`, and
compares each address's repr and leaf.  It expects the original Python
project at /Users/bytedance/workspace/viba; set VIBA_PYTHON_ROOT to override.
"""

import importlib
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORIGINAL = os.environ.get("VIBA_PYTHON_ROOT", "/Users/bytedance/workspace/viba")
sys.path.insert(0, REPO)
sys.path.insert(0, ORIGINAL)

CASES = [
    (
        "Box",
        "Box",
        """Box =
  Object
  * $a int
  * $b str
  * $xs list[int]
  * $d dict[str, int]
""",
        lambda ast: ast.ProductChain([
            ast.TypeRef("Object"),
            ast.Tagged("$a", ast.Constant(7)),
            ast.Tagged("$b", ast.Constant("hi")),
            ast.Tagged("$xs", ast.TypeApp("ListLiteral", [ast.Constant(1), ast.Constant(2)])),
            ast.Tagged("$d", ast.TypeApp("DictLiteral", [
                ast.Tuple([ast.Constant("k"), ast.Constant(9)])])),
        ]),
    ),
    (
        "Maybe",
        "Maybe",
        """Maybe =
  Oneof
  | $none Nil
  | $some int
""",
        lambda ast: ast.Tagged("$some", ast.Constant(7)),
    ),
    (
        "Nested",
        "Nested",
        """Nested =
  Object
  * $rows list[list[int]]
  * $lookup dict[str, list[int]]
  * $flag bool
""",
        lambda ast: ast.ProductChain([
            ast.TypeRef("Object"),
            ast.Tagged("$rows", ast.TypeApp("ListLiteral", [
                ast.TypeApp("ListLiteral", [ast.Constant(1), ast.Constant(2)]),
                ast.TypeApp("ListLiteral", [ast.Constant(3)]),
            ])),
            ast.Tagged("$lookup", ast.TypeApp("DictLiteral", [
                ast.Tuple([ast.Constant("a"), ast.TypeApp("ListLiteral", [ast.Constant(4)])]),
                ast.Tuple([ast.Constant("b"), ast.TypeApp("ListLiteral", [ast.Constant(5), ast.Constant(6)])]),
            ])),
            ast.Tagged("$flag", ast.Constant(True)),
        ]),
    ),
]


def walk(package: str, name: str, source: str, body_of):
    ast = importlib.import_module(f"{package}.viba_ast")
    types = importlib.import_module(f"{package}.type")
    descriptor = importlib.import_module(f"{package}.viba_type_descriptor")
    reflect = importlib.import_module(f"{package}.reflect")

    pool = descriptor.empty_pool()
    parsed = descriptor.parse_viba_file(pool, source, f"{name}.viba", name)
    assert isinstance(parsed, types.Ok), parsed
    pool = descriptor.pool_add_file(pool, parsed.ok_value).ok_value
    definition = descriptor.pool_find_definition(pool, f"{name}.{name}").ok_value
    rooted = reflect.access.root(definition, reflect.VibaData(body_of(ast)))
    assert isinstance(rooted, types.Ok), rooted

    nodes = []
    for node in reflect.access._walk(rooted.ok_value):
        path = [(step.kind, step.value) for step in node.path]
        leaf = reflect.access.leaf(node)
        if isinstance(leaf, types.Ok):
            given = ["ok", leaf.ok_value]
        else:
            given = ["err", leaf.msg]
        nodes.append({"path": path, "repr": repr(node), "leaf": given})
    nodes.sort(key=lambda item: json.dumps(item["path"], sort_keys=True))
    return nodes


def probes(package: str, name: str, source: str, body_of):
    ast = importlib.import_module(f"{package}.viba_ast")
    types = importlib.import_module(f"{package}.type")
    descriptor = importlib.import_module(f"{package}.viba_type_descriptor")
    reflect = importlib.import_module(f"{package}.reflect")

    pool = descriptor.empty_pool()
    parsed = descriptor.parse_viba_file(pool, source, f"{name}.viba", name)
    pool = descriptor.pool_add_file(pool, parsed.ok_value).ok_value
    definition = descriptor.pool_find_definition(pool, f"{name}.{name}").ok_value
    root = reflect.access.root(definition, reflect.VibaData(body_of(ast))).ok_value

    def normalized(action):
        try:
            return ["ok", repr(action())]
        except Exception as error:
            return ["raise", type(error).__name__, str(error)]

    return [
        normalized(lambda: root.by_tag("$a")),
        normalized(lambda: root.by_tag("$missing")),
        normalized(lambda: root.at_index(0)),
        normalized(lambda: root.at_key("k")),
        normalized(lambda: root.by_field_index(0)),
        normalized(lambda: reflect.access.leaf(root)),
    ]


def main() -> int:
    total = 0
    for name, definition, source, body_of in CASES:
        left = walk("viba", definition, source, body_of)
        right = walk("rs_viba", definition, source, body_of)
        total += len(right)
        python_probes = probes("viba", definition, source, body_of)
        rust_probes = probes("rs_viba", definition, source, body_of)
        probes_same = python_probes == rust_probes
        print(f"{name}: python={len(left)} rust={len(right)} same={left == right} probes_same={probes_same}")
        if not probes_same:
            print(" python probes:", python_probes)
            print(" rust probes:  ", rust_probes)
            return 1
        if left != right:
            for a, b in zip(left, right):
                if a != b:
                    print("first difference")
                    print(" python:", a)
                    print(" rust:  ", b)
                    break
            return 1
    print(f"compared {total} nodes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
