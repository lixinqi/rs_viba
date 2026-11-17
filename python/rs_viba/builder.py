"""viba.builder — spell Viba source with Python operators.

    import viba.builder

    vb = viba.builder.Builder("import store_core as sc")
    tag = viba.builder.tag

    vb.UserName = str

    vb.Optional[vb.T] = vb.T | None

    vb.List[vb.T] = (
        vb.Oneof
        | vb.Object * tag.head(vb.T) * tag.tail(vb.List[vb.T])
        | None
    )

    vb.latest_file[vb.Ctx] = (
        vb.sc.Result[vb.sc.FileState]
        ** tag.ctx(vb.Ctx)
        ** tag.file(vb.sc.FileId)
    )

    print(str(vb))

The operators are the language's operators:

    |            sum            A | B
    *            product        A * B
    **           exponent       A <- B, and the right side must be a tag
    <<           partial        A << B, the argument B given to A; B is a tag too
    tag.x(body)  a tagged field $x body
    tag(body)    a group kept whole
    vb.nil       nil            (None is nil too)
    vb.never     never
    ...          the ellipsis
    vb.Name      a name in the source, and the slot a definition lands in
    vb.Name[T] = body           Name[T] = body
    vb.Name = body              Name = body
    vb.Name[arg0, arg1]         Name[arg0, arg1], an application
    vb.Name[()]                 Name[], an application of nothing
    vb.a.b.Name                 a dotted name, e.g. an import's module
    add_import(vb, m, a)        import m as a

Python's own `str` / `int` / `float` / `bool` / `list` / `set` / `dict` are the
language's names, `int | str` (what Python itself makes of a union) is a sum,
`None` is `nil`, `...` is the ellipsis, and any other Python value is a literal. A Python `list` / `set` / `dict` is the container literal it
means — `[a, b]` is `ListLiteral[a, b]`, `{a, b}` is `SetLiteral[a, b]` (spelled
in sorted order, a set has none of its own) and `{a: 0}` is
`DictLiteral[(a, 0)]`, in the order given.

The helpers stand outside the builder, so that every `vb.<name>` is a
definition and never a method:

    viba.builder.literal(value)                a plain Python value, to start with
    viba.builder.hint(text)                    { text }
    viba.builder.add_import(vb, module, alias) import module as alias
    viba.builder.comment(vb, text)             # text
    viba.builder.check(vb)                     parse str(vb) back; raises otherwise

Starting from a file that is already there appends to it:

    vb = viba.builder.Builder(Path("store.viba").read_text())
    vb.Added = vb.Object * tag.x(vb.T)     # a new definition, after the old ones
    Path("store.viba").write_text(str(vb))

An exponent's right side is a tagged field and nothing else, so a slip shows up
right away: `A ** tag.count(vb.T)` is `A <- $count T`, `A ** tag(B ** tag.p(C))`
is `A <- (B <- $p C)`, and `A ** vb.T` is a TypeError — `**` is not a power
here.

A definition is spelled `vb.Name = body` or `vb.Name[Params] = body`, and comes
out as `Name = body` or `Name[Params] = body`: the same `=` in both places, and
no second spelling to remember.

Associativity: the language's `|` and `*` are left-associative and Python's are
too, so `A | B | C` grows the one chain and `A | (B | C)` keeps a branch — the
same chains the parser takes back. Python's `**` is right-associative while
`<-` is left-associative, so `A ** B ** C` is taken as the source chain
`A <- B <- C`; the nested group `A <- (B <- C)` needs `vb(...)`.

The source form is canonical: definitions come out in one style, chains as
chains, so `str(vb)` is stable — parse it and unparse it and nothing moves.

The expression walk and the validation live in Rust (`rs_viba._native`); the
classes below are the Python face of them, so that Python's operators can spell
the language's.
"""

from __future__ import annotations

from rs_viba._native import (
    builder_add_import_py as _add_import,
    builder_ast_py as _ast,
    builder_branch_py as _branch,
    builder_check_py as _check,
    builder_comment_body_py as _comment_body,
    builder_comment_py as _comment,
    builder_define_generic_py as _define_generic,
    builder_define_py as _define,
    builder_definition_name_py as _definition_name,
    builder_elements_py as _elements,
    builder_existing_names_py as _existing_names,
    builder_exponent_append_py as _exponent_append,
    builder_free_py as _free,
    builder_hint_py as _hint,
    builder_items_py as _items,
    builder_literal_py as _literal,
    builder_lshift_py as _lshift,
    builder_name_attr_py as _name_attr,
    builder_name_py as _name,
    builder_name_setattr_py as _name_setattr,
    builder_origin_name_py as _origin_name,
    builder_param_py as _param,
    builder_path_py as _path,
    builder_power_py as _power,
    builder_product_append_py as _product_append,
    builder_product_py as _product,
    builder_rpower_py as _rpower,
    builder_slot_py as _slot,
    builder_source_py as _source,
    builder_sum_append_py as _sum_append,
    builder_sum_py as _sum,
    builder_tag_call_py as _tag_call,
    builder_tagged_py as _tagged,
    builder_wrap_py as _wrap,
)

__all__ = ["Builder", "tag", "literal", "hint", "add_import", "comment",
           "check", "name"]


class _Expr:
    """One type expression in the source."""

    def to_ast(self):
        return _ast(self)

    def __repr__(self):
        return f"<{self.__class__.__name__}>"

    # `|` and `*` grow the chain on the left, as the language does; a
    # parenthesized right operand lands as one element, which is a branch.
    def __or__(self, other):
        return _sum(self, other)

    def __ror__(self, other):
        return _sum(other, self)

    def __mul__(self, other):
        return _product(self, other)

    def __rmul__(self, other):
        return _product(other, self)

    # `<-` is left-associative in the language and `**` is right-associative in
    # Python: `A ** B ** C` arrives as `A ** (B ** C)` and the native splice
    # puts it back into the source order A <- B <- C.
    def __pow__(self, other):
        return _power(self, other)

    def __rpow__(self, other):
        return _rpower(other, self)

    # `<<` gives one argument in the source to a function: left-associative in the
    # language too, so the nest Python builds is the nest the source has.
    def __lshift__(self, other):
        return _lshift(self, other)


class _Sum(_Expr):
    def __init__(self, elements):
        self.elements = _elements(elements)

    def __or__(self, other):
        return _sum_append(self, other)


class _Product(_Expr):
    def __init__(self, elements):
        self.elements = _elements(elements)

    def __mul__(self, other):
        return _product_append(self, other)


class _Exponent(_Expr):
    def __init__(self, elements):
        self.elements = _elements(elements)

    def __pow__(self, other):
        return _exponent_append(self, other)


class _Branch(_Expr):
    """A group kept whole: one element of the chain around it."""

    def __init__(self, inner):
        self.inner = _wrap(inner)


class _Tagged(_Expr):
    def __init__(self, name: str, body):
        self.name = name.lstrip("$")
        self.body = _wrap(body)


class _Tag:
    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return f"tag.{self.name}"

    def __call__(self, body) -> _Tagged:
        return _Tagged(self.name, body)


class _TagFactory:
    """`tag.head(T)` spells `$head T`; `tag(expr)` keeps a group whole."""

    def __getattr__(self, name: str) -> _Tag:
        if name.startswith("_"):
            raise AttributeError(name)
        return _Tag(name)

    def __call__(self, *args) -> _Expr:
        return _tag_call(args)


tag = _TagFactory()


class _TypeApp(_Expr):
    def __init__(self, constructor, args):
        self.constructor = _path(constructor)
        self.args = _elements(args)


class _Partial(_Expr):
    def __init__(self, function, argument):
        self.function = function
        self.argument = argument


class _Literal(_Expr):
    def __init__(self, value):
        self.value = value


class _Tuple(_Expr):
    def __init__(self, elements):
        self.elements = _elements(elements)


class _Nil(_Expr):
    pass


class _Ellipsis(_Expr):
    pass


class _Hint(_Expr):
    def __init__(self, text: str):
        self.text = text


class _Name(_Expr):
    """A name in the source: a reference, an application head, a definition slot."""

    def __init__(self, path: str, owner=None):
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "owner", owner)

    def __repr__(self):
        return self.path

    def __setattr__(self, name: str, body) -> None:
        """`vb.a.b = body` is a slip: a definition's name is one name."""
        _name_setattr(self, name, body)

    def __getattr__(self, name: str) -> "_Name":
        if name.startswith("_"):
            raise AttributeError(name)
        return _name_attr(self, name)

    def __getitem__(self, key) -> _TypeApp:
        return _TypeApp(self.path, _items(key))

    def __setitem__(self, key, body) -> None:
        """`vb.Name[T] = body` spells `Name[T] = body`."""
        _define_generic(self, key, body)


def literal(value) -> _Expr:
    """A plain Python value as the left end of an expression.

    `"a" | 1` is Python's to refuse — neither side knows the builder — so the
    left end is spelled `literal("a") | 1`.
    """
    return _literal(value)


def name(path: str) -> _Name:
    """The name in the source `path`: what `vb.<name>` gives.

    A builder spells names as its own attributes, so a name it cannot reach that
    way — one beginning with an underscore, the two `__dyn_*` names among them —
    is spelled here instead.
    """
    return _name(path)


def hint(text: str) -> _Hint:
    """`{ text }` — a hint block, kept verbatim."""
    return _hint(text)


def add_import(vb: "Builder", module: str, alias: str | None = None) -> "Builder":
    """`import module [as alias]` on that builder.

    An import stands above the definitions, so this only works on a file that
    has none yet: one built from existing text can only take new definitions.
    """
    return _add_import(vb, module, alias)


def comment(vb: "Builder", text: str) -> "Builder":
    """A `# text` line, spelled where it stands between the definitions."""
    return _comment(vb, text)


def check(vb: "Builder"):
    """Parse `str(vb)` back into a Module; raises ValueError when the source
    form is not Viba source."""
    return _check(vb)


class _Comment:
    """A `# ...` line: not a node, kept as the source has it."""

    def __init__(self, text: str):
        self.text = _comment_body(text)

    @staticmethod
    def text(item: "_Comment") -> str:
        return f"# {item.text}" if item.text else "#"


class Builder:
    """One Viba file under construction: what it starts from, then definitions.

    `Builder()` starts an empty file, `Builder("import a.b as c")` starts one
    with that header, and `Builder(existing)` starts from a whole file that is
    already there — only new definitions are appended to it, and a name it
    already defines is refused. `str(vb)` gives the file: what it started from
    as the source has it, then the new definitions canonically.
    """

    def __init__(self, existing: str = ""):
        object.__setattr__(self, "_existing", existing.strip())
        object.__setattr__(self, "_body", [])
        object.__setattr__(self, "_taken", _existing_names(existing))

    def __getattr__(self, name: str) -> _Name:
        if name.startswith("_"):
            raise AttributeError(name)
        return _slot(self, name)

    def __call__(self, expr) -> _Branch:
        """`vb(expr)` keeps a group whole: one element of the chain around it.

        Only an exponent needs it — `vb(A ** B ** C)` is `A <- (B <- C)`,
        where `A ** B ** C` alone is the chain `A <- B <- C` — a sum or a
        product gets the same grouping from plain parentheses.
        """
        return _branch(expr)

    def __setattr__(self, name: str, value) -> None:
        """`vb.Name = body` spells `Name = body`."""
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        _define(self, name, value)

    def _free(self, name: str) -> str:
        """The name, unless the file this builder continues already has it."""
        return _free(self, name)

    def _add(self, node) -> "Builder":
        self._body.append(node)
        return self

    def __str__(self) -> str:
        """The whole file: what it started from, then the new definitions."""
        return _source(self)

    def __repr__(self):
        return f"<Builder {len(self._body)} node(s)>"
