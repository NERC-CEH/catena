"""Docstrings and comments are Markdown, which mkdocstrings renders.

reStructuredText constructs either show up verbatim in the API reference or, for
attribute comments, drop the documentation altogether, since griffe does not read
them. Tests and recipes are held to the same rules because they are read and
imitated.
"""

import ast
import io
import pathlib
import re
import tokenize

import pytest

ROOT = pathlib.Path(__file__).parent.parent
FILES = sorted(
    p
    for top in ("src", "recipes", "tests")
    for p in (ROOT / top).rglob("*.py")
    if "__marimo__" not in p.parts
)

DOCSTRING_RULES = {
    "Sphinx role; use `name`, or [`name`][dotted.path] to link": re.compile(
        r":[a-z]+(:[a-z]+)?:`"
    ),
    "reST directive; use a !!! admonition": re.compile(r"^\s*\.\. [a-z-]+::"),
    "reST literal block; use a fenced code block": re.compile(r"(\S|^\s*)::\s*$"),
    "reST double-backtick literal; use single backticks": re.compile(r"(?<!`)``(?!`)"),
}
COMMENT_RULES = {
    "Sphinx attribute comment; put a docstring below the assignment": re.compile(
        r"^#:"
    ),
    "reST double-backtick literal; use single backticks": re.compile(r"(?<!`)``(?!`)"),
}


def _violations(path: pathlib.Path) -> list[str]:
    src = path.read_text()
    lines = src.splitlines()
    found = []
    for node in ast.walk(ast.parse(src)):
        if not (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            continue
        in_fence = False
        for i in range(node.lineno - 1, node.end_lineno or node.lineno):
            if lines[i].lstrip().startswith("```"):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            found += [
                f"{path.relative_to(ROOT)}:{i + 1}: {why}"
                for why, rule in DOCSTRING_RULES.items()
                if rule.search(lines[i])
            ]
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT:
            found += [
                f"{path.relative_to(ROOT)}:{tok.start[0]}: {why}"
                for why, rule in COMMENT_RULES.items()
                if rule.search(tok.string)
            ]
    return found


def test_files_are_found():
    assert any(p.name == "pipeline.py" for p in FILES)


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_docstrings_and_comments_are_markdown(path):
    assert _violations(path) == []
