"""Run catena in-process for the documentation's executed code blocks.

markdown-exec runs a `python exec="true"` block inside the documentation
build, so catena is imported once per build rather than once per block. Each
helper returns Markdown for the block to `print`: markdown-exec captures only
the block's own `print`, not output written by imported code.

A page's first block puts this directory on `sys.path`:

```python exec="true" session="page"
import sys

sys.path.insert(0, "docs/_scripts")
from catena_blocks import cli
```

Paths are relative to the repository root, where the build runs.
"""

import runpy
import shlex
import sys
import tempfile
from base64 import b64encode
from pathlib import Path

import graphviz
from dotdoc import graph_html
from typer.testing import CliRunner

from catena.cli.app import app

GRAPH_EXAMPLE = Path("docs/guides/validate/graph_example").resolve()


def _invoke(args: list[str], returncode: int) -> str:
    """Run `catena ARGS`, returning its output; raise on an unexpected exit."""
    result = CliRunner().invoke(app, args, catch_exceptions=False)
    if result.exit_code != returncode:
        raise RuntimeError(
            f"`catena {shlex.join(args)}` exited with {result.exit_code}, "
            f"expected {returncode}:\n{result.output}"
        )
    return result.output


def cli(command: str, *, returncode: int = 0) -> str:
    """Show `catena COMMAND` and its output, as two code blocks.

    A backslash-newline in `command` continues the line, as in a shell, and is
    shown as written.
    """
    output = _invoke(shlex.split(command.replace("\\\n", " ")), returncode)
    return f"```bash\ncatena {command}\n```\n\n```text\n{output.rstrip()}\n```\n"


def png(path: str, alt: str) -> str:
    """Return an `<img>` with the PNG at `path` inlined."""
    data = b64encode(Path(path).read_bytes()).decode()
    return (
        f'<img src="data:image/png;base64,{data}" alt="{alt}" style="max-width:100%">'
    )


def make_data(recipe: str) -> None:
    """Write a recipe's synthetic input with its `make_data.write_inputs`."""
    runpy.run_path(f"recipes/{recipe}/make_data.py")["write_inputs"]()


def graph_example(*options: str) -> str:
    """Show `catena graph config.toml OPTIONS` for the graph example, and its graph.

    The command is shown as run from the example's directory, and the graph
    is drawn with `dotdoc.graph_html`. A `--style` file is taken from the
    example's directory, which is put on `sys.path` so the style function it
    names can be imported.
    """
    if str(GRAPH_EXAMPLE) not in sys.path:
        sys.path.insert(0, str(GRAPH_EXAMPLE))
    args = list(options)
    if "--style" in args:
        i = args.index("--style") + 1
        args[i] = str(GRAPH_EXAMPLE / args[i])
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "pipeline"
        config = str(GRAPH_EXAMPLE / "config.toml")
        _invoke(["graph", config, *args, "--output", str(out)], returncode=0)
        dot = out.with_suffix(".dot").read_text()
    prefix = "PYTHONPATH=. " if "--style" in options else ""
    command = shlex.join(["catena", "graph", "config.toml", *options])
    return f"```sh\n{prefix}{command}\n```\n\n" + graph_html(
        graphviz.Source(dot), overlay=None
    )
