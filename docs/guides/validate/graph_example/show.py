"""Draw the example pipeline into the visualise-the-graph guide.

`python docs/guides/validate/graph_example/show.py [OPTION ...]` runs
`catena graph config.toml [OPTION ...]` in this directory and prints the
command, as a Markdown code block, followed by the graph it drew.
"""

import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import graphviz
from dotdoc import graph_html

HERE = Path(__file__).parent


def main(options: list[str]) -> None:
    """Print the command with these options, then the graph it draws."""
    command = ["catena", "graph", "config.toml", *options]
    # style.toml names a style function in styling.py, which must be importable.
    prefix = "PYTHONPATH=. " if "--style" in options else ""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "pipeline"
        result = subprocess.run(
            [*command, "--output", str(out)],
            cwd=HERE,
            env={**os.environ, "PYTHONPATH": str(HERE)},
            capture_output=True,
            text=True,
        )
        if result.returncode:
            sys.exit(result.stderr or result.stdout)
        dot = out.with_suffix(".dot").read_text()
    print(f"```sh\n{prefix}{shlex.join(command)}\n```\n")
    print(graph_html(graphviz.Source(dot), overlay=None))


if __name__ == "__main__":
    main(sys.argv[1:])
