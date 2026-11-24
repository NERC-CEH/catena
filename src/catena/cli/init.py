"""`catena init`: write a starting config from a set of modules."""

import os
from collections.abc import Callable, Collection
from pathlib import Path
from typing import Annotated, Any

import typer

from ..errors import CatenaFileExistsError
from ..importing import discover_registered_modules, is_file_form
from ..scaffold import generate_config

Postprocess = Callable[[dict[str, Any]], None]


def make_init_app(
    *, exclude: Collection[str] = (), postprocess: Postprocess | None = None
) -> typer.Typer:
    """Return a typer app holding the `init` command.

    Parameters
    ----------
    exclude
        Registered modules not to offer in the interactive menu. They can still
        be named with `--module`.
    postprocess
        Called on the generated config's data, in place, before it is written.
    """
    app = typer.Typer()

    @app.command()
    def init(
        modules: Annotated[
            list[str] | None,
            typer.Option(
                "--module",
                "-m",
                help="A module to include: a registered name, a dotted module "
                "path, or a .py file relative to the working directory. Repeat "
                "for several. Omit to choose interactively.",
            ),
        ] = None,
        output: Annotated[
            Path, typer.Option("--output", "-o", help="Where to write the config.")
        ] = Path("config.toml"),
        force: Annotated[
            bool, typer.Option("--force", help="Replace an existing config.")
        ] = False,
    ) -> None:
        """Write a starting config for a set of modules.

        The config lists the inputs the modules need and nothing among them
        produces, the outputs nothing consumes, and each module's settings at
        their defaults. Its file paths are placeholders, and anything left to
        fill in by hand is listed in comments at the top of the file.
        """
        if output.exists() and not force:
            raise CatenaFileExistsError(
                f"There is already a file at {output}. Pass --force to replace it."
            )
        chosen = modules or _choose(exclude)
        base = output.resolve().parent
        generated = generate_config(
            [_relative_to(name, base) for name in chosen], base=base
        )
        if postprocess is not None:
            postprocess(generated.data)
        generated.dump(output, overwrite_ok=True)
        typer.echo(f"Wrote {output}.")
        for note in generated.notes:
            typer.echo(f"  TODO: {note}")

    return app


def _relative_to(module: str, base: Path) -> str:
    """Rewrite a `.py` path given from the working directory relative to `base`."""
    if not is_file_form(module) or Path(module).is_absolute():
        return module
    return Path(os.path.relpath(Path(module).resolve(), base)).as_posix()


def _choose(exclude: Collection[str]) -> list[str]:
    """Ask for modules: registered ones by number or name, or import paths."""
    offered = [name for name in discover_registered_modules() if name not in exclude]
    chosen: list[str] = []
    typer.echo(
        "Toggle modules by number or name, or add a dotted module path or .py "
        "file. Enter nothing when done."
    )
    while True:
        for i, name in enumerate(offered, start=1):
            mark = "x" if name in chosen else " "
            typer.echo(f"  [{mark}] {i}: {name}")
        extra = [name for name in chosen if name not in offered]
        if extra:
            typer.echo(f"  Also: {', '.join(extra)}")
        answer = typer.prompt("Module", default="", show_default=False).strip()
        if not answer:
            if chosen:
                return chosen
            typer.echo("Choose at least one module.")
            continue
        for token in answer.replace(",", " ").split():
            name = _resolve_token(token, offered)
            if name is None:
                typer.echo(f"  No module number {token}.")
            elif name in chosen:
                chosen.remove(name)
            else:
                chosen.append(name)


def _resolve_token(token: str, offered: list[str]) -> str | None:
    """Turn a menu entry into a module name; `None` for a number out of range."""
    if token.isdigit():
        index = int(token)
        return offered[index - 1] if 1 <= index <= len(offered) else None
    return token


app = make_init_app()

__all__ = ["app", "make_init_app"]
