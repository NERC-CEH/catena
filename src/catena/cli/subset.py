"""`catena subset`: create shared Zarr stores for subset runs, and merge the results."""

from pathlib import Path
from typing import Annotated

import typer

from ..config import load_config
from ..subsets import create_output_store, merge_subset_outputs

app = typer.Typer(
    help="Parallel subset runs: pre-create shared Zarr stores and merge the results."
)


@app.command()
def create_store(
    config_file: Annotated[
        Path, typer.Argument(exists=True, file_okay=True, dir_okay=False, readable=True)
    ],
    dim: Annotated[
        str | None,
        typer.Option(
            "--dim",
            help="The dimension the subset runs will partition, e.g. y. Defaults "
            "to the config's [subset] dim.",
        ),
    ] = None,
    chunk: Annotated[
        int | None,
        typer.Option(
            "--chunk",
            help="Chunk size of the store along the partition dimension. Subset "
            "boundaries must be multiples of it. Defaults to [blocking].block_size "
            "when [blocking] partitions the same dimension, else one chunk.",
        ),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option(
            "--overwrite",
            help="Recreate stores that already exist. This erases any data already "
            "written into them by subset runs.",
        ),
    ] = False,
) -> None:
    """Create empty Zarr stores so subset runs can region-write concurrently.

    Run this once before launching independent `catena run` processes that
    each handle a different subset of the same inputs and write to a shared
    Zarr store.
    """
    parsed = load_config(config_file)

    if dim is None and parsed.subset_spec is not None:
        dim = parsed.subset_spec.dim
    blocking = parsed.blocking_spec
    if chunk is None and blocking is not None and blocking.dim == dim:
        chunk = blocking.block_size

    created = create_output_store(parsed, dim=dim, chunk=chunk, overwrite=overwrite)

    if not created:
        typer.echo("No Zarr outputs in config; nothing to create.")
        return
    for path in created:
        typer.echo(f"Created store: {path}")


@app.command()
def merge(
    config_file: Annotated[
        Path, typer.Argument(exists=True, file_okay=True, dir_okay=False, readable=True)
    ],
    out: Annotated[
        Path | None,
        typer.Option(
            "--out",
            "-o",
            help="Explicit destination for the merged NetCDF output. Only valid "
            "with a single output section. Defaults to the config's path.",
        ),
    ] = None,
    dim: Annotated[
        str | None,
        typer.Option(
            "--dim",
            help="The dimension the subset runs partitioned, which names the "
            "NetCDF parts to merge. Defaults to the config's [subset] dim.",
        ),
    ] = None,
) -> None:
    """Finish a set of subset runs: one whole, verified dataset per output.

    NetCDF parts (`*_<dim><start>-<stop>.nc`) are concatenated and written to the
    config's declared path, or to `--out`. A shared Zarr store is already whole,
    so it is checked instead: a chunk no run wrote is an error.

    The partition dimension is `--dim` if given, else the config's `[subset]`
    dim. NetCDF parts carry it in their filenames, so it must be the one the
    subset runs used.
    """
    parsed = load_config(config_file)
    if dim is None and parsed.subset_spec is not None:
        dim = parsed.subset_spec.dim
    written = merge_subset_outputs(parsed.output_specs, out=out, dim=dim)

    if not written:
        typer.echo("No mergeable outputs in config.")
        return
    for path in written:
        typer.echo(f"Complete: {path}")
