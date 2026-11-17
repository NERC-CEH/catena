"""Gridded (CRS) I/O: the optional geospatial + parallel-Zarr layer.

This is the domain-specific counterpart to the domain-agnostic `catena.io`. It
owns:

- **CRS-aware coordinates**: checking that gridded inputs share one grid, and
  computing 2-D `latitude`/`longitude` by reprojecting it to EPSG:4326
  (`compute_lat_lon`);
- **parallel subset I/O**: pre-creating a shared Zarr store and region-writing
  independent `[subset]` slices into it, and reassembling NetCDF parts
  (`create_output_store`, `save_zarr_region`, `merge_subset_outputs`).

See `catena.gridded` for why every import of this module is lazy, and why importing
it is nonetheless cheap and dependency-free.
"""

import re
from os import PathLike
from pathlib import Path

import numpy as np
import xarray as xr

from ..errors import (
    CatenaFileExistsError,
    CatenaFileNotFoundError,
    CatenaValueError,
)
from ..formats import format_for
from ..specs import IOSpec, ParsedConfig, SubsetSpec


class MisalignedGridError(CatenaValueError):
    """Raised when two datasets do not share a common CRS and coordinates."""


def _ensure_rio() -> None:
    """Import rioxarray (registering the `.rio` accessor) or raise a clear error.

    The geospatial layer (grid checks + lat/lon reprojection) is optional;
    `rioxarray`/`pyproj` are only needed when CRS-bearing inputs are present.
    """
    try:
        import rioxarray as rioxarray  # registers the .rio accessor
    except ImportError as exc:  # pragma: no cover - only hit without the extra
        raise ImportError(
            "geospatial (CRS-bearing) inputs require the optional 'geo' extra; "
            "install it with "
            '`pip install "ukceh-catena[geo] @ '
            'git+https://github.com/NERC-CEH/catena"`.'
        ) from exc


def has_crs(ds: xr.Dataset) -> bool:
    """Cheaply detect CRS metadata without importing rioxarray.

    Looks for the markers CF/rioxarray round-trips through NetCDF/Zarr: a `crs`
    global attribute, a `spatial_ref` grid-mapping coordinate, or a per-variable
    `grid_mapping` attribute. Used to decide whether to activate the (lazy)
    geospatial path, so non-gridded pipelines never touch rioxarray/pyproj.
    """
    if ds.attrs.get("crs") or "spatial_ref" in ds.coords:
        return True
    return any("grid_mapping" in da.attrs for da in ds.data_vars.values())


# ---------------------------------------------------------------------------
# Grid computation (CRS -> latitude/longitude)
# ---------------------------------------------------------------------------


def _check_common_grid(
    ds1: xr.Dataset,
    ds2: xr.Dataset,
    label1: str = "ds1",
    label2: str = "ds2",
    atol: float = 1e-6,
) -> None:
    """Raise MisalignedGridError if CRS and coordinates do not match."""
    _ensure_rio()
    if ds1.rio.crs != ds2.rio.crs:
        raise MisalignedGridError(
            f"Mismatched CRS! {label1}={ds1.rio.crs} ≠ {label2}={ds2.rio.crs}"
        )
    x1, y1 = ds1.rio.x_dim, ds1.rio.y_dim
    x2, y2 = ds2.rio.x_dim, ds2.rio.y_dim
    if not (x1 == x2 and y1 == y2):
        raise MisalignedGridError(
            f"Mismatched dimension names: {label1}=({x1}, {y1}) ≠ {label2}=({x2}, {y2})"
        )
    try:
        np.testing.assert_allclose(ds1[x1].values, ds2[x2].values, atol=atol)
        np.testing.assert_allclose(ds1[y1].values, ds2[y2].values, atol=atol)
    except AssertionError as e:
        raise MisalignedGridError(
            f"Mismatched coordinate values between {label1} and {label2}!"
        ) from e


def compute_lat_lon(
    spatial_datasets: dict[str, xr.Dataset],
) -> tuple[xr.DataArray, xr.DataArray]:
    """Compute 2-D latitude and longitude DataArrays from CRS-bearing datasets.

    Both are in EPSG:4326, on the inputs' own spatial dimensions in `(y, x)` order
    (whatever `rioxarray` infers them to be called), and carry CF metadata:
    `units` of `degrees_north` / `degrees_east` and the matching `standard_name`.
    Under the CF unit registry either satisfies a declared `Unit("degrees")`.

    Raises `MisalignedGridError` unless every dataset shares one CRS and grid.
    """
    from pyproj import Transformer

    _ensure_rio()
    items = list(spatial_datasets.items())
    ref_name, ref_ds = items[0]
    for name, ds in items[1:]:
        _check_common_grid(ref_ds, ds, label1=ref_name, label2=name)

    x_dim, y_dim = ref_ds.rio.x_dim, ref_ds.rio.y_dim
    # indexing="ij": y varies along axis 0, x along axis 1.
    y_grid, x_grid = np.meshgrid(
        ref_ds[y_dim].values, ref_ds[x_dim].values, indexing="ij"
    )
    transformer = Transformer.from_crs(ref_ds.rio.crs, "EPSG:4326", always_xy=True)
    lon_grid, lat_grid = transformer.transform(x_grid, y_grid)

    coords = {y_dim: ref_ds[y_dim], x_dim: ref_ds[x_dim]}
    lat = xr.DataArray(
        lat_grid,
        dims=(y_dim, x_dim),
        coords=coords,
        name="latitude",
        attrs={"units": "degrees_north", "standard_name": "latitude"},
    )
    lon = xr.DataArray(
        lon_grid,
        dims=(y_dim, x_dim),
        coords=coords,
        name="longitude",
        attrs={"units": "degrees_east", "standard_name": "longitude"},
    )
    return lat, lon


# ---------------------------------------------------------------------------
# Parallel subset I/O: region writes + store creation + merge
# ---------------------------------------------------------------------------


def subset_suffix(spec: SubsetSpec) -> str:
    """Filename suffix that uniquely identifies a subset, e.g. `_y0-500`."""
    return f"_{spec.dim}{spec.start}-{spec.stop}"


def subset_path(path: str, spec: SubsetSpec) -> Path:
    """Insert the subset suffix before the file extension."""
    p = Path(path)
    return p.with_name(f"{p.stem}{subset_suffix(spec)}{p.suffix}")


def _recreate_hint(path: str) -> str:
    return (
        f"Re-create the store from the current config with "
        f"`catena gridded create-store --overwrite`. (Store: '{path}'.)"
    )


def _assert_coords_match(
    ds: xr.Dataset, store: xr.Dataset, path: str, spec: SubsetSpec
) -> None:
    """Raise unless `ds`'s coordinates are the ones the store holds for its region.

    A region write does not write coordinates, so a disagreement would silently
    mislabel the store. Along `spec.dim` the store's coordinate over
    `[start, stop)` must equal `ds`'s; along every other dimension the whole
    coordinate must.
    """
    region = slice(spec.start, spec.stop)
    for dim in ds.dims:
        if dim not in ds.coords:
            continue
        if dim not in store.coords:
            raise CatenaValueError(
                f"Zarr store '{path}' has no '{dim}' coordinate, but the data being "
                f"written into it does. {_recreate_hint(path)}"
            )
        ours, theirs = ds.indexes[dim], store.indexes[dim]
        if dim == spec.dim:
            theirs = theirs[region]
        if len(ours) != len(theirs) or not (ours == theirs).all():
            where = f" over {spec.start}-{spec.stop}" if dim == spec.dim else ""
            raise CatenaValueError(
                f"'{dim}' coordinate does not match Zarr store '{path}'{where}: the "
                f"store has {len(theirs)} value(s) ({theirs[0]} … {theirs[-1]}), "
                f"the data being written has {len(ours)} ({ours[0]} … {ours[-1]}). "
                f"Region writes do not write coordinates, so this would silently "
                f"mislabel the store. {_recreate_hint(path)}"
            )


def open_region_store(path: str, spec: SubsetSpec) -> xr.Dataset:
    """Open the Zarr store a subset run writes into, or raise if it cannot take it.

    The store must exist (see `catena gridded create-store`), have the subset's
    dimension, and be chunked so that the subset's boundaries fall on chunk
    boundaries: two processes writing parts of one chunk would corrupt it, and
    xarray only detects that for dask-backed data. None of this needs the data,
    so it is checked before a run computes anything.
    """
    dim = spec.dim
    if not Path(path).exists():
        raise CatenaFileNotFoundError(
            f"Zarr store '{path}' does not exist. Create it once before running "
            f"subset processes with: `catena gridded create-store <config>`."
        )
    store = xr.open_zarr(path, consolidated=False)
    if dim not in store.dims:
        raise CatenaValueError(
            f"Zarr store '{path}' has no {dim!r} dimension, so a subset over "
            f"{dim!r} cannot be region-written into it. The store's dimensions are "
            f"{sorted(str(d) for d in store.dims)}."
        )
    n_along_dim = store.sizes[dim]
    chunk = _chunk_along(store, dim)
    if spec.stop > n_along_dim:
        raise CatenaValueError(
            f"Subset range {spec.start}-{spec.stop} runs past the end of {dim!r} "
            f"in Zarr store '{path}', which has {n_along_dim} value(s)."
        )
    if spec.start % chunk != 0 or (spec.stop % chunk != 0 and spec.stop != n_along_dim):
        raise CatenaValueError(
            f"Subset range {spec.start}-{spec.stop} is not aligned to the {dim!r} "
            f"chunk size ({chunk}) of Zarr store '{path}'. Concurrent region writes "
            f"require subset boundaries to fall on chunk boundaries. Re-create the "
            f"store with a matching --chunk, or adjust the subset range."
        )
    return store


def save_zarr_region(ds: xr.Dataset, path: str, spec: SubsetSpec) -> None:
    """Write a subset into an existing Zarr store via a region write.

    Only the data variables are written; coordinates already live in the store,
    and writing them in region mode is both unnecessary and disallowed by xarray.
    Because the coordinates are *not* written, they are checked instead
    (`_assert_coords_match`).

    Every chunk is written, even one holding only fill values, so that
    `merge_subset_outputs` can tell a region that was written from one that was
    not.
    """
    _assert_coords_match(ds, open_region_store(path, spec), path, spec)
    data_only = ds.drop_vars(list(ds.coords))
    region = {spec.dim: slice(spec.start, spec.stop)}
    data_only.to_zarr(path, region=region, consolidated=False, write_empty_chunks=True)


def _chunk_along(store: xr.Dataset, dim: str) -> int:
    """Return the on-disk chunk size of the store's data variables along `dim`."""
    sample = next(var for var in store.data_vars.values() if dim in var.dims)
    chunks = sample.encoding.get("chunks")
    return chunks[list(sample.dims).index(dim)] if chunks else store.sizes[dim]


def create_output_store(
    parsed: ParsedConfig,
    dim: str | None = None,
    chunk: int | None = None,
    overwrite: bool = False,
) -> list[str]:
    """Pre-create empty Zarr stores for parallel subset runs.

    For each Zarr output, build an all-NaN template spanning the whole of `dim`
    in the layout a run writes, then write only the metadata and coordinates
    (data arrays are dask-backed and deferred, so nothing is materialised).
    NetCDF outputs are skipped: they don't need a shared store. Returns the list
    of store paths created.

    The store's layout is **computed from the outputs**, by running the pipeline
    over a single index of `dim` and reading the real coordinates, dims and
    dtype off the result. That makes the store what the subset runs will actually
    write *by construction*, and a derived axis (a `[[resample]]`'s weekly time
    axis, say) needs no input file to have that axis already. Only `dim` is
    resized, to its full length in the input files.

    Parameters
    ----------
    parsed
        The parsed config.
    dim
        The dimension the subset runs will partition. Defaults to the config's
        `[subset]` `dim`.
    chunk
        The store's chunk size along `dim`. Subset boundaries must fall on
        multiples of it. Defaults to the whole of `dim` as one chunk.
    overwrite
        Recreate stores that already exist. Without it an existing store is an
        error, since re-creating one erases what subset runs have written.
    """
    import dask.array as da

    from ..build import build_driver_from_config
    from ..io import (
        dim_size,
        get_final_vars,
        get_outputs,
        load_inputs,
        load_raw_datasets,
    )

    zarr_specs = {
        label: spec
        for label, spec in parsed.output_specs.items()
        if format_for(spec.path).needs_store
    }
    if not zarr_specs:
        return []

    if dim is None:
        if parsed.subset_spec is None:
            raise CatenaValueError(
                "A Zarr output store is partitioned over one dimension. Name it "
                "(CLI: --dim), or give the config a [subset] section."
            )
        dim = parsed.subset_spec.dim
    if not overwrite:
        existing = [
            spec.path for spec in zarr_specs.values() if Path(spec.path).exists()
        ]
        if existing:
            raise CatenaFileExistsError(
                f"Zarr store(s) already exist: {existing}. Re-creating them would "
                f"erase data already written by subset processes. Pass overwrite=True "
                f"(CLI: --overwrite) to recreate them from scratch."
            )

    # Probe the pipeline over one index to learn each output's true layout. Do not
    # replace this with inspecting the input files: an output can carry an axis no
    # input has (a [[resample]]'s time axis), and the store must match what subset
    # runs write. Caching is deliberately not enabled: the probe is an
    # implementation detail of building the store and should neither consult nor
    # populate the user's cache.
    probe_inputs = load_inputs(
        parsed.input_specs, subset_spec=SubsetSpec(start=0, stop=1, dim=dim)
    )
    raw = load_raw_datasets(parsed.input_specs)
    n_along_dim = dim_size(raw, dim, "subset")
    dim_coord = next((ds[dim] for ds in raw.values() if dim in ds.coords), None)
    dr = build_driver_from_config(parsed)
    results = dr.execute(
        get_final_vars(zarr_specs),  # type: ignore[reportArgumentType]
        inputs=probe_inputs,
    )

    created: list[str] = []
    for label, probed in get_outputs(results, zarr_specs).items():
        data_vars = {}
        for name, var in probed.data_vars.items():
            if dim not in var.dims:
                raise CatenaValueError(
                    f"[outputs.{label}] variable {name!r} has no {dim!r} dimension "
                    f"(its dimensions are {list(var.dims)}), so runs over subsets "
                    f"of {dim!r} cannot share one store for it. Write it to a "
                    f"NetCDF output, or leave it out."
                )
            shape = tuple(n_along_dim if d == dim else var.sizes[d] for d in var.dims)
            chunks = tuple(
                (chunk or n_along_dim) if d == dim else var.sizes[d] for d in var.dims
            )
            fill = da.full(shape, np.nan, chunks=chunks, dtype=var.dtype)
            data_vars[name] = (var.dims, fill)
        # Coordinates that do not vary along `dim` are the probe's own; `dim`'s is
        # the input files' full one.
        coords = {
            name: coord
            for name, coord in probed.coords.items()
            if dim not in coord.dims
        }
        if dim_coord is not None:
            coords[dim] = dim_coord.variable

        template = xr.Dataset(data_vars=data_vars, coords=coords)
        template.to_zarr(
            zarr_specs[label].path, compute=False, consolidated=False, mode="w"
        )
        created.append(zarr_specs[label].path)

    return created


def _find_subset_parts(path: Path, dim: str) -> list[Path]:
    """Discover, order and validate the NetCDF subset parts written for `path`.

    Parts are named `{stem}_{dim}{start}-{stop}{suffix}` (`subset_suffix`). They
    are ordered by their integer `start` — a lexicographic sort puts
    `_y1000-1500` before `_y500-1000` — and the ranges must *chain*:
    start at 0, then each start equal to the previous stop. A failed subset run (a
    gap) or a double-written region (an overlap) therefore raises here, rather than
    becoming silent NaN stripes in the merged output.

    Matching on the full `_{dim}<int>-<int>` shape also means a neighbouring
    `{stem}_partial.nc` is not swept into the merge.
    """
    pattern = re.compile(rf"{re.escape(path.stem)}_{re.escape(dim)}(\d+)-(\d+)")
    found: list[tuple[int, int, Path]] = []
    for part in path.parent.glob(f"{path.stem}_{dim}*{path.suffix}"):
        match = pattern.fullmatch(part.stem)
        if match is not None:
            found.append((int(match[1]), int(match[2]), part))

    if not found:
        raise CatenaFileNotFoundError(
            f"No subset parts found matching "
            f"'{path.stem}_{dim}<start>-<stop>{path.suffix}' in {path.parent}."
        )

    found.sort()
    covered = ", ".join(f"{start}-{stop}" for start, stop, _ in found)
    expected = 0
    for start, stop, part in found:
        if start != expected:
            kind = "gap" if start > expected else "overlap"
            raise CatenaValueError(
                f"Subset parts for {path.name} do not cover the {dim!r} dimension: "
                f"{kind} at {dim} {expected} (part '{part.name}' starts at {start}). "
                f"Ranges found: {covered}. Every {dim} must be written exactly once "
                f"or the merged output would contain unwritten regions."
            )
        expected = stop

    return [part for _, _, part in found]


def merge_subset_outputs(
    output_specs: dict[str, IOSpec],
    out: str | PathLike | None = None,
    dim: str | None = None,
) -> list[str]:
    """Finish a set of subset runs: one whole, verified dataset per output.

    For NetCDF outputs, concatenates the per-subset `*_{dim}<start>-<stop>.nc`
    parts along `dim` and writes the result to the config's declared path, or to
    `out` (valid only with a single output section, as the destination would
    otherwise be ambiguous). The parts must chain end to end
    (`_find_subset_parts`).

    A Zarr output needs no merging: the subset runs wrote regions of the store in
    its final layout. It is checked instead, and a store with a chunk no run wrote
    is an error. Returns the list of files written or verified.
    """
    if out is not None and len(output_specs) > 1:
        raise CatenaValueError(
            f"'out' cannot be used with multiple [outputs.*] sections "
            f"({sorted(output_specs)}); the destination would be ambiguous. "
            f"Omit it to use the per-output defaults."
        )

    written: list[str] = []
    for label, spec in output_specs.items():
        path = Path(spec.path)
        if format_for(spec.path).needs_store:
            if out is not None:
                raise CatenaValueError(
                    f"[outputs.{label}] is a Zarr store, which subset runs write in "
                    f"place; 'out' applies to NetCDF outputs only."
                )
            _assert_store_complete(path)
            written.append(str(path))
            continue
        if dim is None:
            raise CatenaValueError(
                f"[outputs.{label}] is merged from NetCDF parts named after the "
                f"dimension the subset runs partitioned. Name it (CLI: --dim), or "
                f"give the config a [subset] section."
            )
        parts = _find_subset_parts(path, dim)
        ds = xr.open_mfdataset(
            parts, combine="nested", concat_dim=dim, decode_coords="all"
        )
        dest = Path(out) if out is not None else path
        ds.to_netcdf(dest, engine="netcdf4")
        written.append(str(dest))

    return written


def _assert_store_complete(path: Path) -> None:
    """Raise unless every chunk of every data variable in the store was written.

    `save_zarr_region` writes every chunk of its region, including ones holding
    only fill values, so a chunk missing from the store is a region no subset run
    reached.
    """
    import zarr

    if not path.exists():
        raise CatenaFileNotFoundError(f"Zarr store '{path}' does not exist.")
    group = zarr.open_group(path, mode="r")
    store = xr.open_zarr(path, consolidated=False)
    missing = []
    for name in store.data_vars:
        array = group[str(name)]
        if array.nchunks_initialized < array.nchunks:  # type: ignore[union-attr]
            missing.append(
                f"{name!r}: {array.nchunks - array.nchunks_initialized} of "  # type: ignore[union-attr]
                f"{array.nchunks} chunk(s)"  # type: ignore[union-attr]
            )
    if missing:
        raise CatenaValueError(
            f"Zarr store '{path}' is incomplete; never written: "
            f"{'; '.join(missing)}. A subset run failed or was not launched. Re-run "
            f"the missing subsets, then check again."
        )
