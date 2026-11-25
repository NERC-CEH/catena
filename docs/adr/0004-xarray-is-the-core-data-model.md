# 4. xarray is the core data model

- **Status:** accepted
- **Date:** 2026-11-25

## Context

catena assumes that the values passed between nodes are `xarray.DataArray`s.
Loading inputs (`io.py`), subsets, resampling, input checks and the contract vocabulary (units, dims, coords, dtype, freq) all build on that assumption.
The question is whether to keep that assumption or to generalise catena to several data types.

xarray is the best fit for gridded data.
For single-point time series it is slower per operation than pandas, but the overhead is avoidable by stacking sites on a shared dimension rather than looping over them.
It can also wrap non-numpy arrays such as JAX, which matters for ML nodes.
Support is partial (JAX arrays are immutable, and they do not combine with dask), so this is realistic for a node that converts at its boundary rather than for a whole pipeline.

Environmental data that xarray handles poorly falls into two groups:

- **Geometry and topology:** point clouds, vector features (catchments, parcels, habitat polygons) and networks (river routing).
- **Ragged or tabular records:** occurrence and event data, animal tracks, and irregularly sampled multi-site observations.

Unstructured meshes are not an exception, since `uxarray` builds on xarray.

In a modelling pipeline, data from both groups is usually rasterised, aggregated to sites or put on a shared time axis before a model consumes it, so it sits at the edge of the DAG rather than inside it.

pandas and polars are not good substitutes for xarray as the node data type.
Neither carries metadata reliably: polars keeps none, and pandas `.attrs` is experimental.
More fundamentally, dims, coords and freq are array concepts, so catena's contract vocabulary does not carry over to tables.
Missing metadata would weaken only the input and run-time checks.
The edge check compares declarations read from type hints and never inspects a value.

## Decision

1. **catena stays xarray-native.**
   Nodes exchange `DataArray`s, and no abstraction over other data types is added.
2. **Data outside xarray's scope enters through user modules** that convert it to a `DataArray`, or is prepared outside catena.
   If a domain needs non-array values inside the DAG (point clouds being the likely first case), it gets a separate package rather than a generalised catena.
3. **The DAG-wide contract check stays in catena for now.**
   `contract_check.py` already uses `xarray-annotated` through only five functions:
   - `declaration_from_hint`
   - `assert_valid_declaration`
   - `declaration_conflicts`
   - `declaration_violations`
   - `resolve_enabled`

   The rest of the file is a Hamilton graph walk that knows nothing about xarray.
   It is extracted into a standalone package, with those five functions as a backend protocol, only once a second backend exists, such as a schema library for tabular data.

## Consequences

- Pipelines whose interior nodes pass tables or geometries do not fit catena.
  Occurrence-based species distribution modelling is the most likely such case.
- Users convert point clouds, vector data and event records at the edge of the DAG, so that conversion is a modelling choice they make explicitly.
- Extracting the contract check later is mechanical, as long as `contract_check.py` keeps its narrow interface to `xarray-annotated`.

## Alternatives considered

- **Generalise catena over several data types.**
  This would need a contract vocabulary that means something for arrays, tables and geometries alike, and adapters for loading, subsetting and resampling each one.
  The cost falls on every module, while the data that needs it mostly sits at the DAG's edge.
- **Extract the contract check now.**
  With one backend, the protocol cannot be tested against a genuinely different model of what a contract is.
  For example, it is unclear which facets are lower bounds and what "provably inconsistent" means for a table.
  Until a second backend exists, the extracted package would just be catena's internals with a release of its own to maintain.
