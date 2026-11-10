---
title: Inline nodes & fan-out
icon: lucide/braces
---

# Inline nodes & fan-out

A `[[node]]` entry defines a node directly in the config, with no Python module.
It suits glue — renames, arithmetic, simple derivations — and, with `for_each`, generating many similar nodes from one entry.

## Inline expressions

The simplest node is a Python `expression`, evaluated with the listed `inputs` and `xr` (xarray) in scope.

```toml
[[node]]
name = "temperature_range_daily"
inputs = ["tmax_daily", "tmin_daily"]
expression = "tmax_daily - tmin_daily"
```

`inputs` are node names, suffix included.
`name` is the node this entry produces, which downstream nodes and `[outputs.*]` can reference.

## Calling an external function

To call a function instead, give `_import_path` and `function` in place of `expression`:

```toml
[[node]]
name = "custom_index_daily"
inputs = ["temperature_daily", "precipitation_daily"]
_import_path = "mypackage.indices"
function = "compute_custom_index"
```

The function must accept keyword arguments matching `inputs` and return an `xarray.DataArray`.
This suits a single derivation; once you have several related functions, shared parameters, or annotated signatures, [write a module](../nodes/bring-your-own-module.md) instead.

## Declaring contracts on a node

catena cannot infer what an expression produces, so declare it.
Any of `units`, `dims`, `dtype`, `coords` or `freq` makes the node a *typed producer*: its output is validated at run time, and the [contract check](../nodes/contracts.md) checks downstream consumers against it.

```toml
[[node]]
name = "aridity_index_daily"
inputs = ["precipitation_daily", "evapotranspiration_daily"]
expression = "precipitation_daily / evapotranspiration_daily"
units = "1"                 # dimensionless ratio
overwrite_units = true      # replace the mm d-1 the ratio kept
dims = ["time", "pixel"]    # optional dimension contract
```

`units` and `dtype` are validated when the config is parsed.
The declared unit is a statement about the expression, and catena treats it as one:

- **An output with no `units` label** is stamped with the declared unit.
  xarray drops the label when the inputs' labels disagree, as in a depth in `mm` divided by a duration in `d`.
- **An output that kept a label** from its inputs must match the declared unit, or the node raises.
  xarray keeps the first input's label through `x * 7.0`, `x.sum()`, and a ratio of two inputs with the same label, so it is usually stale when the expression changes the unit.
- **`overwrite_units = true`** replaces a kept label with the declared one.
  It also hides a forgotten conversion, so for a real conversion use [`xra.units.convert`](../nodes/contracts.md#converting-units), which is in scope, and leave `overwrite_units` off.

The stamp is a claim, not a check: it labels the values with what the config says, whether or not the expression computed them in that unit.
A node that declares nothing has no output contract, so edges leaving it go unchecked.

## Fan-out with `for_each`

`for_each` generates one node per value, substituting `{var}` into the string fields (`name`, `inputs`, `expression`).
It is the config-level equivalent of Hamilton's `@parameterize`.

```toml
[[node]]
for_each = ["temperature", "precipitation", "humidity"]
name = "{var}_anomaly_daily"
inputs = ["{var}_daily"]
expression = "{var}_daily - {var}_daily.mean('time')"
```

This expands to three nodes, `temperature_anomaly_daily`, `precipitation_anomaly_daily` and `humidity_anomaly_daily`, each wired to its own input.

## The `[[resample]]` preset

`[[resample]]` aggregates to a coarser time frequency.
It expands to one [passthrough](../nodes/contracts.md#contracts-across-a-passthrough-node) node per variable, each applying `catena.transforms.resample`.

```toml
[[resample]]
vars = ["temperature", "precipitation"]
from = "daily"
to = "weekly"
freq = "7D"
aggfunc = "mean"          # mean | sum | max | min | first | last (default: mean)
```

This produces `temperature_weekly` and `precipitation_weekly` from their daily counterparts, carrying each source's units and dims across, with `freq` as the declared output frequency.
Choose `aggfunc` with care: [no check can tell a wrong one](../nodes/resampling-and-units.md).

`from` and `to` are just the node-name suffixes to read and write; `freq` alone sets the frequency, so any pair of labels works:

```toml
[[resample]]
vars = ["temperature"]
from = "hourly"
to = "daily"
freq = "1D"
aggfunc = "max"
```

`from = ""` reads bare names, for inputs loaded with `suffix = ""`: with `vars = ["temperature"]` and `to = "weekly"`, it reads `temperature` and writes `temperature_weekly`.
`to` cannot be empty, since the result would take its source's name.

Every key is in the [configuration reference](../../reference/configuration.md#nodes).
