---
title: Resampling & units
icon: lucide/timer
---

# Resampling & units

A resample keeps the input's units whether it takes a `mean` or a `sum`: catena copies the attributes, CF `units` included, straight to the result.
This matches `pint-xarray`, which does not multiply by the timestep on a sum either.

!!! warning "Unit validation cannot catch a wrong `aggfunc`"

    The units are the same either way, so **no contract check will catch this**.
    A wrong `aggfunc` gives a result that is dimensionally consistent and physically meaningless.

## Match `aggfunc` to the *kind* of quantity

| Kind of quantity | Example units | Use | Result |
|---|---|---|---|
| **Rate** (intensive) | `g C m-2 day-1`, `mm hr-1`, `W m-2` | `mean` | the mean rate over the window, same units |
| **Amount per period** (extensive) | `g C m-2` accumulated that day, `mm` fallen that day | `sum` | the window total, same units |

## The trap: summing a rate

To get a window total from a rate, the correct operation is an integral — Σ rateᵢ · Δt — which cancels the time dimension in the units (`g C m-2 day-1` × `day` → `g C m-2`).
But xarray's `.sum()` omits the Δt factor, so you get:

```
7 × (g C m-2 day-1)  ->  still labelled "g C m-2 day-1"
```

a number seven times too large, labelled with the units of a rate.

```toml
# WRONG: gpp is a rate (g C m-2 day-1); summing it does not integrate it.
[[resample]]
vars = ["gpp"]
from = "daily"
to = "weekly"
freq = "7D"
aggfunc = "sum"

# RIGHT: the weekly mean daily rate, same units.
[[resample]]
vars = ["gpp"]
from = "daily"
to = "weekly"
freq = "7D"
aggfunc = "mean"
```

Summing a daily rate does give the weekly amount, but only the label is wrong: seven days of `g C m-2 day-1` sum to `g C m-2 week-1`.
catena does not infer that from `aggfunc`, so if a `sum` is what you mean, declare the unit of the result and replace the kept label:

```toml
[[resample]]
vars = ["gpp"]
from = "daily"
to = "weekly"
freq = "7D"
aggfunc = "sum"
units = "g C m-2 week-1"
overwrite_units = true  # the sum keeps the daily rate's label
```

Now the declared units say what the number is, and the contract check has something to check against.
Declaring `units` without `overwrite_units` fails when the node runs, since the kept `g C m-2 day-1` label contradicts it.
`overwrite_units` relabels without looking at the values, so it is a claim you are responsible for.

## Resampling an axis with holes

A resample returns a regular axis, with NaN in the empty bins, for `mean` and `sum` alike.
Two Januaries of daily values at `freq = "1ME"` give 13 bins, 11 of them NaN, and those pass the node's declared `Freq("1ME")`.
