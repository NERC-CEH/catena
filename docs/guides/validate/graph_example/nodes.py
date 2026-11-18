"""Nodes for the example pipeline drawn in the visualise-the-graph guide."""

from typing import Annotated

import xarray as xr
from hamilton.function_modifiers import tag
from xarray_annotated import Freq, Unit

Kelvin = Annotated[xr.DataArray, Unit("K"), Freq("D")]
Celsius = Annotated[xr.DataArray, Unit("degC"), Freq("D")]
Rain = Annotated[xr.DataArray, Unit("mm"), Freq("D")]


@tag(stage="qc")
def precipitation_checked(precipitation: Rain) -> Rain:
    """Daily precipitation with negative values set to zero."""
    return precipitation.clip(min=0)


def temperature_daily(temperature: Kelvin) -> Celsius:
    """Daily mean temperature in degrees Celsius."""
    return temperature - 273.15


def aridity_index_daily(
    precipitation_checked: Rain, temperature_daily: Celsius, *, offset: float = 10.0
) -> Annotated[xr.DataArray, Unit("mm degC-1"), Freq("D")]:
    """De Martonne aridity index for each day."""
    return precipitation_checked / (temperature_daily + offset)
