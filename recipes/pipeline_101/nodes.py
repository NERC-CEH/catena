"""Hamilton nodes for the Pipeline 101 recipe.

An ordinary xarray function is an ordinary DAG node: the function name is the
node name, and each positional parameter name is the node it consumes. A
keyword-only parameter is config, set in the module's config section. The
annotations declare the units catena checks across the whole DAG before anything
runs.
"""

from typing import Annotated

import xarray as xr
import xarray_annotated as xra
from xarray_annotated import Unit
from xarray_annotated.units import use_cf_units

use_cf_units()

Temperature = Annotated[xr.DataArray, Unit("degC")]


@xra.check_annotations
def temperature_anomaly_climate(
    temperature_climate: Temperature, *, baseline_days: int | None = None
) -> Temperature:
    """Departure of each day's temperature from the mean of a baseline period.

    The baseline is the first ``baseline_days`` days, or the whole record if unset.
    """
    baseline = temperature_climate.isel(time=slice(0, baseline_days))
    return temperature_climate - baseline.mean("time")
