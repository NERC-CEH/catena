"""Before-compute validation of the file<->node wiring.

The contract check (`catena.contract_check`) covers the *declared* contracts;
this covers the *wiring*. It compares the external inputs the built DAG requires (to
produce the requested ``final_vars``) against the inputs actually loaded from the
config, and reports the two failure modes that otherwise only surface part way
through a run:

- **unbound** — a node requires an external input that was neither loaded nor set
  in the config (an error: the run cannot proceed);
- **unused** — a loaded input that no node consumes (a warning: usually a
  suffix/alias typo, e.g. loading ``temperature_daily`` for a node that wants
  ``temperature_weekly``).

It reads the graph structure only, so it needs no compute. The unbound check is
Hamilton's own ``Driver.validate_execution``, which names the nodes that need
each missing input.
"""

import warnings
from typing import Any

from hamilton import driver, graph_types

from .errors import CatenaValueError


class WiringWarning(UserWarning):
    """Warning that a loaded input is not consumed by any node."""


def terminal_nodes(dr: driver.Driver) -> list[str]:
    """Return the computed nodes no other node depends on, sorted by name.

    These are the targets that cover the whole DAG, for checking the wiring of a
    config that requests no outputs.
    """
    hg = graph_types.HamiltonGraph.from_graph(dr.graph)
    consumed = {
        dep
        for node in hg.nodes
        for dep in (*node.required_dependencies, *node.optional_dependencies)
    }
    return sorted(
        node.name
        for node in hg.nodes
        if not node.is_external_input and node.name not in consumed
    )


def check_wiring(
    dr: driver.Driver,
    final_vars: list[str],
    inputs: dict[str, Any],
    *,
    exempt: "frozenset[str] | set[str]" = frozenset(),
) -> None:
    """Validate that loaded inputs satisfy (and match) the DAG's external inputs.

    Parameters
    ----------
    dr
        A built Hamilton driver.
    final_vars
        The node names to be computed (as from `catena.io.get_final_vars`).
    inputs
        The loaded input dict (as from `catena.io.load_inputs`).
    exempt
        Loaded input names to exclude from the *unused* check — auto-derived
        inputs (date indices, lat/lon) that ``load_inputs`` always emits but no
        node is required to consume. See `catena.io.auxiliary_input_names`.

    Raises
    ------
    CatenaValueError
        If Hamilton cannot plan the execution: a required external input has no
        loaded or config value (unbound), or a loaded value has the wrong type.

    Warns
    -----
    WiringWarning
        If a loaded input is consumed by no node (unused).
    """
    external = {
        node.name
        for node in dr.what_is_upstream_of(*final_vars)
        if node.is_external_input
    }

    # Warn before raising, so an unbound input does not hide an unused one.
    unused = set(inputs) - external - set(exempt)
    if unused:
        warnings.warn(
            f"loaded input(s) consumed by no node: {sorted(unused)}",
            WiringWarning,
            stacklevel=2,
        )

    try:
        dr.validate_execution(final_vars, inputs=inputs)  # type: ignore[reportArgumentType]
    except ValueError as exc:
        raise CatenaValueError(f"the DAG cannot be executed: {exc}") from None
