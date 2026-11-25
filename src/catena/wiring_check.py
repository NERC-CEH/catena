"""Before-compute validation of the file<->node wiring.

The contract check (`catena.contract_check`) covers the *declared* contracts;
this covers the *wiring*. It compares the external inputs the built DAG requires (to
produce the requested `final_vars`) against the inputs actually loaded from the
config, and reports the two failure modes that otherwise only surface part way
through a run:

- **unbound** — a node requires an external input that was neither loaded nor set
  in the config (an error: the run cannot proceed);
- **unused** — a loaded input that no node consumes (a warning: usually a
  suffix/alias typo, e.g. loading `temperature_daily` for a node that wants
  `temperature_weekly`).

It reads the graph structure only, so it needs no compute. `unbound_inputs`
lists each unbound input with the nodes that consume it, resolving inputs and
config values as Hamilton's `Driver.validate_execution` does; `check_wiring`
then runs `validate_execution` for anything else that would stop the run.
"""

import warnings
from typing import Any

from hamilton import driver, graph_types
from hamilton.execution import graph_functions

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


def unused_inputs(
    dr: driver.Driver, final_vars: list[str], inputs: dict[str, Any]
) -> list[str]:
    """Return the loaded inputs no node upstream of `final_vars` consumes, sorted."""
    external = {
        node.name
        for node in dr.what_is_upstream_of(*final_vars)
        if node.is_external_input
    }
    return sorted(set(inputs) - external)


def unbound_inputs(
    dr: driver.Driver, final_vars: list[str], inputs: dict[str, Any]
) -> dict[str, list[str]]:
    """Return the required inputs with no loaded or config value, sorted by name.

    Each maps to the sorted names of the nodes upstream of `final_vars` that
    consume it. An input with a default value is not required, and a scalar set
    in a module section binds an input as a loaded one does.
    """
    nodes, user_nodes = dr.graph.get_upstream_nodes(final_vars, inputs)  # type: ignore[reportArgumentType]
    bound = graph_functions.combine_config_and_inputs(dr.graph.config, inputs)
    return {
        node.name: sorted(c.name for c in node.depended_on_by if c in nodes)
        for node in sorted(user_nodes, key=lambda n: n.name)
        if node.name not in bound
        and graph_functions.node_is_required_by_anything(node, nodes)
    }


def unbound_message(name: str, consumers: list[str]) -> str:
    """Return the sentence reporting that input `name` of `consumers` is unbound."""
    return (
        f"required input {name!r} of {', '.join(consumers)} is neither loaded nor "
        f"set in the config"
    )


def check_wiring(
    dr: driver.Driver,
    final_vars: list[str],
    inputs: dict[str, Any],
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

    Raises
    ------
    CatenaValueError
        If a required external input has no loaded or config value (unbound),
        naming each one on its own line, or if Hamilton cannot otherwise plan the
        execution, e.g. a loaded value has the wrong type.

    Warns
    -----
    WiringWarning
        If a loaded input is consumed by no node (unused).
    """
    # Warn before raising, so an unbound input does not hide an unused one.
    if unused := unused_inputs(dr, final_vars, inputs):
        warnings.warn(
            f"loaded input(s) consumed by no node: {unused}",
            WiringWarning,
            stacklevel=2,
        )

    if unbound := unbound_inputs(dr, final_vars, inputs):
        lines = "\n".join(
            f"  {unbound_message(name, consumers)}"
            for name, consumers in unbound.items()
        )
        raise CatenaValueError(f"the DAG cannot be executed:\n{lines}")

    try:
        dr.validate_execution(final_vars, inputs=inputs)  # type: ignore[reportArgumentType]
    except ValueError as exc:
        raise CatenaValueError(f"the DAG cannot be executed: {exc}") from None
