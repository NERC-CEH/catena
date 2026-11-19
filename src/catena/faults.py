"""Faults a dry run reports, as records a program can read.

`catena.pipeline.dry_run` puts one `Fault` per fault in each stage's `errors`, and
one per soft issue in its `findings`. A fault reads as its `message`, the wording the
CLI prints. Its other fields say the same for a CI script, an editor or an agent,
so none of them needs to parse the message, and `Fault.to_dict` gives them as
JSON-ready data.

`kind` says what went wrong. It is public API: a kind is added only for a new way to
fail, and never renamed or reused.

| `kind` | Stage | Meaning |
| --- | --- | --- |
| `"config_invalid"` | `config` | The config file does not parse |
| `"input_unreadable"` | `inputs` | An `[inputs.*]` source could not be opened or read |
| `"blocking_dim_missing"` | `inputs` | `[blocking]` names a dimension no input has |
| `"input_check_failed"` | `checks` | A `[validation].checks` entry failed |
| `"dag_build_failed"` | `dag` | The DAG could not be built from the config |
| `"contract_malformed"` | `dag`, `contracts` | A contract declaration is malformed |
| `"contract_conflict"` | `dag` | Two declarations on one DAG edge conflict |
| `"contract_violation"` | `contracts` | A loaded input fails a declaration on it |
| `"unknown_target"` | `plan` | A requested node is not in the DAG |
| `"plan_invalid"` | `plan` | Hamilton cannot plan the run, e.g. an unbound input |
| `"unused_input"` | `plan` | A loaded input no node consumes, reported as a finding |
| `"output_not_writable"` | `outputs` | An `[outputs.*]` destination cannot be written |

The other fields are described on `Fault`. Where a contract is at fault, `reason`,
`expected` and `found` come from the records `xarray_annotated.declaration_conflicts`
and `xarray_annotated.declaration_violations` return, which list every `reason` a
facet can fail for.
"""

from dataclasses import asdict, dataclass
from typing import Any, Literal

type FaultKind = Literal[
    "config_invalid",
    "input_unreadable",
    "blocking_dim_missing",
    "input_check_failed",
    "dag_build_failed",
    "contract_malformed",
    "contract_conflict",
    "contract_violation",
    "unknown_target",
    "plan_invalid",
    "unused_input",
    "output_not_writable",
]


@dataclass(frozen=True)
class Fault:
    """One fault, as a message and as the facts behind it.

    `str()` gives `message`. Each field but `kind` and `message` is `None` where
    it does not apply.

    Attributes
    ----------
    kind
        What went wrong, as listed in the table above.
    message
        The fault in words, as the CLI prints it.
    node
        The variable a `contract_conflict` is on, the input a `contract_violation`
        is about, the node an `unknown_target` names, or the `unused_input`.
    source
        One end of a `contract_conflict`'s edge, e.g.
        `"output of partition_fluxes"`: the producer, where the edge has one.
    target
        The other end of a `contract_conflict`'s edge, e.g.
        `"input of to_mass_flux"`, or the consumers whose declarations a
        `contract_violation` fails, e.g. `"input of a, b"`.
    facet
        The contract facet at fault: `"unit"`, `"dims"`, `"coords"`, `"dtype"` or
        `"freq"`. Coords never conflict, so only a violation has `"coords"`.
    reason
        Why a contract is at fault, as the xarray-annotated `kind`, e.g.
        `"unit_different"`; or the name of the check an `input_check_failed`
        ran, e.g. `"time_equal"`.
    expected
        For a conflict, `target`'s declaration of `facet`; for a violation, the
        declaration it fails. Each is a dict, e.g. `{"unit": "degC"}`.
    found
        For a conflict, `source`'s declaration of `facet`, as a dict; for a
        violation, what the input has for the facet, e.g. `"K"` or
        `["time", "y", "x"]`.
    hint
        What to do about a `contract_violation`, as worded in `message`.
    location
        Where in the config: the inputs an `input_check_failed` compared, e.g.
        `"[inputs.met], [inputs.soil]"`, or what made the request an
        `unknown_target` names, e.g. `"[outputs.*]"`.

    Examples
    --------
    >>> fault = Fault("unknown_target", "'x' is not a node", node="x")
    >>> fault.to_dict()["node"]
    'x'
    """

    kind: FaultKind
    message: str
    node: str | None = None
    source: str | None = None
    target: str | None = None
    facet: str | None = None
    reason: str | None = None
    expected: Any = None
    found: Any = None
    hint: str | None = None
    location: str | None = None

    def __str__(self) -> str:
        """Return `message`."""
        return self.message

    def to_dict(self) -> dict[str, Any]:
        """Return the fields as JSON-ready data."""
        return asdict(self)
