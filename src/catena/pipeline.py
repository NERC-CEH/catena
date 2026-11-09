"""Run a configured pipeline end to end, or validate one without running it.

`run` executes everything a ``config.toml`` describes. `dry_run` performs the
same setup and validation but executes nothing and writes nothing, returning a
`DryRunReport` of what each stage found.

Both accept either a path to a config file or an already-parsed `ParsedConfig`,
so you can parse, adjust the spec in Python, and run the result. Only the path
form stamps output provenance, which needs the config text.
"""

import hashlib
import logging
import time
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import xarray as xr

from .blocking import execute_blocked
from .build import build_driver
from .config import load_config
from .errors import CatenaError, CatenaValueError
from .io import (
    assert_dim_in_inputs,
    assert_output_paths_writable,
    auxiliary_input_names,
    get_final_vars,
    get_outputs,
    load_inputs,
    save_outputs,
)
from .specs import CacheSpec, ParsedConfig
from .wiring_check import check_wiring, terminal_nodes

__all__ = [
    "DryRunError",
    "DryRunReport",
    "RunReport",
    "Stage",
    "WrittenOutput",
    "dry_run",
    "run",
]

logger = logging.getLogger(__name__)

#: Either a path to a TOML config file or an already-parsed config.
ConfigSource = str | Path | ParsedConfig


def prepare_config(config: ConfigSource) -> tuple[ParsedConfig, Path | None]:
    """Resolve a config source to a parsed config and its source path, if any.

    Accepts a path to a TOML file or an already-parsed `ParsedConfig`. The returned
    path is ``None`` for a `ParsedConfig`, which has no file to stamp provenance from.
    """
    if isinstance(config, ParsedConfig):
        parsed, config_file = config, None
    else:
        config_file = Path(config)
        parsed = load_config(config_file)
    return parsed, config_file


@dataclass(frozen=True)
class WrittenOutput:
    """One destination a run wrote, and what went into it."""

    #: The ``[outputs.<label>]`` section this came from.
    label: str
    #: Where it was actually written. Under ``[subset]`` this carries the
    #: subset's suffix, so it is not always the path the config asked for.
    path: Path
    variables: tuple[str, ...]
    #: Size on disk, summed over a directory store. ``None`` if it could not be
    #: measured.
    size_bytes: int | None = None


@dataclass(frozen=True)
class RunReport:
    """What a run computed, where it went, and how long it took.

    A hard failure raises out of `run`, so a report exists only for a pipeline
    that ran to completion.
    """

    #: The config's path, or ``None`` when `run` was given a `ParsedConfig`.
    config_file: Path | None
    #: The dataset written for each ``[outputs.*]`` section, keyed by section
    #: name. Empty for a config that declares no outputs.
    outputs: dict[str, xr.Dataset] = field(default_factory=dict)
    written: tuple[WrittenOutput, ...] = ()
    #: Wall-clock seconds, config parse to last byte written.
    elapsed: float = 0.0


def run(
    config: ConfigSource,
    *,
    allow_overrides: bool = False,
    cache: bool | None = None,
    cache_dir: Path | None = None,
) -> RunReport:
    """Execute a pipeline and write its outputs.

    Parameters
    ----------
    config
        Path to a TOML config file, or a `ParsedConfig` to run directly.
    allow_overrides
        Allow a later module to override a node defined by an earlier one.
    cache
        Enable or disable result caching, overriding the config's ``[cache]``
        section. ``None`` defers to the config.
    cache_dir
        Directory for cached results. Implies caching is enabled.

    Returns
    -------
    RunReport
        The datasets written, where each one went, and how long it took. A
        config declaring no outputs still parses, loads inputs, runs the input
        checks and builds the DAG, and comes back with nothing written.

    Notes
    -----
    Progress is logged to the ``catena.pipeline`` logger at ``INFO`` as each
    stage completes, so a caller can route it wherever it wants. ``catena run``
    prints it.
    """
    started = time.perf_counter()
    parsed, config_file = prepare_config(config)
    for source in _registered_sources(parsed):
        logger.info("%s", source)
    cache_spec = _resolve_cache(parsed.cache_spec, cache, cache_dir)
    if cache_spec is not None:
        logger.info("caching enabled: %s", cache_spec.path)

    inputs = load_inputs(
        parsed.input_specs,
        subset_spec=parsed.subset_spec,
    )
    logger.info(
        "inputs loaded: %d variable(s) from %d source(s)",
        len(inputs),
        len(parsed.input_specs),
    )

    if n_checks := _run_input_checks(parsed):
        logger.info("input checks passed (%d)", n_checks)

    # Before the driver is built, not after: building applies the [cache] spec,
    # which creates the cache directory, and a run that is going to abort on an
    # unwritable output should not leave one behind.
    if parsed.output_specs:
        assert_output_paths_writable(parsed.output_specs, parsed.subset_spec)

    dr = build_driver(
        modules=parsed.modules,
        config=parsed.driver_config,
        node_specs=parsed.node_specs,
        allow_module_overrides=allow_overrides,
        cache=cache_spec,
        base=parsed.base,
        registered=parsed.registered_modules,
    )

    if not parsed.output_specs:
        logger.info("DAG built; no [outputs.*] configured, so nothing to execute")
        return RunReport(config_file, elapsed=time.perf_counter() - started)

    target_vars = get_final_vars(parsed.output_specs)
    logger.info("DAG built: executing %d output node(s)", len(target_vars))
    check_wiring(dr, target_vars, inputs, exempt=auxiliary_input_names(inputs))
    if parsed.blocking_spec is not None:
        results = execute_blocked(dr, inputs, target_vars, parsed.blocking_spec)
    else:
        results = dr.execute(target_vars, inputs=inputs)  # type: ignore[reportArgumentType]
    stacked = parsed.subset_spec is not None
    output_datasets = get_outputs(results, parsed.output_specs, stacked=stacked)
    paths = save_outputs(
        output_datasets,
        parsed.output_specs,
        subset_spec=parsed.subset_spec,
        provenance=_config_provenance(config_file),
    )
    written = tuple(
        WrittenOutput(
            label=label,
            path=path,
            variables=tuple(str(name) for name in output_datasets[label].data_vars),
            size_bytes=_size_on_disk(path),
        )
        for label, path in paths.items()
    )
    return RunReport(
        config_file=config_file,
        outputs=output_datasets,
        written=written,
        elapsed=time.perf_counter() - started,
    )


def _size_on_disk(path: Path) -> int | None:
    """Bytes ``path`` occupies, summed over a directory store, or None if unreadable.

    A Zarr output is a directory, and a subset run region-writes into a shared
    store whose size is not this run's doing, so the number is best-effort.
    """
    try:
        if path.is_dir():
            return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        return path.stat().st_size
    except OSError:
        return None


@dataclass(frozen=True)
class Stage:
    """One step of a dry run, and what it found.

    ``detail`` is a complete phrase, e.g. "inputs loaded: 18 variable(s) from 4
    source(s)". A ``failed`` stage carries each fault it found in ``errors``;
    ``findings`` are soft issues, such as an input no node consumes, and do not fail
    the stage.
    """

    name: str
    status: Literal["ok", "skipped", "failed"]
    # A whole phrase, never a fragment for the caller to assemble: the wording of a
    # stage is fixed here so every renderer (the CLI, a CI script, a test) words it
    # the same way.
    detail: str
    findings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class DryRunReport:
    """The outcome of a dry run: what each stage checked and found."""

    #: The config's path, or ``None`` when `dry_run` was given a `ParsedConfig`.
    config_file: Path | None
    stages: tuple[Stage, ...] = ()

    @property
    def failed(self) -> tuple[Stage, ...]:
        """The stages that found a fault."""
        return tuple(stage for stage in self.stages if stage.status == "failed")


class DryRunError(CatenaValueError):
    """A dry run found at least one fault.

    The message lists every fault; ``report`` holds the full `DryRunReport`,
    passing stages included.
    """

    def __init__(self, report: DryRunReport) -> None:
        self.report = report
        lines = [
            f"[{stage.name}] {error}"
            for stage in report.failed
            for error in stage.errors
        ]
        super().__init__(f"dry run found {len(lines)} fault(s):\n" + "\n".join(lines))


def dry_run(config: ConfigSource, *, allow_overrides: bool = False) -> DryRunReport:
    """Validate everything a real run depends on, without executing it.

    Runs the same setup as `run` up to (but excluding) execution: parse the
    config, load inputs (lazily — file metadata only), run the input checks, build
    the driver and check the contracts declared across it, check the wiring and
    the execution plan, validate the loaded inputs' contracts (units +
    dims/coords/dtype + freq) against what the DAG declares, and confirm the
    output destinations are writable.

    The wiring is checked against the nodes the ``[outputs.*]`` sections request
    or, with no outputs configured, against every terminal node of the DAG.

    A stage that finds a fault is marked ``failed`` with the fault in
    `Stage.errors`, and the dry run carries on through every later stage that does
    not depend on it, so one run reports every fault it can reach. A stage that
    cannot run because one it depends on failed is ``skipped``. If any stage
    failed, `DryRunError` is raised at the end, carrying the report. A config
    that fails to parse raises its own error at once, as nothing else can run.

    Contract stages report ``skipped`` when `xarray-annotated` is switched off. No
    model runs and nothing is written.
    """
    from xarray_annotated import resolve_enabled

    from .contract_check import check_dag_contracts, check_input_contracts

    parsed, config_file = prepare_config(config)
    # Reported in `detail` rather than `findings`: a finding renders as a warning,
    # and where a module came from is information, not a problem.
    sources = _registered_sources(parsed)
    stages: list[Stage] = [
        Stage("config", "ok", "; ".join(("config parsed", *sources)))
    ]

    inputs: dict[str, Any] | None
    try:
        inputs = load_inputs(
            parsed.input_specs,
            subset_spec=parsed.subset_spec,
        )
    except CatenaError as exc:
        inputs = None
        stages.append(_failed("inputs", "inputs failed to load", exc))
    else:
        detail = (
            f"inputs loaded: {len(inputs)} variable(s) "
            f"from {len(parsed.input_specs)} source(s)"
        )
        try:
            if parsed.blocking_spec is not None:
                assert_dim_in_inputs(inputs, parsed.blocking_spec.dim, "blocking")
        except CatenaError as exc:
            stages.append(_failed("inputs", detail, exc))
        else:
            stages.append(Stage("inputs", "ok", detail))

    stages.append(_checks_stage(parsed, inputs_loaded=inputs is not None))

    checking = resolve_enabled()
    # Caching is an execution-time adapter; disable it so the dry run creates no
    # cache directory. The graph structure and unit checks are unaffected.
    try:
        dr = build_driver(
            modules=parsed.modules,
            config=parsed.driver_config,
            node_specs=parsed.node_specs,
            allow_module_overrides=allow_overrides,
            cache=None,
            base=parsed.base,
            registered=parsed.registered_modules,
            check_contracts=False,
        )
    except CatenaError as exc:
        dr = None
        stages.append(_failed("dag", "DAG failed to build", exc))
    else:
        try:
            check_dag_contracts(dr)
        except CatenaError as exc:
            stages.append(
                _failed("dag", "DAG built; static contract check failed", exc)
            )
        else:
            stages.append(
                Stage(
                    "dag",
                    "ok",
                    "DAG built (static contract check passed)"
                    if checking
                    else "DAG built (contract checks switched off)",
                )
            )

    if dr is None or inputs is None:
        stages.append(_skipped("plan", "execution plan", dr, inputs))
    else:
        stages.append(_plan_stage(dr, inputs, parsed))

    if not checking:
        stages.append(
            Stage("contracts", "skipped", "input contracts: skipped (switched off)")
        )
    elif dr is None or inputs is None:
        stages.append(_skipped("contracts", "input contracts", dr, inputs))
    else:
        try:
            check_input_contracts(dr, inputs)
        except CatenaError as exc:
            stages.append(_failed("contracts", "input contracts invalid", exc))
        else:
            stages.append(Stage("contracts", "ok", "input contracts validated"))

    if parsed.output_specs:
        try:
            assert_output_paths_writable(parsed.output_specs, parsed.subset_spec)
        except CatenaError as exc:
            stages.append(_failed("outputs", "output paths not writable", exc))
        else:
            stages.append(
                Stage(
                    "outputs",
                    "ok",
                    f"output paths writable: {len(parsed.output_specs)} destination(s)",
                )
            )
    else:
        stages.append(
            Stage(
                "outputs",
                "skipped",
                "output paths: skipped (no [outputs.*] configured)",
            )
        )

    report = DryRunReport(config_file=config_file, stages=tuple(stages))
    if report.failed:
        raise DryRunError(report)
    return report


def _failed(name: str, detail: str, exc: Exception) -> Stage:
    return Stage(name, "failed", detail, errors=(str(exc),))


def _skipped(name: str, what: str, dr: object, inputs: object) -> Stage:
    """Report a stage that cannot run because the DAG or the inputs did not load."""
    missing = " and ".join(
        part
        for part, absent in (("DAG", dr is None), ("inputs", inputs is None))
        if absent
    )
    return Stage(name, "skipped", f"{what}: skipped ({missing} unavailable)")


def _checks_stage(parsed: ParsedConfig, *, inputs_loaded: bool) -> Stage:
    if not parsed.checks:
        return Stage("checks", "skipped", "input checks: none configured")
    if not inputs_loaded:
        return Stage("checks", "skipped", "input checks: skipped (inputs unavailable)")
    try:
        n_checks = _run_input_checks(parsed)
    except CatenaError as exc:
        return _failed("checks", "input checks failed", exc)
    if not n_checks:
        return Stage(
            "checks", "skipped", "input checks: skipped (running under [subset])"
        )
    return Stage("checks", "ok", f"input checks passed ({n_checks})")


def _plan_stage(dr: Any, inputs: dict[str, Any], parsed: ParsedConfig) -> Stage:
    """Check the wiring and execution plan for the requested or terminal nodes."""
    if parsed.output_specs:
        requested = get_final_vars(parsed.output_specs)
        what = "output node(s)"
    else:
        requested = terminal_nodes(dr)
        what = "terminal node(s)"
    known = set(dr.graph.nodes)
    targets = [var for var in requested if var in known]
    errors: list[str] = []
    if unknown := [var for var in requested if var not in known]:
        errors.append(f"[outputs.*] request node(s) the DAG does not define: {unknown}")
    with warnings.catch_warnings(record=True) as wiring_warnings:
        warnings.simplefilter("always")
        try:
            # Before Hamilton's own validation: an unbound input reads more
            # clearly here, and an unused one surfaces as a finding.
            check_wiring(dr, targets, inputs, exempt=auxiliary_input_names(inputs))
        except CatenaError as exc:
            errors.append(str(exc))
    findings = tuple(str(w.message) for w in wiring_warnings)
    if not errors:
        try:
            dr.validate_execution(targets, inputs=inputs)
        except ValueError as exc:  # Hamilton's, naming a fault in the config
            errors.append(str(exc))
    if errors:
        return Stage(
            "plan", "failed", "execution plan invalid", findings, tuple(errors)
        )
    return Stage(
        "plan",
        "ok",
        f"execution plan valid: {len(targets)} {what} reachable",
        findings,
    )


def _registered_sources(parsed: ParsedConfig) -> tuple[str, ...]:
    """Name each section whose module came from an installed package.

    A section with no ``_import_path`` says nothing about where its code lives, so
    the run reports what the environment supplied rather than resolving it silently.
    """
    return tuple(
        f"[{mod.section}] provided by {mod.distribution}: {mod.import_path}"
        for mod in parsed.registered_modules
    )


def _run_input_checks(parsed: ParsedConfig) -> int:
    """Run the configured input-compatibility checks before the DAG is built.

    Returns the number of checks run (0 if none configured). Under ``[subset]``
    the checks operate on a pixel slice rather than the full domain, so they are
    skipped with a warning recommending a full-domain dry run. A failure raises
    `catena.input_checks.InputCheckError`.
    """
    if not parsed.checks:
        return 0
    if parsed.subset_spec is not None:
        warnings.warn(
            "input checks skipped under [subset]; run `catena run --dry-run` on "
            "the full domain to validate them",
            stacklevel=2,
        )
        return 0
    from .input_checks import run_input_checks
    from .io import load_raw_datasets

    run_input_checks(load_raw_datasets(parsed.input_specs), parsed.checks)
    return len(parsed.checks)


def _config_provenance(config_file: Path | None) -> dict[str, str]:
    """Config text + its SHA-256, stamped onto outputs so a store is self-describing.

    A `ParsedConfig` carries no source path and no round-trippable text, so a run
    driven from an in-memory config stamps nothing rather than stamping something
    that cannot be trusted to reproduce it.
    """
    if config_file is None:
        return {}
    text = Path(config_file).read_text()
    return {
        "catena_config": text,
        "catena_config_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }


def _resolve_cache(
    config_cache: CacheSpec | None,
    cache_flag: bool | None,
    cache_dir: Path | None,
) -> CacheSpec | None:
    """Combine the config's [cache] spec with caller overrides.

    ``cache=False`` always wins. ``cache=True`` or a ``cache_dir`` enable caching
    with defaults when the config has no [cache] section.
    """
    if cache_flag is False:
        return None
    spec = config_cache
    if cache_flag is True and spec is None:
        spec = CacheSpec()
    if cache_dir is not None:
        spec = replace(spec or CacheSpec(), path=str(cache_dir))
    return spec
