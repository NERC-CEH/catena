---
title: CLI
icon: lucide/square-terminal
---

The `catena` command ships in the `cli` extra.
Each subcommand wraps a [Python API](python-api.md) function, so anything below can also be done from Python.

## `catena run`

::: mkdocs-typer2
    :module: catena.cli.app
    :name: catena
    :command: run
    :termynal: true
    :width: 80
    :prompt: ❯

## `catena graph`

::: mkdocs-typer2
    :module: catena.cli.app
    :name: catena
    :command: graph
    :termynal: true
    :width: 80
    :prompt: ❯

## `catena subset`

Commands that bracket a set of parallel `[subset]` runs: create the shared store, then merge the parts.
See [Scale up](../guides/run/scale-up.md#parallel-subset-runs) for the workflow.

### `catena subset create-store`

::: mkdocs-typer2
    :module: catena.cli.app
    :name: catena
    :command: subset create-store
    :termynal: true
    :width: 80
    :prompt: ❯

### `catena subset merge`

::: mkdocs-typer2
    :module: catena.cli.app
    :name: catena
    :command: subset merge
    :termynal: true
    :width: 80
    :prompt: ❯
