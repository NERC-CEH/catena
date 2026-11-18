---
title: Installation
icon: lucide/download
---

# Installation

catena needs Python 3.12 or later, and is installed from GitHub rather than PyPI.
It installs as `ukceh-catena` and imports as `catena`.

## Install

=== "pip"

    ```sh
    pip install git+https://github.com/NERC-CEH/catena
    ```

=== "uv"

    ```sh
    uv add git+https://github.com/NERC-CEH/catena
    ```

This installs the library and its Python API from the latest `main`.
To install a specific release, append its tag to the URL, e.g. `git+https://github.com/NERC-CEH/catena@v0.4.0`.
The `catena` command and graph rendering are optional extras.

## Extras

| Extra | Installs | Needed for |
| --- | --- | --- |
| `cli` | `typer` | the `catena` command (`catena run`, `catena graph`, `catena subset`) |
| `viz` | `apache-hamilton[visualization]` | rendering the DAG with `catena graph` |
| `all` | everything above | every optional feature |

Name extras in square brackets:

=== "pip"

    ```sh
    pip install "ukceh-catena[cli] @ git+https://github.com/NERC-CEH/catena"
    pip install "ukceh-catena[all] @ git+https://github.com/NERC-CEH/catena"
    ```

=== "uv"

    ```sh
    uv add "ukceh-catena[cli] @ git+https://github.com/NERC-CEH/catena"
    uv add "ukceh-catena[all] @ git+https://github.com/NERC-CEH/catena"
    ```

## Graphviz

`catena graph` also needs the Graphviz system binaries, which the `viz` extra cannot install:

```sh
# Ubuntu/Debian
sudo apt install graphviz

# macOS
brew install graphviz
```

## Verify installation

```sh
python -c "import catena; print(catena.__version__)"
catena --version  # with the cli extra
```

Then try [Pipeline 101](../recipes/pipeline-101.md).
To set up a development checkout, see [`CONTRIBUTING.md`](https://github.com/NERC-CEH/catena/blob/main/CONTRIBUTING.md).
