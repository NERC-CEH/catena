_: lint typecheck test

# Format and lint the package using ruff.
lint:
  ruff format
  ruff check --fix

# Variant of `lint` that doesn't cause any changes to files.
lint-check:
  ruff format --check
  ruff check

# Run static type checker.
typecheck:
  pyright

# Run the full test suite.
test:
  pytest --verbose # --log-cli-level=INFO

# Run tests with coverage report.
test-cov:
  pytest --cov=catena --cov-report=term-missing --cov-fail-under=90

# Export executable notebooks and build the documentation using Zensical.
docs: docs-recipes
  zensical build

# Export the executable recipe notebooks with their rendered outputs.
docs-recipes:
  marimo-md-export recipes/pipeline_101/demo.py docs/recipes/pipeline-101.md
  marimo-md-export recipes/flux_pipeline/demo.py docs/recipes/flux-pipeline.md

# Tag the version in pyproject.toml and push the tag.
tag:
  #!/usr/bin/env bash
  set -euo pipefail
  v=$(uv version --short)
  [[ $v != *dev* ]] || { echo "refusing to tag dev version $v" >&2; exit 1; }
  [[ $(git branch --show-current) == main ]] || { echo "not on main" >&2; exit 1; }
  git diff --quiet HEAD || { echo "uncommitted changes" >&2; exit 1; }
  git fetch --quiet origin main
  [[ $(git rev-parse HEAD) == $(git rev-parse origin/main) ]] \
    || { echo "main differs from origin/main" >&2; exit 1; }
  git tag -a "v$v" -m "v$v"
  git push origin "v$v"

# Create a GitHub release for the tagged version, deploying the docs if stable.
release:
  #!/usr/bin/env bash
  set -euo pipefail
  v=$(uv version --short)
  git rev-parse --quiet --verify "refs/tags/v$v" >/dev/null \
    || { echo "no tag v$v: run 'just tag' first" >&2; exit 1; }
  pre=()
  [[ ! $v =~ (a|b|rc)[0-9]+$ ]] || pre=(--prerelease)
  gh release create "v$v" --verify-tag --generate-notes "${pre[@]}"
