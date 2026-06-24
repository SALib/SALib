# Developers Guide

## Project setup

For new developers: `git` must be installed.

The project and its dependencies are managed via `hatch` and `uv`.

While it is possible to maintain a local version of this project using venv/pip/mamba/conda,
to simplify this guide, `uv` and `hatch` is the focus.

Install `uv` following instructions from [here](https://docs.astral.sh/uv/getting-started/installation/).

```bash
$ uv venv                    # equivalent: python -m venv .venv
$ .venv/Scripts/activate
```

On Windows, the commands above should work for those using PowerShell.

Otherwise, run the below for classic command prompt:

```bash
> uv venv
> .venv/Scripts/activate.bat
```

Then install `hatch` as a tool:

```bash
$ uv tool install hatch      # equivalent: pip install --user hatch (but isolated)
```

Verify installation after restarting terminal with:

```bash
$ hatch --version
```

Build SALib with:

```bash
$ hatch build
```

## Setup (with existing Python installation)

Activate the project environment with:

```bash
$ source .venv\Scripts\activate
```

Or on Windows:

```bash
> .venv\Scripts\activate.bat
```

Then install SALib and its dependencies. Here, additional development dependencies are
also installed.

```bash
$ uv sync                      # equivalent: pip install -e .
$ uv sync --extra dev          # equivalent: pip install -e ".[dev]"

# Install pre-commit as a tool and set up the hook
$ uv tool install pre-commit   # equivalent: pip install --user pre-commit (but isolated)
$ pre-commit install

# To run tests:
$ uv run pytest    # equivalent: pytest (but runs within the uv-managed environment)
```

## Prior to submitting a PR

Run the below to catch any formatting issues.

```bash
pre-commit run --all
```

## Building documentation locally

Notes here assume you are at the root of the project directory.

Install dependencies for documentation:

```bash
$ uv sync --extra doc    # equivalent: pip install -e ".[doc]"
```

### On *nix

```bash
$ cd docs
$ make html
```

### On Windows

In a command prompt:

```bash
> cd docs
> sphinx-build . ./html
```
