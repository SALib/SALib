# Developers Guide

## Project setup

For new developers: `git` must be installed.

The project and its dependencies are managed via `hatch` and `uv`.

While it is possible maintain a local version of this project using venv/pip/mamba/conda
to simplify this guide, `uv` and `hatch` is the focus.

Install `uv` following instructions from [here](https://docs.astral.sh/uv/getting-started/installation/).

```bash
$ uv venv .salib
$ source .salib\Scripts\activate
```

On Windows:

```bash
> uv venv .salib
> .salib\Scripts\activate.bat
```

Then install `hatchling`:

```bash
$ uv pip install hatchling
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
$ source .salib\Scripts\activate
```

Or on Windows:

```bash
> .salib\Scripts\activate.bat
```

Then install a locally editable version of SALib.

```bash
$ uv pip install -e .

# You may also want the ipython and ipykernel
$ uv pip install ipykernel ipython

# Install pre-commit hook support
$ uv pip install pre-commit
$ pre-commit install
```

## Prior to submitting a PR

Run the below to catch any formatting issues.

```bash
pre-commit run --all-files
```

## Running tests

To run tests, run the following command from the SALib project directory.

```bash
$ uv pip install pytest
$ uv run pytest
```

## Building documentation locally

Notes here assumes you are at the root of the project directory.

Install dependencies for documentation

```bash
$ uv pip install -e .[doc]
```

### On *nix

```bash
$ cd docs
$ make html
```

### On Windows

In a command prompt

```bash
> cd docs
> sphinx-build . ./html
```
