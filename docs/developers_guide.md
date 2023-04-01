# Developers Guide


## Running tests

To run tests, run the following command from the SALib project directory.

```bash
$ pytest
```


## Building documentation locally

Notes here assumes you are at the root of the project directory.

Install dependencies for documentation

```bash
$ pip install -e .[doc]
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

## Prior to submitting a PR

Run the below to catch any formatting issues.

```bash
# pre-commit install

pre-commit run --all-files
```
