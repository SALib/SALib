# Developers Guide


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
