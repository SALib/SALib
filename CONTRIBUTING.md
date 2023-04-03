# Contributing

Admins: [Will Usher](https://github.com/willu47) and [Jon Herman](https://github.com/jdherman)

Thanks to all those who have [contributed so far](https://github.com/SALib/SALib/graphs/contributors)!

## Asking Questions, Reporting Bugs

We use GitHub issues to keep track of bugs and to answer questions about the use
of the library.

For bugs, [create a new issue](https://github.com/SALib/SALib/issues/new)
on GitHub. Use this to describe the nature of the bug and
the conditions needed to recreate it, including operating system and Python version.

If you have a question on interpretation of results, then we may be able to help.

We cannot answer specific implementation questions (such as 'how do I run my model
with SALib?')

You can format your questions using GitHub Markdown, which makes it easy to [paste in short snippets of code](https://help.github.com/articles/creating-and-highlighting-code-blocks/). If including a very long Python error traceback, please use a GitHub [gist](https://gist.github.com/).

## Contributing Code

To contribute new code, submit a pull request. There are two instances in which you may want to contribute code: to fix a bug, or to add a new feature, such as a new sensitivity analysis method.

### Making a development environment

Note: We **strongly** recommend using a virtual environment setup, such as
`venv` or `conda`.

First, fork a copy of the main SALib repository in GitHub onto your own
account and then create your local repository via:

```sh
git clone git@github.com:YOURUSERNAME/SALib.git
cd SALib
```

Next, set up your development environment.

With `conda` installed (through
[Miniforge or Mambaforge](https://github.com/conda-forge/miniforge),
[Miniconda](https://docs.conda.io/en/latest/miniconda.html) or
[Anaconda](https://www.anaconda.com/products/individual)), execute the
following commands at the terminal from the base directory of your
[SALib](https://github.com/SALib/SALib) clone:

```sh
# Create an environment with all development dependencies
conda env create -f environment.yml  # works with `mamba` too
# Activate the environment
conda activate SALib
```

Finally, you can install SALib in editable mode in your environment:

```sh
pip install -e .
```

### Fixing a Bug

First, create a new issue on GitHub with the label `bug`. Use this to describe the nature of the bug and the conditions needed to recreate it.

Then, please create a new branch with the name `bug_xxx` where xxx is the number of the issue.

If possible, write a test which reproduces the bug. The tests are stored in the `SALib/tests/` folder, and are run using `pytest` from the root folder of the library.
You can run the tests with the command `pytest` from the root project directory. Individual tests can be run with by specifying a file, or file and test function.

For example:

```bash
$ pytest  # run all tests
$ pytest tests/test_file.py  # run tests within a specific file
$ pytest tests/test_file.py::specific_function  # run a specific test
```

Then, fix the bug in the code so that the test passes.

Submit a pull request with a descriptive title and reference the issue in the text. Once a pull request is submitted, the tests will run on Travis CI. If these tests pass, we will review and merge in your changes.

### Adding a new method

Methods in SALib follow a decoupled sample/analysis workflow. In other words, the generation of parameter samples and the calculation of sensitivity indices can be performed in two separate steps. This is because many users have models in languages other than Python, so sending data to/from the model is left to the user. All methods should support a command-line interface on top of the Python functions.

To add a new method, create an issue on GitHub with the label `add_method`, if one does not already exist. Please describe the method, and link to the peer reviewed article in which the method is described. The master branch should only contain published methods. First check the current [open issues with this label](https://github.com/SALib/SALib/labels/add_method) for inspiration or to see if someone is already working on a certain method.

We use GitHub issues to track ideas for  and [enhancements](https://github.com/SALib/SALib/labels/enhancement).  If you are looking to contribute new methods to the library, check the labels for inspiration or to ensure you are not duplicating another's work.

Then, create a new branch with a useful name, such as `new_method_method_name`. Methods should consist of:

* A sampling module (a `method_name.py` file in `SALib.sample`). This will contain, among other things, a function `sample(problem, ...)` that accepts a problem dictionary and returns a numpy array of samples, one column for each parameter. See [SALib.sample.saltelli](https://github.com/SALib/SALib/blob/main/SALib/sample/saltelli.py) for an example.

* An analysis module (a `method_name.py` file in `SALib.analyze`). This will contain a function analyze(problem, ...) that returns a dictionary of sensitivity indices. See [SALib.analyze.sobol](https://github.com/SALib/SALib/blob/main/SALib/analyze/sobol.py) for an example.

* An example shell script and python file in the `examples` folder, ideally using a test function included in SALib such as the Ishigami or Sobol-G functions.

* Docstrings for the `sample` and `analyze` functions that include citations. Please add an entry to `docs/index.rst` to add your method documentation to the concise API reference.

* All contributed methods should also provide functions to support their use through the command line interface (CLI). These are `cli_parse()` and `cli_action()` to parse command line options and to run the sampling and analysis respectively. See the implementations in [SALib.analyze.delta](https://github.com/SALib/SALib/blob/consolidate-cli/SALib/analyze/delta.py) for an example.

* Tests in the `tests` folder.  We're using Travis CI and Coveralls. Ideally, every new function will have one or more corresponding tests to check that errors are raised for invalid inputs, and that functions return matrices of the proper sizes. (For example [see here](https://github.com/SALib/SALib/blob/main/tests/test_sobol.py). But at a minimum, please include a regression test for the Ishigami function, in the same format as all of the other methods [see here](https://github.com/SALib/SALib/blob/main/tests/test_regression.py). This will at least ensure that future updates don't break your code!

Finally, submit a pull request. Either @willu47 or @jdherman will review the pull request and merge in your changes.


### Other Enhancements

Contributions not related to new methods are also welcome. These might include new test functions (see [SALib.test_functions](https://github.com/SALib/SALib/tree/main/SALib/test_functions) for how these are set up), or other code that is general across some or all of the methods. This general code is currently included in [SALib.util.\_\_init\_\_.py](https://github.com/SALib/SALib/blob/main/SALib/util/__init__.py).


### Other Development Comments

Most of the sampling techniques make heavy use of pseudo-random number
generators.
We use primarily `numpy.random` as the python standard library
`random` library is inconsistent across Python 2 and 3.
When writing tests for methods which use these random number generators, set the seeds using `numpy.random.seed(SEED)`
where `SEED` is a fixed integer.
This will ensure that your tests are repeatable.

### Notes about scope

* SALib contains a few basic types of plots, especially for the Morris method. Indicative results can be made by calling the [`.plot()` method](https://salib.readthedocs.io/en/main/basics.html#basic-plotting)
* However, we generally assume that plot types and styles are left to the user, as these are often application-specific. Users interested in more complex plot types should check out the [savvy](https://github.com/houghb/savvy) library, which is built on top of SALib.

Thanks again!

## Making a release

Following is the process that the development team follows in order to make
a release:

1. Document overview of changes since last release in `CHANGELOG.MD`
2. Update the version in the main `__init__.py`.
3. Build locally using `hatch build`, and verify the content of the artifacts
4. Submit PR, wait for tests to pass, and merge release into `main`
5. Tag release with version number and push to SALib repo
6. Check that release has been deployed to PyPI
7. Check documentation is built and deployed to readthedocs (http://salib.readthedocs.org)
8. Check that auto-generated PR is auto-merged on the conda-forge feedstock repo (conda-forge/salib-feedstock)
9. Update development roadmap on GitHub

## Building a local copy of the documentation

Assuming the current location is the project root (the `SALib` directory):

```bash
$ conda install pydata-sphinx-theme myst-parser -c conda-forge
$ sphinx-build -b html docs docs/html
```

A copy of the documentation will be in the `docs/html` directory.
Open `index.html` to view it.
