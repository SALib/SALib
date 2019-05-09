# Contributing

Admins: [Will Usher](https://github.com/willu47) and [Jon Herman](https://github.com/jdherman)

Thanks to all those who have [contributed so far](https://github.com/SALib/SALib/graphs/contributors)!

## Asking Questions, Reporting Bugs

We use Github issues to keep track of bugs and to answer questions about the use
of the library.  

For bugs, [create a new issue](https://github.com/SALib/SALib/issues/new)
on Github with the label `bug`. Use this to describe the nature of the bug and
the conditions needed to recreate it, including operating system and Python version.

If you have a question on interpretation of results, than we may be able to help.
Again, create an issue with your question and use the label `question_interpretation`.

We cannot answer specific implementation questions (such as 'how do I run my model
with SALib?')

You can format your questions using Github Markdown, which makes it easy to [paste in short snippets of code](https://help.github.com/articles/creating-and-highlighting-code-blocks/). If including a very long Python error traceback, please use a Github [gist](https://gist.github.com/).

## Contributing Code

To contribute new code, submit a pull request. There are two instances in which you may want to contribute code: to fix a bug, or to add a new feature, such as a new sensitivity analysis method.

### Fixing a Bug

First, create a new issue on Github with the label `bug`. Use this to describe the nature of the bug and the conditions needed to recreate it.

Then, please create a new branch with the name `bug_xxx` where xxx is the number of the issue.

If possible, write a test which reproduces the bug. The tests are stored in the `SALib/tests/` folder, and are run using `pytest`.  You can run the tests with the command `python setup.py test` from the root folder of the library.

Then, fix the bug in the code so that the test passes.

Submit a pull request with a descriptive title and reference the issue in the text. Once a pull request is submitted, the tests will run on Travis CI. If these tests pass, we will review and merge in your changes.

### Adding a new method

Methods in SALib follow a decoupled sample/analysis workflow. In other words, the generation of parameter samples and the calculation of sensitivity indices can be performed in two separate steps. This is because many users have models in languages other than Python, so sending data to/from the model is left to the user. All methods should support a command-line interface on top of the Python functions.

To add a new method, create an issue on Github with the label `add_method`, if one does not already exist. Please describe the method, and link to the peer reviewed article in which the method is described. The master branch should only contain published methods. First check the current [open issues with this label](https://github.com/SALib/SALib/labels/add_method) for inspiration or to see if someone is already working on a certain method.

We use Github issues to track ideas for  and [enhancements](https://github.com/SALib/SALib/labels/enhancement).  If you are looking to contribute new methods to the library, check the labels for inspiration or to ensure you are not duplicating another's work.

Then, create a new branch with a useful name, such as `new_method_method_name`. Methods should consist of:

* A sampling module (a `method_name.py` file in `SALib.sample`). This will contain, among other things, a function `sample(problem, ...)` that accepts a problem dictionary and returns a numpy array of samples, one column for each parameter. See [SALib.sample.saltelli](https://github.com/SALib/SALib/blob/master/SALib/sample/saltelli.py) for an example.

* An analysis module (a `method_name.py` file in `SALib.analyze`). This will contain a function analyze(problem, ...) that returns a dictionary of sensitivity indices. See [SALib.analyze.sobol](https://github.com/SALib/SALib/blob/master/SALib/analyze/sobol.py) for an example.

* An example shell script and python file in the `examples` folder, ideally using a test function included in SALib such as the Ishigami or Sobol-G functions.

* Docstrings for the `sample` and `analyze` functions that include citations. Please add an entry to `docs/index.rst` to add your method documentation to the concise API reference.

* All contributed methods should also provide functions to support their use through the command line interface (CLI). These are `cli_parse()` and `cli_action()` to parse command line options and to run the sampling and analysis respectively. See the implementations in [SALib.analyze.delta](https://github.com/SALib/SALib/blob/consolidate-cli/SALib/analyze/delta.py) for an example.

* Tests in the `tests` folder.  We're using Travis CI and Coveralls. Ideally, every new function will have one or more corresponding tests to check that errors are raised for invalid inputs, and that functions return matrices of the proper sizes. (For example [see here](https://github.com/SALib/SALib/blob/master/tests/test_sobol.py). But at a minimum, please include a regression test for the Ishigami function, in the same format as all of the other methods [see here](https://github.com/SALib/SALib/blob/master/tests/test_regression.py). This will at least ensure that future updates don't break your code!

Finally, submit a pull request. Either @willu47 or @jdherman will review the pull request and merge in your changes.

### Other Enhancements

Contributions not related to new methods are also welcome. These might include new test functions (see [SALib.test_functions](https://github.com/SALib/SALib/tree/master/SALib/test_functions) for how these are set up), or other code that is general across some or all of the methods. This general code is currently included in [SALib.util.\_\_init\_\_.py](https://github.com/SALib/SALib/blob/master/SALib/util/__init__.py).


### Other Development Comments

Most of the sampling techniques make heavy use of pseudo-random number
generators.
We use primarily `numpy.random` as the python standard library
`random` library is inconsistent across Python 2 and 3.
When writing tests for methods which use these random number generators, set the seeds using `numpy.random.seed(SEED)`
where `SEED` is a fixed integer.
This will ensure that your tests are repeatable.

### Notes about scope

* We've had some discussions about moving to an OOP setup. At this point, it would require a pretty big refactor for all of the methods. It's tough to group the methods into classes when they're decoupled, and preserving state would force people to learn a whole API. The current functional(ish) approach is easier to test, and users can do whatever they want with the numpy matrices. That's our preference for now, unless someone sees a clear way to restructure everything into classes. Further discussion on this can be found [here](https://github.com/SALib/SALib/issues/216#issuecomment-435647632).



* SALib contains a few basic types of plots, especially for the Morris method. However, we generally assume that plot types and styles are left to the user, as these are often application-specific. Users interested in more complex plot types should check out the [savvy](https://github.com/houghb/savvy) library, which is built on top of SALib.

Thanks again!
