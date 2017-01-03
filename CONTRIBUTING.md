# Contributing

Admins: [Will Usher](https://github.com/willu47) and [Jon Herman](https://github.com/jdherman)

Thanks to all those who contributed so far, including: [Chris Mutel](https://github.com/cmutel), [Bernardo Trindade](https://github.com/bernardoct), [Dave Hadka](https://github.com/dhadka), [Matt Woodruff](https://github.com/matthewjwoodruff), [Fernando Rios](https://github.com/zoidy), and [Dan Hyams](https://github.com/dhyams), [xantares](https://github.com/xantares). The library would be in much worse shape without these efforts.

## General Objectives

Development should follow a few objectives:

* Maintain the decoupled sample/analysis workflow, including the command-line interface. People may not have their models written in Python, so the ability to work from data files rather than Python objects is important. Sending parameter samples to the model should be left to the user.

* Include only published methods. There is a separate branch that includes experimental methods, but these shouldn't be included in the master branch.

* Include citations to specific implementations. There are lots of different ways to calculate Sobol indices, for example. When there isn't an obvious "right way", we'll need citations to justify the choice.

* We've had some discussions about moving to an OOP setup. At this point, it would require a pretty big refactor for all of the methods. It's tough to group the methods into classes when they're decoupled, and preserving state would force people to learn a whole API. The current functional(ish) approach is easier to test, and users can do whatever they want with the numpy matrices. That's our preference for now, unless someone sees a clear way to restructure everything into classes.

* Write tests! We're using Travis CI and Coveralls. See the `SALib/tests` directory for examples. Please try to provide tests for new methods (and the existing tests can be improved, too!)

## Contributing Code

If you wish to contribute some code you have written, we use pull requests to manage the process.  There are two instances in which you may contribute code.  One is to fix a bug, and the second is to add a new sensitivity analysis method, or some other new feature.

### Fixing a Bug

First, create a new issue on Github with the label `bug`. Use this to describe the nature of the bug and the conditions needed to recreate it.

Then, please create a new branch with the name `bug_xxx` where xxx is the number of the issue.

Then, write a test which reproduces the bug. The tests are stored in the `SALib/tests/` folder, and are run using pytest.  You can run the tests with the command `python setup.py test` from the root folder of the library.

Then, fix the bug in the code so that the test passes.

Submit a pull request with a descriptive title and reference the issue in the text. Once a pull request is submitted, our test harness will run on Travis CI. If these tests pass, then we will review and merge in your changes.

### Adding a new method

First, create a new issue on Github with the label `add_method` (if one does not already exist). Please describe the method, and link to the peer reviewed article in which the method is described.

Then, create a new branch with a useful name, such as `new_method_method_name`. Methods usually consist of a sample module (a `method_name.py` file in `SALib.sample`) and an analysis module (a `method_name.py` file in `SALib.analyze`). In addition, create an example shell script and python file in the `examples` folder, tests in the `tests` folder, and use docstrings to document your procedures. You will need to add an entry to `docs/index.rst` to add your method documentation to the concice API entry.

Finally, submit a pull request. Either @willu47 or @jdherman will review the pull request and merge in your changes.

## New Methods and Enhancements

We use Github issues to track ideas for [new methods](https://github.com/SALib/SALib/labels/add_method) and [enhancements](https://github.com/SALib/SALib/labels/enhancement).  If you are looking to contribute new methods to the library, check the labels for inspiration or to ensure you are not duplicating another's work.

Thanks again!
