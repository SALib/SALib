###Contributing

Thanks to all those who contributed so far, including: [Will Usher](https://github.com/willu47), [Chris Mutel](https://github.com/cmutel), [Matt Woodruff](https://github.com/matthewjwoodruff), [Fernando Rios](https://github.com/zoidy), and [Dan Hyams](https://github.com/dhyams), [xantares](https://github.com/xantares). The library would be in much worse shape without these efforts.

In my view, development should follow a few objectives:

* Maintain the decoupled sample/analysis workflow, including the command-line interface. People may not have their models written in Python, so the ability to work from data files rather than Python objects is important. Sending parameter samples to the model should be left to the user.

* Include only published methods. There is a separate branch that includes experimental methods, but these shouldn't be included in the master branch.

* Include citations to specific implementations. There are lots of different ways to calculate Sobol indices, for example. When there isn't an obvious "right way", we'll need citations to justify the choice.

* We've had some discussions about moving to an OOP setup. At this point, it would require a pretty big refactor for all of the methods. It's tough to group the methods into classes when they're decoupled, and preserving state would force people to learn a whole API. The current functional(ish) approach is easier to test, and users can do whatever they want with the numpy matrices. That's my preference for now, unless someone sees a clear way to restructure everything into classes.

* Write tests! We're using Travis CI and Coveralls. See the `SALib/tests` directory for examples. Please try to provide tests for new methods (and the existing tests can be improved, too!)

Here are a few ideas for new methods/features to add:

* "groups" functionality for Sobol. @willu47 added a really nice feature for the Morris method to read groups from the parameter file, and it would be great to extend this to other methods that can handle groups. For Sobol, the biggest difference would be in the cross-sampling.

* "template" functionality for all methods, to put parameter samples into a model input file (like lots of older Fortran/C++ models have). Here is an example: https://gist.github.com/jdherman/fe0b81100ee83e1f6532. If you had 1000 samples, this would create 1000 copies of the model input file, with the variable names substituted with values.

* Methods to consider adding: [PAWN](http://www.sciencedirect.com/science/article/pii/S1364815215000237), and [Distance-Based Generalized Sensitivity Analysis](http://link.springer.com/article/10.1007/s11004-014-9530-5)

* Lots of the existing methods could be vectorized for speed. I wrote several of these before I learned the full power of numpy.

Thanks again!



