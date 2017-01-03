###Contributing

Admins: [Will Usher](https://github.com/willu47) and [Jon Herman](https://github.com/jdherman)

Thanks to all those who contributed so far, including: [Chris Mutel](https://github.com/cmutel), [Bernardo Trindade](https://github.com/bernardoct), [Dave Hadka](https://github.com/dhadka), [Matt Woodruff](https://github.com/matthewjwoodruff), [Fernando Rios](https://github.com/zoidy), and [Dan Hyams](https://github.com/dhyams), [xantares](https://github.com/xantares). The library would be in much worse shape without these efforts.

Development should follow a few objectives:

* Maintain the decoupled sample/analysis workflow, including the command-line interface. People may not have their models written in Python, so the ability to work from data files rather than Python objects is important. Sending parameter samples to the model should be left to the user.

* Include only published methods. There is a separate branch that includes experimental methods, but these shouldn't be included in the master branch.

* Include citations to specific implementations. There are lots of different ways to calculate Sobol indices, for example. When there isn't an obvious "right way", we'll need citations to justify the choice.

* We've had some discussions about moving to an OOP setup. At this point, it would require a pretty big refactor for all of the methods. It's tough to group the methods into classes when they're decoupled, and preserving state would force people to learn a whole API. The current functional(ish) approach is easier to test, and users can do whatever they want with the numpy matrices. That's my preference for now, unless someone sees a clear way to restructure everything into classes.

* Write tests! We're using Travis CI and Coveralls. See the `SALib/tests` directory for examples. Please try to provide tests for new methods (and the existing tests can be improved, too!)

We use Github issues to track ideas for [new methods](https://github.com/SALib/SALib/labels/add_method) and [enhancements](https://github.com/SALib/SALib/labels/enhancement).  If you are looking to contribute new methods to the library, check the labels for inspiration or to check you are not duplicating another's work.

Thanks again!
