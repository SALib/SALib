==========================
Frequently Asked Questions
==========================


Q. How do I wrap my model?
---------------

See this guide on :doc:`wrapping models </user_guide/wrappers>`.


Q. Which technique can I use for pre-existing results?
------------------------------------------------------

DMIM, RBD-FAST, PAWN and HDMR methods are "given-data" approaches and can be independently applied.


Q. How do I get the sensitivity results?
----------------------------------------

Call the :code:`.to_df()` method if you would like Pandas DataFrames.

If using the SALib Interface, see :doc:`SALib Interface Basics </user_guide/basics_with_interface>`.


Q. How can I plot my results?
-----------------------------

SALib provides some basic plotting functionality. See the subsection "Basic Plotting" in :doc:`SALib Interface Basics </user_guide/basics_with_interface>`

See also, `these examples <https://github.com/SALib/SALib/tree/main/examples/plotting>`_.


Q. Why does the Sobol' method implemented in SALib normalize outputs?
---------------------------------------------------------------------

Estimates of Sobol' indices can be biased in cases where model outputs are non-centered.
We have opted to normalize outputs with the standard deviation.

See the discussion `here <https://github.com/SALib/SALib/issues/109#issuecomment-268499001>`_.

In practice, non-normalized model outputs are still usable but requires larger sample sizes for
the indices to converge.


Q. Why is ``sigma`` ``NaN`` when running Morris analysis with groups?
---------------------------------------------------------------------

In grouped Morris sampling, all parameters belonging to a group are perturbed
together within each trajectory step using independent ``+delta`` or
``-delta`` step signs. For groups with two or more parameters, co-perturbed
effects can reinforce or cancel across trajectories, leaving signed group
effects (``mu``) and their standard deviation (``sigma``) undefined. SALib
returns ``NaN`` for multi-member groups so plots (such as
``SALib.plotting.morris.covariance_plot``) and tables do not mistake an
undefined value for zero interaction. Single-member groups follow ungrouped
sampling behavior and still report a numeric ``sigma``. Use ``mu_star`` to
rank multi-member groups.
