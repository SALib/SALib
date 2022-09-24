__all__ = ["plot"]

# magic string indicating DF columns holding conf bound values
CONF_COLUMN = "_conf"


def plot(Si_df, ax=None):
    """Create bar chart of results.

    Parameters
    ----------
    * Si_df: pd.DataFrame, of sensitivity results

    Returns
    ----------
    * ax : matplotlib axes object

    Examples
    ----------
    >>> from SALib.plotting.bar import plot as barplot
    >>> from SALib.test_functions import Ishigami
    >>>
    >>> # See README for example problem specification
    >>>
    >>> X = saltelli.sample(problem, 512)
    >>> Y = Ishigami.evaluate(X)
    >>> Si = sobol.analyze(problem, Y, print_to_console=False)
    >>> total, first, second = Si.to_df()
    >>> barplot(total)
    """
    conf_cols = Si_df.columns.str.contains(CONF_COLUMN)

    confs = Si_df.loc[:, conf_cols]
    confs.columns = [c.replace(CONF_COLUMN, "") for c in confs.columns]

    Sis = Si_df.loc[:, ~conf_cols]

    ax = Sis.plot(kind="bar", yerr=confs, ax=ax)
    return ax
