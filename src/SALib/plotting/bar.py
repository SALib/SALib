
__all__ = ['plot']

# magic string indicating DF columns holding conf bound values
CONF_COLUMN = '_conf'

def plot(Si_df, ax=None):
    '''Create bar chart of results

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
    >>> X = saltelli.sample(problem, 1000)
    >>> Y = Ishigami.evaluate(X)
    >>> Si = sobol.analyze(problem, Y, print_to_console=False)
    >>> Si_df = Si.to_df()
    >>> barplot(Si_df)
    '''
    conf_cols = Si_df.columns.str.contains(CONF_COLUMN)

    confs = Si_df.loc[:, conf_cols]
    confs.columns = [c.replace(CONF_COLUMN, '') for c in confs.columns]

    Sis = Si_df.loc[:, ~conf_cols]

    ax = Sis.plot(kind='bar', yerr=confs, ax=ax)
    return ax