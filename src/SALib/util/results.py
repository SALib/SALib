import pandas as pd
from SALib.plotting.bar import plot as barplot

class ResultDict(dict):
    '''Dictionary holding analysis results.

    Conversion methods (e.g. to Pandas DataFrames) to be attached as necessary
    by each implementing method
    '''
    def __init__(self, *args, **kwargs):
        super(ResultDict, self).__init__(*args, **kwargs)

    def to_df(self):
        '''Convert dict structure into Pandas DataFrame.'''
        return pd.DataFrame({k: v for k, v in self.items() if k is not 'names'},
                            index=self['names'])

    def plot(self):
        '''Create bar chart of results'''
        Si_df = self.to_df()

        if isinstance(Si_df, (list, tuple)):
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, len(Si_df))
            for idx, f in enumerate(Si_df):
                axes[idx] = barplot(f, ax=axes[idx])

        else:
            axes = barplot(Si_df)

        return axes
