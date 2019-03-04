import pandas as pd


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
