from SALib.analyze import delta
import numpy as np
import pandas as pd


df = pd.DataFrame() # read in dataframe here
col_output = 'title_outcol' # column name in df which is output Y
bootstrap_fn = "fn_bootstrap_inspect.parquet" # parquet file to inspect bootstrap subset

Y = np.asarray(df[col_output].values)
cols_input = [col for col in df.columns if col!=col_output]
X = df[cols_input].values

problem = {
    'num_vars': len(cols_input),
    'names': cols_input,
    'bounds': [[X[:, i].min(), X[:, i].max()] for i in range(len(cols_input))]
}

Si = delta.analyze(problem, X, Y, print_to_console=True, bootstrap_savedf=bootstrap_fn)
