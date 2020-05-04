from multiprocess import Pool, cpu_count
from functools import partial
import numpy as np

from .results import ResultDict

ptqdm_available = True
try:
    from p_tqdm import p_imap
except ImportError:
    ptqdm_available = False


class ProblemSpec(dict):
    '''Dictionary holding problem specifications.

    '''
    def __init__(self, *args, **kwargs):
        super(ProblemSpec, self).__init__(*args, **kwargs)

        self._samples = None
        self._model_output = None
        self._analysis = None

        self['num_vars'] = len(self['names'])

    def sample(self, func, *args, **kwargs):
        """Create sample using given function.
        
        Sampling function must accept the problem spec as the first parameter.
        """
        self._samples = func(self, *args, **kwargs)
        # Clear model output and analysis results as well?
        # might avoid confusion if the samples are forcibly changed...

        return self

    def run(self, func, *args, **kwargs):
        """Run a given model.

        The target model must accept the samples as the first parameter.
        """
        self._model_output = func(self._samples, *args, **kwargs)

        return self
    
    def run_parallel(self, func, *args, **kwargs):

        # Create wrapped partial function to allow passing of additional args
        partial_f = partial(func, *args, **kwargs)
        def tmp_f(X_i):
            """Helper function to run each sample independently"""
            return partial_f(X_i)

        # Split into even chunks
        nprocs = cpu_count()
        chunks = np.array_split(self._samples, int(nprocs), axis=0)

        final_res = np.array([])
        if ptqdm_available:
            # Display progress bar
            res = p_imap(tmp_f, chunks, num_cpus=nprocs)
            for r in res:
                final_res = np.concatenate((final_res, r))
        else:
            with Pool(nprocs) as pool:
                res = pool.imap(tmp_f, chunks)
                for r in res:
                    final_res = np.concatenate((final_res, r))
        
        self._model_output = final_res

        return self

    def analyze(self, func, *args, **kwargs):
        """Analyze sampled results using given function.

        Analysis function must accept the problem spec as the first parameter.
        """
        # TODO: Analyze function signature to see if X and Y are both needed...
        if len(self['outputs']) > 1:
            self._analysis = {}
            for i, out in enumerate(self['outputs']):
                self._analysis[out] = func(self, self._model_output[:, i], *args, **kwargs)
        else:
            self._analysis = func(self, self._model_output, *args, **kwargs)

        return self

    def to_df(self):
        an_res = self._analysis
        is_rd_type = isinstance(an_res, ResultDict)
        if is_rd_type:
            return an_res.to_df()
        elif isinstance(an_res, dict):
            # is a dict of ResultDicts
            return [an.to_df() for an in list(an_res.values())]
        
        raise RuntimeError("Analysis not yet conducted")

    def __repr__(self):
        if self._samples is not None:
            print('Samples:', self._samples.shape, "\n")
        if self._model_output is not None:
            print('Outputs:', self._model_output.shape, "\n")
        if self._analysis is not None:
            print('Analysis:\n')  # , self._model_output.shape, "\n")
            an_res = self._analysis
            if isinstance(an_res, ResultDict):
                for df in an_res.to_df():
                    print(df, '\n')
            elif isinstance(an_res, dict):
                for vals in list(an_res.values()):
                    for df in vals.to_df():
                        print(df, "\n")
        return ''
    