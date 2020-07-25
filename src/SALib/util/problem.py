import warnings
from multiprocess import Pool, cpu_count
from functools import partial, wraps
import importlib

import numpy as np

import SALib.sample as samplers
import SALib.analyze as analyzers

from SALib.util import avail_approaches
from SALib.util.results import ResultDict


ptqdm_available = True
try:
    from p_tqdm import p_imap
except ImportError:
    ptqdm_available = False

__all__ = ['ProblemSpec']


class ProblemSpec(dict):
    """Dictionary-like object representing an SALib Problem specification.
    """
    def __init__(self, *args, **kwargs):
        super(ProblemSpec, self).__init__(*args, **kwargs)

        self._samples = None
        self._results = None
        self._analysis = None

        self['num_vars'] = len(self['names'])

        self._add_samplers()
        self._add_analyzers()

    @property
    def samples(self):
        return self._samples

    @property
    def results(self):
        return self._results
    
    @property
    def analysis(self):
        return self._analysis

    def sample(self, func, *args, **kwargs):
        """Create sample using given function.
        
        Sampling function must accept the problem spec as the first parameter,
        and return a numpy array.

        Parameters
        ----------
        func : function,
            Sampling method to use.
        
        *args : list,
            Additional arguments to be passed to `func`
        
        **kwargs : dict,
            Additional keyword arguments passed to `func`

        Returns
        ----------
        self : ProblemSpec object
        """
        self._samples = func(self, *args, **kwargs)
        # Clear model output and analysis results as well?
        # might avoid confusion if the samples are forcibly changed...

        return self

    def evaluate(self, func, *args, **kwargs):
        """Evaluate a given model.

        Usage Conditions:
        * The provided function needs to accept a numpy array of inputs as 
          its first parameter
        * `func` must return a numpy array of results

        Parameters
        ----------
        func : function,
            Model, or function that wraps a model, to be run in parallel
        
        *args : list,
            Additional arguments to be passed to `func`
        
        **kwargs : dict,
            Additional keyword arguments passed to `func`

        Returns
        ----------
        self : ProblemSpec object
        """
        self._results = func(self._samples, *args, **kwargs)

        return self
    
    def evaluate_parallel(self, func, *args, nprocs=None, **kwargs):
        """Evaluate model in parallel.

        Usage Conditions:
        * The provided function needs to accept a numpy array of inputs as 
          its first parameter
        * The provided function must return a numpy array of results

        All detected processors will be used if `nprocs` is not specified.

        Parameters
        ----------
        func : function,
            Model, or function that wraps a model, to be run in parallel
        
        nprocs : int,
            Number of processors to use.
        
        *args : list,
            Additional arguments to be passed to `func`
        
        **kwargs : dict,
            Additional keyword arguments passed to `func`

        Returns
        ----------
        self : ProblemSpec object
        """
        warnings.warn("This is an experimental feature and may break.")

        if self._samples is None:
            raise RuntimeError("Sampling not yet conducted")
        
        if nprocs is None:
            nprocs = cpu_count()

        # Create wrapped partial function to allow passing of additional args
        tmp_f = func
        if (len(args) > 0) or (len(kwargs) > 0):
            tmp_f = partial(func, *args, **kwargs)

        # Split into even chunks
        chunks = np.array_split(self._samples, int(nprocs), axis=0)

        # Set up result array
        if len(self['outputs']) > 1:
            res_shape = (len(self._samples), len(self['outputs']))
        else:
            res_shape = len(self._samples)

        final_res = np.empty(res_shape)

        if ptqdm_available:
            # Display progress bar if available
            res = p_imap(tmp_f, chunks, num_cpus=nprocs)
        else:
            with Pool(nprocs) as pool:
                res = list(pool.imap(tmp_f, chunks))

        # Collect results
        i = 0
        for r in res:
            r_len = len(r)
            final_res[i:i+r_len] = r
            i += r_len

        self._results = final_res

        return self

    def analyze(self, func, *args, **kwargs):
        """Analyze sampled results using given function.

        Analysis function must accept the problem spec as the first parameter
        and return a numpy array.

        Parameters
        ----------
        func : function,
            Analysis method to use.
        
        *args : list,
            Additional arguments to be passed to `func`
        
        **kwargs : dict,
            Additional keyword arguments passed to `func`

        Returns
        ----------
        self : ProblemSpec object
        """
        if 'X' in func.__code__.co_varnames:
            # enforce passing of X if expected
            func = partial(func, *args, X=self._samples, **kwargs)

        if len(self['outputs']) > 1:
            self._analysis = {}
            for i, out in enumerate(self['outputs']):
                self._analysis[out] = func(self, *args, Y=self._results[:, i], **kwargs)
        else:
            self._analysis = func(self, *args, Y=self._results, **kwargs)

        return self

    def to_df(self):
        """Convert results to Pandas DataFrame."""
        an_res = self._analysis
        if isinstance(an_res, ResultDict):
            return an_res.to_df()
        elif isinstance(an_res, dict):
            # case where analysis result is a dict of ResultDicts
            return [an.to_df() for an in list(an_res.values())]
        
        raise RuntimeError("Analysis not yet conducted")

    def _add_samplers(self):
        """Dynamically add available SALib samplers as ProblemSpec methods."""
        for sampler in avail_approaches(samplers):
            func = getattr(importlib.import_module('SALib.sample.{}'.format(sampler)), 'sample')

            @wraps(func)
            def modfunc(*args, **kwargs):
                self.sample(func, *args, **kwargs)
                return self

            self.__setattr__("sample_{}".format(sampler.replace('_sampler', '')), modfunc)
    
    def _add_analyzers(self):
        """Dynamically add available SALib analyzers as ProblemSpec methods."""
        for analyzer in avail_approaches(analyzers):
            func = getattr(importlib.import_module('SALib.analyze.{}'.format(analyzer)), 'analyze')
            
            @wraps(func)
            def modfunc(*args, **kwargs):
                self.analyze(func, *args, **kwargs)
                return self

            self.__setattr__("analyze_{}".format(analyzer.replace('_analyzer', '')), modfunc)

    def __repr__(self):
        if self._samples is not None:
            print('Samples:', self._samples.shape, "\n")
        if self._results is not None:
            print('Outputs:', self._results.shape, "\n")
        if self._analysis is not None:
            print('Analysis:\n')  # , self._results.shape, "\n")
            an_res = self._analysis
            if isinstance(an_res, ResultDict):
                for df in an_res.to_df():
                    print(df, '\n')
            elif isinstance(an_res, dict):
                for vals in list(an_res.values()):
                    for df in vals.to_df():
                        print(df, "\n")
        return ''
    