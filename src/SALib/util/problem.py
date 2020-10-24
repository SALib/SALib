import warnings
import importlib
from types import MethodType

from multiprocess import Pool, cpu_count
from pathos.pp import ParallelPythonPool as pp_Pool
from functools import partial, wraps

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

    @samples.setter
    def samples(self, vals):
        cols = vals.shape[1]
        if cols != self['num_vars']:
            msg = "Mismatched sample size: Expected "
            msg += "{} cols, got {}".format(self['num_vars'], cols)
            raise ValueError(msg)

        self._samples = vals

    @property
    def results(self):
        return self._results
    
    @results.setter
    def results(self, vals):
        cols = vals.shape[1]
        if cols != len(self['outputs']):
            msg = "Mismatched sample size: Expected "
            msg += "{} cols, got {}".format(self['outputs'], cols)
            raise ValueError(msg)

        self._results = vals
    
    @property
    def analysis(self):
        return self._analysis

    def sample(self, func, *args, **kwargs):
        """Create sample using given function.

        Parameters
        ----------
        func : function,
            Sampling method to use. The given function must accept the SALib
            problem specification as the first parameter and return a numpy
            array.
        
        *args : list,
            Additional arguments to be passed to `func`
        
        **kwargs : dict,
            Additional keyword arguments passed to `func`

        Returns
        ----------
        self : ProblemSpec object
        """
        # Clear model output and analysis results to avoid confusion
        # especially if samples are forcibly changed...
        self._analysis = None
        self._results = None
        self._samples = func(self, *args, **kwargs)

        return self

    def evaluate(self, func, *args, **kwargs):
        """Evaluate a given model.

        Parameters
        ----------
        func : function,
            Model, or function that wraps a model, to be run/evaluated.
            The provided function is required to accept a numpy array of
            inputs as its first parameter and must return a numpy array of
            results.
        
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
        """Evaluate model locally in parallel.

        All detected processors will be used if `nprocs` is not specified.

        Parameters
        ----------
        func : function,
            Model, or function that wraps a model, to be run in parallel.
            The provided function needs to accept a numpy array of inputs as 
            its first parameter and must return a numpy array of results.
        
        nprocs : int,
            Number of processors to use. Uses all available if not specified.
        
        *args : list,
            Additional arguments to be passed to `func`
        
        **kwargs : dict,
            Additional keyword arguments passed to `func`

        Returns
        ----------
        self : ProblemSpec object
        """
        warnings.warn("This is an experimental feature and may not work.")

        if self._samples is None:
            raise RuntimeError("Sampling not yet conducted")
        
        if nprocs is None:
            nprocs = cpu_count()

        # Create wrapped partial function to allow passing of additional args
        tmp_f = self._wrap_func(func, *args, **kwargs)

        # Split into even chunks
        chunks = np.array_split(self._samples, int(nprocs), axis=0)

        if ptqdm_available:
            # Display progress bar if available
            res = p_imap(tmp_f, chunks, num_cpus=nprocs)
        else:
            with Pool(nprocs) as pool:
                res = list(pool.imap(tmp_f, chunks))

        self._results = self._collect_results(res)

        return self

    def evaluate_distributed(self, func, *args, nprocs=1, servers=None, verbose=False, **kwargs):
        """Distribute model evaluation across a cluster.

        Usage Conditions:
        * The provided function needs to accept a numpy array of inputs as 
          its first parameter
        * The provided function must return a numpy array of results

        Parameters
        ----------
        func : function,
            Model, or function that wraps a model, to be run in parallel
        
        nprocs : int,
            Number of processors to use for each node. Defaults to 1.
        
        servers : list[str] or None,
            IP addresses or alias for each server/node to use.

        verbose : bool,
            Display job execution statistics. Defaults to False.
        
        *args : list,
            Additional arguments to be passed to `func`
        
        **kwargs : dict,
            Additional keyword arguments passed to `func`

        Returns
        ----------
        self : ProblemSpec object
        """
        if verbose:
            from pathos.parallel import stats

        warnings.warn("This is an untested experimental feature and may not work.")

        workers = pp_Pool(nprocs, servers=servers)

        # Split into even chunks
        chunks = np.array_split(self._samples, int(nprocs)*len(servers), axis=0)

        tmp_f = self._wrap_func(func)

        res = list(workers.map(tmp_f, chunks))

        self._results = self._collect_results(res)

        if verbose:
            print(stats(), '\n')

        workers.clear()

        return self


    def analyze(self, func, *args, **kwargs):
        """Analyze sampled results using given function.


        Parameters
        ----------
        func : function,
            Analysis method to use. The provided function must accept the 
            problem specification as the first parameter, X values if needed, 
            Y values, and return a numpy array.
        
        *args : list,
            Additional arguments to be passed to `func`
        
        **kwargs : dict,
            Additional keyword arguments passed to `func`

        Returns
        ----------
        self : ProblemSpec object
        """
        if self._results is None:
            raise RuntimeError("Model not yet evaluated")

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
    
    def plot(self):
        """Plot results"""
        if self._analysis is None:
            raise RuntimeError("Analysis not yet conducted")

        return self._analysis.plot()


    def _wrap_func(self, func, *args, **kwargs):
        # Create wrapped partial function to allow passing of additional args
        tmp_f = func
        if (len(args) > 0) or (len(kwargs) > 0):
            tmp_f = partial(func, *args, **kwargs)
        
        return tmp_f
    
    def _setup_result_array(self):
        if len(self['outputs']) > 1:
            res_shape = (len(self._samples), len(self['outputs']))
        else:
            res_shape = len(self._samples)

        return np.empty(res_shape)

    def _collect_results(self, res):
        final_res = self._setup_result_array()

        # Collect results
        # Cannot enumerate over this as the length
        # of individual results may vary
        i = 0
        for r in res:
            r_len = len(r)
            final_res[i:i+r_len] = r
            i += r_len
        
        return final_res

    def _method_creator(self, func, method):
        @wraps(func)
        def modfunc(self, *args, **kwargs):
            return getattr(self, method)(func, *args, **kwargs)
        
        return modfunc

    def _add_samplers(self):
        """Dynamically add available SALib samplers as ProblemSpec methods."""
        for sampler in avail_approaches(samplers):
            func = getattr(importlib.import_module('SALib.sample.{}'.format(sampler)), 'sample')
            method_name = "sample_{}".format(sampler.replace('_sampler', ''))            

            self.__setattr__(method_name, MethodType(self._method_creator(func, 'sample'), self))
    
    def _add_analyzers(self):
        """Dynamically add available SALib analyzers as ProblemSpec methods."""
        for analyzer in avail_approaches(analyzers):
            func = getattr(importlib.import_module('SALib.analyze.{}'.format(analyzer)), 'analyze')
            method_name = "analyze_{}".format(analyzer.replace('_analyzer', ''))

            self.__setattr__(method_name, MethodType(self._method_creator(func, 'analyze'), self))

    def __repr__(self):
        if self._samples is not None:
            print('Samples:', self._samples.shape, "\n")
        if self._results is not None:
            print('Outputs:', self._results.shape, "\n")
        if self._analysis is not None:
            print('Analysis:\n')
            an_res = self._analysis
            if isinstance(an_res, ResultDict):
                dfs = an_res.to_df()
                if isinstance(dfs, list):
                    for df in dfs:
                        print(df, "\n")
                else:
                    print(an_res.to_df(), "\n")
            elif isinstance(an_res, dict):
                for vals in list(an_res.values()):
                    for df in vals.to_df():
                        print(df, "\n")
        return ''
    