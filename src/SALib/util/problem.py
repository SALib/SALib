import warnings
import importlib
from types import MethodType
from functools import partial, wraps

from multiprocess import Pool, cpu_count
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

try:
    from pathos.pp import ParallelPythonPool as pp_Pool

    pathos_available = True
except ImportError:
    pathos_available = False


__all__ = ["ProblemSpec"]


class ProblemSpec(dict):
    """Dictionary-like object representing an SALib Problem specification.

    Attributes
    ----------
    samples : np.array, of generated samples
    results : np.array, of evaluations (i.e., model outputs)
    analysis : np.array or dict, of sensitivity indices
    """

    def __init__(self, *args, **kwargs):
        super(ProblemSpec, self).__init__(*args, **kwargs)

        _check_spec_attributes(self)

        self._samples = None
        self._results = None
        self._analysis = None

        self["num_vars"] = len(self["names"])
        if "groups" not in self:
            self["groups"] = None

        self._add_samplers()
        self._add_analyzers()

    @property
    def samples(self):
        return self._samples

    @samples.setter
    def samples(self, vals):
        cols = vals.shape[1]
        if cols != self["num_vars"]:
            msg = "Mismatched sample size: Expected "
            msg += "{} cols, got {}".format(self["num_vars"], cols)
            raise ValueError(msg)

        self._samples = vals

        # Clear results to avoid confusion
        self._results = None

    @property
    def results(self):
        return self._results

    @results.setter
    def results(self, vals):
        val_shape = vals.shape
        if len(val_shape) == 1:
            cols = 1
        else:
            cols = vals.shape[1]

        out_cols = self.get("outputs", None)
        if out_cols is None:
            if cols == 1:
                self["outputs"] = ["Y"]
            else:
                self["outputs"] = [f"Y{i}" for i in range(1, cols + 1)]
        else:
            if cols != len(self["outputs"]):
                msg = "Mismatched sample size: Expected "
                msg += "{} cols, got {}".format(self["outputs"], cols)
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
        -------
        self : ProblemSpec object
        """
        # Clear model output and analysis results to avoid confusion
        # especially if samples are forcibly changed...
        self._analysis = None
        self._results = None
        self._samples = func(self, *args, **kwargs)

        return self

    def set_samples(self, samples: np.ndarray):
        """Set previous samples used."""
        self.samples = samples

        return self

    def set_results(self, results: np.ndarray):
        """Set previously available model results."""
        if self.samples is not None:
            assert self.samples.shape[0] == results.shape[0], (
                "Provided result array does not match existing number of"
                " existing samples!"
            )

        self.results = results

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

        nprocs : int,
            If specified, attempts to parallelize model evaluations

        **kwargs : dict,
            Additional keyword arguments passed to `func`

        Returns
        -------
        self : ProblemSpec object
        """
        if "nprocs" in kwargs:
            nprocs = kwargs.pop("nprocs")
            return self.evaluate_parallel(func, *args, nprocs=nprocs, **kwargs)

        self.results = func(self._samples, *args, **kwargs)

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
            Number of processors to use.
            Capped to the number of available processors.

        *args : list,
            Additional arguments to be passed to `func`

        **kwargs : dict,
            Additional keyword arguments passed to `func`

        Returns
        -------
        self : ProblemSpec object
        """
        warnings.warn(
            "Parallel evaluation is an experimental feature and may not work."
        )

        if self._samples is None:
            raise RuntimeError("Sampling not yet conducted")

        max_procs = cpu_count()
        if nprocs is None:
            nprocs = max_procs
        else:
            if nprocs > max_procs:
                warnings.warn(
                    f"{nprocs} processors requested but only {max_procs} found."
                )
            nprocs = min(max_procs, nprocs)

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

        self.results = self._collect_results(res)

        return self

    def evaluate_distributed(
        self, func, *args, nprocs=1, servers=None, verbose=False, **kwargs
    ):
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
        -------
        self : ProblemSpec object
        """
        if not pathos_available:
            raise RuntimeError(
                "Pathos is required to run in distributed mode. Please install"
                " with `pip install pathos` or"
                " `conda install pathos -c conda-forge`"
            )

        if verbose:
            from pathos.parallel import stats

        warnings.warn(
            "Distributed evaluation is an untested experimental feature and"
            " may not work."
        )

        workers = pp_Pool(nprocs, servers=servers)

        # Split into even chunks
        chunks = np.array_split(self._samples, int(nprocs) * len(servers), axis=0)

        tmp_f = self._wrap_func(func)

        res = list(workers.map(tmp_f, chunks))

        self.results = self._collect_results(res)

        if verbose:
            print(stats(), "\n")

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

        nprocs : int,
            If specified, attempts to parallelize model evaluations

        **kwargs : dict,
            Additional keyword arguments passed to `func`

        Returns
        -------
        self : ProblemSpec object
        """
        if self["num_vars"] == 1 or (self["groups"] and len("groups") == 1):
            msg = (
                "There is only a single parameter or group defined. There is "
                "no point in conducting sensitivity analysis as any and all"
                "effect(s) will be mapped to the single parameter/group."
            )
            raise ValueError(msg)

        if "nprocs" in kwargs:
            # Call parallel method instead
            return self.analyze_parallel(func, *args, **kwargs)

        if self._results is None:
            raise RuntimeError("Model not yet evaluated")

        if "X" in func.__code__.co_varnames:
            # enforce passing of X if expected
            func = partial(func, *args, X=self._samples, **kwargs)
        else:
            func = partial(func, *args, **kwargs)

        out_cols = self.get("outputs", None)
        if out_cols is None:
            if len(self._results.shape) == 1:
                self["outputs"] = ["Y"]
            else:
                num_cols = self._results.shape[1]
                self["outputs"] = [f"Y{i}" for i in range(1, num_cols + 1)]

        if len(self["outputs"]) > 1:
            self._analysis = {}
            for i, out in enumerate(self["outputs"]):
                self._analysis[out] = func(self, Y=self._results[:, i])
        else:
            self._analysis = func(self, Y=self._results)

        return self

    def analyze_parallel(self, func, *args, nprocs=None, **kwargs):
        """Analyze sampled results using the given function in parallel.

        Parameters
        ----------
        func : function,
            Analysis method to use. The provided function must accept the
            problem specification as the first parameter, X values if needed,
            Y values, and return a numpy array.

        *args : list,
            Additional arguments to be passed to `func`

        nprocs : int,
            Number of processors to use.
            Capped to the number of outputs or available processors.

        **kwargs : dict,
            Additional keyword arguments passed to `func`

        Returns
        -------
        self : ProblemSpec object
        """
        warnings.warn("Parallel analysis is an experimental feature and may not work.")

        if self._results is None:
            raise RuntimeError("Model not yet evaluated")

        if "X" in func.__code__.co_varnames:
            # enforce passing of X if expected
            func = partial(func, *args, X=self._samples, **kwargs)
        else:
            func = partial(func, *args, **kwargs)

        out_cols = self.get("outputs", None)
        if out_cols is None:
            if len(self._results.shape) == 1:
                self["outputs"] = ["Y"]
            else:
                num_cols = self._results.shape[1]
                self["outputs"] = [f"Y{i}" for i in range(1, num_cols + 1)]

        # Cap number of processors used
        Yn = len(self["outputs"])
        if Yn == 1:
            # Only single output, cannot parallelize
            warnings.warn(
                f"Analysis was not parallelized: {nprocs} processors requested"
                f" for 1 output."
            )

            res = func(self, Y=self._results)
        else:
            max_procs = cpu_count()
            if nprocs is None:
                nprocs = max_procs
            else:
                nprocs = min(Yn, nprocs, max_procs)

            if ptqdm_available:
                # Display progress bar if available
                res = p_imap(
                    lambda y: func(self, Y=y),
                    [self._results[:, i] for i in range(Yn)],
                    num_cpus=nprocs,
                )
            else:
                with Pool(nprocs) as pool:
                    res = list(
                        pool.imap(
                            lambda y: func(self, Y=y),
                            [self._results[:, i] for i in range(Yn)],
                        )
                    )

        # Assign by output name if more than 1 output, otherwise
        # attach directly
        if Yn > 1:
            self._analysis = {}
            for out, Si in zip(self["outputs"], list(res)):
                self._analysis[out] = Si
        else:
            self._analysis = res

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

    def plot(self, **kwargs):
        """Plot results as a bar chart.

        Returns
        -------
        axes : matplotlib axes object
        """
        if self._analysis is None:
            raise RuntimeError("Analysis not yet conducted")

        num_rows = len(self["outputs"])
        if num_rows == 1:
            return self._analysis.plot(**kwargs)

        try:
            plt
        except NameError:
            import matplotlib.pyplot as plt

        num_cols = 1
        fk = list(self._analysis.keys())[0]
        if isinstance(self._analysis[fk].to_df(), (list, tuple)):
            # have to divide by 2 to account for CI columns
            num_cols = len(self._analysis[fk]) // 2

        p_width = max(num_cols * 3, 5)
        p_height = max(num_rows * 3, 6)
        _, axes = plt.subplots(
            num_rows, num_cols, sharey=True, figsize=(p_width, p_height)
        )
        for res, ax in zip(self._analysis, axes):
            self._analysis[res].plot(ax=ax, **kwargs)

            try:
                ax[0].set_title(res)
            except TypeError:
                ax.set_title(res)

        plt.tight_layout()

        return axes

    def heatmap(
        self, metric: str = None, index: str = None, title: str = None, ax=None
    ):
        """Plot results as a heatmap.

        Parameters
        ----------
        metric : str or None, name of output to analyze (display all if `None`)
        index : str or None, name of index to plot, dependent on what
                    analysis was conducted (ST, S1, etc; displays all if `None`)
        title : str, title of plot to use (defaults to the same as `metric`)
        ax : axes object, matplotlib axes object to use for plot.
                Creates a new figure if not provided.

        Returns
        -------
        ax : matplotlib axes object
        """
        from SALib.plotting.heatmap import heatmap  # type: ignore

        return heatmap(self, metric, index, title, ax)

    def _wrap_func(self, func, *args, **kwargs):
        # Create wrapped partial function to allow passing of additional args
        tmp_f = func
        if (len(args) > 0) or (len(kwargs) > 0):
            tmp_f = lambda x: func(x, *args, **kwargs)  # noqa

        return tmp_f

    def _collect_results(self, res):
        res_shape = res[0].shape
        if len(res_shape) > 1:
            res_shape = (len(self._samples), *res_shape[1:])
        else:
            res_shape = len(self._samples)

        final_res = np.empty(res_shape)

        # Collect results
        # Cannot enumerate over this as the length
        # of individual results may vary
        i = 0
        for r in res:
            r_len = len(r)
            final_res[i : i + r_len] = r
            i += r_len

        return final_res

    def _method_creator(self, func, method):
        """Generate convenience methods for specified `method`."""

        @wraps(func)
        def modfunc(self, *args, **kwargs):
            return getattr(self, method)(func, *args, **kwargs)

        return modfunc

    def _add_samplers(self):
        """Dynamically add available SALib samplers as ProblemSpec methods."""
        for sampler in avail_approaches(samplers):
            func = getattr(
                importlib.import_module("SALib.sample.{}".format(sampler)), "sample"
            )
            method_name = "sample_{}".format(sampler.replace("_sampler", ""))

            self.__setattr__(
                method_name, MethodType(self._method_creator(func, "sample"), self)
            )

    def _add_analyzers(self):
        """Dynamically add available SALib analyzers as ProblemSpec methods."""
        for analyzer in avail_approaches(analyzers):
            func = getattr(
                importlib.import_module("SALib.analyze.{}".format(analyzer)), "analyze"
            )
            method_name = "analyze_{}".format(analyzer.replace("_analyzer", ""))

            self.__setattr__(
                method_name, MethodType(self._method_creator(func, "analyze"), self)
            )

    def _repr_pretty_(self, p, cycle):
        p.text(str(self) if not cycle else "...")

    def __str__(self):
        rep = ""
        if self._samples is not None:
            arr_shape = self._samples.shape
            if len(arr_shape) == 1:
                arr_shape = (arr_shape[0], 1)
            nr, nx = arr_shape
            rep += (
                "Samples:\n"
                f"\t{nx} parameters: {self['names']}\n"
                f"\t{nr} evaluations\n"
            )
        if self._results is not None:
            arr_shape = self._results.shape
            if len(arr_shape) == 1:
                arr_shape = (arr_shape[0], 1)
            nr, ny = arr_shape
            rep += (
                "Outputs:\n"
                f"\t{ny} outputs: {self['outputs']}\n"
                f"\t{nr} evaluations\n"
            )
        if self._analysis is not None:
            rep += "Analysis:\n"
            an_res = self._analysis

            allowed_types = (list, tuple)
            if isinstance(an_res, ResultDict):
                an_res = an_res.to_df()
                if not isinstance(an_res, allowed_types):
                    rep += f"{an_res}\n"
                else:
                    for df in an_res:
                        rep += f"{df}\n"
            elif isinstance(an_res, dict):
                for res_name in an_res:
                    rep += "{}:\n".format(res_name)

                    dfs = an_res[res_name].to_df()
                    if isinstance(dfs, allowed_types):
                        for df in dfs:
                            rep += f"{df}:\n"
                    else:
                        rep += f"{dfs}:\n"

        if len(rep) == 0:
            rep = "ProblemSpec does not currently contain any samples, evaluations or results."

        return rep


def _check_spec_attributes(spec: ProblemSpec):
    assert "names" in spec, "Names not defined"
    assert "bounds" in spec, "Bounds not defined"
    assert len(spec["bounds"]) == len(
        spec["names"]
    ), f"""Number of bounds do not match number of names
        Number of names:
        {len(spec['names'])} | {spec['names']}
        ----------------
        Number of bounds: {len(spec['bounds'])}
        """
