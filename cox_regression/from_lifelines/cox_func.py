import numpy as np
import pandas as pd
from datetime import datetime


from pandas import DataFrame
from typing import Optional, List, Iterable, Any, Dict, Union, Tuple, Callable
import formulaic
from autograd import elementwise_grad
import time
from scipy import stats
import warnings


from numpy import sum as array_sum_to_scalar
from autograd import numpy as anp

from scipy.linalg import solve as spsolve, LinAlgError

from numpy.linalg import norm, inv

matrix_axis_0_sum_to_1d_array = lambda m: np.sum(m, 0)
string_rjustify = lambda width: lambda s: s.rjust(width, " ")

def format_p_value(decimals) -> Callable:
    threshold = 0.5 * 10 ** (-decimals)
    return lambda p: "<%s" % threshold if p < threshold else "{:4.{prec}f}".format(p, prec=decimals)

def format_exp_floats(decimals) -> Callable:
    """
    sometimes the exp. column can be too large
    """
    threshold = 10 ** 5
    return lambda n: "{:.{prec}e}".format(n, prec=decimals) if n > threshold else "{:4.{prec}f}".format(n, prec=decimals)

def format_floats(decimals) -> Callable:
    return lambda f: "{:4.{prec}f}".format(f, prec=decimals)

def leading_space(s) -> str:
    return " %s" % s

def map_leading_space(list) -> List[str]:
    return [leading_space(c) for c in list]

class Printer:
    def __init__(
        self,
        model,
        headers: List[Tuple[str, Any]],
        footers: List[Tuple[str, Any]],
        justify: Callable,
        header_kwargs: Dict,
        decimals: int,
        columns: Optional[List],
    ):
        self.headers = headers
        self.model = model
        self.decimals = decimals
        self.columns = columns
        self.justify = justify
        self.footers = footers

        for tuple_ in header_kwargs.items():
            self.add_to_headers(tuple_)

    def add_to_headers(self, tuple_):
        self.headers.append(tuple_)

    def print_specific_style(self, style):
        if style == "html":
            return self.html_print()
        elif style == "ascii":
            return self.ascii_print()
        elif style == "latex":
            return self.latex_print()
        else:
            raise ValueError("style not available.")

    def print(self, style=None):
        if style is not None:
            self.print_specific_style(style)
        else:
            try:
                from IPython.display import display

                display(self)
            except ImportError:
                self.ascii_print()

    def latex_print(self):
        print(self.to_latex())

    def to_latex(self):
        summary_df = self.model.summary
        if self.columns is None:
            columns = summary_df.columns
        else:
            columns = summary_df.columns & self.columns
        return summary_df[columns].to_latex(float_format="%." + str(self.decimals) + "f")

    def html_print(self):
        print(self.to_html())

    def to_html(self):
        summary_df = self.model.summary

        decimals = self.decimals
        if self.columns is None:
            columns = summary_df.columns
        else:
            columns = summary_df.columns & self.columns

        headers = self.headers.copy()
        headers.insert(0, ("model", "lifelines." + self.model._class_name))

        header_df = pd.DataFrame.from_records(headers).set_index(0)

        header_html = header_df.to_html(header=False, notebook=True, index_names=False)

        summary_html = summary_df[columns].to_html(
            col_space=12,
            index_names=False,
            float_format=format_floats(decimals),
            formatters={
                **{c: format_exp_floats(decimals) for c in columns if "exp(" in c},
                **{"p": format_p_value(decimals)},
            },
        )

        if self.footers:
            footer_df = pd.DataFrame.from_records(self.footers).set_index(0)
            footer_html = "<br>" + footer_df.to_html(header=False, notebook=True, index_names=False)
        else:
            footer_html = ""
        return header_html + summary_html + footer_html

    def to_ascii(self):
        df = self.model.summary
        justify = self.justify
        ci = 100 * (1 - self.model.alpha)
        decimals = self.decimals

        repr_string = ""

        repr_string += repr(self.model) + "\n"
        for string, value in self.headers:
            repr_string += "{} = {}".format(justify(string), value) + "\n"

        repr_string += "\n" + "---" + "\n"

        df.columns = map_leading_space(df.columns)

        if self.columns is not None:
            columns = df.columns.intersection(map_leading_space(self.columns))
        else:
            columns = df.columns

        if len(columns) <= 7:
            # only need one row of display
            first_row_set = [
                "coef",
                "exp(coef)",
                "se(coef)",
                "coef lower %d%%" % ci,
                "coef upper %d%%" % ci,
                "exp(coef) lower %d%%" % ci,
                "exp(coef) upper %d%%" % ci,
                "z",
                "p",
                "-log2(p)",
            ]
            second_row_set = []

        else:
            first_row_set = [
                "coef",
                "exp(coef)",
                "se(coef)",
                "coef lower %d%%" % ci,
                "coef upper %d%%" % ci,
                "exp(coef) lower %d%%" % ci,
                "exp(coef) upper %d%%" % ci,
            ]
            second_row_set = ["z", "p", "-log2(p)"]

        repr_string += df[columns].to_string(
            float_format=format_floats(decimals),
            formatters={
                **{c: format_exp_floats(decimals) for c in columns if "exp(coef)" in c},
                **{leading_space("p"): format_p_value(decimals)},
            },
            columns=[c for c in map_leading_space(first_row_set) if c in columns],
        )

        if second_row_set:
            repr_string += "\n\n"
            repr_string += df[columns].to_string(
                float_format=format_floats(decimals),
                formatters={
                    **{c: format_exp_floats(decimals) for c in columns if "exp(" in c},
                    **{leading_space("p"): format_p_value(decimals)},
                },
                columns=map_leading_space(second_row_set),
            )

        with np.errstate(invalid="ignore", divide="ignore"):

            repr_string += "\n" + "---" + "\n"
            for string, value in self.footers:
                repr_string += "{} = {}".format(string, value) + "\n"
        return repr_string

    def ascii_print(self):
        print(self.to_ascii())

    def _repr_latex_(self,):
        return self.to_latex()

    def _repr_html_(self):
        return self.to_html()

    def __repr__(self):
        return self.to_ascii()

def quiet_log2(p):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"divide by zero encountered in log2")
        return np.log2(p)

def _to_list(x) -> List[Any]:
    if not isinstance(x, list):
        return [x]
    return x

def _to_1d_array(x) -> np.array:
    v = np.atleast_1d(x)
    try:
        if v.shape[0] > 1 and v.shape[1] > 1:
            raise ValueError("Wrong shape (2d) given to _to_1d_array")
    except IndexError:
        pass
    return v

class StatisticalResult:
    """
    This class holds the result of statistical tests with a nice printer wrapper to display the results.

    Note
    -----
    This class' API changed in version 0.16.0.

    Parameters
    ----------
    p_value: iterable or float
        the p-values of a statistical test(s)
    test_statistic: iterable or float
        the test statistics of a statistical test(s). Must be the same size as p-values if iterable.
    test_name: string
        the test that was used. lifelines should set this.
    name: iterable or string
        if this class holds multiple results (ex: from a pairwise comparison), this can hold the names. Must be the same size as p-values if iterable.
    kwargs:
        additional information to attach to the object and display in ``print_summary()``.

    """

    def __init__(self, p_value, test_statistic, name=None, test_name=None, **kwargs):
        self.p_value = p_value
        self.test_statistic = test_statistic
        self.test_name = test_name

        self._p_value = _to_1d_array(p_value)
        self._test_statistic = _to_1d_array(test_statistic)

        assert len(self._p_value) == len(self._test_statistic)

        if name is not None:
            self.name = _to_list(name)
            assert len(self.name) == len(self._test_statistic)
        else:
            self.name = None

        for kw, value in kwargs.items():
            setattr(self, kw, value)

        kwargs["test_name"] = test_name
        self._kwargs = kwargs

    def print_specific_style(self, style, decimals=2, **kwargs):
        """
        Parameters
        -----------

        style: str
          One of {'ascii', 'html', 'latex'}

        """
        if style == "html":
            return self.html_print(decimals=decimals, **kwargs)
        elif style == "ascii":
            return self.ascii_print(decimals=decimals, **kwargs)
        elif style == "latex":
            return self.latex_print(decimals=decimals, **kwargs)
        else:
            raise ValueError("style not available.")

    def print_summary(self, decimals=2, style=None, **kwargs):
        """
        Print summary statistics describing the fit, the coefficients, and the error bounds.

        Parameters
        -----------
        decimals: int, optional (default=2)
            specify the number of decimal places to show
        kwargs:
            print additional meta data in the output (useful to provide model names, dataset names, etc.) when comparing
            multiple outputs.

        """
        if style is not None:
            self.print_specific_style(style)
        else:
            try:
                from IPython.display import display

                display(self)
            except ImportError:
                self.ascii_print()

    def html_print(self, decimals=2, **kwargs):
        print(self.to_html(decimals, **kwargs))

    def to_html(self, decimals=2, **kwargs):
        extra_kwargs = dict(list(self._kwargs.items()) + list(kwargs.items()))
        summary_df = self.summary

        headers = []
        for k, v in extra_kwargs.items():
            headers.append((k, v))

        header_df = pd.DataFrame.from_records(headers).set_index(0)
        header_html = header_df.to_html(header=False, notebook=True, index_names=False)

        summary_html = summary_df.to_html(float_format=format_floats(decimals), formatters={**{"p": format_p_value(decimals)}})

        return header_html + summary_html

    def latex_print(self, decimals=2, **kwargs):
        print(self.to_latex(decimals, **kwargs))

    def to_latex(self, decimals=2, **kwargs):
        return self.summary.to_latex()

    @property
    def summary(self):
        """

        Returns
        -------
        DataFrame
            a DataFrame containing the test statistics and the p-value

        """
        cols = ["test_statistic", "p"]

        # test to see if self.names is a tuple
        if self.name and isinstance(self.name[0], tuple):
            index = pd.MultiIndex.from_tuples(self.name)
        else:
            index = self.name

        df = pd.DataFrame(list(zip(self._test_statistic, self._p_value)), columns=cols, index=index).sort_index()
        df["-log2(p)"] = -quiet_log2(df["p"])
        return df

    def to_ascii(self, decimals=2, **kwargs):
        extra_kwargs = dict(list(self._kwargs.items()) + list(kwargs.items()))
        meta_data = self._stringify_meta_data(extra_kwargs)

        df = self.summary

        s = "<lifelines.StatisticalResult: {0}>".format(self.test_name)
        s += "\n" + meta_data + "\n"
        s += "---\n"
        s += df.to_string(
            float_format=format_floats(decimals), index=self.name is not None, formatters={"p": format_p_value(decimals)}
        )

        return s

    def _stringify_meta_data(self, dictionary):
        longest_key = max([len(k) for k in dictionary])
        justify = string_rjustify(longest_key)
        s = ""
        for k, v in dictionary.items():
            s += "{} = {}\n".format(justify(k), v)

        return s

    def __add__(self, other):
        """useful for aggregating results easily"""
        p_values = np.r_[self._p_value, other._p_value]
        test_statistics = np.r_[self._test_statistic, other._test_statistic]
        names = self.name + other.name
        kwargs = dict(list(self._kwargs.items()) + list(other._kwargs.items()))
        return StatisticalResult(p_values, test_statistics, name=names, **kwargs)

    def ascii_print(self, decimals=2, **kwargs):
        print(self.to_ascii(decimals, **kwargs))

    def _repr_latex_(
        self,
    ):
        return self.to_latex()

    def _repr_html_(self):
        return self.to_html()

    def __repr__(self):
        return self.to_ascii()

def _chisq_test_p_value(U, degrees_freedom) -> float:
    p_value = stats.chi2.sf(U, degrees_freedom)
    return p_value

def _get_index(X) -> List[Any]:
    # we need a unique index because these are about to become column names.
    if isinstance(X, pd.DataFrame) and X.index.is_unique:
        index = list(X.index)
    elif isinstance(X, pd.DataFrame) and not X.index.is_unique:
        warnings.warn("DataFrame Index is not unique, defaulting to incrementing index instead.")
        index = list(range(X.shape[0]))
    elif isinstance(X, pd.Series):
        return list(X.index)
    else:
        # If it's not a dataframe or index is not unique, order is up to user
        index = list(range(X.shape[0]))
    return index

def inv_normal_cdf(p) -> float:
    return stats.norm.ppf(p)

class StepSizer:
    """
    This class abstracts complicated step size logic out of the fitters. The API is as follows:

    > step_sizer = StepSizer(initial_step_size)
    > step_size = step_sizer.next()
    > step_sizer.update(some_convergence_norm)
    > step_size = step_sizer.next()


    ATM it contains lots of "magic constants"
    """

    def __init__(self, initial_step_size: Optional[float]) -> None:
        initial_step_size = initial_step_size or 0.90

        self.initial_step_size = initial_step_size
        self.step_size = initial_step_size
        self.temper_back_up = False
        self.norm_of_deltas: List[float] = []

    def update(self, norm_of_delta: float) -> "StepSizer":
        SCALE = 1.3
        LOOKBACK = 3

        self.norm_of_deltas.append(norm_of_delta)

        # speed up convergence by increasing step size again
        if self.temper_back_up:
            self.step_size = min(self.step_size * SCALE, self.initial_step_size)

        # Only allow small steps
        if norm_of_delta >= 15.0:
            self.step_size *= 0.1
            self.temper_back_up = True
        elif 15.0 > norm_of_delta > 5.0:
            self.step_size *= 0.25
            self.temper_back_up = True

        # recent non-monotonically decreasing is a concern
        if len(self.norm_of_deltas) >= LOOKBACK and not self._is_monotonically_decreasing(self.norm_of_deltas[-LOOKBACK:]):
            self.step_size *= 0.98

        # recent monotonically decreasing is good though
        if len(self.norm_of_deltas) >= LOOKBACK and self._is_monotonically_decreasing(self.norm_of_deltas[-LOOKBACK:]):
            self.step_size = min(self.step_size * SCALE, 1.0)

        return self

    @staticmethod
    def _is_monotonically_decreasing(array: Union[List[float], List[float]]) -> bool:
        return np.all(np.diff(array) < 0)

    def next(self) -> float:
        return self.step_size

def normalize(X, mean=None, std=None):
    """
    Normalize X. If mean OR std is None, normalizes
    X to have mean 0 and std 1.
    """
    if mean is None or std is None:
        mean = X.mean(0)
        std = X.std(0)
    return (X - mean) / std

def check_for_instantaneous_events_at_death_time(events, start, stop):
    """
    avoid rows like
        start=1, stop=5, event=False
        start=5, stop=5, event=True
    """
    if ((start == stop) & (events)).any():

        warning_text = (
            dedent(
                """There exist rows in your DataFrame with start and stop equal and a death event.

            This will likely throw an
        error during convergence. For example, investigate subjects with id's: %s.

        You can fix this by collapsing this row into the subject's previous row. Example:

        start=1, stop=5, event=False
        start=5, stop=5, event=True

        to

        start=1, stop=5, event=True

        """
            )
            % events.index[(start == stop) & (events)].tolist()[:5]
        )
        warnings.warn(warning_text, ConvergenceWarning)

def check_for_instantaneous_events_at_time_zero(start, stop):
    if ((start == stop) & (stop == 0)).any():
        warning_text = dedent(
            """There exist rows in your DataFrame with start and stop both at time 0:

        >>> df.loc[(df[start_col] == df[stop_col]) & (df[start_col] == 0)]

        These can be safely dropped, which should improve performance.

        >>> df = df.loc[~((df[start_col] == df[stop_col]) & (df[start_col] == 0))]"""
        )
        warnings.warn(warning_text, RuntimeWarning)

def check_for_immediate_deaths(events, start, stop):
    # Only used in CTV. This checks for deaths immediately, that is (0,0) lives.
    if ((start == stop) & (stop == 0) & events).any():
        raise ValueError(
            dedent(
                """The dataset provided has subjects that die on the day of entry. (0, 0)
    is not allowed in CoxTimeVaryingFitter. If suffices to add a small non-zero value to their end - example Pandas code:

    >>> df.loc[ (df[start_col] == df[stop_col]) & (df[start_col] == 0) & df[event_col], stop_col] = 0.5

    Alternatively, add 1 to every subjects' final end period."""
            )
        )

def check_for_nonnegative_intervals(start, stop):
    if (stop < start).any():
        raise ValueError(dedent("""There exist values in the `stop_col` column that are less than `start_col`."""))

def check_for_numeric_dtypes_or_raise(df):
    nonnumeric_cols = [col for (col, dtype) in df.dtypes.iteritems() if dtype.name == "category" or dtype.kind not in "biuf"]
    if len(nonnumeric_cols) > 0:  # pylint: disable=len-as-condition
        raise TypeError(
            "DataFrame contains nonnumeric columns: %s. Try 1) using pandas.get_dummies to convert the non-numeric column(s) to numerical data, 2) using it in stratification `strata=`, or 3) dropping the column(s)."
            % nonnumeric_cols
        )

def check_complete_separation_low_variance(df: pd.DataFrame, events: np.ndarray, event_col: str):

    events = events.astype(bool)
    deaths_only = df.columns[_low_var(df.loc[events])]
    censors_only = df.columns[_low_var(df.loc[~events])]
    total = df.columns[_low_var(df)]
    problem_columns = censors_only.union(deaths_only).difference(total).tolist()
    if problem_columns:
        warning_text = """Column {cols} have very low variance when conditioned on death event present or not. This may harm convergence. This could be a form of 'complete separation'. For example, try the following code:

    >>> events = df['{event_col}'].astype(bool)
    >>> print(df.loc[events, '{cols}'].var())
    >>> print(df.loc[~events, '{cols}'].var())

    A very low variance means that the column {cols} completely determines whether a subject dies or not. See https://stats.stackexchange.com/questions/11109/how-to-deal-with-perfect-separation-in-logistic-regression.\n""".format(
            cols=problem_columns[0], event_col=event_col
        )
        warnings.warn(dedent(warning_text), ConvergenceWarning)

def _low_var(df):
    return df.var(0) < 1e-4

def check_low_var(df, prescript="", postscript=""):
    low_var = _low_var(df)
    if low_var.any():
        cols = str(list(df.columns[low_var]))
        warning_text = (
            "%sColumn(s) %s have very low variance. \
    This may harm convergence. 1) Are you using formula's? Did you mean to add '-1' to the end. 2) Try dropping this redundant column before fitting \
    if convergence fails.%s\n"
            % (prescript, cols, postscript)
        )
        warnings.warn(dedent(warning_text), ConvergenceWarning)

def check_nans_or_infs(df_or_array):
    if isinstance(df_or_array, (pd.Series, pd.DataFrame)):
        return check_nans_or_infs(df_or_array.values)

    if pd.isnull(df_or_array).any():
        raise TypeError("NaNs were detected in the dataset. Try using pd.isnull to find the problematic values.")

    try:
        infs = np.isinf(df_or_array)
    except TypeError:
        warning_text = (
            """Attempting to convert an unexpected datatype '%s' to float. Suggestion: 1) use `lifelines.datetimes_to_durations` to do conversions or 2) manually convert to floats/booleans."""
            % df_or_array.dtype
        )
        warnings.warn(warning_text, UserWarning)
        try:
            infs = np.isinf(df_or_array.astype(float))
        except:
            raise TypeError("Wrong dtype '%s'." % df_or_array.dtype)

    if infs.any():
        raise TypeError("Infs were detected in the dataset. Try using np.isinf to find the problematic values.")

class CovariateParameterMappings:
    """
    This class controls the mapping, possible trivial, between covariates and parameters. User, or lifelines, create
    a seed mapping, and this class takes over and creates logic and pieces to make the transformation simple.

    Ideally all transformation of datasets to parameters handled by this class.

    Parameters
    -----------

    seed_mapping: dict
        a mapping of parameters to covariates, specified through a list of column names, or formula.
    df: DataFrame
        the training dataset
    force_intercept:
        True to always add an constant column.
    force_no_intercept:
        True to always remove an constant column.
    """

    INTERCEPT_COL = "Intercept"

    def __init__(self, seed_mapping: Dict, df: pd.DataFrame, force_intercept: bool = False, force_no_intercept: bool = False):
        self.mappings = {}
        self.force_intercept = force_intercept
        self.force_no_intercept = force_no_intercept

        for param, seed_transform in seed_mapping.items():

            if isinstance(seed_transform, str):
                self.mappings[param] = self._string_seed_transform(seed_transform, df)

            elif isinstance(seed_transform, list):
                # user inputted a list of column names, as strings
                self.mappings[param] = self._list_seed_transform(seed_transform)

            elif isinstance(seed_transform, pd.DataFrame):
                # use all the columns in df
                self.mappings[param] = self._list_seed_transform(seed_transform.columns.tolist())

            elif seed_transform is None:
                # use all the columns in df
                self.mappings[param] = self._list_seed_transform(df.columns.tolist())

            elif isinstance(seed_transform, pd.Index):
                # similar to providing a list
                self.mappings[param] = self._list_seed_transform(seed_transform.tolist())

            else:
                raise ValueError("Unexpected transform.")

    @classmethod
    def add_intercept_col(cls, df):
        df = df.copy()
        df[cls.INTERCEPT_COL] = 1
        return df

    def transform_df(self, df: pd.DataFrame):

        Xs = {}
        for param_name, transform in self.mappings.items():
            if isinstance(transform, formulaic.formula.Formula):
                X = transform.get_model_matrix(df)
            elif isinstance(transform, list):
                if self.force_intercept:
                    df = self.add_intercept_col(df)
                X = df[transform]
            else:
                raise ValueError("Unexpected transform.")

            # some parameters are constants (like in piecewise and splines) and so should
            # not be dropped.
            if self.force_no_intercept and X.shape[1] > 1:
                try:
                    X = X.drop(self.INTERCEPT_COL, axis=1)
                except:
                    pass

            Xs[param_name] = X

        # in pandas 0.23.4, the Xs as a dict is sorted differently from the Xs as a DataFrame's columns
        # hence we need to reorder, see https://github.com/CamDavidsonPilon/lifelines/issues/931
        Xs_df = pd.concat(Xs, axis=1, names=("param", "covariate")).astype(float)
        Xs_df = Xs_df[list(self.mappings.keys())]

        # we can't concat empty dataframes and return a column MultiIndex,
        # so we create a "fake" dataframe (acts like a dataframe) to return.
        # This should be removed because it's gross.
        if Xs_df.size == 0:
            return {p: pd.DataFrame(index=df.index) for p in self.mappings.keys()}
        else:
            return Xs_df

    def keys(self):
        yield from self.mappings.keys()

    def _list_seed_transform(self, list_: List):
        list_ = list_.copy()
        if self.force_intercept:
            list_.append(self.INTERCEPT_COL)
        return list_

    def _string_seed_transform(self, formula: str, df: pd.DataFrame):
        # user input a formula, hopefully
        if self.force_intercept:
            formula += "+ 1"

        design_info = formulaic.Formula(formula)

        return design_info

def pass_for_numeric_dtypes_or_raise_array(x):
    """
    Use the utility `to_numeric` to check that x is convertible to numeric values, and then convert. Any errors
    are reported back to the user.

    Parameters
    ----------
    x: list, array, Series, DataFrame

    Notes
    ------
    This actually allows objects like timedeltas (converted to microseconds), and strings as numbers.

    """
    try:
        if isinstance(x, (pd.Series, pd.DataFrame)):
            v = pd.to_numeric(x.squeeze())
        else:
            v = pd.to_numeric(np.asarray(x).squeeze())

        if v.size == 0:
            raise ValueError("Empty array/Series passed in.")
        return v

    except:
        raise ValueError("Values must be numeric: no strings, datetimes, objects, etc.")

def coalesce(*args) -> Any:
    for arg in args:
        if arg is not None:
            return arg
    return None

class BaseFitter:

    weights: np.array
    event_observed: np.array

    def __init__(self, alpha: float = 0.05, label: str = None):
        if not (0 < alpha <= 1.0):
            raise ValueError("alpha parameter must be between 0 and 1.")
        self.alpha = alpha
        self._class_name = self.__class__.__name__
        self._label = label
        self._censoring_type = None

    def __repr__(self) -> str:
        classname = self._class_name
        if self._label:
            label_string = """"%s",""" % self._label
        else:
            label_string = ""
        try:
            s = """<lifelines.%s:%s fitted with %g total observations, %g %s-censored observations>""" % (
                classname,
                label_string,
                self.weights.sum(),
                self.weights.sum() - self.weights[self.event_observed > 0].sum(),
                CensoringType.str_censoring_type(self),
            )
        except AttributeError:
            s = """<lifelines.%s>""" % classname
        return s

    #@CensoringType.right_censoring
    def fit(*args, **kwargs):
        raise NotImplementedError()

    #@CensoringType.right_censoring
    def fit_right_censoring(self, *args, **kwargs):
        """Alias for ``fit``

        See Also
        ---------
        fit
        """
        return self.fit(*args, **kwargs)

class RegressionFitter(BaseFitter):

    _KNOWN_MODEL = False
    _FAST_MEDIAN_PREDICT = False
    _ALLOWED_RESIDUALS = {"schoenfeld", "score", "delta_beta", "deviance", "martingale", "scaled_schoenfeld"}

    def __init__(self, *args, **kwargs):
        super(RegressionFitter, self).__init__(*args, **kwargs)

    def plot_covariate_groups(*args, **kwargs):
        """
        Deprecated as of v0.25.0. Use ``plot_partial_effects_on_outcome`` instead.
        """
        warnings.warn("This method name is deprecated. Use `plot_partial_effects_on_outcome` instead.", DeprecationWarning)
        return self.plot_partial_effects_on_outcome(*args, **kwargs)

    def _compute_central_values_of_raw_training_data(self, df, strata=None, name="baseline"):
        """
        Compute our "baseline" observation for function like plot_partial_effects_on_outcome.
        - Categorical are transformed to their mode value.
        - Numerics are transformed to their median value.
        """
        if df.size == 0:
            return pd.DataFrame(index=["baseline"])

        if strata is not None:
            # apply this function within each stratified dataframe
            central_stats = []
            for stratum, df_ in df.groupby(strata):
                central_stats_ = self._compute_central_values_of_raw_training_data(df_, name=stratum)
                try:
                    central_stats_ = central_stats_.drop(strata, axis=1)
                except:
                    pass
                central_stats.append(central_stats_)
            v = pd.concat(central_stats)
            v.index.rename(make_simpliest_hashable(strata), inplace=True)
            return v

        else:
            from distversion import LooseVersion

            if LooseVersion(pd.__version__) >= "1.1.0":
                # silence deprecation warning
                describe_kwarg = {"datetime_is_numeric": True}
            else:
                describe_kwarg = {}
            described = df.describe(include="all", **describe_kwarg)
            if "top" in described.index and "50%" not in described.index:
                central_stats = described.loc["top"].copy()
            elif "50%" in described.index and "top" not in described.index:
                central_stats = described.loc["50%"].copy()
            elif "top" in described.index and "50%" in described.index:
                central_stats = described.loc["top"].copy()
                central_stats.update(described.loc["50%"])

            central_stats = central_stats.to_frame(name=name).T.astype(df.dtypes)
            return central_stats

    def compute_residuals(self, training_dataframe: pd.DataFrame, kind: str) -> pd.DataFrame:
        """
        Compute the residuals the model.

        Parameters
        ----------
        training_dataframe : DataFrame
            the same training DataFrame given in `fit`
        kind : string
            One of {'schoenfeld', 'score', 'delta_beta', 'deviance', 'martingale', 'scaled_schoenfeld'}

        Notes
        -------
        - ``'scaled_schoenfeld'``: *lifelines* does not add the coefficients to the final results, but R does when you call ``residuals(c, "scaledsch")``



        """
        assert kind in self._ALLOWED_RESIDUALS, "kind must be in %s" % self._ALLOWED_RESIDUALS
        if self.entry_col is not None:
            raise NotImplementedError("Residuals for entries not implemented.")

        warnings.filterwarnings("ignore", category=exceptions.ConvergenceWarning)
        X, Ts, E, weights, _, shuffled_original_index, _ = self._preprocess_dataframe(training_dataframe)

        resids = getattr(self, "_compute_%s" % kind)(X, Ts, E, weights, index=shuffled_original_index)
        return resids

class SemiParametricRegressionFitter(RegressionFitter):
    @property
    def AIC_partial_(self) -> float:
        """
        "partial" because the log-likelihood is partial
        """
        return -2 * self.log_likelihood_ + 2 * self.params_.shape[0]
    
class ProportionalHazardMixin:
    def check_assumptions(
        self,
        training_df: DataFrame,
        advice: bool = True,
        show_plots: bool = False,
        p_value_threshold: float = 0.01,
        plot_n_bootstraps: int = 15,
        columns: Optional[List[str]] = None,
    ) -> None:
        """
        Use this function to test the proportional hazards assumption. See usage example at
        https://lifelines.readthedocs.io/en/latest/jupyter_notebooks/Proportional%20hazard%20assumption.html


        Parameters
        -----------

        training_df: DataFrame
            the original DataFrame used in the call to ``fit(...)`` or a sub-sampled version.
        advice: bool, optional
            display advice as output to the user's screen
        show_plots: bool, optional
            display plots of the scaled Schoenfeld residuals and loess curves. This is an eyeball test for violations.
            This will slow down the function significantly.
        p_value_threshold: float, optional
            the threshold to use to alert the user of violations. See note below.
        plot_n_bootstraps:
            in the plots displayed, also display plot_n_bootstraps bootstrapped loess curves. This will slow down
            the function significantly.
        columns: list, optional
            specify a subset of columns to test.

        Returns
        --------
            A list of list of axes objects.


        Examples
        ----------

        .. code:: python

            from lifelines.datasets import load_rossi
            from lifelines import CoxPHFitter

            rossi = load_rossi()
            cph = CoxPHFitter().fit(rossi, 'week', 'arrest')

            axes = cph.check_assumptions(rossi, show_plots=True)


        Notes
        -------
        The ``p_value_threshold`` is arbitrarily set at 0.01. Under the null, some covariates
        will be below the threshold (i.e. by chance). This is compounded when there are many covariates.

        Similarly, when there are lots of observations, even minor deviances from the proportional hazard
        assumption will be flagged.

        With that in mind, it's best to use a combination of statistical tests and eyeball tests to
        determine the most serious violations.


        References
        -----------
        section 5 in https://socialsciences.mcmaster.ca/jfox/Books/Companion/appendices/Appendix-Cox-Regression.pdf,
        http://www.mwsug.org/proceedings/2006/stats/MWSUG-2006-SD08.pdf,
        http://eprints.lse.ac.uk/84988/1/06_ParkHendry2015-ReassessingSchoenfeldTests_Final.pdf
        """

        if not training_df.index.is_unique:
            raise IndexError(
                "`training_df` index should be unique for this exercise. Please make it unique or use `.reset_index(drop=True)` to force a unique index"
            )

        residuals = self.compute_residuals(training_df, kind="scaled_schoenfeld")
        test_results = proportional_hazard_test(self, training_df, time_transform=["rank", "km"], precomputed_residuals=residuals)

        residuals_and_duration = residuals.join(training_df[self.duration_col])
        Xs = self.regressors.transform_df(training_df)

        counter = 0
        n = residuals_and_duration.shape[0]
        axes = []

        for variable in self.params_.index & (columns or self.params_.index):
            minumum_observed_p_value = test_results.summary.loc[variable, "p"].min()
            if np.round(minumum_observed_p_value, 2) > p_value_threshold:
                continue

            counter += 1

            if counter == 1:
                if advice:
                    print(
                        fill(
                            """The ``p_value_threshold`` is set at %g. Even under the null hypothesis of no violations, some covariates will be below the threshold by chance. This is compounded when there are many covariates. Similarly, when there are lots of observations, even minor deviances from the proportional hazard assumption will be flagged."""
                            % p_value_threshold,
                            width=100,
                        )
                    )
                    print()
                    print(
                        fill(
                            """With that in mind, it's best to use a combination of statistical tests and visual tests to determine the most serious violations. Produce visual plots using ``check_assumptions(..., show_plots=True)`` and looking for non-constant lines. See link [A] below for a full example.""",
                            width=100,
                        )
                    )
                    print()
                test_results.print_summary()
                print()

            print()
            print(
                "%d. Variable '%s' failed the non-proportional test: p-value is %s."
                % (counter, variable, format_p_value(4)(minumum_observed_p_value)),
                end="\n\n",
            )

            if advice:
                values = Xs["beta_"][variable]
                value_counts = values.value_counts()
                n_uniques = value_counts.shape[0]

                # Arbitrary chosen to check for ability to use strata col.
                # This should capture dichotomous / low cardinality values.
                if n_uniques <= 6 and value_counts.min() >= 5:
                    print(
                        fill(
                            "   Advice: with so few unique values (only {0}), you can include `strata=['{1}', ...]` in the call in `.fit`. See documentation in link [E] below.".format(
                                n_uniques, variable
                            ),
                            width=100,
                        )
                    )
                else:
                    print(
                        fill(
                            """   Advice 1: the functional form of the variable '{var}' might be incorrect. That is, there may be non-linear terms missing. The proportional hazard test used is very sensitive to incorrect functional forms. See documentation in link [D] below on how to specify a functional form.""".format(
                                var=variable
                            ),
                            width=100,
                        ),
                        end="\n\n",
                    )
                    print(
                        fill(
                            """   Advice 2: try binning the variable '{var}' using pd.cut, and then specify it in `strata=['{var}', ...]` in the call in `.fit`. See documentation in link [B] below.""".format(
                                var=variable
                            ),
                            width=100,
                        ),
                        end="\n\n",
                    )
                    print(
                        fill(
                            """   Advice 3: try adding an interaction term with your time variable. See documentation in link [C] below.""",
                            width=100,
                        ),
                        end="\n\n",
                    )

            if show_plots:
                axes.append([])
                print()
                print("   Bootstrapping lowess lines. May take a moment...")
                print()
                from matplotlib import pyplot as plt

                fig = plt.figure()

                # plot variable against all time transformations.
                for i, (transform_name, transformer) in enumerate(TimeTransformers().iter(["rank", "km"]), start=1):
                    p_value = test_results.summary.loc[(variable, transform_name), "p"]

                    ax = fig.add_subplot(1, 2, i)

                    y = residuals_and_duration[variable]
                    tt = transformer(self.durations, self.event_observed, self.weights)[self.event_observed.values]

                    ax.scatter(tt, y, alpha=0.75)

                    y_lowess = lowess(tt.values, y.values)
                    ax.plot(tt, y_lowess, color="k", alpha=1.0, linewidth=2)

                    # bootstrap some possible other lowess lines. This is an approximation of the 100% confidence intervals
                    for _ in range(plot_n_bootstraps):
                        ix = sorted(np.random.choice(n, n))
                        tt_ = tt.values[ix]
                        y_lowess = lowess(tt_, y.values[ix])
                        ax.plot(tt_, y_lowess, color="k", alpha=0.30)

                    best_xlim = ax.get_xlim()
                    ax.hlines(0, 0, tt.max(), linestyles="dashed", linewidths=1)
                    ax.set_xlim(best_xlim)

                    ax.set_xlabel("%s-transformed time\n(p=%.4f)" % (transform_name, p_value), fontsize=10)
                    axes[-1].append(ax)

                fig.suptitle("Scaled Schoenfeld residuals of '%s'" % variable, fontsize=14)
                plt.tight_layout()
                plt.subplots_adjust(top=0.90)

        if advice and counter > 0:
            print(
                dedent(
                    r"""
                ---
                [A]  https://lifelines.readthedocs.io/en/latest/jupyter_notebooks/Proportional%20hazard%20assumption.html
                [B]  https://lifelines.readthedocs.io/en/latest/jupyter_notebooks/Proportional%20hazard%20assumption.html#Bin-variable-and-stratify-on-it
                [C]  https://lifelines.readthedocs.io/en/latest/jupyter_notebooks/Proportional%20hazard%20assumption.html#Introduce-time-varying-covariates
                [D]  https://lifelines.readthedocs.io/en/latest/jupyter_notebooks/Proportional%20hazard%20assumption.html#Modify-the-functional-form
                [E]  https://lifelines.readthedocs.io/en/latest/jupyter_notebooks/Proportional%20hazard%20assumption.html#Stratification
            """
                )
            )

        if counter == 0:
            print("Proportional hazard assumption looks okay.")
        return axes

    @property
    def hazard_ratios_(self):
        return pd.Series(np.exp(self.params_), index=self.params_.index, name="exp(coef)")

    def compute_followup_hazard_ratios(self, training_df: DataFrame, followup_times: Iterable) -> DataFrame:
        """
        Recompute the hazard ratio at different follow-up times (lifelines handles accounting for updated censoring and updated durations).
        This is useful because we need to remember that the hazard ratio is actually a weighted-average of period-specific hazard ratios.

        Parameters
        ----------

        training_df: pd.DataFrame
            The same dataframe used to train the model
        followup_times: Iterable
            a list/array of follow-up times to recompute the hazard ratio at.


        """
        results = {}
        for t in sorted(followup_times):
            assert t <= training_df[self.duration_col].max(), "all follow-up times must be less than max observed duration"
            df = training_df.copy()
            # if we "rollback" the df to time t, who is dead and who is censored
            df[self.event_col] = (df[self.duration_col] <= t) & df[self.event_col]
            df[self.duration_col] = np.minimum(df[self.duration_col], t)

            model = self.__class__(penalizer=self.penalizer, l1_ratio=self.l1_ratio).fit(
                df, self.duration_col, self.event_col, weights_col=self.weights_col, entry_col=self.entry_col
            )
            results[t] = model.hazard_ratios_
        return DataFrame(results).T

class CoxTimeVaryingFitter(SemiParametricRegressionFitter, ProportionalHazardMixin):
    r"""
    This class implements fitting Cox's time-varying proportional hazard model:

        .. math::  h(t|x(t)) = h_0(t)\exp((x(t)-\overline{x})'\beta)

    Parameters
    ----------
    alpha: float, optional (default=0.05)
       the level in the confidence intervals.
    penalizer: float, optional
        the coefficient of an L2 penalizer in the regression

    Attributes
    ----------
    params_ : Series
        The estimated coefficients. Changed in version 0.22.0: use to be ``.hazards_``
    hazard_ratios_ : Series
        The exp(coefficients)
    confidence_intervals_ : DataFrame
        The lower and upper confidence intervals for the hazard coefficients
    event_observed: Series
        The event_observed variable provided
    weights: Series
        The event_observed variable provided
    variance_matrix_ : DataFrame
        The variance matrix of the coefficients
    strata: list
        the strata provided
    standard_errors_: Series
        the standard errors of the estimates
    baseline_cumulative_hazard_: DataFrame
    baseline_survival_: DataFrame
    """

    _KNOWN_MODEL = True

    def __init__(self, alpha=0.05, penalizer=0.0, l1_ratio: float = 0.0, strata=None):
        super(CoxTimeVaryingFitter, self).__init__(alpha=alpha)
        self.alpha = alpha
        self.penalizer = penalizer
        self.strata = strata
        self.l1_ratio = l1_ratio

    def fit(
        self,
        df,
        event_col,
        start_col="start",
        stop_col="stop",
        weights_col=None,
        id_col=None,
        show_progress=False,
        step_size=None,
        robust=False,
        strata=None,
        initial_point=None,
        formula: str = None,
    ):  # pylint: disable=too-many-arguments
        """
        Fit the Cox Proportional Hazard model to a time varying dataset. Tied survival times
        are handled using Efron's tie-method.

        Parameters
        -----------
        df: DataFrame
            a Pandas DataFrame with necessary columns `duration_col` and
           `event_col`, plus other covariates. `duration_col` refers to
           the lifetimes of the subjects. `event_col` refers to whether
           the 'death' events was observed: 1 if observed, 0 else (censored).
        event_col: string
           the column in DataFrame that contains the subjects' death
           observation. If left as None, assume all individuals are non-censored.
        start_col: string
            the column that contains the start of a subject's time period.
        stop_col: string
            the column that contains the end of a subject's time period.
        weights_col: string, optional
            the column that contains (possibly time-varying) weight of each subject-period row.
        id_col: string, optional
            A subject could have multiple rows in the DataFrame. This column contains
           the unique identifier per subject. If not provided, it's up to the
           user to make sure that there are no violations.
        show_progress: since the fitter is iterative, show convergence
           diagnostics.
        robust: bool, optional (default: True)
            Compute the robust errors using the Huber sandwich estimator, aka Wei-Lin estimate. This does not handle
          ties, so if there are high number of ties, results may significantly differ. See
          "The Robust Inference for the Cox Proportional Hazards Model", Journal of the American Statistical Association, Vol. 84, No. 408 (Dec., 1989), pp. 1074- 1078
        step_size: float, optional
            set an initial step size for the fitting algorithm.
        strata: list or string, optional
            specify a column or list of columns n to use in stratification. This is useful if a
            categorical covariate does not obey the proportional hazard assumption. This
            is used similar to the `strata` expression in R.
            See http://courses.washington.edu/b515/l17.pdf.
        initial_point: (d,) numpy array, optional
            initialize the starting point of the iterative
            algorithm. Default is the zero vector.
        formula: str, optional
            A R-like formula for transforming the covariates

        Returns
        --------
        self: CoxTimeVaryingFitter
            self, with additional properties like ``hazards_`` and ``print_summary``

        """
        self.strata = coalesce(strata, self.strata)
        self.robust = robust
        if self.robust:
            raise NotImplementedError("Not available yet.")

        self.event_col = event_col
        self.id_col = id_col
        self.stop_col = stop_col
        self.start_col = start_col
        self.formula = formula
        self._time_fit_was_called = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S") + " UTC"

        df = df.copy()

        if not (event_col in df and start_col in df and stop_col in df):
            raise KeyError("A column specified in the call to `fit` does not exist in the DataFrame provided.")

        if weights_col is None:
            self.weights_col = None
            assert "__weights" not in df.columns, "__weights is an internal lifelines column, please rename your column first."
            df["__weights"] = 1.0
        else:
            self.weights_col = weights_col
            if (df[weights_col] <= 0).any():
                raise ValueError("values in weights_col must be positive.")

        df = df.rename(columns={event_col: "event", start_col: "start", stop_col: "stop", weights_col: "__weights"})

        if self.strata is not None and self.id_col is not None:
            df = df.set_index(_to_list(self.strata) + [id_col])
            df = df.sort_index()
        elif self.strata is not None and self.id_col is None:
            df = df.set_index(_to_list(self.strata))
        elif self.strata is None and self.id_col is not None:
            df = df.set_index([id_col])

        events, start, stop = (
            pass_for_numeric_dtypes_or_raise_array(df.pop("event")).astype(bool),
            df.pop("start"),
            df.pop("stop"),
        )
        weights = df.pop("__weights").astype(float)

        self.regressors = CovariateParameterMappings({"beta_": self.formula}, df, force_no_intercept=True)
        X = self.regressors.transform_df(df)["beta_"]

        self._check_values(X, events, start, stop)

        self._norm_mean = X.mean(0)
        self._norm_std = X.std(0)

        params_ = self._newton_rhaphson(
            normalize(X, self._norm_mean, self._norm_std),
            events,
            start,
            stop,
            weights,
            initial_point=initial_point,
            show_progress=show_progress,
            step_size=step_size,
        )

        self.params_ = pd.Series(params_, index=pd.Index(X.columns, name="covariate"), name="coef") / self._norm_std
        self.variance_matrix_ = pd.DataFrame(-inv(self._hessian_) / np.outer(self._norm_std, self._norm_std), index=X.columns)
        self.standard_errors_ = self._compute_standard_errors(
            normalize(X, self._norm_mean, self._norm_std), events, start, stop, weights
        )
        self.confidence_intervals_ = self._compute_confidence_intervals()
        self.baseline_cumulative_hazard_ = self._compute_cumulative_baseline_hazard(df, events, start, stop, weights)
        self.baseline_survival_ = self._compute_baseline_survival()
        self.event_observed = events
        self.start_stop_and_events = pd.DataFrame({"event": events, "start": start, "stop": stop})
        self.weights = weights

        self._n_examples = X.shape[0]
        self._n_unique = X.index.unique().shape[0]
        return self

    def _check_values(self, df, events, start, stop):
        # check_for_overlapping_intervals(df) # this is currently too slow for production.
        check_nans_or_infs(df)
        check_low_var(df)
        check_complete_separation_low_variance(df, events, self.event_col)
        check_for_numeric_dtypes_or_raise(df)
        check_for_nonnegative_intervals(start, stop)
        check_for_immediate_deaths(events, start, stop)
        check_for_instantaneous_events_at_time_zero(start, stop)
        check_for_instantaneous_events_at_death_time(events, start, stop)

    def _partition_by_strata(self, X, events, start, stop, weights):
        for stratum, stratified_X in X.groupby(self.strata):
            stratified_W = weights.loc[stratum]
            stratified_start = start.loc[stratum]
            stratified_events = events.loc[stratum]
            stratified_stop = stop.loc[stratum]
            yield (
                stratified_X.values,
                stratified_events.values,
                stratified_start.values,
                stratified_stop.values,
                stratified_W.values,
            ), stratum

    def _partition_by_strata_and_apply(self, X, events, start, stop, weights, function, *args):
        for ((stratified_X, stratified_events, stratified_start, stratified_stop, stratified_W), _) in self._partition_by_strata(
            X, events, start, stop, weights
        ):
            yield function(stratified_X, stratified_events, stratified_start, stratified_stop, stratified_W, *args)

    def _compute_z_values(self):
        return self.params_ / self.standard_errors_

    def _compute_p_values(self):
        U = self._compute_z_values() ** 2
        return stats.chi2.sf(U, 1)

    def _compute_confidence_intervals(self):
        ci = 100 * (1 - self.alpha)
        z = inv_normal_cdf(1 - self.alpha / 2)
        se = self.standard_errors_
        hazards = self.params_.values
        return pd.DataFrame(
            np.c_[hazards - z * se, hazards + z * se],
            columns=["%g%% lower-bound" % ci, "%g%% upper-bound" % ci],
            index=self.params_.index,
        )

    @property
    def summary(self):
        """Summary statistics describing the fit.

        Returns
        -------
        df : DataFrame
            Contains columns coef, np.exp(coef), se(coef), z, p, lower, upper"""
        ci = 100 * (1 - self.alpha)
        z = inv_normal_cdf(1 - self.alpha / 2)
        with np.errstate(invalid="ignore", divide="ignore", over="ignore", under="ignore"):
            df = pd.DataFrame(index=self.params_.index)
            df["coef"] = self.params_
            df["exp(coef)"] = self.hazard_ratios_
            df["se(coef)"] = self.standard_errors_
            df["coef lower %g%%" % ci] = self.confidence_intervals_["%g%% lower-bound" % ci]
            df["coef upper %g%%" % ci] = self.confidence_intervals_["%g%% upper-bound" % ci]
            df["exp(coef) lower %g%%" % ci] = self.hazard_ratios_ * np.exp(-z * self.standard_errors_)
            df["exp(coef) upper %g%%" % ci] = self.hazard_ratios_ * np.exp(z * self.standard_errors_)
            df["z"] = self._compute_z_values()
            df["p"] = self._compute_p_values()
            df["-log2(p)"] = -quiet_log2(df["p"])
            return df

    def _newton_rhaphson(
        self,
        df,
        events,
        start,
        stop,
        weights,
        show_progress=False,
        step_size=None,
        precision=10e-6,
        max_steps=50,
        initial_point=None,
    ):  # pylint: disable=too-many-arguments,too-many-locals,too-many-branches,too-many-statements
        """
        Newton Rhaphson algorithm for fitting CPH model.

        Parameters
        ----------
        df: DataFrame
        stop_times_events: DataFrame
             meta information about the subjects history
        show_progress: bool, optional (default: True)
            to show verbose output of convergence
        step_size: float
            > 0 to determine a starting step size in NR algorithm.
        precision: float
            the convergence halts if the norm of delta between
                     successive positions is less than epsilon.

        Returns
        --------
        beta: (1,d) numpy array.
        """
        assert precision <= 1.0, "precision must be less than or equal to 1."

        # soft penalizer functions, from https://www.cs.ubc.ca/cgi-bin/tr/2009/TR-2009-19.pdf
        soft_abs = lambda x, a: 1 / a * (anp.logaddexp(0, -a * x) + anp.logaddexp(0, a * x))
        penalizer = (
            lambda beta, a: n
            * (self.penalizer * (self.l1_ratio * (soft_abs(beta, a)) + 0.5 * (1 - self.l1_ratio) * (beta ** 2))).sum()
        )
        d_penalizer = elementwise_grad(penalizer)
        dd_penalizer = elementwise_grad(d_penalizer)

        n, d = df.shape

        # make sure betas are correct size.
        if initial_point is not None:
            beta = initial_point
        else:
            beta = np.zeros((d,))

        i = 0
        converging = True
        ll, previous_ll = 0, 0
        start_time = time.time()

        step_sizer = StepSizer(step_size)
        step_size = step_sizer.next()

        while converging:
            i += 1

            if self.strata is None:
                h, g, ll = self._get_gradients(df.values, events.values, start.values, stop.values, weights.values, beta)
            else:
                g = np.zeros_like(beta)
                h = np.zeros((d, d))
                ll = 0
                for _h, _g, _ll in self._partition_by_strata_and_apply(
                    df, events, start, stop, weights, self._get_gradients, beta
                ):
                    g += _g
                    h += _h
                    ll += _ll

            if i == 1 and np.all(beta == 0):
                # this is a neat optimization, the null partial likelihood
                # is the same as the full partial but evaluated at zero.
                # if the user supplied a non-trivial initial point, we need to delay this.
                self._log_likelihood_null = ll

            if isinstance(self.penalizer, np.ndarray) or self.penalizer > 0:
                ll -= penalizer(beta, 1.5 ** i)
                g -= d_penalizer(beta, 1.5 ** i)
                h[np.diag_indices(d)] -= dd_penalizer(beta, 1.5 ** i)

            try:
                # reusing a piece to make g * inv(h) * g.T faster later
                inv_h_dot_g_T = spsolve(-h, g, sym_pos=True)
            except ValueError as e:
                if "infs or NaNs" in str(e):
                    raise ConvergenceError(
                        """hessian or gradient contains nan or inf value(s). Convergence halted. Please see the following tips in the lifelines documentation:
    https://lifelines.readthedocs.io/en/latest/Examples.html#problems-with-convergence-in-the-cox-proportional-hazard-model
    """,
                        e,
                    )
                else:
                    # something else?
                    raise e
            except LinAlgError as e:
                raise ConvergenceError(
                    """Convergence halted due to matrix inversion problems. Suspicion is high colinearity. Please see the following tips in the lifelines documentation:
    https://lifelines.readthedocs.io/en/latest/Examples.html#problems-with-convergence-in-the-cox-proportional-hazard-model
    """,
                    e,
                )

            delta = step_size * inv_h_dot_g_T

            if np.any(np.isnan(delta)):
                raise ConvergenceError(
                    """delta contains nan value(s). Convergence halted. Please see the following tips in the lifelines documentation:
    https://lifelines.readthedocs.io/en/latest/Examples.html#problems-with-convergence-in-the-cox-proportional-hazard-model
    """
                )
            # Save these as pending result
            hessian, gradient = h, g
            norm_delta = norm(delta)
            newton_decrement = g.dot(inv_h_dot_g_T) / 2

            if show_progress:
                print(
                    "\rIteration %d: norm_delta = %.5f, step_size = %.5f, ll = %.5f, newton_decrement = %.5f, seconds_since_start = %.1f"
                    % (i, norm_delta, step_size, ll, newton_decrement, time.time() - start_time),
                    end="",
                )

            # convergence criteria
            if norm_delta < precision:
                converging, completed = False, True
            elif previous_ll > 0 and abs(ll - previous_ll) / (-previous_ll) < 1e-09:
                # this is what R uses by default
                converging, completed = False, True
            elif newton_decrement < 10e-8:
                converging, completed = False, True
            elif i >= max_steps:
                # 50 iterations steps with N-R is a lot.
                # Expected convergence is less than 10 steps
                converging, completed = False, False
            elif step_size <= 0.0001:
                converging, completed = False, False
            elif abs(ll) < 0.0001 and norm_delta > 1.0:
                warnings.warn(
                    "The log-likelihood is getting suspiciously close to 0 and the delta is still large. There may be complete separation in the dataset. This may result in incorrect inference of coefficients. \
    See https://stats.stackexchange.com/questions/11109/how-to-deal-with-perfect-separation-in-logistic-regression\n",
                    ConvergenceWarning,
                )
                converging, completed = False, False

            step_size = step_sizer.update(norm_delta).next()

            beta += delta

        self._hessian_ = hessian
        self._score_ = gradient
        self.log_likelihood_ = ll

        if show_progress and completed:
            print("Convergence completed after %d iterations." % (i))
        elif show_progress and not completed:
            print("Convergence failed. See any warning messages.")

        # report to the user problems that we detect.
        if completed and norm_delta > 0.1:
            warnings.warn(
                "Newton-Rhapson convergence completed but norm(delta) is still high, %.3f. This may imply non-unique solutions to the maximum likelihood. Perhaps there is colinearity or complete separation in the dataset?"
                % norm_delta,
                ConvergenceWarning,
            )
        elif not completed:
            warnings.warn("Newton-Rhapson failed to converge sufficiently in %d steps." % max_steps, ConvergenceWarning)

        return beta

    @staticmethod
    def _get_gradients(X, events, start, stop, weights, beta):  # pylint: disable=too-many-locals
        """
        Calculates the first and second order vector differentials, with respect to beta.

        Returns
        -------
        hessian: (d, d) numpy array,
        gradient: (1, d) numpy array
        log_likelihood: float
        """

        _, d = X.shape
        hessian = np.zeros((d, d))
        gradient = np.zeros(d)
        log_lik = 0
        unique_death_times = np.unique(stop[events])

        for t in unique_death_times:

            # I feel like this can be made into some tree-like structure
            ix = (start < t) & (t <= stop)

            X_at_t = X[ix]
            weights_at_t = weights[ix]
            stops_events_at_t = stop[ix]
            events_at_t = events[ix]

            phi_i = weights_at_t * np.exp(np.dot(X_at_t, beta))
            phi_x_i = phi_i[:, None] * X_at_t
            phi_x_x_i = np.dot(X_at_t.T, phi_x_i)

            # Calculate sums of Risk set
            risk_phi = array_sum_to_scalar(phi_i)
            risk_phi_x = matrix_axis_0_sum_to_1d_array(phi_x_i)
            risk_phi_x_x = phi_x_x_i

            # Calculate the sums of Tie set
            deaths = events_at_t & (stops_events_at_t == t)

            tied_death_counts = array_sum_to_scalar(deaths.astype(int))  # should always at least 1. Why? TODO

            xi_deaths = X_at_t[deaths]

            x_death_sum = matrix_axis_0_sum_to_1d_array(weights_at_t[deaths, None] * xi_deaths)

            weight_count = array_sum_to_scalar(weights_at_t[deaths])
            weighted_average = weight_count / tied_death_counts

            #
            # This code is near identical to the _batch algorithm in CoxPHFitter. In fact, see _batch for comments.
            #

            if tied_death_counts > 1:

                # A good explanation for how Efron handles ties. Consider three of five subjects who fail at the time.
                # As it is not known a priori that who is the first to fail, so one-third of
                # (φ1 + φ2 + φ3) is adjusted from sum_j^{5} φj after one fails. Similarly two-third
                # of (φ1 + φ2 + φ3) is adjusted after first two individuals fail, etc.

                # a lot of this is now in Einstein notation for performance, but see original "expanded" code here
                # https://github.com/CamDavidsonPilon/lifelines/blob/e7056e7817272eb5dff5983556954f56c33301b1/lifelines/fitters/cox_time_varying_fitter.py#L458-L490

                tie_phi = array_sum_to_scalar(phi_i[deaths])
                tie_phi_x = matrix_axis_0_sum_to_1d_array(phi_x_i[deaths])
                tie_phi_x_x = np.dot(xi_deaths.T, phi_i[deaths, None] * xi_deaths)

                increasing_proportion = np.arange(tied_death_counts) / tied_death_counts
                denom = 1.0 / (risk_phi - increasing_proportion * tie_phi)
                numer = risk_phi_x - np.outer(increasing_proportion, tie_phi_x)

                a1 = np.einsum("ab, i->ab", risk_phi_x_x, denom) - np.einsum(
                    "ab, i->ab", tie_phi_x_x, increasing_proportion * denom
                )
            else:
                # no tensors here, but do some casting to make it easier in the converging step next.
                denom = 1.0 / np.array([risk_phi])
                numer = risk_phi_x
                a1 = risk_phi_x_x * denom

            summand = numer * denom[:, None]
            a2 = summand.T.dot(summand)

            gradient = gradient + x_death_sum - weighted_average * summand.sum(0)
            log_lik = log_lik + np.dot(x_death_sum, beta) + weighted_average * np.log(denom).sum()
            hessian = hessian + weighted_average * (a2 - a1)

        return hessian, gradient, log_lik

    def predict_log_partial_hazard(self, X) -> pd.Series:
        r"""
        This is equivalent to R's linear.predictors.
        Returns the log of the partial hazard for the individuals, partial since the
        baseline hazard is not included. Equal to :math:`(x - \bar{x})'\beta`


        Parameters
        ----------
        X: numpy array or DataFrame
            a (n,d) covariate numpy array or DataFrame. If a DataFrame, columns
            can be in any order. If a numpy array, columns must be in the
            same order as the training data.

        Returns
        -------
        DataFrame

        Note
        -----
        If X is a DataFrame, the order of the columns do not matter. But
        if X is an array, then the column ordering is assumed to be the
        same as the training dataset.
        """
        hazard_names = self.params_.index

        if isinstance(X, pd.DataFrame):
            X = self.regressors.transform_df(X)["beta_"]
            X = X.values

        X = X.astype(float)
        index = _get_index(X)
        X = normalize(X, self._norm_mean.values, 1)
        return pd.Series(np.dot(X, self.params_), index=index)

    def predict_partial_hazard(self, X) -> pd.Series:
        r"""
        Returns the partial hazard for the individuals, partial since the
        baseline hazard is not included. Equal to :math:`\exp{(x - \bar{x})'\beta }`

        Parameters
        ----------
        X: numpy array or DataFrame
            a (n,d) covariate numpy array or DataFrame. If a DataFrame, columns
            can be in any order. If a numpy array, columns must be in the
            same order as the training data.

        Returns
        -------
        DataFrame

        Note
        -----
        If X is a DataFrame, the order of the columns do not matter. But
        if X is an array, then the column ordering is assumed to be the
        same as the training dataset.

        """
        return np.exp(self.predict_log_partial_hazard(X))

    def print_summary(self, decimals=2, style=None, columns=None, **kwargs):
        """
        Print summary statistics describing the fit, the coefficients, and the error bounds.

        Parameters
        -----------
        decimals: int, optional (default=2)
            specify the number of decimal places to show
        style: string
            {html, ascii, latex}
        columns:
            only display a subset of ``summary`` columns. Default all.
        kwargs:
            print additional meta data in the output (useful to provide model names, dataset names, etc.) when comparing
            multiple outputs.

        """
        justify = string_rjustify(18)

        headers = []

        if self.event_col:
            headers.append(("event col", "'%s'" % self.event_col))
        if self.weights_col:
            headers.append(("weights col", "'%s'" % self.weights_col))
        if isinstance(self.penalizer, np.ndarray) or self.penalizer > 0:
            headers.append(("penalizer", self.penalizer))
        if self.strata:
            headers.append(("strata", self.strata))

        headers.extend(
            [
                ("number of subjects", self._n_unique),
                ("number of periods", self._n_examples),
                ("number of events", self.event_observed.sum()),
                ("partial log-likelihood", "{:.{prec}f}".format(self.log_likelihood_, prec=decimals)),
                ("time fit was run", self._time_fit_was_called),
            ]
        )

        sr = self.log_likelihood_ratio_test()
        footers = []
        footers.extend(
            [
                ("Partial AIC", "{:.{prec}f}".format(self.AIC_partial_, prec=decimals)),
                (
                    "log-likelihood ratio test",
                    "{:.{prec}f} on {} df".format(sr.test_statistic, sr.degrees_freedom, prec=decimals),
                ),
                ("-log2(p) of ll-ratio test", "{:.{prec}f}".format(-quiet_log2(sr.p_value), prec=decimals)),
            ]
        )

        p = Printer(self, headers, footers, justify, kwargs, decimals, columns)
        p.print(style=style)

    def log_likelihood_ratio_test(self):
        """
        This function computes the likelihood ratio test for the Cox model. We
        compare the existing model (with all the covariates) to the trivial model
        of no covariates.

        Conveniently, we can actually use CoxPHFitter class to do most of the work.

        """
        if hasattr(self, "_log_likelihood_null"):
            ll_null = self._log_likelihood_null

        else:
            trivial_dataset = self.start_stop_and_events
            trivial_dataset = trivial_dataset.join(self.weights)
            trivial_dataset = trivial_dataset.reset_index()
            ll_null = (
                CoxTimeVaryingFitter()
                .fit(
                    trivial_dataset,
                    start_col=self.start_col,
                    stop_col=self.stop_col,
                    event_col=self.event_col,
                    id_col=self.id_col,
                    weights_col="__weights",
                    strata=self.strata,
                )
                .log_likelihood_
            )

        ll_alt = self.log_likelihood_
        test_stat = 2 * (ll_alt - ll_null)
        degrees_freedom = self.params_.shape[0]
        p_value = _chisq_test_p_value(test_stat, degrees_freedom=degrees_freedom)
        return StatisticalResult(
            p_value, test_stat, name="log-likelihood ratio test", degrees_freedom=degrees_freedom, null_distribution="chi squared"
        )

    def plot(self, columns=None, ax=None, **errorbar_kwargs):
        """
        Produces a visual representation of the coefficients, including their standard errors and magnitudes.

        Parameters
        ----------
        columns : list, optional
            specify a subset of the columns to plot
        errorbar_kwargs:
            pass in additional plotting commands to matplotlib errorbar command

        Returns
        -------
        ax: matplotlib axis
            the matplotlib axis that be edited.

        """
        from matplotlib import pyplot as plt

        if ax is None:
            ax = plt.gca()

        errorbar_kwargs.setdefault("c", "k")
        errorbar_kwargs.setdefault("fmt", "s")
        errorbar_kwargs.setdefault("markerfacecolor", "white")
        errorbar_kwargs.setdefault("markeredgewidth", 1.25)
        errorbar_kwargs.setdefault("elinewidth", 1.25)
        errorbar_kwargs.setdefault("capsize", 3)

        z = inv_normal_cdf(1 - self.alpha / 2)

        if columns is None:
            user_supplied_columns = False
            columns = self.params_.index
        else:
            user_supplied_columns = True

        yaxis_locations = list(range(len(columns)))
        symmetric_errors = z * self.standard_errors_[columns].values.copy()
        hazards = self.params_[columns].values.copy()

        order = list(range(len(columns) - 1, -1, -1)) if user_supplied_columns else np.argsort(hazards)

        ax.errorbar(hazards[order], yaxis_locations, xerr=symmetric_errors[order], **errorbar_kwargs)
        best_ylim = ax.get_ylim()
        ax.vlines(0, -2, len(columns) + 1, linestyles="dashed", linewidths=1, alpha=0.65, color="k")
        ax.set_ylim(best_ylim)

        tick_labels = [columns[i] for i in order]

        ax.set_yticks(yaxis_locations)
        ax.set_yticklabels(tick_labels)
        ax.set_xlabel("log(HR) (%g%% CI)" % ((1 - self.alpha) * 100))

        return ax

    def _compute_cumulative_baseline_hazard(self, tv_data, events, start, stop, weights):  # pylint: disable=too-many-locals

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hazards = self.predict_partial_hazard(tv_data).values

        unique_death_times = np.unique(stop[events.values])
        baseline_hazard_ = pd.DataFrame(np.zeros_like(unique_death_times), index=unique_death_times, columns=["baseline hazard"])

        for t in unique_death_times:
            ix = (start.values < t) & (t <= stop.values)

            events_at_t = events.values[ix]
            stops_at_t = stop.values[ix]
            weights_at_t = weights.values[ix]
            hazards_at_t = hazards[ix]

            deaths = events_at_t & (stops_at_t == t)

            death_counts = (weights_at_t.squeeze() * deaths).sum()  # should always be atleast 1.
            baseline_hazard_.loc[t] = death_counts / hazards_at_t.sum()

        return baseline_hazard_.cumsum()

    def _compute_baseline_survival(self):
        survival_df = np.exp(-self.baseline_cumulative_hazard_)
        survival_df.columns = ["baseline survival"]
        return survival_df

    def __repr__(self):
        classname = self._class_name
        try:
            s = """<lifelines.%s: fitted with %d periods, %d subjects, %d events>""" % (
                classname,
                self._n_examples,
                self._n_unique,
                self.event_observed.sum(),
            )
        except AttributeError:
            s = """<lifelines.%s>""" % classname
        return s

    def _compute_residuals(self, df, events, start, stop, weights):
        raise NotImplementedError()

    def _compute_delta_beta(self, df, events, start, stop, weights):
        """ approximate change in betas as a result of excluding ith row"""

        score_residuals = self._compute_residuals(df, events, start, stop, weights) * weights[:, None]

        naive_var = inv(self._hessian_)
        delta_betas = -score_residuals.dot(naive_var) / self._norm_std.values

        return delta_betas

    def _compute_sandwich_estimator(self, X, events, start, stop, weights):

        delta_betas = self._compute_delta_beta(X, events, start, stop, weights)

        if self.cluster_col:
            delta_betas = pd.DataFrame(delta_betas).groupby(self._clusters).sum().values

        sandwich_estimator = delta_betas.T.dot(delta_betas)
        return sandwich_estimator

    def _compute_standard_errors(self, X, events, start, stop, weights):
        if self.robust:
            se = np.sqrt(self._compute_sandwich_estimator(X, events, start, stop, weights).diagonal())
        else:
            se = np.sqrt(self.variance_matrix_.values.diagonal())
        return pd.Series(se, index=self.params_.index, name="se")
