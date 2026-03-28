"""Performance and risk-adjusted return metrics for financial data."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import polars as pl
from scipy.stats import norm

from ._core import columnwise_stat, to_frame

# ── Performance statistics mixin ─────────────────────────────────────────────


class _PerformanceStatsMixin:
    """Mixin providing performance, drawdown, and benchmark/factor metrics.

    Covers: Sharpe ratio, Sortino ratio, adjusted Sortino, drawdown series,
    max drawdown, prices, R-squared, information ratio, and Greeks (alpha/beta).

    Attributes (provided by the concrete subclass):
        data: The :class:`~jquantstats._data.Data` object.
        all: Combined DataFrame for efficient column selection.
    """

    if TYPE_CHECKING:
        from ._protocol import DataLike

        data: DataLike
        all: pl.DataFrame | None

        def autocorr_penalty(self) -> dict[str, float]:
            """Defined on _BasicStatsMixin."""

        def geometric_mean(self) -> dict[str, float]:
            """Defined on _BasicStatsMixin."""

    # ── Sharpe & Sortino ──────────────────────────────────────────────────────

    @columnwise_stat
    def sharpe(self, series: pl.Series, periods: int | float | None = None) -> float:
        """Calculate the Sharpe ratio of asset returns.

        Args:
            series (pl.Series): The series to calculate Sharpe ratio for.
            periods (int, optional): Number of periods per year. Defaults to 252.

        Returns:
            float: The Sharpe ratio value.

        """
        periods = periods or self.data._periods_per_year

        std_val = series.std(ddof=1)
        mean_val = series.mean()
        divisor = cast(float, std_val) if std_val is not None else 0.0
        mean_f = cast(float, mean_val) if mean_val is not None else 0.0

        _eps = np.finfo(np.float64).eps
        if divisor <= _eps * max(abs(mean_f), _eps) * 10:
            return float("nan")

        res = mean_f / divisor
        factor = periods or 1
        return float(res * np.sqrt(factor))

    @columnwise_stat
    def sharpe_variance(self, series: pl.Series, periods: int | float | None = None) -> float:
        r"""Calculate the asymptotic variance of the Sharpe Ratio.

        .. math::
            \text{Var}(SR) = \frac{1 + \frac{S \cdot SR}{2} + \frac{(K - 3) \cdot SR^2}{4}}{T}

        where:
            - \(S\) is the skewness of returns
            - \(K\) is the kurtosis of returns
            - \(SR\) is the Sharpe ratio (unannualized)
            - \(T\) is the number of observations

        Args:
            series (pl.Series): The series to calculate Sharpe ratio variance for.
            periods (int | float, optional): Number of periods per year. Defaults to data periods.

        Returns:
            float: The asymptotic variance of the Sharpe ratio.
            If number of periods per year is provided or inferred from the data, the result is annualized.

        """
        t = series.count()
        mean_val = cast(float, series.mean())
        std_val = cast(float, series.std(ddof=1))
        if mean_val is None or std_val is None or std_val == 0:
            return float(np.nan)
        sr = mean_val / std_val

        skew_val = series.skew(bias=False)
        kurt_val = series.kurtosis(bias=False)

        if skew_val is None or kurt_val is None:
            return float(np.nan)
        # Base variance calculation using unannualized Sharpe ratio
        # Formula: (1 + skew*SR/2 + (kurt-3)*SR²/4) / T
        base_variance = (1 + (float(skew_val) * sr) / 2 + ((float(kurt_val) - 3) / 4) * sr**2) / t
        # Annualize by scaling with the number of periods
        periods = periods or self.data._periods_per_year
        factor = periods or 1
        return float(base_variance * factor)

    @columnwise_stat
    def probabilistic_sharpe_ratio(self, series: pl.Series) -> float:
        r"""Calculate the probabilistic sharpe ratio (PSR).

        Args:
            series (pl.Series): The series to calculate probabilistic Sharpe ratio for.

        Returns:
            float: Probabilistic Sharpe Ratio.

        Note:
            PSR is the probability that the observed Sharpe ratio is greater than a
            given benchmark Sharpe ratio.

        """
        t = series.count()

        # Calculate observed unannualized Sharpe ratio
        mean_val = cast(float, series.mean())
        std_val = cast(float, series.std(ddof=1))
        if mean_val is None or std_val is None or std_val == 0:
            return float(np.nan)
        # Unannualized observed Sharpe ratio
        observed_sr = mean_val / std_val

        skew_val = series.skew(bias=False)
        kurt_val = series.kurtosis(bias=False)

        if skew_val is None or kurt_val is None:
            return float(np.nan)

        benchmark_sr = 0.0
        # Calculate variance using unannualized benchmark Sharpe ratio
        var_bench_sr = (1 + (float(skew_val) * benchmark_sr) / 2 + ((float(kurt_val) - 3) / 4) * benchmark_sr**2) / t

        if var_bench_sr <= 0:
            return float(np.nan)
        return float(norm.cdf((observed_sr - benchmark_sr) / np.sqrt(var_bench_sr)))

    @columnwise_stat
    def hhi_positive(self, series: pl.Series) -> float:
        r"""Calculate the Herfindahl-Hirschman Index (HHI) for positive returns.

        This quantifies how concentrated the positive returns are in a series.

        .. math::
            w^{\plus} = \frac{r_{t}^{\plus}}{\sum{r_{t}^{\plus}}} \\
            HHI^{\plus} = \frac{N_{\plus} \sum{(w^{\plus})^2} - 1}{N_{\plus} - 1}

        where:
            - \(r_{t}^{\plus}\) are the positive returns
            - \(N_{\plus}\) is the number of positive returns
            - \(w^{\plus}\) are the weights of positive returns

        Args:
            series (pl.Series): The series to calculate HHI for.

        Returns:
            float: The HHI value for positive returns. Returns NaN if fewer than 3
                positive returns are present.

        Note:
            Values range from 0 (perfectly diversified gains) to 1 (all gains
            concentrated in a single period).
        """
        positive_returns = series.filter(series > 0).drop_nans()
        if positive_returns.len() <= 2:
            return float(np.nan)
        weight = positive_returns / positive_returns.sum()
        return float((weight.len() * (weight**2).sum() - 1) / (weight.len() - 1))

    @columnwise_stat
    def hhi_negative(self, series: pl.Series) -> float:
        r"""Calculate the Herfindahl-Hirschman Index (HHI) for negative returns.

        This quantifies how concentrated the negative returns are in a series.

        .. math::
            w^{\minus} = \frac{r_{t}^{\minus}}{\sum{r_{t}^{\minus}}} \\
            HHI^{\minus} = \frac{N_{\minus} \sum{(w^{\minus})^2} - 1}{N_{\minus} - 1}

        where:
            - \(r_{t}^{\minus}\) are the negative returns
            - \(N_{\minus}\) is the number of negative returns
            - \(w^{\minus}\) are the weights of negative returns

        Args:
            series (pl.Series): The returns series to calculate HHI for.

        Returns:
            float: The HHI value for negative returns. Returns NaN if fewer than 3
                negative returns are present.

        Note:
            Values range from 0 (perfectly diversified losses) to 1 (all losses
            concentrated in a single period).
        """
        negative_returns = series.filter(series < 0).drop_nans()
        if negative_returns.len() <= 2:
            return float(np.nan)
        weight = negative_returns / negative_returns.sum()
        return float((weight.len() * (weight**2).sum() - 1) / (weight.len() - 1))

    @columnwise_stat
    def sortino(self, series: pl.Series, periods: int | float | None = None) -> float:
        """Calculate the Sortino ratio.

        The Sortino ratio is the mean return divided by downside deviation.
        Based on Red Rock Capital's Sortino ratio paper.

        Args:
            series (pl.Series): The series to calculate Sortino ratio for.
            periods (int, optional): Number of periods per year. Defaults to 252.

        Returns:
            float: The Sortino ratio value.

        """
        periods = periods or self.data._periods_per_year
        downside_sum = ((series.filter(series < 0)) ** 2).sum()
        downside_deviation = float(np.sqrt(float(downside_sum) / series.count()))
        mean_val = cast(float, series.mean())
        mean_f = mean_val if mean_val is not None else 0.0
        if downside_deviation == 0.0:
            if mean_f > 0:
                return float("inf")
            elif mean_f < 0:  # pragma: no cover  # unreachable: no negatives ⟹ mean ≥ 0
                return float("-inf")
            else:
                return float("nan")
        ratio = mean_f / downside_deviation
        return float(ratio * np.sqrt(periods))

    @columnwise_stat
    def omega(
        self,
        series: pl.Series,
        rf: float = 0.0,
        required_return: float = 0.0,
        periods: int | float | None = None,
    ) -> float:
        """Calculate the Omega ratio.

        The Omega ratio is the probability-weighted ratio of gains to losses
        relative to a threshold return.  It is computed as the sum of returns
        above the threshold divided by the absolute sum of returns below it.

        Args:
            series (pl.Series): The series to calculate Omega ratio for.
            rf (float): Annualised risk-free rate. Defaults to 0.0.
            required_return (float): Annualised minimum acceptable return
                threshold. Defaults to 0.0.
            periods (int | float | None): Number of periods per year. Defaults
                to the value inferred from the data.

        Returns:
            float: The Omega ratio, or NaN when the denominator is zero or
                when ``required_return <= -1``.

        Note:
            See https://en.wikipedia.org/wiki/Omega_ratio for details.

        """
        if required_return <= -1:
            return float("nan")

        periods = periods or self.data._periods_per_year

        # Subtract per-period risk-free rate from returns when rf is non-zero.
        if rf != 0.0:
            rf_per_period = float((1.0 + rf) ** (1.0 / periods) - 1.0)
            series = series - rf_per_period

        # Convert annualised required return to a per-period threshold.
        return_threshold = float((1.0 + required_return) ** (1.0 / periods) - 1.0)

        returns_less_thresh = series - return_threshold

        numer = float(returns_less_thresh.filter(returns_less_thresh > 0.0).sum())
        denom = float(-returns_less_thresh.filter(returns_less_thresh < 0.0).sum())

        if denom <= 0.0:
            return float("nan")
        return numer / denom

    # ── Cumulative returns ────────────────────────────────────────────────────

    @to_frame
    def compsum(self, series: pl.Expr) -> pl.Expr:
        """Calculate the rolling compounded (cumulative) returns.

        Computed as cumprod(1 + r) - 1 for each period.

        Args:
            series (pl.Expr): The expression to calculate cumulative returns for.

        Returns:
            pl.Expr: Cumulative compounded returns expression.

        """
        return (1.0 + series).cum_prod() - 1.0

    def ghpr(self) -> dict[str, float]:
        """Calculate the Geometric Holding Period Return.

        Shorthand for geometric_mean() — the per-period geometric average return.

        Returns:
            dict[str, float]: Dictionary mapping asset names to GHPR values.

        """
        return self.geometric_mean()

    # ── Drawdown ──────────────────────────────────────────────────────────────

    @to_frame
    def drawdown(self, series: pl.Series) -> pl.Series:
        """Calculate the drawdown series for returns.

        Args:
            series (pl.Series): The series to calculate drawdown for.

        Returns:
            pl.Series: The drawdown series.

        """
        equity = self.prices(series)
        d = (equity / equity.cum_max()) - 1
        return -d

    @staticmethod
    def prices(series: pl.Series) -> pl.Series:
        """Convert returns series to price series.

        Args:
            series (pl.Series): The returns series to convert.

        Returns:
            pl.Series: The price series.

        """
        return (1.0 + series).cum_prod()

    @staticmethod
    def max_drawdown_single_series(series: pl.Series) -> float:
        """Compute the maximum drawdown for a single returns series.

        Args:
            series: A Polars Series of returns values.

        Returns:
            float: The maximum drawdown as a positive fraction (e.g. 0.2 for 20%).
        """
        price = _PerformanceStatsMixin.prices(series)
        peak = price.cum_max()
        drawdown = price / peak - 1
        dd_min = cast(float, drawdown.min())
        return dd_min if dd_min is not None else 0.0

    @columnwise_stat
    def max_drawdown(self, series: pl.Series) -> float:
        """Calculate the maximum drawdown for each column.

        Args:
            series (pl.Series): The series to calculate maximum drawdown for.

        Returns:
            float: The maximum drawdown value.

        """
        return _PerformanceStatsMixin.max_drawdown_single_series(series)

    @staticmethod
    def _probabilistic_ratio_from_base(base: float, series: pl.Series) -> float:
        """Compute the probabilistic ratio given an observed unannualized base ratio.

        Uses the formula: norm.cdf(base / sigma), where
        sigma = sqrt((1 + 0.5·base² - skew·base + (kurt-3)/4·base²) / (n-1)).

        Args:
            base (float): Unannualized observed ratio (e.g. Sortino).
            series (pl.Series): The original returns series (for moments and n).

        Returns:
            float: Probabilistic ratio in [0, 1].

        """
        n = series.count()
        skew_val = series.skew(bias=False)
        kurt_val = series.kurtosis(bias=False)
        if skew_val is None or kurt_val is None or n <= 1:
            return float(np.nan)
        variance = (1 + 0.5 * base**2 - float(skew_val) * base + ((float(kurt_val) - 3) / 4) * base**2) / (n - 1)
        if variance <= 0:
            return float(np.nan)
        return float(norm.cdf(base / np.sqrt(variance)))

    @columnwise_stat
    def probabilistic_sortino_ratio(self, series: pl.Series, periods: int | float | None = None) -> float:
        """Calculate the Probabilistic Sortino Ratio.

        The probability that the observed Sortino ratio is greater than zero,
        accounting for estimation uncertainty via skewness and kurtosis.

        Args:
            series (pl.Series): The series to calculate the ratio for.
            periods (int | float, optional): Accepted for API compatibility; has no effect
                since the base ratio is un-annualized.

        Returns:
            float: Probabilistic Sortino ratio in [0, 1].

        """
        downside_sum = ((series.filter(series < 0)) ** 2).sum()
        downside_deviation = float(np.sqrt(float(downside_sum) / series.count()))
        mean_val = cast(float, series.mean())
        mean_f = mean_val if mean_val is not None else 0.0
        if downside_deviation == 0.0:
            return float(np.nan)
        base = float(mean_f / downside_deviation)
        return self._probabilistic_ratio_from_base(base, series)

    @columnwise_stat
    def probabilistic_adjusted_sortino_ratio(self, series: pl.Series, periods: int | float | None = None) -> float:
        """Calculate the Probabilistic Adjusted Sortino Ratio.

        The probability that the observed adjusted Sortino ratio (divided by sqrt(2)
        for Sharpe comparability) is greater than zero, accounting for estimation
        uncertainty via skewness and kurtosis.

        Args:
            series (pl.Series): The series to calculate the ratio for.
            periods (int | float, optional): Accepted for API compatibility; has no effect
                since the base ratio is un-annualized.

        Returns:
            float: Probabilistic adjusted Sortino ratio in [0, 1].

        """
        downside_sum = ((series.filter(series < 0)) ** 2).sum()
        downside_deviation = float(np.sqrt(float(downside_sum) / series.count()))
        mean_val = cast(float, series.mean())
        mean_f = mean_val if mean_val is not None else 0.0
        if downside_deviation == 0.0:
            return float(np.nan)
        base = float(mean_f / downside_deviation) / np.sqrt(2)
        return self._probabilistic_ratio_from_base(base, series)

    def smart_sharpe(self, periods: int | float | None = None) -> dict[str, float]:
        """Calculate the Smart Sharpe ratio (Sharpe with autocorrelation penalty).

        Divides the Sharpe ratio by the autocorrelation penalty to account for
        return autocorrelation that can artificially inflate risk-adjusted metrics.

        Args:
            periods (int | float, optional): Number of periods per year. Defaults to periods_per_year.

        Returns:
            dict[str, float]: Dictionary mapping asset names to Smart Sharpe ratios.

        """
        sharpe_data = self.sharpe(periods=periods)
        penalty_data = self.autocorr_penalty()
        return {k: sharpe_data[k] / penalty_data[k] for k in sharpe_data}

    def smart_sortino(self, periods: int | float | None = None) -> dict[str, float]:
        """Calculate the Smart Sortino ratio (Sortino with autocorrelation penalty).

        Divides the Sortino ratio by the autocorrelation penalty to account for
        return autocorrelation that can artificially inflate risk-adjusted metrics.

        Args:
            periods (int | float, optional): Number of periods per year. Defaults to periods_per_year.

        Returns:
            dict[str, float]: Dictionary mapping asset names to Smart Sortino ratios.

        """
        sortino_data = self.sortino(periods=periods)
        penalty_data = self.autocorr_penalty()
        return {k: sortino_data[k] / penalty_data[k] for k in sortino_data}

    def adjusted_sortino(self, periods: int | float | None = None) -> dict[str, float]:
        """Calculate Jack Schwager's adjusted Sortino ratio.

        This adjustment allows for direct comparison to Sharpe ratio.
        See: https://archive.is/wip/2rwFW.

        Args:
            periods (int, optional): Number of periods per year. Defaults to 252.

        Returns:
            dict[str, float]: Dictionary mapping asset names to adjusted Sortino ratios.

        """
        sortino_data = self.sortino(periods=periods)
        return {k: v / np.sqrt(2) for k, v in sortino_data.items()}

    # ── Benchmark & factor ────────────────────────────────────────────────────

    @columnwise_stat
    def r_squared(self, series: pl.Series, benchmark: str | None = None) -> float:
        """Measure the straight line fit of the equity curve.

        Args:
            series (pl.Series): The series to calculate R-squared for.
            benchmark (str, optional): The benchmark column name. Defaults to None.

        Returns:
            float: The R-squared value.

        Raises:
            AttributeError: If no benchmark data is available.

        """
        if self.data.benchmark is None:
            raise AttributeError("No benchmark data available")  # noqa: TRY003

        benchmark_col = benchmark or self.data.benchmark.columns[0]

        # Evaluate both series and benchmark as Series
        all_data = cast(pl.DataFrame, self.all)
        dframe = all_data.select([series, pl.col(benchmark_col).alias("benchmark")])

        # Drop nulls
        dframe = dframe.drop_nulls()

        matrix = dframe.to_numpy()
        # Get actual Series

        strategy_np = matrix[:, 0]
        benchmark_np = matrix[:, 1]

        corr_matrix = np.corrcoef(strategy_np, benchmark_np)
        r = corr_matrix[0, 1]
        return float(r**2)

    def r2(self) -> dict[str, float]:
        """Shorthand for r_squared().

        Returns:
            dict[str, float]: Dictionary mapping asset names to R-squared values.

        """
        return self.r_squared()

    @columnwise_stat
    def information_ratio(
        self, series: pl.Series, periods_per_year: int | float | None = None, benchmark: str | None = None
    ) -> float:
        """Calculate the information ratio.

        This is essentially the risk return ratio of the net profits.

        Args:
            series (pl.Series): The series to calculate information ratio for.
            periods_per_year (int, optional): Number of periods per year. Defaults to 252.
            benchmark (str, optional): The benchmark column name. Defaults to None.

        Returns:
            float: The information ratio value.

        """
        ppy = periods_per_year or self.data._periods_per_year

        benchmark_data = cast(pl.DataFrame, self.data.benchmark)
        benchmark_col = benchmark or benchmark_data.columns[0]

        active = series - benchmark_data[benchmark_col]

        mean_val = cast(float, active.mean())
        std_val = cast(float, active.std())

        try:
            mean_f = mean_val if mean_val is not None else 0.0
            std_f = std_val if std_val is not None else 1.0
            return float((mean_f / std_f) * (ppy**0.5))
        except ZeroDivisionError:
            return 0.0

    @columnwise_stat
    def greeks(
        self, series: pl.Series, periods_per_year: int | float | None = None, benchmark: str | None = None
    ) -> dict[str, float]:
        """Calculate alpha and beta of the portfolio.

        Args:
            series (pl.Series): The series to calculate greeks for.
            periods_per_year (int, optional): Number of periods per year. Defaults to 252.
            benchmark (str, optional): The benchmark column name. Defaults to None.

        Returns:
            dict[str, float]: Dictionary containing alpha and beta values.

        """
        ppy = periods_per_year or self.data._periods_per_year

        benchmark_data = cast(pl.DataFrame, self.data.benchmark)
        benchmark_col = benchmark or benchmark_data.columns[0]

        # Evaluate both series and benchmark as Series
        all_data = cast(pl.DataFrame, self.all)
        dframe = all_data.select([series, pl.col(benchmark_col).alias("benchmark")])

        # Drop nulls
        dframe = dframe.drop_nulls()
        matrix = dframe.to_numpy()

        # Get actual Series
        strategy_np = matrix[:, 0]
        benchmark_np = matrix[:, 1]

        # 2x2 covariance matrix: [[var_strategy, cov], [cov, var_benchmark]]
        cov_matrix = np.cov(strategy_np, benchmark_np)

        cov = cov_matrix[0, 1]
        var_benchmark = cov_matrix[1, 1]

        beta = float(cov / var_benchmark) if var_benchmark != 0 else float("nan")
        alpha = float(np.mean(strategy_np) - beta * np.mean(benchmark_np))

        return {"alpha": float(alpha * ppy), "beta": beta}
