"""Performance and risk-adjusted return metrics for financial data."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import polars as pl
from scipy.stats import norm

from ._core import _drawdown_series, _to_float, columnwise_stat, to_frame

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

        std_val = cast(float, series.std(ddof=1))
        mean_val = cast(float, series.mean())
        divisor = std_val if std_val is not None else 0.0
        mean_f = mean_val if mean_val is not None else 0.0

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
    def prob_sharpe_ratio(self, series: pl.Series, benchmark_sr: float) -> float:
        r"""Calculate the probabilistic sharpe ratio (PSR).

        Args:
            series (pl.Series): The series to calculate probabilistic Sharpe ratio for.
            benchmark_sr (float): The target Sharpe ratio to compare against. This should be unannualized.

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
        return -dd_min if dd_min is not None else 0.0

    @columnwise_stat
    def max_drawdown(self, series: pl.Series) -> float:
        """Calculate the maximum drawdown for each column.

        Args:
            series (pl.Series): The series to calculate maximum drawdown for.

        Returns:
            float: The maximum drawdown value.

        """
        return _PerformanceStatsMixin.max_drawdown_single_series(series)

    @columnwise_stat
    def ulcer_index(self, series: pl.Series) -> float:
        """Calculate the Ulcer Index (root mean square of drawdowns).

        A higher Ulcer Index indicates deeper and/or longer drawdowns.
        Returns ``nan`` when the series is empty.

        Args:
            series (pl.Series): Series of additive daily returns.

        Returns:
            float: The Ulcer Index value.

        """
        return _PerformanceStatsMixin._ulcer_index_single(series)

    @staticmethod
    def _ulcer_index_single(series: pl.Series) -> float:
        """Compute the Ulcer Index for a single returns series.

        Args:
            series: A Polars Series of additive returns.

        Returns:
            float: The Ulcer Index, or ``nan`` when the series is empty.
        """
        dd = _drawdown_series(series)
        if dd.len() == 0:
            return float("nan")
        return float(np.sqrt(_to_float((dd**2).mean())))

    @columnwise_stat
    def ulcer_performance_index(self, series: pl.Series, rf: float = 0.0) -> float:
        """Calculate the Ulcer Performance Index (UPI).

        A Sharpe-like ratio that penalises drawdown depth and duration.
        Returns ``nan`` when the Ulcer Index is zero or the series is empty.

        Args:
            series (pl.Series): Series of additive daily returns.
            rf (float): Risk-free rate (total, not annualised). Defaults to 0.

        Returns:
            float: The Ulcer Performance Index value.

        """
        ui = _PerformanceStatsMixin._ulcer_index_single(series)
        if ui == 0.0 or np.isnan(ui):
            return float("nan")
        total_return = _to_float(np.expm1(np.log1p(series.cast(pl.Float64)).sum()))
        return (total_return - rf) / ui

    def upi(self, rf: float = 0.0) -> dict[str, float]:
        """Shorthand for :meth:`ulcer_performance_index`.

        Args:
            rf (float): Risk-free rate (total, not annualised). Defaults to 0.

        Returns:
            dict[str, float]: Dictionary mapping asset names to UPI values.

        """
        return self.ulcer_performance_index(rf=rf)

    @columnwise_stat
    def serenity_index(self, series: pl.Series, rf: float = 0.0) -> float:
        """Calculate the Serenity Index.

        A Calmar-like ratio that uses the Ulcer Index instead of max drawdown,
        additionally weighted by a loss ratio and a win/loss ratio.
        Returns ``nan`` when the Ulcer Index is zero or the series is empty.

        Args:
            series (pl.Series): Series of additive daily returns.
            rf (float): Risk-free rate (total, not annualised). Defaults to 0.

        Returns:
            float: The Serenity Index value.

        """
        ui = _PerformanceStatsMixin._ulcer_index_single(series)
        if ui == 0.0 or np.isnan(ui):
            return float("nan")
        total_return = _to_float(np.expm1(np.log1p(series.cast(pl.Float64)).sum()))
        dd = _drawdown_series(series)
        loss_ratio = 1.0 - _to_float(dd.mean())
        positive = series.filter(series >= 0)
        negative = series.filter(series < 0)
        pos_sum = _to_float(positive.sum())
        neg_sum = abs(_to_float(negative.sum()))
        if neg_sum == 0.0:
            return float("nan")
        win_ratio = pos_sum / neg_sum
        return (total_return - rf) / ui * loss_ratio * win_ratio

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
