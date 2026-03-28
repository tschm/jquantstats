"""Plotting utilities for financial returns data."""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING

import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

if TYPE_CHECKING:
    from ._protocol import DataLike


def _plot_performance_dashboard(returns: pl.DataFrame, log_scale: bool = False) -> go.Figure:
    """Build a multi-panel performance dashboard figure for the given returns.

    Args:
        returns: A Polars DataFrame with a date column followed by one column per asset.
        log_scale: Whether to use a logarithmic y-axis for cumulative returns.

    Returns:
        A Plotly Figure containing cumulative returns, drawdowns, and monthly returns panels.

    """

    def hex_to_rgba(hex_color: str, alpha: float = 0.5) -> str:
        """Convert a hex colour string to an RGBA CSS string.

        Args:
            hex_color: A hex colour string (with or without a leading ``#``).
            alpha: Opacity in the range [0, 1]. Defaults to 0.5.

        Returns:
            An RGBA CSS string suitable for use in Plotly colour arguments.

        """
        hex_color = hex_color.lstrip("#")
        r, g, b = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
        return f"rgba({r}, {g}, {b}, {alpha})"

    # Get the date column name from the first column of the DataFrame
    date_col = returns.columns[0]

    # Get the tickers (all columns except the date column)
    tickers = [col for col in returns.columns if col != date_col]

    # Calculate cumulative returns (prices)
    prices = returns.with_columns([((1 + pl.col(ticker)).cum_prod()).alias(f"{ticker}_price") for ticker in tickers])

    palette = px.colors.qualitative.Plotly
    colors = {ticker: palette[i % len(palette)] for i, ticker in enumerate(tickers)}
    colors.update({f"{ticker}_light": hex_to_rgba(colors[ticker]) for ticker in tickers})

    # Resample to monthly returns
    monthly_returns = returns.group_by_dynamic(
        index_column=date_col, every="1mo", period="1mo", closed="right", label="right"
    ).agg([((pl.col(ticker) + 1.0).product() - 1.0).alias(ticker) for ticker in tickers])

    # Create subplot grid with domain for stats table
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=["Cumulative Returns", "Drawdowns", "Monthly Returns"],
        vertical_spacing=0.05,
    )

    # --- Row 1: Cumulative Returns
    for ticker in tickers:
        price_col = f"{ticker}_price"
        fig.add_trace(
            go.Scatter(
                x=prices[date_col],
                y=prices[price_col],
                mode="lines",
                name=ticker,
                legendgroup=ticker,
                line={"color": colors[ticker], "width": 2},
                hovertemplate=f"<b>%{{x|%b %Y}}</b><br>{ticker}: %{{y:.2f}}x",
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    # --- Row 2: Drawdowns
    for ticker in tickers:
        price_col = f"{ticker}_price"
        # Calculate drawdowns using polars
        price_series = prices[price_col]
        cummax = prices.select(pl.col(price_col).cum_max().alias("cummax"))
        dd_values = ((price_series - cummax["cummax"]) / cummax["cummax"]).to_list()

        fig.add_trace(
            go.Scatter(
                x=prices[date_col],
                y=dd_values,
                mode="lines",
                fill="tozeroy",
                fillcolor=colors[f"{ticker}_light"],
                line={"color": colors[ticker], "width": 1},
                name=ticker,
                legendgroup=ticker,
                hovertemplate=f"{ticker} Drawdown: %{{y:.2%}}",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    fig.add_hline(y=0, line_width=1, line_color="gray", row=2, col=1)

    # --- Row 3: Monthly Returns
    for ticker in tickers:
        # Get monthly returns values as a list for coloring
        monthly_values = monthly_returns[ticker].to_list()

        # If there's only one ticker, use green for positive returns and red for negative returns
        if len(tickers) == 1:
            bar_colors = ["green" if val > 0 else "red" for val in monthly_values]
        else:
            bar_colors = [colors[ticker] if val > 0 else colors[f"{ticker}_light"] for val in monthly_values]

        fig.add_trace(
            go.Bar(
                x=monthly_returns[date_col],
                y=monthly_returns[ticker],
                name=ticker,
                legendgroup=ticker,
                marker={
                    "color": bar_colors,
                    "line": {"width": 0},
                },
                opacity=0.8,
                hovertemplate=f"{ticker} Monthly Return: %{{y:.2%}}",
                showlegend=False,
            ),
            row=3,
            col=1,
        )

    # Layout
    fig.update_layout(
        title=f"{' vs '.join(tickers)} Performance Dashboard",
        height=1200,
        hovermode="x unified",
        plot_bgcolor="white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        xaxis={
            "rangeselector": {
                "buttons": [
                    {"count": 6, "label": "6m", "step": "month", "stepmode": "backward"},
                    {"count": 1, "label": "1y", "step": "year", "stepmode": "backward"},
                    {"count": 3, "label": "3y", "step": "year", "stepmode": "backward"},
                    {"step": "year", "stepmode": "todate", "label": "YTD"},
                    {"step": "all", "label": "All"},
                ]
            },
            "rangeslider": {"visible": False},
            "type": "date",
        },
    )

    fig.update_yaxes(title_text="Cumulative Return", row=1, col=1, tickformat=".2f")
    fig.update_yaxes(title_text="Drawdown", row=2, col=1, tickformat=".0%")
    fig.update_yaxes(title_text="Monthly Return", row=3, col=1, tickformat=".0%")

    fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor="lightgrey")
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor="lightgrey")

    if log_scale:
        fig.update_yaxes(type="log", row=1, col=1)

    return fig


@dataclasses.dataclass(frozen=True)
class DataPlots:
    """Visualization tools for financial returns data.

    This class provides methods for creating various plots and visualizations
    of financial returns data, including:

    - Returns bar charts
    - Portfolio performance snapshots
    - Monthly returns heatmaps

    The class is designed to work with the _Data class and uses Plotly
    for creating interactive visualizations.

    Attributes:
        data: The _Data object containing returns and benchmark data to visualize.

    """

    data: DataLike

    def __repr__(self) -> str:
        """Return a string representation of the DataPlots object."""
        return f"DataPlots(assets={self.data.assets})"

    def plot_snapshot(self, title: str = "Portfolio Summary", log_scale: bool = False) -> go.Figure:
        """Create a comprehensive dashboard with multiple plots for portfolio analysis.

        This function generates a three-panel plot showing:
        1. Cumulative returns over time
        2. Drawdowns over time
        3. Daily returns over time

        This provides a complete visual summary of portfolio performance.

        Args:
            title (str, optional): Title of the plot. Defaults to "Portfolio Summary".
            compounded (bool, optional): Whether to use compounded returns. Defaults to True.
            log_scale (bool, optional): Whether to use logarithmic scale for cumulative returns.
                Defaults to False.

        Returns:
            go.Figure: A Plotly figure object containing the dashboard.

        Example:
            >>> import polars as pl
            >>> from jquantstats import Data
            >>> # minimal demo dataset with a Date column and one asset
            >>> returns = pl.DataFrame({
            ...     "Date": ["2023-01-01", "2023-01-02", "2023-01-03"],
            ...     "Asset": [0.01, -0.02, 0.03],
            ... }).with_columns(pl.col("Date").str.to_date())
            >>> data = Data.from_returns(returns=returns)
            >>> fig = data.plots.plot_snapshot(title="My Portfolio Performance")
            >>> # Optional: display the interactive figure
            >>> fig.show()  # doctest: +SKIP

        """
        fig = _plot_performance_dashboard(returns=self.data.all, log_scale=log_scale)
        return fig

    def distribution(self, compounded: bool = True) -> go.Figure:
        """Plot the return distribution for each asset with a normal distribution overlay.

        Displays a histogram of returns for every asset in the dataset and
        overlays a fitted Gaussian curve. Key percentiles (5th, 25th, 75th,
        95th) are drawn as vertical dashed lines so the tails and spread are
        immediately visible.

        Args:
            compounded (bool, optional): When True the cumulative compounded
                return series ``(1 + r).cumprod() - 1`` is displayed; when
                False raw daily returns are used. Defaults to True.

        Returns:
            go.Figure: An interactive Plotly figure with one trace group per
            asset (histogram + normal curve + percentile markers).

        Example:
            >>> import polars as pl
            >>> from jquantstats import Data
            >>> returns = pl.DataFrame({
            ...     "Date": ["2023-01-01", "2023-01-02", "2023-01-03"],
            ...     "Asset": [0.01, -0.02, 0.03],
            ... }).with_columns(pl.col("Date").str.to_date())
            >>> data = Data.from_returns(returns=returns)
            >>> fig = data.plots.distribution()
            >>> fig.show()  # doctest: +SKIP

        """
        returns_df = self.data.all
        date_col = returns_df.columns[0]
        tickers = [c for c in returns_df.columns if c != date_col]

        palette = px.colors.qualitative.Plotly
        colors = {ticker: palette[i % len(palette)] for i, ticker in enumerate(tickers)}

        fig = go.Figure()

        for ticker in tickers:
            raw = returns_df[ticker].drop_nulls()
            values = ((raw + 1.0).cum_prod() - 1.0).to_list() if compounded else raw.to_list()

            if len(values) < 2:
                continue

            n = len(values)
            mean = sum(values) / n
            variance = sum((v - mean) ** 2 for v in values) / (n - 1)
            # Fallback to a tiny positive value to avoid division by zero in
            # the Gaussian formula when all returns are identical.
            std = math.sqrt(variance) if variance > 0 else 1e-10

            color = colors[ticker]

            # Histogram
            fig.add_trace(
                go.Histogram(
                    x=values,
                    name=ticker,
                    legendgroup=ticker,
                    marker_color=color,
                    opacity=0.5,
                    histnorm="probability density",
                    hovertemplate=f"{ticker}: %{{x:.3%}}<extra></extra>",
                )
            )

            # Normal distribution overlay
            x_min = min(values)
            x_max = max(values)
            x_range = [x_min + (x_max - x_min) * i / 200 for i in range(201)]
            normal_y = [math.exp(-0.5 * ((x - mean) / std) ** 2) / (std * math.sqrt(2 * math.pi)) for x in x_range]
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=normal_y,
                    mode="lines",
                    name=f"{ticker} normal fit",
                    legendgroup=ticker,
                    line={"color": color, "width": 2, "dash": "dot"},
                    hoverinfo="skip",
                    showlegend=True,
                )
            )

            # Percentile lines using linear interpolation on sorted values
            sorted_vals = sorted(values)
            percentile_specs = [(5, "dash"), (25, "dashdot"), (75, "dashdot"), (95, "dash")]
            for pct, dash in percentile_specs:
                # Linear interpolation: index = pct/100 * (n-1)
                float_idx = pct / 100 * (n - 1)
                low = int(float_idx)
                high = min(low + 1, n - 1)
                frac = float_idx - low
                pct_val = sorted_vals[low] + frac * (sorted_vals[high] - sorted_vals[low])
                fig.add_vline(
                    x=pct_val,
                    line={"color": color, "width": 1, "dash": dash},
                    annotation_text=f"p{pct}",
                    annotation_font_size=9,
                )

        fig.update_layout(
            title="Return Distribution",
            xaxis_title="Return",
            yaxis_title="Density",
            barmode="overlay",
            plot_bgcolor="white",
            hovermode="x unified",
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        )
        fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor="lightgrey", tickformat=".1%")
        fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor="lightgrey")

        return fig

    def histogram(self, resample: str = "ME", compounded: bool = True) -> go.Figure:
        """Plot a histogram of returns resampled to the requested frequency.

        Aggregates daily returns to the requested calendar period (e.g.
        monthly, quarterly, yearly) and displays a bar histogram.

        Args:
            resample (str, optional): Polars offset alias controlling the
                aggregation period.  Common values:

                - ``"ME"`` — month-end (default, equivalent to monthly)
                - ``"QE"`` — quarter-end
                - ``"YE"`` — year-end

                Any valid Polars ``group_by_dynamic`` ``every`` offset string
                is accepted (e.g. ``"1w"``, ``"3mo"``).
            compounded (bool, optional): When True each period's return is
                computed as ``prod(1 + r_d) - 1``; when False returns are
                summed. Defaults to True.

        Returns:
            go.Figure: An interactive Plotly figure showing resampled return
            histograms, one colour per asset.

        Example:
            >>> import polars as pl
            >>> from jquantstats import Data
            >>> returns = pl.DataFrame({
            ...     "Date": ["2023-01-01", "2023-01-02", "2023-01-03"],
            ...     "Asset": [0.01, -0.02, 0.03],
            ... }).with_columns(pl.col("Date").str.to_date())
            >>> data = Data.from_returns(returns=returns)
            >>> fig = data.plots.histogram(resample="ME")
            >>> fig.show()  # doctest: +SKIP

        """
        resample_aliases = {
            "ME": "1mo",
            "QE": "3mo",
            "YE": "1y",
            "MS": "1mo",
            "QS": "3mo",
            "YS": "1y",
        }

        returns_df = self.data.all
        date_col = returns_df.columns[0]
        tickers = [c for c in returns_df.columns if c != date_col]

        every = resample_aliases.get(resample, resample)

        if compounded:
            agg_exprs = [((pl.col(t) + 1.0).product() - 1.0).alias(t) for t in tickers]
        else:
            agg_exprs = [pl.col(t).sum().alias(t) for t in tickers]

        resampled = returns_df.group_by_dynamic(
            index_column=date_col,
            every=every,
            period=every,
            closed="right",
            label="right",
        ).agg(agg_exprs)

        palette = px.colors.qualitative.Plotly
        colors = {ticker: palette[i % len(palette)] for i, ticker in enumerate(tickers)}

        freq_label = resample

        fig = go.Figure()
        for ticker in tickers:
            values = resampled[ticker].drop_nulls().to_list()
            fig.add_trace(
                go.Histogram(
                    x=values,
                    name=ticker,
                    marker_color=colors[ticker],
                    opacity=0.7,
                    hovertemplate=f"{ticker}: %{{x:.2%}}<extra></extra>",
                )
            )

        fig.update_layout(
            title=f"Return Histogram ({freq_label})",
            xaxis_title="Return",
            yaxis_title="Count",
            barmode="overlay",
            plot_bgcolor="white",
            hovermode="x unified",
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        )
        fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor="lightgrey", tickformat=".1%")
        fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor="lightgrey")

        return fig
