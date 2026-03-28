"""Plotting utilities for financial returns data."""

from __future__ import annotations

import dataclasses
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

    def rolling_beta(
        self,
        benchmark: str,
        window1: int = 126,
        window2: int = 252,
    ) -> go.Figure:
        """Plot rolling beta of each asset vs a benchmark at two window lengths.

        Computes ``beta = cov(asset, benchmark) / var(benchmark)`` over rolling
        windows of ``window1`` and ``window2`` periods and renders one line per
        (asset, window) pair.

        Args:
            benchmark: Name of the benchmark column within ``data.all``.
            window1: Shorter rolling-window size in periods. Defaults to 126.
            window2: Longer rolling-window size in periods. Defaults to 252.

        Returns:
            A Plotly Figure with one trace per (asset, window) combination.

        Raises:
            ValueError: If either window is not a positive integer.
        """
        if not isinstance(window1, int) or window1 <= 0:
            raise ValueError("window1 must be a positive integer")  # noqa: TRY003
        if not isinstance(window2, int) or window2 <= 0:
            raise ValueError("window2 must be a positive integer")  # noqa: TRY003

        all_df = self.data.all
        # Identify date column (first column that is not an asset and not the benchmark)
        date_col = [c for c in all_df.columns if c not in self.data.assets and c != benchmark]
        # Exclude the benchmark itself from the per-asset beta traces
        asset_cols = [a for a in self.data.assets if a != benchmark]

        def _rolling_beta_df(window: int) -> pl.DataFrame:
            """Compute rolling beta for all assets vs the benchmark over *window* periods."""
            return all_df.select(
                [pl.col(c) for c in date_col]
                + [
                    (
                        pl.rolling_cov(asset, benchmark, window_size=window)
                        / pl.col(benchmark).rolling_var(window_size=window)
                    ).alias(asset)
                    for asset in asset_cols
                ]
            )

        rolling1 = _rolling_beta_df(window1)
        rolling2 = _rolling_beta_df(window2)

        fig = go.Figure()
        x1 = rolling1[date_col[0]] if date_col else None
        x2 = rolling2[date_col[0]] if date_col else None

        for col in asset_cols:
            fig.add_trace(
                go.Scatter(
                    x=x1,
                    y=rolling1[col],
                    mode="lines",
                    name=f"{col} ({window1}d)",
                    line={"width": 1},
                )
            )

        for col in asset_cols:
            fig.add_trace(
                go.Scatter(
                    x=x2,
                    y=rolling2[col],
                    mode="lines",
                    name=f"{col} ({window2}d)",
                    line={"width": 1, "dash": "dash"},
                )
            )

        fig.add_hline(y=1, line_width=1, line_dash="dot", line_color="gray")

        fig.update_layout(
            title=f"Rolling Beta vs {benchmark} ({window1}/{window2}-period windows)",
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
        fig.update_yaxes(title_text="Beta")
        fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor="lightgrey")
        fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor="lightgrey")
        return fig
