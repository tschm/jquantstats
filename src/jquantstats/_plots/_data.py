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

# ── Module-level styling helpers ──────────────────────────────────────────────


def _hex_to_rgba(hex_color: str, alpha: float = 0.5) -> str:
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


def _ticker_colors(tickers: list[str]) -> dict[str, str]:
    """Map ticker names to Plotly qualitative palette colours.

    Args:
        tickers: Ordered list of ticker / column names.

    Returns:
        dict mapping each ticker to a hex colour string.

    """
    palette = px.colors.qualitative.Plotly
    return {ticker: palette[i % len(palette)] for i, ticker in enumerate(tickers)}


def _date_range_selector() -> dict:
    """Return a standard Plotly date range-selector configuration.

    Returns:
        A dict suitable for ``xaxis.rangeselector``.

    """
    return {
        "buttons": [
            {"count": 6, "label": "6m", "step": "month", "stepmode": "backward"},
            {"count": 1, "label": "1y", "step": "year", "stepmode": "backward"},
            {"count": 3, "label": "3y", "step": "year", "stepmode": "backward"},
            {"step": "year", "stepmode": "todate", "label": "YTD"},
            {"step": "all", "label": "All"},
        ]
    }


def _apply_base_layout(
    fig: go.Figure,
    title: str,
    height: int = 600,
    with_range_selector: bool = True,
) -> go.Figure:
    """Apply the standard jquantstats Plotly layout to a figure.

    Sets white background, light-grey grid, horizontal legend, and an
    optional date range-selector on the primary x-axis.

    Args:
        fig: The Plotly figure to style in-place.
        title: Chart title.
        height: Figure height in pixels. Defaults to 600.
        with_range_selector: Attach a date range-selector to ``xaxis``.
            Defaults to True.

    Returns:
        The same figure, mutated in-place and returned for chaining.

    """
    layout_kw: dict = {
        "title": title,
        "height": height,
        "hovermode": "x unified",
        "plot_bgcolor": "white",
        "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    }
    if with_range_selector:
        layout_kw["xaxis"] = {
            "rangeselector": _date_range_selector(),
            "rangeslider": {"visible": False},
            "type": "date",
        }
    fig.update_layout(**layout_kw)
    fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor="lightgrey")
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor="lightgrey")
    return fig


def _compute_drawdown_periods(prices: list[float], n: int) -> list[dict]:
    """Identify the top *n* drawdown periods from a cumulative price series.

    Args:
        prices: Cumulative price (NAV) values as a plain Python list.
        n: Maximum number of drawdown periods to return.

    Returns:
        List of dicts with keys ``start_idx``, ``end_idx``, ``valley_idx``,
        ``max_drawdown`` (fraction ≤ 0), sorted by severity (worst first).

    """
    length = len(prices)
    hwm: list[float] = [0.0] * length
    hwm[0] = prices[0]
    for i in range(1, length):
        hwm[i] = max(hwm[i - 1], prices[i])

    in_dd = [prices[i] < hwm[i] for i in range(length)]
    periods: list[dict] = []
    i = 0
    while i < length:
        if not in_dd[i]:
            i += 1
            continue
        start = i
        while i < length and in_dd[i]:
            i += 1
        end = i - 1
        valley = start + min(range(end - start + 1), key=lambda k: prices[start + k])
        max_dd = (prices[valley] - hwm[valley]) / hwm[valley]
        periods.append({"start_idx": start, "end_idx": end, "valley_idx": valley, "max_drawdown": max_dd})

    periods.sort(key=lambda p: p["max_drawdown"])
    return periods[:n]


# ── Dashboard (existing) ──────────────────────────────────────────────────────


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


# ── DataPlots ──────────────────────────────────────────────────────────────────


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

    def plot_returns(self, title: str = "Cumulative Returns", log_scale: bool = False) -> go.Figure:
        """Cumulative compounded returns over time.

        Plots ``(1 + r).cumprod()`` for every column in the dataset (including
        benchmark when present).

        Args:
            title: Chart title. Defaults to ``"Cumulative Returns"``.
            log_scale: Use a logarithmic y-axis. Defaults to False.

        Returns:
            go.Figure: Interactive Plotly line chart.

        """
        df = self.data.all
        date_col = df.columns[0]
        tickers = [c for c in df.columns if c != date_col]
        colors = _ticker_colors(tickers)

        prices = df.with_columns([(1.0 + pl.col(t)).cum_prod().alias(t) for t in tickers])

        fig = go.Figure()
        for ticker in tickers:
            fig.add_trace(
                go.Scatter(
                    x=prices[date_col],
                    y=prices[ticker],
                    mode="lines",
                    name=ticker,
                    line={"color": colors[ticker], "width": 2},
                    hovertemplate=f"<b>%{{x|%b %Y}}</b><br>{ticker}: %{{y:.2f}}x",
                )
            )

        _apply_base_layout(fig, title)
        fig.update_yaxes(title_text="Cumulative Return", tickformat=".2f")
        if log_scale:
            fig.update_yaxes(type="log")
        return fig

    def plot_log_returns(self, title: str = "Log Returns") -> go.Figure:
        """Cumulative log returns over time.

        Plots ``log((1 + r).cumprod())`` — the natural log of the compounded
        growth factor — which linearises exponential growth and makes
        multi-asset comparisons on a common scale.

        Args:
            title: Chart title. Defaults to ``"Log Returns"``.

        Returns:
            go.Figure: Interactive Plotly line chart.

        """
        import math

        df = self.data.all
        date_col = df.columns[0]
        tickers = [c for c in df.columns if c != date_col]
        colors = _ticker_colors(tickers)

        log_prices = df.with_columns([(1.0 + pl.col(t)).cum_prod().log(math.e).alias(t) for t in tickers])

        fig = go.Figure()
        for ticker in tickers:
            fig.add_trace(
                go.Scatter(
                    x=log_prices[date_col],
                    y=log_prices[ticker],
                    mode="lines",
                    name=ticker,
                    line={"color": colors[ticker], "width": 2},
                    hovertemplate=f"<b>%{{x|%b %Y}}</b><br>{ticker}: %{{y:.4f}}",
                )
            )

        _apply_base_layout(fig, title)
        fig.update_yaxes(title_text="Log Return")
        return fig

    def plot_daily_returns(self, title: str = "Daily Returns") -> go.Figure:
        """Daily returns as a bar chart.

        Each bar is coloured green for positive returns and red for negative
        returns.  When multiple assets are present each asset gets its own
        trace in the palette colour with opacity used for positive/negative
        differentiation.

        Args:
            title: Chart title. Defaults to ``"Daily Returns"``.

        Returns:
            go.Figure: Interactive Plotly bar chart.

        """
        df = self.data.all
        date_col = df.columns[0]
        tickers = [c for c in df.columns if c != date_col]
        colors = _ticker_colors(tickers)
        single = len(tickers) == 1

        fig = go.Figure()
        for ticker in tickers:
            values = df[ticker].to_list()
            if single:
                bar_colors = ["#2ca02c" if v is not None and v > 0 else "#d62728" for v in values]
            else:
                pos_color = colors[ticker]
                neg_color = _hex_to_rgba(pos_color, alpha=0.4)
                bar_colors = [pos_color if v is not None and v > 0 else neg_color for v in values]

            fig.add_trace(
                go.Bar(
                    x=df[date_col],
                    y=df[ticker],
                    name=ticker,
                    marker={"color": bar_colors, "line": {"width": 0}},
                    opacity=0.85,
                    hovertemplate=f"{ticker}: %{{y:.2%}}",
                )
            )

        _apply_base_layout(fig, title)
        fig.update_yaxes(title_text="Return", tickformat=".1%")
        return fig

    def plot_yearly_returns(self, title: str = "Yearly Returns", compounded: bool = True) -> go.Figure:
        """Annual compounded (or summed) returns as a grouped bar chart.

        Args:
            title: Chart title. Defaults to ``"Yearly Returns"``.
            compounded: Compound returns within each year. Defaults to True.

        Returns:
            go.Figure: Interactive Plotly grouped bar chart.

        """
        df = self.data.all
        date_col = df.columns[0]
        tickers = [c for c in df.columns if c != date_col]
        colors = _ticker_colors(tickers)

        agg_exprs = (
            [((1.0 + pl.col(t)).product() - 1.0).alias(t) for t in tickers]
            if compounded
            else [pl.col(t).sum().alias(t) for t in tickers]
        )
        yearly = (
            df.with_columns(pl.col(date_col).dt.year().alias("_year")).group_by("_year").agg(agg_exprs).sort("_year")
        )

        fig = go.Figure()
        for ticker in tickers:
            values = yearly[ticker].to_list()
            bar_colors = [
                colors[ticker] if v is not None and v >= 0 else _hex_to_rgba(colors[ticker], 0.5) for v in values
            ]
            fig.add_trace(
                go.Bar(
                    x=yearly["_year"],
                    y=yearly[ticker],
                    name=ticker,
                    marker={"color": bar_colors, "line": {"width": 0}},
                    opacity=0.85,
                    hovertemplate=f"{ticker}: %{{y:.2%}}",
                )
            )

        _apply_base_layout(fig, title, with_range_selector=False)
        fig.update_layout(barmode="group", xaxis_title="Year")
        fig.update_yaxes(title_text="Annual Return", tickformat=".1%")
        return fig

    def plot_monthly_returns(self, title: str = "Monthly Returns", compounded: bool = True) -> go.Figure:
        """Monthly compounded (or summed) returns as a bar chart.

        Args:
            title: Chart title. Defaults to ``"Monthly Returns"``.
            compounded: Compound returns within each month. Defaults to True.

        Returns:
            go.Figure: Interactive Plotly bar chart.

        """
        df = self.data.all
        date_col = df.columns[0]
        tickers = [c for c in df.columns if c != date_col]
        colors = _ticker_colors(tickers)
        single = len(tickers) == 1

        monthly = df.group_by_dynamic(
            index_column=date_col, every="1mo", period="1mo", closed="right", label="right"
        ).agg(
            [((1.0 + pl.col(t)).product() - 1.0).alias(t) if compounded else pl.col(t).sum().alias(t) for t in tickers]
        )

        fig = go.Figure()
        for ticker in tickers:
            values = monthly[ticker].to_list()
            if single:
                bar_colors = ["#2ca02c" if v is not None and v > 0 else "#d62728" for v in values]
            else:
                pos_color = colors[ticker]
                neg_color = _hex_to_rgba(pos_color, alpha=0.4)
                bar_colors = [pos_color if v is not None and v > 0 else neg_color for v in values]

            fig.add_trace(
                go.Bar(
                    x=monthly[date_col],
                    y=monthly[ticker],
                    name=ticker,
                    marker={"color": bar_colors, "line": {"width": 0}},
                    opacity=0.85,
                    hovertemplate=f"{ticker}: %{{y:.2%}}",
                )
            )

        _apply_base_layout(fig, title)
        fig.update_yaxes(title_text="Monthly Return", tickformat=".1%")
        return fig

    def plot_monthly_heatmap(
        self,
        title: str = "Monthly Returns Heatmap",
        compounded: bool = True,
        asset: str | None = None,
    ) -> go.Figure:
        """Monthly returns calendar heatmap (year x month).

        One heatmap is produced per call for a single asset.  Green cells
        indicate positive months; red cells indicate negative months.

        Args:
            title: Chart title. Defaults to ``"Monthly Returns Heatmap"``.
            compounded: Compound intra-month returns. Defaults to True.
            asset: Asset column name to display.  Defaults to the first
                non-date column in the dataset.

        Returns:
            go.Figure: Interactive Plotly heatmap.

        """
        df = self.data.all
        date_col = df.columns[0]
        tickers = [c for c in df.columns if c != date_col]
        col = asset if asset in tickers else tickers[0]

        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        agg_expr = ((1.0 + pl.col(col)).product() - 1.0).alias("ret") if compounded else pl.col(col).sum().alias("ret")
        monthly = (
            df.with_columns(
                [
                    pl.col(date_col).dt.year().alias("_year"),
                    pl.col(date_col).dt.month().alias("_month"),
                ]
            )
            .group_by(["_year", "_month"])
            .agg(agg_expr.alias("ret"))
            .sort(["_year", "_month"])
        )

        years = sorted(monthly["_year"].unique().to_list())
        z = [[None] * 12 for _ in years]
        text = [[""] * 12 for _ in years]
        year_idx = {y: i for i, y in enumerate(years)}

        for row in monthly.iter_rows(named=True):
            yi = year_idx[row["_year"]]
            mi = row["_month"] - 1
            val = row["ret"]
            z[yi][mi] = val * 100 if val is not None else None
            text[yi][mi] = f"{val:.1%}" if val is not None else ""

        fig = go.Figure(
            go.Heatmap(
                x=month_names,
                y=[str(y) for y in years],
                z=z,
                text=text,
                texttemplate="%{text}",
                colorscale=[[0, "#d62728"], [0.5, "#ffffff"], [1, "#2ca02c"]],
                zmid=0,
                showscale=True,
                colorbar={"title": "Return (%)"},
                hovertemplate="<b>%{y} %{x}</b><br>Return: %{text}<extra></extra>",
            )
        )

        fig.update_layout(
            title=f"{title} — {col}",
            height=max(300, 40 * len(years) + 100),
            plot_bgcolor="white",
            xaxis={"side": "top"},
        )
        return fig

    def plot_histogram(self, title: str = "Returns Distribution", bins: int = 50) -> go.Figure:
        """Return histogram with a kernel density overlay.

        Each asset is shown as a semi-transparent histogram overlaid on the
        same axes so distributions can be compared visually.

        Args:
            title: Chart title. Defaults to ``"Returns Distribution"``.
            bins: Number of histogram bins. Defaults to 50.

        Returns:
            go.Figure: Interactive Plotly histogram figure.

        """
        df = self.data.all
        date_col = df.columns[0]
        tickers = [c for c in df.columns if c != date_col]
        colors = _ticker_colors(tickers)

        fig = go.Figure()
        for ticker in tickers:
            values = df[ticker].drop_nulls().to_list()
            fig.add_trace(
                go.Histogram(
                    x=values,
                    name=ticker,
                    nbinsx=bins,
                    marker_color=colors[ticker],
                    opacity=0.6,
                    hovertemplate=f"{ticker}: %{{x:.2%}}<extra></extra>",
                )
            )

        _apply_base_layout(fig, title, with_range_selector=False)
        fig.update_layout(barmode="overlay")
        fig.update_xaxes(title_text="Return", tickformat=".1%")
        fig.update_yaxes(title_text="Count")
        return fig

    def plot_distribution(
        self,
        title: str = "Return Distribution by Period",
        compounded: bool = True,
    ) -> go.Figure:
        """Return distributions across daily, weekly, monthly, quarterly and yearly periods.

        Renders a box plot for each aggregation period so the user can compare
        how the distribution widens as the holding period lengthens.  One
        subplot column is produced per asset.

        Args:
            title: Chart title. Defaults to ``"Return Distribution by Period"``.
            compounded: Compound returns within each period. Defaults to True.

        Returns:
            go.Figure: Interactive Plotly figure with one subplot per asset.

        """
        df = self.data.all
        date_col = df.columns[0]
        tickers = [c for c in df.columns if c != date_col]
        colors = _ticker_colors(tickers)

        periods = [
            ("Daily", None),
            ("Weekly", "1w"),
            ("Monthly", "1mo"),
            ("Quarterly", "3mo"),
            ("Yearly", "1y"),
        ]

        n_assets = len(tickers)
        fig = make_subplots(
            rows=1,
            cols=n_assets,
            subplot_titles=tickers,
            shared_yaxes=True,
        )

        for col_idx, ticker in enumerate(tickers, start=1):
            for period_name, trunc in periods:
                if trunc is None:
                    values = df[ticker].drop_nulls().to_list()
                else:
                    agg_expr = (
                        ((1.0 + pl.col(ticker)).product() - 1.0).alias("ret")
                        if compounded
                        else pl.col(ticker).sum().alias("ret")
                    )
                    agg_df = (
                        df.with_columns(pl.col(date_col).dt.truncate(trunc).alias("_period"))
                        .group_by("_period")
                        .agg(agg_expr)
                    )
                    values = agg_df["ret"].drop_nulls().to_list()

                fig.add_trace(
                    go.Box(
                        y=values,
                        name=period_name,
                        marker_color=colors[ticker],
                        showlegend=(col_idx == 1),
                        legendgroup=period_name,
                        boxpoints="outliers",
                        hovertemplate=f"{period_name}: %{{y:.2%}}<extra></extra>",
                    ),
                    row=1,
                    col=col_idx,
                )

        fig.update_layout(
            title=title,
            height=500,
            plot_bgcolor="white",
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.05, "xanchor": "right", "x": 1},
        )
        fig.update_yaxes(tickformat=".1%", showgrid=True, gridwidth=0.5, gridcolor="lightgrey")
        fig.update_xaxes(showgrid=False)
        return fig

    def plot_drawdown(self, title: str = "Drawdowns") -> go.Figure:
        """Underwater equity curve (drawdown) chart.

        Shows the percentage decline from the running peak for every column
        in the dataset (assets and benchmark where present).

        Args:
            title: Chart title. Defaults to ``"Drawdowns"``.

        Returns:
            go.Figure: Interactive Plotly filled-area chart.

        """
        df = self.data.all
        date_col = df.columns[0]
        tickers = [c for c in df.columns if c != date_col]
        colors = _ticker_colors(tickers)

        prices = df.with_columns([(1.0 + pl.col(t)).cum_prod().alias(t) for t in tickers])

        fig = go.Figure()
        for ticker in tickers:
            price_s = prices[ticker]
            hwm = price_s.cum_max()
            dd = ((price_s - hwm) / hwm).to_list()

            fig.add_trace(
                go.Scatter(
                    x=prices[date_col],
                    y=dd,
                    mode="lines",
                    fill="tozeroy",
                    fillcolor=_hex_to_rgba(colors[ticker], 0.3),
                    line={"color": colors[ticker], "width": 1.5},
                    name=ticker,
                    hovertemplate=f"{ticker}: %{{y:.2%}}",
                )
            )

        fig.add_hline(y=0, line_width=1, line_color="gray")
        _apply_base_layout(fig, title)
        fig.update_yaxes(title_text="Drawdown", tickformat=".0%")
        return fig

    def plot_drawdowns_periods(
        self,
        n: int = 5,
        title: str = "Top Drawdown Periods",
        asset: str | None = None,
    ) -> go.Figure:
        """Cumulative returns chart with the worst *n* drawdown periods shaded.

        Identifies the *n* deepest drawdown periods and overlays coloured
        rectangular shading on the cumulative returns line.  One asset is
        shown per call.

        Args:
            n: Number of worst drawdown periods to highlight. Defaults to 5.
            title: Chart title. Defaults to ``"Top Drawdown Periods"``.
            asset: Asset column name.  Defaults to the first non-date column.

        Returns:
            go.Figure: Interactive Plotly figure.

        """
        df = self.data.all
        date_col = df.columns[0]
        tickers = [c for c in df.columns if c != date_col]
        col = asset if asset in tickers else tickers[0]

        price_series = (1.0 + df[col].cast(pl.Float64)).cum_prod()
        price_list = price_series.to_list()
        dates = df[date_col].to_list()

        drawdown_periods = _compute_drawdown_periods(price_list, n)

        dd_colors = px.colors.qualitative.Plotly

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=price_list,
                mode="lines",
                name=col,
                line={"color": "#1f77b4", "width": 2},
                hovertemplate=f"<b>%{{x|%b %Y}}</b><br>{col}: %{{y:.2f}}x",
            )
        )

        for i, period in enumerate(drawdown_periods):
            start_date = dates[period["start_idx"]]
            end_date = dates[min(period["end_idx"] + 1, len(dates) - 1)]
            max_dd = period["max_drawdown"]
            shade_color = _hex_to_rgba(dd_colors[i % len(dd_colors)], alpha=0.2)

            fig.add_vrect(
                x0=start_date,
                x1=end_date,
                fillcolor=shade_color,
                line_width=0,
                annotation_text=f"#{i + 1} {max_dd:.1%}",
                annotation_position="top left",
                annotation_font_size=10,
            )

        _apply_base_layout(fig, f"{title} — {col}")
        fig.update_yaxes(title_text="Cumulative Return", tickformat=".2f")
        return fig

    def plot_earnings(
        self,
        start_balance: float = 1e5,
        title: str = "Portfolio Earnings",
        compounded: bool = True,
    ) -> go.Figure:
        """Dollar equity curve showing portfolio value over time.

        Scales cumulative returns by *start_balance* so the y-axis reflects
        an absolute portfolio value rather than a dimensionless growth factor.

        Args:
            start_balance: Starting portfolio value in currency units.
                Defaults to 100 000.
            title: Chart title. Defaults to ``"Portfolio Earnings"``.
            compounded: Use compounded returns (``cumprod``). When False uses
                cumulative sum. Defaults to True.

        Returns:
            go.Figure: Interactive Plotly line chart.

        """
        df = self.data.all
        date_col = df.columns[0]
        tickers = [c for c in df.columns if c != date_col]
        colors = _ticker_colors(tickers)

        if compounded:
            equity = df.with_columns([(start_balance * (1.0 + pl.col(t)).cum_prod()).alias(t) for t in tickers])
        else:
            equity = df.with_columns([(start_balance * (1.0 + pl.col(t).cum_sum())).alias(t) for t in tickers])

        fig = go.Figure()
        for ticker in tickers:
            fig.add_trace(
                go.Scatter(
                    x=equity[date_col],
                    y=equity[ticker],
                    mode="lines",
                    name=ticker,
                    line={"color": colors[ticker], "width": 2},
                    hovertemplate=f"<b>%{{x|%b %Y}}</b><br>{ticker}: $%{{y:,.0f}}",
                )
            )

        _apply_base_layout(fig, title)
        fig.update_yaxes(
            title_text=f"Portfolio Value (starting ${start_balance:,.0f})",
            tickprefix="$",
            tickformat=",.0f",
        )
        return fig

    def plot_rolling_sharpe(
        self,
        rolling_period: int = 126,
        periods_per_year: int = 252,
        title: str = "Rolling Sharpe Ratio",
    ) -> go.Figure:
        """Rolling annualised Sharpe ratio over time.

        Computes ``rolling_mean / rolling_std * sqrt(periods_per_year)`` with a
        trailing window of *rolling_period* observations for every column in the
        dataset (assets and benchmark when present).

        Args:
            rolling_period: Trailing window size. Defaults to 126 (6 months).
            periods_per_year: Annualisation factor. Defaults to 252.
            title: Chart title. Defaults to ``"Rolling Sharpe Ratio"``.

        Returns:
            go.Figure: Interactive Plotly line chart.

        """
        import math

        df = self.data.all
        date_col = df.columns[0]
        tickers = [c for c in df.columns if c != date_col]
        colors = _ticker_colors(tickers)
        scale = math.sqrt(periods_per_year)

        rolling = df.with_columns(
            [
                (
                    pl.col(t).rolling_mean(window_size=rolling_period)
                    / pl.col(t).rolling_std(window_size=rolling_period)
                    * scale
                ).alias(t)
                for t in tickers
            ]
        )

        fig = go.Figure()
        for ticker in tickers:
            fig.add_trace(
                go.Scatter(
                    x=rolling[date_col],
                    y=rolling[ticker],
                    mode="lines",
                    name=ticker,
                    line={"color": colors[ticker], "width": 1.5},
                    hovertemplate=f"{ticker}: %{{y:.2f}}",
                )
            )

        fig.add_hline(y=0, line_width=1, line_color="gray", line_dash="dash")
        _apply_base_layout(fig, title)
        fig.update_yaxes(title_text=f"Sharpe ({rolling_period}-period rolling)")
        return fig

    def plot_rolling_sortino(
        self,
        rolling_period: int = 126,
        periods_per_year: int = 252,
        title: str = "Rolling Sortino Ratio",
    ) -> go.Figure:
        """Rolling annualised Sortino ratio over time.

        Computes ``rolling_mean / rolling_downside_std * sqrt(periods_per_year)``
        where downside deviation considers only negative returns.

        Args:
            rolling_period: Trailing window size. Defaults to 126 (6 months).
            periods_per_year: Annualisation factor. Defaults to 252.
            title: Chart title. Defaults to ``"Rolling Sortino Ratio"``.

        Returns:
            go.Figure: Interactive Plotly line chart.

        """
        import math

        df = self.data.all
        date_col = df.columns[0]
        tickers = [c for c in df.columns if c != date_col]
        colors = _ticker_colors(tickers)
        scale = math.sqrt(periods_per_year)

        exprs = []
        for t in tickers:
            mean_r = pl.col(t).rolling_mean(window_size=rolling_period)
            downside = (
                pl.when(pl.col(t) < 0)
                .then(pl.col(t) ** 2)
                .otherwise(0.0)
                .rolling_mean(window_size=rolling_period)
                .sqrt()
            )
            exprs.append((mean_r / downside * scale).alias(t))

        rolling = df.with_columns(exprs)

        fig = go.Figure()
        for ticker in tickers:
            fig.add_trace(
                go.Scatter(
                    x=rolling[date_col],
                    y=rolling[ticker],
                    mode="lines",
                    name=ticker,
                    line={"color": colors[ticker], "width": 1.5},
                    hovertemplate=f"{ticker}: %{{y:.2f}}",
                )
            )

        fig.add_hline(y=0, line_width=1, line_color="gray", line_dash="dash")
        _apply_base_layout(fig, title)
        fig.update_yaxes(title_text=f"Sortino ({rolling_period}-period rolling)")
        return fig

    def plot_rolling_volatility(
        self,
        rolling_period: int = 126,
        periods_per_year: int = 252,
        title: str = "Rolling Volatility",
    ) -> go.Figure:
        """Rolling annualised volatility over time.

        Computes ``rolling_std * sqrt(periods_per_year)`` for every column in
        the dataset.

        Args:
            rolling_period: Trailing window size. Defaults to 126 (6 months).
            periods_per_year: Annualisation factor. Defaults to 252.
            title: Chart title. Defaults to ``"Rolling Volatility"``.

        Returns:
            go.Figure: Interactive Plotly line chart.

        """
        import math

        df = self.data.all
        date_col = df.columns[0]
        tickers = [c for c in df.columns if c != date_col]
        colors = _ticker_colors(tickers)
        scale = math.sqrt(periods_per_year)

        rolling = df.with_columns(
            [(pl.col(t).rolling_std(window_size=rolling_period) * scale).alias(t) for t in tickers]
        )

        fig = go.Figure()
        for ticker in tickers:
            fig.add_trace(
                go.Scatter(
                    x=rolling[date_col],
                    y=rolling[ticker],
                    mode="lines",
                    name=ticker,
                    line={"color": colors[ticker], "width": 1.5},
                    hovertemplate=f"{ticker}: %{{y:.2%}}",
                )
            )

        _apply_base_layout(fig, title)
        fig.update_yaxes(title_text=f"Volatility ({rolling_period}-period rolling)", tickformat=".0%")
        return fig

    def plot_rolling_beta(
        self,
        rolling_period: int = 126,
        rolling_period2: int | None = 252,
        title: str = "Rolling Beta",
    ) -> go.Figure:
        """Rolling beta versus the benchmark.

        Plots one line per asset per window size.  Beta is estimated via the
        standard OLS formula: ``cov(asset, bench) / var(bench)`` computed over
        a trailing window.

        Args:
            rolling_period: Primary trailing window size. Defaults to 126.
            rolling_period2: Optional second window size overlaid on the same
                chart. Defaults to 252. Pass ``None`` to omit.
            title: Chart title. Defaults to ``"Rolling Beta"``.

        Returns:
            go.Figure: Interactive Plotly line chart.

        Raises:
            AttributeError: If no benchmark columns are present in the data.

        """
        df = self.data.all
        date_col = df.columns[0]

        benchmark_df = getattr(self.data, "benchmark", None)
        if benchmark_df is None:
            raise AttributeError("No benchmark data available")  # noqa: TRY003

        bench_col = benchmark_df.columns[0]
        returns_df = getattr(self.data, "returns", None)
        assets = (
            list(returns_df.columns)
            if returns_df is not None
            else [c for c in df.columns if c != date_col and c != bench_col]
        )
        colors = _ticker_colors(assets)
        windows = [w for w in (rolling_period, rolling_period2) if w is not None]
        line_styles = ["solid", "dash"]

        fig = go.Figure()
        for asset in assets:
            for w, dash in zip(windows, line_styles, strict=False):
                mean_x = pl.col(asset).rolling_mean(window_size=w)
                mean_y = pl.col(bench_col).rolling_mean(window_size=w)
                mean_xy = (pl.col(asset) * pl.col(bench_col)).rolling_mean(window_size=w)
                mean_y2 = (pl.col(bench_col) ** 2).rolling_mean(window_size=w)
                beta_expr = ((mean_xy - mean_x * mean_y) / (mean_y2 - mean_y**2)).alias("beta")

                beta_df = df.with_columns(beta_expr)
                label = f"{asset} ({w}d)"
                fig.add_trace(
                    go.Scatter(
                        x=beta_df[date_col],
                        y=beta_df["beta"],
                        mode="lines",
                        name=label,
                        line={"color": colors[asset], "width": 1.5, "dash": dash},
                        hovertemplate=f"{label}: %{{y:.2f}}",
                    )
                )

        fig.add_hline(y=1, line_width=1, line_color="gray", line_dash="dash")
        _apply_base_layout(fig, title)
        fig.update_yaxes(title_text="Beta")
        return fig
