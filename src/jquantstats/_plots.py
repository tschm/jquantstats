import calendar
import dataclasses

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots


def _plot_performance_dashboard(returns: pd.DataFrame, log_scale=False) -> go.Figure:
    def hex_to_rgba(hex_color: str, alpha: float = 0.5) -> str:
        hex_color = hex_color.lstrip("#")
        r, g, b = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
        return f"rgba({r}, {g}, {b}, {alpha})"

    tickers = returns.columns.tolist()
    prices = (1 + returns).cumprod()

    palette = px.colors.qualitative.Plotly
    COLORS = {ticker: palette[i % len(palette)] for i, ticker in enumerate(tickers)}
    COLORS.update({f"{ticker}_light": hex_to_rgba(COLORS[ticker]) for ticker in tickers})

    monthly_returns = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    monthly_returns.index = monthly_returns.index.to_period("M").to_timestamp()

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
        fig.add_trace(
            go.Scatter(
                x=prices.index,
                y=prices[ticker],
                mode="lines",
                name=ticker,
                legendgroup=ticker,
                line=dict(color=COLORS[ticker], width=2),
                hovertemplate=f"<b>%{{x|%b %Y}}</b><br>{ticker}: %{{y:.2f}}x",
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    # --- Row 2: Drawdowns
    for ticker in tickers:
        dd = (prices[ticker] - prices[ticker].cummax()) / prices[ticker].cummax()
        fig.add_trace(
            go.Scatter(
                x=dd.index,
                y=dd,
                mode="lines",
                fill="tozeroy",
                fillcolor=COLORS[f"{ticker}_light"],
                line=dict(color=COLORS[ticker], width=1),
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
        fig.add_trace(
            go.Bar(
                x=monthly_returns.index,
                y=monthly_returns[ticker],
                name=ticker,
                legendgroup=ticker,
                marker=dict(
                    color=[COLORS[ticker] if val > 0 else COLORS[f"{ticker}_light"] for val in monthly_returns[ticker]],
                    line=dict(width=0),
                ),
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
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(
            rangeselector=dict(
                buttons=list(
                    [
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(count=3, label="3y", step="year", stepmode="backward"),
                        dict(step="year", stepmode="todate", label="YTD"),
                        dict(step="all", label="All"),
                    ]
                )
            ),
            rangeslider=dict(visible=False),
            type="date",
        ),
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
class Plots:
    """
    Visualization tools for financial returns data.

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

    data: "Data"  # type: ignore

    def plot_snapshot(self, title: str = "Portfolio Summary", log_scale: bool = False) -> go.Figure:
        """
        Creates a comprehensive dashboard with multiple plots for portfolio analysis.

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
            >>> from jquantstats._data import Data
            >>> import polars as pl
            >>> returns = pl.DataFrame(...)
            >>> data = Data(returns=returns)
            >>> fig = data.plots.plot_snapshot(title="My Portfolio Performance")
            >>> fig.show()
        """
        fig = _plot_performance_dashboard(returns=self.data.all_pd, log_scale=log_scale)
        return fig

    def monthly_heatmap(
        self,
        col: str,
        annot_size: int = 13,
        cbar: bool = True,
        returns_label: str = "Strategy",
        fontname: str = "Arial",
        ylabel: bool = True,
    ) -> go.Figure:
        """
        Creates a heatmap of monthly returns by year.

        This visualization displays returns as a color-coded grid with months on the x-axis
        and years on the y-axis. It provides an intuitive way to identify seasonal patterns
        and compare performance across different time periods.

        Args:
            col (str): The column name of the asset to plot.
            annot_size (int, optional): Font size for annotations. Defaults to 13.
            cbar (bool, optional): Whether to display a color bar. Defaults to True.
            returns_label (str, optional): Label for the returns in the title. Defaults to "Strategy".
            compounded (bool, optional): Whether to use compounded returns. Defaults to False.
            fontname (str, optional): Font family to use. Defaults to "Arial".
            ylabel (bool, optional): Whether to display the y-axis label. Defaults to True.

        Returns:
            go.Figure: A Plotly figure object containing the heatmap.

        Example:
            >>> fig = data.plots.monthly_heatmap("AAPL", returns_label="Apple Inc.")
            >>> fig.show()
        """

        cmap = "RdYlGn"
        date_col = self.data.index.columns[0]

        # Resample monthly
        data = self.data.resample(every="1mo")

        # Prepare DataFrame with Year, Month, Return (%)
        result = data.all.with_columns(
            pl.col(date_col).dt.year().alias("Year"),
            pl.col(date_col).dt.month().alias("Month"),
            (pl.col(col) * 100).alias("Return"),
        )

        # Pivot table (Year x Month)
        pivot = result.pivot(
            values="Return",
            index="Year",
            columns="Month",
            aggregate_function="first",  # Should be fine with monthly data
        ).sort("Year", descending=True)

        # Sort columns by calendar month
        month_cols = [str(m) for m in range(1, 13)]
        pivot = pivot.select("Year", *month_cols)

        # Rename columns to month abbreviations
        new_col_names = ["Year"] + [calendar.month_abbr[int(m)] for m in month_cols]
        pivot.columns = new_col_names

        # Extract z-matrix for heatmap
        z = np.round(pivot.drop("Year").to_numpy(), 2)
        y = pivot["Year"].to_numpy().astype(str)
        x = new_col_names[1:]

        zmin = -np.nanmax(np.abs(z))
        zmax = np.nanmax(np.abs(z))

        fig = go.Figure(
            data=go.Heatmap(
                z=z,
                x=x,
                y=y,
                text=z,
                texttemplate="%{text:.2f}%",
                colorscale=cmap,
                zmid=0,
                zmin=zmin,
                zmax=zmax,
                colorbar=dict(
                    title="Return (%)",
                    ticksuffix="%",
                    tickfont=dict(size=annot_size),
                )
                if cbar
                else None,
                hovertemplate="Year: %{y}<br>Month: %{x}<br>Return: %{z:.2f}%",
            )
        )

        fig.update_layout(
            title={
                "text": f"{returns_label} - Monthly Returns (%)",
                "y": 0.95,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
                "font": dict(family=fontname, size=16, color="black"),
            },
            xaxis=dict(
                title="",
                side="top",
                showgrid=False,
                tickfont=dict(family=fontname, size=annot_size),
            ),
            yaxis=dict(
                title="Years" if ylabel else "",
                autorange="reversed",
                showgrid=False,
                tickfont=dict(family=fontname, size=annot_size),
            ),
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=0, r=0, t=80, b=0),
        )

        return fig
