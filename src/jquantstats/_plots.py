import calendar
import dataclasses

import numpy as np
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots


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

    _FLATUI_COLORS = [
        "#FEDD78",  # Yellow
        "#348DC1",  # Blue
        "#BA516B",  # Rose
        "#4FA487",  # Green
        "#9B59B6",  # Purple
        "#613F66",  # Dark Purple
        "#84B082",  # Light Green
        "#DC136C",  # Pink
        "#559CAD",  # Light Blue
        "#4A5899",  # Navy Blue
    ]

    @staticmethod
    def _get_colors() -> tuple[list[str], str, float]:
        """
        Returns the default color palette and styling parameters for plots.

        Returns:
            tuple[list[str], str, float]: A tuple containing:
                - colors (list[str]): List of hex color codes
                - ls (str): Line style ("-" for solid)
                - alpha (float): Opacity value (0.8)
        """
        colors = Plots._FLATUI_COLORS
        ls = "-"  # Line style
        alpha = 0.8  # Opacity
        return colors, ls, alpha

    def plot_returns_bars(self) -> go.Figure:
        """
        Creates a bar chart of returns for each asset in the data.

        This function visualizes the returns of each asset as bars, making it easy
        to compare performance across different time periods.

        Returns:
            go.Figure: A Plotly figure object containing the bar chart.
                The figure shows returns for each asset with a horizontal line at y=0.

        Example:
            >>> from jquantstats._data import Data
            >>> import polars as pl
            >>> returns = pl.DataFrame(...)
            >>> data = Data(returns=returns)
            >>> fig = data.plots.plot_returns_bars()
            >>> fig.show()
        """
        # Get color palette
        colors, _, _ = Plots._get_colors()

        # Create figure
        fig = go.Figure()

        # Add a bar trace for each asset
        for idx, col in enumerate(self.data.returns.columns):
            fig.add_trace(
                go.Bar(
                    x=self.data.index,
                    y=self.data.returns[col],
                    name=col,
                    marker_color=colors[idx % len(colors)],  # Cycle through colors if more assets than colors
                )
            )

        # Update layout for better readability
        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis=dict(
                tickformat="%Y",  # Format x-axis as years
                showgrid=False,
            ),
            yaxis=dict(
                tickformat=".0%",  # Format y-axis as percentages
                showgrid=True,
                gridcolor="lightgray",
            ),
        )

        # Add horizontal line at y=0 to distinguish positive and negative returns
        fig.add_hline(y=0, line=dict(color="black", width=1, dash="dash"))

        return fig

    def plot_snapshot(
        self, title: str = "Portfolio Summary", compounded: bool = True, log_scale: bool = False
    ) -> go.Figure:
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
        # Calculate drawdowns
        dd = self.data.stats.drawdown(compounded=compounded, initial=100)

        # Create subplot structure
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,  # Share x-axis across all subplots
            row_heights=[0.5, 0.25, 0.25],  # Allocate more space to cumulative returns
            vertical_spacing=0.03,
            subplot_titles=["Cumulative Return", "Drawdown", "Daily Return"],
        )

        # Plot cumulative returns for each asset
        for col in self.data.returns.columns:
            cum_returns = 100 * ((1 + self.data.returns[col]).cum_prod())  # Convert to percentage
            fig.add_trace(
                go.Scatter(
                    x=self.data.index[self.data.index.columns[0]],
                    y=cum_returns,
                    name=col,
                    mode="lines",
                ),
                row=1,
                col=1,
            )

        # Plot drawdowns for each asset
        for col in self.data.returns.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.data.index[self.data.index.columns[0]],
                    y=dd[col],
                    name=f"DD: {col}",
                    mode="lines",
                ),
                row=2,
                col=1,
            )

        # Plot daily returns for each asset
        for col in self.data.assets:
            fig.add_trace(
                go.Scatter(
                    x=self.data.index[self.data.index.columns[0]],
                    y=self.data.all[col] * 100,  # Convert to percentage
                    name=f"{col} Return",
                    mode="lines",
                ),
                row=3,
                col=1,
            )

        # Configure layout
        fig.update_layout(
            height=800,  # Taller figure for better visibility
            title_text=title,
            showlegend=True,
            template="plotly_white",  # Clean white template
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        # Apply log scale to cumulative returns if requested
        if log_scale:
            fig.update_yaxes(type="log", row=1, col=1)

        # Format y-axes
        fig.update_yaxes(title="Cumulative Return (%)", row=1, col=1)
        fig.update_yaxes(title="Drawdown", tickformat=".1%", row=2, col=1)
        fig.update_yaxes(title="Daily Return (%)", row=3, col=1)

        return fig

    def monthly_heatmap(
        self,
        col: str,
        annot_size: int = 13,
        cbar: bool = True,
        returns_label: str = "Strategy",
        compounded: bool = False,
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
        data = self.data.resample(every="1mo", compounded=compounded)

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
