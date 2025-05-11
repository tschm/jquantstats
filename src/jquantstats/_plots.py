import calendar
import dataclasses

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


@dataclasses.dataclass(frozen=True)
class Plots:
    data: "_Data"  # type: ignore

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
    def _get_colors():
        """
        Returns the default color palette and styling parameters for plots.

        Returns:
            tuple: A tuple containing:
                - colors (list): List of hex color codes
                - ls (str): Line style ("-" for solid)
                - alpha (float): Opacity value (0.8)
        """
        colors = Plots._FLATUI_COLORS
        ls = "-"  # Line style
        alpha = 0.8  # Opacity
        return colors, ls, alpha

    @staticmethod
    def _compsum(returns):
        """Calculates rolling compounded returns"""
        return returns.add(1).cumprod(axis=0) - 1

    def plot_returns_bars(self):
        """
        Creates a bar chart of returns for each asset in the data.

        This function visualizes the returns of each asset as bars, making it easy
        to compare performance across different time periods.

        Args:
            data (_Data): A Data object containing returns data to plot.

        Returns:
            plotly.graph_objects.Figure: A Plotly figure object containing the bar chart.
                The figure shows returns for each asset with a horizontal line at y=0.

        Example:
            >>> from jquantstats.api import _Data
            >>> import pandas as pd
            >>> returns = pd.DataFrame(...)
            >>> data = _Data(returns=returns)
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

    def plot_snapshot(self, title="Portfolio Summary", compounded=True, log_scale=False):
        """
        Creates a comprehensive dashboard with multiple plots for portfolio analysis.

        This function generates a three-panel plot showing:
        1. Cumulative returns over time
        2. Drawdowns over time
        3. Daily returns over time

        This provides a complete visual summary of portfolio performance.

        Args:
            data (_Data): A Data object containing returns data.
            title (str, optional): Title of the plot. Defaults to "Portfolio Summary".
            compounded (bool, optional): Whether to use compounded returns. Defaults to True.
            log_scale (bool, optional): Whether to use logarithmic scale for cumulative returns.
                Defaults to False.

        Returns:
            plotly.graph_objects.Figure: A Plotly figure object containing the dashboard.

        Example:
            >>> from jquantstats.api import _Data
            >>> import pandas as pd
            >>> returns = pd.DataFrame(...)
            >>> data = _Data(returns=returns)
            >>> fig = snapshot_plotly(data, title="My Portfolio Performance")
            >>> fig.show()
        """
        # Calculate drawdowns
        dd = self.data.drawdown(compounded=compounded)

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
        for col in self.returns.columns:
            cum_returns = 100 * ((1 + self.data.returns[col]).cumprod())  # Convert to percentage
            fig.add_trace(
                go.Scatter(
                    x=cum_returns.index,
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
                    x=dd[col].index,
                    y=dd[col],
                    name=f"DD: {col}",
                    mode="lines",
                ),
                row=2,
                col=1,
            )

        # Plot daily returns for each asset
        for col in self.data.names:
            fig.add_trace(
                go.Scatter(
                    x=self.data.returns[col].index,
                    y=self.data.returns[col] * 100,  # Convert to percentage
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
        annot_size=13,
        cbar=True,
        returns_label="Strategy",
        compounded=True,
        eoy=False,
        fontname="Arial",
        ylabel=True,
    ):
        """
        Creates a heatmap of monthly returns by year.

        This function visualizes monthly returns in a calendar-like heatmap format,
        with years on the y-axis and months on the x-axis. Positive returns are shown
        in green, negative in red, with color intensity proportional to return magnitude.

        Args:
            data (_Data): A Data object containing returns data.
            annot_size (int, optional): Font size for annotations. Defaults to 13.
            cbar (bool, optional): Whether to show the color bar. Defaults to True.
            returns_label (str, optional): Label for the returns data. Defaults to "Strategy".
            compounded (bool, optional): Whether to use compounded returns. Defaults to True.
            eoy (bool, optional): Whether to include end-of-year summary. Defaults to False.
            fontname (str, optional): Font family to use. Defaults to "Arial".
            ylabel (bool, optional): Whether to show y-axis label. Defaults to True.

        Returns:
            plotly.graph_objects.Figure: A Plotly figure object containing the heatmap.

        Example:
            >>> from jquantstats.api import _Data
            >>> import pandas as pd
            >>> returns = pd.DataFrame(...)
            >>> data = _Data(returns=returns)
            >>> fig = data.plots.monthly_heatmap(returns_label="My Portfolio")
            >>> fig.show()
        """
        # Define color map (Red-Yellow-Green)
        cmap = "RdYlGn"

        # Prepare monthly returns as percentage
        data = self.data.resample(every="1m", compounded=compounded)

        # Extract returns for the first asset
        returns = data.returns[data.returns.columns[0]] * 100
        # returns.index = pd.to_datetime(returns.index)

        # Convert to DataFrame for manipulation
        # returns = returns.to_frame()
        print(data.index)
        print(dir(data.index))
        # Add Year and Month columns
        returns["Year"] = returns.index.year
        returns["Month"] = returns.index.month

        # Create pivot table with years as rows and months as columns
        returns = returns.pivot(index="Year", columns="Month", values=self.data.names[0]).fillna(0)

        # Rename month numbers to month abbreviations
        returns = returns.rename(columns=lambda x: calendar.month_abbr[x])

        # Calculate color scale limits to ensure symmetry around zero
        zmin = -max(abs(returns.min().min()), abs(returns.max().max()))
        zmax = max(abs(returns.min().min()), abs(returns.max().max()))

        # Set index name and reverse order (most recent years on top)
        returns.index.name = "Year"
        returns.columns.name = None
        returns = returns.iloc[::-1]

        # Create heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=returns.values,
                x=[col for col in returns.columns],  # Month abbreviations
                y=returns.index.astype(str),  # Years
                text=np.round(returns.values, 2),
                texttemplate="%{text:.2f}%",  # Annotate cells with return values
                colorscale=cmap,
                zmid=0,  # Center color scale at zero
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

        # Configure layout
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
                side="top",  # Show months at the top
                showgrid=False,
                tickfont=dict(family=fontname, size=annot_size),
            ),
            yaxis=dict(
                title="Years" if ylabel else "",
                autorange="reversed",  # Most recent years at the top
                showgrid=False,
                tickfont=dict(family=fontname, size=annot_size),
            ),
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=0, r=0, t=80, b=0),
        )

        return fig

    def plot_distribution(self, fontname="Arial", compounded=True, title=None):
        """
        Creates a box plot showing the distribution of returns across different time periods.

        This function visualizes the distribution of returns at daily, weekly, monthly,
        quarterly, and yearly frequencies, making it easy to compare volatility and
        central tendency across different time horizons.

        Args:
            data (_Data): A Data object containing returns data.
            fontname (str, optional): Font family to use. Defaults to "Arial".
            compounded (bool, optional): Whether to use compounded returns. Defaults to True.
            title (str, optional): Title of the plot. If None, defaults to "Return Quantiles".

        Returns:
            plotly.graph_objects.Figure: A Plotly figure object containing the distribution plot.

        Example:
            >>> from jquantstats.api import _Data
            >>> import pandas as pd
            >>> returns = pd.DataFrame(...)
            >>> data = _Data(returns=returns)
            >>> fig = data.plots.plot_distribution(title="Portfolio Returns Distribution")
            >>> fig.show()
        """
        # Get color palette
        colors = Plots._FLATUI_COLORS

        # Extract returns for the first asset and ensure DataFrame format
        port = pd.DataFrame(self.data.returns[self.data.returns.columns[0]]).fillna(0)
        port.columns = ["Daily"]

        # Define function to apply when resampling (compound or sum)
        apply_fnc = Plots._compsum if compounded else np.sum

        # Resample returns to different frequencies
        port["Weekly"] = port["Daily"].resample("W-MON").apply(apply_fnc).ffill()
        port["Monthly"] = port["Daily"].resample("ME").apply(apply_fnc).ffill()
        port["Quarterly"] = port["Daily"].resample("QE").apply(apply_fnc).ffill()
        port["Yearly"] = port["Daily"].resample("YE").apply(apply_fnc).ffill()

        # Create figure
        fig = go.Figure()

        # Add box plots for each time frequency
        for i, col in enumerate(["Daily", "Weekly", "Monthly", "Quarterly", "Yearly"]):
            fig.add_trace(
                go.Box(
                    y=port[col],
                    name=col,
                    marker_color=colors[i],
                    boxmean="sd",  # Show mean and standard deviation
                )
            )

        # Set title
        if not title:
            title = "Return Quantiles"
        else:
            title = f"{title} - Return Quantiles"

        # Create date range string for subtitle
        date_range = f"{self.data.index.min():%d %b '%y} - {self.data.returns.index.max():%d %b '%y}"

        # Configure layout
        fig.update_layout(
            title={
                "text": f"<b>{title}</b><br><sub>{date_range}</sub>",
                "y": 0.9,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
            },
            font=dict(family=fontname, size=14),
            yaxis_title="Returns (%)",
            yaxis_tickformat=".0%",
            boxmode="group",
            plot_bgcolor="white",
            paper_bgcolor="white",
        )

        return fig
