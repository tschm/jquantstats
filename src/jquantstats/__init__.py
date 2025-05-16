"""
jquantstats: Portfolio analytics for quants

This package provides tools for analyzing and visualizing financial returns data.
It includes functionality for calculating various financial metrics, generating
reports, and creating visualizations.

The main entry point is the build_data function in the api module, which creates
a Data object from returns and optional benchmark data.

Example:
    >>> import polars as pl
    >>> from jquantstats.api import build_data
    >>> returns = pl.DataFrame(...)
    >>> data = build_data(returns=returns)
    >>> sharpe = data.stats.sharpe()
    >>> fig = data.plots.plot_returns_bars()
"""
from .api import build_data
