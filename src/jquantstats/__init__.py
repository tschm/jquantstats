"""jQuantStats: Portfolio analytics for quants.

Overview
--------
jQuantStats is a Python library for portfolio analytics that helps quants and portfolio
managers understand their performance through in-depth analytics and risk metrics.
It provides tools for calculating various performance metrics and visualizing
portfolio performance.

Features
--------
- Performance Metrics: Calculate key metrics like Sharpe ratio, Sortino ratio,
  drawdowns, volatility, and more
- Risk Analysis: Analyze risk through metrics like Value at Risk (VaR),
  Conditional VaR, and drawdown analysis
- Visualization: Create interactive plots for portfolio performance, drawdowns,
  return distributions, and monthly heatmaps
- Benchmark Comparison: Compare your portfolio performance against benchmarks
- Support for both pandas and polars DataFrames

Installation
-----------
```bash
pip install jquantstats
```

Usage
-----
The main entry point is the `build_data` function in the api module, which creates
a Data object from returns and optional benchmark data.

```python
import polars as pl
from jquantstats.api import build_data

# Create sample returns data
returns = pl.DataFrame({
    "Date": ["2023-01-01", "2023-01-02", "2023-01-03"],
    "Asset1": [0.01, -0.02, 0.03],
    "Asset2": [0.02, 0.01, -0.01]
}).with_columns(pl.col("Date").str.to_date())

# Create a Data object
data = build_data(returns=returns)

# Calculate statistics
sharpe = data.stats.sharpe()
volatility = data.stats.volatility()

# Create visualizations
fig = data.plots.plot_snapshot(title="Portfolio Performance")
fig.show()
```

For more information, visit the [jQuantStats Documentation](https://tschm.github.io/jquantstats/book).
"""

import importlib.metadata

from .api import build_data  # noqa: F401

__version__ = importlib.metadata.version("jquantstats")
