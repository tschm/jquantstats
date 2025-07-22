# [jQuantStats](https://tschm.github.io/jquantstats/book): Portfolio Analytics for Quants

[![PyPI version](https://badge.fury.io/py/jquantstats.svg)](https://badge.fury.io/py/jquantstats)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE.txt)
[![CI](https://github.com/tschm/jquantstats/actions/workflows/ci.yml/badge.svg)](https://github.com/tschm/jquantstats/actions/workflows/ci.yml)
[![Downloads](https://static.pepy.tech/personalized-badge/jquantstats?period=month&units=international_system&left_color=black&right_color=orange&left_text=PyPI%20downloads%20per%20month)](https://pepy.tech/project/jquantstats)
[![Coverage Status](https://coveralls.io/repos/github/tschm/jquantstats/badge.svg?branch=main)](https://coveralls.io/github/tschm/jquantstats?branch=main)
[![CodeFactor](https://www.codefactor.io/repository/github/tschm/jquantstats/badge)](https://www.codefactor.io/repository/github/tschm/jquantstats)
[![Renovate enabled](https://img.shields.io/badge/renovate-enabled-brightgreen.svg)](https://github.com/renovatebot/renovate)

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/tschm/jquantstats)

## 📊 Overview

**jQuantStats** is a Python library for portfolio analytics
that helps quants and portfolio managers understand their performance
through in-depth analytics and risk metrics. It provides tools
for calculating various performance metrics and visualizing
portfolio performance using interactive Plotly charts.

The library is inspired by [QuantStats](https://github.com/ranaroussi/quantstats),
but focuses on providing a clean, modern API with
enhanced visualization capabilities. Key improvements include:

- Support for both pandas and polars DataFrames
- Modern interactive visualizations using Plotly
- Comprehensive test coverage with pytest
- Clean, well-documented API
- Efficient data processing with polars

## ✨ Features

- **Performance Metrics**: Calculate key metrics like Sharpe ratio,
Sortino ratio, drawdowns, volatility, and more
- **Risk Analysis**: Analyze risk through metrics like
Value at Risk (VaR), Conditional VaR, and drawdown analysis
- **Interactive Visualizations**: Create interactive
plots for portfolio performance, drawdowns, and
return distributions
- **Benchmark Comparison**: Compare your portfolio performance against benchmarks
- **Pandas & Polars Support**: Work with either pandas or polars DataFrames as input

## 📦 Installation

```bash
pip install jquantstats
```

For development:

```bash
pip install jquantstats[dev]
```

## 🚀 Quick Start

```python
>>> # Import jquantstats
>>> import polars as pl
>>> from jquantstats import build_data

>>> # Create sample returns data
>>> returns = pl.DataFrame({
...     "Date": ["2023-01-01", "2023-01-02", "2023-01-03"],
...     "Asset1": [0.01, -0.02, 0.03],
...     "Asset2": [0.02, 0.01, -0.01]
... }).with_columns(pl.col("Date").str.to_date())
>>> returns
shape: (3, 3)
┌────────────┬────────┬────────┐
│ Date       ┆ Asset1 ┆ Asset2 │
│ ---        ┆ ---    ┆ ---    │
│ date       ┆ f64    ┆ f64    │
╞════════════╪════════╪════════╡
│ 2023-01-01 ┆ 0.01   ┆ 0.02   │
│ 2023-01-02 ┆ -0.02  ┆ 0.01   │
│ 2023-01-03 ┆ 0.03   ┆ -0.01  │
└────────────┴────────┴────────┘

>>> # Basic usage
>>> data = build_data(returns=returns)
>>>
>>> # With benchmark and risk-free rate
>>> benchmark = pl.DataFrame({
...     "Date": ["2023-01-01", "2023-01-02", "2023-01-03"],
...     "Market": [0.005, -0.01, 0.02]
... }).with_columns(pl.col("Date").str.to_date())
>>> benchmark
shape: (3, 2)
┌────────────┬────────┐
│ Date       ┆ Market │
│ ---        ┆ ---    │
│ date       ┆ f64    │
╞════════════╪════════╡
│ 2023-01-01 ┆ 0.005  │
│ 2023-01-02 ┆ -0.01  │
│ 2023-01-03 ┆ 0.02   │
└────────────┴────────┘

>>> data = build_data(
...     returns=returns,
...     benchmark=benchmark,
...     rf=0.0002,  # risk-free rate (e.g., 0.02% per day)
... )

>>> # Calculate statistics
>>> sharpe = data.stats.sharpe()
>>> sharpe
{'Asset1': np.float64(4.909200099205072), 'Asset2': np.float64(8.08795106197808), 'Market': np.float64(6.113591415853696)}

>>> volatility = data.stats.volatility()
>>> volatility
{'Asset1': np.float64(0.4807979478602905), 'Asset2': np.float64(0.2918332857414772), 'Market': np.float64(0.286574597618142)}

>>> # Create visualizations
>>> fig = data.plots.plot_snapshot(title="Portfolio Performance")
>>> type(fig)
<class 'plotly.graph_objs._figure.Figure'>
>>> # End of example
```

## 📚 Documentation

For detailed documentation, visit [jQuantStats Documentation](https://tschm.github.io/jquantstats/book).

## 🔧 Requirements

- Python 3.10+
- numpy
- polars
- pandas
- plotly
- scipy

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## ⚖️ License

This project is licensed under the Apache
License 2.0 - see the [LICENSE.txt](LICENSE.txt) file for details.
