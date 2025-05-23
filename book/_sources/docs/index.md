# [jQuantStats](https://tschm.github.io/jquantstats/book): Portfolio analytics for quants

[![PyPI version](https://badge.fury.io/py/jquantstats.svg)](https://badge.fury.io/py/jquantstats)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.txt)
[![CI](https://github.com/tschm/jquantstats/actions/workflows/ci.yml/badge.svg)](https://github.com/tschm/jquantstats/actions/workflows/ci.yml)
[![Coverage Status](https://coveralls.io/repos/github/tschm/jquantstats/badge.svg?branch=main)](https://coveralls.io/github/tschm/jquantstats?branch=main)
[![CodeFactor](https://www.codefactor.io/repository/github/tschm/jquantstats/badge)](https://www.codefactor.io/repository/github/tschm/jquantstats)
[![Renovate enabled](https://img.shields.io/badge/renovate-enabled-brightgreen.svg)](https://github.com/renovatebot/renovate)

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/tschm/jquantstats)

## Overview

**jQuantStats** is a Python library for portfolio analytics
that helps quants and portfolio managers understand
their performance through in-depth analytics and risk metrics.
It provides tools for calculating various performance metrics
and visualizing portfolio performance.

The library is inspired by and currently exposes a subset of the
functionality of [QuantStats](https://github.com/ranaroussi/quantstats),
focusing on providing a clean, modern API with enhanced
visualization capabilities using Plotly.

We have made the following changes when compared to quantstats:

- added tests (based on pytest), pre-commit hooks and
  github ci/cd workflows
- removed a direct dependency on yfinance to inject data
- moved all graphical output to plotly and removed the matplotlib dependency
- removed some statistical metrics but intend to bring them back
- moved to Marimo for demos
- gave up on the very tight coupling with pandas

Along the way we broke downwards compatibility with quantstats but the
underlying usage pattern is too different. Users familiar with
Dataclasses may find the chosen path appealing.
A data class is
constructed using the `build_data` function.
This function is essentially
the only viable entry point into jquantstats.
It constructs and returns
a `_Data` object which exposes plots and stats via its member attributes.

At this early stage the user would have to define a benchmark
and set the underlying risk-free rate.

## Features

- **Performance Metrics**: Calculate key metrics like Sharpe ratio, Sortino ratio,
  drawdowns, volatility, and many more
- **Risk Analysis**: Analyze risk through metrics like Value at Risk (VaR),
  Conditional VaR, and drawdown analysis
- **Visualization**: Create interactive plots for portfolio performance, drawdowns,
  return distributions, and monthly heatmaps
- **Benchmark Comparison**: Compare your portfolio performance against benchmarks

## Installation

```bash
pip install jquantstats
```

For development:

```bash
pip install jquantstats[dev]
```

## Quick Start

```python
import polars as pl
from jquantstats.api import build_data

# Create a Data object from returns
returns = pl.DataFrame(...)  # Your returns data

# Basic usage
data = build_data(returns=returns)

# With benchmark and risk-free rate
benchmark = pl.DataFrame(...)  # Your benchmark returns
data = build_data(
    returns=returns,
    benchmark=benchmark,
    rf=0.0002,      # risk-free rate (e.g., 0.02% per day)
)

# Calculate statistics
sharpe = data.stats.sharpe()
volatility = data.stats.volatility()

# Create visualizations
fig = data.plots.plot_snapshot(title="Portfolio Performance")
fig.show()

# Monthly returns heatmap
fig = data.plots.monthly_heatmap()
fig.show()
```

## Documentation

For detailed documentation,
visit [jQuantStats Documentation](https://tschm.github.io/jquantstats/book).

## Requirements

- Python 3.10+
- numpy
- polars
- plotly
- kaleido (for static image export)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the
Apache License 2.0 - see the [LICENSE.txt](LICENSE.txt) file for details.
