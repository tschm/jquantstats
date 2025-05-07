"""global fixtures"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import polars as pl
import pytest

from jquantstats.api import build_data


@pytest.fixture(scope="session", name="resource_dir")
def resource_fixture():
    """resource fixture"""
    return Path(__file__).parent / "resources"


@pytest.fixture
def returns(resource_dir):
    # Only feed in frames. No series.
    df = pl.read_csv(resource_dir / "meta.csv", try_parse_dates=True)
    #df = df.rename({"Close": "Meta"})
    frame = df.select(["Date", "Meta"])
    frame = frame.to_pandas().set_index("Date")
    frame.index = frame.index.astype('datetime64[ns]')

    return frame


@pytest.fixture
def benchmark(resource_dir):
    #x = pd.read_csv(resource_dir / "benchmark.csv", parse_dates=True, index_col=0)["Close"]

    df = pl.read_csv(resource_dir / "benchmark.csv", try_parse_dates=True)
    frame = df.select(["Date", "SPY -- Benchmark"])
    frame = frame.to_pandas().set_index("Date")
    frame.index = frame.index.astype('datetime64[ns]')

    return frame["SPY -- Benchmark"]


@pytest.fixture
def portfolio(resource_dir):
    df = pl.read_csv(resource_dir / "portfolio.csv", try_parse_dates=True).with_columns([
        pl.col("AAPL").cast(pl.Float64, strict=False),
        pl.col("META").cast(pl.Float64, strict=False)
    ])
    frame = df.to_pandas().set_index("Date")
    frame.index = frame.index.astype('datetime64[ns]')

    #frame = pd.read_csv(resource_dir / "portfolio.csv", parse_dates=True, index_col=0)
    return frame

@pytest.fixture
def data(portfolio, benchmark):
    return build_data(returns=portfolio, benchmark=benchmark)


@pytest.fixture
def edge(data):
    returns = pd.Series(index=data.index, data=0.0)
    benchmark = pd.Series(index=data.index, data=0.0)
    return build_data(returns=returns, benchmark=benchmark)
