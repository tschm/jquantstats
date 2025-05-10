"""global fixtures"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from jquantstats.api import build_data


@pytest.fixture(scope="session", name="resource_dir")
def resource_fixture():
    """resource fixture"""
    return Path(__file__).parent / "resources"


@pytest.fixture
def returns(resource_dir) -> pl.DataFrame:
    # Only feed in frames. No series.
    df = pl.read_csv(resource_dir / "meta.csv", try_parse_dates=True)
    return df.select(["Date", "Meta"])


@pytest.fixture
def benchmark(resource_dir) -> pl.DataFrame:
    df = pl.read_csv(resource_dir / "benchmark.csv", try_parse_dates=True)
    return df.select(["Date", "SPY -- Benchmark"])



@pytest.fixture
def portfolio(resource_dir):
    # that's interesting, polars thinks META is a str type
    return pl.read_csv(resource_dir / "portfolio.csv", try_parse_dates=True).with_columns([
        pl.col("AAPL").cast(pl.Float64, strict=False),
        pl.col("META").cast(pl.Float64, strict=False),
        pl.col("Date").cast(pl.Date, strict=False)
    ])

@pytest.fixture
def data(portfolio, benchmark):
    return build_data(returns=portfolio, benchmark=benchmark)

# @pytest.fixture
# def edge(data):
#     index = data.index
#     returns = pl.DataFrame({"index": index, "returns": [0.0] * len(index)})
#     benchmark = pl.DataFrame({"index": index, "benchmark": [0.0] * len(index)})
#
#     benchmark = benchmark.to_pandas().set_index("index")
#     benchmark.index = benchmark.index.astype('datetime64[ns]')
#
#     returns = returns.to_pandas().set_index("index")
#     returns.index = returns.index.astype('datetime64[ns]')
#
#     #returns = pd.Series(index=data.index, data=0.0)
#     #benchmark = pd.Series(index=data.index, data=0.0)
#     return build_data(returns=returns, benchmark=benchmark)
