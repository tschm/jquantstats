# SPDX-License-Identifier: MIT
"""jQuantStats API module — backward-compatible convenience wrappers.

The primary public API is :class:`~jquantstats.data.Data` with its
``from_returns`` classmethod.  ``build_data`` is kept as a thin alias
for backward compatibility.
"""

from ._types import NativeFrame, NativeFrameOrScalar
from .data import Data


def build_data(
    returns: NativeFrame,
    rf: NativeFrameOrScalar = 0.0,
    benchmark: NativeFrame | None = None,
    date_col: str = "Date",
) -> Data:
    """Convenience alias for :meth:`Data.from_returns`.

    Prefer ``Data.from_returns(...)`` directly for new code.

    Parameters
    ----------
    returns : NativeFrame
        Financial returns data. First column should be the date column,
        remaining columns are asset returns.

    rf : float | NativeFrame, optional
        Risk-free rate. Default is 0.0 (no risk-free rate adjustment).

    benchmark : NativeFrame | None, optional
        Benchmark returns. Default is None (no benchmark).

    date_col : str, optional
        Name of the date column in the DataFrames. Default is "Date".

    Returns:
    -------
    Data
        Object containing excess returns and benchmark (if any).

    See Also:
    --------
    :meth:`Data.from_returns` : primary spelling of this factory.
    :class:`~jquantstats.analytics.Portfolio` : entry point for price + position data.

    """
    return Data.from_returns(returns=returns, rf=rf, benchmark=benchmark, date_col=date_col)
