# SPDX-License-Identifier: MIT
"""Type aliases for narwhals-compatible input frames."""

from typing import TypeAlias

import narwhals as nw

NativeFrame: TypeAlias = nw.typing.IntoDataFrame
NativeFrameOrScalar: TypeAlias = nw.typing.IntoDataFrame | float
