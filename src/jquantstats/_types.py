# SPDX-License-Identifier: MIT
"""Type aliases for narwhals-compatible input frames."""

from typing import TypeAlias

import narwhals.typing

NativeFrame: TypeAlias = narwhals.typing.IntoDataFrame
NativeFrameOrScalar: TypeAlias = narwhals.typing.IntoDataFrame | float
