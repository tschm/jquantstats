# SPDX-License-Identifier: MIT
"""Type aliases for narwhals-compatible input frames."""

from typing import TypeAlias

import narwhals.typing as nw_typing

NativeFrame: TypeAlias = nw_typing.IntoDataFrame
NativeFrameOrScalar: TypeAlias = nw_typing.IntoDataFrame | float
