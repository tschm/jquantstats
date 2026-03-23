"""Reports subpackage for jquantstats."""

from ._data import Reports as Reports
from ._portfolio import Report as Report
from ._portfolio import _fmt as _fmt
from ._portfolio import _stats_table_html as _stats_table_html
from ._protocol import DataLike as DataLike
from ._protocol import PortfolioLike as PortfolioLike

__all__ = ["DataLike", "PortfolioLike", "Report", "Reports", "_fmt", "_stats_table_html"]
