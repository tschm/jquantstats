import dataclasses

import polars as pl


@dataclasses.dataclass(frozen=True)
class Reports:
    """
    A class for generating financial reports from Data objects.

    This class provides methods for calculating and formatting various financial metrics
    into report-ready formats such as DataFrames.

    Attributes:
        data (Data): The financial data object to generate reports from.
    """

    data: "Data"  # type: ignore

    def metrics(self) -> pl.DataFrame:
        """
        Calculate various financial metrics and return them as a DataFrame.

        Returns:
            pl.DataFrame: A DataFrame containing financial metrics as rows and assets as columns.
                          The first column is 'Metric' containing the name of each metric.
        """
        metrics: dict[str, dict[str, float]] = {
            "Sharpe Ratio": self.data.stats.sharpe(),
            "Sortino Ratio": self.data.stats.sortino(),
            # "Calmar Ratio": self.data.stats.calmar(),
            "Max Drawdown": self.data.stats.max_drawdown(),
            "Volatility": self.data.stats.volatility(),
            # "CAGR": self.data.stats.cagr(),
            "Value at Risk (5%)": self.data.stats.value_at_risk(alpha=0.05),
            "Win/Loss Ratio": self.data.stats.win_loss_ratio(),
            # "Mean Daily Return": self.data.returns.mean(),
            # "Std Dev": self.data.returns.std().to_dict(),
            "Skew": self.data.stats.skew(),
            "Kurtosis": self.data.stats.kurtosis(),
        }

        # convert to Polars DataFrame with metrics as rows, assets as columns
        return pl.DataFrame([{"Metric": name, **vals} for name, vals in metrics.items()])
