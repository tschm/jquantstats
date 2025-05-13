import dataclasses

import polars as pl


@dataclasses.dataclass(frozen=True)
class Reports:
    data: "Data"  # type: ignore

    def metrics(self):
        metrics = {
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
