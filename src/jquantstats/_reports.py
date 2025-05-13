import dataclasses

import quantstats as qs


@dataclasses.dataclass(frozen=True)
class Reports:
    data: "Data"  # type: ignore

    def basic(self, grayscale=False, figsize=(8, 5), display=True, compounded=True, periods_per_year=None, **kwargs):
        periods_per_year = periods_per_year or self.data._periods_per_year
        return qs.reports.basic(
            returns=self.data.returns_pd.iloc[:, 0],
            benchmark=self.data.benchmark_pd,
            rf=0.0,
            grayscale=grayscale,
            figsize=figsize,
            display=display,
            compounded=compounded,
            periods_per_year=periods_per_year,
            match_dates=True,
            **kwargs,
        )
