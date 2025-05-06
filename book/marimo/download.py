import marimo

__generated_with = "0.13.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    import yfinance as yf

    def download_returns(ticker, period="max", proxy=None):
        params = {"tickers": ticker, "auto_adjust": True, "multi_level_index": False, "progress": False, "proxy": proxy}
        if isinstance(period, pd.DatetimeIndex):
            params["start"] = period[0]
        else:
            params["period"] = period
        df = yf.download(**params)["Close"].pct_change()
        df = df.tz_localize(None)
        return df

    return (download_returns,)


@app.cell
def _(download_returns):
    stock = download_returns("META")
    stock.to_csv("meta.csv")
    return


@app.cell
def _(download_returns):
    spy = download_returns("SPY")
    spy.to_csv("benchmark.csv")
    return


@app.cell
def _(download_returns):
    portfolio = download_returns(ticker=["AAPL", "META"])
    portfolio.to_csv("portfolio.csv")
    return


if __name__ == "__main__":
    app.run()
