from jquantstats._plots import _plot_performance_dashboard

if __name__ == '__main__':
    import yfinance as yf

    # === Config ===
    tickers = ["TSLA", "SPY", "AAPL"]  # Change this list freely
    start_date = "2020-01-01"
    end_date = "2025-01-01"

    # === Download data ===
    data = yf.download(tickers, start=start_date, end=end_date)["Close"]
    returns = data.pct_change().dropna()

    fig = _plot_performance_dashboard(returns)
    fig.show()
