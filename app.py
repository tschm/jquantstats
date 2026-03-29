from fastapi import FastAPI, UploadFile
from fastapi.responses import HTMLResponse
import polars as pl
from jquantstats import Portfolio

app = FastAPI(title="jquantstats API")


@app.get("/")
def root() -> dict:
    """Health check endpoint."""
    return {"status": "jquantstats API running"}


@app.post("/report", response_class=HTMLResponse)
async def generate_report(prices: UploadFile, positions: UploadFile) -> str:
    """Accept prices and positions CSVs and return a full HTML analytics report."""
    prices_df = pl.read_csv(await prices.read())
    positions_df = pl.read_csv(await positions.read())
    pf = Portfolio.from_cash_position(
        prices=prices_df,
        cash_position=positions_df,
        aum=1_000_000,
    )
    return pf.report.to_html()
