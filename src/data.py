from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

ASSETS = {
    "SP500": "^GSPC",
    "BTC": "BTC-USD",
    "GOLD": "GC=F",
    "EM": "EEM",
}


def download_prices(ticker, start="2010-01-01", end=None, force=False):
    """Download adjusted close prices, cached as CSV in data/."""
    DATA_DIR.mkdir(exist_ok=True)
    safe = ticker.replace("^", "").replace("=", "")
    path = DATA_DIR / f"{safe}.csv"

    if path.exists() and not force:
        return pd.read_csv(path, index_col=0, parse_dates=True).squeeze("columns")

    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")

    prices = df["Close"].squeeze()
    prices.name = ticker
    prices.to_csv(path)
    return prices


def log_returns(prices):
    """Daily log returns in percent (arch works best at this scale)."""
    r = 100 * np.log(prices).diff().dropna()
    r.name = prices.name
    return r


def load_returns(name, **kwargs):
    """Accepts an alias from ASSETS ('SP500') or a raw ticker ('AAPL')."""
    ticker = ASSETS.get(name, name)
    return log_returns(download_prices(ticker, **kwargs))


if __name__ == "__main__":
    rows = []
    for alias in ASSETS:
        r = load_returns(alias)
        rows.append({
            "asset": alias,
            "start": r.index[0].date(),
            "end": r.index[-1].date(),
            "n_obs": len(r),
            "mean_%": round(r.mean(), 4),
            "std_%": round(r.std(), 4),
            "skew": round(r.skew(), 3),
            "kurtosis": round(r.kurtosis() + 3, 3),
        })
    print(pd.DataFrame(rows).to_string(index=False))