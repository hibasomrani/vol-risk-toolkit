from pathlib import Path

import pandas as pd

from src.backtest import backtest_table
from src.data import ASSETS
from src.evaluate import evaluate

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

REGIMES = {
    "Calm 2017-19": ("2017-01-01", "2019-12-31"),
    "COVID 2020": ("2020-02-01", "2020-06-30"),
    "Turmoil 2022": ("2022-01-01", "2022-12-31"),
}


def regime_qlike(assets=list(ASSETS), regimes=REGIMES):
    """QLIKE per model, asset and regime."""
    rows = []
    for asset in assets:
        for name, (s, e) in regimes.items():
            tbl = evaluate(asset, s, e)
            for model, row in tbl.iterrows():
                rows.append({
                    "asset": asset, "regime": name, "model": model,
                    "QLIKE": row["QLIKE"], "RMSE": row["RMSE"],
                })
    return pd.DataFrame(rows)


def regime_backtest(alpha=0.05, assets=list(ASSETS), regimes=REGIMES):
    """Violation counts, rates and tests per model, asset and regime."""
    parts = []
    for asset in assets:
        for name, (s, e) in regimes.items():
            tbl = backtest_table(asset, alphas=(alpha,), start=s, end=e)
            tbl["regime"] = name
            parts.append(tbl)
    return pd.concat(parts, ignore_index=True)


if __name__ == "__main__":
    RESULTS_DIR.mkdir(exist_ok=True)

    q = regime_qlike()
    q.round(5).to_csv(RESULTS_DIR / "regime_qlike.csv", index=False)
    print("\nQLIKE by regime (lower is better)")
    for asset, g in q.groupby("asset", sort=False):
        print(f"\n=== {asset} ===")
        print(g.pivot(index="model", columns="regime", values="QLIKE")[list(REGIMES)].round(4))

    b = regime_backtest(0.05)
    b.round(4).to_csv(RESULTS_DIR / "regime_backtest_95.csv", index=False)
    print("\n95% VaR violation rate (%) by regime (expected: 5.0)")
    for asset, g in b.groupby("asset", sort=False):
        print(f"\n=== {asset} ===")
        print(g.pivot(index="model", columns="regime", values="rate_%")[list(REGIMES)].round(2))