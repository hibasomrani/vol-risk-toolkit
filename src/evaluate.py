from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.data import ASSETS
from src.forecast import load_forecasts
from src.models import MODELS

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def load_panel(asset, models=MODELS):
    """Aligned squared returns (proxy) and variance forecasts for all models."""
    fs = {m: load_forecasts(asset, m) for m in models}
    idx = fs[models[0]].index
    for f in fs.values():
        idx = idx.intersection(f.index)
    r2 = fs[models[0]].loc[idx, "ret"] ** 2
    h = pd.DataFrame({m: fs[m].loc[idx, "variance"] for m in models})
    return r2, h


def qlike_loss(r2, h):
    return np.log(h) + (1 / h).mul(r2, axis=0)


def squared_loss(r2, h):
    return h.sub(r2, axis=0) ** 2


def dm_test(d, lags=5):
    """Diebold-Mariano: H0 = mean loss differential is zero (HAC standard errors).
    Returns (t-stat, p-value). Negative t means the first model has lower loss."""
    res = sm.OLS(d.values, np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(res.tvalues[0]), float(res.pvalues[0])


def evaluate(asset, start=None, end=None, benchmark="GARCH", models=MODELS):
    """Loss table for one asset, optionally restricted to a date range."""
    r2, h = load_panel(asset, models)
    r2, h = r2.loc[start:end], h.loc[start:end]

    ql = qlike_loss(r2, h)
    sq = squared_loss(r2, h)

    rows = []
    for m in models:
        row = {
            "model": m,
            "QLIKE": ql[m].mean(),
            "RMSE": np.sqrt(sq[m].mean()),
            "DM t vs " + benchmark: np.nan,
            "DM p vs " + benchmark: np.nan,
        }
        if m != benchmark:
            t, p = dm_test(ql[m] - ql[benchmark])
            row["DM t vs " + benchmark] = t
            row["DM p vs " + benchmark] = p
        rows.append(row)
    out = pd.DataFrame(rows).set_index("model")
    out.attrs["n_obs"] = len(r2)
    return out


def evaluate_all(start=None, end=None, assets=list(ASSETS)):
    return {a: evaluate(a, start, end) for a in assets}


if __name__ == "__main__":
    RESULTS_DIR.mkdir(exist_ok=True)
    for asset, tbl in evaluate_all().items():
        print(f"\n=== {asset} (n = {tbl.attrs['n_obs']}) ===")
        print(tbl.round(4).to_string())
        tbl.round(5).to_csv(RESULTS_DIR / f"loss_table_{asset}.csv")