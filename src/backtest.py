from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import chi2, norm, t

from src.data import ASSETS
from src.forecast import load_forecasts
from src.models import MODELS

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def var_es(f, alpha):
    """VaR and ES (positive numbers, in % of the position) from stored forecasts.

    Rows with a t degrees-of-freedom value use the standardized Student-t,
    rows without one (HIST, EWMA) use the normal distribution.
    """
    sigma = np.sqrt(f["variance"].values)
    mu = f["mu"].values
    nu = f["nu"].values

    q = np.full(len(f), norm.ppf(alpha))                      # quantile of the standardized shock
    es_z = np.full(len(f), -norm.pdf(norm.ppf(alpha)) / alpha)  # E[z | z < q]

    m = ~np.isnan(nu)
    if m.any():
        n = nu[m]
        tq = t.ppf(alpha, n)
        scale = np.sqrt((n - 2) / n)
        q[m] = tq * scale
        es_z[m] = -(t.pdf(tq, n) / alpha) * (n + tq**2) / (n - 1) * scale

    return pd.DataFrame(
        {"VaR": -(mu + sigma * q), "ES": -(mu + sigma * es_z)}, index=f.index
    )


def kupiec(viol, alpha):
    """Unconditional coverage test. H0: violation rate = alpha."""
    n, x = len(viol), int(viol.sum())
    pi = x / n
    ll0 = xlogy(n - x, 1 - alpha) + xlogy(x, alpha)
    ll1 = xlogy(n - x, 1 - pi) + xlogy(x, pi)
    lr = max(-2 * (ll0 - ll1), 0.0)
    return lr, 1 - chi2.cdf(lr, 1)


def christoffersen(viol):
    """Independence test. H0: a violation today does not change tomorrow's odds."""
    v = np.asarray(viol, dtype=int)
    a, b = v[:-1], v[1:]
    n00 = ((a == 0) & (b == 0)).sum()
    n01 = ((a == 0) & (b == 1)).sum()
    n10 = ((a == 1) & (b == 0)).sum()
    n11 = ((a == 1) & (b == 1)).sum()

    p01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    p11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    p = (n01 + n11) / (n00 + n01 + n10 + n11)

    def ll(n0, n1, q):
        return xlogy(n0, 1 - q) + xlogy(n1, q)

    lr = -2 * (ll(n00 + n10, n01 + n11, p) - ll(n00, n01, p01) - ll(n10, n11, p11))
    lr = max(lr, 0.0)
    return lr, 1 - chi2.cdf(lr, 1)


def backtest(f, alpha):
    r = f["ret"]
    risk = var_es(f, alpha)
    viol = (r < -risk["VaR"]).astype(int)
    n, x = len(viol), int(viol.sum())

    lr_uc, p_uc = kupiec(viol, alpha)
    lr_ind, p_ind = christoffersen(viol)
    p_cc = 1 - chi2.cdf(lr_uc + lr_ind, 2)

    hit = viol.astype(bool)
    es_ratio = float((-r[hit] / risk["ES"][hit]).mean()) if x else np.nan

    return {
        "n_obs": n,
        "violations": x,
        "expected": round(alpha * n, 1),
        "rate_%": 100 * x / n,
        "Kupiec p": p_uc,
        "Christoffersen p": p_ind,
        "Cond. coverage p": p_cc,
        "ES ratio": es_ratio,
        "avg VaR_%": float(risk["VaR"].mean()),
    }


def backtest_table(asset, alphas=(0.05, 0.01), start=None, end=None, models=MODELS):
    rows = []
    for alpha in alphas:
        for m in models:
            f = load_forecasts(asset, m).loc[start:end]
            rows.append({"asset": asset, "alpha": alpha, "model": m, **backtest(f, alpha)})
    return pd.DataFrame(rows)


def backtest_all(alphas=(0.05, 0.01), start=None, end=None, assets=list(ASSETS)):
    return pd.concat(
        [backtest_table(a, alphas, start, end) for a in assets], ignore_index=True
    )


if __name__ == "__main__":
    RESULTS_DIR.mkdir(exist_ok=True)
    out = backtest_all()
    out.round(4).to_csv(RESULTS_DIR / "backtest_all.csv", index=False)
    for (asset, alpha), g in out.groupby(["asset", "alpha"], sort=False):
        print(f"\n=== {asset}  VaR {int((1 - alpha) * 100)}% ===")
        print(g.drop(columns=["asset", "alpha", "avg VaR_%"]).round(4).to_string(index=False))
        