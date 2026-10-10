import numpy as np
import pandas as pd
from arch import arch_model

MODELS = ["HIST", "EWMA", "GARCH", "GJR", "EGARCH"]

GARCH_SPECS = {
    "GARCH": dict(vol="GARCH", p=1, o=0, q=1),
    "GJR": dict(vol="GARCH", p=1, o=1, q=1),
    "EGARCH": dict(vol="EGARCH", p=1, o=1, q=1),
}


def hist_variance(window, n=250):
    """Variance of the last n returns."""
    return float(window.iloc[-n:].var(ddof=1))


def ewma_variance(window, lam=0.94):
    """RiskMetrics EWMA: s2_t = lam * s2_{t-1} + (1 - lam) * r_{t-1}^2.
    Runs through the window and returns the forecast for the next day."""
    var = float(window.var(ddof=1))
    for x in window.values:
        var = lam * var + (1 - lam) * x**2
    return var


def fit_garch(window, kind="GARCH", dist="t"):
    """Fit a GARCH-family model with constant mean on a window of returns (in %)."""
    am = arch_model(window, mean="Constant", dist=dist, **GARCH_SPECS[kind])
    return am.fit(disp="off", show_warning=False)


def forecast_one_step(window, model):
    """One-step-ahead forecast for any model in MODELS.

    Returns a dict: variance (in %^2), mu (mean in %), nu (t degrees of freedom,
    None means use the normal distribution for VaR).
    """
    if model == "HIST":
        return {"variance": hist_variance(window), "mu": 0.0, "nu": None}
    if model == "EWMA":
        return {"variance": ewma_variance(window), "mu": 0.0, "nu": None}

    res = fit_garch(window, kind=model)
    var = float(res.forecast(horizon=1).variance.iloc[-1, 0])
    return {"variance": var, "mu": float(res.params["mu"]), "nu": float(res.params["nu"])}


def persistence(res):
    """Persistence of a fitted GARCH or GJR model (close to 1 = shocks fade slowly)."""
    p = res.params
    gamma = p.get("gamma[1]", 0.0)
    if "omega" in p and "alpha[1]" in p and "beta[1]" in p:
        if res.model.volatility.__class__.__name__ == "EGARCH":
            return float(p["beta[1]"])
        return float(p["alpha[1]"] + 0.5 * gamma + p["beta[1]"])
    return np.nan


if __name__ == "__main__":
    from src.data import load_returns

    r = load_returns("SP500")
    window = r.iloc[-1000:]
    rows = []
    for m in MODELS:
        f = forecast_one_step(window, m)
        rows.append({
            "model": m,
            "next-day vol (%)": round(np.sqrt(f["variance"]), 3),
            "nu": None if f["nu"] is None else round(f["nu"], 2),
        })
    print(pd.DataFrame(rows).to_string(index=False))