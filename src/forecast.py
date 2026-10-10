import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from arch import arch_model

from src.data import ASSETS, load_returns
from src.models import GARCH_SPECS, MODELS, fit_garch, forecast_one_step

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "forecasts"


def rolling_forecasts(r, model, window=1000, refit_every=5):
    """One-step-ahead forecasts for every day from index `window` onward.

    The forecast stored at date t only uses returns strictly before t.
    """
    dates, rows = [], []
    params = None

    for i, t in enumerate(range(window, len(r))):
        w = r.iloc[t - window:t]

        if model in GARCH_SPECS:
            if params is None or i % refit_every == 0:
                try:
                    params = fit_garch(w, model).params
                except Exception:
                    if params is None:
                        raise  # first fit failed, nothing to fall back on
            am = arch_model(w, mean="Constant", dist="t", **GARCH_SPECS[model])
            res = am.fix(params)
            var = float(res.forecast(horizon=1).variance.iloc[-1, 0])
            f = {"variance": var, "mu": float(params["mu"]), "nu": float(params["nu"])}
        else:
            f = forecast_one_step(w, model)

        dates.append(r.index[t])
        rows.append({
            "ret": float(r.iloc[t]),
            "variance": f["variance"],
            "mu": f["mu"],
            "nu": np.nan if f["nu"] is None else f["nu"],
        })

    return pd.DataFrame(rows, index=pd.DatetimeIndex(dates, name="date"))


def run_all(assets, models, window=1000, refit_every=5, force=False):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for asset in assets:
        r = load_returns(asset)
        for model in models:
            path = RESULTS_DIR / f"{asset}_{model}.csv"
            if path.exists() and not force:
                print(f"skip {asset} {model} (already done)")
                continue
            t0 = time.time()
            df = rolling_forecasts(r, model, window, refit_every)
            df.to_csv(path)
            print(f"done {asset:6s} {model:7s} {len(df)} forecasts in {time.time() - t0:.0f}s")


def load_forecasts(asset, model):
    path = RESULTS_DIR / f"{asset}_{model}.csv"
    return pd.read_csv(path, index_col=0, parse_dates=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", nargs="+", default=list(ASSETS))
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--window", type=int, default=1000)
    ap.add_argument("--refit-every", type=int, default=5)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    run_all(a.assets, a.models, a.window, a.refit_every, a.force)