"""Conditional-on-shock cross-input forecast test.

Operator-relevant question: when one input has a known unusually large
move (>1.5 SD), does the BVAR — which knows about cross-input
covariance — forecast the OTHER inputs better than naive AR(1) which
treats them independently?

Honest finding: NO. BVAR loses by 4-16% across both verticals.

This test belongs in the eval suite because it documents the empirical
ceiling on the cross-input transmission story. The right operator-
facing value is the scenario engine + FEVD, not point forecast accuracy.

Run::

    uv run python truflation-operate/verticals/conditional_shock_test.py
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from thales.models.archetypes.bvar_minnesota import (    # noqa: E402
    _ar_matrices,
    fit_bvar_minnesota,
)

OUT_DIR = ROOT / "truflation-operate" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def run_test(panel: pd.DataFrame, label: str,
                  train_min: int = 60,
                  sd_threshold: float = 1.5) -> pd.DataFrame:
    var_cols = list(panel.columns)
    Y = panel.values
    k = Y.shape[1]
    R = np.diff(Y, axis=0)
    R_sd = R.std(axis=0)
    n_t = len(R) - 1

    bvar_errs: dict[str, list[float]] = {c: [] for c in var_cols}
    naive_errs: dict[str, list[float]] = {c: [] for c in var_cols}
    n_origins = 0
    for j in range(k):
        for t in range(train_min, n_t):
            if abs(R[t, j]) < sd_threshold * R_sd[j]:
                continue
            n_origins += 1
            Y_train = Y[: t + 1]
            fit = fit_bvar_minnesota(Y_train, p=1)
            A_list = _ar_matrices(fit.coefs, fit.k, fit.p)
            intercept = fit.coefs[:, 0]
            y_pred = intercept + A_list[0] @ Y_train[-1]
            for i in range(k):
                if i == j:
                    continue
                bvar_errs[var_cols[i]].append(float(Y[t + 1, i] - y_pred[i]))
                x = Y_train[:, i]
                d = np.diff(x)
                if len(d) < 4:
                    continue
                a, b = d[:-1], d[1:]
                X = np.column_stack([np.ones_like(a), a])
                coef, *_ = np.linalg.lstsq(X, b, rcond=None)
                alpha, phi = float(coef[0]), float(coef[1])
                naive_pred = x[-1] + alpha + phi * d[-1]
                naive_errs[var_cols[i]].append(float(Y[t + 1, i] - naive_pred))

    rows = []
    for c in var_cols:
        b_arr = np.array(bvar_errs[c])
        n_arr = np.array(naive_errs[c])
        if len(b_arr) == 0:
            continue
        b_rmse = float(np.sqrt((b_arr ** 2).mean()))
        n_rmse = float(np.sqrt((n_arr ** 2).mean()))
        red = (1 - b_rmse / n_rmse) * 100 if n_rmse > 0 else float("nan")
        rows.append({
            "target":      c,
            "n":           int(len(b_arr)),
            "bvar_rmse":   b_rmse,
            "naive_rmse":  n_rmse,
            "red_pct":     red,
        })
    df = pd.DataFrame(rows)
    return df, n_origins


def main() -> None:
    print("=" * 78)
    print("Conditional-on-shock cross-input forecast test")
    print("=" * 78)
    print("\nTest: at each origin where some input had |Δlog|>1.5 SD, predict")
    print("the OTHER inputs one step ahead with BVAR (knows cross-input)")
    print("vs naive AR(1) per input (no cross-effects).")

    for label, path in [
        ("paris_auto_importer",
         ROOT / "truflation-operate" / "verticals" / "import_export_auto.py"),
        ("us_textile_importer",
         ROOT / "truflation-operate" / "verticals" / "import_export_textile.py"),
    ]:
        mod = _load(label, path)
        panel = mod.load_panel()
        df, n_origins = run_test(panel, label)
        df = df.round(5)
        print()
        print(f"── {label} ── (qualifying origins: {n_origins}) ──")
        print(df.to_string(index=False))
        df.to_csv(OUT_DIR / f"conditional_shock_test_{label}_{date.today()}.csv",
                       index=False)


if __name__ == "__main__":
    main()
