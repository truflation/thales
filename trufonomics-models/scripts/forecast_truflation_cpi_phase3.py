"""Truflation US CPI YoY — Phase 3 (Almosova LSTM, component-level).

Trained on the 12 top-level Truflation CPI per-component daily streams.
Forecasts each component's cumulative log-return at h ∈ {1, 7, 14, 30,
90} days; composes via M2 to headline YoY; anchor-corrected to actual
at origin (matches Phase 1 spec exactly so RMSE is apples-to-apples).

Train-once + walk-forward inference. Training data is windows ending
**before 2018-01-01** (the walk-forward start) — strict no-peek-ahead.

Density: MC-dropout (Gal & Ghahramani 2016) — keep dropout active at
inference, run multiple passes, sample Gaussian per pass. Per
(component, horizon, origin) we collect ~200 samples; compose to
headline samples; quantile bands.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from thales.models.almosova_lstm import (    # noqa: E402
    HORIZONS_DEFAULT,
    AlmosovaLSTM,
    TrainConfig,
    gaussian_nll,
    mc_predict,
)
from forecast_truflation_cpi_bottomup import (    # noqa: E402
    VINTAGE_DB,
    load_component_levels,
    load_truflation_headline_yoy,
)

OUT_DIR = ROOT / "results" / "truflation_cpi_forecast"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = OUT_DIR / "phase3_almosova.pt"

WALK_FORWARD_START = "2018-01-01"
WINDOW = 90    # days of context
HORIZONS = HORIZONS_DEFAULT


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ─── Build supervised tensors from the panel ─────────────────────────────


def build_training_tensors(panel: pd.DataFrame,
                              window: int = WINDOW,
                              horizons: list[int] = HORIZONS,
                              cutoff: str = WALK_FORWARD_START,
                              ) -> tuple[torch.Tensor, torch.Tensor,
                                          torch.Tensor, dict[int, int],
                                          np.ndarray]:
    """Build (X, c, y) training tensors plus the (component_id → idx) lookup
    and per-horizon target standardization SD.

    For each (component, t) where the entire window [t-window+1..t] lies
    before cutoff AND all horizon targets [t+h] also lie before cutoff:
      X[i] = log_return panel for component over the window
      c[i] = component index
      y[i, hi] = cumulative log_return from t to t+h, normalized by
                 per-horizon training-set SD.
    """
    cutoff_ts = pd.Timestamp(cutoff)
    max_h = max(horizons)
    components = panel.columns.tolist()
    cid_to_idx = {int(c): i for i, c in enumerate(components)}

    log_levels = np.log(panel.values)    # (T, C)
    log_returns = np.diff(log_levels, axis=0)    # (T-1, C)
    # log_returns[i] = log(panel[i+1]) - log(panel[i])
    # so log_returns[i] is the return ON day panel.index[i+1].

    # Index alignment: panel.index has T entries; log_returns has T-1.
    # We treat log_returns[i] as the return at panel.index[i+1].
    return_index = panel.index[1:]
    cutoff_pos = int(np.searchsorted(return_index, cutoff_ts, side="left"))

    X_list, c_list, y_list = [], [], []
    for ci, col in enumerate(components):
        col_returns = log_returns[:, ci]
        # Loop over every t such that:
        #   - last input return is at index [t]    → window [t-window+1, t]
        #   - last target lookahead is at index [t + max_h]
        #   - cutoff: t + max_h < cutoff_pos
        t_start = window - 1
        t_end = cutoff_pos - max_h
        for t in range(t_start, t_end):
            X_list.append(col_returns[t - window + 1: t + 1])    # (window,)
            c_list.append(ci)
            # target: cumulative log-return from end of window (t) to t+h
            # = sum of col_returns[t+1 : t+h+1]
            target = np.array([
                float(col_returns[t + 1: t + h + 1].sum())
                for h in horizons
            ], dtype=np.float32)
            y_list.append(target)

    X = np.stack(X_list, axis=0).astype(np.float32)
    c = np.array(c_list, dtype=np.int64)
    y = np.stack(y_list, axis=0).astype(np.float32)

    # Standardize y per horizon by training-set SD (no shift — keep mean as-is)
    sd = y.std(axis=0).astype(np.float32)
    sd = np.where(sd > 1e-8, sd, 1.0)
    y_norm = y / sd

    return (torch.from_numpy(X),
              torch.from_numpy(c),
              torch.from_numpy(y_norm),
              cid_to_idx,
              sd)


# ─── Training ─────────────────────────────────────────────────────────────


def train(model: AlmosovaLSTM,
            X: torch.Tensor, c: torch.Tensor, y: torch.Tensor,
            cfg: TrainConfig,
            device: torch.device,
            ) -> dict:
    """Train with NLL loss + early stopping on a held-out validation split.

    Validation split is the *last* val_frac fraction of the supervised
    samples (a temporal holdout, since training samples are roughly in
    time-then-component order).
    """
    n = len(X)
    n_val = max(int(cfg.val_frac * n), 1)
    n_train = n - n_val
    # Temporal-tail validation split
    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(n_train)    # shuffle within train block
    train_idx = perm
    val_idx = np.arange(n_train, n)

    X_tr, c_tr, y_tr = X[train_idx].to(device), c[train_idx].to(device), y[train_idx].to(device)
    X_va, c_va, y_va = X[val_idx].to(device), c[val_idx].to(device), y[val_idx].to(device)

    opt = optim.Adam(model.parameters(), lr=cfg.lr,
                       weight_decay=cfg.weight_decay)
    best_val = float("inf")
    bad_epochs = 0
    history = {"epoch": [], "train_nll": [], "val_nll": []}

    for epoch in range(1, cfg.n_epochs + 1):
        model.train()
        idx = torch.randperm(len(X_tr), device=device)
        running = 0.0
        seen = 0
        for start in range(0, len(idx), cfg.batch_size):
            batch = idx[start: start + cfg.batch_size]
            xb, cb, yb = X_tr[batch], c_tr[batch], y_tr[batch]
            opt.zero_grad()
            pred = model(xb, cb)
            loss = gaussian_nll(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += float(loss.item()) * len(xb)
            seen += len(xb)
        train_nll = running / seen

        # Validation
        model.eval()
        with torch.no_grad():
            pred_va = model(X_va, c_va)
            val_nll = float(gaussian_nll(pred_va, y_va).item())
        history["epoch"].append(epoch)
        history["train_nll"].append(train_nll)
        history["val_nll"].append(val_nll)
        print(f"  epoch {epoch:>3d}  train_nll={train_nll:.4f}  "
                f"val_nll={val_nll:.4f}", flush=True)
        if val_nll < best_val - 1e-4:
            best_val = val_nll
            bad_epochs = 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                print(f"  early stop at epoch {epoch} (no improve for "
                        f"{cfg.patience} epochs; best val={best_val:.4f})")
                break

    # Restore best
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    history["best_val_nll"] = best_val
    return history


# ─── Walk-forward inference ───────────────────────────────────────────────


def walk_forward_inference(model: AlmosovaLSTM,
                              panel: pd.DataFrame,
                              actual_yoy: pd.Series,
                              cid_to_idx: dict[int, int],
                              target_sd: np.ndarray,
                              weights_pct: dict[int, float],
                              start_date: str = WALK_FORWARD_START,
                              step_days: int = 30,
                              n_passes: int = 50,
                              samples_per_pass: int = 4,
                              window: int = WINDOW,
                              horizons: list[int] = HORIZONS,
                              device: torch.device = None,
                              ) -> pd.DataFrame:
    """Walk-forward inference at the same 102-origin schedule as Phase 1.

    For each origin: take the last `window` log-returns per component →
    MC-dropout sample paths → compose via M2 → anchor offset → bands.
    """
    if device is None:
        device = next(model.parameters()).device
    base_date = panel.index.min()
    end_date = panel.index.max()
    origins = pd.date_range(start_date, end_date, freq=f"{step_days}D")
    log_levels = np.log(panel.values)
    log_returns = np.diff(log_levels, axis=0)
    return_index = panel.index[1:]

    # Per-origin loop
    rows = []
    n_samples = n_passes * samples_per_pass
    for origin in origins:
        if origin not in panel.index:
            continue
        history = panel.loc[panel.index <= origin]
        if len(history) < window + 365 + 30:
            continue
        # Position of `origin` in return_index (the day whose log-return
        # is log(panel[origin]) - log(panel[origin-1]))
        pos = int(np.searchsorted(return_index, origin, side="left"))
        if pos < window:
            continue
        # Window: log_returns[pos - window + 1 : pos + 1] is shape (window, C)
        win = log_returns[pos - window + 1: pos + 1, :]    # (window, C)
        if win.shape[0] != window:
            continue

        # Build batch: one row per component
        components = panel.columns.tolist()
        Xb = np.stack([win[:, ci] for ci in range(len(components))],
                        axis=0).astype(np.float32)    # (C, window)
        cb = np.array([cid_to_idx[int(c)] for c in components], dtype=np.int64)
        Xt = torch.from_numpy(Xb).to(device)
        ct = torch.from_numpy(cb).to(device)

        # MC-dropout samples: (C, H, S) of normalized cumulative log-returns
        samples_norm = mc_predict(model, Xt, ct,
                                       n_passes=n_passes,
                                       samples_per_pass=samples_per_pass,
                                       seed=int(origin.value % 1_000_000))
        # De-normalize per horizon
        samples = samples_norm * target_sd[None, :, None]    # (C, H, S)

        # Convert to per-component level samples
        last_levels = panel.loc[origin].values    # (C,)
        # Per-component projected level at horizon h, sample s:
        #   level[c, h, s] = last_level[c] * exp(samples[c, h, s])
        comp_levels = (last_levels[:, None, None]
                          * np.exp(samples))    # (C, H, S)

        # Composition: weighted sum of rebased per-component levels
        base_levels = panel.loc[base_date].values    # (C,)
        weights_arr = np.array([weights_pct[int(components[ci])]
                                  for ci in range(len(components))])    # (C,)

        # composed_target_samples[h, s] = sum_c w_c * (level[c,h,s] / base[c]) / 100 * 100 / 100
        # = (1/100) * sum_c w_c * (level[c,h,s] / base[c]) * 100
        composed_target = (
            (comp_levels / base_levels[:, None, None]) * weights_arr[:, None, None]
        ).sum(axis=0) / 100.0    # (H, S)

        # Composed denom level at target_date - 365d (using actual panel)
        for hi, h in enumerate(horizons):
            target = origin + pd.Timedelta(days=h)
            denom_date = target - pd.Timedelta(days=365)
            if denom_date < panel.index.min():
                continue
            if denom_date in panel.index:
                denom_levels = panel.loc[denom_date].values
            else:
                avail = panel.index[panel.index <= denom_date]
                if len(avail) == 0:
                    continue
                denom_levels = panel.loc[avail[-1]].values
            composed_denom = (
                (denom_levels / base_levels) * weights_arr
            ).sum() / 100.0

            # Anchor offset (Phase 1 mechanism): use composed YoY at origin
            origin_levels = panel.loc[origin].values
            denom_origin_date = origin - pd.Timedelta(days=365)
            if denom_origin_date in panel.index:
                denom_origin_levels = panel.loc[denom_origin_date].values
            else:
                avail = panel.index[panel.index <= denom_origin_date]
                if len(avail) == 0:
                    continue
                denom_origin_levels = panel.loc[avail[-1]].values
            composed_origin = ((origin_levels / base_levels) * weights_arr).sum() / 100.0
            composed_origin_denom = (
                (denom_origin_levels / base_levels) * weights_arr).sum() / 100.0
            composed_yoy_at_origin = (composed_origin / composed_origin_denom - 1.0) * 100.0
            anchor_yoy = (float(actual_yoy.loc[origin])
                            if origin in actual_yoy.index else None)
            anchor_offset = ((anchor_yoy - composed_yoy_at_origin)
                              if anchor_yoy is not None else 0.0)

            yoy_samples = (composed_target[hi, :] / composed_denom - 1.0) * 100.0
            yoy_samples = yoy_samples + anchor_offset
            point = float(np.median(yoy_samples))
            lo80 = float(np.quantile(yoy_samples, 0.10))
            hi80 = float(np.quantile(yoy_samples, 0.90))
            lo95 = float(np.quantile(yoy_samples, 0.025))
            hi95 = float(np.quantile(yoy_samples, 0.975))
            actual = (float(actual_yoy.loc[target])
                      if target in actual_yoy.index else None)
            err = (point - actual) if actual is not None else None
            in_80 = (lo80 <= actual <= hi80) if actual is not None else None
            in_95 = (lo95 <= actual <= hi95) if actual is not None else None
            rows.append({
                "origin": origin,
                "horizon_days": h,
                "target_date": target,
                "point": point,
                "lo80": lo80, "hi80": hi80,
                "lo95": lo95, "hi95": hi95,
                "actual": actual,
                "error_pp": err,
                "in_80": in_80,
                "in_95": in_95,
                "width80_pp": hi80 - lo80,
                "width95_pp": hi95 - lo95,
            })
    return pd.DataFrame(rows)


# ─── Main ─────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", action="store_true",
                        help="force retrain even if checkpoint exists")
    parser.add_argument("--skip-eval", action="store_true",
                        help="skip walk-forward inference (training only)")
    parser.add_argument("--n-epochs", type=int, default=50)
    parser.add_argument("--n-passes", type=int, default=50)
    parser.add_argument("--samples-per-pass", type=int, default=4)
    args = parser.parse_args()

    print("=" * 78)
    print("Truflation US CPI YoY — Phase 3 (Almosova LSTM)")
    print("=" * 78)

    device = _device()
    print(f"\nDevice: {device}")

    # Load data
    print("\nLoading 12 top-level CPI component levels…")
    con = duckdb.connect(str(VINTAGE_DB), read_only=True)
    panel, weights_pct = load_component_levels(con, crosswalk_level="top12")
    con.close()
    print(f"  Panel: {len(panel)} days × {panel.shape[1]} components")
    actual_yoy = load_truflation_headline_yoy()
    print(f"  Actual YoY: n={len(actual_yoy)}")

    # Build training tensors
    print(f"\nBuilding training tensors (window={WINDOW}, "
            f"cutoff={WALK_FORWARD_START}, horizons={HORIZONS})…")
    X, c, y, cid_to_idx, target_sd = build_training_tensors(
        panel, window=WINDOW, horizons=HORIZONS,
        cutoff=WALK_FORWARD_START)
    print(f"  X={tuple(X.shape)}  c={tuple(c.shape)}  y={tuple(y.shape)}")
    print(f"  Per-horizon target SD: "
            f"{dict(zip(HORIZONS, [round(float(s), 5) for s in target_sd]))}")

    cfg = TrainConfig(n_epochs=args.n_epochs)
    model = AlmosovaLSTM(n_components=panel.shape[1],
                              n_horizons=len(HORIZONS),
                              cfg=cfg).to(device)

    if MODEL_PATH.exists() and not args.retrain:
        print(f"\nLoading existing checkpoint: {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else:
        print(f"\nTraining (epochs={cfg.n_epochs}, batch={cfg.batch_size}, "
                f"lr={cfg.lr})…")
        t0 = time.monotonic()
        history = train(model, X, c, y, cfg, device)
        dt = time.monotonic() - t0
        print(f"  train time: {dt:.1f}s  best val_nll={history['best_val_nll']:.4f}")

    if args.skip_eval:
        print("\n--skip-eval set; exiting after training.")
        return

    # Walk-forward
    print("\nWalk-forward inference at 30d-step origins from "
            f"{WALK_FORWARD_START}…")
    t0 = time.monotonic()
    df = walk_forward_inference(
        model, panel, actual_yoy, cid_to_idx, target_sd, weights_pct,
        n_passes=args.n_passes, samples_per_pass=args.samples_per_pass,
        device=device,
    )
    dt = time.monotonic() - t0
    print(f"  inference time: {dt:.1f}s   {len(df)} forecast points "
            f"across {df['origin'].nunique()} origins")

    if len(df):
        print("\nWalk-forward summary by horizon:")
        scored = df.dropna(subset=["actual"])
        agg = scored.groupby("horizon_days").agg(
            n=("actual", "count"),
            rmse=("error_pp", lambda x: float(np.sqrt(np.mean(x ** 2)))),
            mae=("error_pp", lambda x: float(np.mean(np.abs(x)))),
            mean_err=("error_pp", "mean"),
            cov_80=("in_80", "mean"),
            cov_95=("in_95", "mean"),
            width80=("width80_pp", "mean"),
            width95=("width95_pp", "mean"),
        ).reset_index()
        print(agg.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        out_csv = OUT_DIR / "walk_forward_summary_phase3.csv"
        df.to_csv(out_csv, index=False)
        agg_csv = OUT_DIR / "walk_forward_aggregate_phase3.csv"
        agg.to_csv(agg_csv, index=False)
        print(f"\nSaved: {out_csv}")
        print(f"Saved: {agg_csv}")


if __name__ == "__main__":
    main()
