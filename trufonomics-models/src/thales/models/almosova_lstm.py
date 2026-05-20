"""Almosova-style LSTM forecaster — Phase 3.

Shared LSTM encoder + component embedding + multi-horizon Gaussian head.
Trained on Truflation per-component log-returns; predicts cumulative
log-return at h ∈ {1, 7, 14, 30, 90} days for the queried component.

Density via MC-dropout (Gal & Ghahramani 2016): keep dropout active at
inference, run S forward passes, draw a Gaussian sample per (pass,
horizon) → S × samples_per_pass paths per component.

Reference: Almosova & Andresen (2020/2023), "Inflation Forecasting
Using LSTMs" — multi-step LSTM with shared encoder over CPI components.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


HORIZONS_DEFAULT = [1, 7, 14, 30, 90]


@dataclass
class TrainConfig:
    window: int = 90
    hidden: int = 128
    embed_dim: int = 16
    n_layers: int = 2
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 256
    n_epochs: int = 50
    patience: int = 8
    val_frac: float = 0.10
    seed: int = 0


class AlmosovaLSTM(nn.Module):
    """Shared LSTM encoder over per-component log-returns.

    Inputs per (sample, component):
      x : (B, T) log-returns of that component over T days
      c : (B,)   integer component id ∈ [0, n_components)

    Output: (B, H, 2) — for each of H horizons, (mu, log_sigma) of the
    cumulative log-return target.
    """

    def __init__(self, n_components: int, n_horizons: int,
                  cfg: TrainConfig = TrainConfig()):
        super().__init__()
        self.cfg = cfg
        self.n_components = n_components
        self.n_horizons = n_horizons
        self.embed = nn.Embedding(n_components, cfg.embed_dim)
        self.lstm = nn.LSTM(
            input_size=1 + cfg.embed_dim,
            hidden_size=cfg.hidden,
            num_layers=cfg.n_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.n_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(cfg.hidden, 2 * n_horizons)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # x: (B, T)  c: (B,)
        B, T = x.shape
        emb = self.embed(c).unsqueeze(1).expand(B, T, -1)    # (B, T, E)
        inp = torch.cat([x.unsqueeze(-1), emb], dim=-1)       # (B, T, 1+E)
        out, _ = self.lstm(inp)                              # (B, T, H)
        last = self.dropout(out[:, -1, :])                   # (B, H)
        head = self.head(last)                               # (B, 2H)
        head = head.view(B, self.n_horizons, 2)              # (B, H, 2)
        return head    # (mu, log_sigma) at last index 0,1


def gaussian_nll(pred: torch.Tensor, target: torch.Tensor,
                  reduction: str = "mean") -> torch.Tensor:
    """Negative log-likelihood under N(mu, sigma=exp(log_sigma)).

    pred:   (B, H, 2)   target: (B, H)    NLL averaged over (B, H).
    """
    mu = pred[..., 0]
    log_sigma = pred[..., 1].clamp(-7, 5)    # avoid extreme sigmas
    sigma = log_sigma.exp()
    nll = 0.5 * (np.log(2 * np.pi)
                  + 2 * log_sigma
                  + ((target - mu) / sigma) ** 2)
    if reduction == "mean":
        return nll.mean()
    if reduction == "sum":
        return nll.sum()
    return nll


def mc_predict(model: AlmosovaLSTM,
                x: torch.Tensor, c: torch.Tensor,
                n_passes: int = 50,
                samples_per_pass: int = 4,
                seed: int = 0,
                ) -> np.ndarray:
    """MC-dropout prediction: keep dropout active, run multiple passes,
    sample Gaussian per pass.

    Returns array of shape (B, H, n_passes * samples_per_pass) — raw
    samples of cumulative log-return at each horizon.
    """
    rng = np.random.default_rng(seed)
    model.train()    # keep dropout ON for MC sampling
    B, T = x.shape
    H = model.n_horizons
    out = np.empty((B, H, n_passes * samples_per_pass))
    with torch.no_grad():
        for p in range(n_passes):
            pred = model(x, c).cpu().numpy()    # (B, H, 2)
            mu = pred[..., 0]
            sigma = np.exp(np.clip(pred[..., 1], -7, 5))
            for s in range(samples_per_pass):
                noise = rng.standard_normal((B, H))
                idx = p * samples_per_pass + s
                out[:, :, idx] = mu + sigma * noise
    model.eval()
    return out
