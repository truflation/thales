"""ComposedForecaster — wraps per-component models + CBDF composer as a
single Forecaster (Protocol-compliant) usable in the harness.

The integration layer between Phase 1 (per-component archetypes) and
Phase 2.1 (CBDF composition). Once you have N per-component
Forecasters and a (CBDF)Composer, this class makes the composed
headline forecast walk-forward-able through `walk_forward → score`.

Each per-component forecaster is itself a Forecaster (e.g.
PersistenceBaseline, BSTS, commodity TVP, …). The composer aggregates
their per-component forecasts into a headline Forecast.

Usage:

    composer = CBDFComposer(weights={"r1": 0.4, "r2": 0.6})
    forecasters = {
        "r1": PersistenceBaseline(target_col="r1"),
        "r2": PersistenceBaseline(target_col="r2"),
    }
    composed = ComposedForecaster(
        components=forecasters, composer=composer, model_id="composed_v1")

    forecasts = walk_forward(composed, panel, target_col="headline",
                                origins=..., horizon=1)
    df = attach_actuals(forecasts, panel["headline"])
    block = score(df)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from thales.evaluation.harness import Forecast, Forecaster
from thales.models.composition.weighted import WeightedComposer


@dataclass
class ComposedForecaster:
    """Forecaster that composes per-component sub-forecasters via a
    weighted/CBDF composer.

    ``components``: mapping component_id → Forecaster. Each sub-forecaster
    is fit-and-predicted independently against its own target column;
    ``components[k]`` should know which column of ``panel`` to read (via
    its own ``target_col`` attribute, e.g. ``PersistenceBaseline(target_col=k)``).

    ``composer``: a WeightedComposer or CBDFComposer with weights matching
    the keys of ``components``.

    ``model_id``: free-form identifier; used as the primary key in the
    scoring DB.
    """
    components: dict[str, Forecaster]
    composer: WeightedComposer
    model_id: str = "composed_v1"

    def fit_predict(self, panel: pd.DataFrame,
                     origin: pd.Timestamp,
                     target: pd.Timestamp) -> Forecast:
        """Run each sub-forecaster, then compose into headline."""
        sub_forecasts: dict[str, Forecast] = {}
        for cid, fcaster in self.components.items():
            sub_forecasts[cid] = fcaster.fit_predict(panel, origin, target)
        return self.composer.compose(sub_forecasts, origin, target)
