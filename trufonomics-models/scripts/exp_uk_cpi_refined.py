"""UK CPI May-2026 — refined nowcast: seasonal backbone + observed DESNZ fuel + Ofgem-flat energy.

Plugs the two observable/known blocks into the seasonal/base-effect baseline:
  - MOTOR FUEL: DESNZ weekly pump prices give the May-vs-Apr move directly (CPI samples ~mid-month),
    so we replace the seasonal fuel guess with the OBSERVED fuel MoM.
  - HOUSEHOLD ENERGY: the Ofgem cap is unchanged in May (Q2 cap effective 1 Apr, runs to 30 Jun), so
    household electricity+gas MoM ≈ 0 (known), replacing the seasonal guess.
Everything else stays on the seasonal/base-effect backbone. Headline NSA YoY off ONS D7BT.

Run::  uv run python -W ignore scripts/exp_uk_cpi_refined.py
"""
from __future__ import annotations
import json, urllib.request, re
import numpy as np, pandas as pd

W_FUEL, W_ENERGY = 0.028, 0.035   # approx CPI weights: fuels&lubricants ~2.8%, household energy ~3.5%
PETROL_SHARE = 0.60               # petrol vs diesel split within CPI fuel (approx)


def get(u):
    req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0 (research)"})
    return urllib.request.urlopen(req, timeout=30).read()


def ons_index(cdid):
    d = json.loads(get(f"https://www.ons.gov.uk/economy/inflationandpriceindices/timeseries/{cdid}/mm23/data"))
    MON = {m: i for i, m in enumerate(["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"], 1)}
    out = {}
    for x in d.get("months", []):
        y, mm = x["date"].split(); out[pd.Timestamp(int(y), MON[mm], 1) + pd.offsets.MonthEnd(0)] = float(x["value"])
    return pd.Series(out).sort_index()


def seasonal_may(idx, k=5):
    mom = (idx / idx.shift(1) - 1) * 100
    vals = [mom.get(pd.Timestamp(2026 - j, 5, 1) + pd.offsets.MonthEnd(0), np.nan) for j in range(1, k + 1)]
    return float(np.nanmean([v for v in vals if np.isfinite(v)]))


def main():
    allidx = ons_index("D7BT"); fuel = ons_index("D7EC"); energy = ons_index("D7CH")
    apr = allidx[pd.Timestamp("2026-04-30")]; may25 = allidx[pd.Timestamp("2025-05-31")]
    apr_yoy = (apr / allidx[pd.Timestamp("2025-04-30")] - 1) * 100

    # ---- DESNZ weekly pump prices -> observed May vs Apr 2026 MoM ----
    html = get("https://www.gov.uk/government/statistics/weekly-road-fuel-prices").decode("utf-8", "ignore")
    csv_url = re.findall(r'https://assets\.publishing\.service\.gov\.uk/[^"\s]+weekly_road_fuel_prices_\d+\.csv', html)[0]
    rows = get(csv_url).decode("utf-8", "ignore").splitlines()
    recs = []
    for ln in rows[1:]:
        p = ln.split(",")
        try:
            dt = pd.to_datetime(p[0], format="%d/%m/%Y"); recs.append((dt, float(p[1]), float(p[2])))
        except Exception: continue
    fp = pd.DataFrame(recs, columns=["d", "petrol", "diesel"]).set_index("d").sort_index()
    m = fp.resample("ME").mean()
    aprp, aprd = m.loc["2026-04", "petrol"].iloc[0], m.loc["2026-04", "diesel"].iloc[0]
    mayp, mayd = m.loc["2026-05", "petrol"].iloc[0], m.loc["2026-05", "diesel"].iloc[0]
    petrol_mom = (mayp / aprp - 1) * 100; diesel_mom = (mayd / aprd - 1) * 100
    obs_fuel_mom = PETROL_SHARE * petrol_mom + (1 - PETROL_SHARE) * diesel_mom

    # ---- assemble ----
    seas_all = seasonal_may(allidx); seas_fuel = seasonal_may(fuel); seas_energy = seasonal_may(energy)
    fuel_delta = W_FUEL * (obs_fuel_mom - seas_fuel)
    energy_delta = W_ENERGY * (0.0 - seas_energy)          # Ofgem cap flat in May -> known ~0
    refined_mom = seas_all + fuel_delta + energy_delta

    def yoy(mom): return (apr * (1 + mom / 100) / may25 - 1) * 100

    print("=" * 72)
    print("UK CPI May-2026 (headline NSA YoY) — refined nowcast")
    print("=" * 72)
    print(f"  Apr-2026: index {apr:.1f}, YoY {apr_yoy:.2f}%   (base: May-2025 index {may25:.1f})")
    print(f"\n  DESNZ pump prices (monthly avg, p/l): Apr→May  petrol {aprp:.1f}→{mayp:.1f} ({petrol_mom:+.1f}%), "
          f"diesel {aprd:.1f}→{mayd:.1f} ({diesel_mom:+.1f}%)")
    print(f"    observed fuel MoM (blend) = {obs_fuel_mom:+.2f}%   vs seasonal-May guess {seas_fuel:+.2f}%")
    print(f"    -> fuel override on all-items: {fuel_delta:+.3f}pp")
    print(f"  Energy (Ofgem cap unchanged in May): known MoM ~0.0%  vs seasonal {seas_energy:+.2f}%")
    print(f"    -> energy override on all-items: {energy_delta:+.3f}pp")
    print(f"\n  seasonal backbone all-items MoM = {seas_all:+.2f}%")
    print(f"  refined all-items MoM           = {refined_mom:+.2f}%")
    print("\n  " + "-" * 50)
    print(f"  BASELINE (pure seasonal):   May-2026 CPI YoY = {yoy(seas_all):.2f}%")
    print(f"  REFINED  (+DESNZ +Ofgem):   May-2026 CPI YoY = {yoy(refined_mom):.2f}%   <<<")
    print("  " + "-" * 50)
    print("\n  Note: fuel spiked Mar→Apr (already in the Apr base); the May fuel move + flat energy are")
    print("  what the refinement captures vs the seasonal guess. Remaining uncertainty: services/airfares/clothing.")


if __name__ == "__main__":
    main()
