"""
sensitivity.py
--------------
Systematic sensitivity analysis on the "open" shipments
(those with more than one non-dominated feasible route).

Perturbations tested
────────────────────
1. Weight perturbation   : ±20% on each of cost / time / risk (renormalised)  → 6 scenarios
2. On-time requirement   : ±5 pp on the shipment's required_on_time_prob      → 2 scenarios
3. Delay probability     : ±20% on all legs simultaneously                    → 2 scenarios
4. Normalization method  : swap min-max for rank-based normalization           → 1 scenario

Tags
────
  "robust"    : baseline recommendation unchanged in ≥ 80 % of scenarios
  "close call": recommendation flips in > 20 % of scenarios

Output
──────
  results/sensitivity_table.csv
  results/sensitivity_summary.txt  (also printed to stdout)

Usage (via main.py):
    python main.py sensitivity

Or directly:
    python sensitivity.py
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from Recommendation_Engine.DataLoader import DataLoader
from Recommendation_Engine.engine import RecommendationEngine, PRIORITY_WEIGHTS, RISK_BUMP, _minmax

RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ─────────────────────────────────────────────────────────────────────────────
# Rank normalisation helper (alternative to min-max)
# ─────────────────────────────────────────────────────────────────────────────

def _rank_norm(x: np.ndarray) -> np.ndarray:
    """Normalise to [0, 1] by rank position (0 = best/lowest)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 1:
        return np.zeros(1)
    order = np.argsort(x)
    ranks = np.empty(n)
    ranks[order] = np.arange(n)
    return ranks / (n - 1)


# ─────────────────────────────────────────────────────────────────────────────
# Perturbed engine — delay probability scenario
# ─────────────────────────────────────────────────────────────────────────────

class _PerturbedDelayEngine(RecommendationEngine):
    """RecommendationEngine with delay_prob scaled by `factor` on all legs."""

    def __init__(self, base_loader: DataLoader, factor: float, **kwargs):
        # Build a loader whose legs have perturbed delay_prob
        perturbed_loader = copy.copy(base_loader)
        perturbed_legs = base_loader.legs.copy()
        perturbed_legs["delay_prob"] = (perturbed_legs["delay_prob"] * factor).clip(0, 1)
        perturbed_loader.legs = perturbed_legs
        super().__init__(perturbed_loader, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Rank-norm scoring helper (reuses engine internals)
# ─────────────────────────────────────────────────────────────────────────────

def _score_with_rank_norm(engine: RecommendationEngine, shipment_id: str) -> str | None:
    """Run the pipeline but replace min-max with rank normalisation for cost/time."""
    s = engine.loader.shipment(shipment_id)
    need = engine._required_on_time(s)
    df = engine.evaluate(shipment_id)
    feas = df[df.feasible].copy()
    if feas.empty:
        return None

    feas["risk"] = (
        0.5 * ((1 - feas.p_on_time) / max(1 - need, 1e-6)).clip(0, 1)
        + 0.5 * _rank_norm(feas.buffer_days.to_numpy())
    )
    pareto_mask = engine._pareto(feas, ["cost", "exp_time", "risk"])
    feas["pareto"] = pareto_mask
    front = feas[feas.pareto].copy()
    if front.empty:
        return None

    w = engine._weights(s)
    front["score"] = (
        w[0] * _rank_norm(front.cost.to_numpy())
        + w[1] * _rank_norm(front.exp_time.to_numpy())
        + w[2] * front.risk.to_numpy()
    )
    return front.sort_values("score").iloc[0]["route_id"]


# ─────────────────────────────────────────────────────────────────────────────
# Per-shipment sensitivity
# ─────────────────────────────────────────────────────────────────────────────

def _analyse_shipment(
    sid: str,
    engine: RecommendationEngine,
    loader: DataLoader,
    seed: int,
) -> dict[str, Any]:
    """Return a dict of scenario results for one open shipment."""
    s = loader.shipment(sid)
    baseline_result = engine.find_best_route(sid)
    baseline_id = baseline_result["best"]["route_id"]

    scenarios: dict[str, str | None] = {}

    # ── 1. Weight perturbations ────────────────────────────────────────────
    base_weights = PRIORITY_WEIGHTS[s.priority]
    bump = RISK_BUMP.get(s.cargo_type, 0.0)
    w_base = np.array(base_weights, dtype=float)
    w_base[2] += bump
    w_base /= w_base.sum()

    for axis, label in enumerate(["cost", "time", "risk"]):
        for sign, suffix in [(+0.2, "+20%"), (-0.2, "-20%")]:
            w_pert = w_base.copy()
            w_pert[axis] *= (1 + sign)
            w_pert /= w_pert.sum()
            result = engine.find_best_route(sid, weights=w_pert.tolist())
            scenarios[f"w_{label}_{suffix}"] = result["best"]["route_id"] if result["feasible"] else None

    # ── 2. On-time requirement perturbation ───────────────────────────────
    base_otp = engine._required_on_time(s)
    for delta, suffix in [(-0.05, "-5pp"), (+0.05, "+5pp")]:
        perturbed_otp = float(np.clip(base_otp + delta, 0.0, 1.0))
        # Temporarily override via custom min_on_time mapping
        orig = engine.min_on_time.copy()
        engine.min_on_time = {k: perturbed_otp for k in engine.min_on_time}
        result = engine.find_best_route(sid)
        engine.min_on_time = orig
        scenarios[f"otp_{suffix}"] = result["best"]["route_id"] if result["feasible"] else None

    # ── 3. Delay probability perturbation ─────────────────────────────────
    for factor, suffix in [(0.8, "-20%"), (1.2, "+20%")]:
        pert_engine = _PerturbedDelayEngine(
            loader, factor=factor, n_sim=20_000, seed=seed
        )
        result = pert_engine.find_best_route(sid)
        scenarios[f"delay_{suffix}"] = result["best"]["route_id"] if result["feasible"] else None

    # ── 4. Rank-norm instead of min-max ──────────────────────────────────
    scenarios["rank_norm"] = _score_with_rank_norm(engine, sid)

    # ── aggregate ─────────────────────────────────────────────────────────
    n_scenarios = len(scenarios)
    n_same = sum(1 for v in scenarios.values() if v == baseline_id)
    stability = n_same / n_scenarios

    return dict(
        shipment_id=sid,
        origin=s.origin,
        destination=s.destination,
        cargo_type=s.cargo_type,
        priority=s.priority,
        baseline_route=baseline_id,
        n_scenarios=n_scenarios,
        n_same_as_baseline=n_same,
        stability_pct=round(stability * 100, 1),
        tag="robust" if stability >= 0.80 else "close call",
        **{f"scenario_{k}": v for k, v in scenarios.items()},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_sensitivity(
    loader: DataLoader | None = None,
    engine: RecommendationEngine | None = None,
    eval_df: pd.DataFrame | None = None,
    seed: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run sensitivity analysis on the open shipments.

    Parameters
    ----------
    loader   : DataLoader (created if None)
    engine   : RecommendationEngine (created if None)
    eval_df  : results DataFrame from evaluate.run_evaluation (loaded from CSV if None)
    seed     : random seed
    verbose  : print summary to stdout
    """
    if loader is None:
        loader = DataLoader()
    if engine is None:
        engine = RecommendationEngine(loader, n_sim=20_000, seed=seed)

    # Determine open shipments
    if eval_df is None:
        csv_path = RESULTS_DIR / "results_table.csv"
        if csv_path.exists():
            eval_df = pd.read_csv(csv_path)
        else:
            # Run evaluation first
            from evaluate import run_evaluation
            eval_df = run_evaluation(loader=loader, engine=engine, seed=seed, verbose=False)

    open_ids = eval_df[eval_df.is_open == True].shipment_id.tolist()

    if not open_ids:
        print("No open shipments found — sensitivity analysis skipped.")
        return pd.DataFrame()

    rows = []
    for sid in open_ids:
        if verbose:
            print(f"  Analysing {sid} ...", end="\r", flush=True)
        rows.append(_analyse_shipment(sid, engine, loader, seed=seed))

    df = pd.DataFrame(rows)

    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(RESULTS_DIR / "sensitivity_table.csv", index=False)

    summary = _build_summary(df, open_ids)
    (RESULTS_DIR / "sensitivity_summary.txt").write_text(summary, encoding="utf-8")

    if verbose:
        print()   # clear the \r line
        print(summary)

    return df


def _build_summary(df: pd.DataFrame, open_ids: list[str]) -> str:
    n_open = len(open_ids)
    n_robust = (df.tag == "robust").sum()
    n_close = (df.tag == "close call").sum()
    mean_stab = df.stability_pct.mean()

    lines = [
        "=" * 65,
        "  SENSITIVITY ANALYSIS SUMMARY",
        "=" * 65,
        f"  Open shipments analysed  : {n_open}",
        f"  Scenarios per shipment   : {df.n_scenarios.iloc[0] if len(df) > 0 else 'N/A'}",
        "",
        f"  Robust     (>=80% stable): {n_robust} / {n_open}  ({n_robust/n_open:.0%})",
        f"  Close call (<80% stable) : {n_close} / {n_open}  ({n_close/n_open:.0%})",
        f"  Mean stability           : {mean_stab:.1f}%",
        "",
        "  Per-shipment breakdown:",
    ]

    for _, row in df.sort_values("stability_pct").iterrows():
        tag_str = f"[{row.tag.upper():10s}]"
        lines.append(
            f"    {row.shipment_id}  {tag_str}  "
            f"stable in {row.n_same_as_baseline}/{row.n_scenarios} scenarios "
            f"({row.stability_pct:.0f}%)  baseline={row.baseline_route}"
        )

    lines.append("=" * 65)
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_sensitivity(verbose=True)
