"""
evaluate.py
-----------
Comprehensive evaluation suite for the Aiqon Recommendation Engine.

Evaluates all shipments against four baselines (Cheapest, Fastest,
Most Reliable, Random), computes multi-objective regret metrics,
classifies decision complexity, and reports on-time reliability distributions.

Outputs:
  results/results_table.csv   -- Per-shipment metrics, scores, and baselines
  results/summary.txt         -- Aggregate performance summary

Usage:
  python evaluate.py                    # Standard benchmark evaluation (40 shipments)
  python evaluate.py --stress-test      # Realistic delivery window stress test (60%-88% on-time)
  python evaluate.py --seed 123         # Custom random seed
  python main.py evaluate               # Via CLI dispatcher
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from Recommendation_Engine.DataLoader import DataLoader
from Recommendation_Engine.engine import RecommendationEngine

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Stress-test delivery window overrides (commercial realistic schedules)
REALISTIC_STRESS_OVERRIDES = {
    "S001": {"deadline": 33.5, "budget": 25000},
    "S003": {"deadline": 36.5, "budget": 12000},
    "S004": {"deadline": 33.0, "budget": 18000},
    "S006": {"deadline": 42.0, "budget": 14000},
    "S007": {"deadline": 14.5, "budget": 15000},
    "S008": {"deadline": 35.0, "budget": 12000},
    "S011": {"deadline": 36.0, "budget": 18000},
    "S016": {"deadline": 27.5, "budget": 22000},
    "S017": {"deadline": 35.0, "budget": 22000},
    "S018": {"deadline": 34.0, "budget": 15000},
    "S027": {"deadline": 40.0, "budget": 30000},
    "S034": {"deadline": 34.0, "budget": 15000},
}


# ─────────────────────────────────────────────────────────────────────────────
# Baseline Selection Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pick_baseline(feas: pd.DataFrame, mode: str, rng: random.Random) -> str | None:
    """Return a route_id from the feasible DataFrame using the chosen strategy."""
    if feas.empty:
        return None
    if mode == "cheapest":
        return feas.loc[feas.cost.idxmin(), "route_id"]
    if mode == "fastest":
        return feas.loc[feas.exp_time.idxmin(), "route_id"]
    if mode == "most_reliable":
        return feas.loc[feas.p_on_time.idxmax(), "route_id"]
    if mode == "random":
        return rng.choice(feas.route_id.tolist())
    raise ValueError(f"Unknown baseline mode: {mode}")


def _regret(engine_val: float, baseline_val: float) -> float:
    """Signed regret: positive means the engine did worse than the single-objective baseline."""
    return engine_val - baseline_val


# ─────────────────────────────────────────────────────────────────────────────
# Main Evaluation Function
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(
    loader: DataLoader | None = None,
    engine: RecommendationEngine | None = None,
    seed: int = 42,
    verbose: bool = True,
    stress_test: bool = False,
    n_sim: int = 20_000,
) -> pd.DataFrame:
    """
    Evaluate all shipments and return a comprehensive results DataFrame.

    Parameters
    ----------
    loader      : DataLoader instance (created if None)
    engine      : RecommendationEngine instance (created if None)
    seed        : Random seed for reproducibility
    verbose     : Print summary to stdout if True
    stress_test : Apply realistic delivery deadlines (60%-88% on-time)
    n_sim       : Monte Carlo simulation iterations per route
    """
    if loader is None:
        loader = DataLoader()
    if engine is None:
        engine = RecommendationEngine(loader, n_sim=n_sim, seed=seed)

    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []

    shipment_ids = loader.shipments.shipment_id.tolist()

    for sid in shipment_ids:
        # Optional: apply realistic stress-test overrides
        if stress_test and sid in REALISTIC_STRESS_OVERRIDES:
            ov = REALISTIC_STRESS_OVERRIDES[sid]
            loader.shipments.loc[loader.shipments.shipment_id == sid, "deadline_days"] = ov["deadline"]
            if "budget" in ov:
                loader.shipments.loc[loader.shipments.shipment_id == sid, "budget_usd"] = ov["budget"]

        s = loader.shipment(sid)

        # ── 1. Engine Recommendation ─────────────────────────────────────────
        result = engine.find_best_route(sid)
        full_table: pd.DataFrame = result["table"]
        feasible_df = full_table[full_table.feasible].copy()

        n_candidates = len(full_table)
        n_feasible = len(feasible_df)

        if "score" in full_table.columns:
            n_pareto = int(full_table["score"].notna().sum())
        elif "pareto" in full_table.columns:
            n_pareto = int(full_table["pareto"].sum())
        else:
            n_pareto = 0

        best = result["best"]
        rec_id = best["route_id"]
        rec_cost = float(best["cost"])
        rec_exp_time = float(best["exp_time"])
        rec_p_on_time = float(best["p_on_time"])
        rec_risk = float(best.get("risk", np.nan))
        rec_score = float(best.get("score", np.nan))

        # ── 2. Baseline Strategies ───────────────────────────────────────────
        cheap_id = _pick_baseline(feasible_df, "cheapest", rng)
        fast_id = _pick_baseline(feasible_df, "fastest", rng)
        rely_id = _pick_baseline(feasible_df, "most_reliable", rng)
        rand_id = _pick_baseline(feasible_df, "random", rng)

        def _metric(route_id: str | None, col: str) -> float:
            if route_id is None:
                return float("nan")
            row = full_table[full_table.route_id == route_id]
            return float(row[col].iloc[0]) if not row.empty else float("nan")

        cheap_cost = _metric(cheap_id, "cost")
        fast_time = _metric(fast_id, "exp_time")
        rely_p = _metric(rely_id, "p_on_time")
        rand_cost = _metric(rand_id, "cost")
        rand_time = _metric(rand_id, "exp_time")

        # ── 3. Multi-Objective Regret ─────────────────────────────────────────
        regret_cost = _regret(rec_cost, cheap_cost)        # + means engine paid more for speed/safety
        regret_time = _regret(rec_exp_time, fast_time)     # + means engine accepted slower transit for savings

        # ── 4. Classification Flags ───────────────────────────────────────────
        is_feasible = bool(result["feasible"])
        is_pareto = bool(best.get("pareto", False)) if "pareto" in best else (n_pareto > 0 and is_feasible)
        is_open = bool(n_pareto > 1 and is_feasible)
        
        # Trivial risk: all feasible routes achieve >= 99.9% on-time probability
        trivial_risk = bool((feasible_df.p_on_time >= 0.999).all()) if n_feasible > 0 else False

        # Reliability category tag
        if rec_p_on_time >= 0.95:
            rel_tier = "High (>=95%)"
        elif rec_p_on_time >= 0.80:
            rel_tier = "Moderate (80-95%)"
        elif rec_p_on_time > 0.0:
            rel_tier = "At-Risk (<80%)"
        else:
            rel_tier = "Missed (0%)"

        rows.append(dict(
            shipment_id=sid,
            origin=s.origin,
            destination=s.destination,
            cargo_type=s.cargo_type,
            priority=s.priority,
            weight_kg=float(s.weight_kg),
            deadline_days=float(s.deadline_days),
            budget_usd=float(s.budget_usd),
            # Counts
            n_candidates=n_candidates,
            n_feasible=n_feasible,
            n_pareto=n_pareto,
            # Engine pick
            recommended_route=rec_id,
            recommended_cost=round(rec_cost, 2),
            recommended_exp_time=round(rec_exp_time, 2),
            recommended_p_on_time=round(rec_p_on_time, 4),
            reliability_tier=rel_tier,
            recommended_risk=round(rec_risk, 4) if not np.isnan(rec_risk) else np.nan,
            recommended_score=round(rec_score, 4) if not np.isnan(rec_score) else np.nan,
            # Baselines
            cheapest_route=cheap_id,
            cheapest_cost=round(cheap_cost, 2) if not np.isnan(cheap_cost) else np.nan,
            fastest_route=fast_id,
            fastest_exp_time=round(fast_time, 2) if not np.isnan(fast_time) else np.nan,
            reliable_route=rely_id,
            reliable_p_on_time=round(rely_p, 4) if not np.isnan(rely_p) else np.nan,
            random_route=rand_id,
            # Regret
            regret_cost=round(regret_cost, 2) if not np.isnan(regret_cost) else np.nan,
            regret_time=round(regret_time, 2) if not np.isnan(regret_time) else np.nan,
            # Flags
            constraint_satisfied=is_feasible,
            is_pareto_member=is_pareto,
            is_open=is_open,
            trivial_risk=trivial_risk,
        ))

    df = pd.DataFrame(rows)

    # ── 5. Save Artifacts ───────────────────────────────────────────────────
    RESULTS_DIR.mkdir(exist_ok=True)
    csv_file = RESULTS_DIR / ("results_table_stress.csv" if stress_test else "results_table.csv")
    df.to_csv(csv_file, index=False)

    summary = _build_summary(df, stress_test=stress_test)
    summary_file = RESULTS_DIR / ("summary_stress.txt" if stress_test else "summary.txt")
    summary_file.write_text(summary, encoding="utf-8")

    if verbose:
        print(summary)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Summary Report Generator
# ─────────────────────────────────────────────────────────────────────────────

def _build_summary(df: pd.DataFrame, stress_test: bool = False) -> str:
    total = len(df)
    n_feasible_ship = int(df.constraint_satisfied.sum())
    n_pareto_members = int(df.is_pareto_member.sum())
    
    # Feasible partition
    feas = df[df.constraint_satisfied].copy()
    n_open = int(feas.is_open.sum())
    n_forced = int((~feas.is_open).sum())
    n_trivial = int(feas.trivial_risk.sum())

    # Baseline agreements
    if not feas.empty:
        agree_cheapest = int((feas.recommended_route == feas.cheapest_route).sum())
        agree_fastest = int((feas.recommended_route == feas.fastest_route).sum())
        agree_reliable = int((feas.recommended_route == feas.reliable_route).sum())

        mean_regret_cost = float(feas.regret_cost.mean())
        mean_regret_time = float(feas.regret_time.mean())
        pct_pos_regret_cost = int((feas.regret_cost > 0).sum())
        pct_pos_regret_time = int((feas.regret_time > 0).sum())
    else:
        agree_cheapest = agree_fastest = agree_reliable = 0
        mean_regret_cost = mean_regret_time = 0.0
        pct_pos_regret_cost = pct_pos_regret_time = 0

    # Reliability tier distribution
    tier_counts = df["reliability_tier"].value_counts().to_dict()

    mode_label = "REALISTIC STRESS-TEST EVALUATION" if stress_test else "STANDARD BENCHMARK EVALUATION"

    lines = [
        "=" * 70,
        f"  AIQON RECOMMENDATION ENGINE -- {mode_label}",
        "=" * 70,
        f"  Total Shipments Evaluated : {total}",
        f"  Constraint Satisfied      : {n_feasible_ship} / {total}  ({n_feasible_ship/total:.0%})",
        f"  Pareto-Efficient Choices  : {n_pareto_members} / {max(1, n_feasible_ship)}  ({n_pareto_members/max(1, n_feasible_ship):.0%})",
        "",
        "  Decision Complexity (Feasible Shipments):",
        f"    * Forced Decisions (1 Pareto option) : {n_forced}",
        f"    * Open Decisions   (>1 Pareto options): {n_open}",
        f"    * Trivial Risk     (all >=99.9%)      : {n_trivial}",
    ]

    if not stress_test and n_trivial > 0:
        lines.append("      [Note: Trivial Risk occurs when synthetic deadlines provide 40-80d excess buffer]")
    elif stress_test:
        lines.append("      [Stress-Test: Tight deadlines eliminate artificial >99.9% reliability]")

    lines.extend([
        "",
        "  On-Time Reliability Distribution (Recommended Routes):",
        f"    * High Reliability (>=95%)   : {tier_counts.get('High (>=95%)', 0)}",
        f"    * Moderate Risk    (80%-95%) : {tier_counts.get('Moderate (80-95%)', 0)}",
        f"    * At-Risk / Tight  (<80%)    : {tier_counts.get('At-Risk (<80%)', 0)}",
        f"    * Complete Miss    (0.0%)    : {tier_counts.get('Missed (0%)', 0)}",
        "",
        "  Baseline Alignment (Feasible Shipments):",
        f"    * Same as Cheapest Route      : {agree_cheapest} / {max(1, len(feas))}  ({agree_cheapest/max(1, len(feas)):.0%})",
        f"    * Same as Fastest Route       : {agree_fastest} / {max(1, len(feas))}  ({agree_fastest/max(1, len(feas)):.0%})",
        f"    * Same as Most-Reliable Route : {agree_reliable} / {max(1, len(feas))}  ({agree_reliable/max(1, len(feas)):.0%})",
        "",
        "  Multi-Objective Regret:",
        f"    * Mean Cost Regret vs. Cheapest   : ${mean_regret_cost:+,.0f}",
        f"      (Engine pays a premium for speed/reliability in {pct_pos_regret_cost}/{max(1, len(feas))} shipments)",
        f"    * Mean Transit Regret vs. Fastest : {mean_regret_time:+.2f} days",
        f"      (Engine accepts slower transit to reduce freight cost in {pct_pos_regret_time}/{max(1, len(feas))} shipments)",
        "=" * 70,
    ])
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the Aiqon Recommendation Engine across all shipments.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for tie-breakers and baselines (default: 42)")
    parser.add_argument("--n-sim", type=int, default=20_000, help="Number of Monte Carlo simulations per route (default: 20000)")
    parser.add_argument("--stress-test", action="store_true", help="Run with realistic delivery windows (60%-88% on-time reliability)")
    parser.add_argument("--quiet", action="store_true", help="Suppress console summary printing")
    args = parser.parse_args()

    run_evaluation(
        seed=args.seed,
        n_sim=args.n_sim,
        stress_test=args.stress_test,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
