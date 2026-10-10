"""
evaluate.py
-----------
Runs the RecommendationEngine on all shipments, compares against four
baselines (cheapest, fastest, most-reliable, random), and produces:

  results/results_table.csv   — per-shipment metrics and baseline comparisons
  results/summary.txt         — aggregate statistics

Usage (via main.py):
    python main.py evaluate

Or directly:
    python evaluate.py
"""

from __future__ import annotations

import random
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

from Recommendation_Engine.DataLoader import DataLoader
from Recommendation_Engine.engine import RecommendationEngine

RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ─────────────────────────────────────────────────────────────────────────────
# helpers
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
    """Signed regret: positive means the engine did *worse* than the baseline."""
    return engine_val - baseline_val


# ─────────────────────────────────────────────────────────────────────────────
# main evaluation function
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(
    loader: DataLoader | None = None,
    engine: RecommendationEngine | None = None,
    seed: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Evaluate all shipments and return a results DataFrame.

    Parameters
    ----------
    loader  : DataLoader instance (created if None)
    engine  : RecommendationEngine instance (created if None)
    seed    : random seed for reproducibility (affects the random baseline)
    verbose : print summary to stdout if True
    """
    if loader is None:
        loader = DataLoader()
    if engine is None:
        engine = RecommendationEngine(loader, n_sim=20_000, seed=seed)

    rng = random.Random(seed)
    rows: list[dict] = []

    shipment_ids = loader.shipments.shipment_id.tolist()

    for sid in shipment_ids:
        s = loader.shipment(sid)

        # ── engine recommendation ────────────────────────────────────────────
        result = engine.find_best_route(sid)
        full_table: pd.DataFrame = result["table"]
        feasible_df = full_table[full_table.feasible].copy()

        n_candidates = len(full_table)
        n_feasible = len(feasible_df)

        # Pareto column: present in merged table only for Pareto-front routes (non-NaN score)
        if "score" in full_table.columns:
            n_pareto = int(full_table["score"].notna().sum())
        elif "pareto" in full_table.columns:
            n_pareto = int(full_table["pareto"].sum())
        else:
            n_pareto = 0

        best = result["best"]
        rec_id = best["route_id"]
        rec_cost = best["cost"]
        rec_exp_time = best["exp_time"]
        rec_p_on_time = best["p_on_time"]
        rec_risk = best.get("risk", float("nan"))
        rec_score = best.get("score", float("nan"))

        # ── baselines ────────────────────────────────────────────────────────
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

        # ── regret ───────────────────────────────────────────────────────────
        regret_cost = _regret(rec_cost, cheap_cost)        # + means engine is pricier
        regret_time = _regret(rec_exp_time, fast_time)     # + means engine is slower

        # ── flags ─────────────────────────────────────────────────────────────
        is_feasible = result["feasible"]
        is_pareto = bool(best.get("pareto", False)) if "pareto" in best else (n_pareto > 0 and result["feasible"])
        is_open = n_pareto > 1
        # trivial risk: all feasible routes have >= 99.9 % on-time probability
        if n_feasible > 0:
            trivial_risk = bool((feasible_df.p_on_time >= 0.999).all())
        else:
            trivial_risk = False

        rows.append(dict(
            shipment_id=sid,
            origin=s.origin,
            destination=s.destination,
            cargo_type=s.cargo_type,
            priority=s.priority,
            weight_kg=s.weight_kg,
            deadline_days=s.deadline_days,
            budget_usd=s.budget_usd,
            # counts
            n_candidates=n_candidates,
            n_feasible=n_feasible,
            n_pareto=n_pareto,
            # engine pick
            recommended_route=rec_id,
            recommended_cost=round(rec_cost, 2),
            recommended_exp_time=round(rec_exp_time, 2),
            recommended_p_on_time=round(rec_p_on_time, 4),
            recommended_risk=round(rec_risk, 4) if not np.isnan(rec_risk) else float("nan"),
            recommended_score=round(rec_score, 4) if not np.isnan(rec_score) else float("nan"),
            # baselines
            cheapest_route=cheap_id,
            cheapest_cost=round(cheap_cost, 2) if not np.isnan(cheap_cost) else float("nan"),
            fastest_route=fast_id,
            fastest_exp_time=round(fast_time, 2) if not np.isnan(fast_time) else float("nan"),
            reliable_route=rely_id,
            reliable_p_on_time=round(rely_p, 4) if not np.isnan(rely_p) else float("nan"),
            random_route=rand_id,
            # regret
            regret_cost=round(regret_cost, 2) if not np.isnan(regret_cost) else float("nan"),
            regret_time=round(regret_time, 2) if not np.isnan(regret_time) else float("nan"),
            # flags
            constraint_satisfied=is_feasible,
            is_pareto_member=is_pareto,
            is_open=is_open,
            trivial_risk=trivial_risk,
        ))

    df = pd.DataFrame(rows)

    # ── save ────────────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(RESULTS_DIR / "results_table.csv", index=False)

    # ── summary ──────────────────────────────────────────────────────────────
    summary = _build_summary(df)
    (RESULTS_DIR / "summary.txt").write_text(summary, encoding="utf-8")

    if verbose:
        print(summary)

    return df


def _build_summary(df: pd.DataFrame) -> str:
    total = len(df)
    n_feasible_ship = df.constraint_satisfied.sum()
    n_pareto_members = df.is_pareto_member.sum()
    n_open = df.is_open.sum()
    n_forced = (~df.is_open & df.constraint_satisfied).sum()
    n_trivial = df.trivial_risk.sum()

    # agreement with baselines (among feasible shipments)
    feas = df[df.constraint_satisfied]
    agree_cheapest = (feas.recommended_route == feas.cheapest_route).sum()
    agree_fastest = (feas.recommended_route == feas.fastest_route).sum()
    agree_reliable = (feas.recommended_route == feas.reliable_route).sum()

    mean_regret_cost = feas.regret_cost.mean()
    mean_regret_time = feas.regret_time.mean()
    pct_pos_regret_cost = (feas.regret_cost > 0).sum()   # engine pricier than cheapest
    pct_pos_regret_time = (feas.regret_time > 0).sum()   # engine slower than fastest

    lines = [
        "=" * 65,
        "  RECOMMENDATION ENGINE -- EVALUATION SUMMARY",
        "=" * 65,
        f"  Total shipments          : {total}",
        f"  Constraint satisfied     : {n_feasible_ship} / {total}  ({n_feasible_ship/total:.0%})",
        f"  Recommended on Pareto    : {n_pareto_members} / {n_feasible_ship}  ({n_pareto_members/n_feasible_ship:.0%})",
        "",
        "  Decision categories (feasible shipments)",
        f"    Forced   (1 Pareto pt) : {n_forced}",
        f"    Open     (>1 Pareto pt): {n_open}",
        f"    Trivial risk           : {n_trivial}  (all routes >=99.9% on-time)",
        "",
        "  Baseline agreement (feasible shipments)",
        f"    Same as cheapest       : {agree_cheapest} / {len(feas)}  ({agree_cheapest/len(feas):.0%})",
        f"    Same as fastest        : {agree_fastest} / {len(feas)}  ({agree_fastest/len(feas):.0%})",
        f"    Same as most reliable  : {agree_reliable} / {len(feas)}  ({agree_reliable/len(feas):.0%})",
        "",
        "  Regret vs cheapest (cost, USD):",
        f"    Mean  : ${mean_regret_cost:+,.0f}",
        f"    Engine pricier than cheapest in {pct_pos_regret_cost}/{len(feas)} cases",
        "",
        "  Regret vs fastest (time, days):",
        f"    Mean  : {mean_regret_time:+.2f} days",
        f"    Engine slower than fastest in {pct_pos_regret_time}/{len(feas)} cases",
        "=" * 65,
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_evaluation(verbose=True)
