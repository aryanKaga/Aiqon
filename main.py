"""
main.py -- Shipping Route Recommendation System CLI
===================================================

Subcommands
-----------
  recommend   Find the best route for a single shipment
  evaluate    Run all 40 shipments and compare against baselines
  sensitivity Sensitivity analysis on the open (trade-off) shipments

Examples
--------
  python main.py recommend --shipment S001
  python main.py recommend --shipment S028 --weights 0.5 0.3 0.2
  python main.py evaluate
  python main.py evaluate --seed 42
  python main.py sensitivity
  python main.py sensitivity --seed 42
"""

from __future__ import annotations

import argparse
import textwrap

from Recommendation_Engine.DataLoader import DataLoader
from Recommendation_Engine.engine import RecommendationEngine


# ─────────────────────────────────────────────────────────────────────────────
# formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _bar(label: str, width: int = 65) -> str:
    pad = max(0, width - len(label) - 4)
    return f"\n{'--'} {label} {'-' * pad}"


def _fmt_table(df, cols: list[str], fmt: dict | None = None) -> str:
    """Return a plain-text table with right-aligned numbers."""
    fmt = fmt or {}
    sub = df[cols].copy()
    for col, f in fmt.items():
        if col in sub.columns:
            sub[col] = sub[col].apply(lambda x: f.format(x) if x == x else "-")
    col_widths = {c: max(len(str(c)), sub[c].astype(str).str.len().max()) for c in cols}
    header = "  ".join(str(c).ljust(col_widths[c]) for c in cols)
    sep = "  ".join("-" * col_widths[c] for c in cols)
    rows = [header, sep]
    for _, row in sub.iterrows():
        rows.append("  ".join(str(row[c]).ljust(col_widths[c]) for c in cols))
    return "\n".join(rows)


# ─────────────────────────────────────────────────────────────────────────────
# recommend
# ─────────────────────────────────────────────────────────────────────────────

def cmd_recommend(args: argparse.Namespace) -> None:
    loader = DataLoader()
    engine = RecommendationEngine(loader, n_sim=20_000, seed=args.seed)

    weights = args.weights  # list[float] or None

    print(_bar(f"Recommendation for {args.shipment}"))
    s = loader.shipment(args.shipment)
    print(f"  Origin      : {s.origin}")
    print(f"  Destination : {s.destination}")
    print(f"  Cargo type  : {s.cargo_type}   Priority: {s.priority}")
    print(f"  Weight      : {s.weight_kg:,.0f} kg   Deadline: {s.deadline_days} days   Budget: ${s.budget_usd:,.0f}")

    result = engine.find_best_route(args.shipment, weights=weights)

    print(_bar("Explanation"))
    print(textwrap.fill(result["explanation"], width=70, initial_indent="  ", subsequent_indent="  "))

    if result.get("weights"):
        w = result["weights"]
        print(f"\n  Applied weights => cost: {w['cost']:.3f}  time: {w['time']:.3f}  risk: {w['risk']:.3f}")

    table = result["table"]
    display_cols = ["route_id", "path", "modes", "cost", "exp_time", "p_on_time",
                    "p_any_delay", "buffer_days", "feasible", "reason"]
    display_cols = [c for c in display_cols if c in table.columns]

    print(_bar("All candidate routes"))
    print(_fmt_table(
        table, display_cols,
        fmt={
            "cost": "${:,.0f}",
            "exp_time": "{:.1f}d",
            "p_on_time": "{:.1%}",
            "p_any_delay": "{:.1%}",
            "buffer_days": "{:.1f}",
        },
    ))

    if result["feasible"]:
        best = result["best"]
        print(_bar("Best route"))
        print(f"  Route   : {best['route_id']}  ({best['path']})")
        print(f"  Cost    : ${best['cost']:,.0f}")
        print(f"  Exp. time: {best['exp_time']:.1f} days")
        print(f"  On-time : {best['p_on_time']:.1%}")
        if result["second"]:
            s2 = result["second"]
            print(_bar("Runner-up"))
            print(f"  Route   : {s2['route_id']}  ({s2['path']})")
            print(f"  Cost    : ${s2['cost']:,.0f}")
            print(f"  Exp. time: {s2['exp_time']:.1f} days")
            print(f"  On-time : {s2['p_on_time']:.1%}")
    else:
        print("\n  [!] No route fully satisfies the constraints. Showing least-violating route.")

    print()


# ─────────────────────────────────────────────────────────────────────────────
# evaluate
# ─────────────────────────────────────────────────────────────────────────────

def cmd_evaluate(args: argparse.Namespace) -> None:
    from evaluate import run_evaluation
    loader = DataLoader()
    engine = RecommendationEngine(loader, n_sim=20_000, seed=args.seed)
    df = run_evaluation(loader=loader, engine=engine, seed=args.seed, verbose=True)
    print(f"\n  Full results saved to: results/results_table.csv")


# ─────────────────────────────────────────────────────────────────────────────
# sensitivity
# ─────────────────────────────────────────────────────────────────────────────

def cmd_sensitivity(args: argparse.Namespace) -> None:
    from evaluate import run_evaluation
    from sensitivity import run_sensitivity
    loader = DataLoader()
    engine = RecommendationEngine(loader, n_sim=20_000, seed=args.seed)

    # Always (re-)run evaluation first so open shipments are up-to-date
    print("Running evaluation first to identify open shipments...")
    eval_df = run_evaluation(loader=loader, engine=engine, seed=args.seed, verbose=False)

    print(f"Running sensitivity analysis on {eval_df.is_open.sum()} open shipments...\n")
    run_sensitivity(loader=loader, engine=engine, eval_df=eval_df, seed=args.seed, verbose=True)
    print(f"\n  Sensitivity table saved to: results/sensitivity_table.csv")


# ─────────────────────────────────────────────────────────────────────────────
# server
# ─────────────────────────────────────────────────────────────────────────────

def cmd_server(args: argparse.Namespace) -> None:
    from server import start_server
    start_server(host=args.host, port=args.port)


# ─────────────────────────────────────────────────────────────────────────────
# argument parser
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Shipping Route Recommendation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python main.py recommend --shipment S001
              python main.py recommend --shipment S028 --weights 0.5 0.3 0.2
              python main.py evaluate
              python main.py sensitivity
              python main.py server --port 5000
        """),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── recommend ──────────────────────────────────────────────────────────
    p_rec = sub.add_parser("recommend", help="Recommend a route for a shipment")
    p_rec.add_argument("--shipment", required=True, metavar="ID",
                       help="Shipment ID (e.g. S001)")
    p_rec.add_argument("--weights", nargs=3, type=float, metavar=("COST", "TIME", "RISK"),
                       default=None,
                       help="Override priority weights (must sum > 0; will be normalised)")
    p_rec.add_argument("--seed", type=int, default=42, help="RNG seed (default 42)")

    # ── evaluate ──────────────────────────────────────────────────────────
    p_eval = sub.add_parser("evaluate", help="Evaluate all shipments against baselines")
    p_eval.add_argument("--seed", type=int, default=42, help="RNG seed (default 42)")

    # ── sensitivity ───────────────────────────────────────────────────────
    p_sens = sub.add_parser("sensitivity", help="Sensitivity analysis on open shipments")
    p_sens.add_argument("--seed", type=int, default=42, help="RNG seed (default 42)")

    # ── server ────────────────────────────────────────────────────────────
    p_srv = sub.add_parser("server", help="Launch web dashboard & REST API server")
    p_srv.add_argument("--host", type=str, default="127.0.0.1", help="Host address (default 127.0.0.1)")
    p_srv.add_argument("--port", type=int, default=5000, help="Port number (default 5000)")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    dispatch = {
        "recommend": cmd_recommend,
        "evaluate": cmd_evaluate,
        "sensitivity": cmd_sensitivity,
        "server": cmd_server,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()

