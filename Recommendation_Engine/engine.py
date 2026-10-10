import numpy as np
import pandas as pd

from Recommendation_Engine.DataLoader import DataLoader

# weights on (cost, time, risk) by the shipper's soft preference
PRIORITY_WEIGHTS = {
    "cost":        (0.6, 0.2, 0.2),
    "speed":       (0.2, 0.6, 0.2),
    "reliability": (0.2, 0.2, 0.6),
    "balanced":    (1 / 3, 1 / 3, 1 / 3),
}
RISK_BUMP = {"perishable": 0.10, "high_value": 0.10}     # extra weight on risk, then renormalised
# used only if shipments.csv has no required_on_time_prob column (a decision-maker policy, not data)
DEFAULT_MIN_ON_TIME = {"general": 0.90, "perishable": 0.95, "high_value": 0.95, "urgent": 0.95}


def _minmax(x):
    x = np.asarray(x, dtype=float)
    span = x.max() - x.min()
    return np.zeros_like(x) if span == 0 else (x - x.min()) / span


class RecommendationEngine:
    def __init__(self, loader: DataLoader, n_sim: int = 20000, seed: int = 0, min_on_time: dict = None):
        self.loader = loader
        self.graph = loader.graph
        self.n_sim, self.seed = n_sim, seed
        self.min_on_time = {**DEFAULT_MIN_ON_TIME, **(min_on_time or {})}
        self._sims = {}
        self._route_index = {rid: i for i, rid in enumerate(loader.routes.route_id)}

    # ------------------------------------------------------------------ candidates
    def find_routes(self, shipment_id: str) -> pd.DataFrame:
        """Candidate routes for the shipment (same origin and destination)."""
        return self.loader.candidates(shipment_id)

    def is_valid_route(self, shipment, route_id: str) -> bool:
        """Route is a connected, loop-free path in the graph from the shipment's origin to its destination."""
        L = self.loader.route_legs(route_id)
        chained = bool((L.to_hub.values[:-1] == L.from_hub.values[1:]).all())
        ends = L.from_hub.iloc[0] == shipment.origin and L.to_hub.iloc[-1] == shipment.destination
        in_graph = all(any(e["leg_id"] == lid for e in self.graph.get(fh, []))
                       for lid, fh in zip(L.leg_id, L.from_hub))
        nodes = [L.from_hub.iloc[0]] + list(L.to_hub)
        return chained and ends and in_graph and len(set(nodes)) == len(nodes)

    # ------------------------------------------------------------------ metrics
    def _times(self, route_id: str) -> np.ndarray:
        """Monte Carlo total-time samples for a route (base transit + gamma delays), simulated once and cached.
        The samples do not depend on the shipment, so every shipment sees the same route distribution."""
        if route_id not in self._sims:
            L = self.loader.route_legs(route_id)
            p, mu, sd = (L.delay_prob.to_numpy(), L.delay_days_mean.to_numpy(), L.delay_days_std.to_numpy())
            rng = np.random.default_rng([self.seed, self._route_index[route_id]])
            hit = rng.random((self.n_sim, len(p))) < p
            delay = rng.gamma((mu / sd) ** 2, sd ** 2 / mu, size=(self.n_sim, len(p)))
            self._sims[route_id] = L.transit_days.sum() + (hit * delay).sum(axis=1)
        return self._sims[route_id]

    def route_metrics(self, shipment, route_id: str) -> dict:
        L = self.loader.route_legs(route_id)
        sims = self._times(route_id)
        late = np.maximum(0.0, sims - shipment.deadline_days)
        return dict(
            route_id=route_id,
            path=self.loader.route(route_id).path,
            modes="+".join(L["mode"]),
            n_legs=len(L),
            cost=float((L.fixed_cost_usd + shipment.weight_kg * L.cost_per_kg_usd).sum()),
            base_time=float(L.transit_days.sum()),
            exp_time=float(sims.mean()),
            p_on_time=float((sims <= shipment.deadline_days).mean()),
            exp_late_days=float(late.mean()),
            p_any_delay=float(1 - np.prod(1 - L.delay_prob.to_numpy())),
            buffer_days=float(np.quantile(sims, 0.95) - sims.mean()),      # extra days needed for a 95% safe plan
            worst_geo_risk=float(max(L.geo_risk_from.max(), L.geo_risk_to.max())),
            worst_congestion=float(max(L.congestion_from.max(), L.congestion_to.max())),
        )

    def _required_on_time(self, shipment) -> float:
        v = shipment.get("required_on_time_prob")
        return float(v) if v is not None and not pd.isna(v) else self.min_on_time[shipment["cargo_type"]]

    def evaluate(self, shipment_id: str) -> pd.DataFrame:
        """Metrics for every valid candidate route, enforcing both Budget and Deadline hard constraints."""
        s = self.loader.shipment(shipment_id)
        deadline = float(s.deadline_days if "deadline_days" in s else s.max_deadline_days)
        budget = float(s.budget_usd)

        rows = [self.route_metrics(s, r) for r in self.find_routes(shipment_id).route_id
                if self.is_valid_route(s, r)]
        df = pd.DataFrame(rows)

        over_budget = df.cost > budget
        over_deadline = df.exp_time > deadline

        df["feasible"] = ~(over_budget | over_deadline)

        reasons = []
        for i in df.index:
            ob = over_budget[i]
            od = over_deadline[i]
            if ob and od:
                reasons.append(f"over budget by ${df.cost[i] - budget:,.0f} & exceeds max deadline of {deadline:g}d (takes {df.exp_time[i]:.1f}d)")
            elif ob:
                reasons.append(f"over budget by ${df.cost[i] - budget:,.0f}")
            elif od:
                reasons.append(f"exceeds max deadline of {deadline:g}d (takes {df.exp_time[i]:.1f}d)")
            else:
                reasons.append("passes budget & deadline hard constraints")
        df["reason"] = reasons
        return df

    # ------------------------------------------------------------------ scoring
    @staticmethod
    def _weights(shipment, weights=None):
        w = np.array(weights if weights is not None else PRIORITY_WEIGHTS[shipment["priority"]], dtype=float)
        if weights is None:
            w[2] += RISK_BUMP.get(shipment["cargo_type"], 0.0)
        return w / w.sum()

    @staticmethod
    def _pareto(df, cols):
        A = df[cols].to_numpy()
        return np.array([not any(np.all(A[j] <= A[i]) and np.any(A[j] < A[i])
                                 for j in range(len(A)) if j != i) for i in range(len(A))])

    def find_best_route(self, shipment_id: str, weights=None) -> dict:
        """Feasibility (budget & deadline) -> Pareto filter -> weighted score. Returns top routes."""
        s = self.loader.shipment(shipment_id)
        need = self._required_on_time(s)
        df = self.evaluate(shipment_id)
        feas = df[df.feasible].copy()

        if feas.empty:                                    # nothing meets hard constraints: least-violating route
            deadline = float(s.deadline_days if "deadline_days" in s else s.max_deadline_days)
            budget = float(s.budget_usd)
            df["violation"] = ((df.cost / budget - 1).clip(lower=0) + (df.exp_time / deadline - 1).clip(lower=0))
            best = df.sort_values("violation").iloc[0]
            return dict(shipment_id=shipment_id, feasible=False, best=best.to_dict(), second=None, table=df,
                        explanation=(f"No route satisfies both budget and deadline hard constraints. Closest is {best.route_id} "
                                     f"({best.path}): {best.reason}."))

        # Continuous risk penalty = Unreliability (1 - p_on_time) + 0.5 * normalized buffer days
        feas["risk"] = (1.0 - feas.p_on_time) + 0.5 * _minmax(feas.buffer_days)
        feas["pareto"] = self._pareto(feas, ["cost", "exp_time", "risk"])
        front = feas[feas.pareto].copy()
        w = self._weights(s, weights)
        front["score"] = w[0] * _minmax(front.cost) + w[1] * _minmax(front.exp_time) + w[2] * front.risk
        ranked = front.sort_values("score").reset_index(drop=True)

        best = ranked.iloc[0]
        second = ranked.iloc[1] if len(ranked) > 1 else None
        table = df.merge(ranked[["route_id", "risk", "score"]], on="route_id", how="left")
        return dict(shipment_id=shipment_id, feasible=True, best=best.to_dict(),
                    second=None if second is None else second.to_dict(), table=table,
                    weights={k: round(float(v), 3) for k, v in zip(("cost", "time", "risk"), w)},
                    explanation=self.explain(s, best, second, feas, df, need))

    # ------------------------------------------------------------------ explanation
    @staticmethod
    def explain(s, best, second, feas, df, need) -> str:
        out = [f"Recommended {best.route_id} ({best.path}): ${best.cost:,.0f}, expected {best.exp_time:.1f} days, "
               f"{best.p_on_time:.0%} on-time probability."]
        if second is None:
            out.append("It is the only non-dominated route that satisfies the budget constraint.")
        else:
            dc, dt = second.cost - best.cost, second.exp_time - best.exp_time
            out.append(f"Compared with {second.route_id} ({second.path}): "
                       f"${abs(dc):,.0f} {'cheaper' if dc > 0 else 'more expensive'}, "
                       f"{abs(dt):.1f} days {'faster' if dt > 0 else 'slower'}, "
                       f"on-time probability {best.p_on_time:.0%} vs {second.p_on_time:.0%}.")
        for r in df[~df.feasible].itertuples():
            out.append(f"{r.route_id} ({r.path}) rejected: {r.reason}.")
        for r in feas[~feas.pareto].itertuples():
            out.append(f"{r.route_id} ({r.path}) dropped: dominated on cost, transit speed, or on-time delay risk.")
        return " ".join(out)