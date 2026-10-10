"""
data_preparation.py
Builds the shipment-routing dataset as a graph problem.

    hubs.csv       NODES      ports/hubs with congestion and geopolitical risk
    legs.csv       EDGES      catalog of port-to-port legs, one row per (from_hub, to_hub, mode);
                              independent of any shipment or route
    routes.csv     PATHS      route catalog: 3-5 candidate routes per (origin, destination) pair, each an
                              ordered list of leg_ids; independent of any shipment
    shipments.csv  REQUESTS   origin, destination, cargo, weight, deadline, budget, on-time requirement.
                              A shipment's candidate routes are the routes with the same origin and destination.

Routes are found by searching the graph (no route "archetype" labels). Route cost / time / delay are NOT
stored: compute them from a route's legs (see README). Cost depends on the shipment's weight, so the
"cheapest" routes kept in the catalog are chosen at several reference weights. The helpers `delay_moments`
and `simulate_route_times` implement the delay model used to set deadlines.

Network: an explicit catalog of plausible sea lanes, rail corridors and air links (an ABSTRACT logistics
network, not a navigation model). Delay model per leg: delayed with probability p; if delayed, the delay is
Gamma-distributed with mean `delay_days_mean` and std `delay_days_std`; otherwise zero.

Run:  python data_preparation.py --n_shipments 40 --seed 42 --out data
"""
import argparse, heapq, itertools, math, os, random
import numpy as np
import pandas as pd

# ------------------------------------------------------------------ nodes
# name: (lat, lon, congestion 0-1, geopolitical_risk 0-1, is_origin, is_destination)
HUBS = {
    "Shanghai":     (31.2, 121.5, 0.70, 0.15, True,  False),
    "Singapore":    (1.3,  103.8, 0.55, 0.05, True,  False),
    "Mumbai":       (18.9,  72.9, 0.50, 0.10, True,  False),
    "Jebel Ali":    (25.0,  55.1, 0.35, 0.25, True,  False),
    "Colombo":      (6.9,   79.8, 0.40, 0.10, False, False),
    "Salalah":      (17.0,  54.1, 0.25, 0.30, False, False),
    "Djibouti":     (11.6,  43.1, 0.30, 0.75, False, False),   # Red Sea, high risk
    "Suez":         (30.0,  32.5, 0.60, 0.65, False, False),   # chokepoint, high risk
    "Cape Town":    (-33.9, 18.4, 0.30, 0.05, False, False),
    "Piraeus":      (37.9,  23.6, 0.45, 0.20, False, False),
    "Rotterdam":    (51.9,   4.5, 0.65, 0.05, False, True),
    "Hamburg":      (53.5,  10.0, 0.55, 0.05, False, True),
    "Felixstowe":   (51.9,   1.3, 0.45, 0.05, False, True),
    "New York":     (40.7, -74.0, 0.60, 0.05, False, True),
    "Los Angeles":  (33.7, -118.3, 0.75, 0.05, False, True),
}
# short codes used to name routes in the catalog, e.g. SHA-RTM-R2
CODE = {"Shanghai": "SHA", "Singapore": "SIN", "Mumbai": "BOM", "Jebel Ali": "JEA", "Rotterdam": "RTM",
        "Hamburg": "HAM", "Felixstowe": "FXT", "New York": "NYC", "Los Angeles": "LAX"}

# ------------------------------------------------------------------ modes
# speed km/day, cost USD per kg-km, base delay prob, base delay days
MODES = {
    "sea":  dict(speed=700,   rate=0.00004, p0=0.12, d0=3.0),
    "rail": dict(speed=600,   rate=0.00012, p0=0.08, d0=1.5),
    "air":  dict(speed=12000, rate=0.00060, p0=0.05, d0=0.7),
}
HANDLING = {"sea": 250, "rail": 150, "air": 400}   # fixed USD per leg
DWELL = {"sea": 1.5, "rail": 1.0, "air": 0.4}      # days at the arrival hub

# ------------------------------------------------------------------ explicit network catalog
# (hub_a, hub_b, route_factor): factor = lane distance / great-circle distance (open ocean ~1.05-1.15,
# canals / coastal / around-continent lanes higher). Lanes are bidirectional; destination hubs are sinks.
SEA_LANES = [
    ("Shanghai", "Singapore", 1.15), ("Singapore", "Colombo", 1.10), ("Singapore", "Mumbai", 1.10),
    ("Colombo", "Mumbai", 1.20), ("Colombo", "Salalah", 1.10), ("Colombo", "Jebel Ali", 1.15),
    ("Mumbai", "Salalah", 1.10), ("Mumbai", "Jebel Ali", 1.20), ("Mumbai", "Djibouti", 1.10),
    ("Salalah", "Jebel Ali", 1.30), ("Salalah", "Djibouti", 1.20), ("Jebel Ali", "Djibouti", 1.35),
    ("Djibouti", "Suez", 1.15), ("Suez", "Piraeus", 1.15),                                   # Red Sea / Suez / Med
    ("Suez", "Rotterdam", 1.45), ("Suez", "Hamburg", 1.50), ("Suez", "Felixstowe", 1.45), ("Suez", "New York", 1.25),
    ("Piraeus", "Rotterdam", 1.55), ("Piraeus", "Hamburg", 1.55), ("Piraeus", "Felixstowe", 1.50),
    ("Piraeus", "New York", 1.15),
    ("Singapore", "Cape Town", 1.05), ("Colombo", "Cape Town", 1.05), ("Mumbai", "Cape Town", 1.05),  # Cape route
    ("Cape Town", "Rotterdam", 1.05), ("Cape Town", "Hamburg", 1.08), ("Cape Town", "Felixstowe", 1.06),
    ("Cape Town", "New York", 1.05),
    ("Shanghai", "Los Angeles", 1.10), ("Singapore", "Los Angeles", 1.10),                    # transpacific
]
RAIL_LANES = [  # a few illustrative land-bridge corridors (China-Europe, Balkans-Central Europe)
    ("Shanghai", "Rotterdam", 1.35), ("Shanghai", "Hamburg", 1.30),
    ("Piraeus", "Rotterdam", 1.30), ("Piraeus", "Hamburg", 1.25),
]
AIR_HUBS = ["Shanghai", "Singapore", "Mumbai", "Jebel Ali", "Colombo", "Piraeus",
            "Rotterdam", "Hamburg", "Felixstowe", "New York", "Los Angeles"]   # other ports are sea-only
AIR_FACTOR = 1.05

# ------------------------------------------------------------------ route search / constraint settings
MAX_LEGS = 5            # legs per route
DETOUR_RATIO = 1.5      # a route may be at most this much longer than the shortest sea/rail path ...
MAX_DETOUR_RATIO = 2.0  # ... except routes via Cape Town (the Suez alternative), which may be up to this much longer
DETOUR_HUBS = {"Cape Town"}
MAX_AIR_LEGS = 1        # at most one air leg per route
MIN_CANDIDATES = 6      # if an origin-destination pair has fewer candidate routes, allow the larger detour for everyone
MIN_AIR_KM = 2500       # air legs only exist for longer hops
# (routes containing an air leg may not move away from the destination between hubs)
N_SIM = 4000            # Monte Carlo samples
REF_WEIGHTS = (200, 2000, 20000)   # kg; the catalog keeps the cheapest route at each of these shipment weights
                                   # (cost = fixed + weight * per_kg, so the cheapest route depends on weight)
MIN_ROUTES, MAX_ROUTES = 3, 5      # routes per origin-destination pair

CARGO = {  # share of shipments, weight range (kg), typical shipper priority, required on-time probability
    "general":    dict(share=0.40, weight=(2000, 20000), priority=["cost", "balanced"],         on_time=0.90),
    "perishable": dict(share=0.20, weight=(500, 8000),   priority=["speed", "reliability"],     on_time=0.95),
    "high_value": dict(share=0.25, weight=(200, 3000),   priority=["reliability", "balanced"],  on_time=0.95),
    "urgent":     dict(share=0.15, weight=(100, 2000),   priority=["speed"],                    on_time=0.95),
}


def haversine(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(h))


def gc_km(a, b):
    return haversine(HUBS[a][:2], HUBS[b][:2])


# ------------------------------------------------------------------ delay model helpers
def delay_moments(p, mu, sd):
    """Mean and std of total route delay. Leg e: delayed w.p. p_e, then delay has mean mu_e and std sd_e.
    E[D_e] = p*mu ; Var[D_e] = p*sd^2 + p*(1-p)*mu^2 ; legs independent."""
    p, mu, sd = map(np.asarray, (p, mu, sd))
    return float((p * mu).sum()), float(np.sqrt((p * sd ** 2 + p * (1 - p) * mu ** 2).sum()))


def simulate_route_times(base_days, p, mu, sd, n, np_rng):
    """Monte Carlo samples of total route time = base transit + sum of leg delays (Gamma if delayed)."""
    p, mu, sd = map(np.asarray, (p, mu, sd))
    shape, scale = (mu / sd) ** 2, sd ** 2 / mu
    hit = np_rng.random((n, len(p))) < p
    delay = np_rng.gamma(shape, scale, size=(n, len(p)))
    return base_days + (hit * delay).sum(axis=1)


# ------------------------------------------------------------------ nodes
def build_hubs():
    return pd.DataFrame([dict(hub=h, lat=v[0], lon=v[1], congestion=v[2], geo_risk=v[3],
                              is_origin=v[4], is_destination=v[5]) for h, v in HUBS.items()])


# ------------------------------------------------------------------ edges
def lane_catalog():
    lanes = []   # (from, to, mode, factor)
    for catalog, mode in ((SEA_LANES, "sea"), (RAIL_LANES, "rail")):
        for a, b, f in catalog:
            for x, y in ((a, b), (b, a)):
                if not HUBS[x][5]:                     # destination hubs have no outgoing edges
                    lanes.append((x, y, mode, f))
    for a in AIR_HUBS:
        if HUBS[a][5]:
            continue
        for b in AIR_HUBS:
            if a != b and gc_km(a, b) * AIR_FACTOR >= MIN_AIR_KM:
                lanes.append((a, b, "air", AIR_FACTOR))
    return lanes


def build_legs(rng, np_rng):
    """Edge attributes are fixed per edge. Hub scores drive delay, but each lane also gets idiosyncratic noise,
    and ~10% of legs carry an (unlabelled) disruption shock, so cost / time / risk are not perfectly aligned."""
    rows = []
    for a, b, mode, factor in lane_catalog():
        va, vb, m = HUBS[a], HUBS[b], MODES[mode]
        dist = gc_km(a, b) * factor
        cong, geo = (va[2] + vb[2]) / 2, (va[3] + vb[3]) / 2            # both endpoints count
        shock = rng.random() < 0.10
        p = float(np.clip(0.5 * m["p0"] + 0.15 * cong + 0.12 * geo + 0.15 * shock + np_rng.normal(0, 0.02), 0.01, 0.95))
        d_mean = m["d0"] * (1 + cong + geo) * (1.5 if shock else 1.0)
        rows.append(dict(
            from_hub=a, to_hub=b, mode=mode, distance_km=round(dist),
            cost_per_kg_usd=round(dist * m["rate"] * np_rng.lognormal(0, 0.20), 5),
            fixed_cost_usd=round(HANDLING[mode] * np_rng.lognormal(0, 0.15), 2),
            transit_days=round(dist / m["speed"] * rng.uniform(0.9, 1.25) + DWELL[mode] * (1 + vb[2]), 2),
            delay_prob=round(p, 3), delay_days_mean=round(d_mean, 2),
            delay_days_std=round(d_mean * rng.uniform(0.3, 0.7), 2),
            congestion_from=va[2], congestion_to=vb[2], geo_risk_from=va[3], geo_risk_to=vb[3]))
    legs = pd.DataFrame(rows)
    legs.insert(0, "leg_id", [f"L{i+1:04d}" for i in range(len(legs))])
    return legs


# ------------------------------------------------------------------ route search
class Graph:
    def __init__(self, legs):
        self.legs = legs
        self.mode = legs["mode"].to_numpy()
        self.pair_edges, self.adj = {}, {}
        for i, (a, b) in enumerate(zip(legs.from_hub, legs.to_hub)):
            self.pair_edges.setdefault((a, b), []).append(i)
            self.adj.setdefault(a, [])
            if b not in self.adj[a]:
                self.adj[a].append(b)
        dist = legs.distance_km.to_numpy()
        # hop distance for pruning: shortest non-air edge (air only if the pair has no sea/rail edge)
        self.hop, self.hop_ground, self.adj_ground = {}, {}, {}
        for (a, b), idx in self.pair_edges.items():
            ground = [dist[i] for i in idx if self.mode[i] != "air"]
            self.hop[(a, b)] = min(ground) if ground else min(dist[i] for i in idx)
            if ground:
                self.hop_ground[(a, b)] = min(ground)
                self.adj_ground.setdefault(a, []).append(b)
        self._cache = {}

    def shortest(self, o, d):
        """Shortest sea/rail distance o -> d (the detour baseline); falls back to all modes if none exists."""
        r = self._dijkstra(o, d, self.adj_ground, self.hop_ground)
        return r if r is not None else self._dijkstra(o, d, self.adj, self.hop)

    @staticmethod
    def _dijkstra(o, d, adj, hop):
        best, heap = {o: 0.0}, [(0.0, o)]
        while heap:
            dd, u = heapq.heappop(heap)
            if u == d:
                return dd
            if dd > best.get(u, 1e18):
                continue
            for v in adj.get(u, []):
                nd = dd + hop[(u, v)]
                if nd < best.get(v, 1e18):
                    best[v] = nd
                    heapq.heappush(heap, (nd, v))
        return None

    def candidates(self, o, d):
        """Simple, loop-free routes from o to d within the leg and detour limits, grouped by hub path."""
        if (o, d) in self._cache:
            return self._cache[(o, d)]
        short = self.shortest(o, d)
        limit = MAX_DETOUR_RATIO * short
        hub_paths = []

        def dfs(node, path, dist):
            if node == d:
                hub_paths.append((list(path), dist))
                return
            if len(path) - 1 >= MAX_LEGS:
                return
            for nxt in self.adj.get(node, []):
                if nxt in path:
                    continue
                nd = dist + self.hop[(node, nxt)]
                rest = 0.0 if nxt == d else gc_km(nxt, d)            # straight-line lower bound for pruning
                if nd + rest > limit:
                    continue
                path.append(nxt)
                dfs(nxt, path, nd)
                path.pop()

        dfs(o, [o], 0.0)
        def expand(ratio):
            out = {}   # hub path -> list of mode combinations (each a tuple of leg indices)
            for hp, dist in hub_paths:
                if dist > ratio * short and not (DETOUR_HUBS & set(hp)):
                    continue
                mid = [gc_km(o, h) for h in hp[1:-1]]
                if any(mid[k + 1] < mid[k] for k in range(len(mid) - 1)):   # no zig-zagging back toward the origin
                    continue
                approach = all(gc_km(hp[k + 1], d) <= gc_km(hp[k], d) * 1.05 for k in range(len(hp) - 1))
                combos = [c for c in itertools.product(*[self.pair_edges[(hp[k], hp[k + 1])] for k in range(len(hp) - 1)])
                          if (n_air := sum(self.mode[i] == "air" for i in c)) <= MAX_AIR_LEGS and (n_air == 0 or approach)]
                if combos:
                    out[tuple(hp)] = combos
            return out

        by_path = expand(DETOUR_RATIO)
        if sum(len(c) for c in by_path.values()) < MIN_CANDIDATES:   # sparse corridor (e.g. transpacific): allow more detour
            by_path = expand(MAX_DETOUR_RATIO)
        self._cache[(o, d)] = by_path
        return by_path


# ------------------------------------------------------------------ route catalog (independent of shipments)
def build_route_catalog(G, legs, rng, np_rng):
    """Routes for every (origin, destination) pair, chosen without reference to any shipment.

    Each pair keeps: the cheapest route at each reference weight (cost depends on weight), the fastest route and the
    most reliable non-air route (so real trade-offs exist), plus random extras up to a target of 3-5 routes.
    No archetype labels are stored. Returns the routes table and, per route, the leg indices, base time and
    Monte Carlo total-time samples (simulated once per route and reused when constraints are set)."""
    fixed, per_kg = legs.fixed_cost_usd.to_numpy(), legs.cost_per_kg_usd.to_numpy()
    tt, P = legs.transit_days.to_numpy(), legs.delay_prob.to_numpy()
    MU, SD = legs.delay_days_mean.to_numpy(), legs.delay_days_std.to_numpy()
    origins = [h for h, v in HUBS.items() if v[4]]
    dests = [h for h, v in HUBS.items() if v[5]]
    rows, data = [], {}

    for o in origins:
        for d in dests:
            by_path = G.candidates(o, d)
            pool = [combo for combos in by_path.values() for combo in combos]
            assert pool, f"no route from {o} to {d}"
            idx = [list(r) for r in pool]
            time = np.array([tt[r].sum() for r in idx])
            p_any = np.array([1 - np.prod(1 - P[r]) for r in idx])
            has_air = np.array([any(G.mode[x] == "air" for x in r) for r in idx])
            ground = np.where(~has_air, p_any, np.inf)       # most reliable option that is not simply "fly it"

            picks = []
            def add(j):
                if pool[j] not in picks:
                    picks.append(pool[j])
            for w in REF_WEIGHTS:
                add(int(np.array([fixed[r].sum() + w * per_kg[r].sum() for r in idx]).argmin()))
            add(int(time.argmin()))
            add(int(ground.argmin()) if (~has_air).any() else int(p_any.argmin()))
            options = picks[:MAX_ROUTES]

            target, keys, tries = rng.randint(MIN_ROUTES, MAX_ROUTES), list(by_path), 0
            while len(options) < target and tries < 500:       # extras: random hub path, then mostly ground modes on it
                tries += 1
                combos = by_path[rng.choice(keys)]
                no_air = [x for x in combos if not any(G.mode[i] == "air" for i in x)]
                combo = rng.choice(no_air if no_air and rng.random() < 0.8 else combos)
                if combo not in options:
                    options.append(combo)
            rng.shuffle(options)                                # route numbering carries no information

            for k, route in enumerate(options):
                r = list(route)
                rid = f"{CODE[o]}-{CODE[d]}-R{k+1}"
                rows.append(dict(route_id=rid, origin=o, destination=d,
                                 path=" > ".join([legs.from_hub[route[0]]] + [legs.to_hub[x] for x in route]),
                                 leg_ids="|".join(legs.leg_id[x] for x in route), n_legs=len(route)))
                base = float(tt[r].sum())
                data[rid] = dict(idx=r, od=(o, d), time=base,
                                 sims=simulate_route_times(base, P[r], MU[r], SD[r], N_SIM, np_rng))
    return pd.DataFrame(rows), data


# ------------------------------------------------------------------ main
def main(n_shipments, seed, out):
    rng, np_rng = random.Random(seed), np.random.default_rng(seed)
    hubs, legs = build_hubs(), build_legs(rng, np_rng)
    G = Graph(legs)
    fixed, per_kg = legs.fixed_cost_usd.to_numpy(), legs.cost_per_kg_usd.to_numpy()

    routes, rdata = build_route_catalog(G, legs, rng, np_rng)
    by_od = {}
    for rid, v in rdata.items():
        by_od.setdefault(v["od"], []).append(rid)

    origins = [h for h, v in HUBS.items() if v[4]]
    dests = [h for h, v in HUBS.items() if v[5]]
    shipments = []

    for i in range(n_shipments):
        sid = f"S{i+1:03d}"
        o, d = rng.choice(origins), rng.choice(dests)
        cargo = rng.choices(list(CARGO), weights=[CARGO[k]["share"] for k in CARGO])[0]
        c = CARGO[cargo]
        weight = rng.randint(*c["weight"])

        # --- constraints, set against the (shipment-independent) routes of this origin-destination pair.
        # The deadline is anchored to one option (faster options for urgent/perishable, slower half for general);
        # the budget is then anchored to the cheapest option that meets the on-time requirement, so every shipment
        # is feasible but cost and time can conflict.
        cand = by_od[(o, d)]
        opt_time = [rdata[r]["time"] for r in cand]
        opt_cost = [float(fixed[rdata[r]["idx"]].sum() + weight * per_kg[rdata[r]["idx"]].sum()) for r in cand]
        order = sorted(range(len(cand)), key=lambda k: opt_time[k])
        half = max(1, len(order) // 2)
        ref_pool = (order[:half] if cargo == "urgent" else order[:max(1, len(order) * 2 // 3)] if cargo == "perishable"
                    else order[-half:] if cargo == "general" else order)
        ref = cand[rng.choice(ref_pool)]
        tight = rng.random() < 0.3                                     # some shipments are tightly constrained, others loose
        deadline = math.ceil(np.quantile(rdata[ref]["sims"], c["on_time"])
                             * (rng.uniform(1.05, 1.25) if tight else rng.uniform(1.25, 2.2)) * 10) / 10
        on_time = [k for k, r in enumerate(cand) if (rdata[r]["sims"] <= deadline).mean() >= c["on_time"]]
        budget = round(min(opt_cost[k] for k in on_time) * (rng.uniform(1.10, 1.40) if tight else rng.uniform(1.4, 3.0)))

        shipments.append(dict(shipment_id=sid, origin=o, destination=d, cargo_type=cargo, weight_kg=weight,
                              deadline_days=deadline, budget_usd=budget, required_on_time_prob=c["on_time"],
                              priority=rng.choice(c["priority"])))

    shipments = pd.DataFrame(shipments)
    validate(hubs, legs, shipments, routes, np.random.default_rng(seed + 1))

    os.makedirs(out, exist_ok=True)
    for name, df in (("hubs", hubs), ("legs", legs), ("shipments", shipments), ("routes", routes)):
        df.to_csv(f"{out}/{name}.csv", index=False)
    print(f"{len(hubs)} hubs, {len(legs)} legs (edges), {len(routes)} routes over {len(by_od)} origin-destination "
          f"pairs, {len(shipments)} shipments -> {out}/")


def validate(hubs, legs, shipments, routes, np_rng):
    """Integrity checks (structure) and a feasibility check (every shipment has an option meeting all constraints)."""
    leg = legs.set_index("leg_id")
    assert legs.leg_id.is_unique and routes.route_id.is_unique
    assert "shipment_id" not in routes.columns, "routes must not depend on shipments"
    assert set(legs.from_hub) | set(legs.to_hub) <= set(hubs.hub)
    assert (legs.delay_days_mean > 0).all() and (legs.delay_days_std > 0).all()
    n_origin, n_dest = int(hubs.is_origin.sum()), int(hubs.is_destination.sum())
    per_od = routes.groupby(["origin", "destination"]).size()
    assert len(per_od) == n_origin * n_dest, "an origin-destination pair has no routes"
    assert per_od.between(MIN_ROUTES, MAX_ROUTES).all(), "an origin-destination pair has too few or too many routes"
    route_legs = {}
    for r in routes.itertuples():
        ids = r.leg_ids.split("|")
        L = leg.loc[ids]
        nodes = [L.from_hub.iloc[0]] + list(L.to_hub)
        assert all(L.to_hub.iloc[k] == L.from_hub.iloc[k + 1] for k in range(len(ids) - 1)), f"{r.route_id}: not connected"
        assert len(set(nodes)) == len(nodes), f"{r.route_id}: revisits a hub"
        assert (L["mode"] == "air").sum() <= MAX_AIR_LEGS
        assert nodes[0] == r.origin and nodes[-1] == r.destination, f"{r.route_id}: endpoints do not match"
        route_legs[r.route_id] = L
    for s in shipments.itertuples():
        cands = routes[(routes.origin == s.origin) & (routes.destination == s.destination)]
        assert len(cands) >= MIN_ROUTES, f"{s.shipment_id}: too few route options"
        ok = False
        for r in cands.itertuples():
            L = route_legs[r.route_id]
            cost = (L.fixed_cost_usd + s.weight_kg * L.cost_per_kg_usd).sum()
            if cost <= s.budget_usd:
                sims = simulate_route_times(L.transit_days.sum(), L.delay_prob, L.delay_days_mean, L.delay_days_std, N_SIM, np_rng)
                if (sims <= s.deadline_days).mean() >= s.required_on_time_prob - 0.02:   # Monte Carlo tolerance
                    ok = True
                    break
        assert ok, f"no feasible route for {s.shipment_id}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_shipments", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    main(a.n_shipments, a.seed, a.out)