# Shipment Routing Dataset (graph formulation)

A synthetic dataset for **multi-objective route recommendation** in global logistics, structured as a graph problem. Routes are found by searching an explicit network, and each shipment carries explicit constraints (deadline, budget, required on-time probability).

| Graph concept | File | What it holds |
|---|---|---|
| Nodes | `hubs.csv` | Ports/hubs with congestion and geopolitical risk |
| Edges | `legs.csv` | Catalog of port-to-port legs (one row per `from_hub`, `to_hub`, `mode`) |
| Requests | `shipments.csv` | Origin, destination, cargo, weight, deadline, budget, on-time requirement |
| Candidate paths | `routes.csv` | For each shipment, 3-5 routes, each an ordered list of `leg_id`s |

`legs.csv` is independent of shipments and routes: the same leg has the same attributes wherever it appears. `routes.csv` stores **no cost, time or risk values**; compute them from the legs in each route. Routes carry no "type" labels.

---

## Quick start

```bash
pip install pandas numpy
python data_preparation.py --n_shipments 40 --seed 42 --out data
```

| Argument | Default | Description |
|---|---|---|
| `--n_shipments` | 40 | Number of shipments |
| `--seed` | 42 | Random seed (same seed gives an identical dataset) |
| `--out` | `data` | Output directory |

Default output (seed 42): 15 hubs, 105 legs, 40 shipments, 171 routes. Before writing, the script checks that every route is a connected, loop-free path from the shipment's origin to its destination with at most one air leg, and that every shipment has at least one route that meets its budget and on-time requirement.

```python
import numpy as np, pandas as pd
from data_preparation import delay_moments, simulate_route_times

legs      = pd.read_csv("data/legs.csv").set_index("leg_id")
shipments = pd.read_csv("data/shipments.csv").set_index("shipment_id")
routes    = pd.read_csv("data/routes.csv")

route = routes.iloc[0]
ship  = shipments.loc[route.shipment_id]
L     = legs.loc[route.leg_ids.split("|")]                 # the legs of this route

base_time = L.transit_days.sum()                           # time with no delays
cost      = (L.fixed_cost_usd + ship.weight_kg * L.cost_per_kg_usd).sum()
d_mean, d_std = delay_moments(L.delay_prob, L.delay_days_mean, L.delay_days_std)

samples   = simulate_route_times(base_time, L.delay_prob, L.delay_days_mean, L.delay_days_std,
                                 n=4000, np_rng=np.random.default_rng(0))
p_on_time = (samples <= ship.deadline_days).mean()          # P(total time <= deadline)
feasible  = cost <= ship.budget_usd and p_on_time >= ship.required_on_time_prob
```

---

## Files and schema

### `hubs.csv`: nodes

| Column | Description |
|---|---|
| `hub` | Port name |
| `lat`, `lon` | Coordinates |
| `congestion` | Congestion score, 0-1 (illustrative) |
| `geo_risk` | Geopolitical risk score, 0-1 (illustrative) |
| `is_origin`, `is_destination` | Whether the hub can be a shipment origin / destination |

### `legs.csv`: edges

One directed edge per `(from_hub, to_hub, mode)`. Destination hubs have no outgoing edges. Edges come from an explicit catalog of lanes (see "Network"), not from all pairs of hubs.

| Column | Description |
|---|---|
| `leg_id` | Unique ID, e.g. `L0023` |
| `from_hub`, `to_hub` | Endpoints |
| `mode` | `sea`, `rail` or `air` |
| `distance_km` | Lane distance (great-circle distance x a per-lane route factor) |
| `cost_per_kg_usd` | Variable cost per kg shipped on this leg |
| `fixed_cost_usd` | Fixed handling cost for this leg |
| `transit_days` | Travel time plus dwell at the arrival hub (no delays) |
| `delay_prob` | Probability this leg is delayed (depends on mode and **both** endpoint hubs) |
| `delay_days_mean` | Mean delay **given that** the leg is delayed |
| `delay_days_std` | Standard deviation of the delay **given that** the leg is delayed |
| `congestion_from`, `congestion_to` | Congestion of the endpoints (copied from `hubs.csv`) |
| `geo_risk_from`, `geo_risk_to` | Geopolitical risk of the endpoints (copied from `hubs.csv`) |

About a third of the legs are not used by any route; the catalog is the full graph, and routes are paths through it.

### `shipments.csv`: requests and constraints

| Column | Description |
|---|---|
| `shipment_id` | e.g. `S001` |
| `origin`, `destination` | Hubs |
| `cargo_type` | `general`, `perishable`, `high_value` or `urgent` |
| `weight_kg` | Shipment weight; scales variable cost |
| `deadline_days` | Latest acceptable arrival, **counting delays**: the route must satisfy `P(total time <= deadline_days) >= required_on_time_prob` |
| `budget_usd` | Hard spending limit: route cost must be at most this |
| `required_on_time_prob` | Required probability of arriving by the deadline (0.90 for general cargo, 0.95 otherwise) |
| `priority` | Shipper's preference: `cost`, `speed`, `reliability` or `balanced`. A soft preference, **not** a constraint |
| `n_routes` | Number of candidate routes (3-5) |

### `routes.csv`: candidate paths

| Column | Description |
|---|---|
| `route_id` | e.g. `S001-R2` |
| `shipment_id` | Parent shipment |
| `path` | Hubs in order, e.g. `Mumbai > Djibouti > Suez > Rotterdam` |
| `leg_ids` | Ordered `leg_id`s joined by `\|`, e.g. `L0016\|L0008` |
| `n_legs` | Number of legs |

---

## Computing route-level values

Join a route's `leg_ids` to `legs.csv`, then combine. The delay model per leg is: **with probability `p` the leg is delayed, and the delay is Gamma-distributed with mean `mu` and std `sd`; otherwise the delay is zero.** Legs are independent.

| Route value | From its legs (`p` = `delay_prob`, `mu` = `delay_days_mean`, `sd` = `delay_days_std`) |
|---|---|
| Total cost | `sum(fixed_cost_usd + weight_kg * cost_per_kg_usd)` |
| Base transit time (baseline, no delays) | `sum(transit_days)` |
| Probability of any delay | `1 - prod(1 - p)` |
| Expected total delay | `sum(p * mu)` |
| Std of total delay | `sqrt(sum(p * sd^2 + p * (1 - p) * mu^2))` |
| On-time probability | `P(sum(transit_days) + total delay <= deadline_days)` by Monte Carlo (see Quick start) |
| Worst congestion / risk | `max` over the `*_from` and `*_to` columns |

For one leg, `E[D] = p * mu` and `Var[D] = p * sd^2 + p * (1 - p) * mu^2`. The second term in the variance comes from uncertainty about *whether* a delay happens; the first from uncertainty in its *duration*. Summing the leg variances gives the route variance (independent legs). `delay_moments` and `simulate_route_times` in `data_preparation.py` implement these.

A route can pass the base-time check (`sum(transit_days) <= deadline_days`) and still fail the on-time-probability requirement, so use the probability for any constraint that matters and keep base time as a baseline.

Cost scales with `weight_kg`, so normalise cost *within each shipment's set of routes* (for example min-max) rather than comparing raw dollars across shipments.

---

## How the data is generated

### Hubs

| Hub | Congestion | Geo risk | Role |
|---|---|---|---|
| Shanghai | 0.70 | 0.15 | Origin |
| Singapore | 0.55 | 0.05 | Origin |
| Mumbai | 0.50 | 0.10 | Origin |
| Jebel Ali | 0.35 | 0.25 | Origin |
| Colombo | 0.40 | 0.10 | Transit |
| Salalah | 0.25 | 0.30 | Transit |
| Djibouti | 0.30 | 0.75 | Transit (Red Sea, high risk) |
| Suez | 0.60 | 0.65 | Transit (chokepoint, high risk) |
| Cape Town | 0.30 | 0.05 | Transit (safe, but a long detour) |
| Piraeus | 0.45 | 0.20 | Transit |
| Rotterdam | 0.65 | 0.05 | Destination |
| Hamburg | 0.55 | 0.05 | Destination |
| Felixstowe | 0.45 | 0.05 | Destination |
| New York | 0.60 | 0.05 | Destination |
| Los Angeles | 0.75 | 0.05 | Destination |

### Network (explicit lane catalog)

This is an **abstract logistics network**: a hand-written catalog of plausible connections, not a navigation model. Distances are approximate.

- **Sea lanes** (31 bidirectional lanes) connect neighbouring hubs: Asia to the Indian Ocean, the Red Sea / Suez / Mediterranean corridor, Atlantic lanes to Europe and New York, the **Cape of Good Hope alternative** (Singapore, Colombo and Mumbai to Cape Town, then on to Europe and New York), and the transpacific lanes (Shanghai and Singapore to Los Angeles). Each lane has a route factor (lane distance / great-circle distance): about 1.05-1.15 for open ocean, higher for canal, coastal and around-continent lanes.
- **Rail corridors** (4): Shanghai to Rotterdam and Hamburg, and Piraeus to Rotterdam and Hamburg.
- **Air links:** between any two air-capable hubs (Shanghai, Singapore, Mumbai, Jebel Ali, Colombo, Piraeus and the five destinations) when the hop is at least 2,500 km. Salalah, Djibouti, Suez and Cape Town are sea-only.
- Destination hubs are sinks (no outgoing edges).

### Modes

| Mode | Speed (km/day) | Cost (USD per kg-km) | Fixed cost per leg (USD) | Dwell (days) | Base delay prob | Base delay (days) |
|---|---|---|---|---|---|---|
| Sea | 700 | 0.00004 | 250 | 1.5 | 0.12 | 3.0 |
| Rail | 600 | 0.00012 | 150 | 1.0 | 0.08 | 1.5 |
| Air | 12,000 | 0.00060 | 400 | 0.4 | 0.05 | 0.7 |

### Leg attributes

```
distance         = great_circle(a, b) * lane_route_factor
lane_congestion  = (congestion[a] + congestion[b]) / 2      # both endpoints count
lane_geo_risk    = (geo_risk[a]   + geo_risk[b])   / 2
shock            = 1 for ~10% of legs (random, not exposed in the data), else 0
cost_per_kg      = distance * mode_rate * LogNormal(0, 0.20)
fixed_cost       = mode_fixed_cost * LogNormal(0, 0.15)
transit_days     = distance / mode_speed * Uniform(0.9, 1.25) + dwell * (1 + congestion[b])
delay_prob       = clip(0.5 * mode_base_p + 0.15 * lane_congestion + 0.12 * lane_geo_risk
                        + 0.15 * shock + N(0, 0.02), 0.01, 0.95)
delay_days_mean  = mode_base_delay * (1 + lane_congestion + lane_geo_risk) * (1.5 if shock else 1)
delay_days_std   = delay_days_mean * Uniform(0.3, 0.7)
```

The lane-specific noise and the hidden disruption shocks mean that cost, time and risk are not simple functions of the hub scores: a lane through a safe hub can still be delay-prone, and a cheap lane can be unreliable.

### Route search (no archetype labels)

For each shipment, the generator searches the graph for simple (loop-free) paths from origin to destination and expands each hub path into mode combinations.

- At most 5 legs and at most 1 air leg per route.
- **Detour limit:** at most 1.5x the shortest sea/rail path, except routes via Cape Town, which may be up to 2.0x (so the Cape alternative to Suez is available). If a corridor has fewer than 6 candidates (for example Shanghai to Los Angeles), the larger limit applies.
- **No zig-zagging:** each intermediate hub must be farther from the origin than the previous one, and routes with an air leg may not move away from the destination between hubs.

From the candidates, the shipment gets 3-5 routes: the **cheapest**, the **fastest**, and the **most reliable non-air** route (lowest probability of any delay), so that real trade-offs are always present, plus random extras (mostly ground modes). Which of these is best for a shipment depends on its constraints and priority, and no label records it.

### Cargo types

| Cargo type | Share | Weight range (kg) | Typical priority | Required on-time prob |
|---|---|---|---|---|
| general | 40% | 2,000-20,000 | cost, balanced | 0.90 |
| perishable | 20% | 500-8,000 | speed, reliability | 0.95 |
| high_value | 25% | 200-3,000 | reliability, balanced | 0.95 |
| urgent | 15% | 100-2,000 | speed | 0.95 |

### Deadlines and budgets

Constraints are generated so that every shipment has at least one feasible route, while the others may fail.

1. **Deadline:** a reference route is chosen (from the faster options for urgent and perishable cargo, the slower half for general cargo, any option for high-value). The deadline is that route's total-time quantile at `required_on_time_prob` (from 4,000 Monte Carlo runs), multiplied by 1.05-1.25 for a *tight* shipment (30%) or 1.25-2.2 for a *loose* one.
2. **Budget:** the cost of the cheapest option that meets the on-time requirement, multiplied by 1.10-1.40 (tight) or 1.4-3.0 (loose).

With seed 42, about half the shipments have two or more feasible routes. The rest, mostly urgent and high-value cargo, are tightly constrained and often have a single feasible option.

---

## Evaluating a recommender on this data

- Compare recommendations against the **explicit constraints** (`deadline_days` with `required_on_time_prob`, and `budget_usd`), and use `priority` and `cargo_type` only to weight preferences among feasible routes.
- Because routes carry no labels, a recommender cannot succeed by recognising a route "type". The cheapest, fastest and most reliable routes are present for every shipment, but which one wins varies by shipment.
- Run **sensitivity analysis** by changing scoring weights, the required on-time probability, or regenerating with different seeds.
- Within a shipment, cheaper routes tend to be slower and riskier. That follows from the physics of the modes and the graph, but the correlation is imperfect (with seed 42 the cheapest route is never the fastest, and it is the most reliable in about 5% of shipments), and about a quarter of routes are dominated on cost, time and delay probability.

---

## Assumptions and limitations

- All costs, speeds, delay rates, risk scores and lane distances are **illustrative**, not real market data. The network is abstract and not a navigation model.
- Leg delays are **independent**; real delays are correlated (for example a port strike or Red Sea disruption hits many routes). Delay duration given a delay is assumed Gamma-distributed.
- Weight affects **cost only**, not transit time or delay probability.
- Hub risk and congestion are **static**; there is no seasonality or time dimension.
- Delay depends on the two endpoint hubs and the mode, plus random lane noise and hidden shocks, not on named real-world lane characteristics.
- Some multimodal routes are geographically odd (for example sailing west before flying to Los Angeles). The search rules reduce this but do not eliminate it.
- The generator always includes the cheapest, fastest and most reliable non-air candidates in each shipment's option set, so every shipment has a visible trade-off by construction.
- Urgent and high-value shipments are often tightly constrained, so their decisions are close to forced; evaluate the more open general-cargo shipments if you want to test trade-off handling.