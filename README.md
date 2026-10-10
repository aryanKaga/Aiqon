# Aiqon — Shipping Route Recommendation System

> **Multi-objective Pareto optimisation** across **cost**, **transit time**, and **on-time reliability** for global cargo shipments — classical decision analysis, no ML training required.

---

## Table of Contents

1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [Dependencies](#dependencies)
4. [Installation & Setup](#installation--setup)
   - [Quickstart (venv)](#quickstart-venv)
   - [Quickstart (conda)](#quickstart-conda)
5. [Using the CLI (Python Commands)](#using-the-cli-python-commands)
   - [recommend](#1-recommend-a-route)
   - [evaluate](#2-evaluate-all-shipments)
   - [sensitivity](#3-sensitivity-analysis)
   - [server](#4-launch-web-server)
6. [Using the Web Interface](#using-the-web-interface)
7. [Optional: LLM Executive Summary](#optional-llm-executive-summary)
8. [Output Files](#output-files)
9. [Methodology](#methodology)
10. [Design Decisions](#design-decisions)
11. [Scoring Formula](#scoring-formula)

---

## Overview

Aiqon recommends the optimal shipping route for a cargo shipment by:

1. **Filtering** routes that violate hard constraints (budget, on-time probability)
2. **Running 20,000 Monte Carlo simulations** per route to model stochastic transit delays
3. **Applying a Pareto filter** — only non-dominated routes on `(cost, expected_time, risk)` proceed
4. **Weighted scoring** within the Pareto frontier using the shipment's stated priority (`cost`, `speed`, `reliability`, or `balanced`)

The system covers **15 global hubs**, **105 directed legs** (sea / rail / air), and **76 pre-catalogued routes** across 20 origin-destination pairs.

---

## Repository Structure

```
Aiqon/
├── Data/
│   └── data/
│       ├── hubs.csv          # 15 port nodes with congestion & geo-risk scores
│       ├── legs.csv          # 105 directed legs (sea/rail/air) with delay model params
│       ├── routes.csv        # 76 routes across 20 O-D pairs
│       └── shipments.csv     # 40 shipment requests with hard constraints
│
├── Recommendation_Engine/
│   ├── __init__.py
│   ├── DataLoader.py         # CSV loader, graph builder, and lookup helpers
│   └── engine.py             # RecommendationEngine: MC simulation, Pareto filter, scorer
│
├── evaluate.py               # Baseline comparison, regret, Pareto membership, tagging
├── sensitivity.py            # Weight / on-time / delay / normalisation perturbations
├── llm_summary.py            # Optional AI executive summary (Gemini / OpenAI / rule-based)
├── server.py                 # Flask REST API + web dashboard backend
│
├── templates/
│   └── index.html            # Single-page web dashboard (Jinja2 template)
├── static/
│   ├── style.css             # Dark-mode glassmorphism design system
│   └── app.js                # Client-side dashboard controller (Chart.js)
│
├── main.py                   # CLI entry-point (recommend / evaluate / sensitivity / server)
├── requirements.txt          # Python dependencies
├── results/                  # Auto-created; holds all output CSVs and summaries
│   ├── results_table.csv
│   ├── summary.txt
│   ├── sensitivity_table.csv
│   └── sensitivity_summary.txt
└── README.md
```

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `Python` | 3.9 + | Runtime |
| `numpy` | ≥ 1.24 | Monte Carlo simulation arrays |
| `pandas` | ≥ 2.0 | Data loading, DataFrame operations |
| `flask` | ≥ 3.0 | Web server & REST API |
| `google-generativeai` | ≥ 0.7 | **Optional** — Gemini LLM summaries |
| `openai` | ≥ 1.0 | **Optional** — OpenAI GPT summaries |

> **No ML frameworks** (PyTorch, TensorFlow, scikit-learn) are required.  
> The `google-generativeai` and `openai` packages are only needed if you want live LLM summaries; the system works fully offline without them.

---

## Installation & Setup

### Quickstart (venv)

```bash
# 1. Clone or download the repository
git clone <repo-url>
cd Aiqon

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate it
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 4. Install required dependencies
pip install -r requirements.txt
```

That's it — no further configuration is needed for the core CLI and web interface.

---

### Quickstart (conda)

```bash
# 1. Clone or download the repository
git clone <repo-url>
cd Aiqon

# 2. Create and activate conda environment
conda create -n aiqon python=3.11
conda activate aiqon

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Using the CLI (Python Commands)

All commands are run from the project root (`Aiqon/`) with the virtual environment active.

---

### 1. Recommend a Route

Find the optimal shipping route for a single shipment and print a full candidate comparison table plus plain-language explanation.

```bash
python main.py recommend --shipment S001
```

**Override priority weights** (cost / time / risk — automatically normalised):

```bash
python main.py recommend --shipment S001 --weights 0.6 0.2 0.2
```

**All options:**

```
usage: main.py recommend [-h] --shipment ID [--weights COST TIME RISK] [--seed SEED]

  --shipment ID          Shipment ID to evaluate (e.g. S001 through S040)
  --weights COST TIME RISK
                         Three floats overriding priority weights. Will be
                         normalised so they sum to 1.0. Example: 0.5 0.3 0.2
  --seed SEED            RNG seed for reproducible Monte Carlo results (default: 42)
```

**Example output:**

```
-- Recommendation for S001 ------------------------------------------
  Origin      : Mumbai
  Destination : Rotterdam
  Cargo type  : Perishable   Priority: speed
  Weight      : 6,237 kg   Deadline: 3.9 days   Budget: $91,649

-- Explanation -------------------------------------------------------
  Route R003 recommended. Satisfies all hard constraints (budget &
  on-time probability). Selected from 3 Pareto-efficient routes using
  speed priority (cost 20%, time 60%, risk 20%). Expected transit
  3.5 days, on-time probability 98.4%, cost $52,190.

-- All candidate routes ----------------------------------------------
  route_id  path            modes        cost        exp_time  p_on_time  feasible  reason
  --------  -----           ------       --------    --------  ---------  --------  ------
  R001      BOM→JEA→RTM     sea,sea      $38,200     7.1d      72.3%      False     deadline
  R003      BOM→SHA→RTM     air,sea      $52,190     3.5d      98.4%      True      OK
  ...

-- Best route --------------------------------------------------------
  Route   : R003  (BOM → SHA → RTM)
  Cost    : $52,190
  Exp. time: 3.5 days
  On-time : 98.4%
```

---

### 2. Evaluate All Shipments

Run the engine across all 40 shipments, compute four baseline strategies, calculate regret, Pareto membership, and decision category tags (open / forced / trivial).

```bash
python main.py evaluate
```

```bash
# With custom RNG seed for reproducibility
python main.py evaluate --seed 123
```

**Output files created:**
- `results/results_table.csv` — per-shipment metrics and baseline comparisons
- `results/summary.txt` — aggregate statistics

---

### 3. Sensitivity Analysis

Tests how stable each open-decision shipment is under **11 perturbation scenarios** (weight shifts, on-time threshold changes, delay distribution changes, normalisation boundary changes). Tags each shipment as **ROBUST** (≥ 80% stable) or **CLOSE CALL** (< 80% stable).

```bash
python main.py sensitivity
```

> Note: `evaluate` is automatically run first to identify open shipments before sensitivity testing begins.

**Output files created:**
- `results/sensitivity_table.csv` — per-scenario stability breakdown
- `results/sensitivity_summary.txt` — robust / close-call tags

---

### 4. Launch Web Server

Start the interactive web dashboard and REST API server:

```bash
python main.py server
```

```bash
# Custom host and port
python main.py server --host 0.0.0.0 --port 8080
```

Then open **`http://127.0.0.1:5000`** in your browser.

**All options:**

```
usage: main.py server [-h] [--host HOST] [--port PORT]

  --host HOST    Network host to bind to (default: 127.0.0.1)
                 Use 0.0.0.0 to expose on your local network
  --port PORT    Port number (default: 5000)
```

---

## Using the Web Interface

Once the server is running at `http://127.0.0.1:5000`, the dashboard has **four tabs**:

| Tab | Description |
|---|---|
| **Route Finder** | Select any of 40 preset shipments, choose a priority strategy, and instantly view: recommended route, candidate comparison table, Pareto status, and plain-language explanation |
| **System Evaluation** | Live KPI cards (constraint satisfaction, Pareto efficiency, decision flexibility) and interactive Chart.js charts for baseline agreement and decision type breakdown |
| **Sensitivity Analysis** | Stability breakdown table for open-decision shipments across 11 perturbation scenarios |
| **Executive Reports** | Generate a printable PDF-ready executive report. Optionally trigger an AI-powered summary (requires API key, see below) |

### REST API Endpoints

The server also exposes a REST API for programmatic access:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/shipments` | List all 40 shipments |
| `GET` | `/api/hubs` | List all 15 hub nodes |
| `POST` | `/api/recommend` | Get route recommendation |
| `GET` | `/api/evaluate` | Run/retrieve full evaluation results |
| `GET` | `/api/sensitivity` | Run/retrieve sensitivity analysis |
| `POST` | `/api/summary` | Generate AI executive summary |

**Example API call (recommend):**

```bash
curl -X POST http://127.0.0.1:5000/api/recommend \
  -H "Content-Type: application/json" \
  -d '{"shipment_id": "S001"}'
```

```bash
# With custom weights
curl -X POST http://127.0.0.1:5000/api/recommend \
  -H "Content-Type: application/json" \
  -d '{"shipment_id": "S028", "weights": {"cost": 0.5, "time": 0.3, "risk": 0.2}}'
```

---

## Optional: LLM Executive Summary

The **Executive Reports** tab can generate AI-written summaries in two modes:

### Mode 1 — Live LLM API

Set one of the following environment variables before starting the server:

```bash
# Google Gemini (recommended — free tier available)
set GEMINI_API_KEY=your_api_key_here          # Windows
export GEMINI_API_KEY=your_api_key_here       # macOS / Linux

# OR OpenAI GPT-4o-mini
set OPENAI_API_KEY=your_api_key_here          # Windows
export OPENAI_API_KEY=your_api_key_here       # macOS / Linux
```

Then install the optional package:

```bash
# For Gemini:
pip install google-generativeai

# For OpenAI:
pip install openai
```

### Mode 2 — Built-in Rule-Based Engine (Default)

If no API key is set, `llm_summary.py` automatically falls back to a high-quality deterministic rule-based decision-intelligence engine that generates structured Markdown executive briefs **100% offline**, with no external dependencies.

---

## Output Files

| File | Description |
|---|---|
| `results/results_table.csv` | Per-shipment: recommended route, cost, time, on-time prob, baseline picks, regret, Pareto flag, decision tag |
| `results/summary.txt` | Aggregate statistics: constraint satisfaction rate, Pareto efficiency rate, mean regret, baseline agreement percentages |
| `results/sensitivity_table.csv` | Per-shipment per-scenario: route pick stability across 11 perturbation scenarios |
| `results/sensitivity_summary.txt` | ROBUST / CLOSE CALL tags and per-shipment stability percentage breakdown |

---

## Methodology

| Step | Method |
|---|---|
| **Delay modelling** | Independent Gamma delays per leg; Bernoulli activation with the leg's `delay_prob` |
| **Cost calculation** | `fixed_cost_usd + weight_kg × cost_per_kg_usd` summed over all legs in the route |
| **Hard constraint check** | `cost ≤ budget_usd` AND `P(on-time) ≥ required_on_time_prob` — routes failing either are marked **INFEASIBLE** |
| **Risk score** | `0.5 × norm(P(miss deadline)) + 0.5 × norm(P95 buffer days)` |
| **Pareto filter** | Non-dominated set on `(cost, expected_time, risk)` among feasible routes only |
| **Final score** | Min-max normalised cost and time + raw risk, weighted by the shipment's priority strategy |
| **Sensitivity** | 11 perturbation scenarios per open shipment (3 weight shifts, 4 on-time threshold shifts, 2 delay distribution shifts, 2 normalisation boundary changes); ROBUST if ≥ 80% of scenarios agree on the same recommendation |

---

## Design Decisions

1. **Classical decision analysis, not ML.**
   The dataset is small (40 shipments), there are no historical outcome labels to train on, and the assignment requires explainable, auditable reasoning. Weighted multi-objective scoring with hard constraints is the appropriate tool.

2. **Hard constraints are inviolable gates.**
   Budget and on-time probability are non-negotiable. Routes that violate either are marked `INFEASIBLE` regardless of priority weights. Priority weights only determine which *feasible, Pareto-efficient* route wins — they can never override a hard constraint.

3. **Monte Carlo over analytical approximations.**
   Transit time is the sum of a deterministic base and Gamma-distributed per-leg delays. With 20,000 samples per route (cached on first call), the full distribution is available cheaply: expected time, on-time probability, expected days late, and the P95 buffer are all read directly from the sample array without assuming normality.

4. **Hub risk and congestion are delay drivers, not double-counted scores.**
   Geo-risk and congestion appear in the legs' `delay_prob` values (higher-risk hubs → higher delay probability). They appear a second time only in the plain-language explanation — not in the numeric score — so they cannot inflate the risk penalty.

5. **Pareto filter before weighted scoring.**
   Only non-dominated routes enter the final weighted scoring step. This guarantees the recommendation is always Pareto-efficient; the weights resolve trade-offs *on* the frontier, not *below* it.

---

## Scoring Formula

For each feasible route on the Pareto frontier:

```
score = w_cost  × norm(cost)     [lower is better → inverted]
      + w_time  × norm(exp_time) [lower is better → inverted]
      + w_risk  × risk_score     [lower is better]
```

Where `norm(x) = (x - x_min) / (x_max - x_min)` across all Pareto routes, and the weight vector `(w_cost, w_time, w_risk)` is determined by the shipment's `priority` field:

| Priority | w_cost | w_time | w_risk |
|---|---|---|---|
| `balanced` | 0.40 | 0.40 | 0.20 |
| `cost` | 0.60 | 0.20 | 0.20 |
| `speed` | 0.20 | 0.60 | 0.20 |
| `reliability` | 0.20 | 0.20 | 0.60 |

The route with the **lowest composite score** is recommended.
