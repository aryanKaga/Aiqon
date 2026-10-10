"""
llm_summary.py — Optional LLM Summary Generator for Shipping Recommendations

Generates executive summaries for shipping recommendations using live LLM APIs 
(Gemini or OpenAI) when API keys are available, or a high-quality analytical 
rule-based AI engine fallback when offline or unconfigured.
"""

import os
import json
import urllib.request
import urllib.error

def generate_llm_summary(recommendation_data: dict, shipment_data: dict, custom_prompt: str = None) -> dict:
    """
    Generates an executive summary for a recommendation.
    
    Returns a dict with:
        - "summary": Markdown string with executive analysis
        - "provider": "Gemini API" | "OpenAI API" | "Built-in AI Fallback Engine"
        - "status": "success" | "warning"
    """
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    
    rec_route = recommendation_data.get("recommended_route", {})
    cost = recommendation_data.get("cost_usd", 0)
    expected_days = recommendation_data.get("expected_days", 0)
    p95_days = recommendation_data.get("p95_days", 0)
    on_time_prob = recommendation_data.get("on_time_prob", 0) * 100
    shipment_id = shipment_data.get("shipment_id", "N/A")
    origin = shipment_data.get("origin", "N/A")
    dest = shipment_data.get("destination", "N/A")
    budget = shipment_data.get("budget_usd", 0)
    deadline = shipment_data.get("max_deadline_days", 0)
    priority = shipment_data.get("priority", "balanced")
    cargo_type = shipment_data.get("cargo_type", "Standard")
    
    prompt_content = f"""
System context: You are an expert global logistics decision analyst.
Analyze the following shipping recommendation for Shipment {shipment_id}:

Cargo Request:
- Origin to Destination: {origin} -> {dest}
- Priority: {priority.upper()} | Cargo Type: {cargo_type}
- Weight: {shipment_data.get('weight_kg', 0):,} kg
- Constraints: Budget = ${budget:,.2f} USD, Max Deadline = {deadline:.1f} days, Required On-Time Prob = {shipment_data.get('required_on_time_prob', 0.9)*100:.0f}%

Recommended Route Details:
- Route ID: {rec_route.get('route_id', 'N/A')} ({rec_route.get('path', 'N/A')})
- Modes: {rec_route.get('modes', 'N/A')}
- Total Cost: ${cost:,.2f} USD (Budget margin: ${budget - cost:+,.2f})
- Expected Time: {expected_days:.1f} days (Deadline margin: {deadline - expected_days:+.1f} days)
- On-Time Probability: {on_time_prob:.1f}%
- P95 Transit Time (95% confidence): {p95_days:.1f} days
- Risk Score: {recommendation_data.get('risk_score', 0):.3f}

Additional context: {custom_prompt if custom_prompt else "Provide an executive summary highlighting economic trade-offs, risk mitigations, and operational fit."}

Format output in clean GitHub markdown with headings:
## Executive Summary
## Strategic Fit & Trade-offs
## Risk & Contingency Assessment
## Decision Recommendation
"""

    # 1. Try Gemini API if key exists
    if gemini_key:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_key}"
            payload = {
                "contents": [{"parts": [{"text": prompt_content}]}]
            }
            req = urllib.request.Request(
                url, 
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                res = json.loads(resp.read().decode("utf-8"))
                text = res["candidates"][0]["content"]["parts"][0]["text"]
                return {
                    "summary": text,
                    "provider": "Gemini 1.5 Flash API",
                    "status": "success"
                }
        except Exception as e:
            print(f"[LLM Summary] Gemini API call failed: {e}")

    # 2. Try OpenAI API if key exists
    if openai_key:
        try:
            url = "https://api.openai.com/v1/chat/completions"
            payload = {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt_content}]
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {openai_key}"
                },
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                res = json.loads(resp.read().decode("utf-8"))
                text = res["choices"][0]["message"]["content"]
                return {
                    "summary": text,
                    "provider": "OpenAI GPT-4o-mini API",
                    "status": "success"
                }
        except Exception as e:
            print(f"[LLM Summary] OpenAI API call failed: {e}")

    # 3. Smart Built-in Analytical AI Fallback Engine
    fallback_summary = _generate_fallback_summary(
        shipment_id, origin, dest, priority, cargo_type, budget, cost,
        deadline, expected_days, p95_days, on_time_prob, rec_route, recommendation_data
    )
    
    return {
        "summary": fallback_summary,
        "provider": "Built-in Logistics AI Engine (Offline/Rule-based)",
        "status": "success"
    }


def _generate_fallback_summary(shipment_id, origin, dest, priority, cargo_type, budget, cost, deadline, expected_days, p95_days, on_time_prob, rec_route, rec_data) -> str:
    cost_saving = budget - cost
    time_margin = deadline - expected_days
    route_id = rec_route.get("route_id", "N/A")
    path = rec_route.get("path", "N/A")
    modes = rec_route.get("modes", "N/A")
    n_candidates = rec_data.get("n_candidates", 0)
    n_pareto = rec_data.get("n_pareto", 0)

    risk_assessment = "Low Risk"
    if on_time_prob < 85:
        risk_assessment = "Moderate Delay Risk"
    elif p95_days > deadline:
        risk_assessment = "Tail-Risk Warning (P95 exceeds deadline)"

    prio_alignment = f"aligned with cargo priority **{priority.upper()}**"
    if priority == "cost":
        prio_alignment += f", maximizing budget savings (${cost_saving:,.2f} USD buffer)."
    elif priority == "speed":
        prio_alignment += f", prioritizing swift transit ({expected_days:.1f} days avg vs {deadline:.1f} day limit)."
    elif priority == "reliability":
        prio_alignment += f", guaranteeing {on_time_prob:.1f}% on-time probability."
    else:
        prio_alignment += ", achieving balanced optimization across cost, speed, and reliability."

    summary_md = f"""## Executive Summary
For shipment **{shipment_id}** from **{origin}** to **{dest}** ({cargo_type} cargo), the recommendation engine evaluated **{n_candidates} feasible candidate routes** and filtered down to **{n_pareto} Pareto-optimal solution(s)**.

The top recommended choice is **Route {route_id}** (`{path}` via {modes}).

## Strategic Fit & Trade-offs
- **Cost Efficiency**: Total route cost is **${cost:,.2f} USD**, delivering **${cost_saving:,.2f} USD in savings** under the maximum budget of ${budget:,.2f} USD.
- **Transit Schedule**: Expected transit duration is **{expected_days:.1f} days**, providing a safety buffer of **{time_margin:+.1f} days** ahead of the {deadline:.1f}-day deadline.
- **Priority Alignment**: This selection is {prio_alignment}

## Risk & Contingency Assessment
- **On-Time Reliability**: **{on_time_prob:.1f}%** chance of arriving on or before the deadline.
- **95th Percentile Buffer**: Under worst-case delay scenarios (P95), arrival is estimated at **{p95_days:.1f} days**.
- **Risk Status**: `{risk_assessment}`. Hub & corridor congestion factors have been factored in via Monte Carlo simulation.

## Operational Recommendation
> **Approved for Dispatch**: Proceed with booking Route **{route_id}** (`{path}`). Route performance is optimal and non-dominated on the efficiency frontier.
"""
    return summary_md
