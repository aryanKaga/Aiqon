"""
server.py — Flask REST API & Web Dashboard Server for Recommendation System
"""

import os
import json
import pandas as pd
from flask import Flask, render_template, request, jsonify
from Recommendation_Engine.DataLoader import DataLoader
from Recommendation_Engine.engine import RecommendationEngine
from llm_summary import generate_llm_summary
import evaluate
import sensitivity

app = Flask(__name__, template_folder='templates', static_folder='static')

# Initialize Engine
loader = DataLoader()
engine = RecommendationEngine(loader)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/shipments', methods=['GET'])
def get_shipments():
    shipments_df = loader.shipments
    records = shipments_df.to_dict(orient='records')
    return jsonify({"shipments": records, "count": len(records)})

@app.route('/api/hubs', methods=['GET'])
def get_hubs():
    hubs_df = loader.hubs
    records = hubs_df.to_dict(orient='records')
    return jsonify({"hubs": records})

HUB_MAP = {
    'BOM': 'Mumbai', 'SHA': 'Shanghai', 'SIN': 'Singapore', 'JEA': 'Jebel Ali',
    'RTM': 'Rotterdam', 'HAM': 'Hamburg', 'FXT': 'Felixstowe', 'NYC': 'New York', 'LAX': 'Los Angeles'
}

@app.route('/api/recommend', methods=['POST'])
def recommend():
    data = request.json or {}
    shipment_id = data.get('shipment_id')
    custom_weights = data.get('weights')
    
    if shipment_id and shipment_id != "CUSTOM":
        try:
            shipment_row = loader.shipment(shipment_id).copy()
            if data.get('cargo_type'):
                shipment_row['cargo_type'] = data['cargo_type']
            if data.get('priority'):
                shipment_row['priority'] = data['priority']

            loader.shipments.loc[shipment_id] = shipment_row
            result = engine.find_best_route(shipment_id, weights=custom_weights)
            shipment_info = shipment_row.to_dict()
        except KeyError:
            return jsonify({"error": f"Shipment {shipment_id} not found"}), 404
    else:
        # Custom shipment parameters
        origin_raw = data.get('origin', 'Mumbai')
        destination_raw = data.get('destination', 'Rotterdam')
        origin = HUB_MAP.get(origin_raw, origin_raw)
        destination = HUB_MAP.get(destination_raw, destination_raw)
        
        weight_kg = float(data.get('weight_kg', 10000))
        budget_usd = float(data.get('budget_usd', 50000))
        max_deadline_days = float(data.get('max_deadline_days', 25.0))
        required_on_time_prob = float(data.get('required_on_time_prob', 0.85))
        priority = data.get('priority', 'balanced')
        cargo_type = data.get('cargo_type', 'Standard')
        
        # Virtual shipment Series
        s_dict = {
            "shipment_id": "CUSTOM",
            "origin": origin,
            "destination": destination,
            "weight_kg": weight_kg,
            "budget_usd": budget_usd,
            "deadline_days": max_deadline_days,
            "max_deadline_days": max_deadline_days,
            "required_on_time_prob": required_on_time_prob,
            "priority": priority,
            "cargo_type": cargo_type
        }
        shipment_info = s_dict

        # Temporarily register or mock in loader.shipments
        loader.shipments.loc["CUSTOM"] = pd.Series(s_dict)
        try:
            result = engine.find_best_route("CUSTOM", weights=custom_weights)
        finally:
            if "CUSTOM" in loader.shipments.index:
                loader.shipments.drop("CUSTOM", inplace=True)

    table = result["table"]
    best = result.get("best", {})
    
    # Format table for JSON response
    table_df = table.copy()
    if "pareto" in table_df.columns:
        table_df["pareto"] = table_df["pareto"].fillna(False)
    
    table_records = table_df.to_dict(orient='records')
    for rec in table_records:
        if isinstance(rec.get("path"), list):
            rec["path"] = " -> ".join(rec["path"])
        
        # Standardize field names for UI
        c_val = float(rec.get("cost", 0) or 0)
        e_val = float(rec.get("exp_time", 0) or 0)
        b_val = float(rec.get("buffer_days", 0) or 0)
        p_val = float(rec.get("p_on_time", 0) or 0)

        rec["cost_usd"] = round(c_val, 2)
        rec["expected_days"] = round(e_val, 2)
        rec["on_time_prob"] = round(p_val, 4)
        rec["p95_days"] = round(e_val + b_val, 2)
        
        # Ensure NaN values become None for JSON serialization
        for k, v in list(rec.items()):
            if pd.isna(v):
                rec[k] = None

    cost_val = round(float(best.get("cost", 0)), 2)
    exp_time_val = round(float(best.get("exp_time", 0)), 2)
    buffer_val = float(best.get("buffer_days", 0))
    p95_val = round(exp_time_val + buffer_val, 2)
    on_time_val = round(float(best.get("p_on_time", 0)), 4)
    risk_val = float(best.get("risk", 0)) if best.get("risk") is not None and not pd.isna(best.get("risk")) else 0.0
    score_val = float(best.get("score", 0)) if best.get("score") is not None and not pd.isna(best.get("score")) else 0.0

    n_candidates = len(table_df)
    n_feasible = int(table_df["feasible"].sum()) if "feasible" in table_df.columns else 0
    n_pareto = int(table_df["score"].notna().sum()) if "score" in table_df.columns else 0

    return jsonify({
        "status": "success",
        "shipment": shipment_info,
        "n_candidates": n_candidates,
        "n_feasible": n_feasible,
        "n_pareto": n_pareto,
        "recommended_route": best,
        "recommended_route_id": best.get("route_id", "N/A"),
        "cost_usd": cost_val,
        "expected_days": exp_time_val,
        "p95_days": p95_val,
        "on_time_prob": on_time_val,
        "risk_score": risk_val,
        "score": score_val,
        "weights": result.get("weights", {}),
        "candidates_table": table_records,
        "explanation": result.get("explanation", "")
    })

@app.route('/api/evaluate', methods=['GET'])
def run_evaluation():
    # Run evaluation script or fetch cached results table
    res_path = os.path.join('results', 'results_table.csv')
    summary_path = os.path.join('results', 'summary.txt')
    
    if not os.path.exists(res_path):
        evaluate.main()
        
    df = pd.read_csv(res_path)
    summary_text = ""
    if os.path.exists(summary_path):
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary_text = f.read()

    # Calculate top-level KPI metrics matching results_table.csv schema
    total = len(df)
    feasible = int((df['constraint_satisfied'] == True).sum()) if 'constraint_satisfied' in df else total
    pareto_rec = int((df['is_pareto_member'] == True).sum()) if 'is_pareto_member' in df else total
    open_dec = int((df['is_open'] == True).sum()) if 'is_open' in df else 0
    forced = total - open_dec
    
    same_cheapest = int((df['recommended_route'] == df['cheapest_route']).sum()) if 'cheapest_route' in df else 0
    same_fastest = int((df['recommended_route'] == df['fastest_route']).sum()) if 'fastest_route' in df else 0
    same_reliable = int((df['recommended_route'] == df['reliable_route']).sum()) if 'reliable_route' in df else 0

    # Fill NaN for clean JSON return
    df_clean = df.copy()
    for col in df_clean.columns:
        df_clean[col] = df_clean[col].apply(lambda x: None if pd.isna(x) else x)

    return jsonify({
        "total_shipments": total,
        "feasible_count": feasible,
        "pareto_count": pareto_rec,
        "forced_count": forced,
        "open_count": open_dec,
        "baseline_agreement": {
            "cheapest": round(same_cheapest / total * 100, 1),
            "fastest": round(same_fastest / total * 100, 1),
            "most_reliable": round(same_reliable / total * 100, 1)
        },
        "summary_text": summary_text,
        "table": df_clean.to_dict(orient='records')
    })

@app.route('/api/sensitivity', methods=['GET'])
def run_sensitivity():
    sens_path = os.path.join('results', 'sensitivity_table.csv')
    summary_path = os.path.join('results', 'sensitivity_summary.txt')
    
    if not os.path.exists(sens_path):
        sensitivity.main()
        
    df = pd.read_csv(sens_path)
    summary_text = ""
    if os.path.exists(summary_path):
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary_text = f.read()

    robust_count = int((df['tag'].astype(str).str.upper() == 'ROBUST').sum())
    close_call_count = int((df['tag'].astype(str).str.upper() == 'CLOSE CALL').sum())
    mean_stability = float(df['stability_pct'].mean())

    return jsonify({
        "open_shipments_analysed": len(df),
        "robust_count": robust_count,
        "close_call_count": close_call_count,
        "mean_stability": round(mean_stability, 1),
        "summary_text": summary_text,
        "table": df.to_dict(orient='records')
    })

@app.route('/api/summary', methods=['POST'])
def get_summary():
    data = request.json or {}
    recommendation_data = data.get('recommendation', {})
    shipment_data = data.get('shipment', {})
    custom_prompt = data.get('prompt', None)
    
    summary_res = generate_llm_summary(recommendation_data, shipment_data, custom_prompt)
    return jsonify(summary_res)

def start_server(host='127.0.0.1', port=5000, debug=False):
    print(f"\n=========================================================")
    print(f"  Aiqon Logistics Recommendation Engine Server")
    print(f"  Dashboard available at: http://{host}:{port}")
    print(f"=========================================================\n")
    app.run(host=host, port=port, debug=debug)

if __name__ == '__main__':
    start_server(port=5000)
