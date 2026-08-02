"""
Generates an interactive HTML dashboard from MediMama triage evaluation results.
Usage: python generate_triage_dashboard.py
"""
import json
import os

JSON_PATH = "reports/medimama_triage_results_2.json"
OUTPUT_HTML_PATH = "reports/medimama_triage_dashboard.html"

def generate_dashboard():
    if not os.path.exists(JSON_PATH):
        print(f"Error: {JSON_PATH} not found. Run evaluation first.")
        return

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    summary = data.get("summary", {})
    results = data.get("results", [])
    total_cases = summary.get("total_cases", len(results))

    LEVEL_STYLES = {
        1: {"bg": "#fee2e2", "text": "#991b1b", "border": "#f87171", "label": "L1 - Resuscitation"},
        2: {"bg": "#ffedd5", "text": "#9a3412", "border": "#fb923c", "label": "L2 - Emergency"},
        3: {"bg": "#fef3c7", "text": "#92400e", "border": "#facc15", "label": "L3 - Urgent"},
        4: {"bg": "#fef9c3", "text": "#854d0e", "border": "#eab308", "label": "L4 - Semi-Urgent"},
        5: {"bg": "#dcfce7", "text": "#166534", "border": "#4ade80", "label": "L5 - Non-Urgent"},
    }

    under_rate = summary.get("under_triage_rate", 0)
    safety_rate = summary.get("clinical_safety_rate", 0)
    under_count = summary.get("under_triage_count", 0)
    safety_badge_color = "#16a34a" if under_count == 0 else "#d97706"

    css_style = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        :root { --primary: #0f172a; --bg: #f8fafc; --card-bg: #ffffff; --text: #334155; }
        body { font-family: 'Inter', sans-serif; background-color: var(--bg); color: var(--text); margin: 0; padding: 40px 20px; line-height: 1.5; }
        .container { max-width: 1280px; margin: 0 auto; }
        header { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); color: white; padding: 32px; border-radius: 16px; display: flex; justify-content: space-between; align-items: center; margin-bottom: 32px;}
        header h1 { margin: 0 0 8px 0; font-size: 28px; }
        header p { margin: 0; color: #94a3b8; font-size: 14px; }
        .badge-safety { color: white; padding: 8px 16px; border-radius: 9999px; font-weight: 600; font-size: 14px; }
        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 20px; margin-bottom: 32px; }
        .card { background: var(--card-bg); padding: 24px; border-radius: 12px; border: 1px solid #e2e8f0; }
        .card-title { font-size: 12px; font-weight: 600; color: #64748b; margin-bottom: 8px; text-transform: uppercase; }
        .card-value { font-size: 32px; font-weight: 700; color: var(--primary); }
        .card-sub { font-size: 13px; color: #64748b; margin-top: 4px; }
        .section-title { font-size: 20px; font-weight: 700; color: var(--primary); margin: 40px 0 20px 0; }
        .matrix-table, .data-table { width: 100%; border-collapse: collapse; text-align: center; font-size: 14px; background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.02); }
        .matrix-table th, .matrix-table td, .data-table th, .data-table td { padding: 12px; border: 1px solid #e2e8f0; }
        .matrix-table th, .data-table th { background: #f8fafc; font-weight: 600; }
        .data-table { text-align: left; }
        .cm-cell-match { background: #dcfce7 !important; color: #15803d; font-weight: 700; }
        .cm-cell-over { background: #fef3c7 !important; color: #b45309; font-weight: 600; }
        .cm-cell-under { background: #fee2e2 !important; color: #b91c1c; font-weight: 700; }
        .level-badge { padding: 4px 10px; border-radius: 6px; font-weight: 600; font-size: 12px; }
        .status-tag { padding: 4px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }
        .input-text { font-size: 13px; max-width: 320px; }
        .response-box { background: #f8fafc; padding: 10px; border-radius: 6px; border: 1px solid #e2e8f0; font-size: 12px; max-height: 120px; overflow-y: auto; white-space: pre-wrap; }
    </style>
    """

    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>MediMama Triage Dashboard</title>{css_style}</head><body><div class="container">
    <header>
        <div><h1>🚑 MediMama Evaluation Dashboard</h1><p>Pediatric Triage Safety & Accuracy Performance Report</p></div>
        <div class="badge-safety" style="background:{safety_badge_color};">🛡️ Clinical Safety Rate: {safety_rate:.1f}%</div>
    </header>

    <div class="metrics-grid">
        <div class="card"><div class="card-title">Exact-Match Accuracy</div><div class="card-value" style="color:#2563eb;">{summary.get('exact_match_accuracy', 0):.1f}%</div><div class="card-sub">{summary.get('exact_match_count', 0)} / {summary.get('successful_api_responses', total_cases)} Cases Matched</div></div>
        <div class="card"><div class="card-title">Clinical Safety Rate</div><div class="card-value" style="color:#16a34a;">{safety_rate:.1f}%</div><div class="card-sub">Critical Recall: {summary.get('critical_emergency_recall', 0):.0f}%</div></div>
        <div class="card"><div class="card-title">Under-Triage Rate</div><div class="card-value" style="color:{'#16a34a' if under_count == 0 else '#dc2626'};">{under_rate:.2f}%</div><div class="card-sub">Dangerous Downgrades: {under_count}</div></div>
        <div class="card"><div class="card-title">Over-Triage Rate</div><div class="card-value" style="color:#d97706;">{summary.get('over_triage_rate', 0):.1f}%</div><div class="card-sub">{summary.get('over_triage_count', 0)} Escalated Safely</div></div>
    </div>

    <div class="section-title">📊 Confusion Matrix</div>
    <table class="matrix-table">
        <thead><tr><th>Expected \\ Predicted</th><th>L1</th><th>L2</th><th>L3</th><th>L4</th><th>L5</th></tr></thead>
        <tbody>
    """
    cm = summary.get("confusion_matrix", {})
    for exp in ["1", "2", "3", "4", "5"]:
        html += f'<tr><th>Level {exp}</th>'
        for pred in ["1", "2", "3", "4", "5"]:
            val = cm.get(exp, {}).get(pred, 0)
            c_class = "cm-cell-match" if exp == pred and val > 0 else "cm-cell-over" if int(pred) < int(exp) and val > 0 else "cm-cell-under" if int(pred) > int(exp) and val > 0 else ""
            html += f'<td class="{c_class}">{val}</td>'
        html += "</tr>"
    
    html += f"""</tbody></table>
    <div class="section-title">🔍 Detailed Case Results ({total_cases} Cases)</div>
    <table class="data-table">
        <thead><tr><th>ID / Category</th><th>Symptoms</th><th>Expected</th><th>Predicted</th><th>Status</th><th>Response</th></tr></thead><tbody>
    """
    
    for item in results:
        exp_s = LEVEL_STYLES.get(item.get("expected_level"), {})
        pred_s = LEVEL_STYLES.get(item.get("predicted_level"), {})
        cls = item.get("triage_classification", "")
        stat_html = '<span class="status-tag" style="background:#dcfce7;color:#15803d">Match</span>' if cls == "exact_match" else '<span class="status-tag" style="background:#fef3c7;color:#b45309">Over-Triage</span>' if cls == "over_triage" else '<span class="status-tag" style="background:#fee2e2;color:#b91c1c">Under-Triage</span>' if cls == "under_triage" else '<span class="status-tag" style="background:#e2e8f0;color:#475569">API Error</span>'
        
        html += f"""<tr>
            <td><strong>{item.get('id')}</strong><br><span style="font-size:11px;color:#64748b;">{item.get('category')}</span></td>
            <td class="input-text">{item.get('input')}</td>
            <td><span class="level-badge" style="background:{exp_s.get('bg')};color:{exp_s.get('text')};border:1px solid {exp_s.get('border')}">{exp_s.get('label')}</span></td>
            <td><span class="level-badge" style="background:{pred_s.get('bg','#e2e8f0')};color:{pred_s.get('text','#475569')};border:1px solid {pred_s.get('border','#cbd5e1')}">{pred_s.get('label', 'N/A')}</span></td>
            <td>{stat_html}</td><td><div class="response-box">{item.get('answer', '') or '—'}</div></td>
        </tr>"""

    html += "</tbody></table></div></body></html>"

    os.makedirs(os.path.dirname(OUTPUT_HTML_PATH), exist_ok=True)
    with open(OUTPUT_HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Dashboard generated: {os.path.abspath(OUTPUT_HTML_PATH)}")

if __name__ == "__main__":
    generate_dashboard()