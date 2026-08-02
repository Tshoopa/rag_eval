"""
HTML report generation.

Renders evaluation output into a self-contained HTML report: summary cards,
an aggregate bar chart (Plotly), and a per-sample results table.
"""
import os
import json
from datetime import datetime

import plotly.graph_objects as go
from plotly.offline import plot


# Human-readable labels for each metric key.
METRIC_LABELS = {
    "faithfulness": "Faithfulness",
    "answer_relevance": "Answer Relevance",
    "context_precision": "Context Precision",
    "context_recall": "Context Recall",
}

# Score thresholds for color coding (green / amber / red).
GOOD_THRESHOLD = 0.8
FAIR_THRESHOLD = 0.5


def _color_for_score(score: float) -> str:
    """Map a score to a traffic-light color."""
    if score >= GOOD_THRESHOLD:
        return "#2ecc71"
    if score >= FAIR_THRESHOLD:
        return "#f39c12"
    return "#e74c3c"


def _build_aggregate_chart(aggregate: dict) -> str:
    """Render the aggregate metrics as an inline Plotly bar chart."""
    labels = [METRIC_LABELS[k] for k in aggregate.keys()]
    # Treat skipped (None) metrics as 0 for plotting purposes.
    values = [v if v is not None else 0 for v in aggregate.values()]
    colors = [_color_for_score(v) for v in values]

    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=values,
            marker_color=colors,
            text=[f"{v:.2f}" for v in values],
            textposition="outside",
        )
    ])
    fig.update_layout(
        title="Average Scores Across All Samples",
        yaxis=dict(range=[0, 1.1], title="Score"),
        xaxis=dict(title="Metric"),
        template="plotly_white",
        height=400,
    )
    return plot(fig, output_type="div", include_plotlyjs="cdn")


def _score_badge(score) -> str:
    """Render a single score as a colored HTML badge."""
    if score is None:
        return '<span style="color:#95a5a6;">N/A</span>'
    color = _color_for_score(score)
    return (f'<span style="background:{color};color:white;padding:3px 8px;'
            f'border-radius:4px;font-weight:bold;">{score:.2f}</span>')


def _build_summary_cards(aggregate: dict, total: int) -> str:
    """Render the top-of-report summary cards."""
    cards = f"""
    <div class="card">
        <div class="card-value">{total}</div>
        <div class="card-label">Total Samples</div>
    </div>
    """
    for metric, score in aggregate.items():
        display = f"{score:.2f}" if score is not None else "N/A"
        cards += f"""
        <div class="card">
            <div class="card-value">{display}</div>
            <div class="card-label">{METRIC_LABELS[metric]}</div>
        </div>
        """
    return cards


def _build_details_table(results: list) -> str:
    """Render the per-sample results table body."""
    rows = ""
    for i, r in enumerate(results, 1):
        s = r["scores"]
        rows += f"""
        <tr>
            <td>{i}</td>
            <td class="question">{r['question']}</td>
            <td>{_score_badge(s['faithfulness'])}</td>
            <td>{_score_badge(s['answer_relevance'])}</td>
            <td>{_score_badge(s['context_precision'])}</td>
            <td>{_score_badge(s['context_recall'])}</td>
        </tr>
        """
    return rows


def generate_html_report(eval_output: dict, output_path: str = "reports/report.html"):
    """Build and write a complete HTML report from evaluation output."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    aggregate = eval_output["aggregate"]
    results = eval_output["results"]
    total = eval_output["total_samples"]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    chart_div = _build_aggregate_chart(aggregate)
    summary_cards = _build_summary_cards(aggregate, total)
    details_rows = _build_details_table(results)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>RAG-Eval Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, sans-serif; margin: 0;
                background: #f5f6fa; color: #2c3e50; }}
        .header {{ background: #34495e; color: white; padding: 30px 40px; }}
        .header h1 {{ margin: 0; font-size: 28px; }}
        .header p {{ margin: 5px 0 0; opacity: 0.8; }}
        .container {{ max-width: 1000px; margin: 30px auto; padding: 0 20px; }}
        .cards {{ display: flex; gap: 15px; flex-wrap: wrap; margin-bottom: 30px; }}
        .card {{ background: white; border-radius: 8px; padding: 20px;
                 flex: 1; min-width: 150px; text-align: center;
                 box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
        .card-value {{ font-size: 32px; font-weight: bold; color: #3498db; }}
        .card-label {{ color: #7f8c8d; font-size: 14px; margin-top: 5px; }}
        .chart-box, .table-box {{ background: white; border-radius: 8px;
                 padding: 20px; margin-bottom: 30px;
                 box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 12px; text-align: center; border-bottom: 1px solid #ecf0f1; }}
        th {{ background: #f8f9fa; color: #34495e; }}
        td.question {{ text-align: right; max-width: 300px; }}
        h2 {{ color: #34495e; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>RAG-Eval Report</h1>
        <p>Generated on {timestamp} · LLM-as-a-judge Evaluation</p>
    </div>
    <div class="container">
        <div class="cards">
            {summary_cards}
        </div>
        <div class="chart-box">
            {chart_div}
        </div>
        <div class="table-box">
            <h2>Per-Sample Results</h2>
            <table>
                <tr>
                    <th>#</th><th>Question</th>
                    <th>Faithfulness</th><th>Answer Rel.</th>
                    <th>Ctx Precision</th><th>Ctx Recall</th>
                </tr>
                {details_rows}
            </table>
        </div>
    </div>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"HTML report generated: {output_path}")
    return output_path