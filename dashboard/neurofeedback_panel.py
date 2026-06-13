"""
dashboard/neurofeedback_panel.py
==================================
Neurofeedback training panel components for the Plotly Dash dashboard.

Renders:
  - Session accuracy progress chart (S1: 62% → S20: 93%)
  - Real-time band power target indicator
  - Gamification score, streak, and badges
  - Adaptive difficulty threshold display
  - Training instructions for clinician

Used by dashboard/callbacks.py to build the neurofeedback tab.
"""

import numpy as np
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import html


# ── Progress chart ────────────────────────────────────────────────

# Expected accuracy progression from presentation
EXPECTED_PROGRESSION = {
    1:  0.62,
    3:  0.71,
    5:  0.79,
    10: 0.88,
    20: 0.93,
}


def create_progress_chart(
    session_accuracies: list[float],
    height: int = 220,
) -> go.Figure:
    """
    Bar chart showing accuracy across all training sessions.

    Green bars = above 80% target.
    Orange bars = 60–80%.
    Red bars = below 60%.
    """
    if not session_accuracies:
        session_accuracies = list(EXPECTED_PROGRESSION.values())

    sessions = list(range(1, len(session_accuracies) + 1))
    pct = [a * 100 for a in session_accuracies]
    colors = [
        "#06d6a0" if a >= 80 else "#f4a261" if a >= 60 else "#ef233c"
        for a in pct
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f"S{s}" for s in sessions],
        y=pct,
        marker_color=colors,
        text=[f"{p:.0f}%" for p in pct],
        textposition="outside",
        textfont=dict(color="white", size=10),
        hovertemplate="Session %{x}<br>Accuracy: %{y:.1f}%<extra></extra>",
    ))

    # 80% target line
    fig.add_hline(
        y=80,
        line_dash="dash",
        line_color="#ffd166",
        line_width=1.5,
        annotation_text="80% target",
        annotation_font=dict(color="#ffd166", size=9),
        annotation_position="bottom right",
    )

    fig.update_layout(
        plot_bgcolor="#0d1b2a",
        paper_bgcolor="#0d1b2a",
        height=height,
        margin=dict(l=30, r=10, t=10, b=30),
        yaxis=dict(
            range=[0, 105],
            title="Accuracy (%)",
            color="white",
            gridcolor="#1e2d3d",
        ),
        xaxis=dict(color="white"),
        showlegend=False,
    )
    return fig


def create_band_power_gauge(
    current_power_normalised: float,
    target_threshold: float,
    target_band: str = "beta",
    height: int = 200,
) -> go.Figure:
    """
    Gauge chart showing current band power vs neurofeedback target.

    Green zone = above threshold (reward condition).
    Red zone   = below threshold.
    """
    color = "#06d6a0" if current_power_normalised >= target_threshold else "#ef233c"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(current_power_normalised * 100, 1),
        title=dict(text=f"{target_band.capitalize()} Band Power (%)", font=dict(color="white")),
        delta=dict(
            reference=target_threshold * 100,
            increasing=dict(color="#06d6a0"),
            decreasing=dict(color="#ef233c"),
        ),
        gauge=dict(
            axis=dict(
                range=[0, 100],
                tickcolor="white",
                tickfont=dict(color="white"),
            ),
            bar=dict(color=color),
            bgcolor="#122236",
            bordercolor="#334455",
            steps=[
                dict(range=[0, target_threshold * 100], color="#1a0a0a"),
                dict(range=[target_threshold * 100, 100], color="#0a1a0a"),
            ],
            threshold=dict(
                line=dict(color="#ffd166", width=3),
                thickness=0.75,
                value=target_threshold * 100,
            ),
        ),
        number=dict(font=dict(color="white"), suffix="%"),
    ))

    fig.update_layout(
        paper_bgcolor="#0d1b2a",
        height=height,
        margin=dict(l=20, r=20, t=40, b=10),
    )
    return fig


# ── Gamification UI components ────────────────────────────────────

def create_score_panel(
    score: int,
    streak: int,
    accuracy: float,
    badges: list[str],
) -> html.Div:
    """Build a gamification score panel for the dashboard."""
    badge_items = [
        dbc.Badge(b, color="warning", className="me-1 mb-1")
        for b in badges
    ] if badges else [
        html.Small("Complete trials to earn badges",
                   style={"color": "#4a6080"})
    ]

    return html.Div([
        dbc.Row([
            dbc.Col([
                html.H4(f"{score}", style={"color": "#ffd166", "margin": "0"}),
                html.Small("Score", style={"color": "#7a9bb5"}),
            ], width=4, className="text-center"),
            dbc.Col([
                html.H4(f"🔥 {streak}", style={"color": "#f4a261", "margin": "0"}),
                html.Small("Streak", style={"color": "#7a9bb5"}),
            ], width=4, className="text-center"),
            dbc.Col([
                html.H4(f"{accuracy:.0%}", style={"color": "#06d6a0", "margin": "0"}),
                html.Small("Session Acc", style={"color": "#7a9bb5"}),
            ], width=4, className="text-center"),
        ], className="mb-2"),
        html.Div(badge_items),
    ])


def create_threshold_indicator(
    current_threshold: float,
    history: list[dict],
) -> html.Div:
    """Show adaptive difficulty threshold with history."""
    direction = ""
    if len(history) >= 2:
        if history[-1]["new_threshold"] > history[-2]["new_threshold"]:
            direction = "↑ (harder)"
        elif history[-1]["new_threshold"] < history[-2]["new_threshold"]:
            direction = "↓ (easier)"

    return html.Div([
        html.Div([
            html.Strong(f"{current_threshold:.0%}",
                        style={"color": "#00b4d8", "fontSize": "20px"}),
            html.Span(f" {direction}",
                      style={"color": "#7a9bb5", "fontSize": "12px"}),
        ]),
        html.Small(
            "Threshold auto-adjusts: >80% success → tighten by 10%",
            style={"color": "#4a6080", "fontSize": "11px"},
        ),
    ])