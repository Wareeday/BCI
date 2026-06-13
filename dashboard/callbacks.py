"""
dashboard/callbacks.py
=======================
Plotly Dash real-time update callbacks.

All callbacks run in response to:
  - interval-fast (100ms): EEG waveform, band power
  - interval-slow (1s):    commands, metrics, calibration status
"""

import time
import random
from typing import Any

import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, callback, html
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

CHANNEL_NAMES = ["Fp1", "Fp2", "C3", "Cz", "C4", "P3", "P4", "Oz"]
CHANNEL_COLORS = ["#00b4d8", "#7b2fbe", "#06d6a0", "#ffd166",
                  "#f4a261", "#ef233c", "#028090", "#1e90ff"]
COMMAND_COLORS = {"left": "#1e90ff", "right": "#7b2fbe",
                  "feet": "#06d6a0", "rest": "#f4a261"}

_command_history: list[dict] = []
_session_accuracies = [0.62, 0.71, 0.79, 0.88, 0.93]   # from presentation data


def register_callbacks(app):
    """Register all Dash callbacks."""

    # ── EEG waveform ──────────────────────────────────────────────
    @app.callback(
        Output("eeg-plot", "figure"),
        Input("interval-fast", "n_intervals"),
    )
    def update_eeg_plot(n):
        t = np.linspace(0, 2.0, 500)   # 2 seconds of data
        fig = go.Figure()
        for i, (name, color) in enumerate(zip(CHANNEL_NAMES, CHANNEL_COLORS)):
            # Synthetic signal: alpha + noise + channel offset
            offset = i * 50
            signal = (
                np.sin(2 * np.pi * 10 * t) * 15       # 10 Hz alpha
                + np.random.randn(500) * 3              # noise
                + offset
            )
            fig.add_trace(go.Scatter(
                x=t.tolist(), y=signal.tolist(),
                mode="lines", name=name,
                line=dict(color=color, width=1),
            ))
        fig.update_layout(
            plot_bgcolor="#0d1b2a", paper_bgcolor="#0d1b2a",
            margin=dict(l=40, r=10, t=10, b=30),
            showlegend=True,
            legend=dict(font=dict(color="white", size=9), bgcolor="#122236"),
            xaxis=dict(title="Time (s)", color="white", gridcolor="#1e2d3d"),
            yaxis=dict(title="µV + offset", color="white", gridcolor="#1e2d3d"),
        )
        return fig

    # ── Band power bars ───────────────────────────────────────────
    @app.callback(
        Output("band-power-plot", "figure"),
        Input("interval-fast", "n_intervals"),
    )
    def update_band_power(n):
        # Simulated band powers (µV²)
        bands = ["Delta\n1-4Hz", "Theta\n4-8Hz", "Alpha\n8-13Hz", "Beta\n13-30Hz"]
        powers = [
            2.1 + random.gauss(0, 0.3),
            4.5 + random.gauss(0, 0.5),
            18.3 + random.gauss(0, 2.0),   # alpha dominant
            12.1 + random.gauss(0, 1.5),
        ]
        colors = ["#028090", "#7b2fbe", "#00b4d8", "#06d6a0"]
        fig = go.Figure(go.Bar(
            x=bands,
            y=powers,
            marker_color=colors,
            text=[f"{p:.1f}" for p in powers],
            textposition="outside",
            textfont=dict(color="white", size=11),
        ))
        fig.update_layout(
            plot_bgcolor="#0d1b2a", paper_bgcolor="#0d1b2a",
            margin=dict(l=20, r=10, t=10, b=40),
            yaxis=dict(title="Power (µV²)", color="white", gridcolor="#1e2d3d"),
            xaxis=dict(color="white"),
            showlegend=False,
        )
        return fig

    # ── Command log ───────────────────────────────────────────────
    @app.callback(
        Output("command-log", "children"),
        Input("interval-slow", "n_intervals"),
    )
    def update_command_log(n):
        commands = ["left", "right", "feet", "rest"]
        # Simulate new command every ~3 updates
        if n % 3 == 0:
            cmd = random.choice(commands)
            conf = random.uniform(0.75, 0.97)
            _command_history.append({
                "time": time.strftime("%H:%M:%S"),
                "command": cmd,
                "confidence": conf,
                "executed": conf >= 0.85,
            })
        if len(_command_history) > 20:
            _command_history.pop(0)

        rows = []
        for entry in reversed(_command_history[-10:]):
            color = COMMAND_COLORS.get(entry["command"], "white")
            badge_color = "success" if entry["executed"] else "warning"
            rows.append(
                dbc.Row([
                    dbc.Col(html.Small(entry["time"], style={"color": "#7a9bb5"}), width=3),
                    dbc.Col(html.Strong(entry["command"].upper(),
                                        style={"color": color}), width=3),
                    dbc.Col(html.Small(f"{entry['confidence']:.2f}",
                                       style={"color": "white"}), width=3),
                    dbc.Col(dbc.Badge(
                        "ISSUED" if entry["executed"] else "HOLD",
                        color=badge_color, className="ms-1",
                    ), width=3),
                ], className="mb-1")
            )
        return rows if rows else [html.P("Waiting for commands...",
                                         style={"color": "#4a6080"})]

    # ── Neurofeedback progress ────────────────────────────────────
    @app.callback(
        Output("nf-progress-plot", "figure"),
        Input("interval-slow", "n_intervals"),
    )
    def update_nf_progress(n):
        sessions = list(range(1, len(_session_accuracies) + 1))
        pct = [a * 100 for a in _session_accuracies]
        colors = ["#ef233c" if a < 75 else "#f4a261" if a < 85 else "#06d6a0" for a in pct]
        fig = go.Figure(go.Bar(
            x=[f"S{s}" for s in sessions],
            y=pct,
            marker_color=colors,
            text=[f"{p:.0f}%" for p in pct],
            textposition="outside",
            textfont=dict(color="white", size=10),
        ))
        fig.add_hline(y=80, line_dash="dash", line_color="#ffd166",
                      annotation_text="80% target", annotation_font_color="#ffd166")
        fig.update_layout(
            plot_bgcolor="#0d1b2a", paper_bgcolor="#0d1b2a",
            margin=dict(l=30, r=10, t=10, b=30),
            yaxis=dict(range=[0, 100], title="Accuracy (%)",
                       color="white", gridcolor="#1e2d3d"),
            xaxis=dict(color="white"),
            showlegend=False,
        )
        return fig

    # ── System metrics ────────────────────────────────────────────
    @app.callback(
        Output("system-metrics", "children"),
        Input("interval-slow", "n_intervals"),
    )
    def update_metrics(n):
        dsp_ms = 7.5 + random.gauss(0, 0.5)
        e2e_ms = 85 + random.gauss(0, 5)
        return [
            _metric_row("DSP Latency", f"{dsp_ms:.1f} ms", "✓", "#06d6a0" if dsp_ms < 10 else "#ef233c"),
            _metric_row("End-to-End", f"{e2e_ms:.0f} ms", "✓", "#06d6a0" if e2e_ms < 100 else "#ef233c"),
            _metric_row("CNN Accuracy", "91%", "✓", "#06d6a0"),
            _metric_row("Kafka Health", "OK", "✓", "#06d6a0"),
            _metric_row("Safe State", "OFF", "✓", "#06d6a0"),
            _metric_row("Active Model", "CNN", "", "#00b4d8"),
        ]

    # ── Impedance display ─────────────────────────────────────────
    @app.callback(
        Output("impedance-display", "children"),
        Input("interval-slow", "n_intervals"),
    )
    def update_impedance(n):
        badges = []
        for i, name in enumerate(CHANNEL_NAMES):
            kohm = round(1.5 + random.uniform(0, 5.0), 1)
            color = "success" if kohm < 5.0 else "danger"
            badges.append(
                dbc.Badge(
                    f"{name}: {kohm}kΩ",
                    color=color, className="me-1 mb-1",
                    style={"fontSize": "11px"},
                )
            )
        return html.Div([
            html.Small("Target: <5 kΩ (ISO 14155 Bench Test)",
                        style={"color": "#7a9bb5", "display": "block", "marginBottom": "6px"}),
            html.Div(badges),
        ])

    # ── Adaptive calibration ──────────────────────────────────────
    @app.callback(
        Output("calibration-status", "children"),
        Input("interval-slow", "n_intervals"),
    )
    def update_calibration(n):
        return html.Div([
            _metric_row("Total Retrains", "3"),
            _metric_row("Last Retrain", "290s ago"),
            _metric_row("Current Acc", "91.2%"),
            _metric_row("Cold-start Time", "4 min (TL)"),
            _metric_row("Alert Threshold", "<80% × 3 sessions"),
            html.Small(
                "SGD updates every 30 trials · lr=0.0001 · frozen conv",
                style={"color": "#4a6080", "fontSize": "10px"},
            ),
        ])

    # ── SAFE_STATE banner ─────────────────────────────────────────
    @app.callback(
        Output("safe-state-banner", "children"),
        Input("interval-slow", "n_intervals"),
    )
    def update_safe_state_banner(n):
        # In real system: read from API /api/v1/devices/status
        safe_active = False
        if safe_active:
            return dbc.Alert("🚨 SAFE_STATE ACTIVE — All actuators halted",
                             color="danger", className="mb-0 py-2")
        return html.Span(
            "🟢 System Normal  |  EEG: Streaming  |  CNN: Active",
            style={"color": "#06d6a0", "fontSize": "13px"},
        )


def _metric_row(label: str, value: str, icon: str = "", color: str = "white") -> Any:
    return dbc.Row([
        dbc.Col(html.Small(label, style={"color": "#7a9bb5"}), width=7),
        dbc.Col(html.Small(f"{icon} {value}", style={"color": color, "fontWeight": "bold"}), width=5),
    ], className="mb-1")