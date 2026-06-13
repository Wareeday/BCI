"""
dashboard/layout.py
====================
Clinical dashboard layout — all panels defined here.

Panels:
  1. Header bar with session status and SAFE_STATE indicator
  2. EEG waveform (8 channels, last 2 seconds)
  3. Band power bars (real-time alpha/beta/theta/delta)
  4. BCI command log (last 10 commands with confidence)
  5. Neurofeedback progress (session accuracy chart)
  6. Signal quality (impedance per channel)
  7. System metrics (latency, throughput)
  8. Adaptive calibration status
"""

import dash_bootstrap_components as dbc
from dash import html, dcc


def create_layout() -> html.Div:
    """Build the full dashboard layout."""
    return html.Div(
        style={"backgroundColor": "#0d1b2a", "minHeight": "100vh", "padding": "10px"},
        children=[
            # ── Refresh interval ──────────────────────────────────
            dcc.Interval(id="interval-fast",  interval=100,  n_intervals=0),   # 10 Hz EEG
            dcc.Interval(id="interval-slow",  interval=1000, n_intervals=0),   # 1 Hz metrics
            dcc.Store(id="eeg-buffer",        data=[]),
            dcc.Store(id="command-history",   data=[]),

            # ── Header ────────────────────────────────────────────
            dbc.Row([
                dbc.Col(html.H3("🧠 BCI Platform — Clinical Dashboard",
                                style={"color": "#00b4d8", "margin": "0"}), width=6),
                dbc.Col(html.Div(id="safe-state-banner",
                                 style={"color": "white", "textAlign": "right"}), width=6),
            ], style={"backgroundColor": "#0a1628", "padding": "12px 20px",
                      "marginBottom": "12px", "borderRadius": "8px"}),

            # ── Row 1: EEG + Band Power ───────────────────────────
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("📡 Live EEG — 8 Channels (250 Hz)",
                                       style={"backgroundColor": "#122236", "color": "#00b4d8"}),
                        dbc.CardBody([
                            dcc.Graph(id="eeg-plot",
                                      style={"height": "280px"},
                                      config={"displayModeBar": False}),
                        ], style={"backgroundColor": "#0d1b2a", "padding": "8px"}),
                    ], style={"border": "1px solid #1e90ff"}),
                ], width=8),

                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("🌊 Band Power (µV²)",
                                       style={"backgroundColor": "#122236", "color": "#00b4d8"}),
                        dbc.CardBody([
                            dcc.Graph(id="band-power-plot",
                                      style={"height": "280px"},
                                      config={"displayModeBar": False}),
                        ], style={"backgroundColor": "#0d1b2a", "padding": "8px"}),
                    ], style={"border": "1px solid #7b2fbe"}),
                ], width=4),
            ], className="mb-3"),

            # ── Row 2: Commands + Neurofeedback ──────────────────
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("⚡ BCI Command Log",
                                       style={"backgroundColor": "#122236", "color": "#00b4d8"}),
                        dbc.CardBody([
                            html.Div(id="command-log",
                                     style={"height": "200px", "overflowY": "auto"}),
                        ], style={"backgroundColor": "#0d1b2a"}),
                    ], style={"border": "1px solid #f4a261"}),
                ], width=4),

                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("🎮 Neurofeedback Progress",
                                       style={"backgroundColor": "#122236", "color": "#00b4d8"}),
                        dbc.CardBody([
                            dcc.Graph(id="nf-progress-plot",
                                      style={"height": "200px"},
                                      config={"displayModeBar": False}),
                        ], style={"backgroundColor": "#0d1b2a", "padding": "8px"}),
                    ], style={"border": "1px solid #06d6a0"}),
                ], width=5),

                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("📊 System Metrics",
                                       style={"backgroundColor": "#122236", "color": "#00b4d8"}),
                        dbc.CardBody([
                            html.Div(id="system-metrics",
                                     style={"color": "white", "fontSize": "13px"}),
                        ], style={"backgroundColor": "#0d1b2a"}),
                    ], style={"border": "1px solid #028090"}),
                ], width=3),
            ], className="mb-3"),

            # ── Row 3: Signal Quality + Adaptive Cal ─────────────
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("🔌 Electrode Impedance",
                                       style={"backgroundColor": "#122236", "color": "#00b4d8"}),
                        dbc.CardBody([
                            html.Div(id="impedance-display",
                                     style={"color": "white"}),
                        ], style={"backgroundColor": "#0d1b2a"}),
                    ], style={"border": "1px solid #1b5e20"}),
                ], width=6),

                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("🤖 Adaptive Calibration",
                                       style={"backgroundColor": "#122236", "color": "#00b4d8"}),
                        dbc.CardBody([
                            html.Div(id="calibration-status",
                                     style={"color": "white", "fontSize": "13px"}),
                        ], style={"backgroundColor": "#0d1b2a"}),
                    ], style={"border": "1px solid #ffd166"}),
                ], width=6),
            ]),

            # ── Footer ────────────────────────────────────────────
            html.Hr(style={"borderColor": "#1e2d3d", "marginTop": "16px"}),
            html.P(
                "Standards: IEEE 2857 · ISO 14155 · FDA 510(k) · GDPR  |  "
                "Al Nafi International College — EduQual Level 6  |  "
                "Topic 42: BCI Platform for Assistive Technology",
                style={"color": "#4a6080", "fontSize": "11px", "textAlign": "center"},
            ),
        ],
    )