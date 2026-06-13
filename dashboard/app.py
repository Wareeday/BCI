"""
dashboard/app.py
=================
Plotly Dash clinical dashboard — real-time BCI monitoring.

Features:
  - Live 8-channel EEG waveform (updated at 250 Hz)
  - Band power bars (alpha, beta, theta, delta)
  - BCI command log with confidence scores
  - Neurofeedback progress chart (sessions 1-20)
  - Signal quality badges (impedance per channel)
  - SAFE_STATE indicator (red banner when active)
  - GradCAM explainability heatmap (IEEE 2857 §7.1)
  - Session accuracy trajectory

Run: python dashboard/app.py
URL: http://localhost:8050
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import dash
import dash_bootstrap_components as dbc
from dash import html, dcc
from loguru import logger

from dashboard.layout import create_layout
from dashboard.callbacks import register_callbacks

# ── App creation ──────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.DARKLY,          # dark clinical theme
        dbc.icons.FONT_AWESOME,
    ],
    title="BCI Platform — Clinical Dashboard",
    update_title=None,
    suppress_callback_exceptions=True,
)

server = app.server   # expose Flask server for gunicorn

# ── Layout ────────────────────────────────────────────────────────
app.layout = create_layout()

# ── Callbacks ─────────────────────────────────────────────────────
register_callbacks(app)

if __name__ == "__main__":
    host = os.getenv("DASHBOARD_HOST", "0.0.0.0")
    port = int(os.getenv("DASHBOARD_PORT", 8050))
    debug = os.getenv("DASHBOARD_DEBUG", "true").lower() == "true"
    logger.info(f"Dashboard starting: http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)