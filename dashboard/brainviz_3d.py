"""
dashboard/brainviz_3d.py
==========================
3D head model EEG topographic power map using Plotly.

Renders a 2D topographic map (topoplot) of EEG band power
projected onto a standard 10-20 electrode layout.

Features:
  - Colour-coded power map (red = high, blue = low)
  - Electrode labels overlaid
  - Updates in real time from band power callbacks
  - Used for clinician monitoring and neurofeedback visualisation

10-20 electrode positions (normalised 0-1 for head circle):
  Fp1, Fp2 (frontal poles)
  C3, Cz, C4 (central)
  P3, P4 (parietal)
  Oz (occipital)
"""

import numpy as np
import plotly.graph_objects as go
from typing import Optional

# 10-20 electrode positions (x, y) normalised to unit circle
# Convention: top=nose, left=left hemisphere
ELECTRODE_POSITIONS = {
    "Fp1": (-0.18, 0.85),
    "Fp2": ( 0.18, 0.85),
    "C3":  (-0.50, 0.00),
    "Cz":  ( 0.00, 0.00),
    "C4":  ( 0.50, 0.00),
    "P3":  (-0.35,-0.50),
    "P4":  ( 0.35,-0.50),
    "Oz":  ( 0.00,-0.85),
}

CHANNEL_ORDER = ["Fp1", "Fp2", "C3", "Cz", "C4", "P3", "P4", "Oz"]


def create_topoplot(
    band_powers: dict,
    title: str = "EEG Band Power",
    colorscale: str = "RdBu_r",
    height: int = 350,
) -> go.Figure:
    """
    Create a 2D topographic EEG power map.

    Args:
        band_powers: dict {channel_name: power_uv2}
                     e.g. {"Fp1": 12.3, "C3": 18.7, ...}
        title:       plot title
        colorscale:  Plotly colorscale name
        height:      figure height in pixels

    Returns:
        Plotly Figure with topoplot
    """
    xs, ys, zs, labels = [], [], [], []

    for ch in CHANNEL_ORDER:
        if ch in ELECTRODE_POSITIONS and ch in band_powers:
            x, y = ELECTRODE_POSITIONS[ch]
            xs.append(x)
            ys.append(y)
            zs.append(band_powers[ch])
            labels.append(f"{ch}<br>{band_powers[ch]:.1f} µV²")

    if not xs:
        return _empty_topoplot(title)

    # Scatter plot of electrode positions with power as colour
    fig = go.Figure()

    # Head outline circle
    theta = np.linspace(0, 2 * np.pi, 100)
    fig.add_trace(go.Scatter(
        x=np.cos(theta).tolist(),
        y=np.sin(theta).tolist(),
        mode="lines",
        line=dict(color="#334455", width=2),
        showlegend=False,
        hoverinfo="skip",
    ))

    # Nose marker
    fig.add_trace(go.Scatter(
        x=[0], y=[1.05],
        mode="markers",
        marker=dict(symbol="triangle-up", size=12, color="#334455"),
        showlegend=False,
        hoverinfo="skip",
    ))

    # Electrode markers coloured by band power
    vmin, vmax = min(zs), max(zs)
    fig.add_trace(go.Scatter(
        x=xs,
        y=ys,
        mode="markers+text",
        marker=dict(
            size=30,
            color=zs,
            colorscale=colorscale,
            cmin=vmin,
            cmax=vmax,
            showscale=True,
            colorbar=dict(
                title="µV²",
                thickness=12,
                len=0.8,
                tickfont=dict(color="white", size=10),
                titlefont=dict(color="white", size=10),
            ),
            line=dict(color="white", width=1),
        ),
        text=[ch for ch in CHANNEL_ORDER if ch in band_powers],
        textposition="top center",
        textfont=dict(color="white", size=9),
        hovertext=labels,
        hoverinfo="text",
        showlegend=False,
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(color="#00b4d8", size=13)),
        plot_bgcolor="#0d1b2a",
        paper_bgcolor="#0d1b2a",
        height=height,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(
            range=[-1.2, 1.2], showgrid=False, zeroline=False,
            showticklabels=False, color="white",
        ),
        yaxis=dict(
            range=[-1.2, 1.2], showgrid=False, zeroline=False,
            showticklabels=False, scaleanchor="x", scaleratio=1,
        ),
    )
    return fig


def create_power_time_series(
    timestamps: list,
    alpha_powers: list,
    beta_powers: list,
    height: int = 200,
) -> go.Figure:
    """
    Time series of alpha and beta band power for neurofeedback.

    Shows whether the user is successfully modulating their
    brain oscillations toward the target band.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps, y=alpha_powers,
        mode="lines", name="Alpha (8–13 Hz)",
        line=dict(color="#00b4d8", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=timestamps, y=beta_powers,
        mode="lines", name="Beta (13–30 Hz)",
        line=dict(color="#06d6a0", width=2),
    ))
    fig.update_layout(
        plot_bgcolor="#0d1b2a",
        paper_bgcolor="#0d1b2a",
        height=height,
        margin=dict(l=40, r=10, t=20, b=30),
        legend=dict(font=dict(color="white", size=10), bgcolor="#122236"),
        xaxis=dict(title="Time (s)", color="white", gridcolor="#1e2d3d"),
        yaxis=dict(title="Power (µV²)", color="white", gridcolor="#1e2d3d"),
    )
    return fig


def _empty_topoplot(title: str) -> go.Figure:
    """Return empty figure when no data available."""
    fig = go.Figure()
    fig.add_annotation(
        text="No EEG data",
        x=0.5, y=0.5,
        xref="paper", yref="paper",
        showarrow=False,
        font=dict(color="#4a6080", size=14),
    )
    fig.update_layout(
        title=title,
        plot_bgcolor="#0d1b2a",
        paper_bgcolor="#0d1b2a",
        height=300,
    )
    return fig


def simulate_band_powers() -> dict:
    """Generate realistic simulated band powers for all channels."""
    np.random.seed(int(42))
    return {
        ch: float(np.random.uniform(5.0, 30.0))
        for ch in CHANNEL_ORDER
    }