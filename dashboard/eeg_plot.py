"""
dashboard/eeg_plot.py
======================
Real-time EEG waveform and spectrum plotting utilities.

Provides reusable Plotly figure builders for:
  - Multi-channel EEG time series (8 channels, offset for clarity)
  - PSD (power spectral density) frequency plot
  - Epoch comparison (pre/post artefact removal)
  - P300 ERP waveform (averaged across trials)

All figures use the dark clinical theme consistent with layout.py.
"""

import numpy as np
import plotly.graph_objects as go
from typing import Optional

CHANNEL_NAMES  = ["Fp1", "Fp2", "C3", "Cz", "C4", "P3", "P4", "Oz"]
CHANNEL_COLORS = [
    "#00b4d8", "#7b2fbe", "#06d6a0", "#ffd166",
    "#f4a261", "#ef233c", "#028090", "#1e90ff",
]
CHANNEL_OFFSET_UV = 50.0   # µV vertical offset between channels


def create_eeg_timeseries(
    data: np.ndarray,
    sample_rate: float = 250.0,
    title: str = "Live EEG — 8 Channels",
    height: int = 300,
    show_channel_labels: bool = True,
) -> go.Figure:
    """
    Multi-channel EEG waveform plot with vertical offsets.

    Args:
        data:        (n_channels, n_samples) float32 array
        sample_rate: Hz
        title:       plot title
        height:      figure height px

    Returns:
        Plotly Figure
    """
    n_channels, n_samples = data.shape
    duration = n_samples / sample_rate
    t = np.linspace(0, duration, n_samples)

    fig = go.Figure()
    for ch in range(min(n_channels, len(CHANNEL_NAMES))):
        offset = ch * CHANNEL_OFFSET_UV
        signal = data[ch] + offset
        fig.add_trace(go.Scatter(
            x=t.tolist(),
            y=signal.tolist(),
            mode="lines",
            name=CHANNEL_NAMES[ch],
            line=dict(color=CHANNEL_COLORS[ch], width=1),
            hovertemplate=(
                f"<b>{CHANNEL_NAMES[ch]}</b><br>"
                "Time: %{x:.3f}s<br>"
                "Amplitude: %{customdata:.1f} µV"
                "<extra></extra>"
            ),
            customdata=data[ch].tolist(),
        ))

    # Channel label annotations on y-axis
    if show_channel_labels:
        for ch in range(min(n_channels, len(CHANNEL_NAMES))):
            offset = ch * CHANNEL_OFFSET_UV
            fig.add_annotation(
                x=0, y=offset,
                text=CHANNEL_NAMES[ch],
                xref="x", yref="y",
                showarrow=False,
                font=dict(color=CHANNEL_COLORS[ch], size=9),
                xanchor="right",
            )

    fig.update_layout(
        title=dict(text=title, font=dict(color="#00b4d8", size=13)),
        plot_bgcolor="#0d1b2a",
        paper_bgcolor="#0d1b2a",
        height=height,
        margin=dict(l=50, r=10, t=35, b=30),
        showlegend=False,
        xaxis=dict(
            title="Time (s)",
            color="white",
            gridcolor="#1e2d3d",
            gridwidth=0.5,
        ),
        yaxis=dict(
            title="µV + offset",
            color="white",
            gridcolor="#1e2d3d",
            gridwidth=0.5,
            showticklabels=False,
        ),
    )
    return fig


def create_psd_plot(
    data: np.ndarray,
    sample_rate: float = 250.0,
    title: str = "Power Spectral Density",
    height: int = 250,
    freq_max: float = 50.0,
) -> go.Figure:
    """
    PSD plot for all channels using Welch method.

    Highlights key frequency bands:
      Delta  1–4 Hz   (grey)
      Theta  4–8 Hz   (purple)
      Alpha  8–13 Hz  (cyan)   ← key for motor imagery
      Beta   13–30 Hz (green)  ← ERD during motor imagery
    """
    from scipy.signal import welch

    fig = go.Figure()

    # Band shading
    bands = [
        (1, 4,   "#334455", "Delta"),
        (4, 8,   "#4a2060", "Theta"),
        (8, 13,  "#004466", "Alpha"),
        (13, 30, "#003322", "Beta"),
    ]
    for low, high, color, name in bands:
        fig.add_vrect(
            x0=low, x1=high,
            fillcolor=color, opacity=0.3,
            line_width=0,
            annotation_text=name,
            annotation_position="top left",
            annotation_font=dict(color="white", size=8),
        )

    # Per-channel PSD
    n_per_seg = min(256, data.shape[1])
    for ch in range(min(data.shape[0], len(CHANNEL_NAMES))):
        freqs, psd = welch(data[ch], fs=sample_rate, nperseg=n_per_seg)
        mask = freqs <= freq_max
        psd_db = 10 * np.log10(psd[mask] + 1e-12)

        fig.add_trace(go.Scatter(
            x=freqs[mask].tolist(),
            y=psd_db.tolist(),
            mode="lines",
            name=CHANNEL_NAMES[ch],
            line=dict(color=CHANNEL_COLORS[ch], width=1.2),
            opacity=0.8,
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(color="#00b4d8", size=13)),
        plot_bgcolor="#0d1b2a",
        paper_bgcolor="#0d1b2a",
        height=height,
        margin=dict(l=50, r=10, t=35, b=30),
        legend=dict(
            font=dict(color="white", size=9),
            bgcolor="#122236",
            orientation="h",
            y=-0.2,
        ),
        xaxis=dict(
            title="Frequency (Hz)",
            color="white",
            gridcolor="#1e2d3d",
            range=[0, freq_max],
        ),
        yaxis=dict(
            title="Power (dB)",
            color="white",
            gridcolor="#1e2d3d",
        ),
    )
    return fig


def create_erp_plot(
    erp_target: np.ndarray,
    erp_nontarget: np.ndarray,
    sample_rate: float = 250.0,
    channel_idx: int = 5,   # Pz = channel 5
    title: str = "P300 ERP (Pz channel)",
    height: int = 220,
) -> go.Figure:
    """
    P300 ERP waveform — target vs non-target comparison.

    Shows the characteristic P300 positive deflection at ~300ms
    for target stimuli, absent in non-target stimuli.
    """
    n_samples = erp_target.shape[1] if erp_target.ndim > 1 else len(erp_target)
    t_ms = np.linspace(0, n_samples / sample_rate * 1000, n_samples)

    target_trace    = erp_target[channel_idx]    if erp_target.ndim > 1    else erp_target
    nontarget_trace = erp_nontarget[channel_idx] if erp_nontarget.ndim > 1 else erp_nontarget

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t_ms.tolist(), y=target_trace.tolist(),
        mode="lines", name="Target (P300 present)",
        line=dict(color="#06d6a0", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=t_ms.tolist(), y=nontarget_trace.tolist(),
        mode="lines", name="Non-target",
        line=dict(color="#ef233c", width=2),
    ))

    # P300 marker at 300ms
    fig.add_vline(
        x=300, line_dash="dash", line_color="#ffd166",
        annotation_text="P300 (300ms)",
        annotation_font=dict(color="#ffd166", size=10),
    )
    fig.add_hline(y=0, line_color="#334455", line_width=0.8)

    fig.update_layout(
        title=dict(text=title, font=dict(color="#00b4d8", size=13)),
        plot_bgcolor="#0d1b2a",
        paper_bgcolor="#0d1b2a",
        height=height,
        margin=dict(l=50, r=10, t=35, b=30),
        legend=dict(font=dict(color="white", size=10), bgcolor="#122236"),
        xaxis=dict(title="Time (ms)", color="white", gridcolor="#1e2d3d"),
        yaxis=dict(title="Amplitude (µV)", color="white", gridcolor="#1e2d3d"),
    )
    return fig


def generate_demo_eeg(n_channels: int = 8, n_samples: int = 500) -> np.ndarray:
    """Generate realistic demo EEG data for dashboard previews."""
    np.random.seed(42)
    data = np.zeros((n_channels, n_samples), dtype=np.float32)
    t = np.linspace(0, n_samples / 250.0, n_samples)
    for ch in range(n_channels):
        data[ch] = np.random.randn(n_samples) * 3.0
        data[ch] += 15.0 * np.sin(2 * np.pi * 10.0 * t + ch * 0.3)  # alpha
        data[ch] += 5.0  * np.sin(2 * np.pi * 20.0 * t + ch * 0.1)  # beta
    return data