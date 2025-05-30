import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
import time
import plotly.graph_objs as go

# Configure the page
st.set_page_config(page_title="Realtime PD Monitor", layout="wide")
st.title("🔴 Real-Time Partial Discharge Monitor")
st.markdown("Simulates a continuous stream of `hold_signal` with windowed PD analysis and live signal view.")

# File upload
uploaded_file = st.file_uploader("Upload a `hold_signal.txt` file:", type=['txt'])

if uploaded_file:
    fs = 200e6  # Hz
    hold_signal = np.loadtxt(uploaded_file, delimiter=',')
    t = np.arange(len(hold_signal)) / fs

    st.success(f"Signal loaded: {len(hold_signal)} samples, fs = {fs / 1e6:.2f} MHz")

    col1, col2 = st.columns(2)
    with col1:
        win_ms = st.slider("Window size (ms)", 5, 100, 20)
    with col2:
        threshold = st.slider("Detection threshold (V)", 0.001, 0.1, 0.01, step=0.001)

    win_samples = int(fs * (win_ms / 1000))
    total_windows = len(hold_signal) // win_samples
    display_windows = 100  # number of recent windows to keep

    placeholder_metrics = st.empty()
    placeholder_plot = st.empty()
    placeholder_signal = st.empty()
    progress_bar = st.progress(0)

    metrics_hist = []

    # Total number of windows
    total_windows = len(hold_signal) // win_samples

    # Slider for manual control
    selected_window = st.slider("Select window", 0, total_windows - 1, 0)

    # Extract selected segment
    start = selected_window * win_samples
    end = (selected_window + 1) * win_samples
    segment = hold_signal[start:end]
    segment_time = t[start:end]
    dt = win_samples / fs

    # Detect pulses in the segment
    crossings = np.where(np.diff(segment > threshold) == 1)[0] + 1
    pulses = segment[crossings] if len(crossings) > 0 else np.array([0])
    pulse_times = segment_time[crossings] if len(crossings) > 0 else np.array([0])
    time_diffs = np.diff(pulse_times) if len(pulse_times) > 1 else np.array([0])

    # Calculate metrics for the current window
    metrics = {
        "Window": selected_window,
        "Pulse Count": len(pulses),
        "Max Amp (V)": np.max(pulses),
        "Min Amp (V)": np.min(pulses),
        "Mean Amp (V)": np.mean(pulses),
        "Median Amp (V)": np.median(pulses),
        "Std Dev": np.std(pulses),
        "Peak-to-Peak (V)": np.ptp(pulses),
        "Repetition Rate (Hz)": len(pulses) / dt,
        "Skewness": skew(pulses),
        "Kurtosis": kurtosis(pulses),
        "Energy": np.sum(pulses ** 2),
        "First Pulse Time (µs)": pulse_times[0] * 1e6 if len(pulse_times) > 0 else 0,
        "Mean Interval (µs)": np.mean(time_diffs) * 1e6 if len(time_diffs) > 0 else 0
    }

    # Display metrics
    st.subheader("📊 Metrics for Selected Window")
    st.write(pd.DataFrame([metrics]).T.rename(columns={0: "Value"}).style.format("{:.4f}"))

    # Signal plot with highlighted window
    fig3 = go.Figure()

    # Downsample full signal
    max_points = 10000
    ds_factor = max(1, len(hold_signal) // max_points)
    t_ds = t[::ds_factor]
    s_ds = hold_signal[::ds_factor]

    fig3.add_trace(go.Scatter(x=t_ds * 1e3, y=s_ds,
                              mode='lines', name='Hold Signal', line=dict(color='gray')))

    fig3.add_vrect(
        x0=t[start] * 1e3, x1=t[end - 1] * 1e3,
        fillcolor='rgba(255, 0, 0, 0.3)', opacity=0.5,
        layer="below", line_width=0
    )

    fig3.update_layout(
        title="Signal with Selected Window Highlighted",
        xaxis_title="Time (ms)",
        yaxis_title="Amplitude (V)",
        showlegend=False
    )

    st.plotly_chart(fig3, use_container_width=True)

    st.success("✅ Simulated transmission completed.")
    progress_bar.empty()
