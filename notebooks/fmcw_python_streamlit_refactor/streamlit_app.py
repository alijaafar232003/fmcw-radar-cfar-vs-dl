"""
FMCW Radar — Range-Doppler Map Interactive Streamlit App
=========================================================
Real-time RD map with configurable radar parameters, multiple targets, and SNR control.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy import signal
from scipy.ndimage import maximum_filter
from scipy.constants import c, pi

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FMCW Radar — Range-Doppler Map",
    page_icon="📡",
    layout="wide",
)

st.markdown("""
<style>
    .radar-header {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 20px 30px;
        margin-bottom: 20px;
    }
    .radar-header h1 { color: #58a6ff; margin: 0; font-size: 2rem; }
    .radar-header p  { color: #8b949e; margin: 4px 0 0; }
    .metric-box {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px 16px;
        text-align: center;
    }
    .metric-box .label { color: #8b949e; font-size: 0.78rem; margin-bottom: 2px; }
    .metric-box .value { color: #58a6ff; font-size: 1.15rem; font-weight: 700; }
    .warn-box {
        background: #3d1f00;
        border: 1px solid #d29922;
        border-radius: 8px;
        padding: 10px 14px;
        color: #e3b341;
        font-size: 0.85rem;
        margin: 6px 0;
    }
    .err-box {
        background: #3d0000;
        border: 1px solid #f85149;
        border-radius: 8px;
        padding: 10px 14px;
        color: #f85149;
        font-size: 0.85rem;
        margin: 6px 0;
    }
    div[data-testid="stNumberInput"] label { font-size: 0.82rem !important; }
    .stSlider label { font-size: 0.82rem !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="radar-header">
  <h1>📡 FMCW Radar — Range-Doppler Map</h1>
  <p>Real-time 24 GHz drone radar simulator · Add targets · Tune parameters · Watch the RD map respond</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Session-state: target list
# ─────────────────────────────────────────────────────────────────────────────
if "targets" not in st.session_state:
    st.session_state.targets = [
        {"range": 30.0, "velocity": 10.0, "power": 20.0},
        {"range": 55.0, "velocity": -6.0, "power": 10.0},
        {"range": 80.0, "velocity":  3.0, "power": -8.0},
    ]

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — radar parameters
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Radar Parameters")

    f_ghz = st.slider("Carrier frequency (GHz)", 1.0, 79.0, 24.0, 0.5,
                      help="f ↑ → λ ↓ → velocity range ↓")
    bw_mhz = st.slider("Bandwidth (MHz)", 50.0, 2000.0, 500.0, 50.0,
                       help="BW ↑ → range resolution ↓ (better)")
    tc_us = st.slider("Chirp duration Tc (µs)", 5.0, 500.0, 40.0, 5.0,
                      help="Tc ↑ → max velocity ↓, velocity resolution ↑")
    M = st.select_slider("Chirps in CPI (M)", [32, 64, 128, 256, 512], 128,
                         help="M ↑ → better velocity resolution")
    N = st.select_slider("Samples per chirp (N)", [256, 512, 1024, 2048, 4096], 2048,
                         help="N ↑ → max range ↑, finer range bins")

    st.markdown("---")
    st.markdown("## 🌊 Noise")
    snr_db = st.slider("SNR (dB)", -20.0, 40.0, 10.0, 1.0,
                       help="Signal-to-noise ratio. Lower = noisier RD map")
    noise_seed = st.number_input("Noise seed", min_value=0, max_value=9999, value=42, step=1)

    st.markdown("---")
    st.markdown("## 🎨 Display")
    colormap = st.selectbox("Colormap", ["Inferno", "Plasma", "Viridis", "Hot", "Jet", "RdBu"], index=0)
    vmin_db = st.slider("Color floor (dB)", -100.0, -10.0, -60.0, 5.0)
    vmax_db = st.slider("Color ceiling (dB)", -10.0, 10.0, 0.0, 1.0)
    show_markers = st.checkbox("Show target markers", value=True)
    show_range_fft = st.checkbox("Show Range FFT panel", value=True)

# ─────────────────────────────────────────────────────────────────────────────
# Derived radar parameters
# ─────────────────────────────────────────────────────────────────────────────
f        = f_ghz * 1e9
bw       = bw_mhz * 1e6
Tc       = tc_us * 1e-6
wavelength = c / f
slope    = bw / Tc
fs       = N / Tc
range_res = c / (2 * bw)
rmax     = c * fs / (4 * slope)
max_vel  = wavelength / (4 * Tc)
vel_res  = wavelength / (2 * M * Tc)

# ─────────────────────────────────────────────────────────────────────────────
# Derived metric display
# ─────────────────────────────────────────────────────────────────────────────
m1, m2, m3, m4, m5, m6 = st.columns(6)
metrics = [
    (m1, "Range res", f"{range_res:.2f} m"),
    (m2, "Max range", f"{rmax:.0f} m"),
    (m3, "Max vel ±", f"{max_vel:.2f} m/s"),
    (m4, "Vel res", f"{vel_res:.3f} m/s"),
    (m5, "ADC fs", f"{fs/1e6:.1f} MHz"),
    (m6, "λ", f"{wavelength*1e3:.2f} mm"),
]
for col, label, value in metrics:
    with col:
        st.markdown(f"""<div class="metric-box">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Target editor
# ─────────────────────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 2], gap="large")

with left_col:
    st.markdown("### 🎯 Targets")
    st.caption("Add, edit, or remove targets. Changes update the RD map instantly.")

    targets_to_delete = []

    for i, tgt in enumerate(st.session_state.targets):
        with st.expander(f"Target {i+1}  —  {tgt['range']:.0f} m, {tgt['velocity']:+.1f} m/s, {tgt['power']:+.0f} dB",
                         expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                new_r = st.number_input(
                    "Range (m)", min_value=1.0, max_value=rmax * 0.98,
                    value=float(tgt["range"]), step=1.0, key=f"r_{i}"
                )
            with c2:
                new_v = st.number_input(
                    "Velocity (m/s)", min_value=-max_vel * 0.98, max_value=max_vel * 0.98,
                    value=float(np.clip(tgt["velocity"], -max_vel * 0.98, max_vel * 0.98)),
                    step=0.5, key=f"v_{i}"
                )
            with c3:
                new_p = st.number_input(
                    "Power (dB)", min_value=-40.0, max_value=40.0,
                    value=float(tgt["power"]), step=1.0, key=f"p_{i}"
                )

            st.session_state.targets[i] = {"range": new_r, "velocity": new_v, "power": new_p}

            if st.button(f"🗑️ Remove", key=f"del_{i}", use_container_width=True):
                targets_to_delete.append(i)

    # Process deletions
    for idx in sorted(targets_to_delete, reverse=True):
        st.session_state.targets.pop(idx)
    if targets_to_delete:
        st.rerun()

    if st.button("➕ Add Target", use_container_width=True, type="primary"):
        st.session_state.targets.append({"range": 40.0, "velocity": 0.0, "power": 0.0})
        st.rerun()

    if st.button("🔄 Reset to Defaults", use_container_width=True):
        st.session_state.targets = [
            {"range": 30.0, "velocity": 10.0, "power": 20.0},
            {"range": 55.0, "velocity": -6.0, "power": 10.0},
            {"range": 80.0, "velocity":  3.0, "power": -8.0},
        ]
        st.rerun()

    # Validation warnings
    st.markdown("#### Validation")
    all_valid = True
    for i, tgt in enumerate(st.session_state.targets):
        r, v, p = tgt["range"], tgt["velocity"], tgt["power"]
        fb = 2 * r * slope / c
        if fb >= fs / 2:
            st.markdown(f'<div class="err-box">❌ Target {i+1}: beat frequency {fb/1e6:.2f} MHz exceeds Nyquist {fs/2/1e6:.2f} MHz. Reduce range or increase N.</div>', unsafe_allow_html=True)
            all_valid = False
        if abs(v) >= max_vel:
            st.markdown(f'<div class="err-box">❌ Target {i+1}: |vel| {abs(v):.2f} m/s exceeds max ±{max_vel:.2f} m/s.</div>', unsafe_allow_html=True)
            all_valid = False
        if r > rmax:
            st.markdown(f'<div class="warn-box">⚠️ Target {i+1}: range {r:.0f} m > max range {rmax:.0f} m.</div>', unsafe_allow_html=True)

    if not st.session_state.targets:
        st.markdown('<div class="warn-box">⚠️ No targets defined. Add at least one target.</div>', unsafe_allow_html=True)
        all_valid = False

    if all_valid and st.session_state.targets:
        st.success(f"✅ {len(st.session_state.targets)} target(s) valid")

# ─────────────────────────────────────────────────────────────────────────────
# Signal processing
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def compute_rd(f, bw, Tc, M, N, snr_db, noise_seed, targets_tuple):
    """Full FMCW processing chain. Cached on all parameters."""
    wavelength = c / f
    slope      = bw / Tc
    fs         = N / Tc
    range_res  = c / (2 * bw)
    rmax       = c * fs / (4 * slope)

    t_fast  = np.arange(N) / fs
    t_abs   = np.arange(M)[:, None] * Tc + t_fast[None, :]
    t_chirp = t_fast[None, :]

    freq_axis    = np.fft.fftfreq(N, d=1/fs)[:N//2]
    range_axis   = freq_axis * c / (2 * slope)
    doppler_axis = np.fft.fftshift(np.fft.fftfreq(M, d=Tc))
    velocity_axis = doppler_axis * wavelength / 2

    # TX
    tx = np.exp(1j * 2*pi * (f * t_abs + 0.5 * slope * t_chirp**2))

    # RX
    rx = np.zeros((M, N), dtype=np.complex128)
    for r0, v, amp_db in targets_tuple:
        amp = 10 ** (amp_db / 20)
        R_t = r0 + v * t_abs
        tau = 2 * R_t / c
        rx += amp * np.exp(1j * 2*pi * (
            f * (t_abs - tau) + 0.5 * slope * (t_chirp - tau)**2
        ))

    cpi = tx * np.conj(rx)

    # AWGN
    rng = np.random.default_rng(int(noise_seed))
    sig_power = np.mean(np.abs(cpi)**2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (
        rng.standard_normal(cpi.shape) + 1j * rng.standard_normal(cpi.shape)
    )
    cpi_noisy = cpi + noise

    # 2D windowing + FFT
    rw = signal.windows.blackman(N)
    dw = signal.windows.blackman(M)
    w2d = dw[:, None] * rw[None, :]
    cpi_w = cpi_noisy * w2d

    range_fft = np.fft.fft(cpi_w, axis=1)[:, :N//2]
    rd_complex = np.fft.fftshift(np.fft.fft(range_fft, axis=0), axes=0)
    rd_mag = np.abs(rd_complex).T

    peak = np.max(rd_mag)
    rd_db = 20 * np.log10(rd_mag / peak + 1e-12)

    # Range FFT (chirp 0, zero-padded)
    zp = N * 4
    freq_zp = np.fft.fftfreq(zp, d=1/fs)[:zp//2]
    range_zp = freq_zp * c / (2 * slope)
    X_noisy = np.fft.fft(cpi_noisy[0] * rw, n=zp)[:zp//2]
    rfft_db = 20 * np.log10(np.abs(X_noisy) / (np.max(np.abs(X_noisy)) + 1e-12) + 1e-12)

    # Clean CPI for noise floor reference
    cpi_clean = tx * np.conj(rx)
    cpi_cw = cpi_clean * w2d
    rfft_clean = np.fft.fft(cpi_clean[0] * rw, n=zp)[:zp//2]
    rfft_clean_db = 20 * np.log10(np.abs(rfft_clean) / (np.max(np.abs(rfft_clean)) + 1e-12) + 1e-12)

    return {
        "rd_db": rd_db,
        "rd_mag": rd_mag,
        "range_axis": range_axis,
        "velocity_axis": velocity_axis,
        "range_zp": range_zp,
        "rfft_db": rfft_db,
        "rfft_clean_db": rfft_clean_db,
        "noise_floor": float(np.median(rd_db)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CFAR processing
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def apply_cfar(rd_mag, range_axis, velocity_axis, targets_tuple, cfar_type, guard_r, guard_d, ref_r, ref_d, desired_pfa, os_rank_fraction, match_tol_r, match_tol_d, use_local_max):
    rd_power = rd_mag ** 2
    num_r, num_d = rd_power.shape

    threshold_map = np.full_like(rd_power, np.nan, dtype=float)
    detection_mask = np.zeros_like(rd_power, dtype=bool)

    window_r = 2 * (ref_r + guard_r) + 1
    window_d = 2 * (ref_d + guard_d) + 1
    guard_window_r = 2 * guard_r + 1
    guard_window_d = 2 * guard_d + 1
    num_ref_cells = window_r * window_d - guard_window_r * guard_window_d

    alpha = num_ref_cells * (desired_pfa ** (-1 / num_ref_cells) - 1)
    os_rank = int(np.ceil(os_rank_fraction * num_ref_cells))
    os_rank = int(np.clip(os_rank, 1, num_ref_cells))

    for r in range(ref_r + guard_r, num_r - ref_r - guard_r):
        for d in range(ref_d + guard_d, num_d - ref_d - guard_d):
            r0 = r - ref_r - guard_r
            r1 = r + ref_r + guard_r + 1
            d0 = d - ref_d - guard_d
            d1 = d + ref_d + guard_d + 1

            window = rd_power[r0:r1, d0:d1].copy()
            cut_r = ref_r + guard_r
            cut_d = ref_d + guard_d

            window[
                cut_r - guard_r: cut_r + guard_r + 1,
                cut_d - guard_d: cut_d + guard_d + 1
            ] = np.nan

            ref_cells = window[~np.isnan(window)]

            if cfar_type == "CA":
                noise_level = np.mean(ref_cells)
            elif cfar_type == "GO":
                left_cells = window[:, :cut_d - guard_d]
                right_cells = window[:, cut_d + guard_d + 1:]
                noise_level = max(np.nanmean(left_cells), np.nanmean(right_cells))
            elif cfar_type == "OS":
                noise_level = np.sort(ref_cells)[os_rank - 1]
            else:
                raise ValueError("cfar_type must be 'CA', 'GO', or 'OS'")

            threshold = alpha * noise_level
            threshold_map[r, d] = threshold
            detection_mask[r, d] = rd_power[r, d] > threshold

    if use_local_max:
        neighborhood = (2 * guard_r + 1, 2 * guard_d + 1)
        local_peak_mask = rd_power == maximum_filter(rd_power, size=neighborhood, mode="nearest")
        detection_mask &= local_peak_mask

    rd_power_db = 10 * np.log10(rd_power / np.nanmax(rd_power) + 1e-12)
    threshold_db = 10 * np.log10(threshold_map / np.nanmax(rd_power) + 1e-12)
    detected_bins = np.argwhere(detection_mask)

    target_bins = []
    for r_true, v_true, amp_db in targets_tuple:
        r_bin = int(np.argmin(np.abs(range_axis - r_true)))
        d_bin = int(np.argmin(np.abs(velocity_axis - v_true)))
        target_bins.append((r_bin, d_bin, r_true, v_true, amp_db))

    matched_targets = set()
    matched_detections = set()
    target_rows = []

    for ti, (tr, td, r_true, v_true, amp_db) in enumerate(target_bins):
        candidates = []
        for di, (dr, dd) in enumerate(detected_bins):
            if abs(dr - tr) <= match_tol_r and abs(dd - td) <= match_tol_d:
                candidates.append((di, dr, dd, rd_power[dr, dd]))

        if candidates:
            di, dr, dd, _ = max(candidates, key=lambda x: x[3])
            matched_targets.add(ti)
            matched_detections.add(di)
            target_rows.append({
                "Target": f"T{ti+1}",
                "Status": "Detected",
                "True R (m)": round(float(r_true), 2),
                "Pred R (m)": round(float(range_axis[dr]), 2),
                "ΔR (m)": round(float(range_axis[dr] - r_true), 2),
                "True V (m/s)": round(float(v_true), 2),
                "Pred V (m/s)": round(float(velocity_axis[dd]), 2),
                "ΔV (m/s)": round(float(velocity_axis[dd] - v_true), 2),
                "Peak (dB)": round(float(rd_power_db[dr, dd]), 1),
            })
        else:
            target_rows.append({
                "Target": f"T{ti+1}",
                "Status": "Missed",
                "True R (m)": round(float(r_true), 2),
                "Pred R (m)": "—",
                "ΔR (m)": "—",
                "True V (m/s)": round(float(v_true), 2),
                "Pred V (m/s)": "—",
                "ΔV (m/s)": "—",
                "Peak (dB)": "—",
            })

    num_true_targets = len(target_bins)
    num_detections = len(detected_bins)
    num_correct = len(matched_targets)
    num_missed = num_true_targets - num_correct
    num_false_alarms = num_detections - len(matched_detections)
    valid_cells = int(np.sum(~np.isnan(threshold_map)))
    measured_pfa = num_false_alarms / valid_cells if valid_cells else np.nan

    return {
        "rd_power_db": rd_power_db,
        "threshold_db": threshold_db,
        "detection_mask": detection_mask,
        "detected_bins": detected_bins,
        "target_bins": target_bins,
        "target_rows": target_rows,
        "matched_detections": sorted(matched_detections),
        "summary": {
            "CFAR type": cfar_type,
            "Reference cells used": num_ref_cells,
            "OS rank": os_rank if cfar_type == "OS" else None,
            "True targets": num_true_targets,
            "Total detections": num_detections,
            "Correct detections": num_correct,
            "Missed targets": num_missed,
            "False alarms": num_false_alarms,
            "Measured Pfa": measured_pfa,
            "Valid CUT cells": valid_cells,
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# Render plots
# ─────────────────────────────────────────────────────────────────────────────
with right_col:

    if not st.session_state.targets or not all_valid:
        st.info("Add valid targets to see the Range-Doppler map.")
    else:
        targets_tuple = tuple(
            (t["range"], t["velocity"], t["power"])
            for t in st.session_state.targets
        )

        with st.spinner("Computing Range-Doppler map…"):
            res = compute_rd(f, bw, Tc, M, N, snr_db, noise_seed, targets_tuple)

        rd_db       = res["rd_db"]
        rd_mag      = res["rd_mag"]
        range_axis  = res["range_axis"]
        vel_axis    = res["velocity_axis"]
        noise_floor = res["noise_floor"]

        r_max_plot = max(t["range"] for t in st.session_state.targets) * 1.3

        # ── RD Map ────────────────────────────────────────────────────────────
        cmap_map = {
            "Inferno": "Inferno", "Plasma": "Plasma", "Viridis": "Viridis",
            "Hot": "Hot", "Jet": "Jet", "RdBu": "RdBu_r"
        }
        cmap = cmap_map[colormap]

        fig_rd = go.Figure()

        fig_rd.add_trace(go.Heatmap(
            z=rd_db,
            x=vel_axis,
            y=range_axis,
            colorscale=cmap,
            zmin=vmin_db,
            zmax=vmax_db,
            colorbar=dict(
                title="dB",
                thickness=14,
                len=0.85,
                tickfont=dict(size=11),
            ),
            hovertemplate="Vel: %{x:.2f} m/s<br>Range: %{y:.1f} m<br>Mag: %{z:.1f} dB<extra></extra>",
        ))

        if show_markers:
            for i, tgt in enumerate(st.session_state.targets):
                fig_rd.add_trace(go.Scatter(
                    x=[tgt["velocity"]],
                    y=[tgt["range"]],
                    mode="markers+text",
                    marker=dict(symbol="x", size=14, color="cyan", line=dict(width=2.5)),
                    text=[f"T{i+1}"],
                    textposition="top right",
                    textfont=dict(color="cyan", size=12),
                    name=f"T{i+1}: {tgt['range']:.0f} m, {tgt['velocity']:+.1f} m/s",
                    hovertemplate=f"Target {i+1}<br>Range: {tgt['range']:.0f} m<br>Vel: {tgt['velocity']:+.1f} m/s<br>Power: {tgt['power']:+.0f} dB<extra></extra>",
                ))

        fig_rd.update_layout(
            title=dict(
                text=f"Range-Doppler Map — SNR = {snr_db:.1f} dB  |  Noise floor: {noise_floor:.1f} dB  |  Peak-to-floor: {-noise_floor:.1f} dB",
                font=dict(size=14, color="#c9d1d9"),
            ),
            xaxis=dict(
                title=dict(text="Velocity (m/s)", font=dict(color="#8b949e")),
                range=[-max_vel, max_vel],
                gridcolor="#30363d",
                color="#8b949e",
            ),
            yaxis=dict(
                title=dict(text="Range (m)", font=dict(color="#8b949e")),
                range=[0, r_max_plot],
                gridcolor="#30363d",
                color="#8b949e",
            ),
            paper_bgcolor="#0d1117",
            plot_bgcolor="#0d1117",
            legend=dict(
                font=dict(color="#8b949e", size=11),
                bgcolor="rgba(22,27,34,0.85)",
                bordercolor="#30363d",
                x=1.08, y=1,
            ),
            height=520,
            margin=dict(l=60, r=160, t=50, b=50),
        )

        st.plotly_chart(fig_rd, use_container_width=True)

        # ── Range FFT ─────────────────────────────────────────────────────────
        if show_range_fft:
            fig_rfft = go.Figure()

            fig_rfft.add_trace(go.Scatter(
                x=res["range_zp"],
                y=res["rfft_clean_db"],
                mode="lines",
                name="Clean",
                line=dict(color="#58a6ff", width=1.2),
            ))
            fig_rfft.add_trace(go.Scatter(
                x=res["range_zp"],
                y=res["rfft_db"],
                mode="lines",
                name=f"Noisy (SNR={snr_db:.0f} dB)",
                line=dict(color="#f78166", width=1.0),
                opacity=0.85,
            ))

            for i, tgt in enumerate(st.session_state.targets):
                fig_rfft.add_vline(
                    x=tgt["range"],
                    line=dict(color="cyan", dash="dash", width=1.2),
                    annotation_text=f"T{i+1}",
                    annotation_font_color="cyan",
                    annotation_font_size=11,
                )

            r_max_rfft = max(t["range"] for t in st.session_state.targets) * 1.4

            fig_rfft.update_layout(
                title=dict(text="Range FFT — Chirp 0 (zero-padded 4×)", font=dict(size=13, color="#c9d1d9")),
                xaxis=dict(
                    title=dict(text="Range (m)", font=dict(color="#8b949e")),
                    range=[0, r_max_rfft],
                    gridcolor="#30363d", color="#8b949e",
                ),
                yaxis=dict(
                    title=dict(text="Relative magnitude (dB)", font=dict(color="#8b949e")),
                    range=[-90, 5],
                    gridcolor="#30363d", color="#8b949e",
                ),
                paper_bgcolor="#0d1117",
                plot_bgcolor="#0d1117",
                legend=dict(font=dict(color="#8b949e", size=11), bgcolor="rgba(22,27,34,0.85)", bordercolor="#30363d"),
                height=280,
                margin=dict(l=60, r=30, t=45, b=50),
            )

            st.plotly_chart(fig_rfft, use_container_width=True)

        # ── Peak validation table ─────────────────────────────────────────────
        with st.expander("📊 Peak Validation Table", expanded=False):
            rd_mag_lin = 10 ** (rd_db / 20)

            rows = []
            for i, tgt in enumerate(st.session_state.targets):
                r, v = tgt["range"], tgt["velocity"]
                ri = int(np.argmin(np.abs(range_axis - r)))
                vi = int(np.argmin(np.abs(vel_axis - v)))
                sr, sv = 8, 4
                r0, r1 = max(0, ri - sr), min(rd_mag_lin.shape[0] - 1, ri + sr)
                v0, v1 = max(0, vi - sv), min(rd_mag_lin.shape[1] - 1, vi + sv)
                patch = rd_mag_lin[r0:r1+1, v0:v1+1]
                pr, pv = np.unravel_index(patch.argmax(), patch.shape)
                r_est = float(range_axis[r0 + pr])
                v_est = float(vel_axis[v0 + pv])
                rel_db = float(20 * np.log10(patch.max() / rd_mag_lin.max() + 1e-12))
                rows.append({
                    "Target": f"T{i+1}",
                    "R_exp (m)": f"{r:.2f}",
                    "R_est (m)": f"{r_est:.2f}",
                    "ΔR (m)": f"{r_est - r:+.2f}",
                    "V_exp (m/s)": f"{v:.2f}",
                    "V_est (m/s)": f"{v_est:.2f}",
                    "ΔV (m/s)": f"{v_est - v:+.2f}",
                    "Rel (dB)": f"{rel_db:.1f}",
                })

            st.markdown(f"**Range res:** {range_res:.3f} m &emsp; **Vel res:** {vel_res:.4f} m/s")
            st.table(rows)

        # ── CFAR Detection ───────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🎯 CFAR Detection")
        st.caption("Choose CFAR type, tune the cells/Pfa, then press the button to run detection on the current RD map.")

        cfar_col1, cfar_col2, cfar_col3, cfar_col4 = st.columns(4)

        with cfar_col1:
            cfar_type = st.selectbox("CFAR type", ["CA", "GO", "OS"], index=2)
            desired_pfa = st.number_input("Desired Pfa", min_value=1e-8, max_value=1e-1, value=1e-4, step=1e-4, format="%.1e")

        with cfar_col2:
            guard_r = st.number_input("Guard cells - range", min_value=1, max_value=20, value=2, step=1)
            guard_d = st.number_input("Guard cells - doppler", min_value=1, max_value=20, value=2, step=1)

        with cfar_col3:
            ref_r = st.number_input("Reference cells - range", min_value=2, max_value=50, value=8, step=1)
            ref_d = st.number_input("Reference cells - doppler", min_value=2, max_value=50, value=8, step=1)

        with cfar_col4:
            os_rank_fraction = st.slider("OS rank fraction", 0.1, 0.95, 0.75, 0.05, disabled=(cfar_type != "OS"))
            use_local_max = st.checkbox("Keep local peaks only", value=True)

        tol_col1, tol_col2, tol_col3 = st.columns([1, 1, 2])
        with tol_col1:
            match_tol_r = st.number_input("Match tolerance R bins", min_value=0, max_value=20, value=2, step=1)
        with tol_col2:
            match_tol_d = st.number_input("Match tolerance V bins", min_value=0, max_value=20, value=2, step=1)
        with tol_col3:
            max_plot_range_cfar = st.slider(
                "CFAR plot max range (m)",
                min_value=10.0,
                max_value=float(max(range_axis[-1], 10.0)),
                value=float(min(max(t["range"] for t in st.session_state.targets) * 1.4, range_axis[-1])),
                step=5.0,
            )

        run_cfar = st.button("🚀 Apply CFAR", type="primary", use_container_width=True)

        if run_cfar:
            cfar_res = apply_cfar(
                rd_mag=rd_mag,
                range_axis=range_axis,
                velocity_axis=vel_axis,
                targets_tuple=targets_tuple,
                cfar_type=cfar_type,
                guard_r=int(guard_r),
                guard_d=int(guard_d),
                ref_r=int(ref_r),
                ref_d=int(ref_d),
                desired_pfa=float(desired_pfa),
                os_rank_fraction=float(os_rank_fraction),
                match_tol_r=int(match_tol_r),
                match_tol_d=int(match_tol_d),
                use_local_max=bool(use_local_max),
            )

            summary = cfar_res["summary"]
            s1, s2, s3, s4, s5 = st.columns(5)
            s1.metric("True targets", summary["True targets"])
            s2.metric("Correct", summary["Correct detections"])
            s3.metric("Missed", summary["Missed targets"])
            s4.metric("False alarms", summary["False alarms"])
            s5.metric("Measured Pfa", f"{summary['Measured Pfa']:.2e}")

            det_bins = cfar_res["detected_bins"]
            det_r_bins = det_bins[:, 0] if len(det_bins) else np.array([], dtype=int)
            det_d_bins = det_bins[:, 1] if len(det_bins) else np.array([], dtype=int)

            fig_cfar = go.Figure()
            fig_cfar.add_trace(go.Heatmap(
                z=cfar_res["rd_power_db"],
                x=vel_axis,
                y=range_axis,
                colorscale=cmap,
                zmin=vmin_db,
                zmax=vmax_db,
                colorbar=dict(title="Power dB"),
                hovertemplate="Vel: %{x:.2f} m/s<br>Range: %{y:.1f} m<br>Power: %{z:.1f} dB<extra></extra>",
            ))

            if len(det_bins):
                fig_cfar.add_trace(go.Scatter(
                    x=vel_axis[det_d_bins],
                    y=range_axis[det_r_bins],
                    mode="markers",
                    marker=dict(symbol="circle-open", size=10, color="lime", line=dict(width=2)),
                    name="Predicted CFAR detections",
                ))

            for i, tgt in enumerate(st.session_state.targets):
                fig_cfar.add_trace(go.Scatter(
                    x=[tgt["velocity"]],
                    y=[tgt["range"]],
                    mode="markers+text",
                    marker=dict(symbol="x", size=16, color="cyan", line=dict(width=3)),
                    text=[f"T{i+1}"],
                    textposition="top right",
                    name=f"True T{i+1}",
                ))

            fig_cfar.update_layout(
                title=f"{cfar_type}-CFAR: True Targets vs Predicted Detections",
                xaxis=dict(title="Velocity (m/s)", range=[-max_vel, max_vel], gridcolor="#30363d", color="#8b949e"),
                yaxis=dict(title="Range (m)", range=[0, max_plot_range_cfar], gridcolor="#30363d", color="#8b949e"),
                paper_bgcolor="#0d1117",
                plot_bgcolor="#0d1117",
                legend=dict(font=dict(color="#8b949e", size=11), bgcolor="rgba(22,27,34,0.85)", bordercolor="#30363d"),
                height=560,
                margin=dict(l=60, r=40, t=60, b=50),
            )
            st.plotly_chart(fig_cfar, use_container_width=True)

            target_ranges = [t["range"] for t in st.session_state.targets]
            target_vels = [t["velocity"] for t in st.session_state.targets]
            r_min = max(0, min(target_ranges) - 15)
            r_max = min(range_axis[-1], max(target_ranges) + 15)
            v_min = max(vel_axis[0], min(target_vels) - 15)
            v_max = min(vel_axis[-1], max(target_vels) + 15)

            r_mask = (range_axis >= r_min) & (range_axis <= r_max)
            d_mask = (vel_axis >= v_min) & (vel_axis <= v_max)
            range_focus = range_axis[r_mask]
            vel_focus = vel_axis[d_mask]

            fig_3d = go.Figure()
            fig_3d.add_trace(go.Surface(
                x=vel_focus,
                y=range_focus,
                z=cfar_res["rd_power_db"][np.ix_(r_mask, d_mask)],
                colorscale="Inferno",
                opacity=0.88,
                name="RD power",
                showscale=True,
                colorbar=dict(title="RD dB", x=0.92),
            ))
            fig_3d.add_trace(go.Surface(
                x=vel_focus,
                y=range_focus,
                z=cfar_res["threshold_db"][np.ix_(r_mask, d_mask)],
                colorscale="Viridis",
                opacity=0.42,
                name="CFAR threshold",
                showscale=False,
            ))

            for i, (r_bin, d_bin, r_true, v_true, amp_db) in enumerate(cfar_res["target_bins"]):
                if r_min <= r_true <= r_max and v_min <= v_true <= v_max:
                    fig_3d.add_trace(go.Scatter3d(
                        x=[vel_axis[d_bin]],
                        y=[range_axis[r_bin]],
                        z=[cfar_res["rd_power_db"][r_bin, d_bin]],
                        mode="markers+text",
                        marker=dict(symbol="x", size=7, color="cyan"),
                        text=[f"T{i+1}"],
                        textposition="top center",
                        name=f"True T{i+1}",
                    ))

            if len(det_bins):
                focus_det = [
                    (dr, dd) for dr, dd in det_bins
                    if r_min <= range_axis[dr] <= r_max and v_min <= vel_axis[dd] <= v_max
                ]
                if focus_det:
                    fr = np.array([x[0] for x in focus_det])
                    fd = np.array([x[1] for x in focus_det])
                    fig_3d.add_trace(go.Scatter3d(
                        x=vel_axis[fd],
                        y=range_axis[fr],
                        z=cfar_res["rd_power_db"][fr, fd],
                        mode="markers",
                        marker=dict(size=4, color="lime", symbol="circle-open"),
                        name="CFAR detections",
                    ))

            fig_3d.update_layout(
                title="Focused 3D View — RD Power Surface vs CFAR Threshold",
                scene=dict(
                    xaxis_title="Velocity (m/s)",
                    yaxis_title="Range (m)",
                    zaxis_title="Relative Power (dB)",
                    zaxis=dict(range=[-80, 5]),
                    camera=dict(eye=dict(x=1.45, y=-1.65, z=1.05)),
                ),
                paper_bgcolor="#0d1117",
                height=680,
                margin=dict(l=0, r=0, t=60, b=0),
            )
            st.plotly_chart(fig_3d, use_container_width=True)

            st.markdown("#### Detection table")
            st.table(cfar_res["target_rows"])

            st.markdown("#### CFAR summary")
            summary_rows = []
            for key, value in summary.items():
                if value is not None:
                    if isinstance(value, float):
                        value = f"{value:.6e}" if "Pfa" in key else f"{value:.4f}"
                    summary_rows.append({"Metric": key, "Value": value})
            st.table(summary_rows)
