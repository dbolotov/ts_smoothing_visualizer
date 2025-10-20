import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import savgol_filter
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.ndimage import gaussian_filter1d
from pykalman import KalmanFilter



# --- Functions ---
def load_dataset(name):
    path = os.path.join(BASE_DIR, "data", f"{name}.csv")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df.set_index("timestamp", inplace=True)
    return df

def load_markdown(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# --- Datasets ---
dataset_options = {
    "Sunspots": "sunspots",
    "Noisy Sine": "noisy_sine",
    "Humidity": "long_term_weather_rh",
    "Wind Speed": "long_term_weather_wv",
    "Process Anomalies": "process_anomalies",
}

# Load data
BASE_DIR = os.path.dirname(__file__)

# Set Layout to "wide"
st.set_page_config(layout="wide")

# --- Styling ---
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# --- Colors ---
method_colors = {
    "Original": "black",
    "MA": "#1f77b4",
    "EMA": "#ff7f0e",
    "SavGol": "#2ca02c",
    "LOESS": "#d62728",
    "Gaussian": "#9467bd",
    "Kalman": "#17becf",
}

st.title("Time Series Smoothing: an Interactive Visualizer")

left_col, spacer, right_col = st.columns([5, 0.5, 6])

with left_col:

    st.markdown("##### Explore how six smoothing techniques transform noisy time series data.")

    with st.expander("Documentation", expanded=True):
        tab_overview, tab_data, tab_ma, tab_ema, tab_sg, tab_loess, tab_gf, tab_kf = st.tabs([
            "Overview", "Datasets", "Moving Average", "Exponential MA", "Savitsky-Golay", 
            "LOESS", "Gaussian Filter", "Kalman Filter"])

        with tab_overview:
            st.markdown(load_markdown("docs/overview.md"))
        with tab_data:
            st.markdown(load_markdown("docs/datasets.md"))
        with tab_ma:
            st.markdown(load_markdown("docs/ma.md"))
        with tab_ema:
            st.markdown(load_markdown("docs/ema.md"))
        with tab_sg:
            st.markdown(load_markdown("docs/sg.md"))
        with tab_loess:
            st.markdown(load_markdown("docs/loess.md"))
        with tab_gf:
            st.markdown(load_markdown("docs/gf.md"))
        with tab_kf:
            st.markdown(load_markdown("docs/kf.md"))


    st.subheader("Data Parameters")
    st.caption(
        "Select the dataset and adjust the amount of data visualized, as well as how visible the raw signal is."
    )
    dp_col1, dp_col2, dp_col3 = st.columns([1, 1, 1])
    with dp_col1:
        selected_label = st.selectbox("Dataset", options=list(dataset_options.keys()))
        dataset_name = dataset_options[selected_label]
        df_full = load_dataset(dataset_name)

    # Dataset blurbs
    dataset_blurbs = {
        "Sunspots": "Daily counts of sunspots on the Sun's surface.",
        "Noisy Sine": "A synthetic sine wave with added noise.",
        "Humidity": "Relative humidity readings from a long-term weather monitoring station.",
        "Wind Speed": "Wind speed readings (m/s) from a long-term weather monitoring station.",
        "Process Anomalies": "Synthetic industrial process with 4 operating modes and 3 injected anomalies.",
    }
    # Show dataset blurb
    st.caption(dataset_blurbs.get(selected_label, ""))

    with dp_col2:
        subset_size = st.slider(
            "Points to display from start of series",
            min_value=50,
            max_value=len(df_full),
            value=len(df_full),
            step=20,
        )
    with dp_col3:
        signal_opacity = st.slider("Noisy signal opacity", 0.0, 1.0, 0.3, step=0.05)
    df = df_full.iloc[:subset_size].copy()

    st.subheader("Smoothing Parameters")
    st.caption(
        "Select which methods to display and adjust their smoothing parameters below."
    )

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        show_ma = st.checkbox("", value=False, key="show_ma")
        st.markdown(
            f'<div class="method-label"><span class="color-dot" style="background-color:{method_colors["MA"]}"></span>Moving Avg</div>',
            unsafe_allow_html=True,
        )
        ma_window = st.slider(
            "Window",
            3,
            51,
            15,
            step=2,
            key="ma",
            help=("Number of data points averaged at each step.\n\n"
            "Larger = smoother but more laggy."),
        )

    with col2:
        show_ema = st.checkbox("", value=True, key="show_ema")
        st.markdown(
            f'<div class="method-label"><span class="color-dot" style="background-color:{method_colors["EMA"]}"></span>EMA</div>',
            unsafe_allow_html=True,
        )
        ema_alpha = st.slider(
            "Alpha",
            0.01,
            1.0,
            0.1,
            step=0.01,
            key="ema",
            help=("Smoothing factor between 0 and 1.\n\n"
            "Higher = quicker response, less smoothing."),
        )

    with col3:
        show_savgol = st.checkbox("", value=False, key="show_sg")
        st.markdown(
            f'<div class="method-label"><span class="color-dot" style="background-color:{method_colors["SavGol"]}"></span>SavGol</div>',
            unsafe_allow_html=True,
        )
        sg_window = st.slider(
            "Window",
            5,
            51,
            15,
            step=2,
            key="sg_win",
            help=("Number of points used in each fit (must be odd).\n\n"),
        )
        sg_poly = st.slider("Poly", 1, 5, 2, key="sg_poly",
                            help=("Controls how flexible the fit is.\n\n"
                            "Higher = more responsive to structure."))

    with col4:
        show_loess = st.checkbox("", value=False, key="show_loess")
        st.markdown(
            f'<div class="method-label"><span class="color-dot" style="background-color:{method_colors["LOESS"]}"></span>LOESS</div>',
            unsafe_allow_html=True,
        )
        loess_frac = st.slider(
            "Frac",
            0.01,
            0.5,
            0.05,
            step=0.01,
            key="loess",
            help=("**Frac**: proportion of the dataset used to compute each local regression."),
        )

    with col5:
        show_gauss = st.checkbox("", value=False, key="show_gauss")
        st.markdown(
            f'<div class="method-label"><span class="color-dot" style="background-color:{method_colors["Gaussian"]}"></span>Gaussian</div>',
            unsafe_allow_html=True,
        )
        gauss_sigma = st.slider(
            "Sigma",
            0.1,
            10.0,
            2.0,
            step=0.1,
            key="gauss",
            help=("**Sigma**: standard deviation of the Gaussian kernel."),
        )

    with col6:
        show_kalman = st.checkbox("", value=True, key="show_kf")
        st.markdown(
            f'<div class="method-label"><span class="color-dot" style="background-color:{method_colors["Kalman"]}"></span>Kalman</div>',
            unsafe_allow_html=True,
        )
        kf_transition_noise = st.slider("Tr std", 0.001, 1.0, 0.05, step=0.01, key="kf_trans",
            help=( "Transition standard deviation. Expected noise in the process’s internal dynamics."),
        )
        kf_obs_noise = st.slider("Obs std", 0.001, 1.0, 0.2, step=0.01, key="kf_obs",
            help=("Observation standard deviation; expected noise in the observed data."))

# --- Smoothing Calculations ---
df["ma"] = df["value"].rolling(window=ma_window, center=True).mean().bfill().ffill()
df["ema"] = df["value"].ewm(alpha=ema_alpha).mean()
df["savgol"] = savgol_filter(df["value"], window_length=sg_window, polyorder=sg_poly)
lowess_smoothed = lowess(df["value"], np.arange(len(df)), frac=loess_frac)
df["loess"] = pd.Series(lowess_smoothed[:, 1], index=df.index)
df["gaussian"] = gaussian_filter1d(df["value"], sigma=gauss_sigma)

kf = KalmanFilter(
    transition_matrices=[1],
    observation_matrices=[1],
    initial_state_mean=df["value"].iloc[0],
    initial_state_covariance=1,
    transition_covariance=kf_transition_noise,
    observation_covariance=kf_obs_noise,
)
kalman_state_means, _ = kf.filter(df["value"].values)
df["kalman"] = kalman_state_means

# --- Plotting ---
smooth_opacity = 0.8
sm_width = 1.5
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["value"],
        name="Original",
        line=dict(color=method_colors["Original"], width=1),
        opacity=signal_opacity,
    )
)
if show_ma:
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["ma"],
            name="MA",
            line=dict(color=method_colors["MA"], width=sm_width),
            opacity=smooth_opacity,
        )
    )
if show_ema:
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["ema"],
            name="EMA",
            line=dict(color=method_colors["EMA"], width=sm_width),
            opacity=smooth_opacity,
        )
    )
if show_savgol:
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["savgol"],
            name="SavGol",
            line=dict(color=method_colors["SavGol"], width=sm_width),
            opacity=smooth_opacity,
        )
    )
if show_loess:
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["loess"],
            name="LOESS",
            line=dict(color=method_colors["LOESS"], width=sm_width),
            opacity=smooth_opacity,
        )
    )
if show_gauss:
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["gaussian"],
            name="Gaussian",
            line=dict(color=method_colors["Gaussian"], width=sm_width),
            opacity=smooth_opacity,
        )
    )
if show_kalman:
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["kalman"],
            name="Kalman",
            line=dict(color=method_colors["Kalman"], width=sm_width),
            opacity=smooth_opacity,
        )
    )

fig.update_layout(
    title="Smoothing Methods Comparison",
    xaxis=dict(showgrid=True),
    xaxis_title="Time",
    yaxis_title="Value",
    height=600,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
)

with right_col:
    st.plotly_chart(fig, use_container_width=True)

    # --- Diagnostics Table ---

    def total_variation(series):
        return np.sum(np.abs(np.diff(series)))

    def rpr(smoothed, original):
        return total_variation(smoothed) / total_variation(original)

    tv_orig = total_variation(df["value"])

    methods = []
    rpr_vals = []

    if show_ma:
        methods.append("MA")
        rpr_vals.append(rpr(df["ma"], df["value"]))
    if show_ema:
        methods.append("EMA")
        rpr_vals.append(rpr(df["ema"], df["value"]))
    if show_savgol:
        methods.append("SavGol")
        rpr_vals.append(rpr(df["savgol"], df["value"]))
    if show_loess:
        methods.append("LOESS")
        rpr_vals.append(rpr(df["loess"], df["value"]))
    if show_gauss:
        methods.append("Gaussian")
        rpr_vals.append(rpr(df["gaussian"], df["value"]))
    if show_kalman:
        methods.append("Kalman")
        rpr_vals.append(rpr(df["kalman"], df["value"]))

    diag_df = pd.DataFrame(
        {
            method: [f"{r:.2f}"]
            for method, r in zip(methods, rpr_vals)
        },
        index=["RPR"],
    )

    st.dataframe(diag_df, use_container_width=False)
    st.caption(
        "**RPR (Roughness Preservation Ratio)** compares jaggedness after smoothing to the original. Lower values mean more smoothing."
    )


# with right_col:
#     st.plotly_chart(fig, use_container_width=True)

#     # --- Diagnostics Table ---
#     def total_variation(series):
#         return np.sum(np.abs(np.diff(series)))

#     tv_orig = total_variation(df["value"])
#     std_orig = df["value"].std()

#     methods = []
#     tvr_vals = []
#     sdr_vals = []

#     if show_ma:
#         methods.append("MA")
#         tvr_vals.append(1 - total_variation(df["ma"]) / tv_orig)
#         sdr_vals.append(1 - df["ma"].std() / std_orig)
#     if show_ema:
#         methods.append("EMA")
#         tvr_vals.append(1 - total_variation(df["ema"]) / tv_orig)
#         sdr_vals.append(1 - df["ema"].std() / std_orig)
#     if show_savgol:
#         methods.append("SavGol")
#         tvr_vals.append(1 - total_variation(df["savgol"]) / tv_orig)
#         sdr_vals.append(1 - df["savgol"].std() / std_orig)
#     if show_loess:
#         methods.append("LOESS")
#         tvr_vals.append(1 - total_variation(df["loess"]) / tv_orig)
#         sdr_vals.append(1 - df["loess"].std() / std_orig)
#     if show_gauss:
#         methods.append("Gaussian")
#         tvr_vals.append(1 - total_variation(df["gaussian"]) / tv_orig)
#         sdr_vals.append(1 - df["gaussian"].std() / std_orig)
#     if show_kalman:
#         methods.append("Kalman")
#         tvr_vals.append(1 - total_variation(df["kalman"]) / tv_orig)
#         sdr_vals.append(1 - df["kalman"].std() / std_orig)

#     diag_df = pd.DataFrame(
#         {
#             method: [f"{tvr:.2f}", f"{sdr:.2f}"]
#             for method, tvr, sdr in zip(methods, tvr_vals, sdr_vals)
#         },
#         index=["TVR", "SDR"],
#     )

#     st.dataframe(diag_df, use_container_width=False)
#     st.caption(
#         "**TVR (Total Variance Reduction)** measures how much each method reduces jaggedness in the signal. Higher values mean smoother output.\n"
#         "**SDR (Standard Deviation Reduction)** compares variability before and after smoothing. Higher values mean more consistent, less noisy signals."
#     )
