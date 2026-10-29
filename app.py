"""EV charging operations portal for the ITS72604 Task 3 assessment."""
from __future__ import annotations

import base64
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from feature_engineering import add_engineered_features, single_prediction_features, validate_group_data


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "cleaned_ev_dataset.csv"
MODEL_PATH = ROOT / "ev_model.pkl"
STATUS_ORDER = ["Normal", "Grid_Surge", "Overheated"]
STATUS_PALETTE = {
    "Normal": "#029456",
    "Grid_Surge": "#F9D779",
    "Overheated": "#D8226C",
}
CHART_COLORS = ["#D8226C", "#029456", "#F9D779", "#005BB3"]

st.set_page_config(
    page_title="Volt | EV Charging Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_theme() -> None:
    st.html(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Figtree:wght@400;500;600;700;800&display=swap');
        html, body, [class*="st-"] { font-family: "Figtree", sans-serif !important; }
        .eyebrow { color:#326080; font-weight:700; font-size:.75rem; letter-spacing:.12em; text-transform:uppercase; margin-bottom:.3rem; }
        .subhead { color:#617382; margin-top:-.7rem; margin-bottom:1.4rem; }
        [data-testid="stSidebarNav"] a[aria-current="page"] { background:#005BB3 !important; color:#fff !important; border-radius:10px; }
        [data-testid="stSidebarNav"] a[aria-current="page"] span { color:#fff !important; }
        [data-testid="stMetric"] { background:#fffaf6; border:1px solid #ead8cb; border-radius:16px; padding:17px 20px; box-shadow:0 5px 20px #102d3608; }
        [data-testid="stMetricLabel"] { color:#617382; font-weight:600; }
        [data-testid="stDataFrame"] { border:1px solid #ead8cb; border-radius:12px; overflow:hidden; }
        .footnote { color:#617382; font-size:.82rem; }
        .about-footer { margin-top:1.8rem; padding:.85rem 1.15rem; border-radius:12px; background:#05070a; color:#f4f7fa; }
        .about-footer h2, .about-footer p { color:#f4f7fa; }
        .about-footer h2 { margin:.05rem 0 .55rem; font-size:1.05rem; }
        .about-kicker { color:#F9D779; font-size:.65rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase; }
        .about-team { display:flex; flex-wrap:wrap; gap:.8rem 1.5rem; align-items:flex-start; }
        .team-group { display:flex; flex-direction:column; gap:.28rem; }
        .about-role { color:#c5d1d9 !important; font-size:12pt; font-weight:700; margin:0; }
        .about-name { display:inline-flex; align-items:center; gap:.32rem; color:#f4f7fa !important; font-size:12pt; font-weight:500; line-height:1.2; text-decoration:none !important; }
        .about-name.leader-link { color:#005BB3 !important; }
        .about-name:hover { text-decoration:underline !important; }
        .github-logo { width:.82rem; height:.82rem; object-fit:contain; filter:invert(1); flex:none; }
        .st-key-inference-metrics [data-testid="stMetricLabel"],
        .st-key-dashboard-metrics [data-testid="stMetricLabel"] { font-weight:800 !important; }
        .st-key-inference-metrics [data-testid="stMetricValue"],
        .st-key-dashboard-metrics [data-testid="stMetricValue"] { font-size:1.15rem !important; }
        @media (max-width:650px) { .about-team { gap:.7rem 1rem; } }
        </style>
        """,
    )


@st.cache_data(show_spinner="Loading the validated charging dataset…")
def load_data() -> pd.DataFrame:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH.name}")
    data = pd.read_csv(DATA_PATH)
    validate_group_data(data)
    return add_engineered_features(data)


@st.cache_resource(show_spinner="Loading the trained grid-status model…")
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH.name}")
    # Import explicitly before unpickling: the saved estimator is a scikit-learn Pipeline.
    import sklearn

    return joblib.load(MODEL_PATH)


def page_heading(kicker: str, title: str, description: str) -> None:
    st.markdown(f'<div class="eyebrow">{kicker}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="subhead">{description}</div>', unsafe_allow_html=True)


def render_our_team() -> None:
    logo_path = ROOT / "github-logo.png"
    logo_data = base64.b64encode(logo_path.read_bytes()).decode("ascii") if logo_path.is_file() else ""
    logo = f'<img class="github-logo" src="data:image/png;base64,{logo_data}" alt="GitHub">' if logo_data else ""
    st.html(
        f"""
        <section class="about-footer" aria-labelledby="our-team-heading">
          <h2 id="our-team-heading">Our team</h2>
          <div class="about-team">
            <div class="team-group">
              <p class="about-role">Group Leader</p>
              <a class="about-name leader-link" href="https://github.com/wazd117" target="_blank" rel="noopener noreferrer">{logo}<span>Li Zhengyuyang</span></a>
            </div>
            <div class="team-group">
              <p class="about-role">Group Member</p>
              <a class="about-name" href="https://github.com/mahmudmasum242-tech" target="_blank" rel="noopener noreferrer">{logo}<span>Masum Mahmud</span></a>
              <a class="about-name" href="https://github.com/1054441617-png" target="_blank" rel="noopener noreferrer">{logo}<span>Jie Jao</span></a>
            </div>
          </div>
        </section>
        """,
        width="stretch",
    )


def shared_filters(data: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    st.sidebar.markdown("### Explore sessions")
    st.sidebar.caption("Filters apply to the dashboard and analytics views.")
    status_options = st.sidebar.multiselect(
        "Grid status", STATUS_ORDER, default=STATUS_ORDER, key="filter_status"
    )
    charger_options = sorted(data["Charger_Type"].dropna().unique().tolist())
    charger = st.sidebar.multiselect(
        "Charger type", charger_options, default=charger_options, key="filter_charger"
    )
    user_options = sorted(data["User_Type"].dropna().unique().tolist())
    user = st.sidebar.multiselect(
        "User type", user_options, default=user_options, key="filter_user"
    )
    peak_choice = st.sidebar.selectbox(
        "Time window", ["All sessions", "Peak only", "Off-peak only"], key="filter_peak"
    )
    rate = st.sidebar.number_input(
        "Estimated tariff ($ / kWh)", min_value=0.0, max_value=5.0,
        value=0.35, step=0.05, help="Used only to estimate revenue from delivered energy.",
    )
    filtered = data[
        data["Grid_Status_Label"].isin(status_options)
        & data["Charger_Type"].isin(charger)
        & data["User_Type"].isin(user)
    ]
    if peak_choice == "Peak only":
        filtered = filtered[filtered["Peak_Hour_Flag"].eq(1)]
    elif peak_choice == "Off-peak only":
        filtered = filtered[filtered["Peak_Hour_Flag"].eq(0)]
    return filtered.copy(), float(rate)


def dashboard() -> None:
    apply_theme()
    try:
        data = load_data()
    except Exception as exc:
        st.error(f"Unable to load the charging dataset: {exc}")
        st.stop()
    filtered, rate = shared_filters(data)
    page_heading(
        "Operations overview", "Executive KPI dashboard",
        "A live view of the supplied charging-session history. Use the sidebar to focus the network.",
    )
    if filtered.empty:
        st.warning("No sessions match these filters. Adjust the sidebar selections to continue.")
        render_our_team()
        return

    peak = filtered[filtered["Peak_Hour_Flag"].eq(1)]
    delivered = float(filtered["Energy_kWh"].sum())
    duration = float(filtered["Duration_Mins"].mean())
    peak_revenue = float(peak["Energy_kWh"].sum() * rate)
    with st.container(key="dashboard-metrics"):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Energy delivered", f"{delivered:,.0f} kWh", help="Sum of delivered session energy.")
        c2.metric("Avg. session duration", f"{duration:,.0f} min", help="Mean charging-session duration.")
        c3.metric("Peak revenue · estimate", f"${peak_revenue:,.0f}", help=f"Peak energy × estimated tariff of ${rate:.2f}/kWh.")
        c4.metric("Sessions in view", f"{len(filtered):,}", help="Sessions matching the current filters.")

    left, right = st.columns([1.35, 1])
    with left:
        st.subheader("Session mix")
        mix = filtered["Grid_Status_Label"].value_counts().reindex(STATUS_ORDER, fill_value=0)
        fig, ax = plt.subplots(figsize=(7.3, 3.4))
        sns.barplot(x=mix.index, y=mix.values, hue=mix.index, palette=STATUS_PALETTE,
                    legend=False, ax=ax)
        ax.set(xlabel="", ylabel="Sessions")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        fig.tight_layout()
        st.pyplot(fig, width="stretch", alt="Session counts by grid status for the filtered charging sessions.")
        plt.close(fig)
    with right:
        st.subheader("Peak and off-peak energy")
        buckets = filtered.assign(
            Window=filtered["Peak_Hour_Flag"].map({1: "Peak", 0: "Off-peak"})
        ).groupby("Window", observed=False)["Energy_kWh"].sum().reindex(["Peak", "Off-peak"], fill_value=0)
        fig, ax = plt.subplots(figsize=(5.4, 3.4))
        sns.barplot(x=buckets.index, y=buckets.values, hue=buckets.index,
                    palette={"Peak": "#005BB3", "Off-peak": "#F9D779"}, legend=False, ax=ax)
        ax.set(xlabel="", ylabel="Energy (kWh)")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        fig.tight_layout()
        st.pyplot(fig, width="stretch", alt="Total energy delivered during peak and off-peak periods.")
        plt.close(fig)

    st.subheader("Filtered charging sessions")
    st.caption(f"Showing {len(filtered):,} of {len(data):,} sessions · sort and browse the table below.")
    temperature_column = "Ambient_Temp_C" if "Ambient_Temp_C" in filtered.columns else "Battery_Temp_C"
    table = filtered[[
        "Session_ID", "Charger_Type", "User_Type", "Start_Hour", "Duration_Mins",
        "Energy_kWh", temperature_column, "Grid_Status_Label",
    ]].rename(columns={
        "Session_ID": "Session", "Charger_Type": "Charger", "User_Type": "User",
        "Start_Hour": "Start hour", "Duration_Mins": "Duration (min)",
        "Energy_kWh": "Energy (kWh)", temperature_column: "Ambient temp (°C)",
        "Grid_Status_Label": "Grid status",
    })
    st.dataframe(table, width="stretch", hide_index=True, height=390)
    st.markdown(
        '<div class="footnote">Revenue is an estimate based on the adjustable tariff; the source data do not include transaction prices.</div>',
        unsafe_allow_html=True,
    )
    render_our_team()


def analytics() -> None:
    apply_theme()
    try:
        data = load_data()
    except Exception as exc:
        st.error(f"Unable to load the charging dataset: {exc}")
        st.stop()
    filtered, _ = shared_filters(data)
    page_heading(
        "Visual analytics studio", "Visual analytics studio",
        "Explore relationships between location temperature, energy, time-of-day and charger mix.",
    )
    if filtered.empty:
        st.warning("No sessions match these filters. Adjust the sidebar selections to continue.")
        render_our_team()
        return

    view = st.selectbox(
        "Choose a chart", ["Energy vs ambient temperature", "Peak vs off-peak comparison", "Charger type distribution"],
        key="analytics_chart",
    )
    if view == "Energy vs ambient temperature":
        temperature_column = "Ambient_Temp_C" if "Ambient_Temp_C" in filtered.columns else "Battery_Temp_C"
        st.write("Each point represents one session. The chart compares location temperature with delivered energy; color identifies the session's grid-status label. The sidebar filters limit which sessions appear.")
        fig, ax = plt.subplots(figsize=(10, 5.3))
        sns.scatterplot(
            data=filtered, x=temperature_column, y="Energy_kWh", hue="Grid_Status_Label",
            hue_order=STATUS_ORDER, palette=STATUS_PALETTE, alpha=.72, s=62, edgecolor="white", linewidth=.4, ax=ax,
        )
        ax.set(xlabel="Ambient temperature (°C)", ylabel="Energy delivered (kWh)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.18)
        ax.legend(title="Grid status", frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(.5, 1.12))
        fig.tight_layout()
        st.pyplot(fig, width="stretch", alt="Scatter plot comparing charging-location ambient temperature and energy delivered, colored by grid status.")
        plt.close(fig)
    elif view == "Peak vs off-peak comparison":
        st.write("The bars compare peak sessions (12:00–16:59) with off-peak sessions. Colors separate grid-status classes; choose whether bar height represents energy, session count, or average duration.")
        measure = st.segmented_control("Compare by", ["Energy delivered", "Session count", "Average duration"], default="Energy delivered")
        metric_column = {"Energy delivered": "Energy_kWh", "Average duration": "Duration_Mins"}.get(measure)
        windowed = filtered.assign(
            Window=filtered["Peak_Hour_Flag"].map({1: "Peak", 0: "Off-peak"})
        )
        if measure == "Session count":
            grouped = windowed.groupby(["Window", "Grid_Status_Label"]).size().rename("Value").reset_index()
        else:
            agg = "sum" if measure == "Energy delivered" else "mean"
            grouped = windowed.groupby(["Window", "Grid_Status_Label"], as_index=False).agg(Value=(metric_column, agg))
        fig, ax = plt.subplots(figsize=(9, 4.7))
        sns.barplot(data=grouped, x="Window", y="Value", hue="Grid_Status_Label",
                    hue_order=STATUS_ORDER, palette=STATUS_PALETTE, ax=ax)
        ylabel = {"Energy delivered": "Energy (kWh)", "Session count": "Sessions", "Average duration": "Minutes"}[measure]
        ax.set(xlabel="", ylabel=ylabel)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
        ax.legend(title="Grid status", frameon=False)
        fig.tight_layout()
        st.pyplot(fig, width="stretch", alt=f"{ylabel} comparison between peak and off-peak sessions, split by grid status.")
        plt.close(fig)
    else:
        st.write("This chart shows the number of filtered sessions for each charger type. Use the sidebar to compare charger mix for selected users or grid statuses.")
        counts = filtered["Charger_Type"].value_counts().reindex(sorted(data["Charger_Type"].unique()), fill_value=0)
        fig, ax = plt.subplots(figsize=(8, 4.7))
        colors = CHART_COLORS
        bars = ax.bar(counts.index, counts.values, color=colors[:len(counts)], width=.56)
        ax.bar_label(bars, padding=5, color="#52656e", fontweight="semibold")
        ax.set(xlabel="Charger type", ylabel="Sessions")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
        fig.tight_layout()
        st.pyplot(fig, width="stretch", alt="Counts of filtered charging sessions by charger type.")
        plt.close(fig)

    st.caption("All charts use the active sidebar filters. Peak hours follow the 12:00–16:59 definition.")
    render_our_team()


def inference() -> None:
    apply_theme()
    page_heading(
        "Live machine learning", "Live ML inference",
        "Enter a charging session to score it with the saved T2 preprocessing and classification pipeline.",
    )
    st.sidebar.markdown("### Charging session")
    charger_type = st.sidebar.selectbox("Charger type", ["AC", "DC"], index=1)
    user_type = st.sidebar.selectbox("User type", ["Student", "Staff", "Visitor"])
    start_hour = st.sidebar.number_input("Start hour", min_value=0, max_value=23, value=17, step=1)
    duration = st.sidebar.number_input("Duration (minutes)", min_value=1.0, max_value=1440.0, value=60.0, step=5.0)
    energy = st.sidebar.number_input("Energy delivered (kWh)", min_value=0.0, max_value=250.0, value=18.0, step=0.5)
    temperature = st.sidebar.slider("Ambient temperature (°C)", min_value=-10.0, max_value=100.0, value=35.5, step=0.5)
    overstay = st.sidebar.slider("Overstay (minutes)", min_value=0, max_value=240, value=5, step=1)

    a, b = st.columns([1.25, 1])
    with a:
        st.subheader("Session profile")
        m1, m2, m3 = st.columns(3)
        with st.container(key="inference-metrics"):
            m1, m2, m3 = st.columns(3)
            m1.metric("Charging power", f"{energy / (duration / 60):.1f} kW")
            m2.metric("Time window", "Peak" if 12 <= start_hour < 17 else "Off-peak")
            m3.metric("Charger", charger_type)
        st.markdown("#### Inputs submitted to the model")
        st.dataframe(pd.DataFrame([{
            "User type": user_type, "Start hour": int(start_hour), "Duration (min)": duration,
            "Energy (kWh)": energy, "Ambient temp (°C)": temperature, "Overstay (min)": overstay,
        }]), use_container_width=True, hide_index=True)
    with b:
        st.subheader("Prediction")
        try:
            model = load_model()
            row = single_prediction_features(
                charger_type=charger_type,
                user_type=user_type,
                start_hour=int(start_hour),
                duration_mins=float(duration),
                energy_kwh=float(energy),
                ambient_temp_c=float(temperature),
                overstay_mins=int(overstay),
            )
            with st.spinner("Scoring this session…"):
                status = str(model.predict(row)[0])
                scores = dict(zip(model.classes_, map(float, model.predict_proba(row)[0]))) if hasattr(model, "predict_proba") else {}
            if status == "Normal":
                st.success("NORMAL · No grid-status alert")
                st.write("The model classifies this session as normal.")
            elif status == "Grid_Surge":
                st.warning("GRID SURGE · Review session conditions")
                st.write("The model flags a potential grid-surge pattern.")
            elif status == "Overheated":
                st.error("OVERHEATED · Review ambient temperature")
                st.write("The model flags an overheated pattern.")
            else:
                st.warning(f"Unrecognized model output: {status}")
            if scores:
                st.markdown("#### Model class scores")
                for label in STATUS_ORDER:
                    if label in scores:
                        st.progress(scores[label], text=f"{label.replace('_', ' ')} · {scores[label]:.0%}")
                st.caption("Scores are model outputs and are not calibrated real-world risk probabilities.")
        except ModuleNotFoundError as exc:
            if exc.name == "sklearn":
                st.error("Inference needs scikit-learn, which is missing from this Python environment.")
                st.code("python -m pip install -r requirements.txt\npython -m streamlit run app.py", language="powershell")
            else:
                st.error(f"A required Python package is missing: {exc.name}")
        except Exception as exc:
            st.error(f"Unable to run inference: {exc}")
            st.info("Check that ev_model.pkl and feature_engineering.py are present beside app.py, and install requirements.txt.")
    st.markdown(
        '<div class="footnote">Decision support only. This model predicts historical labels and is not a safety-critical control system.</div>',
        unsafe_allow_html=True,
    )
    render_our_team()


def main() -> None:
    pages = {
        "Network": [
            st.Page(dashboard, title="Executive KPI Dashboard", icon="📊", default=True),
            st.Page(analytics, title="Visual Analytics Studio", icon="📈"),
        ],
        "Intelligence": [
            st.Page(inference, title="Live ML Inference", icon="⚡"),
        ],
    }
    navigation = st.navigation(pages, position="sidebar")
    navigation.run()


if __name__ == "__main__":
    main()
