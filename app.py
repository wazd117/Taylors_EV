"""EV charging operations portal for the ITS72604 Task 3 assessment."""
from __future__ import annotations

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
    "Normal": "#16a37a",
    "Grid_Surge": "#e9a23b",
    "Overheated": "#dc5b63",
}

st.set_page_config(
    page_title="Volt | EV Charging Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
        :root { --ink:#182b3a; --muted:#70818d; --teal:#087f78; --line:#e5eceb; }
        html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
        .stApp { background:linear-gradient(180deg,#f5faf9 0%,#f7f9fb 38%,#f7f9fb 100%); color:var(--ink); }
        h1,h2,h3 { font-family:'Manrope',sans-serif; letter-spacing:-.035em; color:var(--ink); }
        h1 { font-size:2rem !important; font-weight:800 !important; }
        [data-testid="stSidebar"] { background:#102d36; }
        [data-testid="stSidebar"] * { color:#eef8f6; }
        [data-testid="stSidebar"] input, [data-testid="stSidebar"] [data-baseweb="select"] * { color:#182b3a; }
        [data-testid="stSidebar"] [data-testid="stMetricValue"] { color:#eef8f6; }
        [data-testid="stMetric"] { background:white; border:1px solid var(--line); border-radius:16px; padding:17px 20px; box-shadow:0 5px 20px #102d3608; }
        [data-testid="stMetricLabel"] { color:#6d7d87; font-weight:600; }
        [data-testid="stMetricValue"] { color:var(--ink); font-family:'Manrope',sans-serif; font-weight:800; }
        [data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:12px; overflow:hidden; }
        div[data-testid="stPlotlyChart"], .stAlert { border-radius:14px; }
        .eyebrow { color:var(--teal); font-weight:700; font-size:.75rem; letter-spacing:.12em; text-transform:uppercase; margin-bottom:.3rem; }
        .subhead { color:var(--muted); margin-top:-.7rem; margin-bottom:1.4rem; }
        .footnote { color:#7a8a93; font-size:.82rem; }
        </style>
        """,
        unsafe_allow_html=True,
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
        "Operations overview", "Charging network at a glance",
        "A live view of the supplied charging-session history. Use the sidebar to focus the network.",
    )
    if filtered.empty:
        st.warning("No sessions match these filters. Adjust the sidebar selections to continue.")
        return

    peak = filtered[filtered["Peak_Hour_Flag"].eq(1)]
    delivered = float(filtered["Energy_kWh"].sum())
    duration = float(filtered["Duration_Mins"].mean())
    peak_revenue = float(peak["Energy_kWh"].sum() * rate)
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
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with right:
        st.subheader("Peak and off-peak energy")
        buckets = filtered.assign(
            Window=filtered["Peak_Hour_Flag"].map({1: "Peak", 0: "Off-peak"})
        ).groupby("Window", observed=False)["Energy_kWh"].sum().reindex(["Peak", "Off-peak"], fill_value=0)
        fig, ax = plt.subplots(figsize=(5.4, 3.4))
        sns.barplot(x=buckets.index, y=buckets.values, hue=buckets.index,
                    palette={"Peak": "#087f78", "Off-peak": "#9bc8c3"}, legend=False, ax=ax)
        ax.set(xlabel="", ylabel="Energy (kWh)")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
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
    st.dataframe(table, use_container_width=True, hide_index=True, height=390)
    st.markdown(
        '<div class="footnote">Revenue is an estimate based on the adjustable tariff; the source data do not include transaction prices.</div>',
        unsafe_allow_html=True,
    )


def analytics() -> None:
    apply_theme()
    try:
        data = load_data()
    except Exception as exc:
        st.error(f"Unable to load the charging dataset: {exc}")
        st.stop()
    filtered, _ = shared_filters(data)
    page_heading(
        "Visual analytics studio", "Find patterns in charging activity",
        "Explore relationships between location temperature, energy, time-of-day and charger mix.",
    )
    if filtered.empty:
        st.warning("No sessions match these filters. Adjust the sidebar selections to continue.")
        return

    view = st.selectbox(
        "Choose a chart", ["Energy vs ambient temperature", "Peak vs off-peak comparison", "Charger type distribution"],
        key="analytics_chart",
    )
    if view == "Energy vs ambient temperature":
        temperature_column = "Ambient_Temp_C" if "Ambient_Temp_C" in filtered.columns else "Battery_Temp_C"
        st.caption("The source field is treated as charging-location ambient temperature per the project data definition.")
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
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    elif view == "Peak vs off-peak comparison":
        measure = st.radio("Compare by", ["Energy delivered", "Session count", "Average duration"], horizontal=True)
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
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        counts = filtered["Charger_Type"].value_counts().reindex(sorted(data["Charger_Type"].unique()), fill_value=0)
        fig, ax = plt.subplots(figsize=(8, 4.7))
        colors = ["#087f78", "#68aaa4", "#b5d8d4"]
        bars = ax.bar(counts.index, counts.values, color=colors[:len(counts)], width=.56)
        ax.bar_label(bars, padding=5, color="#52656e", fontweight="semibold")
        ax.set(xlabel="Charger type", ylabel="Sessions")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with st.expander("How to read these charts"):
        st.write(
            "All charts use the active sidebar filters. Peak hours follow the supplied definition "
            "(12:00–16:59). The cleaned dataset labels the source temperature as ambient temperature, "
            "Power-rate bands are derived proxies and are not measured conversion efficiency."
        )


def inference() -> None:
    apply_theme()
    page_heading(
        "Live machine learning", "Grid status inference",
        "Enter a charging session to score it with the saved T2 preprocessing and classification pipeline.",
    )
    st.sidebar.markdown("### Charging session")
    charger_type = st.sidebar.selectbox("Charger type", ["AC", "DC"], index=1)
    user_type = st.sidebar.selectbox("User type", ["Student", "Staff", "Visitor"])
    start_hour = st.sidebar.number_input("Start hour", min_value=0, max_value=23, value=17, step=1)
    duration = st.sidebar.number_input("Duration (minutes)", min_value=15.0, max_value=300.0, value=60.0, step=5.0)
    energy = st.sidebar.number_input("Energy delivered (kWh)", min_value=5.0, max_value=120.0, value=18.0, step=0.5)
    temperature = st.sidebar.slider("Ambient temperature (°C)", min_value=24.0, max_value=42.0, value=35.5, step=0.5)
    overstay = st.sidebar.slider("Overstay (minutes)", min_value=0, max_value=240, value=5, step=1)

    a, b = st.columns([1.25, 1])
    with a:
        st.subheader("Session profile")
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
