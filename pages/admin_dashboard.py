import streamlit as st
import pandas as pd
import altair as alt

from src.auth import check_password
from src.database import get_analytics, get_all_queries

st.set_page_config(
    page_title="SKCET Admin Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

if not check_password():
    st.stop()

# ── Shared dark style ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .stApp { background: #0a0a0f; }
    [data-testid="stSidebar"] { background: rgba(15,15,25,0.98) !important; border-right: 1px solid rgba(99,102,241,0.2) !important; }
    #MainMenu, footer, header { visibility: hidden; }
    .metric-card {
        background: rgba(15,18,35,0.85);
        border: 1px solid rgba(99,102,241,0.2);
        border-radius: 14px;
        padding: 1.25rem 1.5rem;
        text-align: center;
    }
    .metric-value { font-size: 2.2rem; font-weight: 800; background: linear-gradient(135deg, #a5b4fc, #e879f9); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
    .metric-label { color: #64748b; font-size: 0.8rem; font-weight: 500; margin-top: 0.2rem; text-transform: uppercase; letter-spacing: 0.05em; }
</style>
""", unsafe_allow_html=True)

st.markdown("## 📊 Admin Analytics Dashboard")
st.markdown("<span style='color:#64748b;font-size:0.9rem;'>Real-time monitoring of student queries and AI performance</span>", unsafe_allow_html=True)
st.markdown("---")

# ── Data ──────────────────────────────────────────────────────────────────────
with st.spinner("Fetching analytics..."):
    analytics = get_analytics()
    queries = get_all_queries()

# ── Top Metrics ───────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
metric_data = [
    (c1, analytics["total_queries"], "Total Queries"),
    (c2, analytics["thumbs_up"], "👍 Helpful"),
    (c3, analytics["thumbs_down"], "👎 Not Helpful"),
    (c4, f"{analytics['avg_response_time_ms']}ms", "Avg Response Time"),
    (c5, f"{round(analytics['thumbs_up'] / max(analytics['total_queries'], 1) * 100)}%", "Satisfaction Rate"),
]
for col, val, label in metric_data:
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{val}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Charts ────────────────────────────────────────────────────────────────────
if analytics["daily_counts"]:
    col_left, col_right = st.columns(2)

    # Daily query volume chart
    with col_left:
        st.markdown("#### 📈 Daily Query Volume")
        df_daily = pd.DataFrame(analytics["daily_counts"])
        df_daily["day"] = pd.to_datetime(df_daily["day"])
        chart_queries = (
            alt.Chart(df_daily)
            .mark_area(
                line={"color": "#6366f1"},
                color=alt.Gradient(
                    gradient="linear",
                    stops=[
                        alt.GradientStop(color="rgba(99,102,241,0.6)", offset=0),
                        alt.GradientStop(color="rgba(99,102,241,0.0)", offset=1),
                    ],
                    x1=1, x2=1, y1=1, y2=0
                ),
                point=alt.OverlayMarkDef(color="#6366f1", size=60),
            )
            .encode(
                x=alt.X("day:T", title="Date", axis=alt.Axis(labelColor="#64748b", titleColor="#64748b", gridColor="#1e293b")),
                y=alt.Y("count:Q", title="Queries", axis=alt.Axis(labelColor="#64748b", titleColor="#64748b", gridColor="#1e293b")),
                tooltip=["day:T", "count:Q"]
            )
            .properties(height=250, background="transparent")
            .configure_view(strokeWidth=0)
        )
        st.altair_chart(chart_queries, use_container_width=True)

    # Avg response time chart
    with col_right:
        st.markdown("#### ⏱️ Avg Response Time (ms)")
        if analytics["daily_perf"]:
            df_perf = pd.DataFrame(analytics["daily_perf"])
            df_perf["day"] = pd.to_datetime(df_perf["day"])
            chart_perf = (
                alt.Chart(df_perf)
                .mark_line(color="#e879f9", strokeWidth=2.5, point=alt.OverlayMarkDef(color="#e879f9", size=60))
                .encode(
                    x=alt.X("day:T", title="Date", axis=alt.Axis(labelColor="#64748b", titleColor="#64748b", gridColor="#1e293b")),
                    y=alt.Y("avg_ms:Q", title="Response Time (ms)", axis=alt.Axis(labelColor="#64748b", titleColor="#64748b", gridColor="#1e293b")),
                    tooltip=["day:T", "avg_ms:Q"]
                )
                .properties(height=250, background="transparent")
                .configure_view(strokeWidth=0)
            )
            st.altair_chart(chart_perf, use_container_width=True)
        else:
            st.info("No response time data yet.")
else:
    st.info("No query data yet. Ask the assistant some questions first!")

# ── Confidence Breakdown ──────────────────────────────────────────────────────
if queries:
    conf_counts = {"High": 0, "Medium": 0, "Low": 0, "unknown": 0}
    for row in queries:
        c = row.get("confidence", "unknown")
        conf_counts[c] = conf_counts.get(c, 0) + 1

    known = {k: v for k, v in conf_counts.items() if k != "unknown" and v > 0}
    if known:
        st.markdown("#### 🎯 Answer Confidence Breakdown")
        df_conf = pd.DataFrame([{"Confidence": k, "Count": v} for k, v in known.items()])
        color_map = {"High": "#22c55e", "Medium": "#eab308", "Low": "#ef4444"}
        bar = (
            alt.Chart(df_conf)
            .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
            .encode(
                x=alt.X("Confidence:N", title=None, axis=alt.Axis(labelColor="#94a3b8")),
                y=alt.Y("Count:Q", title="Count", axis=alt.Axis(labelColor="#64748b", gridColor="#1e293b")),
                color=alt.Color("Confidence:N", scale=alt.Scale(domain=list(color_map.keys()), range=list(color_map.values())), legend=None),
                tooltip=["Confidence:N", "Count:Q"]
            )
            .properties(height=200, background="transparent")
            .configure_view(strokeWidth=0)
        )
        st.altair_chart(bar, use_container_width=True)

st.markdown("---")

# ── Interaction Logs ──────────────────────────────────────────────────────────
st.markdown("#### 📝 Recent Interaction Logs")
if queries:
    df = pd.DataFrame(queries)
    available_cols = [c for c in ["timestamp", "rating", "confidence", "response_time_ms", "user_input", "assistant_response"] if c in df.columns]
    df = df[available_cols]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    col_config = {
        "timestamp": st.column_config.DatetimeColumn("Date/Time", format="MMM DD, hh:mm A"),
        "rating": st.column_config.TextColumn("Feedback"),
        "confidence": st.column_config.TextColumn("Confidence"),
        "response_time_ms": st.column_config.NumberColumn("Response (ms)"),
        "user_input": st.column_config.TextColumn("User Query", width="large"),
        "assistant_response": st.column_config.TextColumn("AI Answer", width="large"),
    }
    st.dataframe(df, column_config=col_config, hide_index=True, use_container_width=True)
else:
    st.info("No interactions logged yet.")

st.markdown("---")

# ── Flagged Queries ───────────────────────────────────────────────────────────
st.markdown("#### 🚩 Flagged for Review")
st.markdown("<span style='color:#64748b;font-size:0.9rem;'>Queries reported by students as incorrect or needing attention.</span>", unsafe_allow_html=True)
if queries:
    flagged = [q for q in queries if q.get("is_flagged") == 1]
    if flagged:
        df_f = pd.DataFrame(flagged)
        av_cols = [c for c in ["timestamp", "user_input", "assistant_response", "confidence"] if c in df_f.columns]
        df_f = df_f[av_cols]
        df_f["timestamp"] = pd.to_datetime(df_f["timestamp"])
        st.dataframe(df_f, column_config=col_config, hide_index=True, use_container_width=True)
    else:
        st.success("No queries currently flagged for review! 🎉")
else:
    st.info("No queries found.")

st.markdown("---")
st.caption("SKCET RAG Admin Dashboard · Phase 5")
