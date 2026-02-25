import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from datetime import date, timedelta
from pathlib import Path
from random_forest import (
    TARGET_COL,
    EVENT_WEIGHTS,
    DEFAULT_LAT,
    DEFAULT_LON,
    build_weather_fallback,
    fetch_weather_forecast,
    initial_week_frame,
    run_forecast,
    train_random_forest,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
ICON_PATH = ROOT_DIR / "images" / "icon.png"
LOGO_PATH = ROOT_DIR / "images" / "logo.png"

# Map German event names to English for display
EVENT_LABELS = {
    "Kein Event": "No event",
    "Nationaler Feiertag": "National holiday",
    "Kantonaler Feiertag": "Cantonal holiday",
    "Lokales Event": "Local event",
    "Konfirmation": "Confirmation",
}

# Map German weekday abbreviations to English
WEEKDAY_EN = {"Mo": "Mon", "Di": "Tue", "Mi": "Wed", "Do": "Thu", "Fr": "Fri", "Sa": "Sat", "So": "Sun"}

st.set_page_config(
    page_title="Staffing Restaurant Sonne",
    page_icon=str(ICON_PATH),
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS (with mobile responsiveness)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ---- Google Font ---- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ---- Header area ---- */
    .main-header {
        display: flex;
        align-items: center;
        gap: 1.2rem;
        margin-bottom: 0.5rem;
    }
    .main-header img {
        height: 48px;
        border-radius: 10px;
        object-fit: contain;
    }
    .main-header h1 {
        margin: 0;
        font-size: 1.6rem;
        font-weight: 700;
        line-height: 1.2;
        background: linear-gradient(135deg, #E8913A 0%, #F4C76B 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* ---- Metric cards row ---- */
    .kpi-row {
        display: flex;
        gap: 1rem;
        margin: 1.25rem 0;
        flex-wrap: wrap;
    }
    .kpi-card {
        flex: 1 1 140px;
        min-width: 140px;
        background: linear-gradient(135deg, #1A1F2E 0%, #232A3B 100%);
        border: 1px solid rgba(232,145,58,0.15);
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        text-align: center;
    }
    .kpi-card .kpi-label {
        font-size: 0.78rem;
        color: #8B95A5;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.3rem;
    }
    .kpi-card .kpi-value {
        font-size: 1.55rem;
        font-weight: 700;
        color: #FAFAFA;
    }
    .kpi-card .kpi-sub {
        font-size: 0.75rem;
        color: #6B7585;
        margin-top: 0.15rem;
    }

    /* ---- Daily overview table ---- */
    .overview-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0 0.4rem;
    }
    .overview-table tr {
        background: linear-gradient(135deg, #1A1F2E 0%, #222840 100%);
        transition: background 0.2s;
    }
    .overview-table tr:hover {
        background: linear-gradient(135deg, #1E2438 0%, #283050 100%);
    }
    .overview-table td {
        padding: 0.7rem 0.9rem;
        vertical-align: middle;
        color: #FAFAFA;
    }
    .overview-table td:first-child {
        border-radius: 10px 0 0 10px;
        font-weight: 600;
        white-space: nowrap;
    }
    .overview-table td:last-child {
        border-radius: 0 10px 10px 0;
    }
    .overview-table .ov-day {
        font-size: 0.95rem;
        min-width: 100px;
    }
    .overview-table .ov-weather {
        font-size: 0.8rem;
        color: #8B95A5;
        white-space: nowrap;
    }
    .overview-table .ov-event {
        font-size: 0.75rem;
        color: #E8913A;
    }
    .overview-table .ov-revenue {
        text-align: right;
        font-weight: 700;
        font-size: 1rem;
        color: #E8913A;
        white-space: nowrap;
        min-width: 120px;
    }
    .overview-table .ov-bar-cell {
        width: 30%;
        padding-right: 1rem;
    }
    .ov-bar-bg {
        background: rgba(232,145,58,0.08);
        border-radius: 4px;
        height: 8px;
        width: 100%;
        overflow: hidden;
    }
    .ov-bar-fill {
        height: 100%;
        border-radius: 4px;
        background: linear-gradient(90deg, #E8913A, #F4C76B);
        transition: width 0.4s ease;
    }

    /* ---- Event badge ---- */
    .event-badge {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 500;
        padding: 0.15rem 0.55rem;
        border-radius: 6px;
        background: rgba(232,145,58,0.15);
        color: #E8913A;
    }

    /* ---- Section titles ---- */
    .section-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #FAFAFA;
        margin: 1.5rem 0 0.75rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* ---- Sidebar polish ---- */
    section[data-testid="stSidebar"] > div:first-child {
        background: linear-gradient(180deg, #0E1117 0%, #151B27 100%);
    }
    section[data-testid="stSidebar"] {
        width: 260px !important;
        min-width: 260px !important;
    }

    /* ---- Hide ALL Streamlit branding ---- */
    #MainMenu {display: none !important;}
    footer {display: none !important;}
    header[data-testid="stHeader"] {display: none !important;}
    .stDeployButton {display: none !important;}
    [data-testid="stStatusWidget"] {display: none !important;}
    [data-testid="stToolbar"] {display: none !important;}
    .viewerBadge_container__r5tak {display: none !important;}
    ._profileContainer_gzau3_53 {display: none !important;}
    [data-testid="stDecoration"] {display: none !important;}
    .stAppDeployButton {display: none !important;}
    .reportview-container .main footer,
    .stApp > footer,
    footer.css-164nlkn,
    footer.st-emotion-cache-164nlkn {display: none !important;}
    [data-testid="manage-app-button"] {display: none !important;}
    button[title="View fullscreen"] {display: none !important;}
    .styles_viewerBadge__CvC9N {display: none !important;}
    ._container_gzau3_1 {display: none !important;}
    .stBottom {display: none !important; height: 0 !important; min-height: 0 !important;}
    /* Embed-safe: no border, allow scrolling */
    .stApp {
        outline: none !important;
        border: none !important;
    }
    /* Reduce top gap from hidden header */
    .stMainBlockContainer, .block-container, [data-testid="stAppViewBlockContainer"] {
        padding-top: 1rem !important;
    }

    /* ================================================================
       MOBILE RESPONSIVE STYLES
       ================================================================ */

    /* Tablets and small screens */
    @media (max-width: 768px) {
        .main-header {
            flex-direction: column;
            align-items: flex-start;
            gap: 0.5rem;
        }
        .main-header h1 {
            font-size: 1.35rem;
        }
        .main-header img {
            height: 44px;
        }

        .kpi-row {
            flex-direction: column;
            gap: 0.65rem;
        }
        .kpi-card {
            min-width: unset;
            padding: 0.85rem 1rem;
        }
        .kpi-card .kpi-value {
            font-size: 1.3rem;
        }

        /* Overview table: hide bar + event columns, keep weather */
        .overview-table .ov-bar-cell,
        .overview-table .ov-event {
            display: none;
        }
        .overview-table .ov-weather {
            white-space: normal;
            font-size: 0.7rem;
        }
        .overview-table td {
            padding: 0.4rem 0.35rem;
        }
        .overview-table .ov-revenue {
            font-size: 0.8rem;
            min-width: unset;
            padding-left: 0.2rem;
        }
        .overview-table .ov-day {
            font-size: 0.82rem;
            min-width: unset;
        }

        .section-title {
            font-size: 1.05rem;
        }
    }

    /* Small phones */
    @media (max-width: 480px) {
        .main-header h1 {
            font-size: 1.15rem;
        }
        .main-header img {
            height: 36px;
        }

        .kpi-card .kpi-label {
            font-size: 0.7rem;
        }
        .kpi-card .kpi-value {
            font-size: 1.15rem;
        }

        .overview-table .ov-weather {
            font-size: 0.65rem;
        }
        .overview-table .ov-revenue {
            font-size: 0.75rem;
        }
        .overview-table .ov-day {
            font-size: 0.78rem;
        }
        .overview-table .ov-day span {
            font-size: 0.65rem;
        }
        .overview-table td {
            padding: 0.35rem 0.25rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def weather_icon(sunshine_min: float, precip_mm: float) -> str:
    """Return a weather emoji based on sunshine & precipitation."""
    if precip_mm > 5:
        return "🌧️"
    if precip_mm > 1:
        return "🌦️"
    if sunshine_min > 400:
        return "☀️"
    if sunshine_min > 200:
        return "⛅"
    return "☁️"


def trend_arrow(value: float, avg: float) -> str:
    """Arrow hint relative to average."""
    if value > avg * 1.05:
        return "↑"
    if value < avg * 0.95:
        return "↓"
    return "→"


# ---------------------------------------------------------------------------
# Model loading (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return train_random_forest()


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main():
    # ---- Header ----
    logo_html = ""
    if LOGO_PATH.exists():
        logo_html = f'<img src="data:image/png;base64,{_img_to_base64(LOGO_PATH)}" />'

    st.markdown(
        f"""
        <div class="main-header">
            {logo_html}
            <h1>Restaurant Sonne · Staffing Forecast</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "📍 [Restaurant Sonne Sempachersee](https://www.sonneseehotel.ch/de) · "
        "7-day revenue forecast for staffing planning"
    )

    # ---- Load model ----
    with st.spinner("Loading model …"):
        model, feature_cols, births_map, weather_fallback, history, last_known_date = (
            load_model()
        )

    # ---- Sidebar ----
    with st.sidebar:
        st.markdown("## ⚙️ Settings")

        start_default = max(
            date.today() + timedelta(days=3),
            last_known_date + timedelta(days=1),
        )
        start_date = st.date_input(
            "📅 Start date",
            value=start_default,
            help="Default: today + 3 days",
        )

        st.markdown("---")

        holiday_level = st.selectbox(
            "🎉 Holiday level (all 7 days)",
            options=[
                ("No holidays", 0.0),
                ("Partial holidays", 0.5),
                ("Full holidays", 1.0),
            ],
            format_func=lambda x: x[0],
        )
        holiday_value = holiday_level[1]

        st.markdown("---")
        st.markdown(
            "<small style='color:#6B7585'>📍 Weather location: Sempachersee<br>"
            f"Lat {DEFAULT_LAT:.4f} · Lon {DEFAULT_LON:.4f}</small>",
            unsafe_allow_html=True,
        )

    # ---- Weather ----
    weather_df, source = fetch_weather_forecast(start_date, 7, DEFAULT_LAT, DEFAULT_LON)
    used_fallback = False
    if weather_df.empty:
        weather_df = build_weather_fallback(start_date, 7, weather_fallback)
        used_fallback = True
        st.warning(
            "⚠️ Weather API unavailable – using monthly median fallback."
        )
    else:
        st.sidebar.success(f"✅ Weather: {source}")

    # ---- Week frame + event selection ----
    week_df = initial_week_frame(start_date, weather_df)
    event_options = list(EVENT_WEIGHTS.keys())
    event_display = [EVENT_LABELS.get(e, e) for e in event_options]

    st.markdown('<div class="section-title">📋 Special Events & Holidays</div>', unsafe_allow_html=True)

    selections = []
    cols = st.columns(7)
    weekday_emoji = {"Mon": "📅", "Tue": "📅", "Wed": "📅", "Thu": "📅", "Fri": "🍻", "Sat": "🎉", "Sun": "☀️"}

    for idx, (_, r) in enumerate(week_df.iterrows()):
        with cols[idx]:
            day_date = r["Datum"].date()
            wd_de = r["Wochentag"]
            wd_en = WEEKDAY_EN.get(wd_de, wd_de)
            emoji = weekday_emoji.get(wd_en, "📅")

            sunshine = r.get("Sonnenscheindauer Tagessumme", 0)
            precip = r.get("Niederschlag Tagessumme 6 UTC", 0)
            temp = r.get("Lufttemperatur 2m Tagesmittel", None)
            w_icon = weather_icon(sunshine if pd.notna(sunshine) else 0, precip if pd.notna(precip) else 0)
            temp_str = f"{temp:.0f}°C" if pd.notna(temp) else "–"

            st.markdown(f"**{emoji} {wd_en}**")
            st.caption(f"{day_date.strftime('%d.%m.')}  {w_icon} {temp_str}")

            ev_idx = st.selectbox(
                "Event",
                range(len(event_options)),
                format_func=lambda i: event_display[i],
                key=f"event-{day_date}",
                label_visibility="collapsed",
            )
            ev = event_options[ev_idx]

        selections.append(
            {
                "Datum": r["Datum"],
                "Wochentag": wd_de,
                "Wochentag_en": wd_en,
                "Special Event": ev,
                "Special-Day-Value": EVENT_WEIGHTS[ev],
                "Ferien-Value": holiday_value,
                "Niederschlag Tagessumme 6 UTC": r["Niederschlag Tagessumme 6 UTC"],
                "Sonnenscheindauer Tagessumme": r["Sonnenscheindauer Tagessumme"],
                "Lufttemperatur 2m Tagesmittel": r["Lufttemperatur 2m Tagesmittel"],
            }
        )

    edited_df = pd.DataFrame(selections)

    # ---- Run forecast ----
    st.markdown("")
    run_btn = st.button("🚀  Run Forecast", use_container_width=True, type="primary")

    if run_btn:
        with st.spinner("Calculating forecast …"):
            forecast_df = run_forecast(
                model, feature_cols, history, births_map, weather_df, edited_df
            )

        # Add English weekday column for display
        forecast_df["Weekday"] = forecast_df["Wochentag"].map(WEEKDAY_EN)

        # ---- KPI metrics ----
        total_rev = forecast_df[TARGET_COL].sum()
        avg_rev = forecast_df[TARGET_COL].mean()
        max_rev = forecast_df[TARGET_COL].max()
        min_rev = forecast_df[TARGET_COL].min()
        max_day = forecast_df.loc[forecast_df[TARGET_COL].idxmax()]
        min_day = forecast_df.loc[forecast_df[TARGET_COL].idxmin()]

        st.markdown(
            f"""
            <div class="kpi-row">
                <div class="kpi-card">
                    <div class="kpi-label">Total Revenue</div>
                    <div class="kpi-value">CHF {total_rev:,.0f}</div>
                    <div class="kpi-sub">7 days</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Daily Average</div>
                    <div class="kpi-value">CHF {avg_rev:,.0f}</div>
                    <div class="kpi-sub">Ø per day</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Best Day</div>
                    <div class="kpi-value">CHF {max_rev:,.0f}</div>
                    <div class="kpi-sub">{max_day['Datum']} ({max_day['Weekday']})</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Weakest Day</div>
                    <div class="kpi-value">CHF {min_rev:,.0f}</div>
                    <div class="kpi-sub">{min_day['Datum']} ({min_day['Weekday']})</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ---- Plotly chart ----
        st.markdown('<div class="section-title">📈 Revenue Forecast</div>', unsafe_allow_html=True)

        labels = [r['Weekday'] for _, r in forecast_df.iterrows()]
        hover_labels = [f"{r['Weekday']} {r['Datum']}" for _, r in forecast_df.iterrows()]
        revenues = forecast_df[TARGET_COL].tolist()

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=labels,
                y=revenues,
                mode="lines+markers",
                name="Revenue",
                fill="tozeroy",
                line=dict(color="#E8913A", width=3),
                marker=dict(size=8, color="#E8913A", line=dict(width=2, color="#0E1117")),
                fillcolor="rgba(232,145,58,0.10)",
                customdata=hover_labels,
                hovertemplate="<b>%{customdata}</b><br>CHF %{y:,.0f}<extra></extra>",
            )
        )
        # Average line
        fig.add_hline(
            y=avg_rev,
            line_dash="dot",
            line_color="rgba(250,250,250,0.25)",
            annotation_text=f"Ø {avg_rev:,.0f}",
            annotation_position="top left",
            annotation_font_color="rgba(250,250,250,0.5)",
            annotation_font_size=11,
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=5, t=20, b=10),
            height=320,
            xaxis=dict(
                showgrid=False,
                color="#8B95A5",
                tickfont=dict(size=11),
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor="rgba(250,250,250,0.05)",
                color="#8B95A5",
                tickfont=dict(size=10),
            ),
            showlegend=False,
            hoverlabel=dict(
                bgcolor="#1A1F2E",
                font_size=13,
                font_family="Inter",
                bordercolor="#E8913A",
            ),
        )
        st.plotly_chart(fig, use_container_width=True)

        # ---- Daily overview table ----
        st.markdown('<div class="section-title">📊 Daily Overview</div>', unsafe_allow_html=True)

        max_rev_val = forecast_df[TARGET_COL].max()
        table_rows = ""
        for _, row in forecast_df.iterrows():
            rev = row[TARGET_COL]
            arrow = trend_arrow(rev, avg_rev)
            precip = row["Niederschlag Tagessumme 6 UTC"]
            sunshine = row["Sonnenscheindauer Tagessumme"]
            temp = row["Lufttemperatur 2m Tagesmittel"]
            w_icon = weather_icon(sunshine, precip)
            bar_pct = (rev / max_rev_val * 100) if max_rev_val > 0 else 0
            event_de = row.get("Spezialevent", "Kein Event")
            event_en = EVENT_LABELS.get(event_de, event_de)
            event_cell = f'<span class="event-badge">{event_en}</span>' if event_de and event_de != "Kein Event" else ""

            table_rows += f"""
            <tr>
                <td class="ov-day">{row['Weekday']}<br><span style="font-size:0.75rem;color:#6B7585;font-weight:400">{row['Datum']}</span></td>
                <td class="ov-weather">{w_icon} {temp:.0f}°C &nbsp; 🌧️{precip:.0f}mm &nbsp; ☀️{sunshine:.0f}min</td>
                <td class="ov-event">{event_cell}</td>
                <td class="ov-bar-cell"><div class="ov-bar-bg"><div class="ov-bar-fill" style="width:{bar_pct:.0f}%"></div></div></td>
                <td class="ov-revenue">{arrow} CHF {rev:,.0f}</td>
            </tr>"""

        st.markdown(
            f'<table class="overview-table">{table_rows}</table>',
            unsafe_allow_html=True,
        )

        # ---- Footer info ----
        source_text = (
            f"✅ Weather: {source}"
            if not used_fallback
            else "⚠️ Weather: monthly median fallback (API unavailable)"
        )
        st.caption(source_text)

    else:
        st.markdown("")
        st.info(
            "👆 Select special events for each day above, then click **Run Forecast**."
        )


def _img_to_base64(path: Path) -> str:
    """Convert an image file to a base64 string for inline HTML."""
    import base64

    return base64.b64encode(path.read_bytes()).decode()


if __name__ == "__main__":
    main()
