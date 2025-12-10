import streamlit as st
import pandas as pd

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

st.set_page_config(
    page_title="Staffing Restaurant Sonne",
    page_icon=str(ICON_PATH),
    layout="wide",
)


@st.cache_resource
def load_model():
    return train_random_forest()


def main():
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=160)
    st.title("Staffing estimation for Restaurant Sonne")
    st.write(
        "Please select upcoming holidays in the left menu.  \n"
        "Set special events for each of the next 7 days here:  \n"
        "[Restaurant website](https://www.sonneseehotel.ch/de)"
    )

    model, feature_cols, births_map, weather_fallback, history, last_known_date = load_model()

    with st.sidebar:
        st.header("Settings")
        start_default = max(date.today() + timedelta(days=3), last_known_date + timedelta(days=1))
        start_date = st.date_input("Start date (day 1 of forecast)", value=start_default, help="Default: today + 3 days")
        st.caption("Weather location is fixed: Restaurant Sonne Sempachersee (coordinates stored in code).")
        st.markdown("---")
        holiday_level = st.selectbox(
            "Holiday level (applied to all 7 days)",
            options=[("No holidays", 0.0), ("Partial holidays", 0.5), ("Full holidays", 1.0)],
            format_func=lambda x: x[0],
        )
        holiday_value = holiday_level[1]

    weather_df, source = fetch_weather_forecast(start_date, 7, DEFAULT_LAT, DEFAULT_LON)
    used_fallback = False
    if weather_df.empty:
        weather_df = build_weather_fallback(start_date, 7, weather_fallback)
        used_fallback = True
        st.warning("Weather API not available or no data for the chosen start date. Using monthly median fallback.")
    else:
        st.caption(f"Weather source: {source} (fixed restaurant coordinates)")

    week_df = initial_week_frame(start_date, weather_df)
    event_options = list(EVENT_WEIGHTS.keys())

    selections = []
    st.subheader("Special events & holidays (per day)")
    for _, r in week_df.iterrows():
        col1, col2 = st.columns([2, 1])
        with col1:
            ev = st.selectbox(
                f"{r['Datum'].date()} – {r['Wochentag']}",
                event_options,
                key=f"event-{r['Datum'].date()}",
            )
        with col2:
            st.write(f"Holiday value: {holiday_value}")
        selections.append(
            {
                "Datum": r["Datum"],
                "Wochentag": r["Wochentag"],
                "Special Event": ev,
                "Special-Day-Value": EVENT_WEIGHTS[ev],
                "Ferien-Value": holiday_value,
                "Niederschlag Tagessumme 6 UTC": r["Niederschlag Tagessumme 6 UTC"],
                "Sonnenscheindauer Tagessumme": r["Sonnenscheindauer Tagessumme"],
                "Lufttemperatur 2m Tagesmittel": r["Lufttemperatur 2m Tagesmittel"],
            }
        )

    edited_df = pd.DataFrame(selections)

    if st.button("Run forecast"):
        forecast_df = run_forecast(model, feature_cols, history, births_map, weather_df, edited_df)
        st.subheader("Forecast (7 days)")

        # Revenue line chart (index as string to avoid time-of-day ticks)
        chart_df = forecast_df.copy()
        chart_df["Label"] = chart_df["Datum"].astype(str)
        st.line_chart(chart_df.set_index("Label")[TARGET_COL])

        # Per-day overview (no table)
        st.markdown("#### Daily overview")
        for i, (_, row) in enumerate(forecast_df.iterrows()):
            col1, col2 = st.columns([1.4, 1])
            with col1:
                st.markdown(
                    f"**{row['Datum']} ({row['Wochentag']})**  \n"
                    f"Revenue: `{row[TARGET_COL]:,.0f}`"
                )
            with col2:
                st.markdown(
                    f"Temp mean: `{row['Lufttemperatur 2m Tagesmittel']:.1f} °C`  \n"
                    f"Precipitation: `{row['Niederschlag Tagessumme 6 UTC']:.1f} mm`  \n"
                    f"Sunshine: `{row['Sonnenscheindauer Tagessumme']:.0f} min`"
                )
            st.divider()

        info_text = (
            f"Weather source: {source}"
            if not used_fallback
            else "Weather source: monthly median fallback (API unavailable)"
        )
        st.caption(info_text)
    else:
        st.info("Set events/holidays per day, then click **Run forecast**.")


if __name__ == "__main__":
    main()
