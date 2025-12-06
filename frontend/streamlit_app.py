import streamlit as st

from datetime import date, timedelta
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


st.set_page_config(
    page_title="7-Tage Prognose",
    page_icon="📈",
    layout="wide",
)


@st.cache_resource
def load_model():
    return train_random_forest()


def main():
    st.title("📈 Wochenforecast mit Spezialevents und Wetter")
    st.write(
        "Trainiert wird auf `data/dataset-merged-2.csv` (Tagesumsatz als Ziel). "
        "Special Events können für die nächsten 7 Tage gesetzt werden; Wetter kommt per API oder Monatsmedian-Fallback."
    )

    model, feature_cols, births_map, weather_fallback, history, last_known_date = load_model()

    with st.sidebar:
        st.header("Einstellungen")
        start_default = max(date.today(), last_known_date + timedelta(days=1))
        start_date = st.date_input("Startdatum (Tag 1 der Prognose)", value=start_default, help="Standard: heute")
        st.caption("Wetterstandort fix: Restaurant Sonne Sempachersee (Koordinaten im Code hinterlegt).")

    weather_df, source = fetch_weather_forecast(start_date, 7, DEFAULT_LAT, DEFAULT_LON)
    used_fallback = False
    if weather_df.empty:
        weather_df = build_weather_fallback(start_date, 7, weather_fallback)
        used_fallback = True
        st.warning("Wetter-API nicht verfügbar oder liefert keine Daten für das Startdatum. Nutze Monatsmedian-Fallback.")
    else:
        st.caption(f"Wetterquelle: {source} (fixe Koordinaten Luzern)")

    week_df = initial_week_frame(start_date, weather_df)
    edited_df = st.data_editor(
        week_df,
        column_config={
            "Datum": st.column_config.DateColumn("Datum", disabled=True),
            "Wochentag": st.column_config.TextColumn("Wochentag", disabled=True),
            "Ferien-Value": st.column_config.NumberColumn("Ferien-Value", min_value=0.0, max_value=1.0, step=0.25),
            "Special Event": st.column_config.SelectboxColumn(
                "Special Event",
                options=list(EVENT_WEIGHTS.keys()),
                required=True,
            ),
            "Niederschlag Tagessumme 6 UTC": st.column_config.NumberColumn("Niederschlag (mm)"),
            "Sonnenscheindauer Tagessumme": st.column_config.NumberColumn("Sonnenschein (Minuten)"),
            "Lufttemperatur 2m Tagesmittel": st.column_config.NumberColumn("Ø Temp (°C)"),
        },
        hide_index=True,
        use_container_width=True,
        num_rows="fixed",
    )

    edited_df["Special-Day-Value"] = edited_df["Special Event"].map(EVENT_WEIGHTS)
    if st.button("Prognose berechnen"):
        forecast_df = run_forecast(model, feature_cols, history, births_map, weather_df, edited_df)
        st.subheader("Prognose (7 Tage)")
        st.dataframe(forecast_df, use_container_width=True)
        st.bar_chart(forecast_df.set_index("Datum")[TARGET_COL])

        info_text = (
            f"Wetterquelle: {source}"
            if not used_fallback
            else "Wetterquelle: Monatsmedian aus Trainingsdaten (API nicht erreichbar)"
        )
        st.caption(info_text)
    else:
        st.info("Events setzen/anpassen und dann auf **Prognose berechnen** klicken.")


if __name__ == "__main__":
    main()
