"""
Hilfsfunktionen für das Random-Forest-Frontend:
- Daten laden und Feature-Engineering
- Modell trainieren
- Iterativen 7-Tage-Forecast erzeugen
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import requests

from datetime import date, timedelta
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from typing import Dict, List, Tuple


DATA_PATH = Path("data/dataset-merged-2.csv")
TARGET_COL = "Tagesumsatz Restaurant"
EVENT_WEIGHTS: Dict[str, float] = {
    "Kein Event": 0.0,
    "Nationaler Feiertag": 1.0,
    "Kantonaler Feiertag": 0.75,
    "Lokales Event": 0.5,
    "Konfirmation": 0.1,
}
WEATHER_COLUMN_MAP = {
    "precipitation_sum": "Niederschlag Tagessumme 6 UTC",
    "sunshine_duration": "Sonnenscheindauer Tagessumme",
    "temperature_2m_mean": "Lufttemperatur 2m Tagesmittel",
}
DEFAULT_LAT = 47.15039838353179  # Restaurant Sonne Sempachersee
DEFAULT_LON = 8.166927097595224


def load_raw_data() -> Tuple[pd.DataFrame, Dict[int, float], pd.DataFrame]:
    df = pd.read_csv(DATA_PATH)
    df["Datum"] = pd.to_datetime(df["Datum"])
    births_map = (
        df.groupby(df["Datum"].dt.month)[
            "Durchschnittliche Anzahl Geburtstage (über 100 Jahre)"
        ]
        .mean()
        .to_dict()
    )
    weather_fallback = (
        df.groupby(df["Datum"].dt.month)[
            [
                "Niederschlag Tagessumme 6 UTC",
                "Sonnenscheindauer Tagessumme",
                "Lufttemperatur 2m Tagesmittel",
            ]
        ]
        .median()
    )
    return df, births_map, weather_fallback


def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    df_feat = df.copy()
    for lag in [1, 2, 3, 7, 14, 21, 28, 30]:
        df_feat[f"lag{lag}"] = df_feat[TARGET_COL].shift(lag)

    df_feat["month"] = df_feat["Datum"].dt.month
    df_feat["dayofyear"] = df_feat["Datum"].dt.dayofyear
    df_feat["quarter"] = df_feat["Datum"].dt.quarter
    df_feat["weekofyear"] = df_feat["Datum"].dt.isocalendar().week.astype(int)
    df_feat["weekday"] = df_feat["Datum"].dt.weekday

    df_feat["roll_mean_7"] = df_feat[TARGET_COL].shift(1).rolling(7).mean()
    df_feat["roll_mean_14"] = df_feat[TARGET_COL].shift(1).rolling(14).mean()
    df_feat["roll_mean_28"] = df_feat[TARGET_COL].shift(1).rolling(28).mean()
    df_feat["roll_std_7"] = df_feat[TARGET_COL].shift(1).rolling(7).std()
    df_feat["roll_std_14"] = df_feat[TARGET_COL].shift(1).rolling(14).std()

    df_feat = df_feat.dropna().reset_index(drop=True)

    num_cols = df_feat.select_dtypes(include=["number"]).columns
    feature_cols = [c for c in num_cols if c != TARGET_COL]
    return df_feat, feature_cols


def train_random_forest():
    df_raw, births_map, weather_fallback = load_raw_data()
    df_feat, feature_cols = engineer_features(df_raw.drop(columns=["Restaurant", "Terrasse"], errors="ignore"))

    model = RandomForestRegressor(
        n_estimators=300,
        max_features="sqrt",
        max_depth=15,
        min_samples_leaf=5,
        random_state=72,
    )
    model.fit(df_feat[feature_cols], df_feat[TARGET_COL])

    history = (
        df_raw.sort_values("Datum")[TARGET_COL]
        .ffill()
        .bfill()
        .tolist()
    )
    last_known_date = df_raw["Datum"].max().date()
    return model, feature_cols, births_map, weather_fallback, history, last_known_date


def fetch_weather_forecast(
    start_date: date, days: int, lat: float, lon: float
) -> Tuple[pd.DataFrame, str]:
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ",".join(WEATHER_COLUMN_MAP.keys()),
        "forecast_days": days,
        "timezone": "Europe/Berlin",
    }
    url = "https://api.open-meteo.com/v1/forecast"
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        daily = resp.json().get("daily", {})
        df_weather = pd.DataFrame(
            {
                "Datum": pd.to_datetime(daily.get("time", [])),
                "precipitation_sum": daily.get("precipitation_sum", []),
                "sunshine_duration": np.array(daily.get("sunshine_duration", [])) / 60.0,
                "temperature_2m_mean": daily.get("temperature_2m_mean", []),
            }
        )
        for src, dst in WEATHER_COLUMN_MAP.items():
            if src in df_weather:
                df_weather[dst] = df_weather[src]
        df_weather = df_weather[["Datum"] + list(WEATHER_COLUMN_MAP.values())]
        df_weather = df_weather[
            (df_weather["Datum"].dt.date >= start_date)
            & (df_weather["Datum"].dt.date < start_date + timedelta(days=days))
        ].reset_index(drop=True)
        return df_weather, "Open-Meteo API"
    except Exception as exc:  # noqa: BLE001
        return pd.DataFrame(), f"API-Fehler: {exc}"


def build_weather_fallback(start_date: date, days: int, fallback_stats: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for offset in range(days):
        d = start_date + timedelta(days=offset)
        month_stats = fallback_stats.loc[d.month]
        rows.append(
            {
                "Datum": pd.Timestamp(d),
                "Niederschlag Tagessumme 6 UTC": month_stats["Niederschlag Tagessumme 6 UTC"],
                "Sonnenscheindauer Tagessumme": month_stats["Sonnenscheindauer Tagessumme"],
                "Lufttemperatur 2m Tagesmittel": month_stats["Lufttemperatur 2m Tagesmittel"],
            }
        )
    return pd.DataFrame(rows)


def initial_week_frame(start_date: date, weather: pd.DataFrame) -> pd.DataFrame:
    dates = [start_date + timedelta(days=i) for i in range(7)]
    weekday = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]
    df = pd.DataFrame(
        {
            "Datum": pd.to_datetime(dates),
            "Wochentag": [weekday[d.weekday()] for d in dates],
            "Ferien-Value": 0.0,
            "Special Event": "Kein Event",
        }
    )
    df = df.merge(weather, on="Datum", how="left")
    return df


def feature_row(
    history: List[float],
    forecast_date: pd.Timestamp,
    weather_row: pd.Series,
    ferien_value: float,
    special_value: float,
    births_map: Dict[int, float],
) -> Dict[str, float]:
    feats: Dict[str, float] = {
        "is_fr_sa": 1 if forecast_date.weekday() in (4, 5) else 0,
        "Ferien-Value": float(ferien_value),
        "Special-Day-Value": float(special_value),
        "Niederschlag Tagessumme 6 UTC": float(weather_row["Niederschlag Tagessumme 6 UTC"]),
        "Sonnenscheindauer Tagessumme": float(weather_row["Sonnenscheindauer Tagessumme"]),
        "Lufttemperatur 2m Tagesmittel": float(weather_row["Lufttemperatur 2m Tagesmittel"]),
        "Durchschnittliche Anzahl Geburtstage (über 100 Jahre)": float(
            births_map.get(forecast_date.month, np.mean(list(births_map.values())))
        ),
        "month": forecast_date.month,
        "dayofyear": forecast_date.dayofyear,
        "quarter": forecast_date.quarter,
        "weekofyear": int(forecast_date.isocalendar().week),
        "weekday": forecast_date.weekday(),
    }

    for lag in [1, 2, 3, 7, 14, 21, 28, 30]:
        feats[f"lag{lag}"] = history[-lag]

    for window, name in [(7, "roll_mean_7"), (14, "roll_mean_14"), (28, "roll_mean_28")]:
        feats[name] = float(np.mean(history[-window:]))
    feats["roll_std_7"] = float(np.std(history[-7:], ddof=1))
    feats["roll_std_14"] = float(np.std(history[-14:], ddof=1))

    return feats


def run_forecast(
    model,
    feature_cols: List[str],
    history: List[float],
    births_map: Dict[int, float],
    weather_df: pd.DataFrame,
    user_df: pd.DataFrame,
) -> pd.DataFrame:
    hist_values = list(history)
    forecasts = []

    for _, row in user_df.iterrows():
        forecast_date = pd.to_datetime(row["Datum"]).normalize()
        weather_row = weather_df.loc[weather_df["Datum"] == forecast_date]
        if weather_row.empty:
            weather_row = weather_df.iloc[[-1]]
        weather_row = weather_row.iloc[0]

        feats = feature_row(
            hist_values,
            forecast_date,
            weather_row,
            row["Ferien-Value"],
            row["Special-Day-Value"],
            births_map,
        )
        pred = float(model.predict(pd.DataFrame([feats])[feature_cols])[0])
        hist_values.append(pred)

        forecasts.append(
            {
                "Datum": forecast_date.date(),
                "Wochentag": row["Wochentag"],
                "Spezialevent": row["Special Event"],
                "Ferien-Value": row["Ferien-Value"],
                TARGET_COL: pred,
                "Niederschlag Tagessumme 6 UTC": feats["Niederschlag Tagessumme 6 UTC"],
                "Sonnenscheindauer Tagessumme": feats["Sonnenscheindauer Tagessumme"],
                "Lufttemperatur 2m Tagesmittel": feats["Lufttemperatur 2m Tagesmittel"],
            }
        )
    return pd.DataFrame(forecasts)
