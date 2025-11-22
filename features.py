import pandas as pd
import numpy as np
import datetime as dt

filepath_Mosen_daily = "https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn/moa/ogd-smn_moa_h_historical_2020-2029.csv"
df_Mosen_weather = pd.read_csv(filepath_Mosen_daily, sep=';')


filepath_Luzern_weather = "https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn/luz/ogd-smn_luz_d_historical.csv"
df_Luzern_weather = pd.read_csv(filepath_Luzern_weather, sep=';')


filepath_Births_Switzerland_100years = "px-x-0102020204_111.csv"
df_Births = pd.read_csv(filepath_Births_Switzerland_100years, sep=';', encoding='latin1')


filepath_Feiertage_Schweiz_2025 = "Feiertage_Schweiz_2025.csv"
df_public_holidays = pd.read_csv(filepath_Feiertage_Schweiz_2025, sep=',', encoding='latin1')

print(df_Mosen_weather)
print(df_Luzern_weather)
print(df_Births)
print(df_public_holidays)


# select only month columns (all except "Jahr")
month_cols = [c for c in df_Births.columns if c != "Jahr"]

# calculate average per month over all years
avg = df_Births[month_cols].mean()

# turn into a DataFrame and clean it up
avg_birthdays = avg.reset_index()
avg_birthdays.columns = ["month", "average number of births (100 year avg)"]

# optional: remove the German prefix from the month names
avg_birthdays["month"] = avg_df["month"].str.replace("Lebendgeburten im ", "", regex=False)
avg_birthdays


#Weather Data:
# convert timestamp column to datetime
df_Luzern_weather["reference_timestamp"] = pd.to_datetime(
    df_Luzern_weather["reference_timestamp"], dayfirst=True)

# define date range
start_date = pd.Timestamp("2020-01-01")
end_date = pd.Timestamp(dt.datetime.today().date())

# filter
df_weather_filtered = df_Luzern_weather[(df_Luzern_weather["reference_timestamp"] >= start_date) & (df_Luzern_weather["reference_timestamp"] <= end_date)]

#select columns with relevant data
selected_columns_weather = ["reference_timestamp","rka150d0","rre150d0","sre000d0","tre200d0",]

df_weather_final = df_weather_filtered[selected_columns_weather]

df_weather_final

df_weather_final = df_weather_final.rename(columns={
    "reference_timestamp": "Datum", "rka150d0": "Niederschlag Tagessumme 0 UTC", "rre150d0": "Niederschlag Tagessumme 6 UTC", "sre000d0": "Sonnenscheindauer Tagessumme", "tre200d0": "Lufttemperatur 2m Tagesmittel"})

df_weather_final = df_weather_final.reset_index(drop=True)
df_weather_final

