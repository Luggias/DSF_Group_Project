import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

def mothers_day(year: int) -> pd.Timestamp:
    """Muttertag CH = 2. Sonntag im Mai."""
    may_first = pd.Timestamp(year, 5, 1)
    first_sunday_offset = (6 - may_first.weekday()) % 7  # Monday=0, Sunday=6
    first_sunday = may_first + pd.Timedelta(days=first_sunday_offset)
    second_sunday = first_sunday + pd.Timedelta(days=7)
    return second_sunday

def is_christmas(date: pd.Timestamp) -> bool:
    """Weihnachten: 24.12."""
    return (date.month == 12) and (date.day == 24)

def is_silvester(date: pd.Timestamp) -> bool:
    """Silvester: 31.12."""
    return (date.month == 12) and (date.day == 31)

def is_mothers_day(date: pd.Timestamp) -> bool:
    """Muttertag (CH) für das entsprechende Jahr."""
    return date == mothers_day(date.year)

def is_special_day(date: pd.Timestamp) -> bool:
    """True, wenn Weihnachten ODER Silvester ODER Muttertag."""
    return is_christmas(date) or is_silvester(date) or is_mothers_day(date)

def holiday_average_last_3_years(df_full: pd.DataFrame, date_func, year: int, target_col: str):
    """
    Durchschnitt des Zielwerts am Feiertag der letzten 3 Jahre (year-1, year-2, year-3).
    date_func(year) muss ein Timestamp des Feiertags zurückgeben.
    """
    years = [year - 1, year - 2, year - 3]
    values = []

    for y_ in years:
        d_ = date_func(y_)
        val = df_full.loc[df_full["Datum"] == d_, target_col]
        if len(val) > 0:
            values.append(val.values[0])

    if len(values) == 0:
        return None
    return np.mean(values)