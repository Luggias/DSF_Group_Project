# Hotel Revenue Forecast
**Authors:** Lukas Kapferer, Maik Klaiber, Matthias Hartmann, Thierry Suhner  
**Institution:** University of St.Gallen (HSG) - Autumn Semester 2025  

---

## 1. Business Case
The objective of this project is forecasting the weekly revenue for **Restaurant Sonne Sempachersee** to improve staffing planning and operational efficiency.

### 1.1 Business Problem
Restaurant Sonne currently faces recurring inefficiencies in weekly staffing: too many employees on low-demand days and understaffing on unexpectedly busy days. This mismatch leads to higher labor costs, reduced service quality and operational stress for the staff.

Every Friday, the manager prepares the staffing plan for the upcoming week. At present, this planning relies on a **naïve revenue model**, which simply uses the average revenue of the past few weeks for each weekday as the basis for forecasting. This leads to:
* **Overstaffing:** unnecessary personnel expenses.
* **Understaffing:** long waiting times, dissatisfied guests, and service bottlenecks.

Given the restaurant's strong dependence on weather, seasonality and local events such as school-holidays and confirmations a systematic forecasting method is needed.

### 1.2 Proposed Solution
Developing a model that predicts daily revenue streams for the next week (Monday–Sunday) using only data that is known by Friday evening. This solution transforms workforce planning from an intuition based to an evidence based approach.

A one week forecast horizon matches the restaurant's operational reality:
* Staff schedules must be completed before Monday.
* Weather forecasts are accurate enough 7 days in advance.
* Weekly revenue patterns exhibit strong weekday-level seasonality, which can be captured well over a one week horizon.

## 2. Feature Engineering Report
Apart from the external features, many features were derived from the time-series data:
* **Lags:** Included lags with a shift of 1, 2, 3, 7, 14, 21, 28, 30 in the past.
* **Date-related:** Month, day of the year, quarter, week of the year and weekday.
* **Rolling Windows:** Mean of the last 7, 14 and 28 days as well as for the standard deviation of the last 7 and 14 days.

### External Features Table
| Feature | Description | Sourced From |
| :--- | :--- | :--- |
| `is_fr` | binary indicator for fridays and saturdays | manually hardcoded |
| `ferien-value` | indicator for regional school holidays | official swiss school holiday calendars |
| `special-day-value` | flag for special days or local events not captured by holidays (Like Erstkommunion, Firmung or Sempacherseelauf) | All dates were confirmed by directly calling the respective parish office |
| `Niederschlag` | Total daily rainfall measured at 06:00 | MeteoSwiss Open Data |
| `Sonnenscheindauer`| total sunshine duration in hours per day | MeteoSwiss Open Data |
| `Lufttemperatur` | daily mean temperature at 2 meters above ground | MeteoSwiss weather station "Luzern" |
| `birthdays` | proxy feature for seasonal event-restaurant visits (birthday celebrations) | Birth rates in Switzerland over the past 100 years |
| `Tagesumsatz` | Daily revenue (CHF) recorded for Restaurant Sonne | Personal contact to restaurant owners |

## 3. Merging Data
To create a consistent modeling dataset, multiple sources were merged and cleaned. Special-events data was imported, harmonized, and combined before being merged with the main revenue dataset. Weather data and long-term average birth statistics were added as additional external features. A binary feature was introduced that assigns a value of 1 to Fridays and Saturdays capturing the elevated weekend demand.

## 4. Analysing and Cleaning
* **Missing Values:** Missing values in event variables were set to zero, and a redundant weather column was removed.
* **Error Correction:** Negative revenue entries caused by data recording errors were replaced using the average revenue of the same weekday in previous weeks.
* **Leakage Prevention:** The columns Restaurant and Terrasse were removed to prevent the model from indirectly "seeing" its own label information.
* **Holiday Boost:** Special days such as Christmas, Mother's Day, and New Year's Eve show extreme revenue deviations. To avoid biasing the model with these outliers, they were excluded from the training set. Predictions on these days were instead replaced by a 3-year historical holiday average, a justified post-processing correction that improves interpretability and robustness.

## 5. Modelling
Before estimating the model, the dataset is split into a training and a testing set to enable proper backtesting and to prevent data leakage. All models use a testing period of 10 months. For comparison purposes, the naïve prediction method currently used by the restaurant was included.

### 5.1 Linear Regression Model
The first model was just a standardized Linear Regression model, which already performed quite well due to the extensive feature engineering and standardizing using a `StandardScaler`.
* **Performance:** MAE: 2311.06 | $R^2$: 0.7685 | MAPE: 21.84%.
* **Ridge Model:** Advanced to a Ridge Model with Cross-Validation using a custom cross validation designed to have an expanding-window time-series splitter (SevenDay Forecast CV).
* **Result:** MAE: 2310.14 | $R^2$: 0.7686 | MAPE: 21.83%.
* **Conclusion:** Linear regression is not particularly well suited to this problem as the data is strongly seasonal and the relationship between time and revenue is not even roughly linear.

### 5.2 Random Forest Model
Random forest models handle nonlinear relationships well, are robust to outliers and multicollinearity and they work naturally with the mix of autoregressive and date-based features. A custom expanding-window cross-validation scheme (SevenDay Forecast CV) was built where each fold simulates a realistic forecasting task: train on the past, predict a 7-day horizon into the future.

Four incremental model setups were compared:
1. **Baseline Random Forest:** 300 Trees, 15 max depth, 5 as minimum number of samples per leaf.
2. **Boosted Random Forest (Hist Gradient Boosting):** Learning-rate 0.05 and 30 Max Leaf Nodes.
3. **Hyperparameter-tuned Random Forest:** Trained using the custom SevenDay ForecastCV.
4. **Boosted model with scaling + log-transformed target:** Log transform stabilizes variance and reduces the influence of large spikes.

The **scaler-system boosted model with log-target transformation** yielded the best overall results.

### 5.3 SARIMAX Model
A SARIMAX model (Seasonal Auto Regressive Integrated Moving Average with exogenous regressors) was employed to forecast the daily revenue.
* **Fourier Series:** To capture smooth yearly seasonality, Fourier series terms were generated using $K=16$ harmonics.
* **Lags:** Lag features (1, 2, 3, 7, 14, 21, 28, and 30 days) were included to significantly enhance the model's responsiveness to short-term dynamics.
* **Performance:** MAPE: 19.01% | MAE: 2281.20 | $R^2$: 0.77.

### 5.4 Prophet Model
The observed time series was modelled as the sum of four different components: Trend, Seasonality, Holidays/Events, and Exogeneous regressors.
* **Custom Seasonality:** Custom yearly seasonality with 16 Fourier terms was added to allow the model to learn fine-grained seasonal curvature (school holiday waves, summer terraces).
* **Holiday Boost:** The built in Holidays function was not used. Instead, the Holiday boost function was applied for stability around Christmas and New Year's Eve.
* **Performance:** MAPE: 19.83% | MAE: 2319.32 | $R^2$: 0.7629.

### 5.5 LSTM Model
The LSTM (Long Short-Term Memory) network is a recurrent neural network architecture designed to capture short-term temporal dependencies in sequential data.
* **Input Structure:** Time series $y(t)$ is learned from a 30-day sliding window of past revenue values and exogenous regressors.
* **Architecture:** Two-layer LSTM with 64 hidden units, trained for 50 epochs.
* **Multivariate Result:** MAPE: 25.65% | MAE: 2744.43 | $R^2$: 0.6740.

## 6. Challenges
During the modelling process, it became apparent that the models exhibited significant instability around December and January. This effect was primarily caused by two substantial outliers: Christmas Eve (revenue consistently extremely low) and New Year's Eve (exponentially higher). This necessitated the manual "Holiday Boost" adjustment. Additionally, computational demands limited applying both cross-validation and grid search simultaneously for the SARIMAX model.

## 7. Overall Conclusion
### Summary of Model Performance Metrics
| Model | $R^2$ | MAE (CHF) | MAPE (%) | Relative MSE |
| :--- | :--- | :--- | :--- | :--- |
| **Naïve** | 0.4691 | 3318.84 | 28.01 | 0.63 |
| **Linear Regression** | 0.7685 | 2311.06 | 21.84 | 0.2544 |
| **Ridge Regression** | 0.7686 | 2310.14 | 21.83 | 0.2542 |
| **Random Forest** | **0.8322** | **1953.02** | **16.61** | **0.2004** |
| **SARIMAX** | 0.77 | 2281.20 | 19.01 | 0.263 |
| **Prophet** | 0.7629 | 2319.32 | 19.83 | 0.2832 |
| **LSTM (univariate)** | 0.6236 | 2862.18 | 26.24 | 0.4135 |
| **LSTM (multivariate)** | 0.6740 | 2744.43 | 25.65 | 0.3582 |



The overall winner model is the **histgradient-boosted random forest** with scaling, custom cross-validation, adjustment for special days and bias correction. It delivers a **MAPE of 16.61%** and an **$R^2$ of 0.8322**, explaining 83.22% of the variability in the data. Compared to the naïve approach, this is an improvement of 11.4% on the MAPE and 1,365.82 CHF on the MAE.

---

# Project Structure
- /data: All of our csv data and excels with special events and daily revenues
- /models: All of our prediction models, title explains which model


# Gewichtungen der Special Days
- Nationale Feiertage: 1
- Kantonale Feiertage: 0.75 (z.B. Firmungen)
- Lokale Feiertage: 0.5
- Konfirmation: 0.1


# Track streamlit uptime
https://stats.uptimerobot.com/3pDOlcAOKW
