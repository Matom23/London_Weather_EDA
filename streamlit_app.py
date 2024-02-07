import streamlit as st
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import statsmodels.api as sm
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose, STL
# from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.forecasting.stl import STLForecast
# stats
from statsmodels.api import tsa # time series analysis
import statsmodels.api as sm

# Load data
@st.cache_data
def load_data():
    # Load your dataset here
    df = pd.read_csv("C:\\Users\\mitth\\Documents\\Brainstation\\Final_Project\\London_Weather_EDA\\clean_data.csv")  # Update with your dataset path
    return df

# Load image
from PIL import Image
image = Image.open("C:\\Users\\mitth\\Documents\\Brainstation\\Final_Project\\London_Weather_EDA\\Demo\\Homepage.png")


lw_df = load_data()

print(lw_df.columns)
print(lw_df.index)
lw_df['DATE'] = pd.to_datetime(lw_df['DATE'], format='%Y-%m-%d')

lw_df.set_index("DATE", inplace=True)


# Table of Contents
st.sidebar.title("PAGE")
options = ["Home", "Time Series Analysis", "Regressor Model"]
selection = st.sidebar.radio("", options)

# Main content
if selection == "Home":
    st.title("Prediction of Precipitation in London")
    
    st.image(image, use_column_width=True)

    
elif selection == "Time Series Analysis":
    st.title("Time Series Analysis")
    
    # Section 1: Forecasting
    st.header("Forecasting")
    # Add code for time series forecasting
    
    # Section 2: Baseline Forecast and evaluation
    st.header("STL decomposition")
    # Add code for baseline forecast and evaluation
    
    lw_df_monthly = lw_df.resample("MS").mean()

    lw_df_monthly.head()

    stl_decomposition = tsa.STL(lw_df_monthly["RR"], robust=True)

    result = stl_decomposition.fit()

    lw_df_monthly["Trend"] = result.trend
    lw_df_monthly["Seasonal"] = result.seasonal
    lw_df_monthly["Residual"] = lw_df_monthly["RR"] - result.trend - result.seasonal

    # Separating into Trend, Seasonality and Residual
    cols = ["Trend", "Seasonal", "Residual"]

    fig_separated = make_subplots(rows=3, cols=1, subplot_titles=cols)

    for i, col in enumerate(cols):
        fig_separated.add_trace(
            go.Scatter(x=lw_df_monthly.index, y=lw_df_monthly[col]),
            row=i+1,
            col=1
        )

    fig_separated.update_layout(height=800, width=1200, showlegend=False)
    st.plotly_chart(fig_separated)



    # Calculate the seasonal difference
    lw_df_monthly["seasonal_difference"] = lw_df_monthly["RR"].diff(12)




    # Section 3: SARIMAX
    st.header("Seasonal Difference using SARIMAX")
    # the "MS" option specifies Monthly frequency by Start day
    
    # Filter the seasonal difference data for train and test ranges
    train = lw_df_monthly.loc[lw_df_monthly.index <= "2020-01-01", "seasonal_difference"].dropna()
    test = lw_df_monthly.loc[lw_df_monthly.index >= "2020-01-01", "seasonal_difference"]

    # Perform STL decomposition on the seasonal difference data
    stl_decomposition = STL(train, robust=True)
    result = stl_decomposition.fit()

    # Obtain decomposed components
    trend = result.trend
    seasonal = result.seasonal
    residual = result.resid

    # Define SARIMA parameters based on results from ADF, ACF & PCAF 
    order = (1, 0, 1)  # ARIMA order
    seasonal_order = (1,0,1,24)  # Seasonal order (monthly data)

    # Create SARIMA model for trend component
    sarima_model_trend = SARIMAX(trend, order=order, seasonal_order=seasonal_order)
    sarima_result_trend = sarima_model_trend.fit()

    # Forecast future values for trend component
    trend_forecast = sarima_result_trend.forecast(steps=len(test))

    # Create SARIMA model for seasonal component
    sarima_model_seasonal = SARIMAX(seasonal, order=order, seasonal_order=seasonal_order)
    sarima_result_seasonal = sarima_model_seasonal.fit()

    # Forecast future values for seasonal component
    seasonal_forecast = sarima_result_seasonal.forecast(steps=len(test))

    # Create SARIMA model for residual component
    sarima_model_residual = SARIMAX(residual, order=order, seasonal_order=seasonal_order)
    sarima_result_residual = sarima_model_residual.fit()

    # Forecast future values for residual component
    residual_forecast = sarima_result_residual.forecast(steps=len(test))

    # Combine forecasts from individual components to obtain final forecast
    final_forecast = trend_forecast + seasonal_forecast + residual_forecast

    # Plot the actual seasonal difference and final forecast
    fig_SARIMA = go.Figure()
    fig_SARIMA.add_trace(go.Scatter(x=train.index, y=train, mode='lines', name="Train (Actual)"))
    fig_SARIMA.add_trace(go.Scatter(x=test.index, y=test, mode='lines', name="Test (Actual)"))
    fig_SARIMA.add_trace(go.Scatter(x=test.index, y=final_forecast.values, mode='lines', name="Forecast"))
    fig_SARIMA.update_xaxes(rangeslider_visible=True)
    fig_SARIMA.update_layout(
        yaxis_title="Seasonal Difference", 
        xaxis_title="Date",
        title="Seasonal Difference Forecast using SARIMA"
    )   

    st.plotly_chart(fig_SARIMA)
    
elif selection == "Regressor Model":
    st.title("Regressor Model")
    
    # Model: XGBoost
    st.header("XGBoost Tuned")
    # Add code for XGBoost model

    from sklearn.preprocessing import PolynomialFeatures

    # Select the features for which you want to create interaction terms
    selected_features = ['CC', 'HU', 'QQ', 'TX', 'TN', 'TG', 'PP', 'SD', 'SS', 'YEAR', 'MONTH', 'DAY', 'DAYOFWEEK', 'DAYOFYEAR', 'sin_dayofyear', 'cos_dayofyear']

    # Extract the selected features from your DataFrame lw_df
    X_selected = lw_df[selected_features]

    # Create an instance of PolynomialFeatures
    poly_features = PolynomialFeatures(degree=2, include_bias=False)

    # Fit and transform the selected features to generate polynomial and interaction features
    X_poly = poly_features.fit_transform(X_selected)

    # Convert the transformed features into a DataFrame for further analysis
    X_poly_df = pd.DataFrame(X_poly)

    print(lw_df.shape)

    print(X_poly_df.shape)

    print(X_poly_df)

    # Concatenate the polynomial features DataFrame with the original DataFrame lw_df
    lw_df_poly = pd.concat([lw_df.reset_index(drop=True), X_poly_df.reset_index(drop=True)], axis=1)

    print(lw_df_poly.set_index(lw_df.index, inplace=True))

    # Get the integer indices of the columns you want to rename
    column_indices = [40, 36, 35, 59, 92, 100, 21]

    # Define the new column names
    new_column_names = ['PFeature1', 'PFeature2', 'PFeature3', 'PFeature4', 'PFeature5', 'PFeature6', 'PFeature7']

    # Rename the columns using the integer indices and new column names
    for i, new_name in zip(column_indices, new_column_names):
        lw_df_poly.rename({lw_df_poly.columns[i]: new_name}, inplace=True, axis=1)

    # Get the column names of lw_df_poly
    column_names = lw_df_poly.columns

    # Check if the desired column names exist in the DataFrame
    desired_columns = ['PP', 'PFeature1', 'PFeature2', 'PFeature3', 'PFeature4', 'PFeature5', 'TX', 'PFeature6', 'PFeature7']
    missing_columns = [col for col in desired_columns if col not in column_names]

    # Step 1: Data Preparation
    X = lw_df_poly[["PP",'PFeature1', 'PFeature2', 'PFeature3', 'PFeature4', 'PFeature5', "TX", 'PFeature6', 'PFeature7']]
    y = lw_df_poly["RR"]

    print(y)

    # Step 2: Split data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=123)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=123)
   
    print(y_train, y_test)

    # Step 3: Model Training - Instantiate and fit XGBRegressor with regularization
    model4 = XGBRegressor(colsample_bytree=0.8, learning_rate=0.01, max_depth=5, min_child_weight=1, n_estimators=300, reg_alpha=0, reg_lambda=1.0, subsample=0.6)  # Apply regularization parameters
    model4.fit(X_train, y_train)  # Fit the model

    # Step 4: Model Evaluation
    y_pred_train = model4.predict(X_train)
    y_pred_val = model4.predict(X_val)
    y_pred_test = model4.predict(X_test)


    fig_XG = go.Figure()
    fig_XG.add_trace(go.Scatter(x=y_train.sort_index(ascending=True).index, y=y_train.sort_index(ascending=True).values, mode='lines', name="Train (Actual)"))
    fig_XG.add_trace(go.Scatter(x=X_test.sort_index(ascending=True).index, y=y_test.sort_index(ascending=True).values, mode='lines', name="Test (Actual)"))
    fig_XG.add_trace(go.Scatter(x=X_test.sort_index(ascending=True).index, y=y_pred_test, mode='lines', name="Forecast"))
    fig_XG.update_xaxes(rangeslider_visible=True)
    fig_XG.update_layout(
        yaxis_title="Precipitation (mm)", 
        xaxis_title="Date",
        title="Precipitation Forecast using XGBoost (Tuned)"
    )  

    st.plotly_chart(fig_XG)



