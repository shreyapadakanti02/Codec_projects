import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# Step 1: Download historical stock data
def get_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data


# Step 2: Prepare the data
def prepare_data(data, forecast_days=30):
    data['Prediction'] = data['Close'].shift(-forecast_days)
    X = np.array(data[['Close']])
    X = X[:-forecast_days]
    y = np.array(data['Prediction'])[:-forecast_days]
    return X, y, data


# Step 3: Train/test split and model training
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, X_test, y_test


# Step 4: Predict and visualize
def predict_and_plot(model, data, forecast_days=30):
    forecast = np.array(data[['Close']])[-forecast_days:]
    prediction = model.predict(forecast)

    # Create a DataFrame for future dates
    future_dates = pd.date_range(start=data.index[-1], periods=forecast_days + 1, inclusive='right')
    forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted_Price': prediction})
    forecast_df.set_index('Date', inplace=True)

    # Plotting
    plt.figure(figsize=(14, 6))
    plt.plot(data['Close'], label='Actual Price')
    plt.plot(forecast_df['Predicted_Price'], label='Predicted Price', linestyle='--')
    plt.title('Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

    return forecast_df


# Main function
def main():
    ticker = 'AAPL'  # Change to any stock symbol
    start_date = '2020-01-01'
    end_date = '2023-12-31'
    forecast_days = 30

    data = get_stock_data(ticker, start_date, end_date)
    X, y, data = prepare_data(data, forecast_days)
    model, X_test, y_test = train_model(X, y)

    # Evaluation
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error: {mse:.2f}")

    # Predict and Plot
    forecast_df = predict_and_plot(model, data, forecast_days)
    print(forecast_df)


if __name__ == "_main_":
    main()