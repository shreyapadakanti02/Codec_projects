# Weather Data Analysis and Prediction Example

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the dataset
df = pd.read_csv("C:\\Users\\LENOVO\\OneDrive\\Desktop\\Task1\\weather_data.csv")

# Step 2: Explore the dataset
print("Dataset Preview:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nDataset Description:")
print(df.describe())

# Step 3: Check for missing values
print("\nMissing Values:\n", df.isnull().sum())
df = df.dropna()  # Remove rows with missing values if any

# Step 4: Visualize temperature trends
plt.figure(figsize=(10,5))
plt.plot(pd.to_datetime(df['Date']), df['Temperature'], marker='o')
plt.title('Temperature Trend Over Time')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.grid(True)
plt.show()

# Step 5: Prepare data for prediction
df['Date'] = pd.to_datetime(df['Date'])
df['Day'] = df['Date'].dt.day
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year

# Features and target
X = df[['Day', 'Month', 'Year']]
y = df['Temperature']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 7: Make predictions
y_pred = model.predict(X_test)

# Step 8: Evaluate model
print("\nMean Squared Error:", mean_squared_error(y_test, y_pred))
print("R-squared Score:", r2_score(y_test, y_pred))

# Step 9: Predict future temperature
future_date = pd.DataFrame({'Day':[15], 'Month':[10], 'Year':[2025]})
future_temp = model.predict(future_date)
print("\nPredicted Temperature on 2025-10-15:", future_temp[0])

# Optional: Plot actual vs predicted temperatures
plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('Actual Temperature')
plt.ylabel('Predicted Temperature')
plt.title('Actual vs Predicted Temperature')
plt.grid(True)
plt.show()