"""
California Housing Price Prediction 
---------------------------------------
- Dataset : Calitornia Housing (from sklearn)
- Model : Linear Regression 
- Steps : Load -> Preprocess -> Train ->  Evaluate -> Predict -> Visualize 
"""

# 1. Imports 

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score  

# 2. Load Data 

housing = fetch_california_housing(as_frame=True).frame

# 3. Features and Target

x = housing.drop("MedHouseVal", axis=1)
y = housing["MedHouseVal"]

# 4. Train/Test Split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 5. Pipeline(Scaling + Model)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])
pipeline.fit(x_train, y_train)

# 6. Predictions& Evaluation

predictions = pipeline.predict(x_test)

mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# 7. Predict new data

new_house = np.array([[8.3252, 41, 6.9841, 1.0238, 322, 2.5556, 37.88, -122.23]])
predicted_price = pipeline.predict(new_house)

# 8. Visualization

sorted_idx = np.argsort(y_test.values)
y_sorted = y_test.values[sorted_idx]
pred_sorted = predictions[sorted_idx]

plt.figure(figsize=(10, 6))
plt.scatter(y_sorted, pred_sorted, alpha=0.3, color='blue', s=10, label="Predicted")


z = np.polyfit(y_sorted, pred_sorted, 1)
p = np.poly1d(z)
plt.plot(y_sorted, p(y_sorted), "r--", label="Regression Line")


plt.plot([y_sorted.min(), y_sorted.max()], [y_sorted.min(), y_sorted.max()], 'gray', linestyle='dotted', label="Perfect Prediction")


plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


print("\nExample Predictions vs Actual:")
for pred, actual in zip(predictions[:5], y_test.values[:5]):
    print(f"Predicted: {pred:.2f} | Actual: {actual:.2f}")

print("\nModel Performance:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

print("\nNew House Prediction:")
print(f"Predicted Price: {predicted_price[0]:,.2f}")

