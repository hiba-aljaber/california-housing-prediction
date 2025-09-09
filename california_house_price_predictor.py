import pandas as pd 
import numpy as np 
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,r2_score 
import matplotlib.pyplot as plt 


# 2. Load the California Housing dataset and Convert the dataset into a pandas DataFrame

housing = fetch_california_housing(as_frame= True ).frame


# 4. Explore the data briefly:


#    - Print the first few rows
print(housing[:5])
#    - Check shape and column names
print("Dataset frame :" , housing.shape )
#    - Check for missing values
print("Columns names :" , housing.columns.tolist() )
# 5. Split the data into features (X) and target (y)

#    - X = all columns except "MedHouseVal"
x = housing.drop("MedHouseVal",axis = 1 )
#    - y = the "MedHouseVal" column (what we want to predict)
y = housing["MedHouseVal"]
# 6. Split the dataset into training and testing sets using train_test_split()
#    - Use test_size=0.2 and random_state=42 for reproducibility
x_train,x_test,y_train,y_test = train_test_split(x , y , test_size = 0.2 , random_state = 42)
# 7. Create a scikit-learn Pipeline to combine:
#    - StandardScaler (for feature scaling)
#    - LinearRegression model (for prediction)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
]) 
# 8. Fit the pipeline to the training data using .fit(X_train, y_train)
pipeline.fit(x_train,y_train)
# 9. Use the pipeline to make predictions on the test set using .predict(X_test)
predictions = pipeline.predict(x_test)
# 10. Evaluate the model using:

#     - Mean Squared Error (MSE)
mse = mean_squared_error(y_test,predictions)
#     - R² Score (to know how well the model explains the variance)
R2 = r2_score(y_test,predictions)
# 11. Print a few predicted values and their actual target values side-by-side

print("Predicted values\tActual values ")
for pred, actual in zip(predictions[:5], y_test[:5].values):
    print(f"{pred:.2f}\t\t\t{actual:.2f}")


# 12. (Optional) Predict the price of a custom house by passing a new feature array
new_house_data = np.array([[8.3252, 41, 6.9841, 1.0238, 322, 2.5556, 37.88, -122.23]])

new_predictions = pipeline.predict(new_house_data)

print(f"New house price : {new_predictions }")
