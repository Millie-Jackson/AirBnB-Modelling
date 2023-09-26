from tabular_data import load_airbnb # Data Processing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt # Data visualisation

# Load data with price per night as the label
features, labels = load_airbnb(label="Price_Night")

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Split the test set into validation and final set (50-50)
X_validation, X_test, y_validation, y_test = train_test_split(X_test, y_test, test_size=0.5)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a linear regression model with gradient descent
model = SGDRegressor(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (mse): {mse:.2f}")
print(f"R-squared: {r2:.2f}")

samples = len(y_pred)

plt.figure()
plt.scatter(np.arange(samples), y_pred, c="r", label="Predictions")
plt.scatter(np.arange(samples), y_test, c="b", label="Test Labels", marker="x")
plt.text(0.1, 0.9, f"MSE: {mse:.2f}", transform=plt.gca().transAxes)
plt.text(0.1, 0.85, f"R2: {r2:.2f}", transform=plt.gca().transAxes)
plt.xlabel("Sample Numbers")
plt.ylabel("Values")
plt.legend()
plt.show()


## READ ME ##
# I set the test size to 20% as my dataset is small (less than 900)
# I used random_state to It's also helpful when you want to compare different models or algorithms. By using the same random_state, you can be confident that any differences in performance are due to the model and not random variations. 42 is convention