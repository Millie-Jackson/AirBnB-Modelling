from sklearn.linear_model import LinearRegression # model creation
from sklearn.metrics import mean_squared_error # model evaluation
from sklearn import datasets, model_selection # loading and splitting datasets
import joblib # saving the model

# Load the dataset
X, y = datasets.fetch_california_housing(return_X_y=True)

# Split into training and testing sets (80-20) and set a random seed for reproducibility
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)

# Create model
model = LinearRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate model performance using the MSE
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error", mse)

# Save model to disk
joblib.dump(model, "linear_regression_model.joblib")