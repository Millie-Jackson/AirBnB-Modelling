from tabular_data import load_airbnb # Data Processing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from itertools import product 
import numpy as np
import os #  Model Saving
import joblib # Model Saving
import json # Model Saving
import matplotlib.pyplot as plt # Data Visualisation

# Load data with price per night as the label
features, labels = load_airbnb(label="Price_Night")

# Split data into training, validation and test sets
X_train, X_test, y_train, y_test = train_test_split(features, labels.values.ravel(), test_size=0.3, random_state=42)
X_validation, X_final_test, y_validation, y_final_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_validation = scaler.transform(X_validation)
X_final_test = scaler.transform(X_final_test)

# Train a linear regression model with gradient descent
model = SGDRegressor(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the training and test set
y_train_pred = model.predict(X_train)
y_final_test_pred = model.predict(X_final_test)

# Evaluate the test set
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
r2_train = r2_score(y_train, y_train_pred)
print(f"Mean Squared Error (mse): {rmse_train:.2f}")
print(f"R-squared: {r2_train:.2f}")

# Evaluate the test set
rmse_final_test = np.sqrt(mean_squared_error(y_final_test, y_final_test_pred))
r2_final_test = r2_score(y_final_test, y_final_test_pred)
print(f"Mean Squared Error (mse): {rmse_final_test:.2f}")
print(f"R-squared: {r2_final_test:.2f}")

samples = len(y_final_test_pred)

plt.figure()
plt.scatter(np.arange(samples), y_final_test_pred, c="r", label="Predictions")
plt.scatter(np.arange(samples), y_final_test, c="b", label="Final Test Labels", marker="x")
plt.text(0.1, 0.9, f"RMSE: {rmse_final_test:.2f}", transform=plt.gca().transAxes)
plt.text(0.1, 0.85, f"R2: {r2_final_test:.2f}", transform=plt.gca().transAxes)
plt.xlabel("Sample Numbers")
plt.ylabel("Values")
plt.legend()
plt.show()

# Tune hyperparameters

# Tune From Scratch
'''
def custom_tune_regression_model_hyperparameters(model_class, X, y, hyperparameters):
    
    """
    Perform a grid search over a range of hyperparameter values for a regression model.

    Parameters:
    - model_class: The class of the regression model (e.g., SGDRegressor).
    - X: The feature matrix.
    - y: The target variable.
    - hyperparameters: A dictionary of hyperparameter names mapping to a list of values to be tried.

    Returns:
    - best_model: The best regression model.
    - best_hyperparameters: A dictionary of the best hyperparameter values.
    - performance_metrics: A dictionary of performance metrics.
    """

    X_train, X_temp, y_train, y_temp, = train_test_split(X, y, test_size=0.3, random_state=42)
    X_validation, X_test, y_validation, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    best_model = None
    best_hyperparameters = None
    best_rmse = float('inf')

    for hyperparam_values in product(*hyperparameters.values()):
        hyperparams = dict(zip(hyperparameters.keys(), hyperparam_values))
        model = model_class(**hyperparams)
        model.fit(X_train, y_train)

        # Calculate RMSE on the validation set
        y_validation_pred = model.predict(X_validation)
        rmse = np.sqrt(mean_squared_error(y_validation, y_validation_pred))

        if rmse < best_rmse:
            best_rmse = rmse
            best_hyperparameters = hyperparams

    y_test_pred = best_model.predict(X_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

    performance_metrics = {
        'validation_RMSE': best_rmse,
        'test_RSME': rmse_test,
    }    

    return best_model, best_hyperparameters, performance_metrics
'''

# Tune Using SKlearn
def tune_regression_model_hyperparameters(model, X, y, hyperparameters, cv=5):

    """
    Perform hyperparameter tuning using GridSearchCV.

    Parameters:
    - model: The regression model (an instance, not a class).
    - hyperparameters: The dictionary of hyperparameter values to be tried.
    - X: The feature matrix.
    - y: The target variable.
    - cv: Number of cross-validation folds.

    Returns:
    - best_model: The best regression model.
    - best_hyperparameters: A dictionary of the best hyperparameter values.
    - performance_metrics: A dictionary of performance metrics.
    """

    # Instantiate the model
    estimator = model()

    grid_search = GridSearchCV(estimator, hyperparameters, scoring='neg_mean_squared_error')
    grid_search.fit(X, y)

    best_model = grid_search.best_estimator_
    best_hyperparameters = grid_search.best_params_

    # Calculate RMSE on the validation set
    y_pred = best_model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    performance_metrics = {
        'best_validation_RMSE': rmse,
        'best_params': best_hyperparameters,
        'cv_results': grid_search.cv_results_,
    }

    return best_model, best_hyperparameters, performance_metrics

hyperparameters = {
    'alpha': [0.0001, 0.001, 0.01, 0.1],
    'learning_rate': ['constant', 'optimal', 'invscaling']
    }

best_model, best_hyperparameters, performance_metrics = tune_regression_model_hyperparameters(SGDRegressor, X_train, y_train, hyperparameters)
print(f"Best hyperparameters: {best_hyperparameters}")
print(f"Performance_metrics: {performance_metrics}")



# SAVING THE MODEL

def save_model(model, hyperparameters, performance_metrics, folder='models/regression/linear_regression'):

    """
    Save the trained model, hyperparameters, and performance metrics.

    Parameters:
    - model: The trained regression model.
    - hyperparameters: A dictionary of hyperparameter values.
    - performance_metrics: A dictionary of performance metrics.
    - folder: The folder where files should be saved.

    Returns:
    - None
    """

    # Create directory if it doesnt exist
    os.makedirs(folder, exist_ok=True)

    # Save the trained model
    model_filename = os.path.join(folder, 'model.joblib')
    joblib.dump(model, model_filename)

    # Save the hyperparameters
    hyperparameters_filename = os.path.join(folder, 'hyperparameters.json')
    with open(hyperparameters_filename, 'w') as json_file:
        json.dump(hyperparameters, json_file, indent=4)

    # Save the performance metrics
    metrics_filename = os.path.join(folder, 'metrics.json')
    with open(metrics_filename, 'w') as json_file:
        json.dump(performance_metrics, json_file, indent=4)

    print(f"Model, hyperparameter and metrics saved to {folder}")

save_model(best_model, best_hyperparameters, performance_metrics)

## READ ME ##
# I set the test size to 30% as my dataset is small (less than 900)
# I used random_state to It's also helpful when you want to compare different models or algorithms. By using the same random_state, you can be confident that any differences in performance are due to the model and not random variations. 42 is convention