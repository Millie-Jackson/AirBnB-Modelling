
import os #  Model Saving
import json # Model Saving
import joblib # Model Saving
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # Data Visualisation

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from itertools import product 

from tabular_data import load_airbnb



"""
modeling.py

This module contains functions for training, evaluating, and saving regression models for predicting Airbnb prices.

Functions:
    - load_airbnb_data(label: str = "Price_Night") -> Tuple[pd.DataFrame, pd.DataFrame]:
        Load cleaned Airbnb data and return numerical features and the specified column as the label.

    - standardize_features(*feature_sets: pd.DataFrame) -> List[pd.DataFrame]:
        Standardize the features using a StandardScaler.

    - train_linear_regression(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
        Train a linear regression model using the training data.

    - predict_and_evaluate(model, X_train, y_train, X_final_test, y_final_test) -> Tuple[float, float, float, float]:
        Make predictions on the training and test sets, and evaluate the model's performance.

    - visualize_predictions(predictions: np.ndarray, labels: np.ndarray, rmse: float, r2: float) -> None:
        Visualize predictions against actual labels and display evaluation metrics.

    - tune_regression_model_hyperparameters(model_class, X, y) -> Tuple[Any, Dict[str, Any], Dict[str, Union[float, Dict[str, Any]]]]:
        Perform hyperparameter tuning using GridSearchCV for the specified regression model class.

    - save_model(model: Any, hyperparameters: Dict[str, Any], performance_metrics: Dict[str, Any], folder: str) -> None:
        Save the trained model, hyperparameters, and performance metrics.

    - evaluate_all_models(X_train, y_train, X_validation, y_validation, model_classes) -> None:
        Evaluate different regression models, tune hyperparameters, and save the best models.

    - find_best_model(model_folders: List[str]) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
        Find the best model among the trained models.

Usage:
    To train models and evaluate their performance, run this module as a script.

    Example:
    ```bash
    python modeling.py
    ```

"""



def get_hyperparameters(model_class) -> dict:
    """Return hyperparameters for the given model class."""

    if model_class == SGDRegressor:
        return {"alpha": [0.0001, 0.001, 0.01, 0.1], "learning_rate": ["constant", "optimal", "invscaling"]}
    elif model_class == DecisionTreeRegressor:
        return {"max_depth": [None, 10, 20, 30], "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2, 4]}
    elif model_class == RandomForestRegressor:
        return {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20, 30], "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2, 4]}
    elif model_class == GradientBoostingRegressor:
        return {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2], "max_depth": [3, 4, 5], "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2, 4]}
    else:
        raise ValueError("Unsupported model class")

def get_estimator_params(model_class) -> dict:
    """Return additional parameters for the given estimator class."""

    if model_class == SGDRegressor:
        return {"max_iter": 1000, "random_state": 42}
    else:
        return{}

def format_hyperparameters(hyperparameters) -> dict:
    """Format hyperparameters for saving."""

    formatted_hyperparameters = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in hyperparameters.items()}

    return formatted_hyperparameters

if __name__ == "__main__":

    # Load data
    features, labels = load_airbnb_data()

    # Split data
    X_train, X_test, y_train, y_test = split_data(features, labels)
    X_validation, X_final_test, y_validation, y_final_test = split_data(X_test, y_test)

    # Standardize features
    X_train, X_validation, X_final_test = standardize_features(X_train, X_validation, X_final_test)

    # Train a linear regression model with gradient descent
    model = SGDRegressor(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions on the training and test set
    y_train_pred = model.predict(X_train)
    y_final_test_pred = model.predict(X_final_test)

    # Evaluate the training set
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    r2_train = r2_score(y_train, y_train_pred)
    print(f"Mean Squared Error (mse): {rmse_train:.2f}")
    print(f"R-squared: {r2_train:.2f}")
    
    # Evaluate the test set
    rmse_final_test = np.sqrt(mean_squared_error(y_final_test, y_final_test_pred))
    r2_final_test = r2_score(y_final_test, y_final_test_pred)
    print(f"Mean Squared Error (mse): {rmse_final_test:.2f}")
    print(f"R-squared: {r2_final_test:.2f}")

    # Evaluate the model
    rmse_train, r2_train, rmse_final_test, r2_final_test = predict_and_evaluate(model, X_train, y_train, X_final_test, y_final_test)
    #rmse_train, r2_train, rmse_test, r2_test = predict_and_evaluate(linear_model, X_train, y_train, X_test, y_test)

    # Visualize predictions
    visualize_predictions(y_final_test_pred, y_final_test, rmse_final_test, r2_final_test)


    # Evaluate all models
    model_classes = [DecisionTreeRegressor, RandomForestRegressor, GradientBoostingRegressor]
    evaluate_all_models(X_train, y_train, X_validation, y_validation, model_classes)

    # Find best model
    model_folders = ['models/regression/decisiontreeregressor', 'models/regression/randomforestregressor', 'models/regression/gradientboostingregressor']
    best_model, best_hyperparameters = find_best_model(model_folders)

    print(f"Best Model: {best_model}")



## READ ME ##
# I set the test size to 30% as my dataset is small (less than 900)
# I used random_state to It's also helpful when you want to compare different models or algorithms. By using the same random_state, you can be confident that any differences in performance are due to the model and not random variations. 42 is convention

'''
OPTIONAL
Comments: Some parts of your code already have comments explaining the purpose of the code, which is great. Make sure to add comments for more complex sections or to explain the rationale behind certain choices.
Configuration Handling: If there are constants or configurations that might change, consider handling them in a separate configuration file. This can include things like file paths, hyperparameter search spaces, or any other settings.
Consistent Hyperparameter Tuning: You are using GridSearchCV for hyperparameter tuning for some models, but you have a custom hyperparameter tuning function commented out. It's good to be consistent. Either use GridSearchCV for all or your custom function for all.Data Exploration: Consider adding a section for exploring and visualizing your data. Understanding your data better can often lead to more informed modeling decisions.
Data Analysis: Add more metrics to each model
Data Sets: Include car price dataset
        Ask Tom if you can use his pluming dataset
Data Structure: Depending on the size of your dataset, you might want to split it further into train/validation/test sets, especially if you're building machine learning models.
Error Handling: Consider adding error handling, especially when dealing with file operations, to catch and handle potential exceptions.
Experiment Tracking: If you are running multiple experiments, you might want to look into experiment tracking tools or frameworks (e.g., MLflow, TensorBoard) to log and compare your experiments easily.
Features: Find the best model for each data column
        Geographical distributions
        Pricing trends
        Impact of various features
File Paths: Be cautious when using file paths. In your save_model function, you are saving models and metrics to specific paths. Ensure these paths are correctly set according to your project structure.
Flexibility: Make your script more flexible by allowing users to specify parameters such as the test size and random seed as arguments.
Handle Edge Cases: Ensure that your script gracefully handles edge cases, such as cases where a folder already exists or when there's an issue with saving models.
Logging: Add logging statements to provide information about the progress of your script. This can be helpful for debugging and understanding where your script might be spending more time.
Model Evaluation: Consider wrapping the model evaluation code into functions. This makes the code modular and easier to understand.
Model Folders: Instead of hardcoding the model folders, consider dynamically obtaining them from the file system. This can be helpful when you have a growing number of models.
Print Statements: The print statements are useful for debugging, but in a production environment or a larger project, consider using a logging framework for more control over log levels and destinations.
Reuse Functions: Consider creating a reusable function for hyperparameter tuning, which can be used for each model. This reduces code redundancy and makes your script more maintainable.
Unit Testing: Consider writing unit tests for your classes and methods. This will help ensure that each part of your code works as expected, and it makes it easier to catch regressions when you modify the code.
Use a Dictionary for Models: Instead of a list, consider using a dictionary to map model names to their corresponding classes. This can make your code more readable and scalable.
Visualization: Consider creating a separate function for the visualization part. This will help if you need to reuse this code or if you decide to make changes to the visualization in the future.
            Show the best model on a graph
            Show all the models on the same graph
            Allow visualizations to be interactive
'''

# END OF FILE