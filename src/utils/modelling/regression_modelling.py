# utils/regression_modelling.py

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

#from tabular_data import load_airbnb
from Modelling import RegressionModelling



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



def main():

    task_folder = "/home/millie/Documents/GitHub/AirBnB/models/regression"

    # Create instance
    regression_model = RegressionModelling(task_folder)

    # Load data
    features, labels = regression_model.load_data(label="Price_Night")

    # Split data
    X_train, X_validation, y_train, y_validation = regression_model.split_data(features, labels)

    # Standardize features
    X_train, X_validation = regression_model.standardize_features(X_train, X_validation)

    # Define regression model classes
    model_classes = [SGDRegressor, RandomForestRegressor, GradientBoostingRegressor]

    # Evaluate regression models
    metrics_comparison = regression_model.evaluate_all_models(X_train, y_train, X_validation, y_validation, model_classes)

    # Print and save the metrics comparison results
    for metric_result in metrics_comparison:
        print(metric_result)

    # Find the best model
    model_folders = [
        os.path.join(task_folder, 'models/regression/sgdregressor'),
        os.path.join(task_folder, 'models/regression/randomforestregressor'),
        os.path.join(task_folder, 'models/regression/gradientboostingregressor')
    ]

    best_model, best_hyperparameters = regression_model.find_best_model(model_folders)
    print(f"Best Model: {best_model}")



if __name__ == "__main__":
    main()



## READ ME ##
# I set the test size to 30% as my dataset is small (less than 900)
# I used random_state to It's also helpful when you want to compare different models or algorithms. By using the same random_state, you can be confident that any differences in performance are due to the model and not random variations. 42 is convention



# END OF FILE