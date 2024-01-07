
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, r2_score
from typing import Any, Dict, List, Tuple

class Modelling:
    
    def __init__(self, task_folder: str):
        self.task_folder = task_folder

    def load_data(self, label: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load data and return features and labels."""

        # Load cleaned data
        try:
            df = pd.read_csv("data/tabular_data/clean_tabular_data.csv")
        except FileNotFoundError:
            raise FileNotFoundError("Can't find cleaned data file")
        
        # Check if the label column is in the data
        if label not in df.columns:
            raise ValueError(f"'{label}' is not a valid label")
        
        # Filter out non_numeric columns
        features = df.select_dtypes(include=[int, float])

        # Remove label column from features
        features.drop(columns=[label], inplace=True, errors="ignore")
        labels = df[[labels]]

        return features, labels[label]
    
    def split_data(self, features, labels) -> Tuple:
        """Split data into training, validation, and test sets."""

        return train_test_split(features, labels, test_size=0.3, random_state=42)

    def standardize_features(self, *feature_sets) -> Any:

        """Standardize the features."""

        scaler = StandardScaler()
        standardize_features = [scaler.fit_transform(X) for X in feature_sets]
        
        return tuple(standardize_features)   

    def train_model(self, model, X_train, y_train) -> Any:
        """Train the model."""

        model.fit(X_train, y_train)

        return model     
    
    def predict_and_evaluate(self, model, X_train, y_train, X_test, y_test) -> Tuple[float, float, float, float]:
        """Make predictions and evaluate the model."""

        # Make predictions on the training and test set
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Evaluate the test set
        rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
        r2_train = r2_score(y_train, y_train_pred)
        print(f"Mean Squared Error (mse): {rmse_train:.2f}")
        print(f"R-squared: {r2_train:.2f}")

        # Evaluate the test set
        rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
        r2_test = r2_score(y_test, y_test_pred)
        print(f"Mean Squared Error (mse): {rmse_test:.2f}")
        print(f"R-squared: {r2_test:.2f}")

        return rmse_train, r2_train, rmse_test, r2_test
    
    def visualize_predictions(self, y_pred, y_true, rmse, r2) -> None:
        """Visualize predictions."""

        samples = len(y_pred)

        plt.figure()
        plt.scatter(np.arange(samples), y_pred, c="r", label="Predictions")
        plt.scatter(np.arange(samples), y_true, c="b", label="True Values", marker="x")
        plt.text(0.1, 0.9, f"RMSE: {rmse:.2f}", transform=plt.gca().transAxes)
        plt.text(0.1, 0.85, f"R2: {r2:.2f}", transform=plt.gca().transAxes)
        plt.xlabel("Sample Numbers")
        plt.ylabel("Values")
        plt.legend()
        plt.show()

        return None
    
    def tune_model_hyperparameters(self, model_class, X, y) -> Tuple[Any, Dict[str, Any]]:
        """Hyperparameter tuning using GridSearchCV."""

        hyperparameters = get_hyperparameters(model_class)
        estimator = model_class(**get_estimator_params(model_class))

        grid_search = GridSearchCV(estimator, hyperparameters, scoring='neg_mean_squared_error')
        grid_search.fit(X, y)

        best_model = grid_search.best_estimator_
        best_hyperparameters = grid_search.best_params_

        return best_model, best_hyperparameters
    
    def save_model(self, y_validation, model_classes) -> None:
        """Save the trained model, hyperparameters, and performance metrics."""

        # Create directory if it doesnt exist
        os.makedirs(folder, exist_ok=True)

        # Save the trained model
        model_filename = os.path.join(folder, 'model.joblib')
        joblib.dump(model, model_filename)

        # Save the hyperparameters
        hyperparameters = format_hyperparameters(hyperparameters)
        hyperparameters_filename = os.path.join(folder, 'hyperparameters.json')
        with open(hyperparameters_filename, 'w') as json_file:
            json.dump(hyperparameters, json_file, indent=4)

        performance_metrics['best_validation_RMSE'] = performance_metrics['best_validation_RMSE'].tolist()
        performance_metrics['best_params'] = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in performance_metrics['best_params'].items()}
        performance_metrics['cv_results'] = {key: value.tolist() if isinstance (value, np.ndarray) else value for key, value in performance_metrics['cv_results'].items()}
        
        # Convert NumPy array to list (because it isnt serializable to json)
        if performance_metrics:
            for key, value in performance_metrics.items():
                if isinstance(value, np.ndarray):
                    performance_metrics[key] = value.tolist()

        # Save the performance metrics
        metrics_filename = os.path.join(folder, 'metrics.json')
        if performance_metrics and 'beast_validation_RMSE' in performance_metrics:
            performance_metrics['best_validation_RMSE'] = performance_metrics['best_validation_RMSE'].tolist()
            with open(metrics_filename, 'w') as json_file:
                json.dump(performance_metrics, json_file, indent=4)
        else:
            print("Error saving metrics: 'best_validation_RMSE' not found in performance metrics")

        print(f"Model, hyperparameter and metrics saved to {folder}")

        return None
    
    def evaluate_all_models(self, X_train, y_train, X_validation, model_classes, label_col) -> None:
        """Evaluate different models and save the best models."""

        for model_class in model_classes:
            # Tune hyperparameters
            best_model, best_hyperparameters = self.tune_model_hyperparameters(model_class, model_folder, label_col)

            # Save the model, hyperparameters and metrics
            model_name, = model_class.__name__.lower()
            model_folder = os. path.join(self.task_folder, model_name)
            self.save_model(best_model, best_hyperparameters, X_validation, y_validation, model_class, model_folder, label_col)

            return None
    
    def find_best_model(self, folders: List[str]) -> Tuple[Any, Dict[str, Any]]:
        """Find the best model among the trained models."""

        best_model = None
        best_hyperparameters = None
        best_performance_metrics = float('inf')

        for folder in folders:
            # Load hyperparameters and performance metrics
            hyperparameters_file = os.path.join(folder, 'hyperparameters.json')

            with open(hyperparameters_file, 'r') as json_file:
                hyperparameters = json.load(json_file)

            # Check if this model has a lower validation RMSE
            if hyperparameters['best_validation_RMSE'] < best_rmse:
                best_model = joblib.load(os.path.join(folder, 'model.joblib'))
                best_hyperparameters = hyperparameters
                best_rmse = hyperparameters["best_validation_RMSE"]

        return best_model, best_hyperparameters
    
class RegressionModelling(Modelling):

    def __init__(self, task_folder: str):
        super().__init__(task_folder)
    
    def evaluate_all_models(self, X_train, y_train, X_validation, model_classes) -> None:
        """Evaluate different regression models and save the best models."""

        for model_class in model_classes:
            # Tune hyperparameters
            best_model, best_hyperparameters = tune_model_hyperparameters(model_class, X_train, y_train)

            # Save the model, hyperparameters, and metrics
            model_name = model_class.__name__.lower()
            folder = os.path.join("models/regression", model_name)
            save_model(best_model, best_hyperparameters, folder=folder)

        return None
    
class ClassificationModelling(Modelling):
    
    def __init__(self, task_folder: str):
        super().__init__(task_folder)
    
    def evaluate_all_models(self, X_train, y_train, X_validation, y_validation, model_classes) -> None:
        """Evaluate different classification models and save the best models."""

        metrics_comparison = []

            # Tune hyperparameters
            best_model, best_hyperparameters, best_performance_metrics = tune_model_hyperparameters(model_class, X_train, y_train, X_validation, y_validation, hyperparameters[model_name], folder= None)

            # Save the model, hyperparameters, and metrics
            folder = os.path.join(task_folder, model_name)
            save_model(best_model, best_hyperparameters, best_performance_metrics, folder=folder)

            # Compare metrics
            metrics_comparison.append({
                'model': model_name,
                'validation_accuracy': best_performance_metrics['validation_accuracy'],
                'best_params': best_hyperparameters,
                'cv_results': best_performance_metrics['cv_results']
            })

        return metrics_comparison