import os #  Model Saving
import json # Model Saving
import joblib # Model Saving
import numpy as np # Model Saving
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from tabular_data import load_airbnb
from modelling import save_model



# Define hyperparameters
dt_hyperparameters = {'max_depth': [None, 10, 20, 30]}
rf_hyperparameters = {'n_estimators': [50, 100, 150], 'max_depth': [None, 10, 20, 30]}
gb_hyperparameters = {'n_estimators': [50, 100, 150], 'learning_rate': [0.01, 0.1, 0.2]}
lr_hyperparameters = {'penalty': ['l2', 'l1'], 'C': [0.001, 0.01, 0.1, 1, 10, 100]}

# Train a logistic regression model
model = LogisticRegression(random_state=42)



def tune_classification_model_hyperparameters(model_class, X_train, y_train, X_validation, y_validation, hyperparameters, folder):

    """
    Perform hyperparameter tuning using GridSearchCV for a classification model.

    Parameters:
    - model_class: The classification model class.
    - X_train, y_train: Training data.
    - X_validation, y_validation: Validation data.
    - hyperparameters: A dictionary of hyperparameter names mapping to a list of values to be tried.

    Returns:
    - best_model: The best classification model.
    - best_hyperparameters: A dictionary of the best hyperparameter values.
    - performance_metrics: A dictionary of performance metrics.
    """

    # Create an instance of the model
    estimator = model_class()

    # Define hyperparameters
    grid_search = GridSearchCV(estimator, hyperparameters, scoring='accuracy', cv=5)
    grid_search.fit(X_train, y_train)

    # Get the best model and hyperparametes
    best_model = grid_search.best_estimator_
    best_hyperparameters = grid_search.best_params_

    # Calculate accuracy on the validation set
    y_validation_pred = best_model.predict(X_validation)
    validation_accuracy = accuracy_score(y_validation, y_validation_pred)

    # Performance metrics
    performance_metrics = {
        'validation_accuracy': validation_accuracy, # Accuracy of the model on the validation set
        'best_params': best_hyperparameters, # Hyperparameters resulting in the best performance durin grid search
        'cv_results': grid_search.cv_results_, # A variety of metrics gathered during cross-validation
    }

    # Convert NumPy array to list (because it isnt serializable to json)
    for key, value in performance_metrics.items():
        if isinstance(value, np.ndarray):
            performance_metrics[key] = value.tolist()


    # Error handeling to find which NumPy array wont save
    try:
        # Save performance metrics
        metrics_file_path = os.path.join(folder, 'metrics.json')
        # Error handeling to find which folder doesnt exist
        try:
            with open(metrics_file_path, 'w') as json_file:
                    if 'best_validation_RMSE' in performance_metrics:
                        performance_metrics['best_validation_RMSE'] = performance_metrics['best_validation_RMSE'].tolist()
                    json.dump(performance_metrics, json_file)
        except Exception as e:
            print(f"Problematic key: 'best_validation_RMSE'")
            print(f"Problematic value: {performance_metrics.get('best_validation_RMSE')}")
    except Exception as e:
        print(f"Error saving metrics: {e}")
        print(f"Problematic key: {key}")
        print(f"Problematic value: {value}")

    return best_model, best_hyperparameters, performance_metrics

def find_best_model(task_folder, metric='validation_accuracy') -> tuple:
    """Find the best model among the trained models in the specified task folder."""

    best_model = None
    best_hyperparameters = None
    best_metric_value = 0.0
    best_performance_metrics = None

    model_folders = [os.path.join(task_folder, model_name) for model_name in os.listdir(task_folder) if os.path.isdir(os.path.join(task_folder, model_name))]

    for folder in model_folders:
        # Load hyperparameters and performance metrics
        hyperparameters_file = os.path.join(folder, 'hyperparameters.json')
        metrics_file_path = os.path.join(folder, 'metrics.json')

        try:
            with open(hyperparameters_file, 'r') as json_file:
                hyperparameters = json.load(json_file)

            # Convert NumPy array to list (because it isnt serializable to json
            for key, value in hyperparameters.items():
                if isinstance(value, np.ndarray):
                    hyperparameters[key] = value.tolist()

            with open(metrics_file_path, 'r') as json_file:
                performance_metrics = json.load(json_file)

            # Convert NumPy array to list (because it isnt serializable to json)
            for key, value in performance_metrics.items():
                if isinstance(value, np.ndarray):
                    performance_metrics[key] = value.tolist()

            # Extract the metric value
            metric_value = performance_metrics.get(metric, 0.0)

            # Check if this model has a higher metric value
            if metric_value > best_metric_value:
                best_model = joblib.load(os.path.join(folder, 'model.joblib'))
                best_hyperparameters = hyperparameters
                best_metric_value = metric_value
                best_performance_metrics = performance_metrics

        except FileNotFoundError:
            print(f"Metrics file not found for {folder}")
        except json.decoder.JSONDecodeError:
            print(f"Error loading metrics from {metrics_file_path}")

    return best_model, best_hyperparameters, best_performance_metrics


if __name__=="__main__":

    # Load Airbnb data with "Catagory" as the label
    features, labels = load_airbnb(label="Category")    

    # Split the data into training, validation and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels.values.ravel(), test_size=0.3, random_state=42)
    X_validation, X_test, y_validation, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model_classes = [DecisionTreeClassifier, RandomForestClassifier, GradientBoostingClassifier, LogisticRegression]

    # Evaluate and tune the classification model
    lr_model, lr_hyperparams, lr_metrics = tune_classification_model_hyperparameters(LogisticRegression, X_train, y_train, X_validation, y_validation, lr_hyperparameters, folder = 'models/classification/logistic_regression')
    # Convert NumPy arrays to lists in metrics
    for key, value in lr_metrics.items():
        if isinstance(value, np.ndarray):
            lr_metrics[key] = value.tolist()
    save_model(lr_model, lr_hyperparams, lr_metrics, 'models/classification/logistic_regression')

    # Tune and evaluate Decision Tree
    dt_model, dt_hyperparams, dt_metrics = tune_classification_model_hyperparameters(DecisionTreeClassifier, X_train, y_train, X_test, y_test, dt_hyperparameters, folder = 'models/classification/decisiontreeclassifier')
    # Convert NumPy array to list (because it isnt serializable to json)
    for key, value in dt_metrics.items():
        if isinstance(value, np.ndarray):
            dt_metrics[key] = value.tolist()
    save_model(dt_model, dt_hyperparams, dt_metrics, folder='models/classification/decision_tree')

    # Tune and evaluate Random Forest
    rf_model, rf_hyperparams, rf_metrics = tune_classification_model_hyperparameters(RandomForestClassifier, X_train, y_train, X_test, y_test, rf_hyperparameters, folder = 'models/classification/randomforestclassifier')
    # Convert NumPy array to list (because it isnt serializable to json)
    for key, value in rf_metrics.items():
        if isinstance(value, np.ndarray):
            rf_metrics[key] = value.tolist()
    # Save the model, hyperparameters and metrics
    save_model(rf_model, rf_hyperparams, rf_metrics, folder='models/classification/random_forest')

    # Tune and evaluate Gradient Boosting
    gb_model, gb_hyperparams, gb_metrics = tune_classification_model_hyperparameters(GradientBoostingClassifier, X_train, y_train, X_test, y_test, gb_hyperparameters, folder = 'models/classification/gradient_boosting_classifier')
    # Convert NumPy array to list (because it isnt serializable to json)
    for key, value in gb_metrics.items():
        if isinstance(value, np.ndarray):
            gb_metrics[key] = value.tolist()
    save_model(gb_model, gb_hyperparams, gb_metrics, folder='models/classification/gradient_boosting')

    # Compare models
    metrics_comparison = evaluate_all_models(X_train, y_train, X_validation, y_validation, model_classes, task_folder='models/classification')

    # Evaluate all models
    evaluate_all_models(X_test, y_test, X_validation, y_validation, model_classes, task_folder='models/classification')
    best_model, best_hyperparameters, best_performance_metrics = find_best_model(task_folder='models/classification', metric='validation_accuracy')
    print("Best Classification Model: ", best_model)
    print("Best hyperparameters:", best_hyperparameters)
    print("Best Performance Metrics: ", best_performance_metrics)


'''
OPTIONAL

Class: Some of this code is also int the modeling.py maybe we should make a class and inherit from it
'''



# END OF FILE