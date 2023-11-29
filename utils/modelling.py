
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



def load_airbnb_data(label="Price_Night") -> tuple:
    """Load Airbnb data and return features and labels."""

    # Load cleaned data
    try:
        df = pd.read_csv("data/tabular_data/clean_tabular_data.csv")
    except FileNotFoundError:
        raise FileNotFoundError("Cant find cleaned data file")

    # Check if the label column is in the data
    if label not in df.columns:
        raise ValueError(f"'{label}' is not a features")

    # Filter out non-numeric columns
    features = df.select_dtypes(include=[int, float])

    # Remove label column from features
    features.drop(columns=[label], inplace=True, errors="ignore")
    labels = df[[label]]

    # Load data with price per night as the label
    features, labels = load_airbnb(label="Price_Night")

    return features, labels[label]

def split_data(features, labels) -> tuple:
    """Split data into training, validation, and test sets."""

    return train_test_split(features, labels, test_size=0.3, random_state=42)

def standardize_features(*feature_sets) -> tuple:
    """Standardize the features."""

    scaler = StandardScaler()
    standardize_features = [scaler.fit_transform(X) for X in feature_sets]
    
    return tuple(standardize_features)

def train_model(model, X_train, y_train) -> object:
    """Train the model."""

    model.fit(X_train, y_train)

    return model

def predict_and_evaluate(model, X_train, y_train, X_test, y_test) -> tuple:
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

def visualize_predictions(y_pred, y_true, rmse, r2) -> None:
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

def tune_regression_model_hyperparameters(model_class, X, y) -> tuple:
    """Hyperparameter tuning using GridSearchCV."""

    hyperparameters = get_hyperparameters(model_class)
    estimator = model_class(**get_estimator_params(model_class))

    grid_search = GridSearchCV(estimator, hyperparameters, scoring='neg_mean_squared_error')
    grid_search.fit(X, y)

    best_model = grid_search.best_estimator_
    best_hyperparameters = grid_search.best_params_

    #performance_metrics = {
    #'best_validation_RMSE': rmse,  # Root Mean Squared Error on the validation set
    #'best_params': best_hyperparameters,  # Best hyperparameters found
    #'cv_results': grid_search.cv_results_,}  # Cross-validated results

    return best_model, best_hyperparameters

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

def save_model(model, hyperparameters, folder='models/regression/linear_regression'):
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

    #performance_metrics['best_validation_RMSE'] = performance_metrics['best_validation_RMSE'].tolist()
    #performance_metrics['best_params'] = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in performance_metrics['best_params'].items()}
    #performance_metrics['cv_results'] = {key: value.tolist() if isinstance (value, np.ndarray) else value for key, value in performance_metrics['cv_results'].items()}

    # Save the performance metrics
    #metrics_filename = os.path.join(folder, 'metrics.json')
    #with open(metrics_filename, 'w') as json_file:
    #   json.dump(performance_metrics, json_file, indent=4)

    print(f"Model, hyperparameter and metrics saved to {folder}")

    return None

def format_hyperparameters(hyperparameters) -> dict:
    """Format hyperparameters for saving."""

    formatted_hyperparameters = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in hyperparameters.items()}

    return formatted_hyperparameters

def evaluate_all_models(X_train, y_train, X_validation, y_validation, model_classes) -> None:
    """Evaluate different regression models and save the best models."""

    for model_class in model_classes:
        # Tune hyperparameters
        best_model, best_hyperparameters = tune_regression_model_hyperparameters(model_class, X_train, y_train)

        # Save the model, hyperparameters, and metrics
        model_name = model_class.__name__.lower()
        model_folder = os.path.join("models/regression", model_name)
        save_model(best_model, best_hyperparameters, folder=model_folder)

    return None

def find_best_model(model_folders) -> tuple:
    """Find the best model among the trained models."""

    best_model = None
    best_hyperparameters = None
    best_performance_metrics = float('inf')

    for folder in model_folders:
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
    #visualize_predictions(y_final_test_pred, y_final_test, rmse_final_test, r2_final_test)
    visualize_predictions(y_final_test_pred, y_final_test, rmse_final_test, r2_final_test)
    #visualize_predictions(y_test_pred, y_test, rmse_test, r2_test)

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
Consistent Hyperparameter Tuning: You are using GridSearchCV for hyperparameter tuning for some models, but you have a custom hyperparameter tuning function commented out. It's good to be consistent. Either use GridSearchCV for all or your custom function for all.Data Exploration: Consider adding a section for exploring and visualizing your data. Understanding your data better can often lead to more informed modeling decisions.
Data Structure: Depending on the size of your dataset, you might want to split it further into train/validation/test sets, especially if you're building machine learning models.
Error Handling: Consider adding error handling, especially when dealing with file operations, to catch and handle potential exceptions.
File Paths: Be cautious when using file paths. In your save_model function, you are saving models and metrics to specific paths. Ensure these paths are correctly set according to your project structure.
Flexibility: Make your script more flexible by allowing users to specify parameters such as the test size and random seed as arguments.
Handle Edge Cases: Ensure that your script gracefully handles edge cases, such as cases where a folder already exists or when there's an issue with saving models.
Logging: Add logging statements to provide information about the progress of your script. This can be helpful for debugging and understanding where your script might be spending more time.
Model Evaluation: Consider wrapping the model evaluation code into functions. This makes the code modular and easier to understand.
Model Folders: Instead of hardcoding the model folders, consider dynamically obtaining them from the file system. This can be helpful when you have a growing number of models.
Print Statements: The print statements are useful for debugging, but in a production environment or a larger project, consider using a logging framework for more control over log levels and destinations.
Reuse Functions: Consider creating a reusable function for hyperparameter tuning, which can be used for each model. This reduces code redundancy and makes your script more maintainable.
Use a Dictionary for Models: Instead of a list, consider using a dictionary to map model names to their corresponding classes. This can make your code more readable and scalable.
Visualization: Consider creating a separate function for the visualization part. This will help if you need to reuse this code or if you decide to make changes to the visualization in the future.
            Show the best model on a graph
            Show all the models on the same graph
'''