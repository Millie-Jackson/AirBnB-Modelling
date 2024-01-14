
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import SGDRegressor, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix, roc_curve, precision_recall_curve, auc
from typing import Any, Dict, List, Tuple
from itertools import cycle

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
        labels = df[[label]]

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
    
    def visualize_predictions(self, model, y_pred, y_true, rmse, r2, X_test, y_test, classes=None) -> None:
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

        # CONFUSION MATRIX
        if classes is None:
            classes = np.unique(y_true)
        
        # Plot confusion matrix
        self.plot_confusion_matric(y_true, y_pred, classes)

        # ROC CURVE
        self.visualize_roc_curve(model, X_test, y_test)

        # PRECISION-RECALL CURVE
        self.visualize_precision_recall_curve(model, X_test, y_test)

        plt.show()

        return None
    
    def plot_confusion_matrix(self, y_true, y_pred, classes) -> None:
        """Plot the confusion matrix"""

        cm = confusion_matrix(y_true, y_pred, labels=classes)
        plt.figure(figsize=(8,6))
        im = plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xlabel("Predicted")
        plt.ylabel("True")

        # Add annotations
        for i in range(len(classes)):
            for j in range(len(classes)):
                plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='w')

    def visualize_roc_curve(self, model, X_test, y_test) -> None:
        """Visualize the ROC curve"""

        # Predict probabilities
        y_pred_prob = model.predict_proba(X_test)[:, 1]

        # Compute ROC curve and ROC area
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)

        # Plot predictions
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()

        return None

    def visualize_precision_recall_curve(self, model, X_test, y_test) -> None:
        """Visualize the precision-recall curve"""

        # Binarize the output
        y_bin = label_binarize(y_test, classes=np.unique(y_test))

        # Compute for each class
        precision = dict()
        recall = dict()
        average_precisions = dict()
        n_classes = y_bin.shape[1]

        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_bin[:, i], model.predict_proba(X_test)[:, i])
            average_precisions[i] = auc(recall[i], precision[i])

        # Plot for each class
        plt.figure(figsize=(8, 6))
        colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(recall[i], precision[i], color=color, lw=2, label='Precision-recall cirve for class {0} (area = {1:0.2f}'.format(i, average_precisions[i]))
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower right")

        plt.show()

        return None
    
    def tune_model_hyperparameters(self, model_class, X, y) -> Tuple[Any, Dict[str, Any]]:
        """Hyperparameter tuning using GridSearchCV."""

        hyperparameters = self.get_hyperparameters(model_class)
        estimator = model_class(**self.get_estimator_params(model_class))

        grid_search = GridSearchCV(estimator, hyperparameters, scoring='neg_mean_squared_error')
        grid_search.fit(X, y)

        best_model = grid_search.best_estimator_
        best_hyperparameters = grid_search.best_params_

        return best_model, best_hyperparameters
    
    def save_model(self, model, hyperparameters, performance_metrics, folder: str) -> None:
        """Save the trained model, hyperparameters, and performance metrics."""

        # Create directory if it doesnt exist
        os.makedirs(folder, exist_ok=True)

        # Save the trained model
        model_filename = os.path.join(folder, 'model.joblib')
        joblib.dump(model, model_filename)

        # Save the hyperparameters
        hyperparameters = self.format_hyperparameters(hyperparameters)
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
    
    def evaluate_all_models(self, X_train, y_train, X_validation, y_validation, model_classes, label_col) -> None:
        """Evaluate different models and save the best models."""

        for model_class in model_classes:
            # Tune hyperparameters
            best_model, best_hyperparameters = self.tune_model_hyperparameters(model_class, model_folder, label_col)

            # Train model
            trained_model = self.train_model(best_model, X_train, y_train)

            # Make predictions on validation set
            y_validation_pred = trained_model.predict(X_validation)

            # Evaluate validation set
            rmse_validation, r2_validation, _, _ = self.predict_and_evaluate(trained_model, X_train, y_train, X_validation, y_validation)
            print(f"Validation Mean Squared Error: {rmse_validation: .2f}")
            print(f"Validation R-squared: {r2_validation: .2f}")

            # Save the model, hyperparameters and metrics
            model_name, = model_class.__name__.lower()
            model_folder = os. path.join(self.task_folder, model_name)
            self.save_model(best_model, best_hyperparameters, X_validation, y_validation, model_class, model_folder, label_col)

            # Visualize predictions
            self.visualize_predictions(y_validation_pred, y_validation, rmse_validation, r2_validation)

        return None
    
    def find_best_model(self, folders: List[str]) -> Tuple[Any, Dict[str, Any]]:
        """Find the best model among the trained models."""

        best_model = None
        best_hyperparameters = None
        best_performance_metrics = float('inf')
        best_rmse = float('inf')

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
    
    def get_hyperparameters(self, model_class) -> Dict[str, Any]:
        """Return hyperparameters for the given model class."""

        if model_class == SGDRegressor:  
            return {"alpha": [0.0001, 0.001, 0.01, 0.1], "learning_rate": ["constant", "optimal", "invscaling"]}
        elif model_class == DecisionTreeRegressor:
            return {"max_depth": [None, 10, 20, 30], "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2, 4]}
        elif model_class == RandomForestRegressor:
            return {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20, 30], "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2, 4]}
        elif model_class == GradientBoostingRegressor:
            return {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2], "max_depth": [3, 4, 5], "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2, 4]}
        elif model_class == LogisticRegression:
            return {"C": [0.001, 0.01, 1, 10, 100], "penalty": ["l1", "l2"]}
        elif model_class == RandomForestClassifier:
            return {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20, 30], "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2, 4]}
        elif model_class == GradientBoostingClassifier:
            return {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2], "max_depth": [3, 4, 5], "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2, 4]}
        else:
            raise ValueError("Unsupported model class")
        
        return None
    
    def get_estimator_params(self, model_class) -> Dict[str, Any]:
        """Return additional parameters for the given estimator class."""

        if model_class == SGDRegressor:
            return {"max_iter": 1000, "random_state": 42}
        else:
            return {}
        
        return None
    
class RegressionModelling(Modelling):

    def __init__(self, task_folder: str):
        super().__init__(task_folder)
    
class ClassificationModelling(Modelling):
    
    def __init__(self, task_folder: str):
        super().__init__(task_folder)
    
    def evaluate_all_models(self, X_train, y_train, X_validation, y_validation, model_classes) -> None:
        """Evaluate different models, visualize and save the best models."""

        for model_class in model_classes:
            # Tune hyperparameters
            best_model, best_hyperparameters = self.tune_model_hyperparameters(model_class, X_train, y_train)

            # Train model
            trained_model = self.train_model(best_model, X_train, y_train)

            # Make predictions on validation set
            y_validation_pred = trained_model.predict(X_validation)

            # Evaluate validation set
            accuracy = accuracy_score(y_validation, y_validation_pred)
            print(f"Validation Accuracy: {accuracy: .2%}")
            report = classification_report(y_validation, y_validation_pred)
            print("Classification Report:\n", report)
            cm = confusion_matrix(y_validation, y_validation_pred)
            print("Confusion Matrix:\n", cm)


            # Save the model, hyperparameters and metrics
            model_name, = model_class.__name__.lower()
            model_folder = os. path.join(self.task_folder, model_name)
            self.save_model(best_model, best_hyperparameters, X_validation, y_validation, model_class, model_folder, label_col)

            # Visualize predictions
            self.visualize_predictions(y_validation_pred, y_validation, None, None)

        return None
    


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
            Have a function that decides which visualisation would be best
'''