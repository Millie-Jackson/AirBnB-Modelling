# classification_modelling.py

from Modelling import ClassificationModelling
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression



def main():
    task_folder = "/home/millie/Documents/GitHub/AirBnB/models/classification"

    # Create instance of ClassificationModelling
    classification_model = ClassificationModelling(task_folder)

    # Load data
    features, labels = classification_model.load_data(label="Catagory")

    # Split data
    X_train, X_validation, y_train, y_validation = classification_model.split_data(features, labels)

    # Standadize features
    X_train, X_validation = classification_model.standardize_features(X_train, X_validation)

    # Define classification model classes
    model_classes = [LogisticRegression, RandomForestClassifier, GradientBoostingClassifier]

    # Evaluate classification models
    metrics_comparison = classification_model.evaluate_all_models(X_train, y_train, X_validation, y_validation, model_classes)

    # Print and save the metrics comparison results
    for metric_result in metrics_comparison:
        print(metric_result)

    # Find the best model
    model_folders = ['models/classification/logisticregression', 'models/classification/randomforestclassifier', 'models/classification/gradientboostingclassifier']
    best_model, best_hyperparameters = classification_model.find_best_model(model_folders)
    print(f"Best Model: {best_model}")



if __name__=="__main__":
    main()

# END OF FILE