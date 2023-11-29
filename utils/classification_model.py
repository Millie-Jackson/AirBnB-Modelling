
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from tabular_data import load_airbnb



# Load Airbnb data with "Catagory" as the label
features, labels = load_airbnb(label="Category")

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, labels.values.ravel(),
    test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a logistic regression model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Make predictions on the training set
y_train_pred = model.predict(X_train)

# Make predictions on the test set
y_test_pred = model.predict(X_test)

# Evaluate on training set
train_report = classification_report(y_train, y_train_pred)
train_accuracy = accuracy_score(y_train, y_train_pred)

# Evaluate on test set
test_report = classification_report(y_test, y_test_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Print the results
print(f"Training Set:\n{train_report}\nAccuray: {train_accuracy:.2f}\n")
print(f"Test Set:\n{test_report}\nAccuracy: {test_accuracy:.2f}")



'''
OPTIONAL

Class: Some of this code is also int the modeling.py maybe we should make a class and inherit from it
'''



# END OF FILE