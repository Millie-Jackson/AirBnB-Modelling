
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
    test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a logistic regression model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)

# Print the evaluation metrics
print(f"Accuracy: {accuracy:.2f}")
print(f"Classification Report:\n{classification_report_str}")



'''
OPTIONAL

Class: some of this code is also int the modeling.py maybe we should make a class and inherit from it
'''



# END OF FILE