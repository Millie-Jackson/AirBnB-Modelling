from sklearn import datasets
from sklearn.model_selection import train_test_split

# Load the dataset
X, y = datasets.fetch_california_housing(return_X_y=True)

print(f"Number of samples in dataset: {len(X)}")

# Split the dataset 30-70
X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Split the test set 30-70
X_test, X_validation, y_test, y_validation = train_test_split(x_test, y_test, test_size=0.3)

print("Number of samples in: ")
print(f"    Training: {len(y_train)}")
print(f"    Validation: {len(y_validation)}")
print(f"    Testing: {len(y_test)}")