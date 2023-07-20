# RESOURCES
# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html

from sklearn import datasets
import pandas as pd

# Load dataset
X, y = datasets.fetch_california_housing(return_X_y=True, as_frame=True)

# Show Data Frame
print(X)
# Show first 10
print(X.head())
# Show median house price
print(y[:5])

print(X.dtypes)
print(X.shape)

df = pd.DataFrame(X)
print(df)

# Mean
print(df.mean())
# Variance
print(df.var())
# Other Descriptive Stats
print(df.describe())