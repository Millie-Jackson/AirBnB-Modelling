from sklearn import datasets, model_selection
import numpy as np

X, y = datasets.fetch_california_housing(return_X_y=True)
print(X.shape)
print(y.shape)

# Split the dataset into training and testing sets (with a 80-20 split)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3)

# Split testng set into validation and final test set (with a 50-50 split)
X_validation, X_test, y_validation, y_test = model_selection.train_test_split(X_test, y_test, test_size=0.5)



class LinearRegression:
    def __init__(self,n_features: int): # Initialise parameters
        
        np.random.seed(10)

        self.W = np.random.randn(n_features, 1) # Randomly initialise weight
        self.b = np.random.randn(1) # Randomly initialise bais

    def __call__(self, X):

        ypred = np.dot(X, self.W) + self.b

        return ypred
    
    def mean_squared_error(y_pred, y_true): # Define our criterion (loss function)

        errors = y_pred - y_true # Calculate errors
        squared_errors = errors ** 2 # Square errors

        return np.mean(squared_errors)
    
    def minimize_loss(X_train, y_train):

        X_with_bias = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
        optimal_w =  np.matmul(np.linalg.inv(np.matmul(X_with_bias.T, X_with_bias)),
                               np.matmul(X_with_bias.T, y_train),)

        return optimal_w[1:], optimal_w[0]

    def update_params(self, W, b) -> None:

        self.W = W # set the wights to the new weight value
        self.b = b # set the bias to the new bias value

        return None
    


model = LinearRegression(n_features=8) # Instantiate the linear model
y_pred = model(X_test) # Make predictions with data

print("Examples:\n", X[:10]) # Print first 10 examples
print("Predictions:\n", y_pred[:10]) # Print first 10 predictions

## The predictions are in a significantly different scale compared to the actual values. 
# The predictions appear to be much larger in magnitude. 
# The actual values (y) are in the range of low positive values (e.g., between 2 and 5), 
# while the predictions are in the range of larger values (e.g., thousands). 
# This suggests that there might be an issue with the model's scaling 
# or the target values (y) might have been rescaled differently during preprocessing. 
# There is a substantial difference between the predictions and actual values. 
# You might want to investigate whether there are any scaling or preprocessing issues in your data pipeline 
# and whether the model's architecture or hyperparameters need adjustment to improve its predictive accuracy. ##



cost = LinearRegression.mean_squared_error(y_pred, y_train)
print("Cost:", cost)

weights, bias = LinearRegression.minimize_loss(X_train, y_train)
print("Weights and Bias:", weights, bias)

model.update_params(weights, bias)
y_pred = model(X_train)
cost = LinearRegression.mean_squared_error(y_pred, y_train)
print("Updated Cost:", cost)



print("Examples:\n", X[:10]) # Print first 10 examples
print("Predictions:\n", y_pred[:10]) # Print first 10 predictions