from sklearn import datasets, model_selection # Dataset 
from sklearn import metrics # Model Evaluation
import matplotlib.pyplot as plt # Data Visualization
import numpy as np # Numerical Operations
import joblib # Saving the Model

# 15% for validation and test, 80% for training in total

# Load and split the california housing dataset into features(X) and labels(y)
X, y = datasets.fetch_california_housing(return_X_y=True)

# Split the dataset into training and testing sets (with a 80-20 split)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3)

# Split testng set into validation and final test set (with a 50-50 split)
X_validation, X_test, y_validation, y_test = model_selection.train_test_split(X_test, y_test, test_size=0.5)

# Print the shapes of the training data arrays (features and labels)
print(X_train.shape, y_train.shape)



class LinearRegression:
    def __init__(self,n_features: int): # Initialise parameters
        
        np.random.seed(10)

        # (n_features, 1) needs to be compatible with the input features
        self.W = np.random.randn(n_features, 1) # Randomly initialise weight
        self.b = np.random.randn(1) # Randomly initialise bais

    # Defines how to make predictions
    def __call__(self, X): # X = Input feature matrix of shape (num_samples, n_features)

        ypred = np.dot(X, self.W) + self.b # Computes predicted output 

        return ypred # Returns prediction
    
    def update_params(self, W, b) -> None:

        self.W = W # set the wights to the new weight value
        self.b = b # set the bias to the new bias value

        return None
    


## VISUALIZATION ##
def plot_predictions(y_pred, y_true):

    samples = len(y_pred)

    plt.figure()
    plt.scatter(np.arange(samples), y_pred, c="r", label="predictions")
    plt.scatter(np.arange(samples), y_true, c="b", label="true labels", marker="x")
    plt.legend()
    plt.xlabel("Sample numbers")
    plt.ylabel("Values")
    plt.show()

model = LinearRegression(n_features=8) # Instantiate the linear model
y_pred = model(X_test) # Make predictions with data
print("Predictions:\n", y_pred[:10]) # Print first 10 predictions

plot_predictions(y_pred[:10], y_test[:10])



## LOSS ##
def mean_squared_error(y_pred, y_true):  # Define our criterion (loss function)
    
    errors = y_pred - y_true  # Calculate errors
    squared_errors = errors ** 2  # Square errors

    return np.mean(squared_errors)

cost = mean_squared_error(y_pred, y_train)
print(cost)

def minimize_loss(X_train, y_train):

    X_with_bias = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    optimal_w =  np.matmul(
        np.linalg.inv(np.matmul(X_with_bias.T, X_with_bias)),
        np.matmul(X_with_bias.T, y_train),
    )

    return optimal_w[1:], optimal_w[0]

weights, bias = minimize_loss(X_train, y_train)
print(weights, bias)



## UPDATE MODEL ##
model.update_params(weights, bias)
y_pred = model(X_train)
cost = mean_squared_error(y_pred, y_train)
print(cost)



## MODEL EVALUATION ##
metrics.mean_squared_error(y, y_pred)



## MODEL PERSISTANCE ##
joblib.dump(model, "model.joblib")