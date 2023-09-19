from sklearn import datasets, model_selection
from aicore.ml import data
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Use 'data.split' to split the data into training, validation and test sets
(X_train, y_train), (X_validation, y_validation), (X_test, y_test) = data.split(datasets.load_boston(return_X_y=True))
X_train, X_validation, X_test = data.standardize_multiple(X_train, X_validation, X_test)



## Previously Used Linear Regression Model ##
class LinearRegression:

    def __init__(self, optimiser, n_features): # initalise parameters 
        self.w = np.random.randn(n_features) ## randomly initialise the weight
        self.b = np.random.randn() ## randomly initialise the bias
        self.optimiser = optimiser
        
    def predict(self, X): # how do we calculate the output from an input in our model?
        ypred = X @ self.w + self.b ## make a prediction using a linear hypothesis
        return ypred # return prediction

    def fit(self, X, y):
        all_costs = [] ## initialise an empty list of costs to plot later
        for epoch in range(self.optimiser.epochs): ## for this many complete the runs through the dataset    

            # MAKE PREDICTIONS AND UPDATE MODEL
            predictions = self.predict(X) ## make predictions
            new_w, new_b = self.optimiser.step(self.w, self.b, X, predictions, y) ## calculate updated params
            self._update_params(new_w, new_b) ## update the model weight and bias
            
            # CALCULATE THE LOSS FOR VISUALISATION
            cost = mse_loss(predictions, y) ## compute the loss 
            all_costs.append(cost) ## add the cost for this batch of examples to the list of costs (for plotting)

        plot_loss(all_costs)
        print('Final cost:', cost)
        print('Weight values:', self.w)
        print('Bias values:', self.b)

    
    def _update_params(self, new_w, new_b):
        self.w = new_w ## set this instance's weights to the new weight value passed to the function
        self.b = new_b ## do the same for the bias

class SGDOptimiser:
    def __init__(self, lr, epochs):
        self.lr = lr
        self.epochs = epochs
    
    def _calc_deriv(self, features, predictions, labels):

        m = len(labels) # m = number of examples
        diffs = predictions - labels # calculate the errors
        dLdw = 2 * np.sum(features.T * diffs).T / m # calculate the loss derivative ith respect to weights
        dLdb = 2 * np.sum(diffs) / m # # calculate the derivative with respect to the bias

        return dLdw, dLdb # return the rate of change in the loss with respect w and b

    def step(self, w, b, features, predictions, labels):
        
        dLdw, dLdb = self._calc_deriv(features, predictions, labels)
        new_w = w - self.lr * dLdw
        new_b = b - self.lr * dLdb

        return new_w, new_b

def mse_loss(y_hat, labels): # define the criterion (loss function)
    
    errors = y_hat - labels # calculate the errors
    squared_errors = errors ** 2 # square the errors
    mean_squared_error = sum(squared_errors) / len(squared_errors) # calculate the mean

    return mean_squared_error # return the loss

def plot_loss(losses):
    
    # Helper function for plotting the loss against the epoch 
    plt.figure() # Make figure
    plt.ylabel("Cost")
    plt.xlabel("Epoch")
    plt.plot(losses) # Plot cost
    plt.show()

num_epochs = 1000
learning_rate = 0.001

optimiser = SGDOptimiser(lr=learning_rate, epoch=num_epochs)
model = LinearRegression(optimiser=optimiser, n_features=X_train.shape[1])
model.fit(X_train, y_train)