# RESOURCE: https://www.youtube.com/watch?v=Souzjv6WfrY

# Gradient Decent for Linear Regression
# yhat = wx + b
# loss = (y_yhat)**2 / N

import numpy as np

# Initialise parameters
x = np.random.rand(10, 1) # features
y = 2*x + np.random.rand() # labels
w = 0.0 # weights
b = 0.0 # bias

# Initialise hyperparameters
learning_rate = 0.01 


# Create gradient descent function
def descend(x, y, w, b, learning_rate):
    dldw = 0.0
    dldb = 0.0
    N = x.shape[0]

    # loss = (y-(wx+b))**2
    for xi, yi in zip(x, y):
        dldw += -2*xi*(yi-(w*xi+b))
        dldb += -2*(yi-(w*xi+b))
    
    # Update parameters
    w = w - learning_rate*(1/N)*dldw
    b = b - learning_rate*(1/N)*dldb

    return w, b

# Itteratively update
for epoch in range(400):

    # Gradient descent
    w, b = descend(x, y, w, b, learning_rate)
    yhat = w*x + b
    loss = np.divide(np.sum((y-yhat)**2, axis=0), x.shape[0])

    print(f'{epoch} loss is {loss}, parameters w:{w}, b:{b}')