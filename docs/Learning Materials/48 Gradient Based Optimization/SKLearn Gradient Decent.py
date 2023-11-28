from sklearn.linear_model import LinearRegression

# Instantiate the linear regression model
linear_regression_model = LinearRegression()

def calculate_loss(model, X, y):

    return mse_loss(model.predict(X), y)

    print(f"Training loss before fit: {calculate_loss(model, X_train, y_train)}")
    print(f"Validation loss before fit: {calculate_loss(model, X_validation, y_validation)}")
    print(f"Test loss before fit: {calculate_loss(model, X_validation, y_validation)}")

# Fit model
model = linear_regression_model.fit(X_train, y_train)
epochs = 10000
model.fit(X_train, y_train)

print(f"Training loss after fit: {calculate_loss(model, X_train, y_train)}")
print(f"Validation loss after fit: {calculate_loss(model, X_validation, y_validation)}")
print(f"Test loss after fit: {calculate_loss(model, X_validation, y_validation)}")

print("Final weights: ", model.coef_)
print("Final bias: ", model.intercept_)