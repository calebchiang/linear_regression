import pandas as pd
import numpy as np

dataset = pd.read_csv('dataset.csv')
X = dataset['Feature_1'].values
y = dataset['Target'].values

def predict(X, w, b):
    y_hat = w * X + b

    return y_hat

def compute_cost(y_hat, y):
    m = len(y)
    total_error = sum((y_hat - y) ** 2)
    cost = total_error / (2 * m)

    return cost

def gradient_descent(X, y, w, b, learning_rate, iterations):
    m = len(y)

    for i in range(iterations):
        # Predict current y_hat using the predict function
        y_hat = predict(X, w, b)

        # Compute gradients
        dw = (1/m) * np.dot((y_hat - y), X)
        db = (1/m) * np.sum(y_hat - y)

        # Update parameters
        w -= learning_rate * dw
        b -= learning_rate * db

    return w, b

def main():
    w = 63.74423098683879
    b = -0.5873894300569021
    learning_rate = 0.01
    iterations = 1000

    # w_opt, b_opt = gradient_descent(X, y, w, b, learning_rate, iterations)

    # print("Optimized w:", w_opt)
    # print("Optimized b:", b_opt)

    y_hat = predict(X, w, b)
    initial_cost = compute_cost(y_hat, y)
    print(initial_cost)

main()