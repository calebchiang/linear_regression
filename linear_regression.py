import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Salary_dataset.csv')
X_train = dataset['YearsExperience'].values
y_train = dataset['Salary'].values

def predict(x, w, b):
    return x * w + b

def compute_cost(x, y, w, b):
    m = x.shape[0]
    f_wb = predict(x, w, b)
    cost = np.sum((f_wb - y) ** 2) / (2 * m)
    return cost

def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    f_wb = predict(x, w, b)

    for i in range(m):
        dj_dw = dj_dw + (f_wb[i] - y[i]) * x[i]
        dj_db = dj_db + (f_wb[i] - y[i])
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db

def gradient_descent(x, y, w_init, b_init, alpha, iterations):
    w = w_init
    b = b_init
    costs = []

    for i in range(iterations):
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        cost = compute_cost(x, y, w, b)
        costs.append(cost)

        if i % 100 == 0:
            cost = compute_cost(x, y, w, b)
            print(f"Iteration {i}: Cost {cost:.4f}")

    return w, b, costs

def main():
    w = 9449.418339718359
    b = 24851.91039472122

    years_of_experience = 5
    predicted_salary = predict(years_of_experience, w, b)

    print(f"The predicted salary for someone with {years_of_experience} years of experience is: ${predicted_salary:.2f}")

main()
