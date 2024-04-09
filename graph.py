import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('dataset.csv')
w = 63.74423098683879
b = -0.5873894300569021

plt.scatter(df['Feature_1'], df['Target'], label='Data points')

X_line = np.linspace(df['Feature_1'].min(), df['Feature_1'].max(), 100)
Y_line = w * X_line + b


plt.plot(X_line, Y_line, color='red', label='Linear regression line')

plt.xlabel('Feature_1 (X)')
plt.ylabel('Target (y)')
plt.title('Plot of Dataset with Linear Regression Line')
plt.legend()
plt.show()