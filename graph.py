import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('Salary_dataset.csv')


plt.scatter(df['YearsExperience'], df['Salary'], label='Data points')

w = 9449.418339718359
b = 24851.91039472122

x_values = np.linspace(df['YearsExperience'].min(), df['YearsExperience'].max(), 100)
y_predicted = w * x_values + b

plt.plot(x_values, y_predicted, color='red', label='Regression Line')

plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Salary vs Years of Experience')
plt.show()
