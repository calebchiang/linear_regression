from sklearn.datasets import make_regression
import pandas as pd

def filter_and_save_data(csv_path):
    df = pd.read_csv(csv_path)
    if df.columns[0] != 'YearsExperience' and df.columns[0] != 'Salary':
        df = df.drop(df.columns[0], axis=1)

    df = df[['YearsExperience', 'Salary']]
    df.to_csv(csv_path, index=False)
    print(f"Updated CSV saved to {csv_path}.")

csv_path = 'Salary_dataset.csv'
filter_and_save_data(csv_path)
