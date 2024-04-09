from sklearn.datasets import make_regression
import pandas as pd

def generate_and_save_data(filename, n_samples=500, n_features=1, noise=15, random_state=42):
    X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise, random_state=random_state)
    df = pd.DataFrame(X, columns=[f'Feature_{i+1}' for i in range(n_features)])
    df['Target'] = y

    df.to_csv(filename, index=False)

generate_and_save_data('dataset.csv')
