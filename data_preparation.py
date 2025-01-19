import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Example dataset generation (replace this with real data)
data = pd.DataFrame({
    'Open': np.random.uniform(100, 200, 1000),
    'High': np.random.uniform(100, 200, 1000),
    'Low': np.random.uniform(100, 200, 1000),
    'Close': np.random.uniform(100, 200, 1000),
    'Volume': np.random.randint(10000, 200000, 1000),
    'MA_10': np.random.uniform(100, 200, 1000),
    'MA_50': np.random.uniform(100, 200, 1000),
    'Target': np.random.choice([0, 1], 1000)  # 0 = Sell, 1 = Buy
})

# Save the dataset to a CSV file
data.to_csv('trading_data.csv', index=False)
print("Dataset saved as 'trading_data.csv'.")
