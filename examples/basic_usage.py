"""
Basic usage example of the LSTM Tools library.

This example demonstrates:
1. Creating Feature objects
2. Creating TimeFrame objects
3. Creating Sample objects
4. Creating Chronicle objects
5. Basic operations on each class
6. Data visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lstm_tools import Feature, Features, TimeFrame, Sample, Chronicle
from lstm_tools.logger import configure_logging

# Configure logging to see informational messages
configure_logging(level=20)  # INFO level

# 1. Create synthetic time series data
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
values1 = np.cumsum(np.random.randn(100)) + 100  # Price
values2 = np.random.randn(100) * 5 + 50  # Volume
values3 = np.random.randn(100) * 0.5 + 2  # Some other feature

# Create a DataFrame
df = pd.DataFrame({
    'price': values1,
    'volume': values2,
    'indicator': values3
}, index=dates)

print("Created synthetic DataFrame:")
print(df.head())

# 2. Creating Feature objects
price_feature = Feature(df['price'][0], 'price')
volume_feature = Feature(df['volume'][0], 'volume')

print("\nFeature objects:")
print(price_feature)
print(volume_feature)

# 3. Creating a Features collection
features_list = [Feature(val, 'price') for val in df['price'][:10]]
price_features = Features(features_list, name='price', time=dates[:10])

print("\nFeatures collection:")
print(price_features)

# 4. Creating a TimeFrame
timeframe_data = [
    Feature(df['price'][0], 'price'),
    Feature(df['volume'][0], 'volume'),
    Feature(df['indicator'][0], 'indicator')
]
timeframe = TimeFrame(timeframe_data, cols=['price', 'volume', 'indicator'], time=dates[0])

print("\nTimeFrame:")
print(timeframe)
print(f"Accessing by name: {timeframe.price}")

# 5. Creating a Sample
sample = Sample(df.values[:20], cols=df.columns, time=dates[:20])

print("\nSample:")
print(sample)
print(f"Shape: {sample.shape}")

# 6. Basic operations on Sample
print("\nSample operations:")
print(f"Mean of price: {sample.feature_mean('price')}")
print(f"Max of volume: {sample.feature_max('volume')}")
print(f"Time range: {sample.time_open} to {sample.time_close}")

# 7. Creating windows
sample.window_settings.historical.window_size = 5
sample.window_settings.future.window_size = 2
historical_windows = sample.historical_sliding_window()

print("\nHistorical windows (Chronicle):")
print(historical_windows)
print(f"Shape: {historical_windows._shape}")

# 8. Save the sample to a file
sample.save("example_sample.pkl")
print("\nSaved sample to example_sample.pkl")

# 9. Load the sample from the file
loaded_sample = Sample.load("example_sample.pkl")
print("\nLoaded sample:")
print(loaded_sample)

# 10. Basic visualization
plt.figure(figsize=(10, 6))
df.plot()
plt.title("Synthetic Time Series Data")
plt.savefig("synthetic_data.png")
print("\nSaved plot to synthetic_data.png") 