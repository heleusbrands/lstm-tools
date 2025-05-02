# LSTM Tools vs Pandas: A Comparison

LSTM Tools provides several advantages over Pandas when working with sequential data, particularly for time series analysis and machine learning preparation. This document highlights key differences through practical examples.

## 1. Windowing Operations

### Creating Historical and Future Windows

**Pandas:**
```python
import pandas as pd
import numpy as np

# Create historical and future windows in pandas
def create_windows_pandas(df, historical_size=10, future_size=5):
    windows = []
    targets = []
    
    for i in range(len(df) - historical_size - future_size + 1):
        historical = df.iloc[i:i+historical_size]
        future = df.iloc[i+historical_size:i+historical_size+future_size]
        windows.append(historical.values)
        targets.append(future.values)
    
    return np.array(windows), np.array(targets)

# Usage with pandas
df = pd.DataFrame({'price': range(100)})
X, y = create_windows_pandas(df)
```

**LSTM Tools:**
```python
from lstm_tools import Sample

# Create a sample from data
sample = Sample.from_dataframe(df)

# Configure window sizes
sample.window_settings.historical.window_size = 50
sample.window_settings.future.window_size = 1 # Default, so unnecessary
# Get windows in one line
historical, future = sample.hf_sliding_window()
```

**Advantage:** LSTM Tools provides a much simpler, more intuitive API for windowing operations. The same operation that requires a custom function in pandas is achieved with a single method call in LSTM Tools.

## 2. Feature Access and Compression

### Accessing and Compressing Features Across Windows

**Pandas:**
```python
import pandas as pd
import numpy as np

# Calculate multiple statistics for windowed data in pandas
def window_stats_pandas(df, window_size=10):
    results = {}
    
    # Calculate mean
    means = df.rolling(window=window_size).mean()
    # Calculate std
    stds = df.rolling(window=window_size).std()
    # Calculate min/max
    mins = df.rolling(window=window_size).min()
    maxs = df.rolling(window=window_size).max()
    
    return pd.DataFrame({
        'mean': means,
        'std': stds,
        'min': mins,
        'max': maxs
    })

# Usage
df = pd.DataFrame({'price': range(100), 'time': pd.date_range(start='2024-01-01', periods=100, freq='D')})
stats = window_stats_pandas(df)
```

**LSTM Tools:**
```python
from lstm_tools import Sample

# Create sample
sample = Sample(df)

# Get windowed view
windows = sample.historical_sliding_window()

# Access feature directly and get all statistics in one call
price_stats = windows.price.batch_compress()
# Returns dictionary with 'mean', 'std', 'min', 'max', etc.
```

**Advantage:** LSTM Tools provides attribute-based access to features and built-in compression methods. The same operations that require multiple rolling window calculations in pandas are handled automatically with a single method call.

## 3. Attribute-Based Access

### Accessing Features Across Dimensions

**Pandas:**
*Note that pandas does allow attribute based access to features, even with rolling window, however gaining access to the underlying array is often very complex. The following example shows two different methods for gaining access to the windowed data directly, with pandas, in comparison to how 3D window data is accessed via LSTM Tools.*
```python
import pandas as pd
import numpy as np

# Sample data
s = pd.DataFrame({
    'price': np.random.randn(100), 
    'volume': np.random.randn(100), 
    'time': pd.date_range(start='2024-01-01', periods=100, freq='D')
})

window_size = 3

# Create the Rolling object
roller = s.rolling(window=window_size)

# Option A: Get windows as NumPy arrays (generally faster)
def get_window_raw(window_np_array):
    # You can process or just store the array here
    # For demonstration, we just return it (apply will build a Series of arrays)
    return window_np_array

# Note: The result's index aligns with the *end* of the window.
# The first few results corresponding to incomplete windows will be NaN
# unless min_periods is set appropriately in rolling().
# apply() often skips the initial incomplete windows by default.
window_arrays = roller.price.apply(get_window_raw, raw=True)
# print("\nResulting Series of arrays (NaN for initial incomplete windows):")
# print(window_arrays) # This might print objects if arrays are returned

# Option B: Get windows as pandas Series (more overhead, but retains index)
def get_window_series(window_series):
    # Return the series or some calculation
    return window_series # Returning the series might not work well within apply's structure

# Applying a function that returns a Series within apply can be tricky;
# it's usually used to return a scalar aggregation per window.
# For just *accessing* the series, printing or collecting in a list is better.

collected_series = []
def collect_window_series(window_series):
    if len(window_series) == window_size: # Process only full windows if needed
        collected_series.append(window_series.copy()) # Store a copy
    return np.nan # Apply needs a scalar return value usually, return dummy

roller.price.apply(collect_window_series, raw=False)

```

**LSTM Tools:**
```python
from lstm_tools import Sample

sample = Sample(df)
windows = sample.historical_sliding_window()

# Access feature across all windows with attribute syntax
# Returns a FeatureChronicle object, which is np.array subclass with built-in statistical properties
price_windows = windows.price # Instant array access. 

```

**Advantage:** While both libraries support attribute-based access, LSTM Tools extends this to windowed data and provides immediate access to statistical properties as well as the underlying data if needed. It does this while keeping the time data intact, properly parsing and passing it to the new windows/classes.The returned objects are specialized for time series analysis with built-in methods for common operations.

In addition, it maintains it's performance, since even with windows, it only creates a view of the object when needed (i.e. Lazy Instantiation).

## Summary of Advantages

1. **Simplified Windowing:** LSTM Tools provides intuitive methods for creating historical and future windows, eliminating the need for custom windowing functions.

2. **Integrated Compression:** Built-in methods for calculating statistics and compressing windowed data, reducing the need for multiple rolling window calculations.

3. **Attribute-Based Access:** Intuitive access to features across windows with built-in statistical properties and compression methods.

4. **Specialized Objects:** Purpose-built classes for time series data that understand the relationships between features, samples, and windows.

These advantages make LSTM Tools particularly well-suited for time series analysis and machine learning tasks, where the focus is on sequential data processing and model preparation.
