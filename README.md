# LSTM Tools
<span style="color:#E83A6B;"> by Bloom Research </span>

A library of custom numpy arrays and objects designed to help with sequential data handling, efficient windowing, and data compression for time series analysis.

Note from Author: 

*"This was a personal tool that I created for my own use during some research, which was created out of frustration with the other tools available. Pandas, as amazing as it is, was not very intuitive for handling complex sequential data. The universal approach made it difficult/repetative to get at the capabilites I needed to access frequently when switching between array shapes. I switched to plain numpy arrays, but soon became frustrated at having to keep track of where each feature was stored, and the confusion caused by dealing with pure numeric representations. The whole process with both libraries felt very 'un-pythonic'. Enter LSTM Tools - Arrays that change structure and methods depending on the current situation."*

## Overview

LSTM Tools provides a high-performance framework for managing and processing sequential data, with a focus on time series analysis and preparation for machine learning models, as well as ease of use. Built on numpy's powerful array operations, the library offers significant advantages:

### Approach

- **Hierarchical Data Structure**: Organizes data in a logical progression from individual data points (Features) to complete windowed datasets (Chronicles), making it intuitive to work with time series at any level of abstraction.
  
- **Lazy Instantiation**: Objects are created only when needed, minimizing memory overhead and processing time, particularly important for large datasets.

- **Attribute-based Access**: Access features by name using standard attribute notation (`sample.price` instead of complex indexing), improving code readability and reducing errors.

- **Seamless ML Integration**: Direct conversion to PyTorch and TensorFlow tensors, with utilities for creating training-ready datasets.

### Performance and Efficiency

- **Optimized Windowing**: Fast window creation using numpy's stride tricks, avoiding unnecessary copying of data, allowing efficient handling of datasets with millions of points.

- **Vectorized Operations**: Statistical calculations leverage numpy's vectorized operations for high performance, up to 100x faster than iterative approaches.

- **Memory Efficiency**: The custom numpy subclassing approach maintains a balance between memory usage and performance, with data stored in optimized numpy arrays while providing a friendly API.

- **Computation Reuse**: Compression operations can be registered and reapplied, saving redundant calculations when processing the same data multiple times.

## Installation

```bash
pip install lstm-tools
```

For development installation:

```bash
git clone https://github.com/heleusbrands/lstm-tools.git
cd lstm-tools
pip install -e .
```

## Features

- **Feature**: A float subclass that represents a single data point with a name attribute. Features can store operations for later execution and integrate with the rest of the LSTM Tools ecosystem.

- **TimeFrame**: A 1D array of Feature objects that represents a snapshot of multiple variables at a specific point in time (e.g., price, volume, indicator values at timestamp X). It provides attribute-based access to named features.

- **Sample**: A 2D array of TimeFrame objects that represents a sequence of multi-variable observations over time. It provides powerful windowing capabilities and feature-specific operations.

- **Chronicle**: A 3D array of windowed Sample objects, designed for working with batches of windowed data. Ideal for compressing Sample windows down to TimeFrame objects, or preparing data for machine learning models in a format ready for LSTM networks.

- **FeatureSample**: A 1D array of Feature objects that represents a time series of a single variable (e.g., price over time). It provides methods for statistical calculations (mean, std, etc.) and allows for custom compression functions to be registered and applied.

- **FeatureChronicle**: A 2D array of a windowed feature, representing multiple (windowed) time series of a single variable (e.g., price over 60 minute windows). It provides easy methods/properties for statistical calculations on the windows contained within it, and is obtained most frequently by accessing a feature through a Chronicle class instance. 

## Quick Start

```python
import numpy as np
import pandas as pd
from lstm_tools import Feature, FeatureSample, TimeFrame, Sample, Chronicle
from lstm_tools.logger import configure_logging

# Load data from a CSV file
# The file should have a 'time' column that will be used as the index
sample = Sample("your_data.csv")

# Alternatively, create from a pandas DataFrame
df = pd.DataFrame({
    'price': [100.0, 101.2, 99.8, 102.5, 103.0],
    'volume': [1000, 1200, 800, 1500, 2000]
}, index=pd.date_range(start='2023-01-01', periods=5, freq='D'))
sample = Sample(df)

# Access features by name (returns a FeatureSample object)
price_data = sample.price
volume_data = sample.volume

# Calculate statistics on features
mean_price = sample.feature_mean('price')
max_volume = sample.feature_max('volume')
price_std = sample.feature_std('price')

# Configure window settings
sample.window_settings.historical.window_size = 3  # 3 time steps for historical data
sample.window_settings.future.window_size = 2     # 2 time steps for future prediction
sample.window_settings.stride = 1                 # Step size for sliding windows

# Working with FeatureSample (1D series)
# Add compression operations to features
price_data.add_compressor(np.mean) # Method added directly, no name necessary 
price_data.add_compressor(lambda x: np.std(x), "std_price") # Method via lambda

# Apply all registered compression operations
compressed = price_data.compress()

# Or use chained operations
compressed = sample.price.add_compressor(np.mean).add_compressor(lambda x: np.std(x), "std_price").compress()

# Or use the convenience method to add standard operations
price_data.batch_compress(custom_compressors=[
    (lambda x: np.max(x) - np.min(x), "range")
])

# Working with Chronicles (3D windowed data)

# Create historical windows (input data for model)
historical_data = sample.historical_sliding_window()

# Create future windows (target data for model)
future_data = sample.future_sliding_window()

# Get both historical and future windows in one call
historical, future = sample.hf_sliding_window() # Returns a tuple[Chronicle, Chronicle]

# Access specific features within the windows
hist_price = historical.price # Direct array access

# Compress with convenience properties
hist_mean_price = hist_price.mean # Converts from FeatureChronicle -> FeatureSample
hist_std_price = hist_price.std
hist_open_price = hist_price.first
hist_close_price = hist_price.last

# Compile back into new Sample, with calculated features
compressed_sample = Sample.from_FeatureSamples([
    hist_mean_price,
    hist_std_price,
    hist_open_price,
    hist_close_price
])

# Extract statistics across all windows in a single operation
stats = historical.batch_compress(
    features=['price', 'volume'],  # Process specific features
    methods={
        'mean': np.mean,           # Calculate mean
        'std': np.std,             # Calculate standard deviation
        'range': lambda x: np.max(x) - np.min(x)  # Custom calculation
    }
)
# Results are returned as a dictionary with keys like 'price_mean', 'volume_std', etc.

# Visualize the data
plot = sample.line_plot()
plot.show()

# Save and load
sample.save("my_sample.pkl")
loaded_sample = Sample.load("my_sample.pkl")

# Convert to tensors for deep learning
import torch
pytorch_tensor = sample.to_ptTensor(device="cuda:0")

# Or TensorFlow
tf_tensor = sample.to_tfTensor()
```

## Chronicle Compression - Quick Example

```python

from lstm_tools import Sample
from lstm_tools.utils import TradeWindowOps
import numpy as np

f = r'files\example.csv'
s = Sample(f)

s.window_settings.historical.window_size = 60*6
c6hr = s.historical_sliding_window()

c6hr.compressors.open = [np.mean, np.std, TradeWindowOps.skew, TradeWindowOps.first]
c6hr.compressors.close = [np.mean, np.std, TradeWindowOps.skew, TradeWindowOps.last]
c6hr.compressors.low = [np.mean, np.std, TradeWindowOps.skew, np.min]
c6hr.compressors.high = [np.mean, np.std, TradeWindowOps.skew, np.max]
c6hr.compressors.volume = [np.mean, np.std, TradeWindowOps.skew, np.sum]
compressed_features = c6hr.compress_all_features()

compressed_sample = Sample.join_samples(compressed_features)

compressed_sample.feature_names

"""
Output: 

['low_mean',
 'low_std',
 'low_skew',
 'low_min',
 'high_mean',
 'high_std',
 'high_skew',
 'high_max',
 'open_mean',
 'open_std',
 'open_skew',
 'open_first',
 'close_mean',
 'close_std',
 'close_skew',
 'close_last',
 'volume_mean',
 'volume_std',
 'volume_skew',
 'volume_sum']
"""
```
## Version Notes
**Version 0.1.0**:

*This is the initial release, so please be aware there will likely be bugs and things that still need to be optimized. Just make sure to report issues, and please feel free to submit feature requests, as these have primarily been tailored to my own usages.*

## Documentation

For full documentation, visit our [documentation site](https://github.com/heleusbrands/lstm-tools/docs).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the GPL-3.0 License - see the LICENSE file for details. 