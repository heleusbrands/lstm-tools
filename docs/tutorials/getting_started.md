# Getting Started with LSTM Tools

This guide will help you get started with LSTM Tools, walking you through the basic concepts and common use cases.

## Installation

First, install LSTM Tools using pip:

```bash
pip install lstm-tools
```

## Basic Concepts

LSTM Tools provides a hierarchical structure for handling time series data:

1. `Feature`: A single data point with a name
2. `FeatureSample`: A 1D array of values for a single feature
3. `TimeFrame`: A 1D array of features at a specific point in time
4. `Sample`: A 2D array of TimeFrames representing a sequence
5. `Chronicle`: A 3D array of windowed samples

## Quick Example

Here's a simple example to get you started:

```python
import numpy as np
from lstm_tools import Sample

# Create sample data
data = np.random.randn(100, 3)  # 100 timeframes, 3 features
feature_names = ['price', 'volume', 'volatility']

# Create a Sample object
sample = Sample(data, cols=feature_names)

# Access features by name
prices = sample.price  # Returns a FeatureSample
volumes = sample.volume

# Create windows for machine learning
sample.window_settings.historical.window_size = 30  # Input/Historical window size
sample.window_settings.future.window_size = 5      # Output/Future window size

# Create historical-future windows
historical, future = sample.hf_sliding_window()

# Convert to tensors (PyTorch or TensorFlow)
x_tensor = historical.to_tensor()
y_tensor = future.to_tensor()
```

## Working with Features

Features are the basic building blocks:

```python
from lstm_tools import Feature

# Create a feature
price = Feature(100.5, name='price')
print(price)  # Feature(price: 100.5)

# Features work like regular floats
doubled = price * 2
print(doubled)  # 201.0
```

## Working with FeatureSamples

FeatureSamples represent a time series of a single feature:

```python
from lstm_tools import FeatureSample

# Create a feature sample
prices = FeatureSample([100.5, 101.2, 99.8, 100.1], name='price')

# Access statistical properties
print(prices.mean)    # 100.4
print(prices.std)     # 0.52440256
print(prices.min)     # 99.8
print(prices.max)     # 101.2

# Add compression functions
def range_calc(x):
    return x.max() - x.min()

prices.add_compressor(range_calc, "range")
prices.add_compressor(lambda x: x.mean(), "avg")

# Compress the sample
compressed = prices.compress()
```

## Working with TimeFrames

TimeFrames represent all features at a specific point in time:

```python
import numpy as np
from lstm_tools import TimeFrame

# Create data for multiple features
data = [100.5, 1000000, 0.15]  # price, volume, volatility
feature_names = ['price', 'volume', 'volatility']

# Create a TimeFrame
frame = TimeFrame(data, cols=feature_names)

# Access features by name
print(frame.price)      # Feature(price: 100.5)
print(frame.volume)     # Feature(volume: 1000000.0)
print(frame.volatility) # Feature(volatility: 0.15000000596046448)
```

## Working with Samples

Samples represent sequences of TimeFrames:

```python
import numpy as np
from lstm_tools import Sample

# Create sample data (100 timeframes, 3 features)
data = np.random.randn(100, 3)
feature_names = ['price', 'volume', 'volatility']

# Create a Sample
sample = Sample(data, cols=feature_names)

# Access entire feature sequences
price_series = sample.price    # Returns FeatureSample
volume_series = sample.volume  # Returns FeatureSample

# Create windows
sample.window_settings.historical.window_size = 30
windows = sample.historical_sliding_window()
print(windows.shape)  # (70, 30, 3) - 70 windows of size 30 timeframes with 3 features each
```

## Working with Chronicles

Chronicles are collections of windowed samples:

```python
import numpy as np
from lstm_tools import Chronicle

# Create windowed data (50 windows, 30 timesteps, 3 features)
data = np.random.randn(50, 30, 3)
feature_names = ['price', 'volume', 'volatility']

# Create a Chronicle
chronicle = Chronicle(data, cols=feature_names)

# Access windows and features
first_window = chronicle[0]           # Returns a Sample
price_data = chronicle['price']       # Returns all price data
subset = chronicle[10:20]            # Returns a Chronicle with 10 windows

# Compress features
mean_prices = chronicle.compress('price', np.mean)
compressed = chronicle.batch_compress(
    features=['price', 'volume'],
    methods={'mean': np.mean, 'std': np.std}
)

# Convert to tensors for deep learning
pt_tensor = chronicle.to_ptTensor()  # PyTorch tensor
tf_tensor = chronicle.to_tfTensor()  # TensorFlow tensor
```

## Next Steps

- Check out the [API Reference](../api/index.md) for detailed documentation
- Look at the [Examples](../examples/index.md) for more complex use cases
- Read the [Tutorials](../tutorials/index.md) for in-depth guides 