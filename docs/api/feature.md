# Feature

`Feature` is a subclass of `float` that represents a single value within a time series or dataset, with a name and integration into the LSTM Tools ecosystem.

## Class Definition

```python
class Feature(float):
    def __new__(cls, value, name, base_dtype=np.float32)
```

## Parameters

- **value** (`float`): The numerical value of the feature
- **name** (`str`): Name of the feature
- **base_dtype** (`numpy.dtype`, optional): Base data type for the feature. Default is `np.float32`.

## Attributes

- **name** (`str`): Name of the feature
- **_base** (`numpy.dtype`): Base value stored as the specified data type
- **operations** (`list`): List of operations that can be applied to the feature

## Methods

### `__repr__`

Returns a string representation of the Feature.

```python
def __repr__(self)
```

**Returns:**
- `str`: String representation in format 'Feature(name: value)'

### `__add__`

Special method to handle addition with both numbers and callables.

```python
def __add__(self, other)
```

**Parameters:**
- **other** (`Union[float, Callable]`): Value to add or callable operation to append

**Returns:**
- `Union[Feature, None]`: Result of addition or None if operation was appended

## Usage Examples

```python
from lstm_tools import Feature

# Create a simple feature
f1 = Feature(3.14159, name='pi')
print(f1)  # Feature(pi: 3.14159)

# Basic arithmetic works like a float
result = f1 * 2
print(result)  # 6.28318

# Add operations to the feature
def square(x):
    return x * x
f1 + square  # Adds the square operation to the feature's operations list
```

## Features

`Features` is a 1D array of `Feature` objects, representing a time series of values for a specific feature or metric.

### Class Definition

```python
class Features(np.ndarray):
    def __new__(cls, input_array, name=None, meta=None, dtype=None)
```

### Parameters

- **input_array** (`array-like`): Input data, can be a list, numpy array, or another Features object
- **name** (`str`, optional): Name of the feature collection. Default is None.
- **meta** (`dict`, optional): Metadata associated with the feature collection. Default is None.
- **dtype** (`dtype`, optional): Data type for the array. Default is None.

### Attributes

- **name** (`str`): Name of the feature collection
- **meta** (`dict`): Metadata dictionary
- **_level** (`int`): Hierarchy level in the LSTM Tools data structure (always 0)
- **_shape** (`tuple`): Shape of the array
- **_compression_functions** (`list`): List of compression functions to apply

### Methods

#### Factory Methods

##### `from_list`

Creates a Features object from a list of values.

```python
@classmethod
def from_list(cls, values, name=None, meta=None)
```

**Parameters:**

- **values** (`list`): List of values to convert
- **name** (`str`, optional): Name for the Features object
- **meta** (`dict`, optional): Metadata for the Features object

**Returns:**

- `Features`: New Features object

##### `from_dataframe`

Creates a Features object from a pandas DataFrame column.

```python
@classmethod
def from_dataframe(cls, df, col_name, meta=None)
```

**Parameters:**

- **df** (`pandas.DataFrame`): Source DataFrame
- **col_name** (`str`): Column name to extract
- **meta** (`dict`, optional): Metadata for the Features object

**Returns:**

- `Features`: New Features object with the column's name

#### Compression Methods

##### `register_compression_function`

Registers a function to be used for compression.

```python
def register_compression_function(self, func)
```

**Parameters:**

- **func** (`callable`): Function that takes a Features object and returns a value or Feature

##### `compress`

Compresses the Features using the registered compression functions.

```python
def compress(self, funcs=None)
```

**Parameters:**

- **funcs** (`list`, optional): List of compression functions to use. If None, uses registered functions.

**Returns:**

- `Feature` or `list`: Compressed feature(s)

#### Common Statistical Functions

The Features class provides several built-in statistical functions:

- **mean**: Calculate the mean of the Features
- **min**: Find the minimum value in the Features
- **max**: Find the maximum value in the Features
- **median**: Calculate the median of the Features
- **std**: Calculate the standard deviation of the Features
- **var**: Calculate the variance of the Features
- **sum**: Calculate the sum of all values in the Features
- **count**: Get the number of elements in the Features

#### Utility Methods

##### `to_dict`

Returns a dictionary representation of the Features.

```python
def to_dict(self)
```

**Returns:**

- `dict`: Dictionary containing the array data and metadata

##### `from_dict`

Creates a Features object from a dictionary.

```python
@classmethod
def from_dict(cls, data)
```

**Parameters:**

- **data** (`dict`): Dictionary containing the feature data and metadata

**Returns:**

- `Features`: A new Features instance

### Usage Examples

```python
import numpy as np
from lstm_tools import Features, Feature

# Create from a list of values
values = [1.0, 2.0, 3.0, 4.0, 5.0]
features = Features(values, name="temperature")

# Create from numpy array
array_data = np.array([10.5, 11.2, 9.8, 10.1])
humidity = Features(array_data, name="humidity")

# Access individual values (returns Feature objects)
first_temp = features[0]
print(first_temp)  # 1.0
print(type(first_temp))  # <class 'lstm_tools.feature.Feature'>

# Calculate statistics
avg_temp = features.mean()
print(avg_temp)  # 3.0

max_humidity = humidity.max()
print(max_humidity)  # 11.2

# Register and use compression functions
features.register_compression_function(lambda x: x.mean())
features.register_compression_function(lambda x: x.max())

# Compress to get summary statistics
compressed = features.compress()
print(compressed)  # [3.0, 5.0]

# Custom compression function
def range_feature(feat):
    return Feature(feat.max() - feat.min(), meta={"type": "range"})

features.register_compression_function(range_feature)
result = features.compress()
print(result[-1])  # 4.0
print(result[-1].meta)  # {"type": "range"}

# Working with pandas DataFrame
import pandas as pd
df = pd.DataFrame({
    'temperature': [20.1, 21.5, 22.0, 21.8],
    'humidity': [45, 48, 51, 47]
})
temp_features = Features.from_dataframe(df, 'temperature')
``` 