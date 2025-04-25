# TimeFrame

`TimeFrame` is a 1D array of `Feature` objects representing a collection of features at a specific point in time (a single observation across multiple features).

## Class Definition

```python
class TimeFrame(FrameBase):
    def __new__(cls, input_data, cols, idx=None, name=None, dtype=None, time=None)
```

## Parameters

- **input_data** (`array-like`): Input data, can be a list, numpy array, or another TimeFrame object
- **cols** (`list`): List of column/feature names
- **idx** (`int`, optional): Index of the timeframe. Default is None.
- **name** (`str`, optional): Name of the timeframe. Default is None.
- **dtype** (`dtype`, optional): Data type for the array. Default is None.
- **time** (`np.ndarray` or `pandas.DatetimeIndex`, optional): Time values for the timeframe. Default is None.

## Attributes

- **_cols** (`list`): List of column/feature names
- **_time** (`np.ndarray`): Time values for the timeframe
- **_shape** (`tuple`): Shape of the array
- **_idx** (`int`): Index of the timeframe
- **_level** (`int`): Hierarchy level in the LSTM Tools data structure (always 0)
- **_original_input** (`array-like`): Original input data

## Properties

### `feature_names`

Returns the list of column names associated with this TimeFrame.

```python
@property
def feature_names(self)
```

**Returns:**
- `list`: List of column names

### `shape`

Returns the shape of the TimeFrame.

```python
@property
def shape(self)
```

**Returns:**
- `tuple`: Shape of the TimeFrame

## Methods

### `__getitem__`

Get items from the TimeFrame by index, feature name, or slice.

```python
def __getitem__(self, item)
```

**Parameters:**
- **item** (`Union[str, int, slice]`): 
  - If string: Feature name to retrieve
  - If integer: Index of the feature to retrieve
  - If slice: Range of features to retrieve

**Returns:**
- `Union[Feature, np.ndarray]`: Feature data based on the input type

### `__array__`

Return the underlying array data.

```python
def __array__(self)
```

**Returns:**
- `np.ndarray`: The underlying NumPy array data

### `__array_finalize__`

Finalize the array creation process.

```python
def __array_finalize__(self, obj)
```

**Parameters:**
- **obj** (`object`): Object to finalize

### `__getattr__`

Enables attribute access for features by name.

```python
def __getattr__(self, name)
```

**Parameters:**
- **name** (`str`): Name of the attribute/feature to access

**Returns:**
- `Feature`: The feature with the specified name

**Raises:**
- `AttributeError`: If the feature name is not found in columns

### Data Conversion Methods

#### `to_ptTensor`

Convert the TimeFrame data to a PyTorch tensor.

```python
def to_ptTensor(self, device='cpu')
```

**Parameters:**
- **device** (`str` or `torch.device`): Device to place the tensor on. Default is 'cpu'.

**Returns:**
- `torch.Tensor`: TimeFrame data as a PyTorch tensor on the specified device

#### `to_tfTensor`

Convert the TimeFrame data to a TensorFlow tensor.

```python
def to_tfTensor(self, device='cpu')
```

**Parameters:**
- **device** (`str` or `tf.device`): Device to place the tensor on. Default is 'cpu'.

**Returns:**
- `tf.Tensor`: TimeFrame data as a TensorFlow tensor on the specified device

#### `to_DataFrame`

Convert the TimeFrame data to a pandas DataFrame.

```python
def to_DataFrame(self)
```

**Returns:**
- `pandas.DataFrame`: TimeFrame data as a pandas DataFrame

### Window Methods

#### `window`

Get a window of the TimeFrame.

```python
def window(self, idx)
```

**Parameters:**
- **idx** (`int`): Index of the window

**Returns:**
- `TimeFrame`: Window of the TimeFrame

## Usage Examples

### Creating a TimeFrame

```python
import numpy as np
from lstm_tools import TimeFrame

# Create a TimeFrame from a list with column names
data = [1.0, 2.0, 3.0]
columns = ['temp', 'humidity', 'pressure']
tf = TimeFrame(data, cols=columns)

# Create with time values
import pandas as pd
time = pd.Timestamp('2023-01-01')
tf2 = TimeFrame(data, cols=columns, time=time)
```

### Accessing Features

```python
# Access features by numerical index
first_feature = tf[0]  # Returns Feature object

# Access features by attribute name (column name)
temp = tf.temp
humidity = tf.humidity
pressure = tf.pressure

# Access features by string indexing
temp_feature = tf['temp']

# Check available column names
print(tf.feature_names)  # ['temp', 'humidity', 'pressure']
```

### Converting to Other Formats

```python
# Convert to DataFrame
df = tf.to_DataFrame()
print(df)

# Convert to PyTorch tensor
import torch
tensor = tf.to_ptTensor(device='cuda:0')

# Convert to TensorFlow tensor
tf_tensor = tf.to_tfTensor()
```

### Working with Windows

```python
# Create a window
window = tf.window(0)
print(window.shape)  # Shows shape of the window
``` 