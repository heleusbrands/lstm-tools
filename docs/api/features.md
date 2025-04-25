# Features

`Features` is a 1D array of `Feature` objects that represents a time series of a single variable (e.g., price over time). It provides methods for statistical calculations and allows for custom compression functions.

## Class Definition

```python
class Features(FrameBase):
    def __new__(
        cls, 
        input_data, 
        name=None, 
        dtype=None, 
        time=None, 
        compressors=[])
```

## Parameters

- **input_data** (`List` or `numpy.ndarray`): The input data to create features from
- **name** (`str`, optional): The name of the feature series. Default is None.
- **dtype** (`numpy.dtype`, optional): The numpy data type to use. Default is `np.float32`.
- **time** (`numpy.ndarray` or `pandas.DatetimeIndex`, optional): Time values for the features. Default is None.
- **compressors** (`List[Callable]`, optional): Initial list of compression functions. Default is empty list.

## Attributes

- **compressors** (`List[Callable]`): List of compression functions to apply
- **_time** (`numpy.ndarray` or `pandas.DatetimeIndex`): Time values for the features
- **_shape** (`tuple`): Shape of the feature array
- **_level** (`int`): Hierarchy level of the object (0 for Features)
- **_original_input** (`List` or `numpy.ndarray`): Original input data
- **operations** (`TradeWindowOps`): Class containing statistical operations

## Methods

### Array Interface Methods

#### `__array__`

Return the underlying array data.

```python
def __array__(self)
```

**Returns:**
- `np.ndarray`: The underlying NumPy array data

#### `__array_finalize__`

Finalize the array creation process.

```python
def __array_finalize__(self, obj)
```

**Parameters:**
- **obj** (`object`): Object to finalize

#### `__getitem__`

Get items from the Features by index or slice.

```python
def __getitem__(self, item)
```

**Parameters:**
- **item** (`Union[int, slice]`): Index or slice to retrieve

**Returns:**
- `Union[Feature, Features]`: Single feature or slice of features

### Compression Methods

#### `__add__`

Special method to add compression functions.

```python
def __add__(self, other)
```

**Parameters:**
- **other** (`Union[Callable, Tuple[str, Callable]]`): Compression function or tuple of (name, function)

#### `add_compressor`

Add a compression function to be applied when the `compress` method is called.

```python
def add_compressor(self, compressor, name=None)
```

**Parameters:**
- **compressor** (`Callable`): Function to be applied to the Features array
- **name** (`str`, optional): Name to assign to the compressor

**Returns:**
- `Features`: Self reference for method chaining

**Example:**
```python
>>> features.add_compressor(np.mean)  # Uses "mean" as name
>>> features.add_compressor(lambda x: x.max() - x.min(), "range")  # Custom name
```

#### `compress`

Apply all registered compression functions and return the results as a TimeFrame.

```python
def compress(self)
```

**Returns:**
- `TimeFrame`: A TimeFrame object containing the results of all compression functions

#### `batch_compress`

Apply a batch of common compression operations to the Features.

```python
def batch_compress(self, common_operations=True, custom_compressors=None)
```

**Parameters:**
- **common_operations** (`bool`, optional): Whether to include common operations (mean, std, etc.). Default is True.
- **custom_compressors** (`List`, optional): List of custom compressor functions to add. Default is None.

**Returns:**
- `Features`: Self reference for method chaining

### Statistical Properties

Features provides convenient property accessors for common statistical operations:

- **mean**: Calculate the mean of the features
- **std**: Calculate the standard deviation of the features
- **var**: Calculate the variance of the features
- **skew**: Calculate the skewness of the features
- **kurtosis**: Calculate the kurtosis of the features
- **variation**: Calculate the coefficient of variation
- **first**: Get the first value
- **last**: Get the last value
- **sum**: Calculate the sum of the features
- **min**: Find the minimum value
- **max**: Find the maximum value
- **median**: Calculate the median value

## Usage Examples

### Creating Features

```python
from lstm_tools import Features
import numpy as np

# From a list of values
values = [1.0, 2.0, 3.0, 4.0, 5.0]
features = Features(values, name='price')

# With time values
import pandas as pd
dates = pd.date_range('2023-01-01', periods=5)
features = Features(values, name='price', time=dates)

# With initial compressors
features = Features(values, name='price', compressors=[np.mean, np.std])
```

### Using Compression Functions

```python
# Add individual compression functions
features.add_compressor(np.mean, "mean")
features.add_compressor(np.std, "std")
features.add_compressor(lambda x: np.max(x) - np.min(x), "range")

# Or use the + operator
features + np.mean  # Adds mean as a compressor
features + ("custom_range", lambda x: np.max(x) - np.min(x))

# Use batch_compress for common statistics
features.batch_compress()

# Execute compression to get a TimeFrame of results
compressed = features.compress()
print(compressed.mean, compressed.std, compressed.range)
```

### Using Statistical Properties

```python
# Access statistical properties directly
print(f"Mean: {features.mean}")
print(f"Standard Deviation: {features.std}")
print(f"Variance: {features.var}")
print(f"Skewness: {features.skew}")
print(f"Kurtosis: {features.kurtosis}")
print(f"Coefficient of Variation: {features.variation}")
print(f"First Value: {features.first}")
print(f"Last Value: {features.last}")
print(f"Sum: {features.sum}")
print(f"Min: {features.min}")
print(f"Max: {features.max}")
print(f"Median: {features.median}")
``` 