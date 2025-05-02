# FeatureSample

`FeatureSample` is a 1D array of `Feature` objects that represents a time series of a single variable (e.g., price over time). It provides methods for statistical calculations and allows for custom compression functions.

## Class Definition

```python
class FeatureSample(FrameBase):
    def __new__(
        cls, 
        input_data: Union[List, np.ndarray], 
        name: Optional[str] = None,
        dtype: Optional[np.dtype] = None,
        time: Optional[Union[np.ndarray, pd.DatetimeIndex]] = None,
        compressors: List[Callable] = []
    ) -> 'FeatureSample'
```

## Parameters

- **input_data** (`Union[List, np.ndarray]`): The input data to create features from
- **name** (`Optional[str]`): The name of the feature series. Default is None.
- **dtype** (`Optional[np.dtype]`): The numpy data type to use. Default is `np.float32`.
- **time** (`Optional[Union[np.ndarray, pd.DatetimeIndex]]`): Time values for the features. Default is None.
- **compressors** (`List[Callable]`): Initial list of compression functions. Default is empty list.

## Attributes

- **compressors** (`List[Callable]`): List of compression functions to apply
- **_time** (`Optional[Union[np.ndarray, pd.DatetimeIndex]]`): Time values for the features
- **_shape** (`tuple`): Shape of the feature array
- **_level** (`int`): Hierarchy level of the object (0 for FeatureSample)
- **_original_input** (`Union[List, np.ndarray]`): Original input data
- **operations** (`TradeWindowOps`): Class containing statistical operations

## Methods

### Array Interface Methods

#### `__array__`

Return the underlying array data.

```python
def __array__(self) -> np.ndarray
```

**Returns:**
- `np.ndarray`: The underlying NumPy array data

#### `__array_finalize__`

Finalize the array creation process.

```python
def __array_finalize__(self, obj: Any) -> None
```

**Parameters:**
- **obj** (`Any`): Object to finalize

#### `__getitem__`

Get items from the FeatureSample by index or slice.

```python
def __getitem__(self, item: Union[int, slice]) -> Union[Feature, 'FeatureSample']
```

**Parameters:**
- **item** (`Union[int, slice]`): Index or slice to retrieve

**Returns:**
- `Union[Feature, FeatureSample]`: Single feature or slice of features

### Compression Methods

#### `__add__`

Special method to add compression functions.

```python
def __add__(self, other: Union[Callable, Tuple[str, Callable]]) -> Optional['FeatureSample']
```

**Parameters:**
- **other** (`Union[Callable, Tuple[str, Callable]]`): Compression function or tuple of (name, function)

**Returns:**
- `Optional[FeatureSample]`: Self reference for method chaining if other is a compression function, None if arithmetic

#### `add_compressor`

Add a compression function to be applied when the `compress` method is called.

```python
def add_compressor(self, compressor: Callable, name: Optional[str] = None) -> 'FeatureSample'
```

**Parameters:**
- **compressor** (`Callable`): Function to be applied to the FeatureSample array
- **name** (`Optional[str]`): Name to assign to the compressor

**Returns:**
- `FeatureSample`: Self reference for method chaining

**Example:**
```python
>>> features.add_compressor(np.mean)  # Uses "mean" as name
>>> features.add_compressor(lambda x: x.max() - x.min(), "range")  # Custom name
```

#### `compress`

Apply all registered compression functions and return the results as a TimeFrame.

```python
def compress(self) -> 'TimeFrame'
```

**Returns:**
- `TimeFrame`: A TimeFrame object containing the results of all compression functions

#### `batch_compress`

Apply a batch of common compression operations to the FeatureSample.

```python
def batch_compress(
    self, 
    common_operations: bool = True, 
    custom_compressors: Optional[List[Union[Callable, Tuple[Callable, str]]]] = None
) -> 'FeatureSample'
```

**Parameters:**
- **common_operations** (`bool`): Whether to include common operations (mean, std, etc.). Default is True.
- **custom_compressors** (`Optional[List[Union[Callable, Tuple[Callable, str]]]]`): List of custom compressor functions to add. Default is None.

**Returns:**
- `FeatureSample`: Self reference for method chaining

### Statistical Properties

FeatureSample provides convenient property accessors for common statistical operations:

- **mean** (`float`): Calculate the mean of the features
- **std** (`float`): Calculate the standard deviation of the features
- **var** (`float`): Calculate the variance of the features
- **skew** (`float`): Calculate the skewness of the features
- **kurtosis** (`float`): Calculate the kurtosis of the features
- **variance** (`float`): Calculate the coefficient of variance
- **first** (`float`): Get the first value
- **last** (`float`): Get the last value
- **sum** (`float`): Calculate the sum of the features
- **min** (`float`): Find the minimum value
- **max** (`float`): Find the maximum value
- **median** (`float`): Calculate the median value

## Usage Examples

### Creating FeatureSample

```python
from lstm_tools import FeatureSample
import numpy as np

# From a list of values
values = [1.0, 2.0, 3.0, 4.0, 5.0]
features = FeatureSample(values, name='price')

# With time values
import pandas as pd
dates = pd.date_range('2023-01-01', periods=5)
features = FeatureSample(values, name='price', time=dates)

# With initial compressors
features = FeatureSample(values, name='price', compressors=[np.mean, np.std])
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
print(f"Coefficient of variance: {features.variance}")
print(f"First Value: {features.first}")
print(f"Last Value: {features.last}")
print(f"Sum: {features.sum}")
print(f"Min: {features.min}")
print(f"Max: {features.max}")
print(f"Median: {features.median}")
``` 