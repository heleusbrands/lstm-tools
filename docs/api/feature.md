# Feature

`Feature` is a subclass of `float` that represents a single value within a time series or dataset. It extends the basic float functionality with a name attribute and integration into the LSTM Tools ecosystem.

## Class Definition

```python
class Feature(float):
    def __new__(cls, value: float, name: str, base_dtype=np.float32) -> 'Feature'
```

## Parameters

- **value** (`float`): The numerical value of the feature
- **name** (`str`): Name of the feature
- **base_dtype** (`numpy.dtype`, optional): Base numpy data type for storing the value internally. Default is `np.float32`.

## Attributes

- **name** (`str`): Name of the feature
- **_base** (`numpy.dtype`): The value stored as the specified numpy data type
- **operations** (`list`): List of operations/compressors that can be applied to the feature

## Methods

### `__new__`

Creates a new Feature instance. This is the constructor method for the Feature class.

```python
def __new__(cls, value: float, name: str, base_dtype=np.float32) -> 'Feature'
```

**Parameters:**
- **value** (`float`): The numerical value of the feature
- **name** (`str`): Name of the feature
- **base_dtype** (`numpy.dtype`): Base data type for internal storage

**Returns:**
- `Feature`: A new Feature instance

### `__repr__`

Returns a string representation of the Feature.

```python
def __repr__(self) -> str
```

**Returns:**
- `str`: String representation in format 'Feature(name: value)'

### `__add__`

Special method to handle addition with both numbers and callables. When adding a callable, it appends it to the operations list instead of performing arithmetic.

```python
def __add__(self, other: Union[float, Callable]) -> Union['Feature', None]
```

**Parameters:**
- **other** (`Union[float, Callable]`): Value to add or callable operation to append

**Returns:**
- `Union[Feature, None]`: Result of addition if other is a number, None if other is a callable (operation was appended)

## Usage Examples

```python
from lstm_tools import Feature

# Create a simple feature
f1 = Feature(3.14159, name='pi')
print(f1)  # Feature(pi: 3.14159)

# Basic arithmetic works like a float
result = f1 * 2
print(result)  # 6.28318

# Add compression operations to the feature
def square(x):
    return x * x
f1 + square  # Adds the square operation to the feature's operations list
```

# FeatureSample

`FeatureSample` is a 1D array of values that represents a time series for a specific feature. It inherits from `FrameBase` and provides methods for compression and statistical operations.

## Class Definition

```python
class FeatureSample(FrameBase):
    def __new__(cls, input_data, name=None, dtype=None, time=None, compressors=[], idx=None)
```

## Parameters

- **input_data** (`Union[list, np.ndarray]`): Input data array
- **name** (`str`, optional): Name of the feature series. Default is None.
- **dtype** (`numpy.dtype`, optional): Data type for the array. Default is np.float32.
- **time** (`Union[pd.DatetimeIndex, str, int, list, np.ndarray], optional`): Time index for the series
- **compressors** (`list`, optional): List of compression functions to apply. Default is empty list.
- **idx** (`Any`, optional): Custom index for the series. Default is None.

## Class Attributes

- **subtype** (`type`): The type of elements in the array (`Feature`)
- **level** (`int`): The level in the hierarchy (0 for 1D array)
- **nptype** (`numpy.dtype`): Default numpy data type (np.float32)
- **operations** (`class`): Class containing statistical operations (TradeWindowOps)

## Instance Attributes

- **name** (`str`): Name of the feature series
- **compressors** (`list`): List of compression functions
- **_time** (`Union[pd.DatetimeIndex, str, int, list, np.ndarray]`): Time index
- **_shape** (`tuple`): Shape of the array
- **_level** (`int`): Hierarchy level (always 0)
- **_idx** (`Any`): Custom index

## Methods

### Compression Methods

#### `add_compressor`

Add a compression function to be applied when the compress method is called.

```python
def add_compressor(self, compressor: Callable, name: str = None) -> 'FeatureSample'
```

**Parameters:**
- **compressor** (`callable`): Function to be applied to the FeatureSample array
- **name** (`str`, optional): Name for the compressor. If None, uses function name

**Returns:**
- `FeatureSample`: Self reference for method chaining

#### `compress`

Apply all registered compression functions to create a TimeFrame.

```python
def compress(self) -> 'TimeFrame'
```

**Returns:**
- `TimeFrame`: A new TimeFrame containing the compressed values

#### `batch_compress`

Apply a batch of common compression operations to the FeatureSample.

```python
def batch_compress(self, common_operations: bool = True, custom_compressors: List[Callable] = None) -> 'FeatureSample'
```

**Parameters:**
- **common_operations** (`bool`): Whether to include common operations (mean, std, etc.)
- **custom_compressors** (`list`): List of custom compressor functions to add

**Returns:**
- `FeatureSample`: Self reference for method chaining

### Statistical Properties

The following statistical properties are available:

- **mean**: Calculate the mean of the series
- **std**: Calculate the standard deviation
- **var**: Calculate the variance
- **skew**: Calculate the skewness
- **kurtosis**: Calculate the kurtosis
- **first**: Get the first value
- **last**: Get the last value
- **sum**: Calculate the sum
- **min**: Get the minimum value
- **max**: Get the maximum value
- **median**: Calculate the median

Each property returns the corresponding statistical measure as a float value.

## Usage Examples

```python
from lstm_tools import FeatureSample

# Create a feature sample
data = [1.0, 2.0, 3.0, 4.0, 5.0]
sample = FeatureSample(data, name="temperature")

# Access statistical properties
print(sample.mean)    # 3.0
print(sample.std)     # 1.4142135623730951
print(sample.min)     # 1.0
print(sample.max)     # 5.0

# Add custom compressors
def range_calc(x):
    return x.max() - x.min()

sample.add_compressor(range_calc, "range")
sample.add_compressor(lambda x: x.mean(), "avg")

# Compress the sample
compressed = sample.compress()
# Returns a TimeFrame with features: temperature_range, temperature_avg

# Batch compress with common operations
sample.batch_compress(common_operations=True)
compressed = sample.compress()
# Returns a TimeFrame with common statistical measures
``` 