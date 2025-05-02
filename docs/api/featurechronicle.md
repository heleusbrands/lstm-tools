# FeatureChronicle

`FeatureChronicle` is a 2D array of `FeatureSample` objects, representing windowed versions of features. This class is used when accessing features from a Chronicle object to return windowed versions of the features.

## Class Definition

```python
class FeatureChronicle(FrameBase):
    def __new__(
        cls, 
        input_data: Union[np.ndarray, List], 
        name: Optional[str] = None, 
        dtype: Optional[np.dtype] = None, 
        time: Optional[np.ndarray] = None, 
        compressors: Optional[List[Callable]] = None, 
        idx: Optional[Any] = None
    ) -> 'FeatureChronicle'
```

## Class Attributes

- **subtype** (`type`): The type of elements in the array (`FeatureSample`)
- **level** (`int`): The level in the hierarchy (1 for 2D array)
- **nptype** (`numpy.dtype`): Default numpy data type (`np.float32`)
- **operations** (`class`): Class containing statistical operations (`TradeWindowOps`)

## Parameters

- **input_data** (`Union[np.ndarray, List]`): Input data to create the FeatureChronicle from
- **name** (`Optional[str]`): Name of the feature. Default is None.
- **dtype** (`Optional[np.dtype]`): Data type for the array. Default is None.
- **time** (`Optional[np.ndarray]`): Time values for the data. Default is None.
- **compressors** (`Optional[List[Callable]]`): List of compression functions. Default is None.
- **idx** (`Optional[Any]`): Index values for the data. Default is None.

## Instance Attributes

- **compressors** (`List[Callable]`): List of compression functions
- **_time** (`Optional[np.ndarray]`): Time values for the data
- **name** (`Optional[str]`): Name of the feature
- **_shape** (`tuple`): Shape of the array
- **_level** (`int`): Hierarchy level (always 1)
- **_idx** (`Optional[Any]`): Index values

## Methods

### Array Interface Methods

#### `__getitem__`

Get items from the FeatureChronicle by index or slice.

```python
def __getitem__(self, item: Union[int, slice]) -> Union[FeatureSample, FeatureChronicle]
```

**Parameters:**
- **item** (`Union[int, slice]`): Index or slice to get

**Returns:**
- `Union[FeatureSample, FeatureChronicle]`: Single FeatureSample for integer index, FeatureChronicle for slice

### Compression Methods

#### `add_compressor`

Add a compression function to be applied when the compress method is called.

```python
def add_compressor(self, compressor: Callable, name: Optional[str] = None) -> 'FeatureChronicle'
```

**Parameters:**
- **compressor** (`Callable`): Function to be applied to the data
- **name** (`Optional[str]`): Name for the compressor. Default is None.

**Returns:**
- `FeatureChronicle`: Self reference for method chaining

**Raises:**
- `TypeError`: If compressor is not callable

#### `compress`

Apply all registered compression functions to the data. Each window is compressed creating a new TimeFrame of features. The newly created TimeFrames are then compiled into a new Sample.

```python
def compress(self) -> 'Sample'
```

**Returns:**
- `Sample`: Compressed data as a Sample

#### `batch_compress`

Apply a batch of compression operations.

```python
def batch_compress(
    self, 
    common_operations: bool = True, 
    custom_compressors: Optional[List[Union[Callable, Tuple[Callable, str]]]] = None
) -> 'FeatureChronicle'
```

**Parameters:**
- **common_operations** (`bool`): Whether to include common operations (mean, std, etc.). Default is True.
- **custom_compressors** (`Optional[List[Union[Callable, Tuple[Callable, str]]]]`): List of custom compression functions or tuples of (function, name). Default is None.

**Returns:**
- `FeatureChronicle`: Self reference for method chaining

### Statistical Properties

The following statistical properties are available as property decorators:

- **mean**: Calculate the mean of each window
- **std**: Calculate the standard deviation
- **var**: Calculate the variance
- **skew**: Calculate the skewness
- **kurtosis**: Calculate the kurtosis
- **first**: Get the first value of each window
- **last**: Get the last value of each window
- **sum**: Calculate the sum
- **min**: Get the minimum value
- **max**: Get the maximum value
- **median**: Calculate the median

Each property returns a `FeatureSample` containing the statistical measure for each window.

## Usage Examples

```python
import numpy as np
from lstm_tools import Chronicle

# Create sample data
data = np.random.randn(100, 30, 3)  # 100 windows, 30 timesteps, 3 features
cols = ['price', 'volume', 'volatility']

# Create a Chronicle
chronicle = Chronicle(data, cols=cols)

# Access a feature's windows
price_windows = chronicle.price  # Returns FeatureChronicle

# Access statistical properties
mean_prices = price_windows.mean    # Mean of each window
std_prices = price_windows.std     # Standard deviation of each window

# Add custom compression
def range_calc(x):
    return x.max() - x.min()

price_windows.add_compressor(range_calc, "range")
price_windows.add_compressor(lambda x: x.mean(), "avg")

# Compress with common operations
price_windows.batch_compress(common_operations=True)
compressed = price_windows.compress()
``` 