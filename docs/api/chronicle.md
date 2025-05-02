# Chronicle

`Chronicle` is a 3D array-like object for representing chronicles in time series data. It extends `FrameBase` to include chronicle-specific attributes and methods, designed for efficient handling of windowed and compressed time series data.

## Class Definition

```python
class Chronicle(FrameBase):
    def __new__(
        cls, 
        input_data: Union[np.ndarray, list], 
        cols: List[str], 
        idx: Optional[int] = None, 
        name: Optional[str] = None, 
        dtype: Optional[np.dtype] = None, 
        is_gen: bool = False, 
        scaler: Optional[Any] = None, 
        preserve_base: bool = True, 
        time: Optional[np.ndarray] = None
    ) -> 'Chronicle'
```

## Class Attributes

- **subtype** (`type`): The type of elements in the array (`Sample`)
- **level** (`int`): The level in the hierarchy (0)
- **nptype** (`numpy.dtype`): Default numpy data type (`np.float32`)

## Parameters

- **input_data** (`Union[np.ndarray, list]`): Input data to create the chronicle
- **cols** (`List[str]`): List of column names for the chronicle
- **idx** (`Optional[int]`): Index of the chronicle. Default is None.
- **name** (`Optional[str]`): Name of the chronicle. Default is None.
- **dtype** (`Optional[np.dtype]`): Data type of the chronicle. Default is None.
- **is_gen** (`bool`): Whether the chronicle is generated. Default is False.
- **scaler** (`Optional[Any]`): Scaler used for the chronicle. Default is None.
- **preserve_base** (`bool`): Whether to preserve the base data. Default is True.
- **time** (`Optional[np.ndarray]`): Time values for the chronicle. Default is None.

## Instance Attributes

- **_cols** (`List[str]`): List of column/feature names
- **_time** (`Optional[np.ndarray]`): Time values for the chronicle
- **_shape** (`tuple`): Shape of the array
- **_idx** (`Optional[int]`): Index of the chronicle
- **_level** (`int`): Hierarchy level (always 0)
- **scaler** (`Optional[Any]`): Scaler used for the chronicle
- **name** (`Optional[str]`): Name of the chronicle
- **is_gen** (`bool`): Whether the chronicle is generated

## Methods

### `__getitem__`

Get items from the Chronicle by index, feature name, or slice.

```python
def __getitem__(self, item: Union[str, int, slice, tuple]) -> Union[Sample, np.ndarray]
```

**Parameters:**
- **item** (`Union[str, int, slice, tuple]`): 
  - If string: Feature name to retrieve
  - If integer: Index of the chronicle to retrieve
  - If slice: Range of chronicles to retrieve
  - If tuple: For multi-dimensional indexing

**Returns:**
- `Union[Sample, np.ndarray]`: Chronicle data based on the input type

**Raises:**
- `IndexError`: If the index is out of bounds

**Example:**
```python
>>> chronicle = sample.historical_sliding_window()
>>> sample_window = chronicle[0]        # Get first sample window
>>> price_data = chronicle['price']     # Get all price data across windows
>>> subset = chronicle[0:10]           # Get first 10 windows
```

### `merge_samples_to_chronicle`

Merge a list of Sample instances into a Chronicle.

```python
@classmethod
def merge_samples_to_chronicle(cls, samples: List[Sample]) -> Chronicle
```

**Parameters:**
- **samples** (`List[Sample]`): List of Sample instances to merge

**Returns:**
- `Chronicle`: Chronicle instance containing the merged samples

**Raises:**
- `EmptyDataError`: If samples list is empty
- `InvalidDataTypeError`: If samples have invalid types
- `DataError`: If samples have different lengths or time values

### Tensor Conversion Methods

#### `to_ptTensor`

Convert the Chronicle to a PyTorch tensor.

```python
def to_ptTensor(self, device: str = 'cpu') -> torch.Tensor
```

**Parameters:**
- **device** (`str`): PyTorch device to place the tensor on. Default is 'cpu'.

**Returns:**
- `torch.Tensor`: PyTorch tensor representation of the Chronicle

#### `to_tfTensor`

Convert the Chronicle to a TensorFlow tensor.

```python
def to_tfTensor(self, device: str = 'cpu') -> tf.Tensor
```

**Parameters:**
- **device** (`str`): TensorFlow device to place the tensor on. Default is 'cpu'.

**Returns:**
- `tf.Tensor`: TensorFlow tensor representation of the Chronicle

#### `to_tensor`

Convert the Chronicle to a tensor based on the available framework.

```python
def to_tensor(self, device: str = 'cpu') -> Union[torch.Tensor, tf.Tensor]
```

**Parameters:**
- **device** (`str`): Device to place the tensor on. Default is 'cpu'.

**Returns:**
- `Union[torch.Tensor, tf.Tensor]`: Tensor representation of the Chronicle

### Data Processing Methods

#### `xy_dataset`

Create an input-output dataset from the Chronicle data.

```python
def xy_dataset(
    self, 
    x: np.ndarray, 
    y: np.ndarray, 
    future_size: int, 
    historical_size: int, 
    step_size: int = 1
) -> Tuple[np.ndarray, np.ndarray]
```

**Parameters:**
- **x** (`np.ndarray`): Input data array
- **y** (`np.ndarray`): Output data array
- **future_size** (`int`): Size of the future window
- **historical_size** (`int`): Size of the historical window
- **step_size** (`int`): Step size between consecutive windows. Default is 1.

**Returns:**
- `Tuple[np.ndarray, np.ndarray]`: Tuple containing (x_windows, y_windows)

#### `batch`

Get a batch of data from the Chronicle.

```python
def batch(self, y: Chronicle, batch_size: int) -> Tuple[np.ndarray, np.ndarray]
```

**Parameters:**
- **y** (`Chronicle`): Output Chronicle
- **batch_size** (`int`): Size of the batch

**Returns:**
- `Tuple[np.ndarray, np.ndarray]`: Tuple containing (batch, y_batch)

#### `subwindow_over_samples`

Create a subwindow view of the Chronicle across all samples.

```python
def subwindow_over_samples(
    self, 
    window_size: int, 
    direction: str = 'backward'
) -> Chronicle
```

**Parameters:**
- **window_size** (`int`): Size of the window to create
- **direction** (`str`): Direction to create the window, either 'forward' or 'backward'. Default is 'backward'.

**Returns:**
- `Chronicle`: A new Chronicle instance containing the subwindow view

### Compression Methods

#### `compress`

Compress a feature using a method.

```python
def compress(self, feature: str, method: callable) -> np.ndarray
```

**Parameters:**
- **feature** (`str`): Feature to compress
- **method** (`callable`): Method to use for compression

**Returns:**
- `np.ndarray`: Compressed feature data

#### `batch_compress`

Compress multiple features using multiple methods.

```python
def batch_compress(
    self, 
    features: Optional[List[str]] = None, 
    methods: Optional[Dict[str, callable]] = None
) -> Dict[str, np.ndarray]
```

**Parameters:**
- **features** (`Optional[List[str]]`): List of feature names to compress. If None, uses all features.
- **methods** (`Optional[Dict[str, callable]]`): Dictionary mapping method names to callable functions. If None, uses standard statistical methods.

**Returns:**
- `Dict[str, np.ndarray]`: Dictionary where keys are '{feature_name}_{method_name}' and values are the compressed results

## Usage Examples

```python
import numpy as np
from lstm_tools import Chronicle, Sample

# Create sample data
data = np.random.randn(100, 10, 3)  # 100 windows, 10 timesteps, 3 features
cols = ['price', 'volume', 'volatility']

# Create a Chronicle
chronicle = Chronicle(data, cols=cols)

# Access data
first_window = chronicle[0]  # Returns a Sample
price_data = chronicle['price']  # Returns all price data
subset = chronicle[10:20]  # Returns a Chronicle with 10 windows

# Create input-output dataset
x_data = np.array(chronicle)
y_data = np.array(future_chronicle)
x_windows, y_windows = chronicle.xy_dataset(
    x_data, y_data, 
    future_size=5, 
    historical_size=30, 
    step_size=1
)

# Compress features
mean_prices = chronicle.compress('price', np.mean)
compressed = chronicle.batch_compress(
    features=['price', 'volume'],
    methods={'mean': np.mean, 'std': np.std}
)

# Convert to tensors
pt_tensor = chronicle.to_ptTensor(device='cuda')  # PyTorch tensor
tf_tensor = chronicle.to_tfTensor()  # TensorFlow tensor
``` 