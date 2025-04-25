# Chronicle

`Chronicle` is a 3D array of windowed `Sample` objects. It's designed for use when windowing and/or compressing samples, providing a powerful way to organize and manipulate multi-window data structures.

## Class Definition

```python
class Chronicle(FrameBase):
    def __new__(
        cls, 
        input_data, 
        cols, 
        idx=None, 
        name=None, 
        dtype=None, 
        is_gen=False, 
        scaler=None, 
        preserve_base=True, 
        time=None)
```

## Parameters

- **input_data** (`array-like`): Input data, can be a list, numpy array, or another Chronicle object
- **cols** (`list`): List of column/feature names
- **idx** (`int`, optional): Index of the chronicle. Default is None.
- **name** (`str`, optional): Name of the chronicle. Default is None.
- **dtype** (`dtype`, optional): Data type for the array. Default is None.
- **is_gen** (`bool`, optional): Whether the chronicle is generated. Default is False.
- **scaler** (`object`, optional): Scaler used for the chronicle. Default is None.
- **preserve_base** (`bool`, optional): Whether to preserve the base data. Default is True.
- **time** (`np.ndarray`, optional): Time values for the chronicle. Default is None.

## Attributes

- **_cols** (`list`): List of column/feature names
- **_time** (`np.ndarray`): Time values for the chronicle
- **_shape** (`tuple`): Shape of the array
- **_idx** (`int`): Index of the chronicle
- **_level** (`int`): Hierarchy level in the LSTM Tools data structure
- **scaler** (`object`): Scaler used for the chronicle
- **name** (`str`): Name of the chronicle
- **is_gen** (`bool`): Whether the chronicle is generated

## Methods

### `__getitem__`

Get items from the Chronicle by index, feature name, or slice.

```python
def __getitem__(self, item)
```

**Parameters:**
- **item** (`Union[str, int, slice, tuple]`): 
  - If string: Feature name to retrieve
  - If integer: Index of the chronicle to retrieve
  - If slice: Range of chronicles to retrieve
  - If tuple: For multi-dimensional indexing

**Returns:**
- `Union[Sample, np.ndarray]`: Data based on the input type

**Example:**
```python
>>> chronicle = sample.historical_sliding_window()
>>> sample_window = chronicle[0]        # Get first sample window
>>> price_data = chronicle['price']     # Get all price data across windows
>>> subset = chronicle[0:10]            # Get first 10 windows
```

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

### `merge_samples_to_chronicle`

Merge a list of Sample instances into a Chronicle by combining their TimeFrames.

```python
@classmethod
def merge_samples_to_chronicle(cls, samples)
```

**Parameters:**
- **samples** (`List[Sample]`): List of Sample instances to merge. All samples must have the same length and time values.

**Returns:**
- `Chronicle`: Chronicle instance containing the merged samples

**Raises:**
- `ValueError`: If samples list is empty, or if samples have different lengths or time values

**Example:**
```python
>>> samples = [Sample(data1, cols), Sample(data2, cols), Sample(data3, cols)]
>>> chronicle = Chronicle.merge_samples_to_chronicle(samples)
```

### `compress`

Compress a feature using a method.

```python
def compress(self, feature, method)
```

**Parameters:**
- **feature** (`str`): Feature to compress
- **method** (`callable`): Method to use for compression

**Returns:**
- `np.ndarray`: Compressed feature data

**Example:**
```python
>>> chronicle = sample.historical_sliding_window()
>>> # Compress the 'price' feature using the mean function
>>> mean_prices = chronicle.compress('price', np.mean)
>>> print(mean_prices.shape)  # One value per window
```

### `batch_compress`

Compress multiple features using multiple methods.

```python
def batch_compress(self, features=None, methods=None)
```

**Parameters:**
- **features** (`list` or `None`, optional): List of feature names or indices to compress. If None, uses all features.
- **methods** (`dict` or `None`, optional): Dictionary mapping method names to callable functions. If None, uses standard statistical methods (mean, std, min, max).

**Returns:**
- `dict`: Dictionary where keys are '{feature_name}_{method_name}' and values are the compressed results

**Example:**
```python
>>> chronicle = sample.historical_sliding_window()
>>> # Compress all features with default methods
>>> compressed = chronicle.batch_compress()
>>> # Compress specific features with specific methods
>>> compressed = chronicle.batch_compress(
...     features=['price', 'volume'],
...     methods={'mean': np.mean, 'range': lambda x: np.max(x) - np.min(x)}
... )
>>> for key, value in compressed.items():
...     print(f"{key}: {value.shape}")
```

### `subwindow_over_samples`

Create a subwindow view of the Chronicle across all samples.

```python
def subwindow_over_samples(self, window_size, direction='backward')
```

**Parameters:**
- **window_size** (`int`): Size of the window to create
- **direction** (`str`, optional): Direction to create the window, either 'forward' or 'backward'. Default is 'backward'.

**Returns:**
- `Chronicle`: A new Chronicle instance containing the subwindow view

**Example:**
```python
>>> chronicle = sample.historical_sliding_window()
>>> # Get the last 5 time steps of each window
>>> last_5 = chronicle.subwindow_over_samples(5, direction='backward')
>>> # Get the first 5 time steps of each window
>>> first_5 = chronicle.subwindow_over_samples(5, direction='forward')
```

### `xy_dataset`

Create an input-output dataset from the Chronicle data.

```python
def xy_dataset(self, x, y, future_size, historical_size, step_size=1)
```

**Parameters:**
- **x** (`np.ndarray`): Input data array
- **y** (`np.ndarray`): Output data array
- **future_size** (`int`): Size of the future window
- **historical_size** (`int`): Size of the historical window
- **step_size** (`int`, optional): Step size between consecutive windows. Default is 1.

**Returns:**
- `tuple`: Tuple containing (x_windows, y_windows) as numpy arrays

**Example:**
```python
>>> chronicle = sample.historical_sliding_window()
>>> # Create input/output windows
>>> x_data = np.array(chronicle)
>>> y_data = np.array(future_chronicle)
>>> x_windows, y_windows = chronicle.xy_dataset(
...     x_data, y_data, future_size=5, historical_size=30, step_size=1
... )
```

### `batch`

Get a batch of data from the Chronicle.

```python
def batch(self, y, batch_size)
```

**Parameters:**
- **y** (`Chronicle`): Output Chronicle
- **batch_size** (`int`): Size of the batch

**Returns:**
- `tuple`: Tuple containing (batch, y_batch)

**Example:**
```python
>>> historical, future = sample.hf_sliding_window()
>>> x_batch, y_batch = historical.batch(future, batch_size=32)
>>> # Use the batches for training
>>> model.train_on_batch(x_batch, y_batch)
```

### Tensor Conversion Methods

#### `to_ptTensor`

Convert the Chronicle data to a PyTorch tensor.

```python
def to_ptTensor(self, device='cpu')
```

**Parameters:**
- **device** (`str` or `torch.device`): Device to place the tensor on. Default is 'cpu'.

**Returns:**
- `torch.Tensor`: Chronicle data as a PyTorch tensor on the specified device

#### `to_tfTensor`

Convert the Chronicle data to a TensorFlow tensor.

```python
def to_tfTensor(self, device='cpu')
```

**Parameters:**
- **device** (`str` or `tf.device`): Device to place the tensor on. Default is 'cpu'.

**Returns:**
- `tf.Tensor`: Chronicle data as a TensorFlow tensor on the specified device

#### `to_tensor`

Alias for `to_ptTensor`.

```python
def to_tensor(self, device='cpu')
```

**Parameters:**
- **device** (`str` or `torch.device`): Device to place the tensor on. Default is 'cpu'.

**Returns:**
- `torch.Tensor`: Chronicle data as a PyTorch tensor on the specified device

## Usage Examples

### Creating a Chronicle

```python
from lstm_tools import Sample, Chronicle
import numpy as np

# Typically created from sliding windows
sample = Sample(data, cols=['price', 'volume', 'volatility'])
sample.window_settings.historical.window_size = 30
chronicle = sample.historical_sliding_window()

# Or manually from a list of samples
samples = [
    Sample(window1_data, cols=['price', 'volume', 'volatility']),
    Sample(window2_data, cols=['price', 'volume', 'volatility']),
    Sample(window3_data, cols=['price', 'volume', 'volatility'])
]
chronicle = Chronicle.merge_samples_to_chronicle(samples)
```

### Feature Compression

```python
# Compress a single feature with a single method
mean_prices = chronicle.compress('price', np.mean)
std_volumes = chronicle.compress('volume', np.std)

# Batch compress multiple features with multiple methods
compressed = chronicle.batch_compress(
    features=['price', 'volume'],
    methods={
        'mean': np.mean,
        'std': np.std,
        'min': np.min,
        'max': np.max,
        'range': lambda x: np.max(x) - np.min(x)
    }
)

# Access the compressed results
price_means = compressed['price_mean']
volume_ranges = compressed['volume_range']
```

### Creating Machine Learning Datasets

```python
# First create historical and future windows
sample.window_settings.historical.window_size = 30  # 30 time steps for input
sample.window_settings.future.window_size = 5      # 5 time steps for output
historical, future = sample.hf_sliding_window()

# Convert to tensors for deep learning
import torch
X = historical.to_ptTensor(device='cuda:0')
y = future.to_ptTensor(device='cuda:0')

# Create PyTorch dataset and dataloader
dataset = torch.utils.data.TensorDataset(X, y)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# Or get batches directly
for i in range(0, len(historical), 32):
    x_batch, y_batch = historical.batch(future, batch_size=32)
    # Use batches for training
    # model.train_on_batch(x_batch, y_batch)
```

### Subwindowing

```python
# Get only the recent part of each historical window
chronicle = sample.historical_sliding_window()
recent_data = chronicle.subwindow_over_samples(window_size=10, direction='backward')

# Convert to tensor for model input
import torch
tensor = recent_data.to_ptTensor()
predictions = model(tensor)
``` 