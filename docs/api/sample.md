# Sample

`Sample` is a 2D array of `TimeFrame` objects representing a sequence of multiple features over time. It allows for attribute access of features, which returns a `Features` object, and provides easy windowing capabilities for both historical and future (offset) windows.

## Class Definition

```python
class Sample(FrameBase):
    def __new__(cls, 
        input_data, 
        cols=None, 
        idx=None, 
        name=None, 
        df_names=None, 
        dtype=None, 
        use_scaler=False, 
        scaler=None, 
        time=None)
```

## Parameters

- **input_data** (`array-like`): Input data, can be a list, numpy array, pandas DataFrame, or another Sample object
- **cols** (`list`, optional): List of column/feature names. Default is None.
- **idx** (`int`, optional): Index of the sample. Default is None.
- **name** (`str`, optional): Name of the sample. Default is None.
- **df_names** (`list`, optional): List of DataFrame names when loading from multiple CSV files. Default is None.
- **dtype** (`dtype`, optional): Data type for the array. Default is None.
- **use_scaler** (`bool`, optional): Whether to use a scaler. Default is False.
- **scaler** (`object`, optional): Scaler to use. Default is None.
- **time** (`np.ndarray`, optional): Time values for the sample. Default is None.

## Attributes

- **_cols** (`list`): List of column/feature names
- **_time** (`np.ndarray`): Time values for the sample
- **_shape** (`tuple`): Shape of the array
- **_idx** (`int`): Index of the sample
- **_level** (`int`): Hierarchy level in the LSTM Tools data structure
- **scaler** (`object`): Scaler used for the sample
- **name** (`str`): Name of the sample
- **window_settings** (`HFWindowSettings`): Settings for historical and future windows

## Properties

### `feature_names`

Returns the list of column names associated with this Sample.

```python
@property
def feature_names(self)
```

**Returns:**
- `list`: List of column names

### `inverse`

Get the inverse-transformed Sample using the fitted scaler.

```python
@property
def inverse(self)
```

**Returns:**
- `Sample`: A new Sample instance with inverse-transformed data

**Raises:**
- `ValueError`: If no scaler is set

### `time_open`

Get the first timestamp in the time series.

```python
@property
def time_open(self)
```

**Returns:**
- `Union[pd.Timestamp, np.datetime64, None]`: First timestamp in the series

### `time_close`

Get the last timestamp in the time series.

```python
@property
def time_close(self)
```

**Returns:**
- `Union[pd.Timestamp, np.datetime64, None]`: Last timestamp in the series

## Methods

### `__getitem__`

Allows indexing into the Sample to retrieve TimeFrame objects by index, slice, or mask.

```python
def __getitem__(self, key)
```

**Parameters:**
- **key** (`str`, `int`, `slice`, or `array-like`): 
  - If `str`: Feature name to retrieve
  - If `int`: Index of the TimeFrame to retrieve
  - If `slice`: Range of TimeFrames to retrieve
  - If `array-like`: Boolean mask or integer indices

**Returns:**
- `Features`: When a string (feature name) is provided
- `TimeFrame`: When a single index is provided
- `Sample`: When a slice or array of indices is provided

**Example:**
```python
# Get the first TimeFrame
first_timeframe = sample[0]

# Get a slice of TimeFrames (returns a Sample)
subset = sample[1:5]

# Get a feature by name (returns a Features object)
temperature = sample['temp']

# Use boolean indexing
mask = np.array([True, False, True, False, True])
filtered = sample[mask]
```

### `__getattr__`

Enables attribute access for features by name, returning a Features object.

```python
def __getattr__(self, name)
```

**Parameters:**
- **name** (`str`): Name of the attribute/feature to access

**Returns:**
- `Features`: A Features object containing all values for the requested feature across all timeframes

**Raises:**
- `AttributeError`: If the feature name is not found in columns

### Windowing Methods

#### `historical_sliding_window`

Creates a window of past data from the Sample based on window settings.

```python
def historical_sliding_window(self)
```

**Returns:**
- `Chronicle`: A Chronicle object containing the historical windows

#### `future_sliding_window`

Creates a window of future data from the Sample based on window settings.

```python
def future_sliding_window(self)
```

**Returns:**
- `Chronicle`: A Chronicle object containing the future windows

#### `hf_sliding_window`

Creates both historical and future windows from the Sample based on window settings.

```python
def hf_sliding_window(self)
```

**Returns:**
- `tuple`: (historical_chronicle, future_chronicle)

### Feature Methods

#### `get_feature`

Retrieves a specific feature by name or index across all timeframes.

```python
def get_feature(self, feature, exc=None)
```

**Parameters:**
- **feature** (`Union[int, str]`): Feature to retrieve, either by index or name
- **exc** (`List[Union[int, str]]`, optional): Features to exclude if feature="all"

**Returns:**
- `np.ndarray`: Feature data

#### Statistical Methods

The following methods calculate various statistics for a feature:

```python
def feature_max(self, feature, exc=None)
def feature_min(self, feature, exc=None)
def feature_mean(self, feature, exc=None)
def feature_std(self, feature, exc=None)
def feature_var(self, feature, exc=None)
def feature_skew(self, feature, exc=None)
def feature_kurtosis(self, feature, exc=None)
def feature_variance(self, feature, exc=None)
def feature_sum(self, feature, exc=None)
def feature_first(self, feature, exc=None)
def feature_last(self, feature, exc=None)
```

Each method takes:
- **feature** (`Union[int, str]`): Feature to analyze, either by index or name
- **exc** (`List[Union[int, str]]`, optional): Features to exclude if feature="all"

And returns the corresponding statistical measure as a float.

### Visualization Methods

#### `line_plot`

Create an interactive line plot of the sample data using plotly.

```python
def line_plot(self, exclude=None, opacity=0.9)
```

**Parameters:**
- **exclude** (`str` or `list`, optional): Feature(s) to exclude from the plot
- **opacity** (`float`, optional): Transparency level (0.0 to 1.0). Default is 0.9.

**Returns:**
- `plotly.graph_objects.Figure`: Interactive line plot figure

### Factory Methods

#### `from_dataframe`

Creates a Sample from a pandas DataFrame.

```python
@classmethod
def from_dataframe(cls, df, use_scaler=False, scaler=None)
```

**Parameters:**
- **df** (`pandas.DataFrame`): Source DataFrame
- **use_scaler** (`bool`, optional): Whether to use a scaler. Default is False.
- **scaler** (`object`, optional): Scaler to use. Default is None.

**Returns:**
- `Sample`: New Sample containing the DataFrame data

### Data Conversion Methods

#### `to_ptTensor`

Converts the Sample to a PyTorch tensor.

```python
def to_ptTensor(self, device='cpu')
```

**Parameters:**
- **device** (`str`, optional): Device to place the tensor on. Default is 'cpu'.

**Returns:**
- `torch.Tensor`: PyTorch tensor representation

#### `to_tfTensor`

Converts the Sample to a TensorFlow tensor.

```python
def to_tfTensor(self, device='cpu')
```

**Parameters:**
- **device** (`str`, optional): Device to place the tensor on. Default is 'cpu'.

**Returns:**
- `tf.Tensor`: TensorFlow tensor representation

#### `to_DataFrame`

Converts the Sample to a pandas DataFrame.

```python
def to_DataFrame(self)
```

**Returns:**
- `pandas.DataFrame`: DataFrame representation

### File Operations

#### `save_dataframe`

Save the sample to a CSV file.

```python
def save_dataframe(self, path)
```

**Parameters:**
- **path** (`str`): Path to save the CSV file

#### `save`

Saves the Sample to a file.

```python
def save(self, path, format='pickle')
```

**Parameters:**
- **path** (`str`): Path to save the file
- **format** (`str`, optional): Format to save in ('pickle', 'csv', 'json'). Default is 'pickle'.

#### `load`

Loads a Sample from a file.

```python
@classmethod
def load(cls, path, format='pickle')
```

**Parameters:**
- **path** (`str`): Path to load the file from
- **format** (`str`, optional): Format to load from ('pickle', 'csv', 'json'). Default is 'pickle'.

**Returns:**
- `Sample`: Loaded Sample

### Machine Learning Methods

#### `create_lstm_dataset`

Create a ready-to-use dataset for LSTM models.

```python
def create_lstm_dataset(self, target_feature=None, lookback=None, forecast=None, 
                       batch_size=32, return_torch=True)
```

**Parameters:**
- **target_feature** (`str`, optional): Target feature to predict
- **lookback** (`int`, optional): Number of historical time steps to use
- **forecast** (`int`, optional): Number of future time steps to predict
- **batch_size** (`int`, optional): Batch size for the dataset. Default is 32.
- **return_torch** (`bool`, optional): If True, returns PyTorch tensors. Default is True.

**Returns:**
- `tuple`: (X_tensor/array, y_tensor/array, X_time, y_time)

## Usage Examples

### Creating a Sample

```python
import numpy as np
from lstm_tools import Sample

# Create a Sample from a 2D array with column names
data = np.array([
    [1.0, 2.0, 3.0],  # timeframe 1
    [4.0, 5.0, 6.0],  # timeframe 2
    [7.0, 8.0, 9.0]   # timeframe 3
])
columns = ['temp', 'humidity', 'pressure']
sample = Sample(data, cols=columns)

# Create with a scaler
from sklearn.preprocessing import RobustScaler
sample_scaled = Sample(data, cols=columns, use_scaler=True, scaler=RobustScaler())
```

### Accessing Features

```python
# Access a single timeframe by index
timeframe = sample[0]  # Returns a TimeFrame object

# Access a specific timeframe and feature
first_temp = sample[0][0]  # Method 1
first_temp = sample[0].temp # Method 2
first_temp = sample.temp[0] # Method 3
print(first_temp)  # 1.0

# Access a feature across all timeframes (returns a Features object)
all_temps = sample.temp
print(all_temps)  # Features([1.0, 4.0, 7.0])

# Calculate statistics on a feature
print(sample.feature_mean('temp'))  # 4.0
print(sample.feature_max('pressure'))  # 9.0
```

### Creating Windows

```python
# Configure window settings
sample.window_settings.historical.window_size = 2
sample.window_settings.future.window_size = 2

# Create historical windows
hist_chronicle = sample.historical_sliding_window()
print(hist_chronicle.shape)  # Shows shape based on window size and sample length

# Create future windows
future_chronicle = sample.future_sliding_window()
print(future_chronicle.shape)  # Shows shape based on window size and sample length

# Create both historical and future windows
historical, future = sample.hf_sliding_window()
```

### Working with pandas DataFrames

```python
import pandas as pd

# Example DataFrame
df = pd.DataFrame({
    'temperature': [20.1, 21.5, 22.0],
    'humidity': [45, 48, 51],
    'pressure': [1013, 1015, 1010]
})

# Create Sample from DataFrame
sample_from_df = Sample.from_dataframe(df)

# Access features
print(sample_from_df.temperature)  # Features([20.1, 21.5, 22.0])
print(sample_from_df.humidity)    # Features([45, 48, 51])

# Convert back to DataFrame
df_converted = sample_from_df.to_DataFrame()
print(df_converted)
```

### Creating LSTM Datasets

```python
# Create a dataset for LSTM training
X, y, X_time, y_time = sample.create_lstm_dataset(
    target_feature='temp',
    lookback=10,
    forecast=5,
    batch_size=64,
    return_torch=True
)

# Create a PyTorch DataLoader
import torch
dataset = torch.utils.data.TensorDataset(X, y)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
```

### Visualization

```python
# Create an interactive plot of all features
fig = sample.line_plot()
fig.show()

# Create a plot excluding certain features
fig = sample.line_plot(exclude=['humidity'], opacity=0.8)
fig.show()
``` 