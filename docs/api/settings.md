# Settings

The `settings` module provides configuration classes for window operations in time series analysis, including window size, stride, and window type settings.

## WindowSettings

`WindowSettings` is a configuration class for window operations.

### Class Definition

```python
class WindowSettings:
    def __init__(self, window_size=10, window_type=WindowType.CENTERED)
```

### Parameters

- **window_size** (`int`): Size of the window. Default is 10.
- **window_type** (`WindowType`): Type of the window. Default is `WindowType.CENTERED`.

### Attributes

- **window_size** (`int`): Size of the window
- **window_type** (`WindowType`): Type of the window
- **centered** (`bool`): Whether the window is centered
- **causal** (`bool`): Whether the window is causal
- **future** (`bool`): Whether the window is future-oriented

### Properties

- **centered**: Returns True if window_type is CENTERED
- **causal**: Returns True if window_type is CAUSAL
- **future**: Returns True if window_type is FUTURE

### Methods

#### `validate_window_size`

Validates the window size.

```python
def validate_window_size(self, window_size)
```

**Parameters:**

- **window_size** (`int`): Window size to validate

**Raises:**

- `ValueError`: If window size is not positive

**Example:**

```python
>>> settings = WindowSettings(window_size=10)
>>> try:
...     settings.validate_window_size(0)
... except ValueError as e:
...     print(str(e))
Window size must be positive
```

### Usage Examples

```python
from lstm_tools.settings import WindowSettings, WindowType

# Create window settings with default values
settings = WindowSettings()
print(settings.window_size)  # 10
print(settings.window_type)  # WindowType.CENTERED

# Create settings for causal windows (looking backwards only)
causal_settings = WindowSettings(window_size=30, window_type=WindowType.CAUSAL)
print(causal_settings.centered)  # False
print(causal_settings.causal)    # True
print(causal_settings.future)    # False

# Update window size
causal_settings.window_size = 20
```

## HFWindowSettings

`HFWindowSettings` extends the configuration with stride and separate settings for historical and future windows.

### Class Definition

```python
class HFWindowSettings:
    def __init__(self, historical_window_size=10, future_window_size=10, 
                 stride=1, historical_window_type=WindowType.CAUSAL, 
                 future_window_type=WindowType.FUTURE)
```

### Parameters

- **historical_window_size** (`int`): Size of the historical window. Default is 10.
- **future_window_size** (`int`): Size of the future window. Default is 10.
- **stride** (`int`): Stride (step size) between consecutive windows. Default is 1.
- **historical_window_type** (`WindowType`): Type of the historical window. Default is `WindowType.CAUSAL`.
- **future_window_type** (`WindowType`): Type of the future window. Default is `WindowType.FUTURE`.

### Attributes

- **historical** (`WindowSettings`): Settings for historical windows
- **future** (`WindowSettings`): Settings for future windows
- **stride** (`int`): Stride (step size) between consecutive windows

### Methods

#### `validate_stride`

Validates the stride value.

```python
def validate_stride(self, stride)
```

**Parameters:**

- **stride** (`int`): Stride value to validate

**Raises:**

- `ValueError`: If stride is not positive

**Example:**

```python
>>> settings = HFWindowSettings(stride=2)
>>> try:
...     settings.validate_stride(0)
... except ValueError as e:
...     print(str(e))
Stride must be positive
```

### Usage Examples

```python
from lstm_tools.settings import HFWindowSettings, WindowType

# Create with default values
settings = HFWindowSettings()
print(settings.historical.window_size)  # 10
print(settings.future.window_size)      # 10
print(settings.stride)                  # 1

# Configure for LSTM prediction (30 steps back to predict 5 ahead)
settings.historical.window_size = 30
settings.future.window_size = 5
settings.stride = 1

# Access or modify individual WindowSettings
print(settings.historical.causal)  # True
print(settings.future.future)      # True

# Configure window types if needed
settings.historical.window_type = WindowType.CENTERED
settings.future.window_type = WindowType.CENTERED
```

## WindowType

An enumeration defining the different types of windows.

### Class Definition

```python
class WindowType(Enum):
    CENTERED = 0
    CAUSAL = 1  
    FUTURE = 2  
```

### Values

- **CENTERED** (`0`): Window is centered around the current time step
- **CAUSAL** (`1`): Window only includes past time steps (historical)
- **FUTURE** (`2`): Window only includes future time steps

### Usage Examples

```python
from lstm_tools.settings import WindowType, WindowSettings

# Create different types of windows
centered = WindowSettings(window_type=WindowType.CENTERED)
causal = WindowSettings(window_type=WindowType.CAUSAL)
future = WindowSettings(window_type=WindowType.FUTURE)

# Check window type properties
print(centered.centered)  # True
print(causal.causal)      # True
print(future.future)      # True
``` 