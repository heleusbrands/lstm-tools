from dataclasses import dataclass, field
from .base import WindowType
from .exceptions import InvalidWindowSizeError
import numpy as np

@dataclass
class FutureWindowSettings:
    """
    Configuration settings for window operations in time series analysis.

    Parameters
    ----------
    window_type : WindowType
        Type of the window (future or historical).
    window_size : int
        Size of the window in time steps.
    """
    window_type: WindowType = WindowType.future
    size: int = 10  # Changed from 60 to a more reasonable default
    offset: int = 0
    
    def __post_init__(self):
        self.validate_window_size()
        
    def validate_window_size(self):
        """Validate that window size is positive."""
        if self.size <= 0:
            raise InvalidWindowSizeError(f"Window size must be positive, got {self.size}")
            
    def __setattr__(self, name, value):
        """Custom attribute setter with validation."""
        # Call the default setter
        object.__setattr__(self, name, value)
        
        # Validate window_size when it changes
        if name == 'window_size':
            self.validate_window_size()

@dataclass
class HistoricalWindowSettings:
    """
    Configuration settings for window operations in time series analysis.

    Parameters
    ----------
    window_type : WindowType
        Type of the window (future or historical).
    window_size : int
        Size of the window in time steps.
    """
    window_type: WindowType = WindowType.historical
    size: int = 10  # Changed from 60 to a more reasonable default
    spacing: int = 1
    
    def __post_init__(self):
        self.validate_window_size()
        
    def validate_window_size(self):
        """Validate that window size is positive."""
        if self.size <= 0:
            raise InvalidWindowSizeError(f"Window size must be positive, got {self.size}")
            
    def __setattr__(self, name, value):
        """Custom attribute setter with validation."""
        # Call the default setter
        object.__setattr__(self, name, value)
        
        # Validate window_size when it changes
        if name == 'window_size':
            self.validate_window_size()


@dataclass
class HFWindowSettings:
    """
    Configuration settings for historical-future window operations.

    Parameters
    ----------
    historical : WindowSettings
        Settings for the historical window.
    future : WindowSettings
        Settings for the future window.
    stride : int
        Step size between consecutive windows.
    """
    historical: HistoricalWindowSettings = field(default_factory=lambda: HistoricalWindowSettings(window_type=WindowType.historical))
    future: FutureWindowSettings = field(default_factory=lambda: FutureWindowSettings(window_type=WindowType.future, size=1))
    stride: int = 1
    
    def __post_init__(self):
        self.validate_stride()
    
    def validate_stride(self):
        """Validate that stride is positive."""
        if self.stride <= 0:
            raise InvalidWindowSizeError(f"Stride must be positive, got {self.stride}")
    
    def __setattr__(self, name, value):
        """Custom attribute setter with validation."""
        # Call the default setter
        object.__setattr__(self, name, value)
        
        # Validate stride when it changes
        if name == 'stride':
            self.validate_stride()

class ExpandingWindowSettings:

    def __init__(self, window_sizes: list[int] = [], window_type: WindowType = WindowType.historical):
        self.window_sizes = window_sizes
        self.window_type = window_type

    def __add__(self, other):
        if isinstance(other, int):
            self.window_sizes.append(other)
        elif isinstance(other, list):
            self.window_sizes.extend(other)
        else:
            raise ValueError(f"Cannot add {type(other)} to ExpandingWindowSettings")
        return self
    
    def __iter__(self):
        return iter(self.window_sizes)
    
    def __next__(self):
        return next(self.window_sizes)
    
    def __len__(self):
        return len(self.window_sizes)
    
    def __getitem__(self, index):
        return self.window_sizes[index]
    
    def __setitem__(self, index, value):
        self.window_sizes[index] = value
    
    def __delitem__(self, index):
        del self.window_sizes[index]
    
    def __contains__(self, item):
        return item in self.window_sizes
    
    def _get_window(self, window_size: int, array: np.ndarray):
        if self.window_type == WindowType.historical:
            return array[-window_size:]
        else:
            return array[:window_size]
        
    def _get_window_array(self, window_size: int, array: np.ndarray):
        if self.window_type == WindowType.historical:
            return array[:, -window_size:]
        else:
            return array[:, :window_size]
    
    def _get_windows(self, array: np.ndarray):
        return [self._get_window(window_size, array) for window_size in self.window_sizes]
    
    def _get_window_arrays(self, array: np.ndarray):
        return [self._get_window_array(window_size, array) for window_size in self.window_sizes]
    
    def __call__(self, array: np.ndarray):
        if len(array.shape) == 1:
            return self._get_windows(array)
        else:
            return self._get_window_arrays(array)
    
    
    
    
    
