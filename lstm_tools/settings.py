from dataclasses import dataclass, field
from .base import WindowType
from .exceptions import InvalidWindowSizeError


@dataclass
class WindowSettings:
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
    window_size: int = 10  # Changed from 60 to a more reasonable default
    
    def __post_init__(self):
        self.validate_window_size()
        
    def validate_window_size(self):
        """Validate that window size is positive."""
        if self.window_size <= 0:
            raise InvalidWindowSizeError(f"Window size must be positive, got {self.window_size}")
            
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
    historical: WindowSettings = field(default_factory=lambda: WindowSettings(window_type=WindowType.historical))
    future: WindowSettings = field(default_factory=lambda: WindowSettings(window_type=WindowType.future, window_size=1))
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