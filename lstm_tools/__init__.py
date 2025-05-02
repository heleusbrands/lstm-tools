"""LSTM Tools - A library for handling sequential data efficiently.

LSTM Tools provides custom numpy array subclasses designed for time series data,
with a focus on easy windowing, feature compression, and machine learning integration.

Main Classes:
- Feature: A float subclass representing an individual data point
- FeatureSample: A 1D array of Feature objects for a single time series
- TimeFrame: A 1D array of Feature objects for a point in time
- Sample: A 2D array of TimeFrame objects for a sequence of observations
- Chronicle: A 3D array of windowed samples for ML model training
"""

__version__ = "0.1.0"
__author__ = "Rose Bloom Research Co"
__email__ = "rosebloomresearch@gmail.com"
__license__ = "GPL-3.0-only"

# Core classes
from .timeframe import TimeFrame
from .sample import Sample
from .feature import Feature, FeatureSample
from .chronicle import Chronicle

# Settings and configuration
from .settings import HFWindowSettings, WindowSettings, WindowType
from .logger import configure_logging, logger, debug, info, warning, error, critical

# Exception classes
from .exceptions import (
    LSTMToolsError, DataError, EmptyDataError, InvalidDataTypeError,
    DimensionError, FeatureError, FeatureNotFoundError, WindowingError,
    InvalidWindowSizeError, SerializationError, CompressionError
)
