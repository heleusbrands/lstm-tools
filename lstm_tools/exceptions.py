"""
Custom exceptions for the LSTM Tools library.

This module contains custom exceptions that provide more specific error information
than generic Python exceptions.
"""

class LSTMToolsError(Exception):
    """Base exception for all LSTM Tools errors."""
    pass

class DataError(LSTMToolsError):
    """Raised when there is an issue with the data format or content."""
    pass

class EmptyDataError(DataError):
    """Raised when an operation is attempted on empty data."""
    pass

class InvalidDataTypeError(DataError):
    """Raised when data of an invalid type is provided."""
    pass

class DimensionError(DataError):
    """Raised when data dimensions do not match expectations."""
    pass

class FeatureError(LSTMToolsError):
    """Raised for issues specific to Feature objects."""
    pass

class FeatureNotFoundError(FeatureError):
    """Raised when a requested feature name is not found."""
    pass

class WindowingError(LSTMToolsError):
    """Raised for issues with windowing operations."""
    pass

class InvalidWindowSizeError(WindowingError):
    """Raised when an invalid window size is specified."""
    pass

class SerializationError(LSTMToolsError):
    """Raised when there is an issue with serialization or deserialization."""
    pass

class CompressionError(LSTMToolsError):
    """Raised when there is an issue with data compression."""
    pass 