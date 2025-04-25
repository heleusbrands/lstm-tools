import numpy as np
import pandas as pd
from line_profiler import profile
from .base import FrameBase
from .utils import TradeWindowOps
from typing import Union, List, Callable, Any, Optional, TypeVar


class Feature(float):

    def __new__(cls, value: float, name: str, base_dtype = np.float32) -> 'Feature':
        feature = super().__new__(cls, value)
        feature.name = name
        feature._base = base_dtype(value)
        feature.operations = []
        return feature

    def __repr__(self) -> str:
        return f'Feature({self.name}: {self.__float__()})'
    
    def __add__(self, other: Union[float, Callable]) -> Union['Feature', None]:
        if callable(other):
            self.operations.append(other)
            return None
        else:
            return super().__add__(other)


class Features(FrameBase):
    """
    1D Array of Features
    """
    subtype = Feature
    level = 0
    nptype = np.float32
    operations = TradeWindowOps

    @profile
    def __new__(
        cls, 
                input_data, 
                name=None,
                dtype=None,
                time = None,
                compressors = []
                ): #TODO: Change this to no longer be a ndarray subclass, but rather a wrapper that includes an __array__ method
        # Create a new instance of the array

        # Validate input_data
        if input_data is None:
            raise ValueError("input_data cannot be None")
        
        if len(input_data) == 0:
            raise ValueError("input_data cannot be empty")
            
        if not isinstance(input_data, (list, np.ndarray)):
            raise TypeError(f"input_data must be a list or numpy array, got {type(input_data)}")
            
        # Validate compressors
        if not isinstance(compressors, list):
            raise TypeError(f"compressors must be a list, got {type(compressors)}")
            
        for comp in compressors:
            if not callable(comp):
                raise TypeError(f"All compressors must be callable, got {type(comp)}")

        def time_handler(time):
            if isinstance(time, (pd.DatetimeIndex, str, int)): return time
            elif isinstance(time, (list, np.ndarray)): return time

        dtype = np.dtype(dtype) if dtype else np.dtype(cls.nptype)
        
        # Store the original data without creating subclass instances eagerly
        if isinstance(input_data[0], cls.subtype):
            # Extract float values from each Feature
            base_data = np.array([float(f) for f in input_data], dtype=cls.nptype)
        else:
            # Just use the input data directly
            base_data = np.array(input_data, dtype=cls.nptype)
        
        # Create a view of the base data and store the data directly in the view
        obj = np.array(base_data, dtype=dtype).view(cls)
        obj.compressors = compressors
        obj._time = time_handler(time)
        obj.name = name
        obj._shape = base_data.shape
        obj._level = 0
        obj._original_input = input_data  # Store original if needed for reference
        return obj
    
    def __array_finalize__(self, obj):
        if obj is None: return
        self.compressors = getattr(obj, 'compressors', [])
        self._time = getattr(obj, '_time', None)
        self.name = getattr(obj, 'name', None)
        self._shape = getattr(obj, '_shape', None)
        self._level = getattr(obj, '_level', None)
        self._original_input = getattr(obj, '_original_input', None)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return Features(super().__getitem__(item), name=self.name, time=self._time[item], compressors=self.compressors)
        elif isinstance(item, int):
            return self.subtype(super().__getitem__(item), name=self.name)
        else:
            return super().__getitem__(item)
        
    def __add__(self, other):
        if callable(other):
            self.compressors.append(other)
        elif isinstance(other, tuple):
            if isinstance(other[0], str) and callable(other[1]):
                self.add_compressor(other[1], other[0])
            elif isinstance(other[1], str) and callable(other[0]):
                self.add_compressor(other[0], other[1])
            else:
                raise ValueError(f"Invalid compressor tuple: {other}. Must be (name, callable) or (callable, name)")
        else:
            return super().__add__(other)

    def _as_nparray(self):
        """Return the underlying array data."""
        return super().__array__()

    def add_compressor(self, compressor, name=None):
        """
        Add a compression function to be applied when the compress method is called.
        
        Parameters
        ----------
        compressor : callable
            Function to be applied to the Features array.
            Should take a numpy array as input and return a scalar value.
        name : str, optional
            Name to assign to the compressor. If None, will use function's name
            or generate a name for lambdas, by default None.
            
        Returns
        -------
        Features
            Self reference for method chaining
            
        Examples
        --------
        >>> features.add_compressor(np.mean)  # Uses "mean" as name
        >>> features.add_compressor(lambda x: x.max() - x.min(), "range")  # Custom name
        """
        if not callable(compressor):
            raise TypeError(f"Compressor must be callable, got {type(compressor)}")

        # Handle name assignment
        if name:
            # For class methods, we need to create a wrapper to preserve the name
            if isinstance(compressor, (classmethod, staticmethod)):
                original = compressor
                compressor = lambda x, c=compressor: c.__get__(None, type(None))(x)
            compressor.__name__ = name
        elif not hasattr(compressor, '__name__') or compressor.__name__ == '<lambda>':
            compressor.__name__ = f"Compressor_{len(self.compressors)+1}"
            
        self.compressors.append(compressor)
        return self  # Return self for method chaining

    def compress(self):
        from .timeframe import TimeFrame
        # Convert to numpy array before compression
        data = self._as_nparray()
        feats = [comp(data) for comp in self.compressors]
        cols = [comp.__name__ for comp in self.compressors]
        return TimeFrame(feats, cols=cols, time=self._time[0] if self._time else None)
    
    def batch_compress(self, common_operations=True, custom_compressors=None):
        """
        Apply a batch of common compression operations to the Features.
        
        Parameters
        ----------
        common_operations : bool, optional
            Whether to include common operations (mean, std, etc.), by default True
        custom_compressors : list, optional
            List of custom compressor functions to add, by default None
            
        Returns
        -------
        Features
            Self reference for method chaining
            
        Examples
        --------
        >>> features.batch_compress().compress()  # Add common operations and compress
        >>> features.batch_compress(custom_compressors=[my_func]).compress()  # Add custom operations
        """
        if common_operations:
            # Create wrapper functions for class methods
            def wrap_op(op_name):
                op = getattr(self.operations, op_name)
                wrapped = lambda x, o=op: o.__get__(None, type(None))(x)
                wrapped.__name__ = op_name
                return wrapped

            # Add common statistical operations in the expected order
            operations = ["mean", "median", "sum", "std", "min", "max", "skew", "kurtosis", "variance", "first", "last"]
            for op_name in operations:
                self.add_compressor(wrap_op(op_name))
            
        # Add custom compressors if provided
        if custom_compressors:
            for comp in custom_compressors:
                if callable(comp):
                    self.add_compressor(comp)
                elif isinstance(comp, tuple) and len(comp) == 2:
                    self.add_compressor(comp[0], comp[1])
                    
        return self
    
    @property
    def mean(self):
        return self.operations.mean(self._as_nparray())
    
    @property
    def std(self):
        return self.operations.std(self._as_nparray())
    
    @property
    def var(self):
        return self.operations.variance(self._as_nparray())
    
    @property
    def skew(self):
        return self.operations.skew(self._as_nparray())
    
    @property
    def kurtosis(self):
        return self.operations.kurtosis(self._as_nparray())
    
    @property
    def first(self):
        return self.operations.first(self._as_nparray())
    
    @property
    def last(self):
        return self.operations.last(self._as_nparray())
        
    @property
    def sum(self):
        return self.operations.sum(self._as_nparray())
    
    @property
    def min(self):
        return self.operations.min(self._as_nparray())
    
    @property
    def max(self):
        return self.operations.max(self._as_nparray())
    
    @property
    def median(self):
        return self.operations.median(self._as_nparray())
        
