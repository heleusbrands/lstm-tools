import numpy as np
import pandas as pd
from typing import List, Tuple, Union
from .base import FrameBase
from .feature import Feature
from .utils import *
from line_profiler import profile
import torch
from tensorflow import convert_to_tensor
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .sample import Sample


class TimeFrame(FrameBase):
    """
    A custom array-like object for representing time frames in time series data.

    This class extends FrameBase to include time frame specific attributes and methods.

    Parameters
    ----------
    input_data : np.ndarray
        Input data to create the time frame.
    cols : list
        List of column names for the time frame.
    idx : int, optional
        Index of the time frame, by default None.
    name : str, optional
        Name of the time frame, by default None.
    dtype : numpy.dtype, optional
        Data type of the time frame, by default None.
    time : np.ndarray, optional
        Time values for the time frame, by default None.

    Attributes
    ----------
    _time : np.ndarray
        Time values for the time frame.
    """
    subtype = Feature
    level = 0
    nptype = np.float32

    """
    1D Array of FeatureSample
    """

    @profile
    def __new__(
        cls, 
                input_data, 
                cols, 
                idx=None, 
                name=None,
                dtype=None,
                time = None,
                source: FrameBase = None
                ): #TODO: Change this to no longer be a ndarray subclass, but rather a wrapper that includes an __array__ method
        # Create a new instance of the array

        def time_handler(time):
            if isinstance(time, (pd.DatetimeIndex, str, int)): return time
            elif isinstance(time, (list, np.ndarray)): return time[idx]

        dtype = np.dtype(dtype) if dtype else np.dtype(cls.nptype)
        
        # Store the original data without creating subclass instances eagerly
        if isinstance(input_data[0], cls.subtype):
            # Extract _base from each Feature
            base_data = np.array([f._base for f in input_data], dtype=cls.nptype)
        elif isinstance(input_data, np.ndarray):
            base_data = input_data
            cls.nptype = base_data.dtype
        else:
            # Just use the input data directly
            base_data = np.array(input_data, dtype=cls.nptype)

        # Create a view of the base data and store the data directly in the view
        obj = base_data.view(cls)
        obj._time = time_handler(time)
        obj._shape = base_data.shape
        obj._cols = cols
        obj._idx = idx
        obj._level = 0
        obj._original_input = input_data  # Store original if needed for reference

        if np.any(source): cls.__load_metadata__(source, obj)
        return obj

    @profile
    def __getitem__(self, item):
        """
        Get items from the TimeFrame by index, feature name, or slice.

        This method implements lazy creation of Feature objects only when needed.

        Parameters
        ----------
                item : Union[str, int, slice]
                    If str: Feature name to retrieve
                    If int: Index of the feature to retrieve
                    If slice: Range of features to retrieve

        Returns
        -------
        Union[Feature, np.ndarray]
            Feature data based on the input type.
        """
        if isinstance(item, str):
            # Get by column name
            idx = self._cols.index(item)
            return self.subtype(self[idx], name=item)
        elif isinstance(item, int):
            # Get by numeric index - create a Feature only when explicitly requested
            if not (0 <= item < len(self)):
                raise IndexError(f"Index {item} out of bounds for TimeFrame with {len(self)} items")
            return self.subtype(np.array(self).view(np.ndarray)[item], name=self._cols[item] if item < len(self._cols) else None)
        elif isinstance(item, tuple) and len(item) == 2:
            # Handle 2D indexing - typical for 2D array access
            return np.array(self).view(np.ndarray)[item]
        else:
            # For slices, return a new TimeFrame with the sliced data
            raw_slice = np.array(self).view(np.ndarray)[item]
            
            # If this is a single element, return a Feature
            if isinstance(raw_slice, np.ndarray) and len(raw_slice.shape) == 0:
                return self.subtype(raw_slice, name=self._cols[item] if hasattr(self._cols, '__getitem__') else self._cols)
            
            # For multi-element slices, create a new TimeFrame
            if isinstance(raw_slice, np.ndarray) and len(raw_slice.shape) > 0:
                # Get the column subset if applicable
                if isinstance(item, slice):
                    cols_subset = self._cols[item] if hasattr(self._cols, '__getitem__') else self._cols
                    time_subset = self._time[item] if hasattr(self._time, '__getitem__') else self._time
                else:
                    cols_subset = self._cols
                    time_subset = self._time
                
                return type(self)(raw_slice, cols_subset, time=time_subset)
            
            # For other cases, just return the raw array
            return raw_slice

    def __array__(self):
        """Return the underlying array data."""
        return self.view(np.ndarray)

    @profile
    def __array_finalize__(self, obj):
        """
        Finalize the array creation process.
        
        Parameters
        ----------
        obj : object
            Object to finalize.
        """
        from .sample import Sample
        if obj is None: return
        self._time = getattr(obj, '_time', None)
        self._shape = getattr(obj, '_shape', None)
        self._cols = getattr(obj, '_cols', None)
        self._idx = getattr(obj, '_idx', None)
        self._level = getattr(obj, '_level', None)
        self._original_input = getattr(obj, '_original_input', None)

    def __getattribute__(self, name: str) -> profile:
        # First get access to the _cols attribute from the base object
        # to avoid recursion when accessing self._cols
        try:
            cols = super().__getattribute__('_cols')
            if cols is not None and name in cols:
                # Get the index using the super implementation to avoid recursion
                arr = super().__getattribute__('__array__')()
                idx = cols.index(name)
                subtype = super().__getattribute__('subtype')
                return subtype(arr[idx], name=name)
        except (AttributeError, ValueError):
            pass
        return super().__getattribute__(name)

    @property
    def feature_names(self): return self._cols

    @property
    def time(self): return self._time

    def to_numpy(self):
        return self.__array__()

    @profile
    def to_ptTensor(self, device = 'cpu'):
        """
        Convert the TimeFrame data to a PyTorch tensor.

        Parameters
        ----------
        device : str or torch.device, optional
            Device to place the tensor on, by default 'cpu'.

        Returns
        -------
        torch.Tensor
            TimeFrame data as a PyTorch tensor on the specified device.
        """
        return torch.from_numpy(self).to(device)

    @profile
    def to_tfTensor(self, device = 'cpu'):
        """
        Convert the TimeFrame data to a TensorFlow tensor.

        Parameters
        ----------
        device : str or tf.device, optional
            Device to place the tensor on, by default 'cpu'.

        Returns
        -------
        tf.Tensor
            TimeFrame data as a TensorFlow tensor on the specified device.
        """
        return convert_to_tensor(self).to(device)

    @profile
    def to_DataFrame(self):
        """
        Convert the TimeFrame data to a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            TimeFrame data as a pandas DataFrame.
        """
        return pd.DataFrame(self, columns=self._cols)

    @profile
    def window(self, idx: int):
        """
        Get a window of the TimeFrame.

        Parameters
        ----------
        idx : int
            Index of the window.

        Returns
        -------
        TimeFrame
            Window of the TimeFrame.
        """
        ws = self.window_settings.future.window_size
        return self[idx:idx+ws]

