from __future__ import annotations
from .base import FrameBase
from .sample import Sample
from .feature import FeatureChronicle
from .utils import *
from line_profiler import profile
from typing import TYPE_CHECKING, List, Union
import numpy as np
import torch
from tensorflow import convert_to_tensor
from .exceptions import EmptyDataError, InvalidDataTypeError, DataError



class Chronicle(FrameBase):
    """
    A custom array-like object for representing chronicles in time series data.

    This class extends FrameBase to include chronicle specific attributes and methods.

    Parameters
    ----------
    input_data : np.ndarray
        Input data to create the chronicle.
    cols : list
        List of column names for the chronicle.
    idx : int, optional
        Index of the chronicle, by default None.
    name : str, optional
        Name of the chronicle, by default None.
    dtype : numpy.dtype, optional
        Data type of the chronicle, by default None.
    is_gen : bool, optional
        Whether the chronicle is generated, by default False.
    scaler : sklearn.preprocessing.RobustScaler, optional
        Scaler used for the chronicle, by default None.
    preserve_base : bool, optional
        Whether to preserve the base data, by default True.
    time : np.ndarray, optional
        Time values for the chronicle, by default None.

    Attributes
    ----------
    _time : np.ndarray
        Time values for the chronicle.
    """
    
    subtype = Sample
    level = 0
    nptype = np.float32

    """
    3D Array of Samples
    """
    @profile
    def __new__(
        cls, 
        input_data: Union[np.ndarray, list], 
        cols, 
        idx=None, 
        name=None, 
        dtype=None, 
        is_gen = False, 
        scaler = None, 
        preserve_base = True,
        time = None
        ):
        
        # Validate input data
        if not np.any(input_data):
            raise EmptyDataError("input_data cannot be None")
            
        if len(input_data) == 0:
            raise EmptyDataError("input_data cannot be empty")
            
        if not isinstance(input_data, (list, tuple, np.ndarray)):
            raise InvalidDataTypeError(f"input_data must be a list, tuple, or numpy array, got {type(input_data)}")
            
        # Validate cols
        if not cols:
            raise DataError("cols cannot be None")
            
        if not isinstance(cols, (list, tuple)):
            raise InvalidDataTypeError(f"cols must be a list or tuple, got {type(cols)}")
            
        # Create a new instance of the array
        if cls.subtype != Sample: cls.subtype = Sample
        dtype = np.dtype(dtype) if dtype else np.dtype(cls.nptype)
        scaler = scaler or getattr(input_data[0], 'scaler', None) if isinstance(input_data, (list, tuple)) and len(input_data) > 0 else None
        
        # Prepare data to store directly in the array view
        if isinstance(input_data[0], cls.subtype): 
            # Use existing Sample's data
            cls.nptype = input_data[0].nptype
            base_data = np.array([np.array(d).view(np.ndarray) for d in input_data], dtype=cls.nptype)
            
            # Handle time for generated data
            if is_gen and time is not None and len(input_data) > 1:
                if not hasattr(time, 'shape') or len(time.shape) < 2:
                    # Convert time to 2D if it's not already
                    if time is not None:
                        time = np.array([time[i] if i < len(time) else None for i in range(len(input_data))])
        else:
            # Use raw data directly
            base_data = np.array(input_data, dtype=cls.nptype)
        
        # Create a view of the base data and store the data directly in the view
        obj = np.array(base_data, dtype=dtype).view(cls)
        
        
        # Infer shape from the data
        obj._shape = base_data.shape
        obj._time = time    
        obj._cols = cols
        obj._idx = idx
        obj._level = 0
        obj.scaler = scaler
        obj.name = name
        obj.compressors = Storage(**{col: [] for col in cols})
        obj.is_gen = is_gen
        return obj
    
    @profile
    def __getitem__(self, item):
        """
        Get items from the Chronicle by index, feature name, or slice.
        
        This method implements lazy creation of Sample objects only when needed.

        Parameters
        ----------
        item : Union[str, int, slice, tuple]
            If str: Feature name to retrieve
            If int: Index of the chronicle to retrieve
            If slice: Range of chronicles to retrieve
            If tuple: For multi-dimensional indexing

        Returns
        -------
        Union[Sample, np.ndarray]
            Chronicle data based on the input type.
        """
        from .feature import FeatureSample, FeatureChronicle
        if isinstance(item, str):
            # Get by column name
            idx = self._cols.index(item)
            return FeatureChronicle(np.array(self).view(np.ndarray)[:, :, idx], name=item, time=self._time, compressors = self.compressors[item]) if len(self._shape) == 3 else FeatureSample(np.array(self).view(np.ndarray)[:, idx], name=item, time=self._time)
        elif isinstance(item, int):
            # Get by numeric index - return a Sample
            if not (0 <= item < len(self)):
                raise IndexError(f"Index {item} out of bounds for Chronicle with {len(self)} items")
            
            # Create a Sample only when explicitly requested by index
            if len(self._shape) == 3:
                sample_data = np.array(self).view(np.ndarray)[item]
            else:
                # Handle special case for non-3D data
                sample_data = np.array(self).view(np.ndarray)[item:item+1]
                
            time_value = self._time[item] if self._time is not None and hasattr(self._time, '__getitem__') else None
            return self.subtype(sample_data, self._cols, idx=item, time=time_value, scaler=self.scaler)
        elif isinstance(item, tuple):
            # Handle multi-dimensional indexing
            return np.array(self).view(np.ndarray)[item]
        else:
            # Slice handling
            raw_slice = np.array(self).view(np.ndarray)[item]
            if self._time is not None and hasattr(self._time, '__getitem__'):
                if isinstance(item, slice):
                    start = item.start or 0
                    stop = item.stop or len(self._time)
                    step = item.step or 1
                    time_slice = self._time[start:stop:step]
                else:
                    time_slice = self._time[item]
            else:
                time_slice = self._time
                
            # Return a new Chronicle for multi-element slices
            if isinstance(raw_slice, np.ndarray) and len(raw_slice.shape) > 0 and raw_slice.shape[0] > 0:
                return type(self)(raw_slice, self._cols, time=time_slice, scaler=self.scaler)
            
            return raw_slice
    
    def __getattribute__(self, name: str):
        # First get access to the _cols attribute from the base object
        # to avoid recursion when accessing self._cols
        try:
            cols = super().__getattribute__('_cols')
            if cols is not None and name in cols:
                # Get the index using the super implementation to avoid recursion
                arr = super().__getattribute__('__array__')()
                idx = cols.index(name)
                return FeatureChronicle(arr[:,:, idx], name=name, time=self._time, compressors = self.compressors[name])
        except (AttributeError, ValueError):
            pass
        return super().__getattribute__(name)

    def __array__(self):
        """Return the underlying array data."""
        return np.array(self).view(np.ndarray)
        
    @profile
    def __array_finalize__(self, obj):
        """
        Finalize the array creation process.
        
        Parameters
        ----------
        obj : object
            Object to finalize.
        """
        if obj is None: return
        self._time = getattr(obj, '_time', None)
        self._shape = getattr(obj, '_shape', None)
        self._cols = getattr(obj, '_cols', None)
        self._idx = getattr(obj, '_idx', None)
        self._level = getattr(obj, '_level', None)
        self.compressors = getattr(obj, 'compressors', None)
        self.scaler = getattr(obj, 'scaler', None)
        self.name = getattr(obj, 'name', None)
        self.is_gen = getattr(obj, 'is_gen', False)

    @property
    def features(self):
        return [self[c] for c in self._cols]

    @classmethod
    @profile
    def merge_samples_to_chronicle(cls, samples: List['Sample']) -> 'Chronicle':
        """
        Merge a list of Sample instances into a Chronicle by combining their TimeFrames.

        Parameters
        ----------
        samples : List[Sample]
            List of Sample instances to merge. All samples must have the same length and time values.

        Returns
        -------
        Chronicle
            Chronicle instance containing the merged samples.

        Raises
        ------
        ValueError
            If samples have different lengths or time values.
        """
        if not samples:
            raise ValueError("Cannot merge empty list of samples")
            
        # Validate samples have same length and time values
        base_length = len(samples[0])
        base_time = samples[0]._time
        for sample in samples[1:]:
            if len(sample) != base_length:
                raise ValueError(f"All samples must have same length. Found lengths {base_length} and {len(sample)}")
            if not np.array_equal(sample._time, base_time):
                raise ValueError("All samples must have same time values")
                
        # Stack the base arrays to create Chronicle's base data
        base_data = np.stack([np.array(sample).view(np.ndarray) for sample in samples])
        
        # Use time from first sample since they're all the same
        time = samples[0]._time
        
        # Use cols from first sample
        cols = ['_'.join(c.split('_')) for c in samples[0]._cols] if '_' in samples[0]._cols[0] else samples[0]._cols
        
        # Use scaler from first sample if it exists
        scaler = getattr(samples[0], 'scaler', None)
        
        return cls(base_data, cols=cols, time=time, scaler=scaler)

    @profile
    def to_ptTensor(self, device = 'cpu'):
        """
        Convert the Chronicle data to a PyTorch tensor.

        Parameters
        ----------
        device : str or torch.device, optional
            Device to place the tensor on, by default 'cpu'.

        Returns
        -------
        torch.Tensor
            Chronicle data as a PyTorch tensor on the specified device.
        """
        return torch.from_numpy(np.array(self).view(np.ndarray)).to(device)

    @profile
    def to_tfTensor(self, device = 'cpu'):
        """
        Convert the Chronicle data to a TensorFlow tensor.

        Parameters
        ----------
        device : str or tf.device, optional
            Device to place the tensor on, by default 'cpu'.

        Returns
        -------
        tf.Tensor
            Chronicle data as a TensorFlow tensor on the specified device.
        """
        return convert_to_tensor(np.array(self).view(np.ndarray)).to(device)

    @profile
    def to_tensor(self, device = 'cpu'):
        """
        Convert the Chronicle data to a PyTorch tensor.

        Parameters
        ----------
        device : str or torch.device, optional
            Device to place the tensor on, by default 'cpu'.

        Returns
        -------
        torch.Tensor
            Chronicle data as a PyTorch tensor on the specified device.
        """
        return torch.from_numpy(np.array(self).view(np.ndarray)).to(device)

    @profile
    def xy_dataset(self, x: np.ndarray, y: np.ndarray, future_size, historical_size, step_size=1):
        """
        Create an input-output dataset from the Chronicle data.

        Parameters
        ----------
        x : np.ndarray
            Input data array.
        y : np.ndarray
            Output data array.
        future_size : int
            Size of the future window.
        historical_size : int
            Size of the historical window.
        step_size : int, optional
            Step size between consecutive windows, by default 1.

        Returns
        -------
        tuple
            Tuple containing (x_windows, y_windows) as numpy arrays.
        """
        x_windows = sliding_window(x[:len(x)-future_size], historical_size, step_size)
        y_windows = sliding_window(y[historical_size:], future_size, step_size)
        return x_windows, y_windows

    @profile
    def batch(self, y: 'Chronicle', batch_size):
        """
        Get a batch of data from the Chronicle.
        
        Parameters
        ----------
        y : Chronicle
            Output Chronicle.
        batch_size : int
            Size of the batch.
            
        Returns
        -------
        tuple
            Tuple containing (batch, y_batch).
        """
        batch_data = np.array(self).view(np.ndarray)[0:0+batch_size,:, :]
        y_data = np.array(y).view(np.ndarray).squeeze()
        y_data = y_data[0:0+batch_size]
        return batch_data, y_data

    @profile
    def subwindow_over_samples(self, window_size, direction='backward'):
        """
        Create a subwindow view of the Chronicle across all samples.

        Parameters
        ----------
        window_size : int
            Size of the window to create.
        direction : str, optional
            Direction to create the window, either 'forward' or 'backward',
            by default 'backward'.

        Returns
        -------
        Chronicle
            A new Chronicle instance containing the subwindow view.
        """
        data = np.array(self).view(np.ndarray)
        subwin = data[:, 0:window_size, :] if direction == 'forward' else data[:, -window_size:, :]
        time = self._time[0:window_size] if direction == 'forward' else self._time[-window_size:]
        return Chronicle(subwin, cols=self._cols, is_gen=False, time=time, scaler=self.scaler)
    
    @profile
    def compress(self, feature: str, method: callable):
        """
        Compress a feature using a method.
        
        This now uses lazy instantiation of Sample objects to improve performance.

        Parameters
        ----------
        feature : str
            Feature to compress.
        method : callable
            Method to use for compression.

        Returns
        -------
        np.ndarray
            Compressed feature data.
        """
        # Create a list to hold the results
        results = []
        data = np.array(self).view(np.ndarray)
        
        # Process each sample without eagerly creating Sample objects
        for i in range(len(data)):
            # Get the data for the feature
            if len(data.shape) == 3:
                feature_idx = self._cols.index(feature) if isinstance(feature, str) else feature
                feature_data = data[i, :, feature_idx]
            else:
                feature_idx = self._cols.index(feature) if isinstance(feature, str) else feature
                feature_data = data[i, feature_idx]
            
            # Apply the method directly to the feature data
            results.append(method(feature_data))
        
        return np.array(results)
    
    def compress_all_features(self):
        compressed_features = [f.compress() for f in self.features]
        return compressed_features

    @profile
    def batch_compress(self, features=None, methods=None):
        """
        Compress multiple features using multiple methods.

        This method allows you to apply multiple compression methods to multiple
        features in a single call, returning the results as a dictionary.

        Parameters
        ----------
        features : list or None, optional
            List of feature names or indices to compress. If None, uses all features.
            By default None.
        methods : dict or None, optional
            Dictionary mapping method names to callable functions. If None, uses
            standard statistical methods (mean, std, min, max).
            By default None.

        Returns
        -------
        dict
            Dictionary where keys are '{feature_name}_{method_name}' and values
            are the compressed results.

        Examples
        --------
        >>> chronicle = sample.historical_sliding_window()
        >>> # Compress all features with default methods
        >>> compressed = chronicle.batch_compress()
        >>> # Compress specific features with specific methods
        >>> compressed = chronicle.batch_compress(
        ...     features=['price', 'volume'],
        ...     methods={'mean': np.mean, 'range': lambda x: np.max(x) - np.min(x)}
        ... )
        """
        from .utils import TradeWindowOps
        
        # Default methods if none provided
        if methods is None:
            methods = {
                'mean': TradeWindowOps.mean,
                'std': TradeWindowOps.std,
                'min': TradeWindowOps.min,
                'max': TradeWindowOps.max
            }
            
        # Default to all features if none specified
        if features is None:
            features = self._cols
            
        # Create result dictionary
        results = {}
        
        # Apply each method to each feature
        for feature in features:
            feature_name = feature if isinstance(feature, str) else self._cols[feature]
            for method_name, method in methods.items():
                result_key = f"{feature_name}_{method_name}"
                results[result_key] = self.compress(feature, method)
                
        return results