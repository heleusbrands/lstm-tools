from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Tuple, Union
from .base import FrameBase
from .feature import FeatureSample
from .timeframe import TimeFrame
from .utils import *
from .settings import HFWindowSettings
from line_profiler import profile
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import torch
from tensorflow import convert_to_tensor
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .chronicle import Chronicle

class Sample(FrameBase):
    """
    A custom array-like object for representing samples in time series data.

    This class extends FrameBase to include sample specific attributes and methods.

    Parameters
    ----------
    input_data : np.ndarray
        Input data to create the sample.
    cols : list
        List of column names for the sample.
    idx : int, optional
        Index of the sample, by default None.
    name : str, optional
        Name of the sample, by default None.
    dtype : numpy.dtype, optional
        Data type of the sample, by default None.
    use_scaler : bool, optional
        Whether to use a scaler for the sample, by default False.
    scaler : sklearn.preprocessing.RobustScaler, optional
        Scaler to use for the sample, by default None.
    time : np.ndarray, optional
        Time values for the sample, by default None.

    Attributes
    ----------
    _time : np.ndarray
        Time values for the sample.
    scaler : sklearn.preprocessing.RobustScaler
        Scaler used for the sample.
    """
    subtype = TimeFrame
    level = 0
    scaler_type = RobustScaler
    nptype = np.float32
    scaler = RobustScaler(copy=False)

    """
    2D Array of TimeFrames
    """

    @profile
    def __new__(
        cls, 
        input_data: Union[np.ndarray, list, str, pd.DataFrame], 
        cols=None, 
        idx=None, 
        name=None, 
        df_names: list = None,
        dtype=None,
        use_scaler: bool = False,
        scaler = None,
        time = None,
        source: FrameBase = None
        ):

        def scaler_handler(input_data, scaler):
            if not scaler: scaler = cls.scaler_type()
            input_data = scaler.fit_transform(input_data)
            return input_data, scaler

        def time_handler(time):
            if not isinstance(time, np.ndarray):
                time = np.array(time) 
            if len(time.shape) == 1:
                return time
            elif len(time.shape) == 2:
                return time[idx]
            
        def scale_check(scaler, feature_names):
            if not scaler: return False
            try:
                center = scaler.center_
                scale = scaler.scale_
                if len(center) != len(feature_names) or len(scale) != len(feature_names):
                    return False
                return True
            except:
                return False

        if isinstance(time, np.ndarray):
            time_exists = bool(np.any(time))
        elif isinstance(time, pd.DatetimeIndex):
            time_exists = not time.empty
        else:
            time_exists = bool(time)
        
        time = time_handler(time) if time_exists else None

        if isinstance(input_data, pd.DataFrame): input_data = input_data.values # Convert DataFrame to array
        elif isinstance(input_data, str) and '.csv' in input_data: # Read in a single csv file
            try:
                df = pd.read_csv(input_data, parse_dates=['time'], index_col='time')
                if not time_exists: time = np.array(df.index) if isinstance(df.index, pd.DatetimeIndex) else df.index
                input_data = df.values # Update input_data
                cols = list(df.columns) # Update cols
            except Exception as e:
                try:
                    print(f"Couldn't find time column. Trying Index.")
                    df = pd.read_csv(input_data)
                    print(f"Loaded Dataframe. Index Type: {type(df.index)} | {type(df.index[0])} | {df.index[0]}")
                    if not time_exists: time = np.array(df.index) if isinstance(df.index, pd.DatetimeIndex) else df.index
                    if isinstance(time[0], str): time = [float(date_converter(t)) for t in time]
                    input_data = df.values # Update input_data
                    cols = list(df.columns) # Update cols
                except Exception as e:
                    raise ValueError(f"Failed to read CSV file: {e}")
                    
        elif isinstance(input_data, list) and isinstance(input_data[0], str) and '.csv' in input_data[0]: # Handle a list of csv files
            names = df_names or [str(x) for x in range(len(input_data))] # Create list of file names
            df = merge_csv(names, input_data) # Merge csv files
            if not time_exists: time = df.index # Update time
            input_data = df.values # Update input_data
            cols = list(df.columns) # Update cols
        elif isinstance(input_data, np.ndarray) and len(input_data.shape) == 2: # Handle a 2D array
            input_data = input_data # Update input_data
        elif isinstance(input_data, list) and isinstance(input_data[0], cls.subtype): # Handle a list of TimeFrames
            if not time_exists: time = np.array([t._time for t in input_data])
            if not cols: cols = input_data[0]._cols
            input_data = np.array(input_data)
        else: # Handle other input types
            raise ValueError(f'Unsupported input type: {type(input_data)}')

        if cls.subtype != TimeFrame: cls.subtype = TimeFrame # Update subtype

        if not scaler and not np.any(source): scaler = cls.scaler

        if use_scaler: # Scaler Handling
            input_data, scaler = scaler_handler(input_data, scaler) # Scale input_data

        # Prepare final data to use in the array view
        dtype = np.dtype(dtype) if dtype else np.dtype(cls.nptype)
        if isinstance(input_data[0], cls.subtype): 
            # Use existing TimeFrame's data
            cls.nptype = input_data[0].nptype 
            base_data = np.array([d.view(np.ndarray) for d in input_data], dtype=cls.nptype)
        elif isinstance(input_data, np.ndarray):
            cls.nptype = dtype
            base_data = input_data.astype(dtype)
            
        else:
            # Use raw data directly
            base_data = np.array(input_data, dtype=cls.nptype)
        
        if not time_exists: # Update time
            if isinstance(input_data, cls):
                time = input_data._time
            elif hasattr(input_data[0], 'time'):
                time = input_data[0].time

        # Create a view of the base data and store the data directly in the view
        obj = base_data.view(cls)
        obj._time = time
        obj._shape = base_data.shape
        obj._cols = cols
        obj._idx = idx
        obj._level = 0
        obj.scaled = scale_check(scaler, cols)
        obj.scaler = scaler if obj.scaled else RobustScaler(copy=False)
        obj.name = name
        obj.window_settings = HFWindowSettings()

        if np.any(source): cls.__load_metadata__(source, obj)
        return obj
    
    @profile
    def __array_finalize__(self, obj):
        """
        Finalize the array creation process.
        
        Parameters
        ----------
        obj : object
            Object to finalize.
        """
        from .chronicle import Chronicle
        if obj is None: return
        self._time = getattr(obj, '_time', None)
        self._shape = getattr(obj, '_shape', None)
        self._cols = getattr(obj, '_cols', None)
        self._idx = getattr(obj, '_idx', None)
        self._level = getattr(obj, '_level', None)
        self.scaler = getattr(obj, 'scaler', None)
        self.name = getattr(obj, 'name', None)
        self.window_settings = getattr(obj, 'window_settings', HFWindowSettings())
     
    @profile
    def __getitem__(self, item):
        """
        Get items from the Sample by index, feature name, or slice.
        
        This method implements lazy creation of TimeFrame objects only when needed.

        Parameters
        ----------
        item : Union[str, int, slice, tuple]
            If str: Feature name to retrieve
            If int: Index of the sample to retrieve
            If slice: Range of samples to retrieve
            If tuple: For 2D indexing (row, col)

        Returns
        -------
        Union[TimeFrame, np.ndarray]
            Sample data based on the input type.
        """
        if isinstance(item, str):
            # Get by column name
            idx = self._cols.index(item)
            return FeatureSample(self.view(np.ndarray)[:, idx], name=item, time=self._time)
        elif isinstance(item, int):
            # Get by numeric index - return a TimeFrame
            if not (0 <= item < len(self)):
                raise IndexError(f"Index {item} out of bounds for Sample with {len(self)} items")
            # Create a TimeFrame only when explicitly requested by index
            raw_data = self.view(np.ndarray)[item]
            time_value = self._time[item] if self._time is not None and hasattr(self._time, '__getitem__') else None
            return self.subtype(raw_data, self._cols, idx=item, time=time_value)
        elif isinstance(item, tuple) and len(item) == 2:
            # Handle 2D indexing
            if isinstance(item[1], str):
                # Handle column name in second position
                col_idx = self._cols.index(item[1])
                modified_item = (item[0], col_idx)
                return self.view(np.ndarray)[modified_item]
            return self.view(np.ndarray)[item]
        else:
            # Slice handling
            raw_slice = self.view(np.ndarray)[item]
            if isinstance(raw_slice, np.ndarray) and len(raw_slice.shape) > 0:
                time_slice = self._time[item] if self._time is not None and hasattr(self._time, '__getitem__') else self._time
                
                # If it's a 2D slice, return a new Sample
                return type(self)(raw_slice, self._cols, time=time_slice, scaler=self.scaler)
            
            return raw_slice
    
    def __array__(self):
        """Return the underlying array data."""
        return self.view(np.ndarray)
    
    def __getattribute__(self, name: str):
        # First get access to the _cols attribute from the base object
        # to avoid recursion when accessing self._cols
        try:
            cols = super().__getattribute__('_cols')
            if cols is not None and name in cols:
                # Get the index using the super implementation to avoid recursion
                arr = super().__getattribute__('__array__')()
                idx = cols.index(name)
                return FeatureSample(arr[:, idx], name=name, time=self._time)
        except (AttributeError, ValueError):
            pass
        return super().__getattribute__(name)

    def __add__(self, other):
        if isinstance(other, Sample):
            return Sample(
                np.concatenate([self._as_nparray(), other._as_nparray()], axis=1), 
                cols=self._cols.extend(other._cols), 
                time=self._time, 
                scaler=self.scaler
            )
        elif isinstance(other, FeatureSample):
            return Sample(
                np.concatenate([self._as_nparray(), np.expand_dims(other._as_nparray(), axis=0)], axis=1), 
                cols=self._cols.extend(other.name), 
                time=self._time, 
                scaler=self.scaler
            )
        else:
            try:
                super().__add__(other)
            except:
                raise ValueError(f"Cannot add {type(other)} to Sample")

    @property
    def feature_names(self): return self._cols

    @property
    def time(self): return self._time

    def rename_columns(self, new_names: List[str]):
        """
        Rename the columns of the Sample.
        """
        self._cols = new_names
        return self
    
    def scale(self, scaler: RobustScaler = None):
        if not scaler: scaler = self.scaler
        scaler.fit(self.to_numpy())
        return Sample(scaler.transform(self.to_numpy()), cols=self._cols, time=self._time, scaler=scaler, use_scaler=False)
    
    def unscale(self, scaler: RobustScaler = None):
        if not scaler: scaler = self.scaler
        data = scaler.inverse_transform(self.to_numpy())
        return Sample(data, cols=self._cols, time=self._time, scaler=scaler, use_scaler=False)
    
    @classmethod
    def split(cls, historical: 'Chronicle', future: 'Sample', test_size: float = 0.2, random_state: int = 42):
        htype = type(historical)
        ftype = type(future)
        hcols = historical.feature_names
        fcols = future.feature_names
        hscaler = historical.scaler
        fscaler = future.scaler
        htime = historical.time
        ftime = future.time
        historical, test_hist, future, test_fut = train_test_split(historical.to_numpy(), future.to_numpy(), test_size=test_size, random_state=random_state)
        return (
            htype(historical, cols=hcols, time=htime, scaler=hscaler), 
            htype(test_hist, cols=hcols, time=htime, scaler=hscaler), 
            ftype(future, cols=fcols, time=ftime, scaler=fscaler), 
            ftype(test_fut, cols=fcols, time=ftime, scaler=fscaler)
            )

    def to_numpy(self):
        return self.__array__()

    @profile
    def to_ptTensor(self, device = 'cpu'):
        """
        Convert the Sample data to a PyTorch tensor.

        Parameters
        ----------
        device : str or torch.device, optional
            Device to place the tensor on, by default 'cpu'.

        Returns
        -------
        torch.Tensor
            Sample data as a PyTorch tensor on the specified device.
        """
        return torch.from_numpy(self.view(np.ndarray)).to(device)

    @profile
    def to_tfTensor(self, device = 'cpu'):
        """
        Convert the Sample data to a TensorFlow tensor.

        Parameters
        ----------
        device : str or tf.device, optional
            Device to place the tensor on, by default 'cpu'.

        Returns
        -------
        tf.Tensor
            Sample data as a TensorFlow tensor on the specified device.
        """
        return convert_to_tensor(self.view(np.ndarray)).to(device)

    @profile
    def to_DataFrame(self):
        """
        Convert the Sample data to a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            Sample data as a pandas DataFrame.
        """
        cols = self._cols
        df = pd.DataFrame(self.view(np.ndarray), columns=self._cols)
        df.insert(0, 'time', self._time)
        return df

    @profile
    def _time_to_hf_sliding_window(self):
        """
        Create a sliding window view of the time data.

        Returns
        -------
        np.ndarray
            Sliding window view of the time data.
        """
        if not bool(np.any(self._time)): return None
        fsize = self.window_settings.future.size
        hsize = self.window_settings.historical.size
        foffset = self.window_settings.future.offset
        step = self.window_settings.stride
        h, f = hf_sliding_window(self._time, hsize, fsize, foffset, step)
        return h, f

    @profile
    def _time_to_h_sliding_window(self):
        """
        Create a historical sliding window view of the time data.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple containing (historical_windows, future_windows).
        """
        if not bool(np.any(self._time)): return None
        fsize = self.window_settings.future.size
        hsize = self.window_settings.historical.size
        foffset = self.window_settings.future.offset
        hspacing = self.window_settings.historical.spacing
        step = self.window_settings.stride
        h, _ = hf_sliding_window(self._time, hsize, fsize, foffset, hspacing, step)
        return h

    @profile
    def _time_to_f_sliding_window(self):
        """
        Create a future sliding window view of the time data.

        Returns
        -------
        np.ndarray
            Future sliding window view of the time data.
        """
        if not bool(np.any(self._time)): return None
        fsize = self.window_settings.future.size
        hsize = self.window_settings.historical.size
        foffset = self.window_settings.future.offset
        hspacing = self.window_settings.historical.spacing
        step = self.window_settings.stride
        _, f = hf_sliding_window(self._time, hsize, fsize, foffset, hspacing, step)
        return f
    
    @profile
    def future_sliding_window(self) -> 'Chronicle':
        """
        Create a sliding window view of the Sample.

        Returns
        -------
        Chronicle
            Sliding window view of the Sample.
        """
        from .chronicle import Chronicle
        hsize = self.window_settings.historical.size
        fsize = self.window_settings.future.size
        foffset = self.window_settings.future.offset
        hspacing = self.window_settings.historical.spacing
        step = self.window_settings.stride
        data = self.view(np.ndarray)
        _, f = hf_sliding_window(data, hsize, fsize, foffset, hspacing, step)
        return Chronicle(
            f, 
            cols=self._cols, 
            is_gen=True, 
            time = self._time_to_f_sliding_window(), 
            scaler=self.scaler
            )

    @profile
    def historical_sliding_window(self) -> 'Chronicle':
        """
        Create a historical sliding window view of the Sample.

        Returns
        -------
        Chronicle
            Historical sliding window view of the Sample.
        """
        from .chronicle import Chronicle
        hsize = self.window_settings.historical.size
        fsize = self.window_settings.future.size
        foffset = self.window_settings.future.offset
        hspacing = self.window_settings.historical.spacing
        step = self.window_settings.stride
        data = self.view(np.ndarray)
        h, _ = hf_sliding_window(data, hsize, fsize, foffset, hspacing, step)
        return Chronicle(
            h, 
            cols=self._cols, 
            is_gen=True, 
            time = self._time_to_h_sliding_window(), 
            scaler=self.scaler
            )
    
    @profile
    def hf_sliding_window(self) -> Tuple['Chronicle', 'Chronicle']:
        """
        Create historical and future sliding window views of the Sample.

        Returns
        -------
        Tuple[Chronicle, Chronicle]
            Tuple containing (historical_windows, future_windows).
        """
        from .chronicle import Chronicle
        hsize = self.window_settings.historical.size
        fsize = self.window_settings.future.size
        foffset = self.window_settings.future.offset
        hspacing = self.window_settings.historical.spacing
        step = self.window_settings.stride
        data = self.view(np.ndarray)
        ht, ft = self._time_to_hf_sliding_window()
        h, f = hf_sliding_window(data, hsize, fsize, foffset, hspacing, step)
        return (
            Chronicle(h, cols=self._cols, is_gen=True, time=ht, scaler=self.scaler, source=self), 
            Chronicle(f, cols=self._cols, is_gen=True, time=ft, scaler=self.scaler, source=self)
            )
    
    @classmethod
    def join_samples_by_window_size(cls, window_size, samples: list):
        samples = [s for s in samples if s.name.split('_')[1] == str(window_size)]
        cols = np.array([s._cols for s in samples]).flatten().tolist()
        data = np.concatenate([s.view(np.ndarray) for s in samples], axis=1)
        return Sample(data, cols=cols, time=samples[0]._time, scaler=samples[0].scaler)
    
    @classmethod
    def join_samples(cls, samples: list, remove_window_sizes=True):
        cols = np.array([s._cols for s in samples]).flatten().tolist()
        if remove_window_sizes:
            col_parts = np.array([c.split('_') for c in cols]).flatten()
            sizes = {p for p in col_parts if p.isdigit()}
            name = f'compressed_{"".join(sizes)}_window'
            cols = ['_'.join([x for x in c.split('_') if not x.isdigit()]) for c in cols]
        data = np.concatenate([s.view(np.ndarray) for s in samples], axis=1)
        return Sample(data, cols=cols, time=samples[0]._time, scaler=samples[0].scaler, name=name)

    @profile
    def line_plot(self, exclude=None, opacity=0.9):
        """
        Create an interactive line plot of the sample data using plotly.

        Parameters
        ----------
        exclude : str or list of str, optional
            Feature(s) to exclude from the plot. Can be a single feature name or a list of feature names.
        opacity : float, optional
            Transparency level for the lines (0.0 to 1.0). Default is 0.7.
            0.0 is completely transparent, 1.0 is completely opaque.

        Returns
        -------
        plotly.graph_objects.Figure
            Interactive line plot figure.
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError("Please install plotly using: pip install plotly")

        # Validate opacity value
        opacity = max(0.0, min(1.0, opacity))  # Ensure opacity is between 0 and 1

        # Convert exclude to list if it's a string
        if isinstance(exclude, str):
            exclude = [exclude]
        # Convert exclude to empty list if None
        exclude = exclude or []

        fig = go.Figure()
        
        # Use time values if available, otherwise use index
        x_values = self._time if self._time is not None else np.arange(self._shape[1])
        data = self.view(np.ndarray)
        
        # Add a line for each column in the data, excluding specified features
        for i, col_name in enumerate(self._cols):
            if col_name not in exclude:
                fig.add_trace(
                    go.Scatter(
                        x=x_values,
                        y=data[0, :, i] if len(data.shape) == 3 else data[:, i],
                        mode='lines',
                        name=col_name,
                        opacity=opacity,
                        line=dict(width=2)  # Added line width for better visibility with transparency
                    )
                )
        
        # Update layout
        fig.update_layout(
            title='Time Series Line Plot',
            xaxis_title='Time' if self._time is not None else 'Index',
            yaxis_title='Value',
            showlegend=True,
        )
        
        return fig

    @profile
    def save_dataframe(self, path):
        """
        Save the sample to a CSV file.

        Parameters
        ----------
        path : str
            Path to save the CSV file.

        Returns
        -------
        None
        """
        df = self.to_DataFrame()
        df.to_csv(path, index=False)

    @profile
    def save(self, path, format='pickle'):
        """
        Save the sample to a file.

        Parameters
        ----------
        path : str
            Path to save the file.
        format : str, optional
            Format to save the file in, by default 'pickle'. Options: 'pickle', 'csv', 'json'.

        Returns
        -------
        None
        
        Raises
        ------
        ValueError
            If the format is not supported.
        """
        import pickle
        import json
        import os

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        if format == 'pickle':
            with open(path, 'wb') as f:
                pickle.dump(self, f)
        elif format == 'csv':
            self.save_dataframe(path)
        elif format == 'json':
            df = self.to_DataFrame()
            with open(path, 'w') as f:
                json.dump(df.to_dict(), f, indent=4)
        else:
            raise ValueError(f"Unsupported format: {format}. Choose from 'pickle', 'csv', or 'json'")
        
        return self
        
    @classmethod
    def load(cls, path, format='pickle'):
        """
        Load a sample from a file.

        Parameters
        ----------
        path : str
            Path to load the file from.
        format : str, optional
            Format to load the file from, by default 'pickle'. Options: 'pickle', 'csv', 'json'.

        Returns
        -------
        Sample
            Loaded sample.
            
        Raises
        ------
        ValueError
            If the format is not supported.
        FileNotFoundError
            If the file does not exist.
        """
        import pickle
        import json
        import os
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        
        if format == 'pickle':
            with open(path, 'rb') as f:
                return pickle.load(f)
        elif format == 'csv':
            return cls(path)
        elif format == 'json':
            import pandas as pd
            with open(path, 'r') as f:
                data = json.load(f)
            df = pd.DataFrame.from_dict(data)
            return cls(df)
        else:
            raise ValueError(f"Unsupported format: {format}. Choose from 'pickle', 'csv', or 'json'")

    @property
    def inverse(self):
        """
        Get the inverse-transformed Sample using the fitted scaler.

        Returns
        -------
        Sample
            A new Sample instance with inverse-transformed data.

        Raises
        ------
        ValueError
            If no scaler is set.
        """
        if not self.scaler:
            raise ValueError("Scaler is not set")
        data = self.view(np.ndarray)
        return Sample(self.scaler.inverse_transform(data), cols=self._cols, scaler=self.scaler, time=self._time)

    @profile
    def get_feature(self, feature: Union[int, str], exc: List[Union[int, str]] = None):
        """
        Get a feature from the Sample.

        Parameters
        ----------
        feature : Union[int, str]
            Feature to retrieve, either by index or name.
        exc : List[Union[int, str]], optional
            FeatureSample to exclude if feature="all", by default None.

        Returns
        -------
        np.ndarray
            Feature data.
        """
        data = self.view(np.ndarray)
        if isinstance(feature, str) and feature.lower() == "all":
            if exc is None:
                exc = []
            # Convert string exclusions to indices
            exc_indices = [self._cols.index(e) if isinstance(e, str) else e for e in exc]
            # Get all column indices except excluded ones
            indices = [i for i in range(len(self._cols)) if i not in exc_indices]
            # Get all features and flatten to 1D array
            return np.concatenate([data[:, i] for i in indices])
        else:
            index = self._cols.index(feature) if isinstance(feature, str) else feature
            return data[:, index]

    @profile
    def feature_max(self, feature: Union[int, str], exc: List[Union[int, str]] = None): 
        return self.get_feature(feature, exc).max()

    @profile
    def feature_min(self, feature: Union[int, str], exc: List[Union[int, str]] = None):
        """
        Calculate the minimum of a feature.

        Parameters
        ----------
        feature : Union[int, str]
            Feature to analyze, either by index or name.
        exc : List[Union[int, str]], optional
            FeatureSample to exclude if feature="all", by default None.

        Returns
        -------
        float
            Minimum of the feature values.
        """
        return self.get_feature(feature, exc).min()

    @profile
    def feature_mean(self, feature: Union[int, str], exc: List[Union[int, str]] = None):
        """
        Calculate the mean of a feature.

        Parameters
        ----------
        feature : Union[int, str]
            Feature to analyze, either by index or name.
        exc : List[Union[int, str]], optional
            FeatureSample to exclude if feature="all", by default None.

        Returns
        -------
        float
            Mean of the feature values.
        """
        return self.get_feature(feature, exc).mean()

    @profile
    def feature_std(self, feature: Union[int, str], exc: List[Union[int, str]] = None):
        """
        Calculate the standard deviation of a feature.

        Parameters
        ----------
        feature : Union[int, str]
            Feature to analyze, either by index or name.
        exc : List[Union[int, str]], optional
            FeatureSample to exclude if feature="all", by default None.

        Returns
        -------
        float
            Standard deviation of the feature values.
        """
        return np.std(self.get_feature(feature, exc))

    @profile
    def feature_var(self, feature: Union[int, str], exc: List[Union[int, str]] = None):
        """
        Calculate the variance of a feature.

        Parameters
        ----------
        feature : Union[int, str]
            Feature to analyze, either by index or name.
        exc : List[Union[int, str]], optional
            FeatureSample to exclude if feature="all", by default None.

        Returns
        -------
        float
            Variance of the feature values.
        """
        return np.var(self.get_feature(feature, exc))

    @profile
    def feature_skew(self, feature: Union[int, str], exc: List[Union[int, str]] = None):
        """
        Calculate the skew of a feature.

        Parameters
        ----------
        feature : Union[int, str]
            Feature to analyze, either by index or name.
        exc : List[Union[int, str]], optional
            FeatureSample to exclude if feature="all", by default None.

        Returns
        -------
        float
            Skew of the feature values.
        """
        return skew(self.get_feature(feature, exc))

    @profile
    def feature_kurtosis(self, feature: Union[int, str], exc: List[Union[int, str]] = None):
        """
        Calculate the kurtosis of a feature.

        Parameters
        ----------
        feature : Union[int, str]
            Feature to analyze, either by index or name.
        exc : List[Union[int, str]], optional
            FeatureSample to exclude if feature="all", by default None.

        Returns
        -------
        float
            Kurtosis of the feature values.
        """
        return kurtosis(self.get_feature(feature, exc))

    @profile
    def feature_variance(self, feature: Union[int, str], exc: List[Union[int, str]] = None):
        """
        Calculate the variance of a feature.

        Parameters
        ----------
        feature : Union[int, str]
            Feature to analyze, either by index or name.
        exc : List[Union[int, str]], optional
            FeatureSample to exclude if feature="all", by default None.

        Returns
        -------
        float
            variance of the feature values.
        """
        return st.variance(self.get_feature(feature, exc))

    @profile
    def feature_sum(self, feature: Union[int, str], exc: List[Union[int, str]] = None):
        """
        Calculate the sum of a feature.

        Parameters
        ----------
        feature : Union[int, str]
            Feature to analyze, either by index or name.
        exc : List[Union[int, str]], optional
            FeatureSample to exclude if feature="all", by default None.

        Returns
        -------
        float
            Sum of the feature values.
        """
        return self.get_feature(feature, exc).sum()

    @profile
    def feature_first(self, feature: Union[int, str], exc: List[Union[int, str]] = None):
        """
        Get the first value of a feature.

        Parameters
        ----------
        feature : Union[int, str]
            Feature to analyze, either by index or name.
        exc : List[Union[int, str]], optional
            FeatureSample to exclude if feature="all", by default None.

        Returns
        -------
        float
            First value of the feature.
        """
        return self.get_feature(feature, exc)[0]

    @profile
    def feature_last(self, feature: Union[int, str], exc: List[Union[int, str]] = None):
        """
        Get the last value of a feature.

        Parameters
        ----------
        feature : Union[int, str]
            Feature to analyze, either by index or name.
        exc : List[Union[int, str]], optional
            FeatureSample to exclude if feature="all", by default None.

        Returns
        -------
        float
            Last value of the feature.
        """
        return self.get_feature(feature, exc)[-1]

    @property
    def time_open(self):
        """
        Get the first timestamp in the time series.

        Returns
        -------
        Union[pd.Timestamp, np.datetime64, None]
            First timestamp in the series.
        """
        return self._time[0] if self._time is not None and len(self._time) > 0 else None

    @property
    def time_close(self):
        """
        Get the last timestamp in the time series.

        Returns
        -------
        Union[pd.Timestamp, np.datetime64, None]
            Last timestamp in the series.
        """
        return self._time[-1] if self._time is not None and len(self._time) > 0 else None

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, use_scaler: bool = False, scaler = None):
        """
        Create a Sample from a pandas DataFrame.

        This is a convenient factory method that properly handles DataFrame conversion
        with appropriate time indexing.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the data. The index will be used as time values.
        use_scaler : bool, optional
            Whether to use a scaler for the sample, by default False.
        scaler : sklearn.preprocessing.RobustScaler, optional
            Scaler to use for the sample. If None and use_scaler is True,
            a new scaler will be created, by default None.

        Returns
        -------
        Sample
            A new Sample instance created from the DataFrame.

        Examples
        --------
        >>> import pandas as pd
        >>> df = pd.DataFrame({'price': [100, 101, 102], 'volume': [1000, 1100, 1200]},
        ...                   index=pd.date_range('2023-01-01', periods=3))
        >>> sample = Sample.from_dataframe(df)
        """
        from .logger import logger
        
        # Extract time from DataFrame index
        if isinstance(df.index, pd.DatetimeIndex):
            time = df.index
            logger.info(f"Using DatetimeIndex as time values with {len(time)} entries")
        else:
            time = None
            logger.warning("DataFrame index is not a DatetimeIndex, no time values will be used")
            
        # Extract column names and values
        cols = list(df.columns)
        values = df.values
        
        # Create the Sample
        return cls(
            input_data=values,
            cols=cols,
            time=time,
            use_scaler=use_scaler,
            scaler=scaler
        )

    def create_lstm_dataset(self, target_feature=None, lookback=None, forecast=None, batch_size=32, return_torch=True):
        """
        Create a ready-to-use dataset for LSTM models.

        This is a convenience method that configures window settings, creates
        historical and future windows, and returns them in a format suitable for
        direct use in LSTM models.

        Parameters
        ----------
        target_feature : str, optional
            The target feature to predict. If None, all features will be used as targets.
            By default None.
        lookback : int, optional
            Number of historical time steps to use. If None, uses current historical window size.
            By default None.
        forecast : int, optional
            Number of future time steps to predict. If None, uses current future window size.
            By default None.
        batch_size : int, optional
            Batch size for the dataset, by default 32.
        return_torch : bool, optional
            If True, returns PyTorch tensors, otherwise NumPy arrays, by default True.

        Returns
        -------
        tuple
            If return_torch is True: (X_tensor, y_tensor, X_time, y_time)
            If return_torch is False: (X_array, y_array, X_time, y_time)
            Where X is input data, y is target data, and both include corresponding time values.

        Examples
        --------
        >>> sample = Sample.from_dataframe(df)
        >>> X, y, X_time, y_time = sample.create_lstm_dataset(
        ...     target_feature='price', lookback=10, forecast=5, batch_size=64
        ... )
        >>> # These can be directly used with PyTorch:
        >>> dataset = torch.utils.data.TensorDataset(X, y)
        >>> dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
        """
        from .logger import logger
        
        # Configure window settings if provided
        if lookback is not None:
            self.window_settings.historical.size = lookback
            logger.info(f"Historical window size set to {lookback}")
        
        if forecast is not None:
            self.window_settings.future.size = forecast
            logger.info(f"Future window size set to {forecast}")
        
        # Create windows
        historical, future = self.hf_sliding_window()
        
        # If target_feature is specified, extract only that feature from future windows
        if target_feature is not None:
            # Get the feature index
            feature_idx = self._cols.index(target_feature) if isinstance(target_feature, str) else target_feature
            
            # Extract only the target feature from future data
            future_data = np.array(future).view(np.ndarray)[:, :, feature_idx:feature_idx+1]
            logger.info(f"Using only '{target_feature}' as target feature")
        else:
            # Use all features as targets
            future_data = np.array(future).view(np.ndarray)
            logger.info("Using all features as targets")
        
        # Extract time values
        X_time = historical._time
        y_time = future._time
        
        # Convert to tensors if requested
        if return_torch:
            import torch
            X = torch.from_numpy(np.array(historical).view(np.ndarray)).float()
            y = torch.from_numpy(future_data).float()
            logger.info(f"Created PyTorch tensors with shapes: X {X.shape}, y {y.shape}")
            return X, y, X_time, y_time
        else:
            X = np.array(historical).view(np.ndarray)
            y = future_data
            logger.info(f"Created NumPy arrays with shapes: X {X.shape}, y {y.shape}")
            return X, y, X_time, y_time
        
    @classmethod
    def from_FeatureSamples(cls, feature_samples: List[FeatureSample], scaler=None):
        lengths = [len(fs) for fs in feature_samples]
        if not all(length == lengths[0] for length in lengths):
            raise ValueError("All feature samples must have the same length")
        
        # Create a new Sample with the combined data
        return cls(
            np.stack([f._as_nparray() for f in feature_samples], axis=1), 
            cols=[f.name for f in feature_samples], 
            time=feature_samples[0]._time,
            scaler=scaler
        )
        
