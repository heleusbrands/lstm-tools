import numpy as np
import pandas as pd
import os
from typing import List
import datetime
import statistics as st


def verify_dataset(dataset, samples, sequence_length):
    """
    Verify the dataset by checking the inputs and targets.

    Parameters
    ----------
    dataset : Dataset
        Dataset to verify.
    samples : Sample
        Sample data to compare with.
    sequence_length : int
        Length of the sequence.

    Returns
    -------
    None
    """
    for batch in dataset:
        inputs, targets = batch
        assert np.array_equal(inputs[0], samples._base[:sequence_length])
        assert np.array_equal(targets[0], samples._base[sequence_length])
        print("Validation Complete")
        break

def merge_csv(names: List[str], files: List[str], output_path: str=None, append:str=None, save=False):
    """
    Merge multiple CSV files into a single DataFrame.

    Parameters
    ----------
    names : List[str]
        List of names to identify each CSV file.
    files : List[str]
        List of file paths to the CSV files to merge.
    output_path : str, optional
        Directory path to save the merged CSV file, by default None.
    append : str, optional
        Append string to the filename, by default None.
    save : bool, optional
        Whether to save the merged CSV file, by default False.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame containing data from all CSV files.
    """
    name = '_'.join(names)
    file_name = f"{name}_{append}" if append else name
    data = [pd.read_csv(f, parse_dates=['time'], index_col='time') for f in files]
    dfs = []
    for i in range(len(data)):
        df = data[i]
        c = df.columns
        cnew = {col: f"{names[i]}_{col}" for col in c}
        df_new = df.rename(columns=cnew)
        dfs.append(df_new)
    
    con_data: pd.DataFrame = pd.concat(dfs, axis=1, join='inner')
    if save:
        os.makedirs(output_path, exist_ok=True)
        if os.path.exists(os.path.join(output_path, f"{file_name}.csv")):
            os.remove(os.path.join(output_path, f"{file_name}.csv"))
        con_data.to_csv(os.path.join(output_path, f"{file_name}.csv"), index=False)
    return con_data

def date_converter(date: str, format: str = "%m-%d-%y-%H-%M"):
    datetime_object = datetime.datetime.strptime(date, format)
    timestamp = datetime_object.timestamp()
    return int(timestamp)

def timestamp_converter(timestamp, formatting = "%m-%d-%y-%H-%M"):
    datetime_object = datetime.datetime.fromtimestamp(timestamp)
    return datetime_object.strftime(formatting)

def calculate_weighted_mean(data):
    """
    Calculate the weighted mean of a list of numbers.

    Weights are assigned such that numbers closer to the end of the list have more weight.
    """
    n = len(data)
    weights = np.arange(1, n + 1)  # assign weights from 1 to n
    weights = weights / weights.sum()  # normalize weights to sum to 1
    return np.average(data, weights=weights)

def weighted_average_right_np(data, weights_type='linear', custom_weights_func=None):
    """
    Calculates a weighted average of a NumPy array, giving more weight to values
    closer to the end (right side) of the array.

    Args:
        data: A NumPy array of numerical values.
        weights_type: The type of weighting scheme. Options:
            'linear' (default): Linearly increasing weights from 0 to 1.
            'quadratic': Weights increase quadratically (x^2).
            'exponential': Weights increase exponentially (2^x, normalized).
            'custom': Allows the user to specify a custom weighting function.
                       If 'custom', `custom_weights_func` must be provided.
        custom_weights_func: A function that takes an index (int) and the array
                             length (int) as arguments and returns a weight (float).
                             Required if weights_type='custom'.

    Returns:
        The weighted average (a float).
        Returns NaN if the input data is empty.

    Raises:
        TypeError: If input data is not a NumPy array.
        ValueError: If an unsupported weights_type is provided or if 'custom'
            is selected but custom_weights_func is not a callable function.
            ValueError: If any weight is negative.
            ValueError: Custom weight function output length is different than the input data length.
        ZeroDivisionError: if the sum of weights is zero.

    """

    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a NumPy array.")

    n = len(data)
    if n == 0:
        return np.nan  # Handle empty array case

    if weights_type == 'linear':
        weights = np.linspace(1/n, 1, n)  # Start from 1/n to avoid zero weight.
    elif weights_type == 'quadratic':
        weights = np.linspace(0, 1, n)**2
    elif weights_type == 'exponential':
        weights = 2**(np.linspace(0, 1, n)) -1 # shift to have the first weight = 0
        weights = weights / np.sum(weights)  # Normalize after exponentiation
    elif weights_type == 'custom':
        if not callable(custom_weights_func):
            raise ValueError("'custom_weights_func' must be a callable function.")
        weights = np.array([custom_weights_func(i, n) for i in range(n)])

    else:
        raise ValueError("Invalid weights_type. Choose 'linear', 'quadratic', 'exponential', or 'custom'.")

    if len(weights) != len(data) :
       raise ValueError("Custom weight function does not return a list of weights of the same length of data")

    if np.any(weights < 0):
          raise ValueError("Weights cannot be negative.")

    sum_weights = np.sum(weights)

    if sum_weights == 0:
        raise ZeroDivisionError("Sum of weights cannot be zero.")

    normalized_weights = weights / sum_weights
    return np.sum(data * normalized_weights)

def sliding_window(arr, window_size, step_size=1):
    """
    Create a sliding window view of the input data.

    Parameters
    ----------
    arr : np.ndarray
        Input array to create windows from.
    window_size : int
        Size of each window.
    step_size : int, optional
        Step size between consecutive windows, by default 1.

    Returns
    -------
    np.ndarray
        Array containing the sliding window views.
    """
    # Pre-calculate the shape and strides for better performance
    n_windows = ((arr.shape[0] - window_size) // step_size) + 1
    window_shape = (n_windows, window_size) + arr.shape[1:]
    
    # Create strided view without unnecessary copies
    new_strides = (arr.strides[0] * step_size,) + arr.strides
    
    # Create the view directly without intermediate copies
    return np.lib.stride_tricks.as_strided(arr, shape=window_shape, strides=new_strides, writeable=False)

def hf_sliding_window(arr, historical_size, future_size, step_size=1):
    """
    Create historical and future sliding window views of the input data.

    Parameters
    ----------
    arr : np.ndarray
        Input array to create windows from.
    historical_size : int
        Size of the historical window.
    future_size : int
        Size of the future window.
    step_size : int, optional
        Step size between consecutive windows, by default 1.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple containing (historical_windows, future_windows).
    """
    # Calculate total required length
    total_size = historical_size + future_size
    if len(arr) < total_size:
        raise ValueError(f"Array length {len(arr)} is less than total window size {total_size}")
    
    # Calculate number of windows
    n_windows = ((len(arr) - total_size) // step_size) + 1
    
    # Pre-allocate the arrays with the correct shape
    hist_shape = (n_windows, historical_size) + arr.shape[1:]
    fut_shape = (n_windows, future_size) + arr.shape[1:]
    
    # Create strided views for both historical and future windows simultaneously
    hist_strides = (arr.strides[0] * step_size,) + arr.strides
    fut_strides = hist_strides
    
    # Create the views directly
    historical_windows = np.lib.stride_tricks.as_strided(
        arr, 
        shape=hist_shape,
        strides=hist_strides,
        writeable=False
    )
    
    future_windows = np.lib.stride_tricks.as_strided(
        arr[historical_size:], 
        shape=fut_shape,
        strides=fut_strides,
        writeable=False
    )
    
    return historical_windows, future_windows

def subwindow(arr, subwindow_size, direction="backward"):
    """
    Create a subwindow view from a sliding window.

    Parameters
    ----------
    arr : np.ndarray
        Original input data.
    subwindow_size : int
        Size of the subwindow to create.
    direction : str, optional
        Direction to create the subwindow ('forward' or 'backward'), by default 'backward'.

    Returns
    -------
    np.ndarray
        Subwindow view of the data.
    """
    if direction == "backward":
        return arr[:, -subwindow_size:]
    elif direction == "forward":
        return arr[:, :subwindow_size]
    else:
        raise ValueError("Invalid direction. Must be 'forward' or 'backward'")

def calculate_weighted_mean(data):
    """
    Calculate the weighted mean of a list of numbers.

    Weights are assigned such that numbers closer to the end of the list have more weight.
    """
    n = len(data)
    weights = np.arange(1, n + 1)  # assign weights from 1 to n
    weights = weights / weights.sum()  # normalize weights to sum to 1
    return np.average(data, weights=weights)

def kurtosis(arr, axis=0):
    """Calculates kurtosis along the specified axis.

  Args:
    arr: The input NumPy array.
    axis: The axis along which to calculate kurtosis.

  Returns:
    The calculated kurtosis.
    """

    n = arr.shape[axis]
    mean = np.mean(arr, axis=axis)
    var = np.var(arr, axis=axis)

    # Check for zero variance
    if np.all(var == 0):
      return np.zeros_like(var)

    std = np.sqrt(var)
    central_moments = np.mean((arr - mean)**4, axis=axis)
    return (n * central_moments) / (var**2) - 3 * (n - 1) / (n - 2)

def skew_with_bias(arr, axis=0, bias=True):
  """
  Calculate the skew of an array along the specified axis.

  Parameters
  ----------
  arr : np.ndarray
      Input array.
  axis : int, optional
      Axis along which to calculate skew, by default 0.
  bias : bool, optional
      If True, uses the biased estimator, otherwise uses the unbiased estimator, by default True.

  Returns
  -------
  float
      Skew value of the array.
  """

  n = arr.shape[axis]
  mean = np.mean(arr, axis=axis)
  std = np.std(arr, axis=axis, ddof=int(not bias))  # ddof for biased/unbiased std
  central_moments = np.mean((arr - mean)**3, axis=axis)

  if bias:
    return n * central_moments / std**3
  else:
    return np.sqrt(n * (n - 1)) * central_moments / ((n - 2) * std**3)

def skew(arr):
    """
    Calculate the skew of a NumPy array efficiently.

    Parameters
    ----------
    arr : np.ndarray
        Input array.

    Returns
    -------
    float
        Skew value of the array.
    """

    # Calculate the mean and standard deviation
    arr_mean = np.mean(arr)
    arr_std = np.std(arr, ddof=1)  # Note: ddof=1 for sample standard deviation

    # Calculate the centered and standardized values
    centered_arr = arr - arr_mean
    standardized_arr = centered_arr / arr_std

    # Calculate the skew using vectorized operations
    n = len(arr)
    return np.sum(standardized_arr**3) / n


class TradeWindowOps:

    @classmethod    
    def max(cls, data: np.ndarray):
        """
        Calculate the maximum of the input data.

        Parameters
        ----------
        data : np.ndarray
            Input array.

        Returns
        -------
        float
            Maximum value of the array.
        """
        return np.max(data)
    
    @classmethod
    def min(cls, data: np.ndarray):
        """
        Calculate the minimum of the input data.

        Parameters
        ----------
        data : np.ndarray
            Input array.

        Returns
        -------
        float
            Minimum value of the array.
        """
        return np.min(data)
    
    @classmethod
    def mean(cls, data: np.ndarray):
        """
        Calculate the mean of the input data.

        Parameters
        ----------
        data : np.ndarray
            Input array.

        Returns
        -------
        float
            Mean value of the array.
        """
        return np.mean(data)
    
    @classmethod
    def median(cls, data: np.ndarray):
        """
        Calculate the median of the input data.
        """
        return np.median(data)
    
    @classmethod
    def sum(cls, data: np.ndarray):
        """
        Calculate the sum of the input data.

        Parameters
        ----------
        data : np.ndarray
            Input array.

        Returns
        -------
        float
            Sum value of the array.
        """
        return np.sum(data)
    
    @classmethod
    def std(cls, data: np.ndarray):
        """
        Calculate the standard deviation of the input data.

        Parameters
        ----------
        data : np.ndarray
            Input array.

        Returns
        -------
        float
            Standard deviation value of the array.
        """
        return np.std(data)
    
    @classmethod
    def skew(cls, data: np.ndarray):
        """
        Calculate the skew of the input data.

        Parameters
        ----------
        data : np.ndarray
            Input array.

        Returns
        -------
        float
            Skew value of the array.
        """
        return skew(data)
    
    @classmethod
    def kurtosis(cls, data: np.ndarray):
        """
        Calculate the kurtosis of the input data.

        Parameters
        ----------
        data : np.ndarray
            Input array.

        Returns
        -------
        float
            Kurtosis value of the array.
        """
        return kurtosis(data)
    
    @classmethod
    def variance(cls, data: np.ndarray):
        """
        Calculate the variance of the input data.

        Parameters
        ----------
        data : np.ndarray
            Input array.

        Returns
        -------
        float
            Variance value of the array.
        """
        return np.var(data)
    
    @classmethod
    def first(cls, data: np.ndarray):
        """
        Get the first value of the input data.

        Parameters
        ----------
        data : np.ndarray
            Input array.

        Returns
        -------
        float
            First value of the array.
        """
        return data[0]
    
    @classmethod
    def last(cls, data: np.ndarray):
        """
        Get the last value of the input data.

        Parameters
        ----------
        data : np.ndarray
            Input array.

        Returns
        -------
        float
            Last value of the array.
        """
        return data[-1]