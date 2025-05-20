import numpy as np
import pandas as pd
import os
from typing import List
import datetime
import statistics as st
from collections import UserDict
from typing import Any
from inspect import signature

_DEFAULT_TARGET_CHUNK_PROCESSING_MEMORY_MB = 256
_DEFAULT_K_MEMORY_MULTIPLIER = 4
_MIN_ARRAY_SIZE_FOR_AUTOCHUNK_MB = 512

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

def has_axis_argument(func):
    return 'axis' in signature(func).parameters

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

def sliding_window(arr, window_size, step_size=1, offset=0):
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
    if offset > 0:
        arr = arr[offset:]
    if offset < 0:
        arr = arr[:offset]
    # Pre-calculate the shape and strides for better performance
    n_windows = ((arr.shape[0] - window_size) // step_size) + 1
    window_shape = (n_windows, window_size) + arr.shape[1:]
    
    # Create strided view without unnecessary copies
    new_strides = (arr.strides[0] * step_size,) + arr.strides
    
    # Create the view directly without intermediate copies
    return np.lib.stride_tricks.as_strided(arr, shape=window_shape, strides=new_strides, writeable=False)

def hf_sliding_window(
    arr,
    h_size,
    f_size,
    f_offset=0,
    h_spacing=1,
    step_size=1
    ): 
    """
    Create historical and future sliding window views of the input data.

    Parameters
    ----------
    arr : np.ndarray or array-like
        Input array to create windows from. Windowing is performed on the first axis.
    h_size : int
        The span of the original data from which the historical window's elements are drawn.
    f_size : int
        Size of the future window (number of elements along the first axis).
    f_offset : int, optional
        Offset for the start of the future window relative to the end of the
        historical window. Default is 0.
    h_spacing : int, optional
        Step size for picking elements within each historical window.
        A value of 1 (default) means all elements within the `historical_size` span are taken.
        A value of k > 1 means every k-th element is taken from the `historical_size` span.
        For example, if historical_size=100 and h_spacing=3, the window
        will contain elements from original indices like [0, 3, 6, ..., 99] relative to
        the start of that window's span.
    step_size : int, optional
        Step size for sliding the entire (historical, future) window pair along `arr`. Default is 1.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple containing (historical_windows, future_windows).

    Raises
    ------
    ValueError
        If various size/step parameters are invalid.
    """
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)

    if arr.ndim == 0:
        raise ValueError("Input array cannot be a scalar.")
    if h_size <= 0:
        raise ValueError("historical_size must be positive.")
    if f_size <= 0:
        raise ValueError("future_size must be positive.")
    if step_size <= 0:
        raise ValueError("step_size must be positive.")
    if h_spacing <= 0: # Validation for new parameter
        raise ValueError("historical_segment_step must be positive.")

    future_data_array_start_idx = h_size + f_offset
    if future_data_array_start_idx < 0:
        raise ValueError(
            f"The effective start index for future windows in the base array "
            f"(historical_size + future_offset = {future_data_array_start_idx}) "
            f"cannot be negative."
        )

    max_reach_from_window_start = max(h_size, future_data_array_start_idx + f_size)

    if arr.shape[0] < max_reach_from_window_start:
        num_possible_steps_for_sliding = -1
    else:
        num_possible_steps_for_sliding = arr.shape[0] - max_reach_from_window_start

    if num_possible_steps_for_sliding < 0:
        n_windows = 0
    else:
        n_windows = (num_possible_steps_for_sliding // step_size) + 1

    # Calculate the actual number of elements in each (potentially segmented) historical window
    if h_size == 0 : # Should be caught by historical_size > 0 check
        num_elements_in_segmented_hist_window = 0
    else:
        num_elements_in_segmented_hist_window = (h_size - 1) // h_spacing + 1


    if n_windows <= 0:
        # For empty hist_shape, use the new segmented size
        empty_hist_shape = (0, num_elements_in_segmented_hist_window) + arr.shape[1:]
        empty_fut_shape = (0, f_size) + arr.shape[1:]
        return np.empty(empty_hist_shape, dtype=arr.dtype), np.empty(empty_fut_shape, dtype=arr.dtype)

    # Define the shape of the resulting window arrays
    # Historical window shape changes based on segmentation
    hist_shape = (n_windows, num_elements_in_segmented_hist_window) + arr.shape[1:]
    fut_shape = (n_windows, f_size) + arr.shape[1:] # Future window shape is unchanged

    # Define the strides for the window arrays
    base_element_strides = arr.strides
    window_step_stride_bytes = base_element_strides[0] * step_size # Stride for the 'n_windows' dimension

    # Strides for dimensions *within* each historical window
    # The first of these (arr.strides[0]) needs to be scaled by historical_segment_step
    strides_within_hist_window_list = list(base_element_strides)
    strides_within_hist_window_list[0] *= h_spacing
    
    hist_strides = (window_step_stride_bytes,) + tuple(strides_within_hist_window_list)

    # Strides for future windows remain as before (no internal segmentation for future windows)
    fut_strides = (window_step_stride_bytes,) + base_element_strides

    historical_windows = np.lib.stride_tricks.as_strided(
        arr,
        shape=hist_shape,
        strides=hist_strides,
        writeable=False
    )

    future_array_base_view = arr[future_data_array_start_idx:]

    future_windows = np.lib.stride_tricks.as_strided(
        future_array_base_view,
        shape=fut_shape,
        strides=fut_strides,
        writeable=False
    )

    return historical_windows, future_windows

def sub_window_over_dimension(
    arr: np.ndarray,
    window_sizes: list[int], # Use list[int] for Python 3.9+
    alignment: str,
    axis_to_window: int = 1) -> list[np.ndarray]: # Use list[np.ndarray]
    """
    Creates sub-window selections (views) over a specified dimension of an input array.

    For each size in window_sizes, a sub-window view is created from the input array.
    This function does not create sliding windows that increase the number of items;
    instead, it extracts a fixed-size segment from the specified axis for each
    existing item along other axes. The created windows will overlap if their
    source regions in the original array overlap (e.g., asking for left-aligned
    windows of size 2 and 5 will result in the size 2 window being a sub-region
    of the size 5 window).

    Parameters
    ----------
    arr : np.ndarray
        Input array. For optimal performance, this should already be a NumPy array.
    window_sizes : list[int]
        A list of integers, where each integer specifies the size of a
        sub-window to be created along the `axis_to_window`.
    alignment : str
        Determines the alignment of the sub-windows:
        - "right": Right-aligned ("historical"). Windows are taken from the end
                  (most recent part) of the `axis_to_window`.
                  E.g., for size `k`, the slice is `arr[..., -k:, ...]`.
        - "left": Left-aligned ("future" or "forward-looking"). Windows are taken
                   from the beginning of the `axis_to_window`.
                   E.g., for size `k`, the slice is `arr[..., :k, ...]`.
    axis_to_window : int, optional
        The dimension (axis) of `arr` from which sub-windows will be extracted.
        Defaults to 1 (the second dimension, as per the example context).

    Returns
    -------
    list[np.ndarray]
        A list of NumPy arrays, where each array is a view into the original `arr`
        and corresponds to a size specified in `window_sizes`. The shapes of these
        views will be the same as `arr` except along `axis_to_window`, where the
        dimension size will be the respective window size.

    Raises
    ------
    TypeError
        If `arr` is not a NumPy array and cannot be converted by `np.array()`.
    ValueError
        If `arr` is 0-dimensional (scalar).
        If `axis_to_window` is out of bounds for `arr.ndim`.
        If any `window_size` in `window_sizes` is not a positive integer.
        If any `window_size` is larger than the dimension size of `arr`
        at `axis_to_window`.
    """
    if not isinstance(arr, np.ndarray):
        # Attempt to convert to NumPy array if it's not.
        # For performance-critical applications, passing an ndarray directly is preferred.
        try:
            arr_np = np.array(arr)
        except Exception as e:
            raise TypeError(
                f"Input 'arr' must be a NumPy array or convertible to one. "
                f"Conversion failed: {e}"
            )
    else:
        arr_np = arr

    if arr_np.ndim == 0:
        raise ValueError("Input array 'arr' cannot be a 0-dimensional scalar.")

    if not (0 <= axis_to_window < arr_np.ndim):
        raise ValueError(
            f"axis_to_window ({axis_to_window}) is out of bounds "
            f"for input array with {arr_np.ndim} dimensions."
        )

    dim_size_to_window_on = arr_np.shape[axis_to_window]
    output_views = []

    if not window_sizes: # Handle empty window_sizes list
        return output_views

    for k_size in window_sizes:
        if not isinstance(k_size, int) or k_size <= 0:
            raise ValueError(
                f"All window sizes must be positive integers, got {k_size}."
            )
        if k_size > dim_size_to_window_on:
            raise ValueError(
                f"Window size {k_size} is larger than the dimension size "
                f"{dim_size_to_window_on} at axis {axis_to_window}."
            )

        # Construct the slice tuple for all dimensions.
        # Start with slice(None) for all dimensions, which means selecting all elements
        # along those axes (equivalent to ':'). Then, overwrite the slice for the target axis.
        slice_obj_list = [slice(None)] * arr_np.ndim

        if alignment == "right":  # Right-aligned (from the end)
            # Slice for the target dimension: from -k_size up to the end.
            # e.g., if k_size=5, indices are -5, -4, -3, -2, -1 relative to end.
            target_dim_slice = slice(-k_size, None)
        elif alignment == "left":  # Left-aligned (from the beginning)
            # Slice for the target dimension: from 0 up to (but not including) k_size.
            # e.g., if k_size=5, indices are 0, 1, 2, 3, 4.
            target_dim_slice = slice(0, k_size)

        slice_obj_list[axis_to_window] = target_dim_slice
        
        # NumPy indexing requires a tuple of slice objects or indices
        sub_window_view = arr_np[tuple(slice_obj_list)]
        output_views.append(sub_window_view)

    return output_views

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

def first(arr: np.ndarray, axis: int) -> np.ndarray:
    """
    Returns the first element (slice) along the specified axis.

    This is functionally equivalent to `arr[..., 0, ...]`, where the `0`
    is at the position specified by `axis`.

    Parameters
    ----------
    arr : array_like
        Input array or object that can be converted to an array.
    axis : int
        The axis along which to select the first element.
        Negative axis values are supported (e.g., -1 for the last axis).

    Returns
    -------
    np.ndarray
        An array containing the first element of each slice along the specified axis.
        The returned array will have one less dimension than the input array.

    Raises
    ------
    np.AxisError
        If the specified axis is out of bounds for the input array's dimensions.
    IndexError
        If the specified axis has a size of 0 (i.e., it's an empty dimension).
    TypeError
        If `arr` cannot be converted to a NumPy array.
    """
    # np.take handles array conversion, axis normalization, and bounds checking.
    try:
        return np.take(arr, indices=0, axis=axis)
    except IndexError as e:
        # Provide a slightly more specific message if the axis exists but is empty
        # First, ensure 'arr' is an ndarray to check its properties
        if not isinstance(arr, np.ndarray):
            try:
                # Attempt conversion once to check properties; np.take will do its own.
                # This is just for a more specific error message.
                temp_arr = np.asarray(arr)
            except TypeError:
                raise e # Let original TypeError from np.take propagate if initial conversion fails
        else:
            temp_arr = arr
        
        # Normalize axis to check shape (np.take would have already raised AxisError if axis is invalid)
        ndim = temp_arr.ndim
        current_axis_normalized = axis
        if axis < 0:
            current_axis_normalized += ndim
        
        if 0 <= current_axis_normalized < ndim and temp_arr.shape[current_axis_normalized] == 0:
            raise IndexError(f"Cannot select the first element along axis {axis} because this axis has size 0.") from e
        raise # Re-raise original IndexError if it's for a different reason or AxisError from np.take
    # np.AxisError and other errors from np.take will propagate naturally.


def last(arr: np.ndarray, axis: int) -> np.ndarray:
    """
    Returns the last element (slice) along the specified axis.

    This is functionally equivalent to `arr[..., -1, ...]`, where the `-1`
    is at the position specified by `axis`.

    Parameters
    ----------
    arr : array_like
        Input array or object that can be converted to an array.
    axis : int
        The axis along which to select the last element.
        Negative axis values are supported.

    Returns
    -------
    np.ndarray
        An array containing the last element of each slice along the specified axis.
        The returned array will have one less dimension than the input array.

    Raises
    ------
    np.AxisError
        If the specified axis is out of bounds for the input array's dimensions.
    IndexError
        If the specified axis has a size of 0 (i.e., it's an empty dimension).
    TypeError
        If `arr` cannot be converted to a NumPy array.
    """
    try:
        return np.take(arr, indices=-1, axis=axis)
    except IndexError as e:
        # Provide a slightly more specific message if the axis exists but is empty
        if not isinstance(arr, np.ndarray):
            try:
                temp_arr = np.asarray(arr)
            except TypeError:
                raise e
        else:
            temp_arr = arr
            
        ndim = temp_arr.ndim
        current_axis_normalized = axis
        if axis < 0:
            current_axis_normalized += ndim
            
        if 0 <= current_axis_normalized < ndim and temp_arr.shape[current_axis_normalized] == 0:
            raise IndexError(f"Cannot select the last element along axis {axis} because this axis has size 0.") from e
        raise

def kurtosis_old(arr, axis=0):
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

def _calculate_kurtosis_on_chunk(chunk_data: np.ndarray,
                                 axis: int | None,
                                 fisher: bool,
                                 bias: bool,
                                 n_for_bias_correction: float) -> np.ndarray | float:
    """
    Core logic to calculate kurtosis on a given data chunk.
    n_for_bias_correction is the N from the original slice for unbiased correction factors.
    """
    # N for current chunk's calculations (should be same as n_for_bias_correction
    # if chunking is not done along the reduction axis itself)
    n_in_chunk_reduction_axis = chunk_data.shape[axis] if axis is not None else chunk_data.size

    if n_in_chunk_reduction_axis == 0:
        if axis is None: return np.nan
        out_shape_list = list(chunk_data.shape)
        current_axis_norm_chunk = axis if axis >= 0 else axis + chunk_data.ndim
        # Ensure axis index is valid before attempting to delete
        if 0 <= current_axis_norm_chunk < len(out_shape_list):
            del out_shape_list[current_axis_norm_chunk]
        elif len(out_shape_list) == 0 and chunk_data.ndim > 0 : # e.g. input (0,), axis=0
             return np.nan # scalar NaN
        else: # Should be caught by wrapper if axis is globally invalid or array.size is 0
            # This case might occur if chunk_data itself has a 0-dim axis, e.g. (X, 0, Y)
            # and this function is called. The out_shape_list logic should handle it.
            pass
        if not out_shape_list and chunk_data.ndim == 1: return np.nan # scalar output
        return np.full(tuple(out_shape_list) if out_shape_list else (), np.nan)

    with np.errstate(divide='ignore', invalid='ignore'): # Handle 0/0 or x/0
        mean_val = np.mean(chunk_data, axis=axis, keepdims=True)
        deviations = chunk_data - mean_val
        dev_sq = deviations**2
        
        # Biased moments (denominator: n_in_chunk)
        m2_biased = np.mean(dev_sq, axis=axis, keepdims=True)
        m4_biased = np.mean(dev_sq**2, axis=axis, keepdims=True) # m4 = E[(X-mu)^4]

        is_m2_zero_mask = np.isclose(m2_biased, 0.0) # True where variance is zero

        if bias:
            # For constant data (m2_biased=0), m4_biased=0. Pearson kurtosis = 0. Excess = -3.
            # This matches SciPy for bias=True.
            pearson_kurtosis_val = np.where(is_m2_zero_mask, 0.0, m4_biased / (m2_biased**2))
            
            if fisher:
                kurtosis_result = pearson_kurtosis_val - 3.0
            else:
                kurtosis_result = pearson_kurtosis_val
        else: # Unbiased calculation (aiming for SciPy bias=False behavior)
            N = n_for_bias_correction # N of the original slice/array
            if N < 4: # Unbiased kurtosis undefined for N < 4
                out_shape_list = list(m4_biased.shape) # Use shape before reduction
                current_axis_norm_chunk = axis if axis is None or axis >=0 else axis + chunk_data.ndim
                if axis is not None: del out_shape_list[current_axis_norm_chunk]
                return np.full(tuple(out_shape_list) if out_shape_list else (), np.nan)

            # For constant data (m2_biased is zero) and N >= 4:
            # SciPy returns -3.0 for Fisher (excess) and 0.0 for Pearson.
            # This specific handling for constant data in unbiased mode is crucial.
            
            # General formula for non-constant data (SciPy's _skewtest_finish for kurtosis)
            # G2_excess = ( (N-1) / ((N-2)*(N-3)) ) * ( (N+1) * (m4/m2^2)_biased_moments - 3*(N-1) )
            # Here, m4 and m2 are the biased moments from the sample.
            g2_term_from_biased_moments = m4_biased / (m2_biased**2) # This will be NaN if m2_biased is 0

            num = (N - 1.0) * ((N + 1.0) * g2_term_from_biased_moments - 3.0 * (N - 1.0))
            den = (N - 2.0) * (N - 3.0)
            
            unbiased_excess_kurtosis = num / den # Will be NaN if g2_term was NaN or den=0 (N<4 handles den=0)

            # Now, apply the special handling for constant data (m2_biased == 0)
            # Squeeze mask if needed. The result of np.where should match the non-constant calculation shape.
            squeezed_m2_zero_mask = is_m2_zero_mask
            # kurtosis_result will be shaped like the output of np.mean(..., axis=axis)
            # mean_val was (..., 1, ...), m2_biased was (..., 1, ...), result is (..., ...)
            if axis is not None and m2_biased.ndim > unbiased_excess_kurtosis.ndim :
                 squeezed_m2_zero_mask = is_m2_zero_mask.squeeze(axis=axis)
            
            if fisher:
                kurtosis_result = np.where(squeezed_m2_zero_mask, -3.0, unbiased_excess_kurtosis)
            else: # Pearson kurtosis
                kurtosis_result = np.where(squeezed_m2_zero_mask, 0.0, unbiased_excess_kurtosis + 3.0)
    
    return kurtosis_result


def kurtosis(arr: np.ndarray,
             axis: int | None = None,
             fisher: bool = True,
             bias: bool = True,
             chunk_size: int | None = None) -> np.ndarray | float:
    """
    Calculates kurtosis (Fisher's or Pearson's) of data in an array
    along a specified axis, with optional and automatic chunking for memory efficiency.

    Parameters
    ----------
    arr : np.ndarray
        Input array containing numbers. Must be a NumPy array or convertible.
    axis : int or None, optional
        Axis along which the kurtosis is computed.
        If None (default), the kurtosis is computed over the entire flattened array.
    fisher : bool, optional
        If True (default), Fisher's definition of kurtosis is used (normal ==> 0.0).
        If False, Pearson's definition is used (normal ==> 3.0).
    bias : bool, optional
        If True (default), the biased kurtosis (based on N-denominator moments)
        is calculated.
        If False, an attempt is made to compute sample kurtosis adjusted for bias,
        similar to `scipy.stats.kurtosis(bias=False)`. Returns NaN for N < 4.
        For constant data (variance=0) and N>=4, returns -3.0 (Fisher) or 0.0 (Pearson).
    chunk_size : int or None, optional
        Number of slices to process at a time along the primary non-reduction axis.
        If None, attempts to auto-calculate a chunk size for large arrays.
        Set to a value (e.g., 1000) to manually control memory usage.

    Returns
    -------
    np.ndarray or float
        The kurtosis values. If `axis` is None, a float is returned.
        Returns 0.0 (Pearson) or -3.0 (Fisher) for slices with zero standard
        deviation when `bias=True`.
        For `bias=False` and zero standard deviation (N>=4), returns 0.0 (Pearson)
        or -3.0 (Fisher).
        Returns np.nan for slices where kurtosis is undefined (e.g., N<4 for
        unbiased, or empty input).
    """
    if not isinstance(arr, np.ndarray):
        try: arr_internal = np.array(arr, dtype=float)
        except Exception as e: raise TypeError(f"Input 'arr' must be NumPy array or convertible. Error: {e}")
    elif np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.bool_):
        arr_internal = arr.astype(float)
    else: arr_internal = arr

    if arr_internal.ndim == 0: # Scalar input
        if bias:
            return -3.0 if fisher else 0.0 # Kurtosis of a single point (N=1, m2=0)
        else: # bias=False, N=1 < 4
            return np.nan

    # Normalize axis and get N for global checks
    current_axis_normalized: int
    n_global_reduction_axis: int
    if axis is None:
        current_axis_normalized = -1 # Indicates no specific axis for shape manipulation later
        n_global_reduction_axis = arr_internal.size
    else:
        if not (-arr_internal.ndim <= axis < arr_internal.ndim):
            raise np.AxisError(f"Axis {axis} is out of bounds for array of dimension {arr_internal.ndim}")
        current_axis_normalized = axis if axis >= 0 else axis + arr_internal.ndim
        n_global_reduction_axis = arr_internal.shape[current_axis_normalized]

    # Handle empty array or zero-length reduction axis globally
    if arr_internal.size == 0 or n_global_reduction_axis == 0:
        if axis is None: return np.nan
        out_shape_list = list(arr_internal.shape)
        # Ensure current_axis_normalized is valid before del
        if 0 <= current_axis_normalized < len(out_shape_list):
            del out_shape_list[current_axis_normalized]
        elif len(out_shape_list)==0 and arr_internal.ndim > 0: # e.g. input (0,), axis=0, result is scalar
            return np.nan
        if not out_shape_list and arr_internal.ndim == 1 : return np.nan
        return np.full(tuple(out_shape_list) if out_shape_list else (), np.nan)

    # Global check for unbiased kurtosis with N < 4
    if not bias and n_global_reduction_axis < 4:
        if axis is None: return np.nan
        out_shape_list = list(arr_internal.shape)
        del out_shape_list[current_axis_normalized]
        if not out_shape_list and arr_internal.ndim == 1: return np.nan
        return np.full(tuple(out_shape_list) if out_shape_list else (), np.nan)

    # --- Chunking Logic ---
    apply_chunking = False
    effective_chunk_size = 0
    iteration_axis = -1

    if axis is None or arr_internal.ndim <= 1: # No chunking for flat or 1D arrays in this scheme
        pass
    else: # ndim > 1 and axis is not None (so current_axis_normalized is set)
        iteration_axis = 0 if current_axis_normalized != 0 else 1
        if iteration_axis >= arr_internal.ndim: # e.g. 2D array, axis=0, iteration_axis=1 (valid)
                                                # e.g. 2D array, axis=1, iteration_axis=0 (valid)
                                                # This check is more for safety if ndim became <2 unexpectedly
            iteration_axis = -1

    if chunk_size is not None: # User specified a chunk size
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError("User-specified chunk_size must be a positive integer.")
        if iteration_axis != -1 and arr_internal.shape[iteration_axis] > chunk_size:
            apply_chunking = True
            effective_chunk_size = chunk_size
    elif iteration_axis != -1: # Auto-chunking mode, and a valid iteration_axis exists
        if arr_internal.nbytes > _MIN_ARRAY_SIZE_FOR_AUTOCHUNK_MB * 1024 * 1024:
            bytes_per_slice_along_iteration_axis = arr_internal.nbytes / arr_internal.shape[iteration_axis]
            if bytes_per_slice_along_iteration_axis > 0:
                target_data_chunk_bytes = (_DEFAULT_TARGET_CHUNK_PROCESSING_MEMORY_MB * 1024 * 1024) / _DEFAULT_K_MEMORY_MULTIPLIER
                desired_chunk_len = max(1, int(np.floor(target_data_chunk_bytes / bytes_per_slice_along_iteration_axis)))
                if arr_internal.shape[iteration_axis] > desired_chunk_len:
                    apply_chunking = True
                    effective_chunk_size = desired_chunk_len
                    print(f"Large Array Detect: Auto-Chunking at Chunk Size: {effective_chunk_size}")
            # else: bytes_per_slice is 0 (e.g. shape=(100,0,10)), no meaningful chunking on this axis.
    
    # --- Actual processing ---
    if not apply_chunking or iteration_axis == -1 : # Added iteration_axis == -1 check
        return _calculate_kurtosis_on_chunk(arr_internal, axis, fisher, bias, float(n_global_reduction_axis))
    else:
        num_total_slices = arr_internal.shape[iteration_axis]
        if effective_chunk_size <= 0 : effective_chunk_size = 1 # Should be caught if user set it. Default for auto.

        num_chunks = int(np.ceil(num_total_slices / effective_chunk_size))
        results_list = []
        slicer = [slice(None)] * arr_internal.ndim

        for i in range(num_chunks):
            start_idx = i * effective_chunk_size
            end_idx = min(start_idx + effective_chunk_size, num_total_slices)
            slicer[iteration_axis] = slice(start_idx, end_idx)
            current_data_chunk = arr_internal[tuple(slicer)]
            
            kurt_for_chunk = _calculate_kurtosis_on_chunk(current_data_chunk,
                                                          axis, fisher, bias,
                                                          float(n_global_reduction_axis))
            results_list.append(kurt_for_chunk)
        
        concat_axis = iteration_axis
        if axis is not None and iteration_axis > current_axis_normalized:
            concat_axis -= 1
        
        # Handle case where results_list might be empty if num_chunks was 0 (e.g. iteration_axis had size 0)
        if not results_list: # Should be caught by earlier n_global_reduction_axis==0 or arr_internal.size==0
            if axis is None: return np.nan
            out_shape_list = list(arr_internal.shape); del out_shape_list[current_axis_normalized]
            return np.full(tuple(out_shape_list) if out_shape_list else (), np.nan)

        return np.concatenate(results_list, axis=concat_axis)

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

def skew_old(arr):
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

def _calculate_skew_on_chunk(chunk_data: np.ndarray,
                              axis: int | None,
                              bias: bool,
                              n_for_unbiased_calc: int) -> np.ndarray | float:
    """
    Core logic to calculate skewness on a given data chunk.
    'n_for_unbiased_calc' is the original N along the reduction axis for bias correction.
    (This function remains the same as in the previous response)
    """
    current_chunk_reduction_axis_size = chunk_data.shape[axis] if axis is not None else chunk_data.size
    
    # Handle case where the effective axis length for the chunk is 0
    if current_chunk_reduction_axis_size == 0:
        if axis is None: return np.nan
        out_shape_list = list(chunk_data.shape)
        # Ensure axis is non-negative for del
        current_axis_normalized_for_chunk = axis if axis >= 0 else axis + chunk_data.ndim
        if 0 <= current_axis_normalized_for_chunk < len(out_shape_list):
            del out_shape_list[current_axis_normalized_for_chunk]
        else: # Should not happen if axis was validated for original array
            return np.nan 
        if not out_shape_list and chunk_data.ndim == 1 : return np.nan # scalar output
        return np.full(tuple(out_shape_list) if out_shape_list else (), np.nan)


    with np.errstate(divide='ignore', invalid='ignore'):
        mean_val = np.mean(chunk_data, axis=axis, keepdims=True)

        if bias:
            std_val = np.std(chunk_data, axis=axis, keepdims=True, ddof=0)
        else:
            if n_for_unbiased_calc == 1:
                std_val = np.full_like(mean_val, np.nan)
            else:
                std_val = np.std(chunk_data, axis=axis, keepdims=True, ddof=1)
        
        zero_std_mask = np.isclose(std_val, 0.0)
        std_val_safe = np.where(zero_std_mask, 1.0, std_val)
        
        deviations = chunk_data - mean_val
        z_scores = deviations / std_val_safe
        z_scores_cubed = z_scores**3
        skewness_intermediate = np.mean(z_scores_cubed, axis=axis)

    if not bias:
        if n_for_unbiased_calc <= 2:
            return np.full_like(skewness_intermediate, np.nan)
        
        correction_factor = np.sqrt(n_for_unbiased_calc * (n_for_unbiased_calc - 1.0)) / (n_for_unbiased_calc - 2.0)
        skewness_result = skewness_intermediate * correction_factor
    else:
        skewness_result = skewness_intermediate
    
    final_zero_std_mask = zero_std_mask
    if axis is not None and mean_val.ndim > skewness_result.ndim: # Check if reduction happened
        final_zero_std_mask = zero_std_mask.squeeze(axis=axis)
    
    skewness_result = np.where(np.logical_and(final_zero_std_mask, ~np.isnan(skewness_result)), 0.0, skewness_result)
    
    return skewness_result


def skew(arr: np.ndarray,
                axis: int | None = None,
                bias: bool = True,
                chunk_size: int | None = None) -> np.ndarray | float:
    """
    Calculates skewness, with optional and automatic chunking for memory efficiency.
    """
    if not isinstance(arr, np.ndarray):
        try:
            arr_internal = np.array(arr, dtype=float)
        except Exception as e:
            raise TypeError(f"Input 'arr' must be a NumPy array or convertible to one. Error: {e}")
    elif np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.bool_):
        arr_internal = arr.astype(float)
    else:
        arr_internal = arr

    if arr_internal.ndim == 0: # Scalar input
        return 0.0 if bias else np.nan

    # Normalize axis and get N for global checks
    current_axis_normalized: int
    n_global_reduction_axis: int
    if axis is None:
        current_axis_normalized = -1 # Placeholder, not used for indexing shape
        n_global_reduction_axis = arr_internal.size
    else:
        if not (-arr_internal.ndim <= axis < arr_internal.ndim):
            raise np.AxisError(f"Axis {axis} is out of bounds for array of dimension {arr_internal.ndim}")
        current_axis_normalized = axis if axis >= 0 else axis + arr_internal.ndim
        n_global_reduction_axis = arr_internal.shape[current_axis_normalized]

    # Handle empty array or zero-length reduction axis globally
    if arr_internal.size == 0 or n_global_reduction_axis == 0:
        if axis is None: return np.nan
        out_shape_list = list(arr_internal.shape)
        del out_shape_list[current_axis_normalized]
        if not out_shape_list and arr_internal.ndim == 1 : return np.nan
        return np.full(tuple(out_shape_list) if out_shape_list else (), np.nan)

    # Global check for unbiased skew with N <= 2
    if not bias and n_global_reduction_axis <= 2:
        if axis is None: return np.nan
        out_shape_list = list(arr_internal.shape)
        del out_shape_list[current_axis_normalized]
        if not out_shape_list and arr_internal.ndim == 1: return np.nan
        return np.full(tuple(out_shape_list) if out_shape_list else (), np.nan)

    # --- Chunking Logic ---
    apply_chunking = False
    effective_chunk_size = 0 # This will be the number of slices along iteration_axis
    iteration_axis = -1  # Axis along which to iterate for chunks

    # Determine iteration_axis (axis to slice for chunks)
    if axis is None or arr_internal.ndim <= 1: # No chunking for flat or 1D arrays in this scheme
        pass
    else: # ndim > 1 and axis is not None
        iteration_axis = 0 if current_axis_normalized != 0 else 1
        # (If current_axis_normalized is 0 and ndim is only 1, previous block catches it)
        # (If current_axis_normalized is 1 and ndim is only 1, previous block catches it)
        # Ensure iteration_axis is valid if ndim just became 2 and iteration_axis was picked as 1
        if iteration_axis >= arr_internal.ndim:
            iteration_axis = -1 # Cannot find a suitable different axis for iteration


    if chunk_size is not None: # User specified a chunk size
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError("User-specified chunk_size must be a positive integer.")
        if iteration_axis != -1 and arr_internal.shape[iteration_axis] > chunk_size:
            apply_chunking = True
            effective_chunk_size = chunk_size
        # Else: user chunk_size is too large or no iteration_axis, process as one.
    elif iteration_axis != -1: # Auto-chunking mode, and a valid iteration_axis exists
        if arr_internal.nbytes > _MIN_ARRAY_SIZE_FOR_AUTOCHUNK_MB * 1024 * 1024:
            bytes_per_slice_along_iteration_axis = arr_internal.nbytes / arr_internal.shape[iteration_axis]
            
            if bytes_per_slice_along_iteration_axis > 0: # Avoid division by zero for shape (...,0,...)
                target_data_chunk_bytes = (_DEFAULT_TARGET_CHUNK_PROCESSING_MEMORY_MB * 1024 * 1024) / _DEFAULT_K_MEMORY_MULTIPLIER
                desired_chunk_len = max(1, int(np.floor(target_data_chunk_bytes / bytes_per_slice_along_iteration_axis)))
                
                if arr_internal.shape[iteration_axis] > desired_chunk_len:
                    apply_chunking = True
                    effective_chunk_size = desired_chunk_len
            # Else: bytes_per_slice is 0 (e.g. shape=(100,0,10)), no chunking needed on this axis.
            print(f"Large Array Detect: Auto-Chunking at Chunk Size: {effective_chunk_size}")
        # Else: array not large enough for auto-chunking, or no iteration_axis.

    # --- Actual processing ---
    if not apply_chunking:
        return _calculate_skew_on_chunk(arr_internal, axis, bias, n_global_reduction_axis)
    else:
        # Chunking logic
        num_total_slices_on_iteration_axis = arr_internal.shape[iteration_axis]
        # Ensure effective_chunk_size is valid if auto-calculated or user-provided
        if effective_chunk_size <=0 : effective_chunk_size = 1 # Should be caught by checks before

        num_chunks = int(np.ceil(num_total_slices_on_iteration_axis / effective_chunk_size))
        
        results_list = []
        slicer = [slice(None)] * arr_internal.ndim

        for i in range(num_chunks):
            start_idx = i * effective_chunk_size
            end_idx = min(start_idx + effective_chunk_size, num_total_slices_on_iteration_axis)
            slicer[iteration_axis] = slice(start_idx, end_idx)
            
            current_data_chunk = arr_internal[tuple(slicer)]
            
            skew_for_chunk = _calculate_skew_on_chunk(current_data_chunk,
                                                      axis,
                                                      bias,
                                                      n_global_reduction_axis)
            results_list.append(skew_for_chunk)
        
        # Determine concatenation axis carefully
        concat_axis = iteration_axis
        if axis is not None and iteration_axis > current_axis_normalized :
            concat_axis -= 1 # Adjust because reduction axis was removed from shape of results
        
        return np.concatenate(results_list, axis=concat_axis)

class Storage(UserDict): # CHECKED
    """
    A dictionary-like class that allows attribute-style access to its items.
    
    This class extends UserDict to provide a more intuitive way to access dictionary
    items as attributes, allowing for both dictionary-style and attribute-style access.
    
    Examples:
        storage = Storage()
        storage['key'] = value  # Dictionary-style access
        storage.key = value     # Attribute-style access
        value = storage.key     # Attribute-style access
    
    Attributes:
        data (dict): The underlying dictionary storing the data.
    
    Methods:
        __setattr__: Sets an attribute or dictionary item.
        __getattr__: Gets an attribute or dictionary item.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __setattr__(self, name: str, value: Any) -> None:
        # Handle special attributes (like 'data') normally
        if name == "data" or name.startswith('_'):
            super().__setattr__(name, value)
        else:
            # Store other attributes in the internal dictionary
            self.data[name] = value
    
    def __getattr__(self, name: str) -> Any:
        # This is only called if the attribute wasn't found through normal means
        # Avoid infinite recursion by not using self.data.get()
        if name in self.data:
            return self.data[name]
        # Raise AttributeError for missing attributes (standard behavior)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def __bool__(self):
        return bool(self.data)

class TradeWindowOps:

    @classmethod    
    def max(cls, data: np.ndarray, axis: int = 1):
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
        return np.max(data, axis)
    
    @classmethod
    def min(cls, data: np.ndarray, axis: int = 1):
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
        return np.min(data, axis)
    
    @classmethod
    def mean(cls, data: np.ndarray, axis: int = 1):
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
        return np.mean(data, axis)
    
    @classmethod
    def median(cls, data: np.ndarray, axis: int = 1):
        """
        Calculate the median of the input data.
        """
        return np.median(data, axis)
    
    @classmethod
    def sum(cls, data: np.ndarray, axis: int = 1):
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
        return np.sum(data, axis)
    
    @classmethod
    def std(cls, data: np.ndarray, axis: int = 1):
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
        return np.std(data, axis)
    
    @classmethod
    def skew(cls, data: np.ndarray, axis: int = 1):
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
        return skew(data, axis).squeeze()
    
    @classmethod
    def kurtosis(cls, data: np.ndarray, axis: int = 1):
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
        return kurtosis(data, axis)
    
    @classmethod
    def variance(cls, data: np.ndarray, axis: int = 1):
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
        return np.var(data, axis)
    
    @classmethod
    def first(cls, data: np.ndarray, axis: int = 1):
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
        return first(data, axis)
    
    @classmethod
    def last(cls, data: np.ndarray, axis: int = 1):
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
        return last(data, axis)