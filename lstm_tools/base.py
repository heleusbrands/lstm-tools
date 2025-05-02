
import plotly.graph_objects as go
import plotly.io as pio
from enum import Enum
import numpy as np

# Create a custom template
my_template = go.layout.Template(
    layout=dict(
        paper_bgcolor='#121314',  # Background color of the entire figure
        plot_bgcolor='#121314',      # Background color of the plotting area
        font=dict(family="Arial", size=12, color="#2fe0b7"), # Example font setting
        # Add other global styling here (e.g., axis colors, gridlines)
        coloraxis=dict( # Default color axis settings
            colorscale='viridis', # Apply the gradient here
        )
    )
)

# Set the custom template as the default
pio.templates["bayesian"] = my_template
pio.templates.default = "bayesian"


class WindowType(Enum):
    """
    Enumeration for specifying the type of window in time series analysis.

    Attributes
    ----------
    future : int
        Represents a window looking into future data points.
    historical : int
        Represents a window looking into historical data points.
    """
    future = 1
    historical = 2


class FrameBase(np.ndarray):
    """
    Base class for creating custom array-like objects.

    This class provides a basic structure for creating custom array-like objects
    with additional attributes and methods.

    Parameters
    ----------
    input_data : np.ndarray
        Input data to create the custom array-like object.
    cols : list
        List of column names for the custom array-like object.
    idx : int, optional
        Index of the custom array-like object, by default None.
    name : str, optional
        Name of the custom array-like object, by default None.
    dtype : numpy.dtype, optional
        Data type of the custom array-like object, by default None.

    Attributes
    ----------
    _cols : list
        List of column names for the custom array-like object.
    _idx : int
        Index of the custom array-like object.
    _level : int
        Level of the custom array-like object.
    """
    
    subtype = None
    level = 0
    _idx = None
    format_input_feature = lambda input_data, cols, subtype: [subtype(input_data[i], name) for i, name in enumerate(cols)]
    format_input_sequence = lambda input_data, cols, subtype: [subtype(data, cols = cols, idx = i) for i, data in enumerate(input_data)]

    def __new__(cls, input_data, cols, idx=None, name=None,dtype=None):
        # Create a new instance of the array
        dtype = np.dtype(cls) if not dtype else np.dtype(dtype)
        if not isinstance(input_data[0], cls.subtype):
            if isinstance(input_data[0], float): input_data = cls.format_input_feature(input_data, cols, cls.subtype)
            elif isinstance(input_data[0], list) or isinstance(input_data[0], np.ndarray): input_data = cls.format_input_sequence(input_data, cols, cls.subtype)
            else:
                raise ValueError(f'Unsupported input type: {type(input_data[0])}')

        obj = np.array(input_data, dtype=object, subok=True).view(cls)
        
        obj._cols = cols
        obj._idx = idx
        obj._level = 0
        return obj

    def __array_finalize__(self, obj):
        # This method is called when a new array is created from an existing one
        if obj is None:
            return
        self.scaler = getattr(obj, 'scaler', None)
        self._cols = getattr(obj, '_cols', None)
        self._idx = getattr(obj, '_idx', None)
        self._level = getattr(obj, '_level', 0)
        self._shape = getattr(obj, '_shape', None)

    @property
    def shape(self):
        if not self._shape and len(self) != 0: self._shape = (len(self), len(self[0]))
        return self._shape
    
    @shape.setter
    def shape(self, value):
        self._shape = value

    def __hrepr__(self, level=None):
        # sourcery skip: use-fstring-for-concatenation
        if not level: level = self._level
        h = '|\t'.expandtabs(4) * (level + 1)
        p = f'\n{h}'
        e = '-' * len(self.shape)
        if hasattr(self[0], '__hrepr__'):
            if len(self) < 100:
                return f'{self.__class__.__name__}[{self._idx}]' + p + f'\n{h}'.join([f.__hrepr__(level + 1) for f in self]) + f'\n{e}'
            else:
                return f'{self.__class__.__name__}[{self._idx}]' + p + f'\n{h}'.join([self[f].__hrepr__(level + 1) for f in range(2)]) + f'\n{e}' + '\n...'
        return f'{self.__class__.__name__}[{self._idx}]' + p + f'\n{h}'.join([f.__repr__() for f in self]) + f'\n{e}'

    def __repr__(self):
        return self.__hrepr__(0)