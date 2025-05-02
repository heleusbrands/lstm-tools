# API Reference

This section provides detailed documentation for all modules, classes, and functions in the LSTM Tools library.

## Core Classes

- [Feature](feature.md): A single data point with a name attribute
- [TimeFrame](timeframe.md): A 1D array of distinct Feature objects for a snapshot in time
- [Sample](sample.md): A 2D array of TimeFrame objects for a sequence
- [Chronicle](chronicle.md): A 3D array of windowed Sample objects
- [FeatureSample](featuresample.md): A 1D array of similar Feature objects over time
- [FeatureChronicle](featurechronicle.md): A 2D array of windowed FeatureSample objects

## Settings and Configuration

- [WindowSettings](window_settings.md): Configuration for window operations
- [HFWindowSettings](window_settings.md#hfwindowsettings): Settings for historical-future windowing

## Utility Modules

- [Utils](utils.md): Helper functions for windowing and calculations
- [Logger](logger.md): Logging utilities for the library
- [Exceptions](exceptions.md): Custom exception classes

## Base Classes

- [FrameBase](frame_base.md): Base class for all array-like objects
- [WindowType](window_settings.md#windowtype): Enumeration for window types 