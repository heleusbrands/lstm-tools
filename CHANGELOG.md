# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-05

### Added
- Initial release of lstm-tools library
- Feature class implementation
- FeatureSample collection implementation
- TimeFrame class implementation
- Sample class implementation
- Chronicle class implementation
- Basic windowing functionality
- Basic feature compression
- Data visualization tools
- Serialization/deserialization utilities
- Custom error handling system
- Logging utilities
- Example scripts 

## [0.2.0] - 2025-05

### Changed
- Renamed `Feature` collection class to `FeatureSample` for better clarity and consistency
- Enhanced `Chronicle.__getitem__` method to support direct feature attribute access
- Updated documentation to reflect new class names and functionality

### Added
- New `FeatureChronicle` class for handling windowed feature data
- Improved feature access in `Chronicle` class through attribute-based syntax
- Integration of `FeatureChronicle` as return type for feature access in `Chronicle` 
- Added `from_FeatureSample` class method within the `Sample` class. 

## [0.3.0] - 2025-05

### Changed
- Streamlined the compression workflow
- Fixed several errors in the compression workflows
- Streamlined several other workflows

### Added
- Line plot capabilities added to the FeatureSample class
- Merging capabilities for Merging multiple Samples to a Chronicle via their TimeFrames.
- Merging capabilities for Merging Multiple Samples to a single Sample
- New Storage class added to Chronicles to allow compressors for each FeatureChronicle to be added to top level class, and then passed to FeatureChronicles
- Added complete support for creating expanding window (i.e. subwindows over a larger window), making creating and compressed timeframes with different granularity views easy.

### Issues
- Need to update the documentation to reflect the new changes.

## [0.3.2] - 2025-05

### Changed
- Fixed Issue with scaler not being adopted properly by the Chronicle class

### Added
- New MetaData base class to help with passing information back and forth between class type
- New custom squeeze() method added to Chronicle class to handle converting instances with an extra unneeded dimension back to Sample
- Added the parameter "source" to all FrameBase subclass's __new__ methods, to allow for obtaining MetaData objects

### Issues
- Need to update the documentation to reflect the new changes.

## [0.3.4] - 2025-05

### Changed
- Fixed error when squeezing a Chronicle with an empty dimension.

### Added
- `time` property added to the `TimeFrame`, `Sample`, and `Chronicle` classes.
- Added more robust window settings, with the addition of an `offset` setting for offsetting a future window from the historical.
- Improved the `hf_sliding_window`method, within the `utils` module, to account for the new `offset` window setting and updated the Sample class.
- Added `to_numpy()` method to `TimeFrame`, `Sample`, and `Chronicle` classes

### Issues
- Need to update the documentation to reflect the new changes.

## [0.3.5] - 2025-05

### Changed
- Fixed error issue with scalers not being passed to a Sample created via `Sample.fromFeatureSamples`

### Added
- HistoricalWindowSettings and FutureWindowSettings are now two separate classes, to accomodate varying parameters
- `hf_sliding_window()` method expanded to add an addition `h_spacing` arguement, to allow for spaced element selection
- Sample class updated to accomodate and integrate the new `spacing` setting of the HistoricalWindowSettings class

### Issues
- Need to update the documentation to reflect the new changes.

## [0.3.6] - 2025-05

### Changed
- Fixed import error, due to previous changes not being correctly updated in the `__init__`

### Issues
- Need to update the documentation to reflect the new changes.

## [0.3.7] - 2025-05

### Changed
- Fixed an error with the Sample class method `fromFeatureSample` incorrectly passing scaler to Sample instance. Added `scaler` arguement.

### Issues
- Need to update the documentation to reflect the new changes.

## [0.3.8] - 2025-05

### Added
- Compressors added to a FeatureChronicle are now passed back to the parent Chronicle.
- New SubWindowsSettings class added and integrated
- sub_window_over_axis() method added to Chronicle class - will eventually replace subwindow_over_samples
- New compress_features_to_sample method added.
- `to_numpy` method added across all FrameBase Subclasses


### Changed
- Optimization of window creation and array form switching
- Major optimizations of TradeOps compression function, to utilize vector opterations.
- Custom `skew` and `kurtosis` method's expaded and optimized to be able to handle extremely large arrays efficiently, with auto-batching.

### Issues
- Need to update the documentation to reflect the new changes.

## [0.3.9] - 2025-05

### Added
- `Chronicle`
-- `scale` method
-- `scale_chronicle` method
-- `scale_sample` method
-- `scale_array` method
-- `unscale_chronicle` method
-- `unscale_sample` method
-- `unscale_array` method
-- `_to_2d` method
-- `_to_3d` method
-- `split` method

- `Sample`
-- `scale` method
-- `unscale` method
-- `splilt` method


### Changed
- Now both the Sample and Chronicle classes now have default class Scalers
- Scalers are now initialized with "copy = False" to avoid unnessary memory usage for large arrays.

### Fixed
- dtypes specified during class initialization are now properly incorperated in all senarios.

### Issues
- Need to update the documentation to reflect the new changes.