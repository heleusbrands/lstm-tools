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