# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2023-10-30

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

## [0.2.0] - 2024-03-19

### Changed
- Renamed `Feature` collection class to `FeatureSample` for better clarity and consistency
- Enhanced `Chronicle.__getitem__` method to support direct feature attribute access
- Updated documentation to reflect new class names and functionality

### Added
- New `FeatureChronicle` class for handling windowed feature data
- Improved feature access in `Chronicle` class through attribute-based syntax
- Integration of `FeatureChronicle` as return type for feature access in `Chronicle` 
- Added `from_FeatureSample` class method within the `Sample` class. 