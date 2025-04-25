# LSTM Tools Documentation
<span style="color:#E83A6B;"> by Bloom Research </span>

Welcome to the official documentation for LSTM Tools, a high-performance library for dynamically handling sequential data.

## Introduction

LSTM Tools is designed to make working with time series data intuitive and efficient, particularly when preparing data for machine learning models. Built on numpy's powerful array operations, it provides a hierarchical data structure that makes complex operations simple. It takes a dynamic approach, by changing the functionality of the array based on the current shape of the data.

## Key Features

- **Intuitive Object Hierarchy**: From individual data points (Feature) to complete windowed datasets (Chronicle)
- **Attribute-based Access**: Access features by name using standard Python attribute notation
- **Efficient Windowing**: Fast creation of sliding windows using numpy's stride tricks
- **Seamless ML Integration**: Direct conversion to PyTorch and TensorFlow tensors
- **Lazy Instantiation**: Objects are created only when needed to minimize memory usage

## Installation

```bash
pip install lstm-tools
```

For development installation:

```bash
git clone https://github.com/heleusbrands/lstm-tools.git
cd lstm-tools
pip install -e .
```

## Quick Navigation

- [Getting Started](tutorials/getting_started.md): First steps with LSTM Tools
- [API Reference](api/index.md): Detailed documentation of all classes and methods
- [Tutorials](tutorials/index.md): Practical guides for common tasks
- [Examples](examples/index.md): Real-world usage examples
- [Changelog](changelog.md): Release history and changes

## Note from Author

*"This was a personal tool that I created for my own use during some research, which was created out of frustration with the other tools available. Pandas, as amazing as it is, was not very intuitive for handling complex sequential data. The universal approach made it difficult/repetitive to get at the capabilities I needed to access frequently when switching between array shapes. I switched to plain numpy arrays, but soon became frustrated at having to keep track of where each feature was stored, and the confusion caused by dealing with pure numeric representations. The whole process with both libraries felt very 'un-pythonic'. Enter LSTM Tools - Arrays that change structure and methods depending on the current situation."* 