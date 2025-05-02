from setuptools import setup, find_packages

setup(
    name="lstm-tools",
    version="0.2.0",
    author="Rose Bloom Research Co",
    author_email="rosebloomresearch@gmail.com",
    description="A high-performance library for dynamically handling sequential data",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/heleusbrands/lstm-tools",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.26.2",
        "pandas>=2.1.4",
        "scikit-learn>=1.5.1",
        "torch>=2.6.0",
        "tensorflow>=2.17.1",
        "plotly>=5.24.1",
        "line_profiler>=4.2.1",
    ],
    keywords=["lstm", "time series", "sequential data", "windowing", "data compression", "dataframe", "numpy", "torch", "tensorflow", "pytorch", "scikit-learn", "plotly", "line_profiler"],
    license="GPL-3.0-only",
) 