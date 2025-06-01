from setuptools import setup, find_packages

setup(
    name="lstm-tools",
    version="0.4.0",
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
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "torch",
        "tensorflow",
        "plotly",
        "line_profiler",
    ],
    keywords=["lstm", "time series", "sequential data", "windowing", "data compression", "dataframe", "numpy", "torch", "tensorflow", "pytorch", "scikit-learn", "plotly", "line_profiler"],
    license="GPL-3.0-only",
) 