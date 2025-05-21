# Embodied-Data

A Python library for data handling, types, and manipulation for embodied learning.

## Overview

Embodied-data is a comprehensive toolkit designed for researchers and developers working with embodied learning data. It provides utilities for handling 3D data, sensory information (images, depth), coordinate transformations, and more.

## Features

- **Coordinate Transformations**: Work with 3D coordinates, planes, and affine transformations
- **Sensory Data Processing**: Handle RGB images, depth data, point clouds
- **Geometric Operations**: PCA, quaternion operations, and other geometric utilities
- **Filtering and Segmentation**: Process point clouds with filters like RANSAC plane segmentation, KDTree filtering
- **Data Types**: Strongly typed data structures with Pydantic integration

## Documentation

- [Coordinate Systems](docs/coordinate.md) - Working with coordinate frames, poses, and planes
- [Transformations](docs/transforms.md) - Affine transformations in 2D and 3D space
- [Filters](docs/filters.md) - Point cloud filtering and processing

## Installation

### Prerequisites

- Python 3.10+
- pip

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/mbodiai/embodied-data.git
cd embodied-data

# Create a conda environment
conda create -n embdata python=3.11
conda activate embdata

# Install the package in development mode
pip install -e .
```

### With Optional Dependencies

Other Options:

```bash
# For all optional dependencies
pip install -e ".[all]"

# For full development workflow
pip install -e ".[workflow]"
```

## Development

```bash
# Install development dependencies
pip install -e ".[workflow]"

# Run tests
pytest
```

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.