# embodied data

## Data, types, pipes, manipulation for embodied learning.

[![PyPI - Version](https://img.shields.io/pypi/v/embdata.svg)](https://pypi.org/project/embdata)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/embdata.svg)](https://pypi.org/project/embdata)

-----

### A good chunk of data wrangling and exploratory data analysis that just works. See [embodied-agents](https://github.com/mbodiai/embodied-agents) for real world usage.

## Plot, filter and transform your data with ease. On any type of data structure.

[![Video Title](https://img.youtube.com/vi/L5JqM2_rIRM/0.jpg)](https://www.youtube.com/watch?v=L5JqM2_rIRM)

## Table of Contents

- [embodied data](#embodied-data)
  - [Data, types, pipes, manipulation for embodied learning.](#data-types-pipes-manipulation-for-embodied-learning)
  - [Plot, filter and transform your data with ease. On any type of data structure.](#plot-filter-and-transform-your-data-with-ease-on-any-type-of-data-structure)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Usage](#usage)
  - [License](#license)
  - [Design Decisions](#design-decisions)
  - [API Reference](#api-reference)

## Installation

```console
pip install embdata
```

## Usage

<details>
<summary><strong>Sample</strong></summary>

The `Sample` class is a flexible base model for serializing, recording, and manipulating arbitrary data.

### Key Features
- Serialization and deserialization of complex data structures
- Flattening and unflattening of nested structures
- Conversion between different formats (e.g., dict, numpy arrays, torch tensors)
- Integration with machine learning frameworks and gym spaces

### Usage Example
```python
from embdata import Sample

# Create a simple Sample
sample = Sample(x=1, y=2, z={"a": 3, "b": 4})

# Flatten the sample
flat_sample = sample.flatten()
print(flat_sample)  # [1, 2, 3, 4]

# Convert to different formats
as_dict = sample.to("dict")
as_numpy = sample.to("np")
as_torch = sample.to("pt")

# Create a random sample based on the structure
random_sample = sample.random_sample()

# Get the corresponding Gym space
space = sample.space()
```

### Methods
- `flatten()`: Flattens the nested structure into a 1D representation
- `unflatten()`: Reconstructs the original nested structure from a flattened representation
- `to(format)`: Converts the sample to different formats (dict, numpy, torch, etc.)
- `random_sample()`: Creates a random sample based on the current structure
- `space()`: Returns the corresponding Gym space for the sample

The `Sample` class provides a wide range of functionality for data manipulation, conversion, and integration with various libraries and frameworks.

</details>

<details>
<summary><strong>Image</strong></summary>

The `Image` class represents image data and provides methods for manipulation and conversion.

### Key Features
- Multiple representation formats (NumPy array, base64, file path, PIL Image)
- Easy conversion between different image formats
- Resizing and encoding capabilities
- Integration with other data processing pipelines

### Usage Example
```python
from embdata import Image
import numpy as np

# Create an Image from a numpy array
array_data = np.random.rand(100, 100, 3)
img = Image(array=array_data)

# Convert to base64
base64_str = img.base64

# Open an image from a file
img_from_file = Image.open("path/to/image.jpg")

# Resize the image
resized_img = Image(img_from_file, size=(50, 50))

# Save the image
img.save("output_image.png")
```

### Methods
- `open(path)`: Opens an image from a file path
- `save(path, encoding, quality)`: Saves the image to a file
- `show()`: Displays the image using matplotlib

### Properties
- `array`: The image as a NumPy array
- `base64`: The image as a base64 encoded string
- `path`: The file path of the image
- `pil`: The image as a PIL Image object
- `size`: The size of the image as a (width, height) tuple
- `encoding`: The encoding format of the image

The `Image` class provides a convenient interface for working with image data in various formats and performing common image operations.

</details>

<details>
<summary><strong>Trajectory</strong></summary>

The `Trajectory` class represents a time series of multidimensional data, such as robot movements or sensor readings.

### Key Features
- Representation of time series data with optional frequency information
- Methods for statistical analysis, visualization, and manipulation
- Support for resampling and filtering operations

### Usage Example
```python
from embdata import Trajectory
import numpy as np

# Create a Trajectory
data = np.random.rand(100, 3)  # 100 timesteps, 3 dimensions
traj = Trajectory(data, freq_hz=10)

# Compute statistics
stats = traj.stats()
print(stats)

# Plot the trajectory
traj.plot()

# Resample the trajectory
resampled_traj = traj.resample(target_hz=5)

# Apply a low-pass filter
filtered_traj = traj.low_pass_filter(cutoff_freq=2)

# Save the plot
traj.save("trajectory_plot.png")
```

### Methods
- `stats()`: Computes statistics for the trajectory
- `plot()`: Plots the trajectory
- `resample(target_hz)`: Resamples the trajectory to a new frequency
- `low_pass_filter(cutoff_freq)`: Applies a low-pass filter to the trajectory
- `save(filename)`: Saves the trajectory plot to a file
- `show()`: Displays the trajectory plot

The `Trajectory` class offers methods for analyzing, visualizing, and manipulating trajectory data, making it easier to work with time series data in robotics and other applications.

</details>

<details>
<summary><strong>Episode</strong></summary>

The `Episode` class provides a list-like interface for a sequence of observations, actions, and other data, particularly useful for reinforcement learning scenarios.

### Key Features
- List-like interface for managing sequences of data
- Methods for appending, iterating, and splitting episodes
- Support for metadata and frequency information
- Integration with reinforcement learning workflows

### Usage Example
```python
from embdata import Episode, Sample

# Create an Episode
episode = Episode()

# Add steps to the episode
episode.append(Sample(observation=[1, 2, 3], action=0, reward=1))
episode.append(Sample(observation=[2, 3, 4], action=1, reward=0))
episode.append(Sample(observation=[3, 4, 5], action=0, reward=2))

# Iterate over the episode
for step in episode.iter():
    print(step.observation, step.action, step.reward)

# Split the episode based on a condition
def split_condition(step):
    return step.reward > 0

split_episodes = episode.split(split_condition)

# Extract a trajectory from the episode
action_trajectory = episode.trajectory(field="action", freq_hz=10)

# Access episode metadata
print(episode.metadata)
print(episode.freq_hz)
```

### Methods
- `append(step)`: Adds a new step to the episode
- `iter()`: Returns an iterator over the steps in the episode
- `split(condition)`: Splits the episode based on a given condition
- `trajectory(field, freq_hz)`: Extracts a trajectory from the episode for a specified field
- `filter(condition)`: Filters the episode based on a given condition

### Properties
- `metadata`: Additional metadata for the episode
- `freq_hz`: The frequency of the episode in Hz

The `Episode` class simplifies the process of working with sequential data in reinforcement learning and other time-series applications.

</details>

<details>
<summary><strong>Pose3D</strong></summary>

The `Pose3D` class represents absolute coordinates for a 3D space with x, y, and theta (orientation).

### Key Features
- Representation of 3D pose with position (x, y) and orientation (theta)
- Conversion between different units (meters, centimeters, radians, degrees)
- Conversion to different formats (list, dict)

### Usage Example
```python
from embdata.geometry import Pose3D
import math

# Create a Pose3D instance
pose = Pose3D(x=1, y=2, theta=math.pi/2)
print(pose)  # Pose3D(x=1.0, y=2.0,

 theta=1.5707963267948966)

# Convert to different units
pose_cm = pose.to("cm")
print(pose_cm)  # Pose3D(x=100.0, y=200.0, theta=1.5707963267948966)

pose_deg = pose.to(angular_unit="deg")
print(pose_deg)  # Pose3D(x=1.0, y=2.0, theta=90.0)

# Convert to different formats
pose_list = pose.to("list")
print(pose_list)  # [1.0, 2.0, 1.5707963267948966]

pose_dict = pose.to("dict")
print(pose_dict)  # {'x': 1.0, 'y': 2.0, 'theta': 1.5707963267948966}
```

### Methods
- `to(container_or_unit, unit, angular_unit)`: Converts the pose to different units or formats

The `Pose3D` class provides methods for converting between different units and representations of 3D poses, making it easier to work with spatial data in various contexts.

</details>

<details>
<summary><strong>HandControl</strong></summary>

The `HandControl` class represents an action for a 7D space, including the pose of a robot hand and its grasp state.

### Key Features
- Representation of robot hand pose and grasp state
- Integration with other motion control classes
- Support for complex nested structures

### Usage Example
```python
from embdata.geometry import Pose
from embdata.motion.control import HandControl

# Create a HandControl instance
hand_control = HandControl(
    pose=Pose(position=[0.1, 0.2, 0.3], orientation=[0, 0, 0, 1]),
    grasp=0.5
)

# Access and modify the hand control
print(hand_control.pose.position)  # [0.1, 0.2, 0.3]
hand_control.grasp = 0.8
print(hand_control.grasp)  # 0.8

# Example with complex nested structure
from embdata.motion import Motion
from embdata.motion.fields import VelocityMotionField

class RobotControl(Motion):
    hand: HandControl
    velocity: float = VelocityMotionField(default=0.0, bounds=[0.0, 1.0])

robot_control = RobotControl(
    hand=HandControl(
        pose=Pose(position=[0.1, 0.2, 0.3], orientation=[0, 0, 0, 1]),
        grasp=0.5
    ),
    velocity=0.3
)

print(robot_control.hand.pose.position)  # [0.1, 0.2, 0.3]
print(robot_control.velocity)  # 0.3
```

### Attributes
- `pose`: The pose of the robot hand (Pose object)
- `grasp`: The openness of the robot hand (float, 0 to 1)

The `HandControl` class allows for easy manipulation and representation of robot hand controls in a 7D space, making it useful for robotics and motion control applications.

</details>

## License

`embdata` is distributed under the terms of the [apache-2.0](https://spdx.org/licenses/apache-2.0.html) license.

## Design Decisions

- [x] Grasp value is [-1, 1] so that the default value is 0.
- [x] Motion rather than Action to distinguish from non-physical actions.

## API Reference

## API Reference

<details>
<summary>Episode</summary>

```python
class Episode(Sample):
    """A list-like interface for a sequence of observations, actions, and/or other data.

    This class is designed to streamline exploratory data analysis and manipulation of time series data.
    It provides methods for appending, iterating, concatenating, and analyzing episodes.

    Attributes:
        steps (list[TimeStep]): A list of TimeStep objects representing the episode's steps.
        metadata (Sample | Any | None): Additional metadata for the episode.
        freq_hz (int | None): The frequency of the episode in Hz.

    Example:
        >>> from embdata.image import Image
        >>> from embdata.motion import Motion
        >>> steps = [
        ...     VisionMotorStep(
        ...         observation=ImageTask(image=Image((224, 224, 3)), task="grasp"),
        ...         action=Motion(position=[0.1, 0.2, 0.3], orientation=[0, 0, 0, 1])
        ...     ),
        ...     VisionMotorStep(
        ...         observation=ImageTask(image=Image((224, 224, 3)), task="lift"),
        ...         action=Motion(position=[0.2, 0.3, 0.4], orientation=[0, 0, 1, 0])
        ...     )
        ... ]
        >>> episode = Episode(steps=steps)
        >>> len(episode)
        2
        >>> for step in episode.iter():
        ...     print(f"Task: {step.observation.task}, Action: {step.action.position}")
        Task: grasp, Action: [0.1, 0.2, 0.3]
        Task: lift, Action: [0.2, 0.3, 0.4]

    To concatenate two episodes, use the `+` operator:
        >>> episode1 = Episode(steps=steps[:1])
        >>> episode2 = Episode(steps=steps[1:])
        >>> combined_episode = episode1 + episode2
        >>> len(combined_episode)
        2
    """

    def trajectory(self, field: str = "action", freq_hz: int = 1) -> Trajectory:
        """Extract a trajectory from the episode for a specified field.

        This method creates a Trajectory object from the specified field of each step in the episode.
        The resulting Trajectory object allows for various operations such as frequency analysis,
        subsampling, super-sampling, and min-max scaling.

        Args:
            field (str, optional): The field to extract from each step. Defaults to "action".
            freq_hz (int, optional): The frequency in Hz of the trajectory. Defaults to 1.

        Returns:
            Trajectory: The trajectory of the specified field.

        Example:
            >>> from embdata.image import Image
            >>> from embdata.motion import Motion
            >>> episode = Episode(
            ...     steps=[
            ...         VisionMotorStep(
            ...             observation=ImageTask(image=Image((224, 224, 3)), task="grasp"),
            ...             action=Motion(position=[0.1, 0.2, 0.3], orientation=[0, 0, 0, 1])
            ...         ),
            ...         VisionMotorStep(
            ...             observation=ImageTask(image=Image((224, 224, 3)), task="move"),
            ...             action=Motion(position=[0.2, 0.3, 0.4], orientation=[0, 0, 1, 0])
            ...         ),
            ...         VisionMotorStep(
            ...             observation=ImageTask(image=Image((224, 224, 3)), task="release"),
            ...             action=Motion(position=[0.3, 0.4, 0.5], orientation=[1, 0, 0, 0])
            ...         ),
            ...     ]
            ... )
            >>> action_trajectory = episode.trajectory(field="action", freq_hz=10)
            >>> action_trajectory.mean()
            array([0.2, 0.3, 0.4, 0.33333333, 0., 0.33333333, 0.33333333])
            >>> observation_trajectory = episode.trajectory(field="observation")
            >>> [step.task for step in observation_trajectory]
            ['grasp', 'move', 'release']
        """
```

</details>
## Classes

<details>
<summary><strong>Trajectory</strong></summary>

### Trajectory

A trajectory of steps representing a time series of multidimensional data.

This class provides methods for analyzing, visualizing, and manipulating trajectory data,
such as robot movements, sensor readings, or any other time-series data.

#### Attributes:
- `steps` (NumpyArray | List[Sample | NumpyArray]): The trajectory data.
- `freq_hz` (float | None): The frequency of the trajectory in Hz.
- `time_idxs` (NumpyArray | None): The time index of each step in the trajectory.
- `dim_labels` (list[str] | None): The labels for each dimension of the trajectory.
- `angular_dims` (list[int] | list[str] | None): The dimensions that are angular.

#### Methods:
- `plot`: Plot the trajectory.
- `map`: Apply a function to each step in the trajectory.
- `make_relative`: Convert the trajectory to relative actions.
- `resample`: Resample the trajectory to a new sample rate.
- `frequencies`: Plot the frequency spectrogram of the trajectory.
- `frequencies_nd`: Plot the n-dimensional frequency spectrogram of the trajectory.
- `low_pass_filter`: Apply a low-pass filter to the trajectory.
- `stats`: Compute statistics for the trajectory.
- `transform`: Apply a transformation to the trajectory.

#### Example:
```python
import numpy as np
from embdata.trajectory import Trajectory

# Create a simple 2D trajectory
steps = np.array([[0, 0], [1, 1], [2, 0], [3, 1], [4, 0]])
traj = Trajectory(steps, freq_hz=10, dim_labels=['X', 'Y'])

# Plot the trajectory
traj.plot().show()

# Compute and print statistics
print(traj.stats())

# Apply a low-pass filter
filtered_traj = traj.low_pass_filter(cutoff_freq=2)
filtered_traj.plot().show()
```

</details>

<details>
<summary><strong>Pose3D</strong></summary>

### Pose3D

Absolute coordinates for a 3D space representing x, y, and theta.

This class represents a pose in 3D space with x and y coordinates for position
and theta for orientation.

#### Attributes:
- `x` (float): X-coordinate in meters.
- `y` (float): Y-coordinate in meters.
- `theta` (float): Orientation angle in radians.

#### Methods:
- `to(container_or_unit=None, unit="m", angular_unit="rad", **kwargs) -> Any`: Convert the pose to a different unit or container.

#### Example:
```python
import math
from embdata.geometry import Pose3D

# Create a Pose3D instance
pose = Pose3D(x=1, y=2, theta=math.pi/2)
print(pose)  # Output: Pose3D(x=1.0, y=2.0, theta=1.5707963267948966)

# Convert to centimeters
pose_cm = pose.to("cm")
print(pose_cm)  # Output: Pose3D(x=100.0, y=200.0, theta=1.5707963267948966)

# Convert theta to degrees
pose_deg = pose.to(angular_unit="deg")
print(pose_deg)  # Output: Pose3D(x=1.0, y=2.0, theta=90.0)

# Convert to a list
pose_list = pose.to("list")
print(pose_list)  # Output: [1.0, 2.0, 1.5707963267948966]

# Convert to a dictionary
pose_dict = pose.to("dict")
print(pose_dict)  # Output: {'x': 1.0, 'y': 2.0, 'theta': 1.5707963267948966}
```

</details>

<details>
<summary><strong>Sample</strong></summary>

### Sample

A base model class for serializing, recording, and manipulating arbitrary data.

This class provides a flexible and extensible way to handle complex data structures,
including nested objects, arrays, and various data types. It offers methods for
flattening, unflattening, converting between different formats, and working with
machine learning frameworks.

#### Attributes:
- `model_config` (ConfigDict): Configuration for the model, including settings for validation, extra fields, and arbitrary types.

#### Methods:
- `__init__(self, item=None, **data)`: Initialize a Sample instance.
- `schema(self, include_descriptions=False)`: Get a simplified JSON schema of the data.
- `to(self, container)`: Convert the Sample instance to a different container type.
- `flatten(self, output_type="list", non_numerical="allow", ignore=None, sep=".", to=None)`: Flatten the Sample instance into a one-dimensional structure.
- `unflatten(cls, one_d_array_or_dict, schema=None)`: Unflatten a one-dimensional array or dictionary into a Sample instance.
- `space(self)`: Return the corresponding Gym space for the Sample instance.
- `random_sample(self)`: Generate a random Sample instance based on its attributes.

#### Example:
```python
from embdata import Sample
import numpy as np

# Create a simple Sample instance
sample = Sample(x=1, y=2, z={"a": 3, "b": 4}, extra_field=5)

# Flatten the sample
flat_sample = sample.flatten()
print(flat_sample)  # Output: [1, 2, 3, 4, 5]

# Get the schema
schema = sample.schema()
print(schema)

# Unflatten a list back to a Sample instance
unflattened_sample = Sample.unflatten(flat_sample, schema)
print(unflattened_sample)  # Output: Sample(x=1, y=2, z={'a': 3, 'b': 4}, extra_field=5)

# Create a complex nested structure
nested_sample = Sample(
    image=Sample(
        data=np.random.rand(32, 32, 3),
        metadata={"format": "RGB", "size": (32, 32)}
    ),
    text=Sample(
        content="Hello, world!",
        tokens=["Hello", ",", "world", "!"],
        embeddings=np.random.rand(4, 128)
    ),
    labels=["greeting", "example"]
)

# Get the schema of the nested structure
nested_schema = nested_sample.schema()
print(nested_schema)
```

</details>

<details>
<summary><strong>Image</strong></summary>

### Image

An image sample that can be represented in various formats.

The image can be represented as a NumPy array, a base64 encoded string, a file path, a PIL Image object,
or a URL. The image can be resized to and from any size and converted to and from any supported format.

#### Attributes:
- `array` (Optional[np.ndarray]): The image represented as a NumPy array.
- `base64` (Optional[Base64Str]): The base64 encoded string of the image.
- `path` (Optional[FilePath]): The file path of the image.
- `pil` (Optional[PILImage]): The image represented as a PIL Image object.
- `url` (Optional[AnyUrl]): The URL of the image.
- `size` (Optional[tuple[int, int]]): The size of the image as a (width, height) tuple.
- `encoding` (Optional[Literal["png", "jpeg", "jpg", "bmp", "gif"]]): The encoding of the image.

#### Methods:
- `from_base64(base64_str: str, encoding: str, size=None, make_rgb=False) -> "Image"`: Decodes a base64 string to create an Image instance.

#### Example:
```python
from embdata import Image

# Create an Image instance from a URL
image_url = Image("https://example.com/image.jpg")

# Create an Image instance from a file path
image_file = Image("/path/to/image.jpg")

# Create an Image instance from a base64 string
base64_str = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD/4Q3zaHR0cDovL25zLmFkb2JlLmNvbS94YXAvMS4wLwA"
image_base64 = Image(base64_str)

# Convert PNG to JPEG
jpeg_from_png = Image("path/to/image.png", encoding="jpeg")

# Resize an image
resized_image = Image(image_url, size=(224, 224))

# Access different representations of the image
pil_image = image_file.pil
array = image_file.array
base64 = image_file.base64

# Create an Image instance from a base64 string
base64_str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAACklEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg=="
image = Image.from_base64(base64_str, encoding="png", size=(1, 1))
print(image.size)  # Output: (1, 1)

# Example with complex nested structure
nested_data = {
    "image": Image.from_base64(base64_str, encoding="png"),
    "metadata": {
        "text": "A small red square",
        "tags": ["red", "square", "small"]
    }
}
print(nested_data["image"].size)  # Output: (1, 1)
print(nested_data["metadata"]["text"])  # Output: A small red square
```

</details>

<details>
<summary><strong>Motion</strong></summary>

### Motion

Base class for defining motion-related data structures.

This class extends the Coordinate class and provides a foundation for creating
motion-specific data models. It does not allow extra fields and enforces
validation of motion type, shape, and bounds.

#### Attributes:
- Inherited from Coordinate

#### Usage:
Subclasses of Motion should define their fields using MotionField or its variants
(e.g., AbsoluteMotionField, VelocityMotionField) to ensure proper validation and
type checking.

#### Example:
```python
from embdata.motion import Motion
from embdata.motion.fields import VelocityMotionField

class Twist(Motion):
    x: float = VelocityMotionField(default=0.0, bounds=[-1.0, 1.0])
    y: float = VelocityMotionField(default=0.0, bounds=[-1.0, 1.0])
    z: float = VelocityMotionField(default=0.0, bounds=[-1.0, 1.0])
    roll: float = VelocityMotionField(default=0.0, bounds=["-pi", "pi"])
    pitch: float = VelocityMotionField(default=0.0, bounds=["-pi", "pi"])
    yaw: float = VelocityMotionField(default=0.0, bounds=["-pi", "pi"])

# Create a Twist instance
twist = Twist(x=0.5, y=-0.3, z=0.1, roll=0.2, pitch=-0.1, yaw=0.8)
print(twist)
# Output: Twist(x=0.5, y=-0.3, z=0.1, roll=0.2, pitch=-0.1, yaw=0.8)

# Attempt to create an invalid Twist instance
try:
    invalid_twist = Twist(x=1.5, y=-0.3, z=0.1, roll=0.2, pitch=-0.1, yaw=0.8)
except ValueError as e:
    print(f"Error: {e}")
    # Output: Error: x value 1.5 is not within bounds [-1.0, 1.0]

# Example with complex nested structure
class RobotMotion(Motion):
    twist: Twist
    gripper: float = VelocityMotionField(default=0.0, bounds=[0.0, 1.0])

robot_motion = RobotMotion(
    twist=Twist(x=0.2, y=0.1, z=0.0, roll=0.0, pitch=0.0, yaw=0.1),
    gripper=0.5
)
print(robot_motion)
# Output: RobotMotion(twist=Twist(x=0.2, y=0.1, z=0.0, roll=0.0, pitch=0.0, yaw=0.1), gripper=0.5)
```

Note: The Motion class is designed to work with complex nested structures.
It can handle various types of motion data, including images and text,
as long as they are properly defined using the appropriate MotionFields.

</details>

<details>
<summary><strong>HandControl</strong></summary>

### HandControl

Action for a 7D space representing x, y, z, roll, pitch, yaw, and openness of the hand.

This class represents the control for a robot hand, including its pose and grasp state.

#### Attributes:
- `pose` (Pose): The pose of the robot hand, including position and orientation.
- `grasp` (float): The openness of the robot hand, ranging from 0 (closed) to 1 (open).

#### Example:
```python
from embdata.geometry import Pose
from embdata.motion.control import HandControl

# Create a HandControl instance
hand_control = HandControl(
    pose=Pose(position=[0.1, 0.2, 0.3], orientation=[0, 0, 0, 1]),
    grasp=0.5
)

# Access and modify the hand control
print(hand_control.pose.position)  # Output: [0.1, 0.2, 0.3]
hand_control.grasp = 0.8
print(hand_control.grasp)  # Output: 0.8

# Example with complex nested structure
from embdata.motion import Motion
from embdata.motion.fields import VelocityMotionField

class RobotControl(Motion):
    hand: HandControl
    velocity: float = VelocityMotionField(default=0.0, bounds=[0.0, 1.0])

robot_control = RobotControl(
    hand=HandControl(
        pose=Pose(position=[0.1, 0.2, 0.3], orientation=[0, 0, 0, 1]),
        grasp=0.5
    ),
    velocity=0.3
)

print(robot_control.hand.pose.position)  # Output: [0.1, 0.2, 0.3]
print(robot_control.velocity)  # Output: 0.3
```

</details>

<details>
<summary><strong>to_features_dict</strong></summary>

### to_features_dict

Convert a dictionary to a Datasets Features object.

This function recursively converts a nested dictionary into a format compatible with
Hugging Face Datasets' Features. It handles various data types including strings,
integers, floats, lists, and PIL Images.

#### Arguments:
- `indict`: The input to convert. Can be a dictionary, string, int, float, list, tuple, numpy array, or PIL Image.
- `exclude_keys`: A set of keys to exclude from the conversion. Defaults to None.

#### Returns:
A dictionary representation of the Features object for Hugging Face Datasets.

#### Raises:
ValueError: If an empty list is provided or if the input type is not supported.

#### Examples:
```python
# Simple dictionary conversion
to_features_dict({"name": "Alice", "age": 30})
# Output: {'name': Value(dtype='string', id=None), 'age': Value(dtype='int64', id=None)}

# List conversion
to_features_dict({"scores": [85, 90, 95]})
# Output: {'scores': [Value(dtype='int64', id=None)]}

# Numpy array conversion
import numpy as np
to_features_dict({"data": np.array([1, 2, 3])})
# Output: {'data': [Value(dtype='int64', id=None)]}

# PIL Image conversion
from PIL import Image
img = Image.new("RGB", (60, 30), color="red")
to_features_dict({"image": img})
# Output: {'image': Image(decode=True, id=None)}

# Nested structure with image and text
complex_data = {
    "user_info": {
        "name": "John Doe",
        "age": 28
    },
    "posts": [
        {
            "text": "Hello, world!",
            "image": Image.new("RGB", (100, 100), color="blue"),
            "likes": 42
        },
        {
            "text": "Another post",
            "image": Image.new("RGB", (200, 150), color="green"),
            "likes": 17
        }
    ]
}
features = to_features_dict(complex_data)
print(features)
# Output:
# {
#     'user_info': {
#         'name': Value(dtype='string', id=None),
#         'age': Value(dtype='int64', id=None)
#     },
#     'posts': [
#         {
#             'text': Value(dtype='string', id=None),
#             'image': Image(decode=True, id=None),
#             'likes': Value(dtype='int64', id=None)
#         }
#     ]
# }
```

</details>
## Classes

<details>
<summary><strong>HandControl</strong></summary>

### HandControl

Action for a 7D space representing x, y, z, roll, pitch, yaw, and openness of the hand.

This class represents the control for a robot hand, including its pose and grasp state.

#### Attributes:
- `pose` (Pose): The pose of the robot hand, including position and orientation.
- `grasp` (float): The openness of the robot hand, ranging from 0 (closed) to 1 (open).

#### Example:
```python
from embdata.geometry import Pose
from embdata.motion.control import HandControl

# Create a HandControl instance
hand_control = HandControl(
    pose=Pose(position=[0.1, 0.2, 0.3], orientation=[0, 0, 0, 1]),
    grasp=0.5
)

# Access and modify the hand control
print(hand_control.pose.position)  # Output: [0.1, 0.2, 0.3]
hand_control.grasp = 0.8
print(hand_control.grasp)  # Output: 0.8

# Example with complex nested structure
from embdata.motion import Motion
from embdata.motion.fields import VelocityMotionField

class RobotControl(Motion):
    hand: HandControl
    velocity: float = VelocityMotionField(default=0.0, bounds=[0.0, 1.0])

robot_control = RobotControl(
    hand=HandControl(
        pose=Pose(position=[0.1, 0.2, 0.3], orientation=[0, 0, 0, 1]),
        grasp=0.5
    ),
    velocity=0.3
)

print(robot_control.hand.pose.position)  # Output: [0.1, 0.2, 0.3]
print(robot_control.velocity)  # Output: 0.3
```

</details>
# embdata

embdata is a Python library for handling and processing various types of data in robotics and AI applications.

## Classes

<details>
<summary><strong>Episode</strong></summary>

### Episode

The `Episode` class provides a list-like interface for a sequence of observations, actions, and/or other data. It's designed to streamline exploratory data analysis and manipulation of time series data.

#### Usage

```python
from embdata import Episode, TimeStep

# Create an episode
episode = Episode()

# Add steps to the episode
step1 = TimeStep(observation={'position': [0, 0, 0]}, action={'move': [1, 0, 0]})
step2 = TimeStep(observation={'position': [1, 0, 0]}, action={'move': [0, 1, 0]})

episode.append(step1)
episode.append(step2)

# Access steps
first_step = episode[0]
print(first_step.observation)  # Output: {'position': [0, 0, 0]}

# Iterate through steps
for step in episode:
    print(step.action)

# Get episode length
print(len(episode))  # Output: 2
```

</details>

<details>
<summary><strong>Image</strong></summary>

### Image

The `Image` class represents an image sample that can be represented in various formats, including NumPy arrays, base64 encoded strings, file paths, PIL Images, or URLs.

#### Usage

```python
from embdata import Image

# Create an Image from a file
img = Image.open('path/to/image.jpg')

# Resize the image
resized_img = img.resize((224, 224))

# Convert to different formats
numpy_array = resized_img.array
base64_string = resized_img.base64
pil_image = resized_img.pil

# Save the image
resized_img.save('path/to/save/resized_image.jpg', encoding='jpeg', quality=95)
```

</details>

<details>
<summary><strong>Sample</strong></summary>

### Sample

The `Sample` class is a base model for serializing, recording, and manipulating arbitrary data. It provides methods for flattening, unflattening, and converting between different formats.

#### Usage

```python
from embdata import Sample

class CustomSample(Sample):
    name: str
    value: float
    nested: dict

# Create a sample
sample = CustomSample(name="example", value=3.14, nested={"key": "value"})

# Flatten the sample
flattened = sample.flatten()
print(flattened)  # Output: {'name': 'example', 'value': 3.14, 'nested.key': 'value'}

# Get field info
field_info = sample.model_field_info('value')
print(field_info)  # Output: {'type': 'float', ...}
```

</details>

<details>
<summary><strong>Trajectory</strong></summary>

### Trajectory

The `Trajectory` class represents a trajectory of steps, typically used for time series of multidimensional data such as robot movements or sensor readings.

#### Usage

```python
import numpy as np
from embdata import Trajectory

# Create a trajectory
data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
traj = Trajectory(steps=data, freq_hz=10, dim_labels=['x', 'y', 'z'])

# Access data
print(traj.array)  # Output: [[1 2 3] [4 5 6] [7 8 9]]

# Get statistics
stats = traj.stats()
print(stats.mean)  # Output: [4. 5. 6.]
print(stats.std)   # Output: [3. 3. 3.]

# Slice the trajectory
sliced_traj = traj[1:3]
print(sliced_traj.array)  # Output: [[4 5 6] [7 8 9]]
```

</details>

<details>
<summary><strong>Motion</strong></summary>

### Motion

The `Motion` class is a base class for defining motion-related data structures. It extends the `Coordinate` class and provides a foundation for creating motion-specific data models.

#### Usage

```python
from embdata.motion import Motion, VelocityMotionField

class Twist(Motion):
    x: float = VelocityMotionField(default=0.0, bounds=[-1.0, 1.0])
    y: float = VelocityMotionField(default=0.0, bounds=[-1.0, 1.0])
    z: float = VelocityMotionField(default=0.0, bounds=[-1.0, 1.0])
    roll: float = VelocityMotionField(default=0.0, bounds=["-pi", "pi"])
    pitch: float = VelocityMotionField(default=0.0, bounds=["-pi", "pi"])
    yaw: float = VelocityMotionField(default=0.0, bounds=["-pi", "pi"])

# Create a Twist motion
twist = Twist(x=0.5, y=-0.3, z=0.1, roll=0.2, pitch=-0.1, yaw=0.8)

print(twist)  # Output: Twist(x=0.5, y=-0.3, z=0.1, roll=0.2, pitch=-0.1, yaw=0.8)

# Access individual fields
print(twist.x)  # Output: 0.5

# Validate bounds
try:
    invalid_twist = Twist(x=1.5)  # This will raise a ValueError
except ValueError as e:
    print(f"Validation error: {e}")
```

</details>
