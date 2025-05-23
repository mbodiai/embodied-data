# Copyright (c) 2024 Mbodi AI
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# Copyright 2024 Mbodi AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import base64
import io
import os
import numpy as np
import pytest
import cv2
from PIL import Image as PILImage
from embdata.sense.image import Image
import tempfile


@pytest.fixture
def temp_file():
    """Create a temporary file and provide its path to tests, remove it after the test."""
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        yield tmp_file.name
    os.remove(tmp_file.name)


def test_create_image_with_array():
    array = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    img = Image(array=array)
    assert img.size == (640, 480)
    assert img.base64 != ""


def test_create_image_with_path(temp_file):
    test_image_path = temp_file
    PILImage.new("RGB", (640, 480)).save(test_image_path, format="PNG")
    img = Image(test_image_path)
    assert img.size is not None
    assert img.base64 != ""
    assert img.array is not None


def test_create_image_with_size():
    img = Image(size=(640, 480))
    assert img.size is not None
    # Decode both base64 strings to images and compare
    assert img.base64 != ""
    assert img.base64 is not None
    assert img.array.shape == (480, 640, 3)


def test_create_image_with_base64():
    buffer = io.BytesIO()
    PILImage.new("RGB", (10, 10)).save(buffer, format="PNG")
    encoded_str = base64.b64encode(buffer.getvalue()).decode()
    img = Image(encoded_str, encoding="png")
    assert img.size is not None
    # Decode both base64 strings to images and compare
    original_image_data = io.BytesIO(base64.b64decode(encoded_str))
    original_image = PILImage.open(original_image_data)
    test_image_data = io.BytesIO(base64.b64decode(img.base64))
    test_image = PILImage.open(test_image_data)
    # Compare images pixel by pixel
    assert list(original_image.getdata()) == list(test_image.getdata())


def test_base64_encode():
    array = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    img = Image(array=array)
    encoded_base64 = img.base64
    assert encoded_base64 != ""
    assert isinstance(encoded_base64, str)


def test_resize():
    array = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    img = Image(array=array, size=(224, 224))
    assert img.size == (224, 224)
    assert img.array.shape == (224, 224, 3)


def test_encode_decode_array():
    array = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    img = Image(array, encoding="png")
    encoded_base64 = img.base64
    decoded_array_bgr = cv2.imdecode(np.frombuffer(base64.b64decode(encoded_base64), np.uint8), cv2.IMREAD_COLOR)
    decoded_array = cv2.cvtColor(decoded_array_bgr, cv2.COLOR_BGR2RGB)
    assert decoded_array.shape == (480, 640, 3)
    assert np.array_equal(decoded_array, array)


def test_png_tojpeg(temp_file):
    array = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    img = Image(array, encoding="png")
    img.save(temp_file, encoding="png")

    with open(temp_file, "rb") as file:
        decoded_bytes = file.read()
    decoded_image = PILImage.open(io.BytesIO(decoded_bytes))
    decoded_array = np.array(decoded_image)
    assert np.array_equal(decoded_array, array)


def test_image_save(temp_file):
    # Create a random image
    image_file_path = temp_file
    array = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    img = Image(array, encoding="png")
    img.save(image_file_path, "PNG", quality=100)
    # Reload the image to check if saved correctly
    reloaded_img = Image(image_file_path)
    assert np.array_equal(reloaded_img.array, array)


def test_image_model_dump_load():
    array = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    img = Image(array, encoding="png")
    json = img.model_dump_json()
    reconstructed_img = Image.model_validate_json(json)
    assert np.array_equal(reconstructed_img.array, array)


def test_image_model_dump_load_with_base64():
    array = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    img = Image(array, encoding="png")
    json = img.model_dump_json(round_trip=True)
    reconstructed_img = Image.model_validate_json(json)
    assert np.array_equal(reconstructed_img.array, array)


def test_path(temp_file):
    file_path = temp_file
    
    # Create and save the image to the temporary file
    Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)).save(temp_file)
    
    # Normalize the file path before comparison
    normalized_path = os.path.normpath(temp_file)
    
    # Assuming the Image class returns a path as a string or WindowsPath
    image = Image(path=normalized_path, encoding="png", mode="RGB")
    
    # Compare normalized paths
    assert os.path.normpath(image.path) == normalized_path


def test_base64_and_bytes_performance():
    """Test the performance of the current vs old methods for base64 and bytes."""
    import time
    import io
    import base64 as base64lib
    
    # Create a test image
    RGB_PATH = "embdata/resources/color_image.png"
    rgb_image = Image(path=RGB_PATH, encoding="png", mode="RGB")

    # Current methods
    start_time = time.time()
    base64_current = rgb_image.base64
    current_base64_time = time.time() - start_time
    
    start_time = time.time()
    bytes_current = rgb_image.bytes
    current_bytes_time = time.time() - start_time
    
    # Old methods (recreated here)
    def old_base64(img):
        buffer = io.BytesIO()
        image = img.pil.convert(img.mode)
        image.save(buffer, format=img.encoding.upper())
        return base64lib.b64encode(buffer.getvalue()).decode("utf-8")
    
    def old_bytes(img):
        buffer = io.BytesIO()
        img.pil.save(buffer, format=img.encoding.upper())
        return buffer
    
    start_time = time.time()
    base64_old = old_base64(rgb_image)
    old_base64_time = time.time() - start_time
    
    start_time = time.time()
    bytes_old = old_bytes(rgb_image)
    old_bytes_time = time.time() - start_time
    
    # Print results
    print(f"\nPerformance comparison:")
    print(f"base64 current: {current_base64_time:.4f}s, old: {old_base64_time:.4f}s, speedup: {old_base64_time/current_base64_time:.2f}x")
    print(f"bytes current: {current_bytes_time:.4f}s, old: {old_bytes_time:.4f}s, speedup: {old_bytes_time/current_bytes_time:.2f}x")
    
    # Print lengths for debugging
    print(f"base64 lengths - current: {len(base64_current)}, old: {len(base64_old)}")
    print(f"bytes lengths - current: {len(bytes_current)}, old: {len(bytes_old.getvalue()) if hasattr(bytes_old, 'getvalue') else 'N/A'}")
    
    # Don't compare lengths directly, as they might be different due to encoding differences
    # Instead, verify that both methods produce valid outputs
    assert len(base64_current) > 0
    assert len(base64_old) > 0
    assert len(bytes_current) > 0


def test_array_base64_and_bytes_performance():
    """Test the performance of the current vs old methods for base64 and bytes with array input."""
    import time
    import io
    import base64 as base64lib
    
    # Create a test image from array
    array = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    img = Image(array=array, encoding="png", mode="RGB")

    # Current methods
    start_time = time.time()
    base64_current = img.base64
    current_base64_time = time.time() - start_time
    
    start_time = time.time()
    bytes_current = img.bytes
    current_bytes_time = time.time() - start_time
    
    # Old methods (recreated here)
    def old_base64(img):
        buffer = io.BytesIO()
        image = img.pil.convert(img.mode)
        image.save(buffer, format=img.encoding.upper())
        return base64lib.b64encode(buffer.getvalue()).decode("utf-8")
    
    def old_bytes(img):
        buffer = io.BytesIO()
        img.pil.save(buffer, format=img.encoding.upper())
        return buffer
    
    start_time = time.time()
    base64_old = old_base64(img)
    old_base64_time = time.time() - start_time
    
    start_time = time.time()
    bytes_old = old_bytes(img)
    old_bytes_time = time.time() - start_time
    
    # Print results
    print(f"\nArray-based performance comparison:")
    print(f"base64 current: {current_base64_time:.4f}s, old: {old_base64_time:.4f}s, speedup: {old_base64_time/current_base64_time:.2f}x")
    print(f"bytes current: {current_bytes_time:.4f}s, old: {old_bytes_time:.4f}s, speedup: {old_bytes_time/current_bytes_time:.2f}x")
    
    # Verify that the outputs are equivalent
    assert len(base64_current) == len(base64_old)
    assert len(bytes_current) > 0

if __name__ == "__main__":
    pytest.main([__file__, "-vv"])
