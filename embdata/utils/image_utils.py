import base64
import base64 as base64lib
import io
import logging
from functools import singledispatch
from pathlib import Path
from typing import Any, Dict, SupportsBytes, Tuple, TypeAlias

import numpy as np
from PIL import Image as PILModule
from PIL import ImageOps
from PIL.Image import Image as PILImage
from pydantic import (
    AnyUrl,
    Base64Str,
    FilePath,
    InstanceOf,
)
from typing_extensions import Literal

from embdata.ndarray import ndarray

ImageLikeArray = ndarray[3, Any, Any, np.uint8] | ndarray[Any, Any, 3, np.uint8]
SupportsImage: TypeAlias = (
    ndarray[3, ..., np.uint8]
    | InstanceOf[PILImage]
    | InstanceOf[Base64Str]
    | InstanceOf[AnyUrl]
    | InstanceOf[FilePath]
    | InstanceOf[bytes]
    | InstanceOf[io.BytesIO]
)

SOURCE_TYPES = ["array", "pil", "base64", "url", "path", "bytes"]

DEFAULT_MODE = "RGB"
def count_image_tokens(width, height, base64_image_size, image_format="JPEG", quality=85):
    """Count the number of tokens based on image dimensions, Base64 size, and image format/quality."""
    h = ceil(height / 512)
    w = ceil(width / 512)
    n = w * h
    total_token_count = 85 + 170 * n

    # Add token count for the base64 encoding size
    base64_token_count = ceil(base64_image_size / 4)  # Approximation: 4 bytes = 1 token in text encoding
    total_token_count += base64_token_count

    # Adjust for image format
    if image_format == "JPEG":
        # JPEG compression; lower quality means fewer tokens
        quality_factor = 1 - (quality / 100)  # As quality decreases, we assume fewer tokens
        total_token_count = int(total_token_count * (1 - 0.3 * quality_factor))  # 30% impact for extreme qualities
    elif image_format == "PNG":
        # PNG has larger sizes; assume a fixed multiplier for more tokens
        total_token_count = int(total_token_count * 1.5)

    return total_token_count

def load_url(
    url: str,
    size: Tuple[int, int] | None = None,
    encoding: str | None = None,
    mode: Literal["RGB", "RGBA", "L", "P", "CMYK", "YCbCr", "I", "F"] | None = None,
    **kwargs,
) -> PILImage | None:
    """Downloads an image from a URL or decodes it from a base64 data URI.

    This method can handle both regular image URLs and base64 data URIs.
    For regular URLs, it downloads the image data. For base64 data URIs,
    it decodes the data directly. It's useful for fetching images from
    the web or working with inline image data.

    Args:
        url (str): The URL of the image to download, or a base64 data URI.
        size (Optional[Tuple[int, int]]): The desired size of the image as a (width, height) tuple. Defaults to None.
        encoding (Optional[str]): The encoding format of the image. Defaults to None.
        mode (Optional[str]): The mode to use for the image. Defaults to None.
        **kwargs: Additional keyword arguments.

    Returns:
        PIL.Image.Image | None: The downloaded and decoded image as a PIL Image object,
                                or None if the download fails or is cancelled.

    Example:
        >>> image = Image.load_url("https://example.com/image.jpg")
        >>> if image:
        ...     print(f"Image size: {image.size}")
        ... else:
        ...     print("Failed to load image")

        >>> data_uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAACklEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg=="
        >>> image = Image.load_url(data_uri)
        >>> if image:
        ...     print(f"Image size: {image.size}")
        ... else:
        ...     print("Failed to load image")
    """
    if url.startswith("data:image"):
        try:
            # Extract the base64 part of the URL
            base64_str = url.split(";base64,", 1)[1]

            # Decode base64 string into raw bytes
            raw_image_data = base64.b64decode(base64_str)

            # Reconstruct the image from raw bytes
            image = PILModule.frombytes(mode=mode, size=size, data=raw_image_data)

            return image.convert(mode) if mode else image

        except Exception as e:
            error_message = f"Failed to decode base64 image from URL: {url}"
            raise ValueError(error_message) from e

    # Handle other URL cases...
    from urllib.request import Request, urlopen

    user_agent = "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7"
    headers = {"User-Agent": user_agent}

    if not url.startswith(("http:", "https:")):
        msg = "URL must start with 'http' or 'https'."
        raise ValueError(msg)

    if not url.endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")) and not url.split("?")[0].endswith(
        (".jpg", ".jpeg", ".png", ".bmp", ".gif"),
    ):
        if url.find("huggingface.co") != -1:
            logging.warning("URL not ending with a valid image extension.")

        else:
            msg = f"URL must end with a valid image extension: {url[:20]}...{url[-20:]}"
            raise ValueError(msg)

    with urlopen(Request(url, None, headers)) as response:  # noqa
        data = response.read()
        image = PILModule.open(io.BytesIO(data))
        return image.convert(mode) if mode else image


def init_base64(
    arg: Base64Str,
    encoding: str | None = None,
    size: Tuple[int, int] | None = None,
    mode: Literal["RGB", "RGBA", "L", "P", "CMYK", "YCbCr", "I", "F"] | None = None,
    **kwargs,
) -> dict:
    """Decodes a base64 string to create an Image instance.

    Args:
        arg (Base64Str): The base64-encoded string of the image.
        encoding (Optional[str]): The format of the image encoding (e.g., "png", "jpeg").
        size (Optional[Tuple[int, int]]): The desired size of the image.
        mode (Optional[str]): The desired mode of the image (e.g., "RGB").
        **kwargs: Additional keyword arguments.

    Returns:
        dict: A dictionary containing image data and metadata.
    """
    # Decode base64 string into bytes
    image_data = base64lib.b64decode(arg)

    # Open the image from bytes and optionally convert mode
    pil_image = PILModule.open(io.BytesIO(image_data))
    if mode and pil_image.mode != mode:
        pil_image = pil_image.convert(mode)

    # Ensure encoding is set
    if encoding is None:
        encoding = pil_image.format.lower()

    return pil_to_data(pil_image, encoding, size, mode)



def pil_to_data(image: PILImage, encoding: str | None = None, size=None, mode: str | None = None, url=None) -> dict:
    """Creates an Image instance from a PIL image.

    Args:
        image (PIL.Image.Image): The source PIL image from which to create the Image instance.
        encoding (str): The format used for encoding the image when converting to base64.
        size (Optional[Tuple[int, int]]): The size of the image as a (width, height) tuple.
        mode (Optional[str]): The mode to use for the image. Defaults to "RGB".
        url (Optional[str]): The URL of the image, if available.

    Returns:
        Image: An instance of the Image class with populated fields.
    """
    if encoding.lower() == "jpg":
        encoding = "jpeg"
    image = image.convert(mode) if mode is not None else image
    base64_encoded = base64lib.b64encode(image.tobytes()).decode("utf-8")
    if encoding is None:
        encoding = image.format.lower()
    data_url = f"data:image/{encoding};base64,{base64_encoded}"
    if size is not None:
        image = ImageOps.fit(image, size, PILModule.Resampling.LANCZOS)
    else:
        size = image.size

    return {
        "array": np.array(image),
        "base64": base64_encoded,
        "pil": image,
        "size": size,
        "url": data_url if url is None else url,
        "encoding": encoding.lower(),
        "mode": mode,
    }

@singledispatch
def dispatch_arg(arg: Any, **kwargs) -> dict:
    msg = f"Unsupported argument type: {type(arg)}"
    raise ValueError(msg)


@dispatch_arg.register
def init_dict(arg: dict, **kwargs) -> None:
    kwargs.update(arg)
    if "pil" not in kwargs:
        for k in SOURCE_TYPES:
            if k in kwargs and kwargs[k] is not None:
                arg = kwargs.pop(k)
                kwargs.update(dispatch_arg(arg, **kwargs))
                return kwargs
        arg = kwargs.pop(next(iter(kwargs.keys())))
        kwargs.update(dispatch_arg(arg, **kwargs))
    return kwargs


@dispatch_arg.register(io.BytesIO)
def init_bytesio(
    arg: SupportsImage | None = None,  # type: ignore
    size: Tuple[int, int] | None = None,
    encoding = None,
    mode: Literal["RGB", "RGBA", "L", "P", "CMYK", "YCbCr", "I", "F"] | None = DEFAULT_MODE,
    **kwargs,
) -> None:
    kwargs.update(pil_to_data(PILModule.open(arg, formats=[encoding.upper()]), encoding, size, mode))
    return kwargs


@dispatch_arg.register(SupportsBytes)
def init_bytes(
    arg: SupportsImage | None = None,  # type: ignore
    size: Tuple[int, int] | None = None,
    encoding=None,
    mode: Literal["RGB", "RGBA", "L", "P", "CMYK", "YCbCr", "I", "F"] | None = None,
    **kwargs,
) -> None:
    kwargs.update(pil_to_data(PILModule.open(io.BytesIO(arg)), encoding.upper(), size, mode))
    return kwargs


@dispatch_arg.register(np.ndarray)
def from_numpy(
    arg: ndarray[3, Any, Any, np.uint8] | ndarray[Any, Any, 3, np.uint8] | None = None,
    size: Tuple[int, int] | None = None,
    encoding=None,
    mode: Literal["RGB", "RGBA", "L", "P", "CMYK", "YCbCr", "I", "F"] | None = None,
    **kwargs,
) -> dict:
    """Creates an Image instance from a NumPy array."""
    if not isinstance(arg, np.ndarray) or arg.ndim not in (2, 3):
        msg = "Input must be a 2D or 3D NumPy array representing an image."
        raise ValueError(msg)
    kwargs.update(pil_to_data(PILModule.fromarray(arg), encoding, size, mode))
    return kwargs

@dispatch_arg.register(str)
def init_str(
    arg: str,
    encoding: str | None = None,
    action: Literal["download", "set"] = "set",
    size: Tuple[int, int] | None = None,
    mode: Literal["RGB", "RGBA", "L", "P", "CMYK", "YCbCr", "I", "F"] | None = None,
    **kwargs,
) -> dict:
    """Decodes a base64 string to create an Image instance.

    This method takes a base64 encoded string representation of an image,
    decodes it, and creates an Image instance from it. It's useful when
    you have image data in base64 format and want to work with it as an Image object.

    Args:
        arg (str): The base64 string to decode.
        encoding (str): The format used for encoding the image when converting to base64.
        action (str): Either "download" or "set" if the URL is a download link.
        size (Optional[Tuple[int, int]]): The size of the image as a (width, height) tuple.
        mode (Optional[str]): The mode to use for the image. Defaults to "
        **kwargs: Additional keyword arguments.

    Returns:
        Image: An instance of the Image class with populated fields.

    Example:
        >>> base64_str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAACklEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg=="
        >>> image = Image.from_base64(base64_str, encoding="png", size=(1, 1))
        >>> print(image.size)
        (1, 1)

        # Example with complex nested structure
        >>> nested_data = {
        ...     "image": Image.from_base64(base64_str, encoding="png"),
        ...     "metadata": {"text": "A small red square", "tags": ["red", "square", "small"]},
        ... }
        >>> print(nested_data["image"].size)
        (1, 1)
        >>> print(nested_data["metadata"]["text"])
        A small red square
    """
    if arg and Path(arg[:120]).exists():
        return dispatch_arg(Path(arg), encoding, size, mode, **kwargs)
    if arg.startswith(("data:image", "http")):
        image = load_url(arg, size, encoding, mode, action=action, **kwargs)
        kwargs.update(pil_to_data(image, encoding, size, mode, url=arg))
        kwargs["url"] = arg
        return kwargs
    try:
        return init_base64(arg, encoding, size, mode, **kwargs)
    except Exception as e:
        msg = f"Path not found, url not found, or invalid base64 string: {arg[:50]}..."
        raise ValueError(msg) from e


@dispatch_arg.register(PILImage)
def init_pil(
    arg: PILImage,
    encoding: str | None = None,
    size: Tuple[int, int] | None = None,
    mode: Literal["RGB", "RGBA", "L", "P", "CMYK", "YCbCr", "I", "F"] | None = None,
    **kwargs,
) -> None:
    """Creates an Image instance from a PIL image.

    Args:
        arg (PIL.Image.Image): The source PIL image from which to create the Image instance.
        encoding (str): The format used for encoding the image when converting to base64.
        size (Optional[Tuple[int, int]]): The size of the image as a (width, height) tuple.
        mode (Optional[str]): The mode to use for the image. Defaults to "RGB".
        **kwargs: Additional keyword arguments.

    Returns:
        Image: An instance of the Image class with populated fields.
    """
    kwargs.update(pil_to_data(arg, encoding, size, mode))
    return kwargs

@dispatch_arg.register(Path)
def open(
    arg: Path,
    encoding: str | None = None,
    size: Tuple[int] | None = None,
    mode: Literal["RGB", "RGBA", "L", "P", "CMYK", "YCbCr", "I", "F"] | None = None,
    **kwargs,
) -> Dict[str, Any]:
    """Opens an image from a file path and creates an Image instance.

    This method reads an image file from the specified path,
    and creates an Image instance from it. It's a convenient way to load images from
    your local file system.

    Args:
        arg (str): The path to the image file.
        encoding (str): The format used for encoding the image when converting to base64.
                        Defaults to "jpeg".
        size (Optional[Tuple[int, int]]): The size of the image as a (width, height) tuple.
                                            If provided, the image will be resized.
        mode (Optional[str]): The mode to use for the image.
        **kwargs: Additional keyword arguments.

    Returns:
        Image: An instance of the Image class with populated fields.

    Example:
        >>> image = Image.open("/path/to/image.jpg", encoding="jpeg", size=(224, 224))
        >>> print(image.size)
        (224, 224)
    """
    image = PILModule.open(arg)
    image = image.convert(mode)
    kwargs["path"] = str(arg)
    kwargs.update(pil_to_data(image, encoding, size, mode))
    return kwargs
