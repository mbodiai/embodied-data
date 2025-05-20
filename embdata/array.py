import base64
import base64 as base64lib
import hashlib
import logging
import sys
import time
import traceback
from dataclasses import dataclass
from functools import lru_cache as _lru_cache
from functools import update_wrapper
from pathlib import Path
from threading import RLock
from typing import cast

import numpy as np
from annotated_types import Len
from cv2 import COLOR_BGR2RGB, INTER_LANCZOS4, cvtColor, resize
from typing_extensions import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Callable,
    Generic,
    Literal,
    NamedTuple,
    ParamSpec,
    Protocol,
    SupportsIndex,
    TypedDict,
    TypeGuard,
    TypeVar,
    TypeVarTuple,
    Unpack,
    overload,
    override,
    runtime_checkable,
)

from embdata.utils.import_utils import smart_import

if TYPE_CHECKING:
    from collections.abc import Sequence

    from PIL.Image import Image as PILImage

h = hashlib.new("sha256")
U = TypeVar("U", bound=Any)
H = TypeVar("H", bound=Any, covariant=True)
W = TypeVar("W", bound=Any, covariant=True)
P = ParamSpec("P")
R = TypeVar("R")
S = TypeVar("S", bound=Any)
T = TypeVar("T", bound=Any)
if sys.version_info >= (3, 11):
    Ts = TypeVarTuple("Ts")
    DT_co = TypeVar("DT_co", bound= Any, covariant=True)
else:
    Ts = TypeVarTuple("Ts")
    DT_co = TypeVar("DT_co", bound=Any, covariant=True)

if False:

    from embdata.utils.image_utils import ImageEncoding, ImageMode

ImageEncoding = Any
ImageMode = Any

Float64 = np.float64 | float
Float = np.float64| np.float32 | float
Int = np.int64 | np.int32 | np.int16 | np.int8 | int
UInt = np.uint64 | np.uint32 | np.uint16 | np.uint8 | int
Bool = np.bool_ | bool
Int8 = np.int8 | int
Int16 = np.int16 | int
Int32 = np.int32 | int
Int64 = np.int64 | int
UInt8 = np.uint8 | int
UInt16 = np.uint16 | int
UInt32 = np.uint32 | int
UInt64 = np.uint64 | int
sz = Literal
N = TypeVar("N", bound=Any)
M = TypeVar("M", bound=Any)

class HasBytes(Protocol):
    def tobytes(self) -> bytes: ...

def hasbytes(obj: Any) -> TypeGuard[HasBytes]:
    return hasattr(obj, "tobytes")


class _CacheInfo(NamedTuple):
    hits: int
    misses: int
    maxsize: int
    currsize: int

if not TYPE_CHECKING:
    class _ArrayOrScalarCommon(Protocol[Unpack[Ts], DT_co]):...
    class ndarray(Protocol):...
else:
    from numpy import ndarray
@runtime_checkable
class array(ndarray,Protocol[Unpack[Ts], DT_co]): # type: ignore
    if TYPE_CHECKING:

        # @property
        # def ndim(self) -> int: ...
        @property
        @override
        def shape(self) -> Sequence[SupportsIndex] | tuple[SupportsIndex, ...]| SupportsIndex: ...
        @shape.setter
        def shape(self, value: Sequence[SupportsIndex]|tuple[SupportsIndex, ...]|SupportsIndex):...


        tobytes = np.ndarray.tobytes
        def __array__(self) -> np.ndarray:...
        tolist = np.ndarray.tolist
        __ne__ = np.ndarray.__ne__
        __neg__ = np.ndarray.__neg__
        __sub__ = np.ndarray.__sub__

        __rtruediv__ = np.ndarray.__rtruediv__
        __truediv__ = np.ndarray.__truediv__
        __floordiv__ = np.ndarray.__floordiv__
        __rfloordiv__ = np.ndarray.__rfloordiv__


        __iter__ = np.ndarray.__iter__
        from embdata.ndarray import ndarray
        squeeze = ndarray.squeeze
        mean = ndarray.mean
        transpose = ndarray.transpose
        std = ndarray.std
        sum = ndarray.sum
        prod = ndarray.prod
        max = ndarray.max
        min = ndarray.min
        __len__ = ndarray.__len__

        __contains__ = ndarray.__contains__



        __matmul__ = ndarray.__matmul__
        __rmatmul__ = ndarray.__rmatmul__
        __imatmul__ = ndarray.__imatmul__
        __radd__ = ndarray.__radd__
        __iadd__ = ndarray.__iadd__

        __rsub__ = ndarray.__rsub__
        __isub__ = ndarray.__isub__
        __imul__ = ndarray.__imul__
        __mul__ = ndarray.__mul__
        __rmul__ = ndarray.__rmul__
        __array_interface__ = np.ndarray.__array_interface__
        view = ndarray.view
        astype = ndarray.astype
        T = ndarray.T
        Inv = ndarray.Inv
        transpose = ndarray.transpose
        view = ndarray.view
    if not TYPE_CHECKING:
        try:
            PYDANTIC = True
            import pydantic
        except ImportError:
            PYDANTIC = False
        if PYDANTIC:
            @classmethod
            def __get_pydantic_core_schema__(cls,*args,**kwargs):
                from embdata.ndarray import ndarray
                return ndarray.__get_pydantic_core_schema__(*args,**kwargs)
ArrayLike = Annotated[list[DT_co], Len(cast("int",cast("object",N)))] | array[N,DT_co] | np.ndarray[Any,DT_co]
a: array[sz[1], sz[2], sz[3], np.uint8] = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

b: array[sz[1], sz[2], sz[3], Any] = np.array([1,2,3],dtype=np.uint8)

NumPyArrayNumeric = np.ndarray | array[Any, Any, Any, np.uint8] if not TYPE_CHECKING else np.ndarray


def detect_changes(old_block, new_block, method: Literal["hash", "histogram"] = "hash") -> bool:
    if not TYPE_CHECKING:
        _= smart_import("cv2")
        averageHash = smart_import("cv2.img_hash.averageHash")
        calcHist = smart_import("cv2.imgproc.calcHist")
        compareHist = smart_import("cv2.imgproc.compareHist")
        HISTCMP_CORREL = smart_import("cv2.imgproc.HISTCMP_CORREL")
    else:
        from cv2 import HISTCMP_CORREL, calcHist, compareHist
        from cv2.img_hash import averageHash
    if method == "hash":
        hash_old = averageHash(old_block)  # type: ignore
        hash_new = averageHash(new_block)  # type: ignore
        return not np.array_equal(hash_old, hash_new)
    if method == "histogram":
        hist_old = calcHist([old_block], [0], None, [256], [0, 256])  # type: ignore
        hist_new = calcHist([new_block], [0], None, [256], [0, 256])  # type: ignore
        return compareHist(hist_old, hist_new, HISTCMP_CORREL) < 0.99  # type: ignore
    msg = f"Unknown method: {method}. Use 'hash' or 'histogram'."
    raise ValueError(msg)


def warn_opencv_contrib() -> None:
    logging.warning("OpenCV contrib modules are not installed. Some functions may not work as expected.")


def error_opencv_contrib() -> None:
    if smart_import("cv2.img_hash", "lazy") == Any:
        msg = "cv2.img_hash not found. Please install the  `opencv-contrib-python`"
        raise ImportError(msg)


@dataclass
class Image(Generic[H, W]):
    array: array[H, W, Int]
    pil: "PILImage" # type: ignore  # noqa
    path: Path
    url: str
    size: tuple[int, int]
    mode: Literal["RGB", "RGBA", "L", "P", "CMYK", "YCbCr", "I", "F"]
    encoding: str
    def tobytes(self) -> bytes: ...

class HashMethods(TypedDict):
    PHash: Callable[[NumPyArrayNumeric], NumPyArrayNumeric]
    AverageHash: Callable[[NumPyArrayNumeric], NumPyArrayNumeric]
    BlockMeanHash: Callable[[NumPyArrayNumeric], NumPyArrayNumeric]
    ColorMomentHash: Callable[[NumPyArrayNumeric], NumPyArrayNumeric]
    MarrHildrethHash: Callable[[NumPyArrayNumeric], NumPyArrayNumeric]
    RadialVarianceHash: Callable[[NumPyArrayNumeric], NumPyArrayNumeric]
    SHA256: Callable[[NumPyArrayNumeric], NumPyArrayNumeric]
    NoHash: Callable[[NumPyArrayNumeric], NumPyArrayNumeric]


def hash_methods() -> HashMethods:
    """Returns a dictionary of hash methods."""
    error_opencv_contrib()
    if TYPE_CHECKING:
        from cv2.img_hash import (  # type: ignore  # noqa
            averageHash,
            blockMeanHash,
            colorMomentHash,
            marrHildrethHash,
            pHash,
            radialVarianceHash,
        )
    else:
        averageHash = smart_import("cv2.img_hash.averageHash")
        blockMeanHash = smart_import("cv2.img_hash.blockMeanHash")
        colorMomentHash = smart_import("cv2.img_hash.colorMomentHash")
        marrHildrethHash = smart_import("cv2.img_hash.marrHildrethHash")
        pHash = smart_import("cv2.img_hash.pHash")
        radialVarianceHash = smart_import("cv2.img_hash.radialVarianceHash")

    def sha256(array: NumPyArrayNumeric) -> NumPyArrayNumeric:
        h = hashlib.new("sha256")
        h.update(array.tobytes())
        return np.frombuffer(h.digest(), dtype=np.uint8)

    return {
        "PHash": pHash,
        "AverageHash": averageHash,
        "BlockMeanHash": blockMeanHash,
        "ColorMomentHash": colorMomentHash,
        "MarrHildrethHash": marrHildrethHash,
        "RadialVarianceHash": radialVarianceHash,
        "SHA256": sha256,
        "NoHash": lambda x: x,
    }


def cached_semantic_hash(
    image: array[H, W, sz[3], Int] | array[H, W, sz[1], Int] | array[H, W, Int],
    method: Literal[
        "PHash",
        "AverageHash",
        "BlockMeanHash",
        "ColorMomentHash",
        "MarrHildrethHash",
        "RadialVarianceHash",
        "SHA256",
        "NoHash",
    ] = "PHash",
) -> bytes:
    """Compute and cache the semantic hash of an image."""
    return hash_methods()[method](np.asarray(image)).tobytes()


class _HashedSeq(list):
    """Guarantee that hash() will be called no more than once per element.

    This is important because the lru_cache() will hash the key multiple times on a cache miss.
    """

    __slots__ = "hashvalue"

    def __init__(self, tup, hash=hash):  # noqa
        self[:] = tup
        self.hashvalue = hash(tup)

    def __hash__(self):
        return self.hashvalue


@overload
def lru_cache(func: Callable[P, R]) -> Callable[P, R]: ...


@overload
def lru_cache(maxsize=128, typed=False) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def lru_cache(*args, **kwargs) -> Any:
    """Least-recently-used cache decorator."""
    if len(args) == 1 and callable(args[0]):
        return _lru_cache()(args[0])
    arglist = list(args)
    maxsize = kwargs.get("maxsize", arglist.pop(0) if arglist else 128)
    typed = kwargs.get("typed", arglist.pop(0) if arglist else False)
    if isinstance(maxsize, int):
        maxsize = max(maxsize, 0)
    elif callable(maxsize) and isinstance(typed, bool):
        user_function, maxsize = maxsize, 128
        wrapper = _lru_cache_wrapper(user_function, maxsize, typed, _CacheInfo)
        wrapper.cache_parameters = lambda: {"maxsize": maxsize, "typed": typed}
        return update_wrapper(wrapper, user_function)
    elif maxsize is not None:
        msg = "Expected first argument to be an integer, a callable, or None."
        raise TypeError(msg)

    def decorating_function(user_function: Callable[P, R]) -> Callable[P, R]:
        wrapper = _lru_cache_wrapper(user_function, maxsize, typed, _CacheInfo)
        wrapper.cache_parameters = lambda: {"maxsize": maxsize, "typed": typed}
        return update_wrapper(wrapper, user_function)  # type: ignore

    return decorating_function


# Helper for cache keys
def _make_key(
    args,
    kwds,
    typed,
    kwd_mark=(object(),),  # noqa
    fasttypes=None,
    tuple=tuple,  # noqa
    type=type,  # noqa
    length=len,
):
    """Modified to handle numpy arrays by converting them to bytes."""
    if fasttypes is None:
        fasttypes = {int, str}

    def _convert_numpy(arg):
        if isinstance(arg, np.ndarray|array):
            return arg.tobytes()
        return arg

    args = tuple(_convert_numpy(arg) for arg in args)
    key = args
    if kwds:
        key += kwd_mark
        for item in kwds.items():
            key += (_convert_numpy(item[0]), _convert_numpy(item[1]))
    if typed:
        key += tuple(type(v) for v in args)
        if kwds:
            key += tuple(type(v) for v in kwds.values())
    elif length(key) == 1 and type(key[0]) in fasttypes:
        return key[0]
    return _HashedSeq(key)


def _lru_cache_wrapper(user_function, maxsize, typed, _CacheInfo):  # noqa
    # Constants shared by all lru cache instances:
    sentinel = object()  # unique object used to signal cache misses
    make_key = _make_key  # build a key from the function arguments
    PREV, NEXT, KEY, RESULT = 0, 1, 2, 3  # names for the link fields

    cache = {}
    hits = misses = 0
    full = False
    cache_get = cache.get  # bound method to lookup a key or return None
    cache_len = cache.__len__  # get cache size without calling len()
    lock = RLock()  # because linkedlist updates aren't threadsafe
    root = []  # root of the circular doubly linked list
    root[:] = [root, root, None, None]  # initialize by pointing to self

    if maxsize == 0:

        def wrapper(*args, **kwds):
            # No caching -- just a statistics update
            nonlocal misses
            misses += 1
            return user_function(*args, **kwds)

    elif maxsize is None:

        def wrapper(*args, **kwds):
            # Simple caching without ordering or size limit
            nonlocal hits, misses
            key = make_key(args, kwds, typed)
            result = cache_get(key, sentinel)
            if result is not sentinel:
                hits += 1
                return result
            misses += 1
            result = user_function(*args, **kwds)
            cache[key] = result
            return result

    else:

        def wrapper(*args, **kwds):
            # Size limited caching that tracks accesses by recency
            nonlocal root, hits, misses, full
            key = make_key(args, kwds, typed)
            with lock:
                link = cache_get(key)
                if link is not None:
                    # Move the link to the front of the circular queue
                    link_prev, link_next, _key, result = link
                    link_prev[NEXT] = link_next
                    link_next[PREV] = link_prev
                    last = root[PREV]
                    last[NEXT] = root[PREV] = link
                    link[PREV] = last
                    link[NEXT] = root
                    hits += 1
                    return result
                misses += 1
            result = user_function(*args, **kwds)
            with lock:
                if key in cache:
                    # Getting here means that this same key was added to the
                    # cache while the lock was released.  Since the link
                    # update is already done, we need only return the
                    # computed result and update the count of misses.
                    pass
                elif full:
                    # Use the old root to store the new key and result.
                    oldroot = root
                    oldroot[KEY] = key
                    oldroot[RESULT] = result
                    # Empty the oldest link and make it the new root.
                    # Keep a reference to the old key and old result to
                    # prevent their ref counts from going to zero during the
                    # update. That will prevent potentially arbitrary object
                    # clean-up code (i.e. __del__) from running while we're
                    # still adjusting the links.
                    root = oldroot[NEXT]
                    oldkey = root[KEY]
                    root[RESULT]
                    root[KEY] = root[RESULT] = None
                    # Now update the cache dictionary.
                    del cache[oldkey]
                    # Save the potentially reentrant cache[key] assignment
                    # for last, after the root and links have been put in
                    # a consistent state.
                    cache[key] = oldroot
                else:
                    # Put result in a new link at the front of the queue.
                    last = root[PREV]
                    link = [last, root, key, result]
                    last[NEXT] = root[PREV] = cache[key] = link
                    # Use the cache_len bound method instead of the len() function
                    # which could potentially be wrapped in an lru_cache itself.
                    full = cache_len() >= maxsize
            return result

    def cache_info():
        """Report cache statistics."""
        with lock:
            return _CacheInfo(hits, misses, maxsize, cache_len())

    def cache_clear():
        """Clear the cache and cache statistics."""
        nonlocal hits, misses, full
        with lock:
            cache.clear()
            root[:] = [root, root, None, None]
            hits = misses = 0
            full = False

    wrapper.cache_info = cache_info
    wrapper.cache_clear = cache_clear
    return wrapper


@lru_cache(maxsize=128)
def to_url(
    image: "array[H, W, np.integer] | array[H,W,sz[3],np.integer] | array[H,W,sz[1],np.integer] | Image[H,W]",
    encoding: Literal["png", "jpg", "jpeg", "bmp", "tiff"] = "png",
) -> str:
    """Convert image array to URL string."""
    try:
        img_base64 = base64.b64encode(image.tobytes()).decode()
        return f"data:image/{encoding};base64,{img_base64}"
    except Exception as e:
        logging.exception(f"Failed to convert image to URL: {e}")
        raise


@lru_cache(maxsize=128)
def load_url(
    url: str,
    size: tuple[H, W] | None = None,
    encoding: ImageEncoding | None = None,
    mode: ImageMode | None  = None,
    **_kwargs,
) -> array[H,W,Int]:
    """Downloads an image from a URL or decodes it from a base64 data URI.

    This method can handle both regular image URLs and base64 data URIs.
    For regular URLs, it downloads the image data. For base64 data URIs,
    it decodes the data directly. It's useful for fetching images from
    the web or working with inline image data.

    Args:
    ----
        url (str): The URL of the image to download, or a base64 data URI.
        size (Optional[Tuple[int, int]]): The desired size of the image as a (width, height) tuple. Defaults to None.
        encoding (Optional[str]): The encoding format of the image. Defaults to None.
        mode (Optional[str]): The mode to use for the image. Defaults to None.
        **kwargs: Additional keyword arguments.

    Returns:
    -------
        PIL.Image.Image | None: The downloaded and decoded image as a PIL Image object,
                                or None if the download fails or is cancelled.

    Example:
    -------
        >>> image = Image.load_url("https://example.com/image.jpg")
        >>> if image:
        ...     print(f"Image size: {image.size}")
        ... else:
        ...     print("Failed to load image")

        >>> data_uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAAABQABDQottAAAAABJRU5ErkJggg=="
        >>> image = Image.load_url(data_uri)
        >>> if image:
        ...     print(f"Image size: {image.size}")
        ... else:
        ...     print("Failed to load image")

    """
    from urllib.request import Request, urlopen

    if isinstance(url, Image):
        url = url.url
    user_agent = "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7"
    headers = {"User-Agent": user_agent}
    if url.startswith("data:image/"):
        return from_base64(url, encoding=encoding, size=size, mode=mode)
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
        return from_bytes(data, size, mode, encoding=encoding)


# @lru_cache(maxsize=128)
def from_base64(
    arg: str,
    encoding: ImageEncoding | None = None,
    size: tuple[H, W] | None = None,
    mode: ImageMode | None = "RGB",
) -> array[H, W, Int]:
    """Decodes a base64 string to an image array efficiently with a single processing path.

    Args:
    ----
        arg: The base64-encoded string or data URI
        encoding: The encoding format (determines bit depth handling)
        size: Optional tuple of (width, height) to resize the image to
        mode: Color mode for output array ('RGB', 'RGBA', 'L', etc.)

    Returns:
    -------
        NumPy array containing the decoded image in requested format

    """
    # Extract the base64 part and encoding if this is a data URI
    if ";" in arg and "base64," in arg:
        if not encoding and "data:image" in arg:
            mime_type = cast("ImageEncoding", arg.split("data:image/", 1)[1].split(";", 1)[0])
            encoding = mime_type
        base64_str = arg.split(";base64,", 1)[1] if ";" in arg else arg
    else:
        base64_str = arg

    # Add padding if needed (before decoding)
    padding = len(base64_str) % 4
    if padding:
        base64_str += "=" * (4 - padding)

    # Decode base64 to bytes
    img_bytes = base64.b64decode(base64_str, validate=True)

    return from_bytes(img_bytes, size, mode, encoding)


def from_bytes(
    arg: bytes,
    size: tuple[H, W] | None = None,
    mode: ImageMode | None = "RGB",
    encoding: ImageEncoding | None = None,
) -> array[H,W,Int]:
    if TYPE_CHECKING:
        import imghdr
        import io

        import cv2
        import numpy as np
        from PIL import Image
    else:
        cv2 = smart_import("cv2")
        np = smart_import("numpy")
        Image = smart_import("PIL.Image")
        io = smart_import("io")
        imghdr = smart_import("imghdr")

    # Validate encoding vs header
    imghdr.what(None, h=arg)

    if mode == "I16":
        arr = np.frombuffer(arg, dtype=np.uint16)
    elif mode == "F":
        arr = np.frombuffer(arg, dtype=np.float32)
    elif mode == "I":
        arr = np.frombuffer(arg, dtype=np.int32)
    else:
        arr = np.frombuffer(arg, dtype=np.uint8)
    read_flag = cv2.IMREAD_COLOR if mode in ("RGB", "YCbCr", "P", "CMYK") else cv2.IMREAD_UNCHANGED
    img = cv2.imdecode(arr, read_flag)
    if img is None:
        msg = "Failed to decode image data (OpenCV returned None)"
        raise ValueError(msg)

    # Color mode conversions
    if mode == "RGB" and img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif mode == "RGBA" and img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    elif mode == "YCbCr":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    elif mode == "CMYK":
        img = np.array(Image.open(io.BytesIO(arg)).convert("CMYK"))
    elif mode == "P":
        img = np.array(Image.open(io.BytesIO(arg)).convert("P"))

    # Resize if requested
    if size and tuple(img.shape[:2]) != size:
        img = cv2.resize(img, size[::-1], interpolation=cv2.INTER_LANCZOS4)

    return np.array(to_pil(img, mode=mode))


# @lru_cache(maxsize=128)
def loads(
    bytes_data: bytes | str | Path,
    size: tuple[H, W] | None = None,
    mode: ImageMode | None = "RGB",
    encoding: ImageEncoding | None = "jpeg",
) -> array[H, W, Int]:
    """Creates an image array from bytes or base64 string.

    OpenCV loads images in BGR format, but this function converts them to RGB
    to maintain consistency with the rest of the library.

    Args:
    ----
        bytes_data: The bytes or base64 string to convert to a numpy array.
        size: Optional tuple of (width, height) to resize the image to.
        mode: Color mode for output array ('RGB', 'RGBA', 'L', etc.)
        encoding: The encoding to use for the image.

    Returns:
    -------
        np.ndarray: RGB image array with shape (height, width, channels)

    """
    if not TYPE_CHECKING:
        smart_import("cv2")
        Path = smart_import("pathlib.Path")

    if isinstance(bytes_data, bytes):
        return from_bytes(bytes_data, size, mode, encoding)
    if isinstance(bytes_data, Path) or (isinstance(bytes_data, str) and Path(bytes_data[:100]).exists()):
        return from_path(bytes_data, size)
    if isinstance(bytes_data, str) and bytes_data.startswith("http"):
        return load_url(url=bytes_data, size=size, encoding=encoding, mode=mode)
    if isinstance(bytes_data, str):
        return from_base64(arg=bytes_data, size=size, encoding=encoding, mode=mode)

    msg = f"Expected bytes, got {type(bytes_data)}"
    raise ValueError(msg)


@lru_cache(maxsize=128)
def from_pil(
    pil_image: bytes,
    size: tuple[H, W] | None = None,
    mode: Literal["RGB", "RGBA", "L", "P", "CMYK", "YCbCr", "I", "F"] | None = "RGB",
) -> array[H, W, Int]:
    """Converts a PIL image to a numpy array.

    Args:
    ----
        pil_image (Any): The PIL image to convert.
        size (Tuple[H, W] | None): The size of the image to resize to.
        mode (Literal["RGB", "RGBA", "L", "P", "CMYK", "YCbCr", "I", "F"] | None): The mode of the image to convert to.

    Returns:
    -------
        np.ndarray: The image as a numpy array.

    """
    if not TYPE_CHECKING:
        INTER_LANCZOS4 = smart_import("cv2.INTER_LANCZOS4")
        resize = smart_import("cv2.resize")
        PILModule = smart_import("PIL.Image")
    else:
        from cv2 import INTER_LANCZOS4, resize  # type: ignore
        from PIL import Image as PILModule  # type: ignore
    arr = np.array(PILModule.open(pil_image).convert(mode))
    if size is not None and tuple(arr.shape[:2]) != size:
        arr = resize(arr, size[::-1], interpolation=INTER_LANCZOS4)
    return cast("array[H, W, Int]", arr)


@lru_cache(maxsize=128)
def from_path(path: str | bytes | Path, size: tuple[H, W] | None = None) -> array[H, W, Int]:
    """Loads an image from a file path using OpenCV.

    Args:
    ----
        path: File path to load the image from.
        size: Optional tuple of (width, height) to resize the image to.

    Returns:
    -------
        np.ndarray: RGB image array with shape (height, width, channels)

    """
    if TYPE_CHECKING:
        from cv2 import IMREAD_COLOR, imread  # type: ignore
    else:
        imread = smart_import("cv2.imread")
        IMREAD_COLOR = smart_import("cv2.IMREAD_COLOR")
        smart_import("cv2")

    # Convert path to absolute Path object
    try:
        path_obj = Path(str(path)).absolute()
        if not path_obj.exists():
            msg = f"Image file not found: {path_obj}"
            raise FileNotFoundError(msg)
        if not path_obj.is_file():
            msg = f"Path is not a file: {path_obj}"
            raise ValueError(msg)

        # Read the image from the file path using OpenCV (returns BGR)
        arr = imread(str(path_obj), IMREAD_COLOR)
        if arr is None:
            msg = f"OpenCV failed to decode image at: {path_obj}"
            raise ValueError(msg)

        # API BOUNDARY: Convert from BGR (OpenCV) to RGB (internal format)
        rgb_arr = cast("array[H, W, Int]", cvtColor(arr, COLOR_BGR2RGB))

        # Resize if needed
        if size is not None:
            # Size is (width, height) which matches cv2.resize expectations
            rgb_arr = cast("array[H, W, Int]", resize(rgb_arr.view(np.ndarray), size, interpolation=INTER_LANCZOS4))

        return rgb_arr
    except Exception as e:
        logging.exception(f"Error loading image from {path}: {e!s}\n{traceback.format_exc()}")
        raise


# @lru_cache(maxsize=128)
def to_bytes(
    image: "array[H,W,Int]",
) -> bytes:
    """Converts an image array or Image-like object to encoded bytes."""
    return image.tobytes()

def to_base64(
    image: "array[H, W, np.integer] | array[H,W,sz[3],np.integer] | array[H,W,sz[1],np.integer] | Image[H,W] | np.ndarray", # noqa
    encoding: ImageEncoding = "PNG",
) -> str:
    """Convert an image array to a base64 string.

    Returns a complete data URI string (e.g., "data:image/png;base64,...")

    Args:
    ----
        image: The image to convert (RGB format expected).
        path: Optional path to save the bytes to.
        encoding: The encoding format to use.

    Returns:
    -------
        str: The data URI containing the base64-encoded image.

    """
    # Encode as base64 and format as data URI
    base64_str = base64lib.b64encode(image.tobytes()).decode("utf-8")
    return f"data:image/{encoding};base64,{base64_str}"


def to_pil(
    image: "array[H, W, Int] | array[H, W, sz[3], Int] | array[H, W, sz[1], Int] | Image[H, W]",
    mode: ImageMode | None = "RGB",
) -> "PILImage":  # type: ignore  # noqa
    """Convert a numpy array to a PIL image.

    Args:
    ----
        image (np.array): The image array to convert.
        mode (str): The mode to convert the image to.
        encoding (str): The encoding to use for the image.

    Returns:
    -------
        Any: The PIL image.

    """
    if not TYPE_CHECKING:
        PILModule = smart_import("PIL.Image")
    else:
        from PIL import Image as PILModule  # type: ignore
    return PILModule.fromarray(np.asarray(image),mode=mode)


def error(
    original: array[H, W, Int],
    modified: array[H, W, Int],
    metric: Literal["mse", "cosine", "hamming", "mae"] = "mse",
) -> float:
    if metric == "cosine":
        similarity = np.dot(original, modified) / (np.linalg.norm(original) * np.linalg.norm(modified))
        return 1 - similarity
    if metric == "hamming":
        return float(np.mean(original != modified))
    if metric == "mae":
        return float(np.mean(np.abs(np.asarray(original) - np.asarray(modified)).astype(float)))
    if metric == "mse":
        return np.mean((np.asarray(original) - np.asarray(modified)) ** 2).astype(float)
    msg = f"Unknown metric: {metric}"
    raise ValueError(msg)


def benchmark_hash_method(
    hash_func: Callable[[array[H, W, Int]], array[H, W, Int]],
    input_data: array[H, W, Int],
    modified_data: array[H, W, Int] | None = None,
    iterations: int = 10,
) -> dict[str, Any]:
    start = time.time()
    hash_value = None
    try:
        for _ in range(iterations):
            hash_value = hash_func(input_data)

        end = time.time()

        reconstruction_error = None
        if modified_data is not None:
            original_hash = hash_func(input_data)
            modified_hash = hash_func(modified_data)

            original_flat = original_hash.ravel()
            modified_flat = modified_hash.ravel()

            if hash_func.__name__ == "NoHash":
                reconstruction_error = error(input_data, modified_data, metric="mae")
            else:
                reconstruction_error = error(original_flat, modified_flat)

        return {
            "time": end - start,
            "time_per_iteration": (end - start) / iterations,
            "hash_value": hash_value,
            "reconstruction_error": reconstruction_error,
        }
    except Exception as e:
        logging.exception(f"Error in benchmark for {hash_func.__name__}: {e!s}")
        return {
            "time": 0,
            "time_per_iteration": 0,
            "hash_value": None,
            "reconstruction_error": None,
            "error": str(e),
        }


def run_benchmark(
    image_sizes: list[tuple[int, int, int]], iterations: int = 10,
) -> dict[tuple[int, int, int], dict[str, Any]]:
    import cv2

    results = {size: {} for size in image_sizes}
    for size in image_sizes:
        test_image = np.random.randint(0, 255, (size[0], size[1], size[2]), dtype=np.uint8)
        noise = np.random.normal(0, 5, test_image.shape).astype(np.uint8)
        modified_image = cv2.add(test_image, noise)

        for name, func in hash_methods().items():
            results[size][name] = benchmark_hash_method(func, test_image, modified_image, iterations)
    return results


def plot_results(results) -> None:
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

    for size, data in results.items():
        times = [v["time_per_iteration"] for v in data.values()]
        errors = [v["reconstruction_error"] for v in data.values()]
        labels = list(data.keys())

        ax1.plot(labels, times, label=f"Time: {size[0]}x{size[1]}")
        ax2.plot(labels, errors, label=f"Reconstruction Error: {size[0]}x{size[1]}")

    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, rotation=45)
    ax1.set_xlabel("Hash Method")
    ax1.set_ylabel("Time Per Iteration (s)")
    ax1.set_title("Hash Method Latency by Image Size")
    ax1.legend()

    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=45)
    ax2.set_xlabel("Hash Method")
    ax2.set_ylabel("Reconstruction Error")
    ax2.set_title("Hash Method Reconstruction Error by Image Size")
    ax2.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    image_sizes = [(256, 256, 3), (512, 512, 3), (1024, 1024, 3)]
    iterations = 100

    results = run_benchmark(image_sizes, iterations)
    plot_results(results)
    from rich.pretty import pprint

    pprint(results)
