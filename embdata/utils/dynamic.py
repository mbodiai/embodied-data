from typing import Any, Callable, Concatenate, Generic, Type

from typing_extensions import ParamSpec, TypeVar, TypeVarTuple, overload

P = ParamSpec("P")
_P = ParamSpec("_P")
_R_co = TypeVar("_R_co", covariant=True)
V = TypeVar("V")
U = TypeVar("U")
T = TypeVar("T")
R = TypeVar("R")
_T = TypeVar("_T")
Ts = TypeVarTuple("Ts")

class dynamic(Generic[_T, _P, _R_co]):  # type: ignore # noqa
    """A descriptor that can be used as both a classmethod and instance method.

    Usage:
    ```python
    class MyClass:
        @dynamic()
        def my_method(self_or_cls, arg1, arg2):
            if self_or_cls is MyClass:
                print("Called as class method")
            else:
                print("Called as instance method")

        @dynamic
        def my_property(self_or_cls):
            if self_or_cls is MyClass:
                print("Class property")
            else:
                print("Instance property")

    >>> MyClass.my_method(1, 2)
    Called as class method
    >>> MyClass().my_method(1, 2)
    Called as instance method
    >>> MyClass.my_property
    'Class property'
    >>> MyClass().my_property
    'Instance property'
    ```
    """

    __name__: str
    __qualname__: str
    __doc__: str | None
    __module__: str

    @classmethod
    def call(cls, f: Callable[Concatenate[Any, _P], _R_co] | None = None) -> "Callable[_P, _R_co]":
        """Return a dynamic descriptor in *method* mode.

        Usage::

            class C:
                @dynamic.call
                def method(self):
                    ...
        """
        deco = dynamic()  # method-mode descriptor awaiting the wrapped func
        return deco if f is None else deco(f)

    def __call__(
        self,
        wrapped: Callable[Concatenate[_T | Type[_T], _P], _R_co]
        | Callable[Concatenate[_T, _P], _R_co]
        | Callable[Concatenate[type[_T], _P], _R_co]
        | Callable[[_T], _R_co]
        | None = None,
    ) -> "dynamic[_T, _P, _R_co]":
        """Dynamic member access. Use @dynamic for methods and @dynamic() for properties.

        Note that properties will return the same class-level object for all instances.
        """
        if wrapped is None:
            msg = "Must provide a callable to @dynamic()"
            raise ValueError(msg)
        # store function and mark as *method* mode
        self._func = wrapped
        self.__name__ = getattr(wrapped, "__name__", type(wrapped).__name__)
        self.__qualname__ = getattr(wrapped, "__qualname__", type(wrapped).__name__)
        self.__doc__ = wrapped.__doc__
        self.__module__ = wrapped.__module__
        self._property = False

        return self

    @property
    def __isabstractmethod__(self) -> bool: ...

    @overload
    def __init__(self, f: Callable[Concatenate[type[_T], _P], _R_co]) -> None: ...
    @overload
    def __init__(self, f: Callable[Concatenate[_T, _P], _R_co]) -> None: ...
    @overload
    def __init__(self, f: None = None) -> None: ...
    def __init__(self, f: Callable[Concatenate[type[_T], _P], _R_co] | None = None) -> None:
        if f is None:
            return
        # store function and mark as *method* mode
        self._func = f
        self.__name__ = f.__name__
        self.__qualname__ = f.__qualname__
        self.__doc__ = f.__doc__
        self.__module__ = f.__module__
        self._property = True

    @overload
    def __get__(self, instance: None, owner: type[_T] | None = None, /) -> _R_co: ...
    @overload
    def __get__(self, instance: _T, owner: type[_T] | None = None, /) -> _R_co: ...
    @overload
    def __get__(self, instance: Any, owner: Any, /) -> Callable[_P, _R_co]: ...

    def __get__(self, instance: _T | None = None, owner: type[_T] | None = None) -> _R_co | Callable[_P, _R_co]:
        if self._property:
            if instance is None:
                return self._func.__get__(owner, owner)()

            return self._func.__get__(instance, owner)()

        if instance is None:
            return self._func.__get__(owner, owner)
        return self._func.__get__(instance, owner)

    __name__: str
    __qualname__: str

    @property
    def __wrapped__(self) -> Callable[Concatenate[type[_T], _P], _R_co]: ...
