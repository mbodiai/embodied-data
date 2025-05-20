import importlib
import sys
from pydoc import ErrorDuringImport
from types import ModuleType, SimpleNamespace
from typing import Callable, TypeVar, cast, overload

from typing_extensions import TYPE_CHECKING, Any, Literal, TypeAlias

if TYPE_CHECKING:
    import matplotlib as mpl
    import plotext
    from matplotlib import pyplot as pyplt
    MplModule: TypeAlias = mpl
    PltModule: TypeAlias = pyplt
    PlotextModule: TypeAlias = plotext

else:
    try:
        import plotext
        from matplotlib import pyplot as mpl
        PltModule: TypeAlias = mpl
        PlotextModule: TypeAlias = plotext
    except (ImportError, ModuleNotFoundError, AttributeError, NameError):
        PltModule = Any
        PlotextModule = Any
    lazy = Any
    eager = Any
    reload = Any

plt = None


def requires(module: str, wrapped_function: Callable | None = None):

    def inner(func):

        def wrapper(*args, **kwargs):
            if module not in globals():
                msg = f"Module {module} is not installed. Please install with `pip install {module}`"
                raise ImportError(msg)
            return func(*args, **kwargs)

        return wrapper

    if wrapped_function:
        return inner(wrapped_function)
    return inner


PlotBackend = Literal[
    "matplotlib",  # Default matplotlib backend
    "agg",
    "cairo",
    "pdf",
    "pgf",
    "ps",
    "svg",
    "template",
    "widget",
    "qt5",
    "qt6",
    "tk",
    "gtk3",
    "wx",
    "qt4",
    "macosx",
    "nbagg",
    "notebook",
    "inline",
    "ipympl",
    "plotly",
    "tkagg",
    "tcl-tk",
    "tcl-tkagg",
]

PlotBackendT = TypeVar("PlotBackendT", bound=PlotBackend)
PlotTextT = TypeVar("PlotTextT", bound=PlotextModule)
MatplotlibT = TypeVar("MatplotlibT", bound=PltModule)


@overload
def import_plt(backend: Literal["plotext"] | PlotTextT) -> PlotTextT:
    ...


@overload
def import_plt(backend: Literal["matplotlib"] | MatplotlibT) -> MatplotlibT:
    ...


def import_plt(
    backend: PlotBackend | Literal["plotext"] | MatplotlibT | PlotTextT,
) -> PlotTextT | MatplotlibT:  # type: ignore [no-untyped-def]
    try:
        global plt
        if backend == "plotext":
            return cast("PlotTextT", smart_import("plotext"))

        if backend == "matplotlib":
            backend = "tkagg" if sys.platform == "darwin" else backend
        mpl = cast("MplModule", smart_import("matplotlib"))
        mpl.use(backend if isinstance(backend, str) else "tkagg")
        return cast("MatplotlibT", smart_import("matplotlib.pyplot"))

    except (ImportError, AttributeError, NameError) as e:
        if sys.platform == "darwin":
            backend = "tcl-tk" if backend.lower() in ("tk",
                                                      "tkagg") else backend
            msg = f"Failed to import {backend} backend. Hint: Install with `brew install {backend}`"

        msg = f"Failed to import {backend} backend. Hint: Install with `pip install {backend}`"
        raise ImportError(msg) from e


def reload(module: str):
    if module in globals():
        return importlib.reload(globals()[module])
    return cast("ModuleType", importlib.import_module(module))


T = TypeVar("T")


# Lazy import functionality
def import_lazy(module_name: str) -> ModuleType:

    def lazy_module(*args, **kwargs):
        return cast("ModuleType",
                    importlib.import_module(module_name.split(".")[0]))

    return cast("ModuleType", lazy_module)


_warned = {}


def smart_import(name: str,
                 mode: Literal["lazy", "eager", "reload",
                               "type_safe_lazy"] = "eager",
                 suppress_warnings: bool = False) -> Any:
    """Import a module and return the resolved object. Supports . and : delimeters for classes and functions."""
    import sys
    from importlib import import_module, reload
    from pydoc import resolve

    name = name.replace(":", ".")
    try:
        resolved_obj, _ = resolve(name) or (None, None)
        # If object is resolved and not in reload mode
        if resolved_obj and mode != "reload":
            return resolved_obj

        # If module is already imported
        if name.split(".")[0] in sys.modules and mode != "reload":
            return resolved_obj

        if mode == "lazy":

            return import_lazy(name)

        # Import the module or reload if needed
        module_name = name.split(".")[0]
        module = import_module(module_name)
        resolved, _ = resolve(name) or (None, None)
        return reload(module) if mode == "reload" else resolved

    except (ImportError, AttributeError, NameError, ErrorDuringImport) as e:
        # Handle type_safe_lazy mod
        if not suppress_warnings and not _warned.get(name) and mode == "type_safe_lazy":
            _warned[name] = True
            # logging.warning(f"'type_safe_lazy' mode will hide errors from attributes not found in [bold blue]{name}[/bold blue]")
        if mode == "type_safe_lazy":

            class Namespace(SimpleNamespace):

                __call__ = lambda *args, **kwargs: import_lazy(name)(*args, **
                                                                     kwargs)

            return type(Namespace(**{name.split(".")[0]: import_lazy(name)}))

        msg = f"Module {name} not found. Install with `pip install `{name}`"
        raise NameError(msg) from e


def default_export(
    obj: T,
    *,
    key: str | None = None,
) -> T:
    """Assign a function to a module's __call__ attr.

    Args:
        obj: function to be made callable
        key (str): module name as it would appear in sys.modules

    Returns:
        Callable[..., T]: the function passed in

    Raises:
        AttributeError: if key is None and exported obj no __module__ attr
        ValueError: if key is not in sys.modules

    """
    try:
        _module: str = key or obj.__module__
    except AttributeError as e:
        msg = f"Object {obj} has no __module__ attribute. Please provide module key"
        raise AttributeError(msg) from e

    class ModuleCls(ModuleType):

        def __call__(self, *args: Any, **kwargs: Any) -> T:
            return cast("T", obj(*args, **kwargs))  # type: ignore[operator]

    class ModuleClsStaticValue(ModuleCls):

        def __call__(self, *args: Any, **kwargs: Any) -> T:
            return obj

    mod_cls = ModuleCls if callable(obj) else ModuleClsStaticValue

    try:
        sys.modules[_module].__class__ = mod_cls
    except KeyError as e:
        msg = f"{_module} not found in sys.modules"
        raise ValueError(msg) from e
    return obj


@default_export
def make_callable(obj: Callable[..., T],
                  *,
                  key: str | None = None) -> Callable[..., T]:
    """Assign a function to a module's __call__ attr.

    Args:
        obj: function to be made callable
        key (str): module name as it would appear in sys.modules

    Returns:
        Callable[..., T]: the function passed in

    """
    return default_export(obj=obj, key=key)


def bootstrap_third_party(modname: str, location: str) -> ModuleType:
    """Bootstrap third-party libraries with debugging."""
    import sys
    from importlib import import_module
    from importlib.util import find_spec, module_from_spec

    try:
        # Find the module spec
        spec = find_spec(modname)
        if not spec:
            msg = f"Module {modname} not found"
            raise ImportError(msg)  # noqa: TRY301

        # Load the module
        mod = module_from_spec(spec)
        spec.loader.exec_module(mod)

        # Import the parent module at the given location
        new_parent = import_module(location)
        qualified_name = f"{location}.{modname.split('.')[-1]}"

        # Debugging: print information about module and parent

        # Attach the module to the parent
        setattr(new_parent, modname.split(".")[-1], mod)
        sys.modules[qualified_name] = mod

        # Update the globals with the new module
        globals().update({qualified_name: mod})
        globals().update({modname: mod})

        # # Recursively bootstrap submodules if necessary, skipping non-modules
        # for k, v in mod.__dict__.items():
        #     if isinstance(v, ModuleType) and k not in sys.modules and v.__name__.startswith(modname):
        #         bootstrap_third_party(k, qualified_name)

        return mod
    except Exception:
        # Debugging: Catch any errors and print the module causing issues
        raise
