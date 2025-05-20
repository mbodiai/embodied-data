from types import ModuleType

from embdata.utils.import_utils import smart_import

try:
  torch = smart_import("torch")
  Tensor = smart_import("torch.tensor:Tensor")
except (ImportError, ModuleNotFoundError, NameError, AttributeError):
  torch: ModuleType = ...
  class Tensor:...

