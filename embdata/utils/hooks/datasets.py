from embdata.utils.import_utils import smart_import

try:
  datasets = smart_import("datasets")
  from datasets import Dataset, Features
except (ImportError, ModuleNotFoundError, NameError, AttributeError):
  class Dataset:...
  class Features:...

__all__ = ["Dataset", "Features", "datasets"]
