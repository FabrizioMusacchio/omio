"""
Compatibility facade for OMIO's historical single-file API.

The implementation is organized into focused modules, but this module keeps
``from omio.omio import ...`` working for existing user code and tests.
"""

from .core import *
from .cache import *
from .templates import *
from .readers.tif import *
from .readers.czi import *
from .readers.thorlabs_raw import *
from .writers.ome_tiff import *
from .viewer import *
from .read import *
from .convert import *
from .batch import *

__all__ = [name for name in globals() if not name.startswith("__")]
