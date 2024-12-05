paramdir = 'parameters/'

from .main_functions import *
from .cfg import Cfg
from .grid import Grid
from .setup import Setup
from .data import Data
from .time_marching import *
from .boundary_condition import *

try:
    from ._version import version as __version__
except ImportError:
    __version__ = 'unknown'