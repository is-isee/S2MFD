"""
S2MFD: A Python package for solving 2D mean field dynamo equations in background of solar/stellar convection zone.

Example
-------
>>> import S2MFD
>>> S2MFD.run_simulation()

"""

paramdir = 'parameters/'

from .main_functions import run_simulation
from .cfg import Cfg
from .grid import Grid
from .setup import Setup
from .data import Data
from .simulation import Simulation
from . import physics
from . import tools

__all__ = [ 'Cfg',
            'Grid',
            'Setup',
            'Data',
            'Simulation',
            'physics',
            'tools',
            'run_simulation',
            ]

try:
    from ._version import version as __version__
except ImportError:
    __version__ = 'unknown'