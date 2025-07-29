"""
S2MFD: A Python package for solving 2D mean field dynamo equations in background of solar/stellar convection zone.

Example
-------
>>> import S2MFD
>>> S2MFD.run_simulation()

"""

paramdir = 'parameters/'

from .main_functions import run_simulation
from .bisection_main_functions import bisection_simulation
from .defunction_main_functions import defunction_simulation
from .cfg import Cfg
from .grid import Grid
from .legendre import Legendre
from .setup import Setup
from .data import Data
from .simulation import Simulation
from . import physics
from . import tools
from .genetic_algorithm_fourier3 import GA_defunction
from .GA_for_OBS import GA_for_OBS
from .make_graph import make_graph


__all__ = [ 'Cfg',
            'Grid',
            'Legendre',
            'Setup',
            'Data',
            'Simulation',
            'physics',
            'tools',
            'run_simulation',
            'bisection_simulation',
            'defunction_simulation',
            'GA_defunction',
            'GA_for_OBS',
            'make_graph'
            ]

try:
    from ._version import version as __version__
except ImportError:
    __version__ = 'unknown'