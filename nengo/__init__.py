"""
Nengo
=====

Nengo provides a package for doing large-scale brain modelling in Python.

The source code repository for this package is found at
https://www.github.com/nengo/nengo. Examples of models can be found
in the `examples` directory of the source code repository.
"""

__copyright__ = "2013-2014, Applied Brain Research"
__license__ = "Free for non-commercial use; see LICENSE.rst"
from .version import version as __version__

import logging

# Nengo namespace (API)
from .config import Config
from .connection import Connection
from .ensemble import Ensemble
from .node import Node
from .neurons import (AdaptiveLIF, AdaptiveLIFRate, Direct, Izhikevich, LIF,
                      LIFRate, RectifiedLinear, Sigmoid)
from .network import Network
from .learning_rules import PES, BCM, Oja
from .params import Default
from .probe import Probe
from .rc import rc, RC_DEFAULTS
from .simulator import Simulator
from .synapses import Alpha, LinearFilter, Lowpass
from .utils.logging import log
from . import dists, networks, processes, utils

logger = logging.getLogger(__name__)
try:
    # Prevent output if no handler set
    logger.addHandler(logging.NullHandler())
except AttributeError:
    pass
