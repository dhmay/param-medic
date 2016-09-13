#cython: boundscheck=False
#cython: cdivision=True

"""
Mixture-model code reused with permission from Jacob Schreiber's Pomegranate:
https://github.com/jmschrei/pomegranate
"""

import logging
cimport numpy
import numpy
import sys

__author__ = "Damon May"
__copyright__ = "Copyright (c) 2016 Damon May"
__license__ = ""
__version__ = ""

logger = logging.getLogger(__name__)

