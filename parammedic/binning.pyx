#cython: boundscheck=False
#cython: cdivision=True
"""
Utility code used by multiple modules
"""

import math
import numpy as np
cimport numpy as np
import logging
cimport cython
from cpython cimport array
import util

NDARRAY_DTYPE = np.float32
ctypedef np.float32_t NDARRAY_DTYPE_t


# mass of a hydrogen atom
HYDROGEN_MASS = 1.00794

# N-terminal and C-terminal modifications are indicated with these keys.
# Amino acid modifications are indicated with the relevant AA
MOD_TYPE_KEY_NTERM = "NTERM"
MOD_TYPE_KEY_CTERM = "CTERM"

# unmodified masses of each amino acid
AA_UNMOD_MASSES = {
    'A': 71.03711,
    'C': 103.00919, # note no iodoacetamide
    'D': 115.02694,
    'E': 129.04259,
    'F': 147.06841,
    'G': 57.02146,
    'H': 137.05891,
    'I': 113.08406,
    'K': 128.09496,
    'L': 113.08406,
    'M': 131.04049,
    'N': 114.04293,
    'P': 97.05276,
    'Q': 128.05858,
    'R': 156.10111,
    'S': 87.03203,
    'T': 101.04768,
    'V': 99.06841,
    'W': 186.07931,
    'Y': 163.06333
}

BINNING_MIN_MZ = 50.5 * util.AVERAGINE_PEAK_SEPARATION
BINNING_MAX_MZ = 6000.5 * util.AVERAGINE_PEAK_SEPARATION
N_BINS = int(float(BINNING_MAX_MZ - BINNING_MIN_MZ) / util.AVERAGINE_PEAK_SEPARATION) + 1

logger = logging.getLogger(__name__)

def calc_binidx_for_mass_precursor(float mass, float min_mass_for_bin=BINNING_MIN_MZ,
                                   float bin_size=util.AVERAGINE_PEAK_SEPARATION):
    """
    Calculate the appropriate bin for a given precursor mass
    Note: offset bin divisions so that they occur at the minima between Averagine peaks
    :param mass: 
    :return: 
    """
    return calc_binidx_for_mz_fragment(mass, min_mz_for_bin=min_mass_for_bin, bin_size=bin_size)


def calc_binidx_for_mz_fragment(float mz, float min_mz_for_bin=BINNING_MIN_MZ,
                                float bin_size=util.AVERAGINE_PEAK_SEPARATION):
    """
    Calculate the appropriate bin for a given fragment m/z value.
    Do *not* do range-checking. Can return a negative number, for input < min_mz_for_bin
    Note: offset bin divisions so that they occur at the minima between Averagine peaks
    :param mz: 
    :return: 
    """
    relative_mz = mz - min_mz_for_bin
    return max(0, int(math.floor(relative_mz / bin_size)))


#def bin_spectrum(mz_array, intensity_array,
#                 fragment_min_mz=BINNING_MIN_MZ, fragment_max_mz=BINNING_MAX_MZ,
#                 bin_size=AVERAGINE_PEAK_SEPARATION):
#    """
#    Given an array of m/z values and an array of intensity values for fragment peaks
#    from one spectrum, produce a binned representation of the spectrum.
#
#    Values for each bin represent the intensity of the most-intense peak falling into the bin,
#    as a proportion of the total signal in the spectrum.
#
#    :param mz_array:
#    :param intensity_array:
#    :param fragment_min_mz: low end of the m/z range to represent. Values below this limit ignored.
#    :param fragment_max_mz: high end of the m/z range to represent. Values below this limit ignored.
#    :param bin_size:
#    :return: an ndarray
#    """
#    nbins = int(float(fragment_max_mz - fragment_min_mz) / float(bin_size)) + 1
#    scan_matrix = np.zeros((nbins,))
#
#    for peak_idx in xrange(0, len(mz_array)):
#        mz = mz_array[peak_idx]
#        intensity = intensity_array[peak_idx]
#        if mz < fragment_min_mz or mz > fragment_max_mz:
#            continue
#        bin_idx = calc_binidx_for_mz_fragment(mz, fragment_min_mz, bin_size)
#        if bin_idx < 0 or bin_idx > nbins - 1:
#            continue
#        scan_matrix[bin_idx,] = max(scan_matrix[bin_idx,], intensity)
#    frag_intensity_sum = scan_matrix.sum()
#    if frag_intensity_sum > 0:
#        # divide intensities by sum
#        scan_matrix /= frag_intensity_sum
#    else:
#        # this can happen if the precursor is the only signal in the spectrum!
#        logger.debug("0-intensity spectrum!")
#    return scan_matrix

def bin_spectrum(mz_array, intensity_array, float fragment_min_mz=BINNING_MIN_MZ, float fragment_max_mz=BINNING_MAX_MZ,
                 float bin_size=util.AVERAGINE_PEAK_SEPARATION):
    """
    Given an array of m/z values and an array of intensity values for fragment peaks
    from one spectrum, produce a binned representation of the spectrum.

    Values for each bin represent the intensity of the most-intense peak falling into the bin.

    This code has been somewhat optimized for speed. It is far faster than a full-Python implementation.

    :param mz_array:
    :param intensity_array:
    :param spectra: a generator or list. type is spectra.MS2Spectrum
    :param fragment_min_mz: low end of the m/z range to represent. Values below this limit ignored.
    :param fragment_max_mz: high end of the m/z range to represent. Values below this limit ignored.
    :param bin_size:
    :param precursor_mz:
    :param should_normalize:
    :param should_normalize_exclude_precursor:
    :param window_exclude_precursor_signal:
    :return: an ndarray
    """
    cdef int bin_idx, peak_idx
    cdef int i
    cdef float mz
    cdef float intensity
    cdef int nbins = int(float(fragment_max_mz - fragment_min_mz) / float(bin_size)) + 1
    cdef np.ndarray[NDARRAY_DTYPE_t, ndim=1] scan_matrix = np.zeros((nbins,), dtype=NDARRAY_DTYPE)
    # amount of signal excluded because it's too close to the precursor, when normalizing
    cdef float precursor_signal_to_exclude = 0.0

    for peak_idx in xrange(0, len(mz_array)):
        mz = mz_array[peak_idx]
        intensity = intensity_array[peak_idx]
        if mz < fragment_min_mz or mz > fragment_max_mz:
            continue
        bin_idx = calc_binidx_for_mz_fragment(mz, fragment_min_mz, bin_size)
        if bin_idx < 0 or bin_idx > nbins - 1:
            continue
        scan_matrix[bin_idx,] = max(scan_matrix[bin_idx,], intensity)
    frag_intensity_sum = scan_matrix.sum()
    if frag_intensity_sum > 0:
        # divide intensities by sum
        scan_matrix /= frag_intensity_sum
    else:
        # this can happen if the precursor is the only signal in the spectrum!
        logger.debug("0-intensity spectrum!")
    return scan_matrix

