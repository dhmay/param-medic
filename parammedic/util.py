#!/usr/bin/env python
"""
Utility code used by multiple modules
"""

# Separation between Averagine peaks. This is used for binning spectra
import abc
import math

AVERAGINE_PEAK_SEPARATION = 1.0005079

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


class MSSpectrum(object):
    """
    represents a single MS spectrum
    """
    def __init__(self, scan_number, retention_time, level, mz_array, intensity_array):
        assert(len(mz_array) == len(intensity_array))
        self.scan_number = scan_number
        self.retention_time = retention_time
        self.mz_array = mz_array
        self.intensity_array = intensity_array
        self.level = level


class MS2Spectrum(MSSpectrum):
    """
    represents a single MS/MS spectrum
    """
    def __init__(self, scan_number, retention_time, mz_array, intensity_array,
                 precursor_mz, charge):
        MSSpectrum.__init__(self, scan_number, retention_time, 2, mz_array, intensity_array)
        self.precursor_mz = precursor_mz
        self.charge = charge

    def generate_mz_intensity_pairs(self):
        for i in xrange(0, len(self.mz_array)):
            yield(self.mz_array[i], self.intensity_array[i])


class RunAttributeDetector(object):
    """
    Abstract superclass for objects to detect different kinds of attributes of runs
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def process_spectrum(self, spectrum):
        """
        Process a single spectrum
        :param spectrum: 
        :return: 
        """
        return
    
    @abc.abstractmethod
    def summarize(self):
        """
        This method gets called after all spectra are processed
        :return: a list of Modifications to be used in search 
        """
        return
    
    @abc.abstractmethod
    def next_file(self):
        """
        Register that a new file is being processed
        :return: 
        """
        return


def calc_binidx_for_mass_precursor(mass):
    """
    Calculate the appropriate bin for a given precursor mass
    Note: offset bin divisions so that they occur at the minima between Averagine peaks
    :param mass: 
    :return: 
    """
    return max(0, int(math.floor((mass + AVERAGINE_PEAK_SEPARATION * 0.5) / AVERAGINE_PEAK_SEPARATION)))


def calc_binidx_for_mz_fragment(mz):
    """
    Calculate the appropriate bin for a given fragment m/z value
    Note: offset bin divisions so that they occur at the minima between Averagine peaks
    :param mz: 
    :return: 
    """
    return max(0, int(math.floor((mz + AVERAGINE_PEAK_SEPARATION * 0.5) / AVERAGINE_PEAK_SEPARATION)))


def calc_mplush_from_mz_charge(mz, charge):
    """
    Given an mz and a charge, calculate the M+H mass of the ion
    :param mz:
    :param charge:
    :return:
    """
    return (mz - HYDROGEN_MASS) * charge + HYDROGEN_MASS


def calc_mz_from_mplush_charge(m_plus_h, charge):
    """
    Given an M+H and a charge, calculate mz
    :param m_plus_h:
    :param charge:
    :return:
    """
    return (m_plus_h - HYDROGEN_MASS) / charge + HYDROGEN_MASS


class Modification(object):
    """
    indicates the location, mass difference and variable or static nature of a modification
    """
    def __init__(self, location, mass_diff, is_variable):
        self.location = location
        self.mass_diff = mass_diff
        self.is_variable = is_variable

    def __str__(self):
        static_variable_str = "Variable" if self.is_variable else "Static"
        location_str = self.location
        if self.location == MOD_TYPE_KEY_NTERM:
            location_str = "N terminus"
        elif self.location == MOD_TYPE_KEY_CTERM:
            location_str = "C terminus"
        return "%s modification of %fDa on %s" % (static_variable_str, self.mass_diff, location_str)
