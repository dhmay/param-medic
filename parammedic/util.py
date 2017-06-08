#!/usr/bin/env python
"""
Utility code used by multiple modules
"""

# Separation between Averagine peaks. This is used for binning spectra
import abc

AVERAGINE_PEAK_SEPARATION = 1.0005079

# mass of a hydrogen atom
HYDROGEN_MASS = 1.00794

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
        :return: 
        """
        return
    
    @abc.abstractmethod
    def next_file(self):
        """
        Register that a new file is being processed
        :return: 
        """
        return
