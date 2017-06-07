#!/usr/bin/env python
"""
Code to infer the presence or absence of a defined set of modifications in 
a given run.
"""

import logging
import math
from util import AVERAGINE_PEAK_SEPARATION, HYDROGEN_MASS

__author__ = "Damon May"
__copyright__ = "Copyright (c) 2016 Damon May"
__license__ = ""
__version__ = ""

logger = logging.getLogger(__name__)


class ReporterIonProportionCalculator(object):
    """
    Class that accumulates the proportion of MS/MS fragment signal that's accounted for
    by a list of reporter ion mzs.
    """
    def __init__(self, reporter_ion_mzs):
        self.n_total_spectra = 0
        self.reporter_ion_bins = set([calc_binidx_for_mz_fragment([mz]) for mz in reporter_ion_mzs])

        self.sum_proportions_in_bins = 0.0
        logger.debug("Reporter ion bins: %s" % str(self.reporter_ion_bins))
        
    def process_spectrum(self, spectrum):
        """
        Handle a spectrum. Check its charge and its number of scans. If passes, find the right
        bin for the precursor. If there's a previous scan in that bin, check to see if we've got
        a pair; if so, record all the peak pair info. Regardless, put the new scan in the bin.
        :param spectrum:
        :return:
        """
        # accounting:
        self.n_total_spectra += 1
        signal_in_bin = 0.0
        signal_total = 0.0
        for i in xrange(0, len(spectrum.mz_array)):
            if calc_binidx_for_mz_fragment(spectrum.mz_array[i]) in self.reporter_ion_bins:
                signal_in_bin += spectrum.intensity_array[i]
            signal_total += spectrum.intensity_array[i]
        self.sum_proportions_in_bins += (signal_in_bin / signal_total)

    def calc_proportion_reporter_ions(self):
        """
        Return the average proportion of signal coming from reporter ions across
        all spectra
        :return: 
        """
        return self.sum_proportions_in_bins / self.n_total_spectra


class PrecursorSeparationProportionCalculator(object):
    """
    Calculate the number of pairs of spectra that are separated by 
    a given set of distances, as a proporation of all pairs of spectra.
    """
    def __init__(self, separation_bin_distances):
        self.bin_distances = separation_bin_distances
        self.spectrum_counts_in_bins = [0] * 10000
        self.highest_binidx_used = 0
        self.n_total_spectra = 0

    def process_spectrum(self, spectrum):
        """
        Just add a count to the bin containing this precursor
        :return: 
        """
        self.n_total_spectra += 1
        precursor_mass = calc_mplush_from_mz_charge(spectrum.precursor_mz, spectrum.charge)
        binidx = calc_binidx_for_mass_precursor(precursor_mass)
        self.highest_binidx_used = max(self.highest_binidx_used, binidx)
        if binidx > len(self.spectrum_counts_in_bins):
            logger.debug("Got a very high precursor mass! %f" % precursor_mass)
            return
        self.spectrum_counts_in_bins[binidx] += 1

    def calc_proportions_with_separations(self):
        """
        Return a dict from mass bin separations to the proportions of spectrum pairs that have
        a precursor having a match with that separation
        :return: 
        """
        # map from separation distances to counts of pairs with that separation
        counts_with_separations = {}
        for bin_distance in self.bin_distances:
            counts_with_separations[bin_distance] = 0
        for i in xrange(0, self.highest_binidx_used - min(self.bin_distances)):
            count_this_bin = self.spectrum_counts_in_bins[i]
            if count_this_bin == 0:
                continue
            for bin_distance in self.bin_distances:
                bin_up = i + bin_distance
                if bin_up < len(self.spectrum_counts_in_bins):
                    count_other_bin = self.spectrum_counts_in_bins[bin_up]
                    counts_with_separations[bin_distance] += count_this_bin * count_other_bin
        proportions_with_separations = {}
        n_total_pairs = self.n_total_spectra * (self.n_total_spectra - 1) / 2
        for bin_distance in self.bin_distances:
            proportions_with_separations[bin_distance] = counts_with_separations[bin_distance] / n_total_pairs
        return proportions_with_separations


def calc_binidx_for_mass_precursor(mass):
    return int(math.floor(mass / AVERAGINE_PEAK_SEPARATION))


def calc_binidx_for_mz_fragment(mz):
    return int(math.floor(mz / AVERAGINE_PEAK_SEPARATION))


def calc_mplush_from_mz_charge(mz, charge):
    """
    Given an mz and a charge, calculate the M+H mass of the ion
    :param mz:
    :param charge:
    :return:
    """
    return (mz - HYDROGEN_MASS) * charge + HYDROGEN_MASS
