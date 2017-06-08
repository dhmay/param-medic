#!/usr/bin/env python
"""
Code to infer the presence or absence of a defined set of modifications in 
a given run.

todo: improve efficiency by sharing calculations like total ms signal between 
detectors
"""

import logging
import math

from parammedic.util import RunAttributeDetector
from util import AVERAGINE_PEAK_SEPARATION, HYDROGEN_MASS

__author__ = "Damon May"
__copyright__ = "Copyright (c) 2016 Damon May"
__license__ = ""
__version__ = ""

# default distances between precursors to check for SILAC labeling
# 6Da rationale:
# 13C6 L-Lysine is a stable isotope of 12C6 L-Lysine and is 6 Da heavier than 12C6 L-Lysine.
# 13C6 L- Arginine is a stable isotope of 12C6 L- Arginine and is 6 Da heavier than 12C6 L- Arginine.
# 4Da and 8Da rationale:
# For lysine three-plex experiments, 4,4,5,5-D4 L-lysine and 13C6 15N2 L-lysine are used to
# generate peptides with 4- and 8-Da mass shifts, respectively, compared to peptides generated
# with light lysine.
DEFAULT_SILAC_MOD_BIN_DISTANCES = [4, 6, 8]


DEFAULT_TMT_REPORTERION_MZS = [126.0, 127.0, 128.0, 129.0, 130.0, 131.0]
ITRAQ_4PLEX_REPORTERION_MZS = [114.0, 115.0, 116.0, 117.0]
ITRAQ_8PLEX_REPORTERION_MZS = [113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 121.0]

# number of bins to keep track of for mass. Approximates maximum precursor mass considered
MAX_BINS_FOR_MASS = 20000

# delta mass representing a loss of phosphorylation
DELTA_MASS_PHOSPHO_LOSS = 80.0


logger = logging.getLogger(__name__)


class PhosphoLossProportionCalculator(RunAttributeDetector):
    """
    Accumulates the proportion of MS/MS fragment signal that's accounted for
    by fragments representing a loss of DELTA_MASS_PHOSPHO_LOSS Da from the precursor mass
    """
    def __init__(self):
        self.n_total_spectra = 0
        self.sum_proportions_in_phosopholoss = 0.0

    def next_file(self):
        """
        Register that a new file is being processed
        :return: 
        """
        return

    def process_spectrum(self, spectrum):
        """
        Handle a spectrum. Calculate precursor mass from m/z and charge, then calculate
        mass of phospho loss and convert back to m/z. Look for charge-1 ion representing
        that loss. accumulate proportion of total signal contained in those ions
        :param spectrum: 
        :return: 
        """
        self.n_total_spectra += 1
        precursor_mass = calc_mplush_from_mz_charge(spectrum.precursor_mz, spectrum.charge)
        phospho_loss_mass = precursor_mass - DELTA_MASS_PHOSPHO_LOSS
        phospho_loss_charge1_mz = calc_mz_from_mplush_charge(phospho_loss_mass, 1)
        phospho_loss_bin = calc_binidx_for_mz_fragment(phospho_loss_charge1_mz)
        signal_in_bin = 0.0
        signal_total = 0.0
        for i in xrange(0, len(spectrum.mz_array)):
            if calc_binidx_for_mz_fragment(spectrum.mz_array[i]) == phospho_loss_bin:
                signal_in_bin += spectrum.intensity_array[i]
            signal_total += spectrum.intensity_array[i]
        self.sum_proportions_in_phosopholoss += (signal_in_bin / signal_total)

    def summarize(self):
        """
        Calculate the average proportion of signal coming from reporter ions across
        all spectra
        :return: 
        """
        proportion_phospholoss = self.sum_proportions_in_phosopholoss / self.n_total_spectra
        print("Phosphorylation loss as proportion of total signal: %.04f" % proportion_phospholoss)
    

class ReporterIonProportionCalculator(RunAttributeDetector):
    """
    Class that accumulates the proportion of MS/MS fragment signal that's accounted for
    by a list of reporter ion mzs.
    """
    def __init__(self, name, reporter_ion_mzs):
        self.name = name
        self.n_total_spectra = 0
        self.reporter_ion_bins = set([calc_binidx_for_mz_fragment(mz) for mz in reporter_ion_mzs])

        self.sum_proportions_in_bins = 0.0
        logger.debug("Reporter ion bins: %s" % str(self.reporter_ion_bins))

    def next_file(self):
        """
        Register that a new file is being processed
        :return: 
        """
        return
        
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

    def summarize(self):
        """
        Calculate the average proportion of signal coming from reporter ions across
        all spectra
        :return: 
        """
        proportion_reporter_ions = self.sum_proportions_in_bins / self.n_total_spectra
        print("%s reporter ions as proportion of total signal: %.03f" % 
              (self.name, proportion_reporter_ions))


class PrecursorSeparationProportionCalculator(RunAttributeDetector):
    """
    Calculate the number of pairs of spectra that are separated by 
    a given set of distances, as a proporation of all pairs of spectra.
    """
    def __init__(self, name, separation_bin_distances):
        self.name = name
        self.bin_distances = separation_bin_distances
        self.spectrum_counts_in_bins = [0] * MAX_BINS_FOR_MASS
        self.highest_binidx_used = 0
        self.n_total_spectra = 0

    def next_file(self):
        """
        Register that a new file is being processed
        :return: 
        """
        return

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

    def summarize(self):
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
        logger.debug("summarize(): n_total_spectra=%d, n_total_pairs=%d" % (self.n_total_spectra, n_total_pairs))
        for bin_distance in self.bin_distances:
            proportions_with_separations[bin_distance] = float(counts_with_separations[bin_distance]) / n_total_pairs
            logger.debug("count with separation %d: %d" % (bin_distance, counts_with_separations[bin_distance]))
        overall_proportion = sum(proportions_with_separations.values())
        print("%s: proportion of precursor pairs with appropriate separation: overall=%.05f" %
              (self.name, overall_proportion))
        for bin_distance in self.bin_distances:
            print("    proportion with separation %dDa: %.05f" %
                  (bin_distance, proportions_with_separations[bin_distance]))


# utility methods

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


def calc_mz_from_mplush_charge(m_plus_h, charge):
    """
    Given an M+H and a charge, calculate mz
    :param m_plus_h:
    :param charge:
    :return:
    """
    return (m_plus_h - HYDROGEN_MASS) / charge + HYDROGEN_MASS
