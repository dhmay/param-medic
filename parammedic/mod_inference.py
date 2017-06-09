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
    by a list of reporter ion mzs for each of multiple types.
    """
    def __init__(self, reporter_ion_type_mzs_map):
        self.n_total_spectra = 0
        # map from reporter type to the bins representing that type
        self.reporter_ion_type_bins_map = {}
        # map from reporter type to sum of proportions of fragment ion intensities in bins for that type
        self.reportertype_sum_proportion_map = {}
        for reporter_ion_type in reporter_ion_type_mzs_map:
            self.reporter_ion_type_bins_map[reporter_ion_type] = []
            for mz in reporter_ion_type_mzs_map[reporter_ion_type]:
                self.reporter_ion_type_bins_map[reporter_ion_type].append(calc_binidx_for_mz_fragment(mz))
            self.reportertype_sum_proportion_map[reporter_ion_type] = 0.0

        logger.debug("Reporter ion type count: %d" % len(self.reporter_ion_type_bins_map))

    def next_file(self):
        """
        Register that a new file is being processed
        :return: 
        """
        return
        
    def process_spectrum(self, spectrum):
        """
        Process a single spectrum, checking all fragment mzs against the lists of 
        mzs for each reporter type
        :param spectrum:
        :return:
        """
        # accounting
        self.n_total_spectra += 1
        signal_total = 0.0
        reportertype_signalsum_map_this_spectrum = {}
        # construct a temporary map from reporter type to sum of signal in this spectrum for that type.
        # I could make this a class variable and just zero it out each time.
        for reporter_type in self.reporter_ion_type_bins_map:
            reportertype_signalsum_map_this_spectrum[reporter_type] = 0.0
        for i in xrange(0, len(spectrum.mz_array)):
            mz_bin = calc_binidx_for_mz_fragment(spectrum.mz_array[i])
            intensity_i = spectrum.intensity_array[i]
            # for each reporter type, check if the mz_bin is one of the bins for that type.
            # if so, add this peak's intensity to the sum for that type.
            for reporter_type in self.reporter_ion_type_bins_map:
                if mz_bin in self.reporter_ion_type_bins_map[reporter_type]:
                    reportertype_signalsum_map_this_spectrum[reporter_type] += intensity_i
            signal_total += spectrum.intensity_array[i]
        for reporter_type in reportertype_signalsum_map_this_spectrum:
            proportion_this_type = reportertype_signalsum_map_this_spectrum[reporter_type] / signal_total
            self.reportertype_sum_proportion_map[reporter_type] += proportion_this_type

    def summarize(self):
        """
        Calculate the average proportion of signal coming from reporter ions across
        all spectra
        :return: 
        """
        for reporter_type in self.reportertype_sum_proportion_map:
            # divide the sum of proportions of signals in ions for this reporter type
            # by 1.0 * the total number of spectra considered
            proportion_reporter_ions = self.reportertype_sum_proportion_map[reporter_type] / self.n_total_spectra
            print("%s: proportion of total signal: %.03f" %
                  (reporter_type, proportion_reporter_ions))


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
