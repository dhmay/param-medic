#!/usr/bin/env python
"""
Code to infer the presence or absence of a defined set of modifications in 
a given run.

todo: improve efficiency by sharing calculations like total ms signal between 
detectors
"""

import logging
import math
import numpy as np

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

# control m/z values to compare with reporter ion groups. These occur below,
# between and above the different groups of reporter ions.
REPORTER_ION_CONTROL_MZS = [111.0, 112.0,  # below
                            122.0, 123.0, 124.0, 125.0,  # between
                            132.0, 133.0]  # above

# number of bins to keep track of for mass. Approximates maximum precursor mass considered
MAX_BINS_FOR_MASS = 20000

# delta mass representing a loss of phosphorylation
# Phospho is lost as (H3PO4, -98Da), according to Villen:
#http://faculty.washington.edu/jvillen/wordpress/wp-content/uploads/2016/04/Beausoleil_PNAS_04.pdf
DELTA_MASS_PHOSPHO_LOSS = 98.0
# offsets from the phospho peak to use as control peaks.
# Don't use 1 or 2 above the phospho peak, because could be M+1 and M+2 peaks for the phospho peak
PHOSPHO_CONTROLPEAK_PHOSPHOPEAK_OFFSETS = [-4, -3, -2, -1, 3, 4, 5, 6]


logger = logging.getLogger(__name__)


class PhosphoLossProportionCalculator(RunAttributeDetector):
    """
    Accumulates the proportion of MS/MS fragment signal that's accounted for
    by fragments representing a loss of DELTA_MASS_PHOSPHO_LOSS Da from the precursor mass
    """
    def __init__(self):
        self.n_total_spectra = 0
        self.sum_proportions_in_phosopholoss = 0.0
        # map from offset from phospho loss bin to sum of proportion
        self.sums_proportions_per_controlpeak = {}
        for offset in PHOSPHO_CONTROLPEAK_PHOSPHOPEAK_OFFSETS:
            self.sums_proportions_per_controlpeak[offset] = 0.0

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
        # look in same charge as precursor, because phospho loss is neutral
        phospho_loss_charge1_mz = calc_mz_from_mplush_charge(phospho_loss_mass, spectrum.charge)
        phospho_loss_bin = calc_binidx_for_mz_fragment(phospho_loss_charge1_mz)
        control_bins = []
        signal_total = sum(spectrum.intensity_array)
        for offset in PHOSPHO_CONTROLPEAK_PHOSPHOPEAK_OFFSETS:
            control_bins.append(phospho_loss_bin + offset)
        for i in xrange(0, len(spectrum.mz_array)):
            frag_binidx = calc_binidx_for_mz_fragment(spectrum.mz_array[i])
            if frag_binidx == phospho_loss_bin:
                self.sum_proportions_in_phosopholoss += spectrum.intensity_array[i] / signal_total
            elif frag_binidx in control_bins:
                self.sums_proportions_per_controlpeak[frag_binidx - phospho_loss_bin] += (spectrum.intensity_array[i] / signal_total)

    def summarize(self):
        """
        Calculate the average proportion of signal coming from reporter ions across
        all spectra
        :return: 
        """
        control_mean = np.mean(self.sums_proportions_per_controlpeak.values())
        control_sd = np.std(self.sums_proportions_per_controlpeak.values())
        proportion_to_control = self.sum_proportions_in_phosopholoss / control_mean
        zscore_to_control = (self.sum_proportions_in_phosopholoss - control_mean) / control_sd
        print("Phospho: ratio phospho-loss to control peaks: %.05f (z=%.03f)" % (proportion_to_control, zscore_to_control))



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
    # bin distances to use for comparison. Param-Medic assumes there won't be any
    # excessive pairing of precursors at these mass distances
    CONTROL_BIN_DISTANCES = [11, 14, 15, 21, 23, 27]
    MAX_SCAN_SEPARATION = 50

    def __init__(self, name, separation_bin_distances):
        self.name = name
        self.bin_distances = separation_bin_distances
        self.scan_numbers = []
        self.precursor_mass_bins = []
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
        self.scan_numbers.append(spectrum.scan_number)
        precursor_mass = calc_mplush_from_mz_charge(spectrum.precursor_mz, spectrum.charge)
        binidx = calc_binidx_for_mass_precursor(precursor_mass)
        self.precursor_mass_bins.append(binidx)

    def summarize(self):
        """
        Return a dict from mass bin separations to the proportions of spectrum pairs that have
        a precursor having a match with that separation
        :return: 
        """
        # map from separation distances to counts of pairs with that separation
        counts_with_separations = {}
        separations_to_evaluate = set(self.bin_distances + PrecursorSeparationProportionCalculator.CONTROL_BIN_DISTANCES)
        if len(separations_to_evaluate) < len(self.bin_distances) + len(PrecursorSeparationProportionCalculator.CONTROL_BIN_DISTANCES):
            logger.warn("A specified separation is also a control separation! Specified: %s" % str(self.bin_distances))
        for separation in separations_to_evaluate:
            counts_with_separations[separation] = 0
        
        minidx = 0
        maxidx = 0
        for i in xrange(0, len(self.scan_numbers)):
            scan_number = self.scan_numbers[i]
            min_scan_number = scan_number - PrecursorSeparationProportionCalculator.MAX_SCAN_SEPARATION
            max_scan_number = scan_number + PrecursorSeparationProportionCalculator.MAX_SCAN_SEPARATION
            while self.scan_numbers[minidx] < min_scan_number:
                minidx += 1
            while self.scan_numbers[maxidx] < max_scan_number and maxidx < len(self.scan_numbers) - 1:
                maxidx += 1
            for j in xrange(minidx, maxidx):
                separation = abs(self.precursor_mass_bins[i] - self.precursor_mass_bins[j])
                if separation in separations_to_evaluate:
                    counts_with_separations[separation] += 1
        mean_control_count = (float(sum([counts_with_separations[separation] for separation in
                                    PrecursorSeparationProportionCalculator.CONTROL_BIN_DISTANCES])) /
                               len(PrecursorSeparationProportionCalculator.CONTROL_BIN_DISTANCES))
        logger.debug("Control separation counts:")
        for separation in PrecursorSeparationProportionCalculator.CONTROL_BIN_DISTANCES:
            logger.debug("  %d: %d" % (separation, counts_with_separations[separation]))
        logger.debug("Mean control separation count: %.05f" % mean_control_count)
        logger.debug("Counts of interest:")
        for separation in self.bin_distances:
            proportion_to_control = float(counts_with_separations[separation]) / mean_control_count
            logger.debug("  %d: %d (proportion=%.05f)" % (separation, counts_with_separations[separation], proportion_to_control))
        print("Ratios of %s mass separations to control separations:" % (self.name))
        control_sd = np.std([counts_with_separations[separation] for separation in PrecursorSeparationProportionCalculator.CONTROL_BIN_DISTANCES])
        for separation in self.bin_distances:
            proportion_to_control = float(counts_with_separations[separation]) / mean_control_count
            zscore_to_control = float(counts_with_separations[separation] - mean_control_count) / control_sd
            print("    %dDa: %.05f (z=%.03f)" % (separation, proportion_to_control, zscore_to_control))

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
