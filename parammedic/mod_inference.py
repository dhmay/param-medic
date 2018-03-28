#!/usr/bin/env python
"""
Code to infer the presence or absence of each of a defined set of modifications in
a given run.

Each type of modification is detected by a separate class that inherits from RunAttributeDetector.
"""

import logging
from parammedic import util

import numpy as np
from scipy.stats import ttest_ind

from parammedic.util import RunAttributeDetector, calc_mplush_from_mz_charge
from parammedic.binning import calc_binidx_for_mass_precursor, calc_binidx_for_mz_fragment

# for finding locations of m/z peaks in sorted lists
import bisect

__author__ = "Damon May"
__copyright__ = "Copyright (c) 2016 Damon May"
__license__ = ""
__version__ = ""

# SILAC constants

# default distances between precursors to check for SILAC labeling
# 6Da rationale:
# 13C6 L-Lysine is a stable isotope of 12C6 L-Lysine and is 6 Da heavier than 12C6 L-Lysine.
# 13C6 L- Arginine is a stable isotope of 12C6 L- Arginine and is 6 Da heavier than 12C6 L- Arginine.
# 4Da and 8Da rationale:
# For lysine three-plex experiments, 4,4,5,5-D4 L-lysine and 13C6 15N2 L-lysine are used to
# generate peptides with 4- and 8-Da mass shifts, respectively, compared to peptides generated
# with light lysine.
SILAC_MOD_BIN_DISTANCES = [4, 6, 8, 10]
# z-score cutoff above which we consider a particular SILAC separation to be present
SILAC_ZSCORE_CUTOFF = 4.0

# TMT constants
# I got the exact values from here: http://lgatto.github.io/MSnbase/reference/TMT6.html
TMT_2PLEX_REPORTERION_MZS = [126.1277, 127.1311]
# the TMT 6-plex reporter ion m/zs that are NOT in the 4plex ion list
TMT_6PLEXONLY_REPORTERION_MZS = [128.1344, 129.1378, 130.1411, 131.1382]

# the TMT 10-plex reporter ion m/zs that are NOT in the 2- and 4-plex ion list.
# Note: because these are so close to the 2-plex and 10-plex ions, it only makes sense to use these
# in a high-resolution pass
#TMT_10PLEXONLY_REPORTERION_MZS = [127.1248, 128.1281, 129.1315, 130.1348]

ITRAQ_4PLEX_REPORTERION_MZS = [114.0, 115.0, 116.0, 117.0]
# the iTRAQ 8-plex reporter ion m/zs that are NOT in the 4plex ion list
ITRAQ_8PLEXONLY_REPORTERION_MZS = [117.0, 118.0, 119.0, 121.0]

# control m/z values to compare with reporter ion groups. These occur below,
# between and above the different groups of reporter ions.
REPORTER_ION_CONTROL_MZS = [111.0, 112.0,  # below
                            120.0, 122.0, 123.0, 124.0, 125.0,  # between
                            133.0, 134.0]  # above
REPORTER_ION_TYPES = ["TMT_6plex", "iTRAQ_4plex", "iTRAQ_8plex"]

# map from reporter ion type to the mzs of that class of reporters
REPORTER_ION_TYPE_MZS_MAP = {
    "TMT_2plex": TMT_2PLEX_REPORTERION_MZS,
    "TMT_6plex": TMT_6PLEXONLY_REPORTERION_MZS,
    "iTRAQ_4plex": ITRAQ_4PLEX_REPORTERION_MZS,
    "iTRAQ_8plex": ITRAQ_8PLEXONLY_REPORTERION_MZS,
    "control": REPORTER_ION_CONTROL_MZS
}

REPORTER_ION_TSTAT_THRESHOLDS_MAP = {
    "TMT_2plex": 2.0,
    "TMT_6plex": 2.0,
    "iTRAQ_4plex": 1.5,
    "iTRAQ_8plex": 1.5,
}

# z-score cutoff above which we consider a particular reporter ion to be present.
# Values derived empirically for each reporter
ITRAQ_REPORTER_ION_ZSCORE_CUTOFF = 2.0
TMT_REPORTER_ION_ZSCORE_CUTOFF = 3.5


# number of bins to keep track of for mass. Approximates maximum precursor mass considered
MAX_BINS_FOR_MASS = 20000

# delta mass representing a loss of phosphorylation
# Phospho is typically lost as a neutral (H3PO4, -98Da), according to Villen (and many others):
#http://faculty.washington.edu/jvillen/wordpress/wp-content/uploads/2016/04/Beausoleil_PNAS_04.pdf
DELTA_MASS_PHOSPHO_LOSS = 98.0
# offsets from the phospho peak to use as control peaks.
# Don't use 1 or 2 above the phospho peak, because could be M+1 and M+2 peaks for the phospho peak
PHOSPHO_CONTROLPEAK_PHOSPHOPEAK_OFFSETS = [-20, -15, -12, -10, 10, 12, 15, 20]

# z-score cutoff above which we consider phosphorylation to be present
PHOSPHO_ZSCORE_CUTOFF = 9.0

# modification masses for different reagents are from www.unimod.org

# http://www.unimod.org/modifications_view.php?editid1=214
# except using this from Jimmy Eng, instead. Average of masses for 114 (144.105918), 115 (144.09599), and 116/117 (144.102063)
SEARCH_MOD_MASS_ITRAQ_4PLEX = 144.10253
# http://www.unimod.org/modifications_view.php?editid1=730
#SEARCH_MOD_MASS_ITRAQ_8PLEX = 304.205360
# using this instead, from Jimmy Eng. Average of masses for 115/118/119/121 (304.199040) and 113/114/116/117 (304.205360)
SEARCH_MOD_MASS_ITRAQ_8PLEX = 304.2022
# http://www.unimod.org/modifications_view.php?editid1=738
SEARCH_MOD_MASS_TMT_2PLEX = 225.155833
# http://www.unimod.org/modifications_view.php?editid1=737
SEARCH_MOD_MASS_TMT_6PLEX = 229.162932
# http://www.unimod.org/modifications_view.php?editid1=21
SEARCH_MOD_MASS_PHOSPHO = 79.966331

# maps giving exact mass difference for nominal mass differences for K and R labels.
# One suggestion is made at random from the options in Unimod
SILAC_MOD_K_EXACTMASS_MAP = {
    4: 4.025107,  # http://www.unimod.org/modifications_view.php?editid1=481
    6: 6.020129,  # http://www.unimod.org/modifications_view.php?editid1=188
    8: 8.014199,  # http://www.unimod.org/modifications_view.php?editid1=259
    10: 10.008269  # http://www.unimod.org/modifications_view.php?editid1=267
}

SILAC_MOD_R_EXACTMASS_MAP = {
    6: 6.020129,  # http://www.unimod.org/modifications_view.php?editid1=188
    10: 10.008269  # http://www.unimod.org/modifications_view.php?editid1=267
}

logger = logging.getLogger(__name__)


class PhosphoLossProportionCalculator(RunAttributeDetector):
    """
    Accumulates the proportion of MS/MS fragment signal that's accounted for
    by fragments representing a loss of DELTA_MASS_PHOSPHO_LOSS Da from the precursor mass,
    and also by control-peak separations. Compares the phospho signal with the control
    peaks to calculate a statistic and a recommendation.
    """
    def __init__(self):
        self.sum_proportions_in_phosopholoss = 0.0
        # map from offset from phospho loss bin to sum of proportion
        self.sums_proportions_per_controlpeak = {}
        self.n_spectra_used = 0
        for offset in PHOSPHO_CONTROLPEAK_PHOSPHOPEAK_OFFSETS:
            self.sums_proportions_per_controlpeak[offset] = 0.0

    def process_spectrum(self, spectrum, binned_spectrum):
        """
        Process a spectrum. Calculate precursor mass from m/z and charge, then calculate
        mass of phospho loss and convert back to m/z. Look for ion representing
        that loss, in same charge as precursor. accumulate proportion of total signal 
        contained in those ions. Do the same thing for several control ions
        :param spectrum: 
        :param binned_spectrum: 
        :return: 
        """
        if spectrum.charge < 1:
            # can't use spectra of unknown charge
            return
        self.n_spectra_used += 1
        
        # phospho loss peak is DELTA_MASS_PHOSPHO_LOSS lighter than precursor, and in the same charge state
        phospho_loss_mz = spectrum.precursor_mz - (DELTA_MASS_PHOSPHO_LOSS / spectrum.charge)
        phospho_loss_bin = calc_binidx_for_mz_fragment(phospho_loss_mz)
        # increment the control bins for this spectrum
        for offset in PHOSPHO_CONTROLPEAK_PHOSPHOPEAK_OFFSETS:
            self.sums_proportions_per_controlpeak[offset] += binned_spectrum[phospho_loss_bin + offset]
        self.sum_proportions_in_phosopholoss += binned_spectrum[phospho_loss_bin]

    def summarize(self):
        """
        Calculate the average proportion of signal coming from phospho-loss separations across
        all spectra. Calculate z-score vs. control separations. Make a determination, report the results.
        :return: 
        """
        if self.n_spectra_used == 0:
            logger.warn("No spectra usable for phosphorylation detection!")
            result = util.RunAttributeResult()
            result.name_value_pairs['phospho_present'] = 'ERROR'
            result.name_value_pairs['phospho_statistic'] = 'ERROR'
            return result

        control_mean = np.mean(self.sums_proportions_per_controlpeak.values())
        control_sd = np.std(self.sums_proportions_per_controlpeak.values())
        logger.debug("Phospho control peaks (mean=%.03f):" % control_mean)
        for control_peak in self.sums_proportions_per_controlpeak:
            logger.debug("    %d: %.04f" % (control_peak, self.sums_proportions_per_controlpeak[control_peak]))
        proportion_to_control = self.sum_proportions_in_phosopholoss / control_mean
        zscore_to_control = (self.sum_proportions_in_phosopholoss - control_mean) / control_sd
        logger.debug("Phospho-loss peak: %.03f" % self.sum_proportions_in_phosopholoss)
        logger.debug("Phospho: ratio phospho-loss to control peaks: %.05f (z=%.03f)" %
                     (proportion_to_control, zscore_to_control))

        # summarize results
        search_modifications = []
        if zscore_to_control > PHOSPHO_ZSCORE_CUTOFF:
            logger.info("Phosphorylation: detected")
            search_modifications.append(util.Modification(util.MOD_TYPE_KEY_CTERM, SEARCH_MOD_MASS_PHOSPHO, True))
            phospho_is_present = True
        else:
            logger.info("Phosphorylation: not detected")
            phospho_is_present = False

        result = util.RunAttributeResult()
        result.name_value_pairs['phospho_present'] = 'T' if phospho_is_present else 'F'
        result.name_value_pairs['phospho_statistic'] = str(zscore_to_control)
        result.search_modifications = search_modifications
        return result

# Constants and data structures used by the TMT6/TMT10 detector

# at least this many of the four TMT10 peaks must comprise at least this proportion of the
# TMT610_WINDOW_WIDTH area around the TMT10 peak and associated TMT6 peak 
# in order to declare TMT10 present
TMT610_MINPEAKS_TMT_PRESENT_FOR_DECISION = 2
TMT610_MIN_TMT10_PROPORTION_FOR_DECISION = 0.2

# define the nominal and precise TMT6 and TMT10 reporter ion masses
# http://www.unimod.org/modifications_view.php?editid1=737
# TMT6 masses: 126.12773 127.12476 128.13443 129.13147 130.14114 131.13818
# TMT10 masses: 127.13108 128.12811 129.13779 130.13482

TMT610_DOUBLE_REAGENT_PEAK_NOMINALMASSES = [127, 128, 129, 130]

# map from nominal mass to precise TMT6 mass
TMT610_NOMINALMASS_TMT6MASS_MAP = {
    127: 127.12476,
    128: 128.13443,
    129: 129.13147,
    130: 130.14114
}

# map from nominal mass to precise TMT10 mass
TMT610_NOMINALMASS_TMT10MASS_MAP = {
    127: 127.13108,
    128: 128.12811,
    129: 129.13779,
    130: 130.13482
}

# Define the window to use for TMT6 and TMT10 peak assessment

# this is roughly the difference between the TMT6 and TMT10 masses
TMT10_MASSDIFF_FROM6 = .00632
# amount of padding on each size of the 6 and 10 peaks. I'm relating this size to the size of
# the difference between TMT6 and TMT10 ions.
TMT10_PADDING_EACHSIDE = TMT10_MASSDIFF_FROM6 * 5
# window width for counting peaks near the TMT6 and TMT10 masses
TMT610_WINDOW_WIDTH = TMT10_PADDING_EACHSIDE * 2 + TMT10_MASSDIFF_FROM6

# size of a small window around each individual expected peak, for use in deciding
# whether each TMT10 peak is present
TMT610_PEAK_WIDTH_FOR_DETECT = 0.003

# map from each nominal mass to the minimum mass in the window to consider
TMT610_NOMINALMASS_MINBINMASS_MAP = {}

# define the minimum TMT mass (6 or 10) for each nominal mass
for nominal_mass in TMT610_NOMINALMASS_TMT6MASS_MAP:
    TMT610_NOMINALMASS_MINBINMASS_MAP[nominal_mass] = min(TMT610_NOMINALMASS_TMT10MASS_MAP[nominal_mass],
                                                          TMT610_NOMINALMASS_TMT6MASS_MAP[nominal_mass]) \
                                                      - TMT10_PADDING_EACHSIDE

# this is just 127, but let's define it cleanly
TMT610_SMALLEST_NOMINAL_MASS = min(TMT610_NOMINALMASS_MINBINMASS_MAP)

# define the range of values that TMT peaks can occur in
TMT610_SMALLEST_BIN_CENTER = TMT610_SMALLEST_NOMINAL_MASS * util.AVERAGINE_PEAK_SEPARATION
TMT610_SMALLEST_BIN_MINMASS = TMT610_SMALLEST_BIN_CENTER - util.AVERAGINE_PEAK_SEPARATION / 2
TMT610_BIGGEST_BIN_MAXMASS = TMT610_NOMINALMASS_MINBINMASS_MAP[
                                 max(TMT610_DOUBLE_REAGENT_PEAK_NOMINALMASSES)] + TMT610_WINDOW_WIDTH

# Minimum number of peaks within the window around each nominal mass in order to try to
# detect the presence of TMT 10
TMT610_MINPEAKS_FOR_DETECT = 30


class TMT6vs10Detector(RunAttributeDetector):
    """
    Class for detecting the presence of TMT10. If not detected, assume TMT6.
    Basic approach is to define a small window around, for each nominal mass that could
    have TMT6 and TMT10 peaks in it, both of those peaks, and assess the proportion of
    observed peaks in that window that fall into a small range around the TMT peak.
    """

    def __init__(self):
        self.nominalmass_allpeaks_map = {}
        for doublepeak_nominalmass in TMT610_DOUBLE_REAGENT_PEAK_NOMINALMASSES:
           self.nominalmass_allpeaks_map[doublepeak_nominalmass] = []

    def process_spectrum(self, spectrum, binned_spectrum):
        """
        Process a single spectrum, checking all fragment mzs against 
        DOUBLE_REAGENT_PEAK_NOMINALMASSES and adding in-range fragments to each
        bin appropriately
        :param spectrum:
        :return: a RunAttributeResult
        """

        if spectrum.level not in (2, 3):
            return
        # find the smallest peak in this scan that might be in one of the double-reagent bins
        idx_smallestpossible = bisect.bisect_left(spectrum.mz_array, TMT610_SMALLEST_BIN_MINMASS)
        assert (spectrum.mz_array[idx_smallestpossible] >= TMT610_SMALLEST_BIN_MINMASS)
        # print(min(spectrum.mz_array))
        curidx = idx_smallestpossible
        while curidx < len(spectrum.mz_array) and spectrum.mz_array[curidx] < TMT610_BIGGEST_BIN_MAXMASS:
            cur_mz = spectrum.mz_array[curidx]

            # int() does floor. Add 1 because 0-index vs. 1-index
            cur_bin = int((cur_mz - util.AVERAGINE_PEAK_SEPARATION / 2) / util.AVERAGINE_PEAK_SEPARATION) + 1
            if TMT610_NOMINALMASS_MINBINMASS_MAP[cur_bin] < cur_mz < TMT610_NOMINALMASS_MINBINMASS_MAP[
                cur_bin] + TMT610_WINDOW_WIDTH:
                # print("{}: {} + {}".format(cur_mz, cur_bin, cur_offset))
                self.nominalmass_allpeaks_map[cur_bin].append(cur_mz)
            curidx += 1

    def summarize(self):
        """
        Calculate the average proportion of signal coming from reporter ions across
        all spectra
        :return: a tuple of values indicating whether TMT10 is present and how many TMT10 peaks were detected
        """
        n_peaks_with_enough_TMT10_signal = 0
        for nominalmass in self.nominalmass_allpeaks_map:
            peaks_this_nominalmass = self.nominalmass_allpeaks_map[nominalmass]
            n_peaks_this_nominalmass = len(peaks_this_nominalmass)
            logger.debug("nominal mass {}: {}".format(nominalmass, n_peaks_this_nominalmass))
            if n_peaks_this_nominalmass >= TMT610_MINPEAKS_FOR_DETECT:
                mzs_this_nominalmass = self.nominalmass_allpeaks_map[nominalmass]
                smallest_mz_thisbin = nominalmass * util.AVERAGINE_PEAK_SEPARATION - util.AVERAGINE_PEAK_SEPARATION / 2
                logger.debug("    {}-{}".format(min(mzs_this_nominalmass), max(mzs_this_nominalmass)))
                # logger.debug(sorted(mzs_this_nominalmass))
                # find the proportion of peaks falling in the expected range for each TMT ion at this nominal mass
                peak6_mz_thisnominalmass = TMT610_NOMINALMASS_TMT6MASS_MAP[nominalmass]
                peak10_mz_thisnominalmass = TMT610_NOMINALMASS_TMT10MASS_MAP[nominalmass]
    
                minmass_peak6 = peak6_mz_thisnominalmass - TMT610_PEAK_WIDTH_FOR_DETECT / 2
                maxmass_peak6 = peak6_mz_thisnominalmass + TMT610_PEAK_WIDTH_FOR_DETECT / 2
                minmass_peak10 = peak10_mz_thisnominalmass - TMT610_PEAK_WIDTH_FOR_DETECT / 2
                maxmass_peak10 = peak10_mz_thisnominalmass + TMT610_PEAK_WIDTH_FOR_DETECT / 2
                #charts.hist(mzs_this_nominalmass, bins=100, title='{}'.format(nominalmass)).show()
    
                n_peaks_near_peak6 = bisect.bisect_left(mzs_this_nominalmass, maxmass_peak6) - bisect.bisect_left(
                    mzs_this_nominalmass, minmass_peak6) + 1
                n_peaks_near_peak10 = bisect.bisect_left(mzs_this_nominalmass, maxmass_peak10) - bisect.bisect_left(
                    mzs_this_nominalmass, minmass_peak10) + 1
                proportion_near_peak6 = float(n_peaks_near_peak6) / n_peaks_this_nominalmass
                proportion_near_peak10 = float(n_peaks_near_peak10) / n_peaks_this_nominalmass
                logger.debug("Near 6: {}. Near 10: {}".format(proportion_near_peak6, proportion_near_peak10))
                if proportion_near_peak10 > TMT610_MIN_TMT10_PROPORTION_FOR_DECISION:
                    logger.debug("Peak {} has TMT10 signal".format(nominalmass))
                    n_peaks_with_enough_TMT10_signal += 1
            else:
                logger.debug("Too few, failing peak.")
        if n_peaks_with_enough_TMT10_signal >= TMT610_MINPEAKS_TMT_PRESENT_FOR_DECISION:
            logger.debug("Detected TMT 10-plex! {} peaks had signal.".format(n_peaks_with_enough_TMT10_signal))
        else:
            logger.debug("Did NOT detect TMT 10-plex. Only {} peaks had signal.".format(n_peaks_with_enough_TMT10_signal))
        return n_peaks_with_enough_TMT10_signal >= TMT610_MINPEAKS_TMT_PRESENT_FOR_DECISION, n_peaks_with_enough_TMT10_signal


class ReporterIonProportionCalculator(RunAttributeDetector):
    """
    Class that accumulates the proportion of MS/MS fragment signal that's accounted for
    by a list of reporter ion mzs for each of multiple types, as well as a set of control m/z bins,
    and then compares each of those types against the controls in order to determine whether the
    label responsible for the reporter ions is present.
    """

    def __init__(self):

        # map from reporter type to the bins representing that type
        self.reporter_ion_type_bins_map = {}
        # map from reporter type to a map from bins for that type to sum of proportions of fragment ion intensities
        # in that bin
        self.reportertype_bin_sum_proportion_map = {}

        # set of all the bins we're considering, for a simple check to see if
        # a peak is in one
        for reporter_ion_type in REPORTER_ION_TYPE_MZS_MAP:
            self.reporter_ion_type_bins_map[reporter_ion_type] = []
            self.reportertype_bin_sum_proportion_map[reporter_ion_type] = {}
            for mz in REPORTER_ION_TYPE_MZS_MAP[reporter_ion_type]:
                binidx = calc_binidx_for_mz_fragment(mz)
                self.reporter_ion_type_bins_map[reporter_ion_type].append(binidx)
                self.reportertype_bin_sum_proportion_map[reporter_ion_type][binidx] = 0.0

        self.tmt10detector = TMT6vs10Detector()
        # keep track of whether we saw any MS3 scans. Those are a reason to check for TMT10
        self.found_ms3_scans = False
        logger.debug("Reporter ion type count (including control): %d" % len(self.reporter_ion_type_bins_map))

    def process_spectrum(self, spectrum, binned_spectrum):
        """
        Process a single spectrum, checking all fragment mzs against the lists of 
        mzs for each reporter type and incrementing the signal in each bin appropriately
        :param spectrum:
        :return: a RunAttributeResult
        """
        if spectrum.level == 3:
            self.found_ms3_scans = True
        for reporter_type in self.reporter_ion_type_bins_map:
            for mz_bin in self.reportertype_bin_sum_proportion_map[reporter_type]:
                self.reportertype_bin_sum_proportion_map[reporter_type][mz_bin] += binned_spectrum[mz_bin]
        self.tmt10detector.process_spectrum(spectrum, binned_spectrum)

    def summarize(self):
        """
        Calculate the average proportion of signal coming from reporter ions across
        all spectra
        :return: 
        """
        # summarize the control bins. These sums are used in the t-statistic calculation
        control_bin_sums = self.reportertype_bin_sum_proportion_map['control'].values()
        # control bin mean and sd are only calculated for debug output. Not used in significance determination
        control_bin_mean = np.mean(control_bin_sums)
        control_bin_sd = np.std(control_bin_sums)
        logger.debug("Reporter ion control bin sums:")
        for control_bin in sorted(self.reportertype_bin_sum_proportion_map['control']):
            # adding 1 to bin number to convert from zero-based index
            logger.debug("  %d: %.02f" % (control_bin + 1, self.reportertype_bin_sum_proportion_map['control'][control_bin]))
        logger.debug("Reporter ion control bin mean: %.02f" % control_bin_mean)

        # check each reporter ion type, determine which are significantly elevated
        significant_reporter_types = set()
        reportertype_tstatistic_map = {}
        for reporter_type in self.reportertype_bin_sum_proportion_map:
            if reporter_type == 'control':
                continue
            # reporter bin sums are used in t-statistic calculation
            reporter_bin_sums = self.reportertype_bin_sum_proportion_map[reporter_type].values()
            # reporter bin mean is only for debug output
            reporter_bin_mean = np.mean(reporter_bin_sums)
            logger.debug("%s, individual ions:" % reporter_type)
            ion_zscores = []
            for reporter_bin in sorted(self.reportertype_bin_sum_proportion_map[reporter_type]):
                bin_sum = self.reportertype_bin_sum_proportion_map[reporter_type][reporter_bin]
                # z-score is only calculated for debug output purposes. Not used in significance determination
                zscore = (bin_sum - control_bin_mean) / control_bin_sd
                # adding 1 to bin number to convert from zero-based index
                logger.debug("    %s, mz=%d: ratio=%.02f, zscore=%.02f" % (reporter_type, reporter_bin + 1,
                                                                     bin_sum / control_bin_mean, zscore))
                ion_zscores.append(zscore)
            logger.debug("%s, ion zscores: %s" % (reporter_type, "\t".join([str(x) for x in ion_zscores])))
            logger.debug("%s bin mean: %.02f" % (reporter_type, reporter_bin_mean))
            # calculate t-statistic of the reporter bin intensity sums vs. the control bin intensity sums
            t_statistic = ttest_ind(reporter_bin_sums, control_bin_sums, equal_var=False)[0]
            reportertype_tstatistic_map[reporter_type] = t_statistic
            # ratio is only calculated for debug output
            ratio = reporter_bin_mean / control_bin_mean
            logger.debug("%s, overall: reporter/control mean ratio: %.04f. t-statistic: %.04f" %
                         (reporter_type, ratio, t_statistic))

            # check the t-statistic against the appropriate threshold and conditionally declare significance
            if t_statistic > REPORTER_ION_TSTAT_THRESHOLDS_MAP[reporter_type]:
                significant_reporter_types.add(reporter_type)

        # Now, all the sets of reporter ions are analyzed and tested. Create the properly-formatted
        # result that summarizes everything.
        result = util.RunAttributeResult()
        
        # handle iTRAQ
        if "iTRAQ_8plex" in significant_reporter_types:
            logger.info("iTRAQ: 8-plex reporter ions detected")
            result.search_modifications.append(util.Modification("K", SEARCH_MOD_MASS_ITRAQ_8PLEX, True))
            result.search_modifications.append(util.Modification(util.MOD_TYPE_KEY_NTERM, SEARCH_MOD_MASS_ITRAQ_8PLEX, True))
            if "iTRAQ_4plex" not in significant_reporter_types:
                logger.warn("    No iTRAQ 4-plex reporters detected, only 8-plex.")
            itraq8_is_present = True
            itraq4_is_present = False
        elif "iTRAQ_4plex" in significant_reporter_types:
            logger.info("iTRAQ: 4-plex reporter ions detected")
            # 8plex mass same as 4plex, more or less
            result.search_modifications.append(util.Modification("K", SEARCH_MOD_MASS_ITRAQ_4PLEX, True))
            result.search_modifications.append(util.Modification(util.MOD_TYPE_KEY_NTERM, SEARCH_MOD_MASS_ITRAQ_4PLEX, True))
            itraq8_is_present = False
            itraq4_is_present = True
        else:
            logger.info("iTRAQ: no reporter ions detected")
            itraq8_is_present = False
            itraq4_is_present = False

        # handle TMT
        
        # first check for TMT10, if checking is justified
        n_tmt10_peaks_detected = 0
        logger.debug("Found MS3 scans? {}".format(self.found_ms3_scans))
        if "TMT_6plex" in significant_reporter_types or self.found_ms3_scans:
            # Either we have TMT6/10, or there are MS3 scans. Either way, we might have TMT10.
            # Let's ask tmt6vs10detector
            tmt10_is_present, n_tmt10_peaks_detected = self.tmt10detector.summarize()
            if tmt10_is_present:
                logger.info("TMT10 is present, {} TMT10 peaks detected.".format(n_tmt10_peaks_detected))
                # declare TMT6 and TMT2 to be absent
                significant_reporter_types.remove("TMT_6plex")
                significant_reporter_types.add("TMT_10plex")
            else:
                logger.info("TMT10 is not present, {} TMT10 peaks detected.".format(n_tmt10_peaks_detected))
                
        if "TMT_6plex" in significant_reporter_types:
            logger.info("TMT: 6-plex reporter ions detected")
            result.search_modifications.append(util.Modification("K", SEARCH_MOD_MASS_TMT_6PLEX, True))
            result.search_modifications.append(util.Modification(util.MOD_TYPE_KEY_NTERM, SEARCH_MOD_MASS_TMT_6PLEX, True))
            if "TMT_2plex" not in significant_reporter_types:
                logger.warn("    No TMT 2-plex reporters detected, only 6-plex")
            tmt2_is_present = False
            tmt6_is_present = True
            tmt10_is_present = False
        elif "TMT_10plex" in significant_reporter_types:
            result.search_modifications.append(util.Modification("K", SEARCH_MOD_MASS_TMT_6PLEX, True))
            result.search_modifications.append(util.Modification(util.MOD_TYPE_KEY_NTERM, SEARCH_MOD_MASS_TMT_6PLEX, True))
            if "TMT_2plex" not in significant_reporter_types:
                logger.warn("    No TMT 2-plex reporters detected, only 10-plex")
            tmt2_is_present = False
            tmt6_is_present = False
            tmt10_is_present = True
        elif "TMT_2plex" in significant_reporter_types:
            logger.info("TMT: 2-plex reporter ions detected")
            result.search_modifications.append(util.Modification("K", SEARCH_MOD_MASS_TMT_2PLEX, True))
            result.search_modifications.append(util.Modification(util.MOD_TYPE_KEY_NTERM, SEARCH_MOD_MASS_TMT_2PLEX, True))
            tmt2_is_present = True
            tmt6_is_present = False
            tmt10_is_present = False
        else:
            logger.info("TMT: no reporter ions detected")
            tmt6_is_present = False
            tmt2_is_present = False
            tmt10_is_present = False



        # declare label to be present or not, and report the appropriate statistic.
        result.name_value_pairs['iTRAQ_8plex_present'] = 'T' if itraq8_is_present else 'F'
        result.name_value_pairs['iTRAQ_8plex_statistic'] = str(reportertype_tstatistic_map['iTRAQ_8plex'])
        result.name_value_pairs['iTRAQ_4plex_present'] = 'T' if itraq4_is_present else 'F'
        result.name_value_pairs['iTRAQ_4plex_statistic'] = str(reportertype_tstatistic_map['iTRAQ_4plex'])
        result.name_value_pairs['TMT_6plex_present'] = 'T' if tmt6_is_present else 'F'
        result.name_value_pairs['TMT_6plex_statistic'] = str(reportertype_tstatistic_map['TMT_6plex'])
        result.name_value_pairs['TMT_10plex_present'] = 'T' if tmt10_is_present else 'F'
        result.name_value_pairs['TMT_10plex_statistic'] = str(n_tmt10_peaks_detected)
        result.name_value_pairs['TMT_2plex_present'] = 'T' if tmt2_is_present else 'F'
        result.name_value_pairs['TMT_2plex_statistic'] = str(reportertype_tstatistic_map['TMT_2plex'])

        return result


class SILACDetector(RunAttributeDetector):
    """
    Detect the presence of SILAC labeling by calculating the number of pairs of (binned) precursor ions that are
    separated by the various SILAC label separations, and comparing the count for each separation against the counts
    for a set of control separations
    """
    # control bin distances to use for comparison. Param-Medic assumes there won't be any
    # excessive pairing of precursors at these mass distances
    CONTROL_BIN_DISTANCES = [11, 14, 15, 21, 23, 27]
    # maximum separation in scans over which to count a pair as present
    MAX_SCAN_SEPARATION = 50

    def __init__(self):
        # a list of all the scan numbers. We determine whether a scan pair falls within MAX_SCAN_SEPARATION
        # by checking the actual scan number, not the number of MS/MS scans in between them
        self.scan_numbers = []
        # a list of the bins representing each scan's precursor mass
        self.precursor_mass_bins = []

    def process_spectrum(self, spectrum, binned_spectrum):
        """
        Just calculate the bin that holds this precursor mass and record the bin index
        :return: 
        """
        if spectrum.charge < 1:
            # can't use spectra of unknown charge
            return
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
        # construct the full set of all separations to summarize, both control and SILAC-label
        separations_to_evaluate = set(SILAC_MOD_BIN_DISTANCES + SILACDetector.CONTROL_BIN_DISTANCES)
        # paranoia
        if len(separations_to_evaluate) < len(SILAC_MOD_BIN_DISTANCES) + len(SILACDetector.CONTROL_BIN_DISTANCES):
            logger.warn("A specified separation is also a control separation! Specified: %s" % str(SILAC_MOD_BIN_DISTANCES))

        # initialize a map from separation distances to counts of pairs with that separation
        counts_with_separations = {}
        for separation in separations_to_evaluate:
            counts_with_separations[separation] = 0

        # keep track of the scan window defined by the minimum and maximum scan index to consider
        minidx = 0
        maxidx = 0
        for i in xrange(0, len(self.scan_numbers)):
            # determine the minimum and maximum scan number currently in range
            scan_number = self.scan_numbers[i]
            min_scan_number = scan_number - SILACDetector.MAX_SCAN_SEPARATION
            max_scan_number = scan_number + SILACDetector.MAX_SCAN_SEPARATION
            while self.scan_numbers[minidx] < min_scan_number:
                minidx += 1
            while self.scan_numbers[maxidx] < max_scan_number and maxidx < len(self.scan_numbers) - 1:
                maxidx += 1

            # within the scan window, increment the separations that we care about with any that involve this scan
            for j in xrange(minidx, maxidx):
                separation = abs(self.precursor_mass_bins[i] - self.precursor_mass_bins[j])
                if separation in separations_to_evaluate:
                    counts_with_separations[separation] += 1

        # summarize the control separations
        mean_control_count = (float(sum([counts_with_separations[separation] for separation in
                                         SILACDetector.CONTROL_BIN_DISTANCES])) /
                              len(SILACDetector.CONTROL_BIN_DISTANCES))
        logger.debug("SILAC: Control separation counts:")
        for separation in SILACDetector.CONTROL_BIN_DISTANCES:
            logger.debug("  %d: %d" % (separation, counts_with_separations[separation]))
        logger.debug("SILAC: Mean control separation count: %.05f" % mean_control_count)
        if mean_control_count > 0:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("SILAC: Counts for each separation:")
                for separation in SILAC_MOD_BIN_DISTANCES:
                    proportion_to_control = float(counts_with_separations[separation]) / mean_control_count
                    logger.debug("  %d: %d (proportion=%.05f)" % (separation, counts_with_separations[separation], proportion_to_control))
        else:
            logger.warn("SILAC: No counts for any control separation pairs! Cannot estimate prevalence of SILAC separations.")
            # make a dummy result with no significant inferences
            result = util.RunAttributeResult()
            for separation in SILAC_MOD_BIN_DISTANCES:
                result.name_value_pairs['SILAC_%dDa_present' % separation] = 'ERROR'
                result.name_value_pairs['SILAC_%dDa_statistic' % separation] = 'ERROR'
            return result
        control_sd = np.std([counts_with_separations[separation] for separation in SILACDetector.CONTROL_BIN_DISTANCES])

        # determine any separations with a significantly elevated number of representatives
        result = util.RunAttributeResult()
        significant_separations = []
        logger.debug("SILAC: Ratios of mass separations to control separations:")

        for separation in SILAC_MOD_BIN_DISTANCES:
            # z-score is checked against a cutoff to determine significance
            zscore_to_control = float(counts_with_separations[separation] - mean_control_count) / control_sd
            if zscore_to_control > SILAC_ZSCORE_CUTOFF:
                significant_separations.append(separation)
                logger.info("SILAC: %dDa separation detected." % separation)
                # paranoia
                if separation not in SILAC_MOD_K_EXACTMASS_MAP and separation not in SILAC_MOD_R_EXACTMASS_MAP:
                    raise ValueError('Unknown SILAC separation %d' % separation)

                # figure out the exact appropriate mass for search.
                if separation in SILAC_MOD_K_EXACTMASS_MAP:
                    result.search_modifications.append(util.Modification("K", SILAC_MOD_K_EXACTMASS_MAP[separation], True))
                if separation in SILAC_MOD_R_EXACTMASS_MAP:
                    result.search_modifications.append(util.Modification("R", SILAC_MOD_R_EXACTMASS_MAP[separation], True))
                result.name_value_pairs['SILAC_%dDa_present' % separation] = 'T'
            else:
                result.name_value_pairs['SILAC_%dDa_present' % separation] = 'F'
            result.name_value_pairs['SILAC_%dDa_statistic' % separation] = str(zscore_to_control)
            # show some details for debug output
            proportion_to_control = float(counts_with_separations[separation]) / mean_control_count
            logger.debug("SILAC:     %dDa: %.05f (z=%.03f)" % (separation, proportion_to_control,
                                                               zscore_to_control))
        if not significant_separations:
            logger.info("SILAC: no labeling detected")
        return result

