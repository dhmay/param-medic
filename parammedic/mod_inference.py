#!/usr/bin/env python
"""
Code to infer the presence or absence of a defined set of modifications in 
a given run.

todo: improve efficiency by sharing calculations like total ms signal between 
detectors
"""

import logging
from parammedic import util

import numpy as np
from scipy.stats import ttest_ind

from parammedic.util import RunAttributeDetector, calc_mplush_from_mz_charge
from parammedic.binning import calc_binidx_for_mass_precursor, calc_binidx_for_mz_fragment

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
SILAC_MOD_BIN_DISTANCES = [4, 6, 8]
# z-score cutoff above which we consider a particular SILAC separation to be present
SILAC_ZSCORE_CUTOFF = 6.0


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
# except using this from Jimmy, instead. Average of masses for 114 (144.105918), 115 (144.09599), and 116/117 (144.102063)
SEARCH_MOD_MASS_ITRAQ_4PLEX = 144.10253
# http://www.unimod.org/modifications_view.php?editid1=730
#SEARCH_MOD_MASS_ITRAQ_8PLEX = 304.205360
# using this instead, from Jimmy. Average of masses for 115/118/119/121 (304.199040) and 113/114/116/117 (304.205360)
SEARCH_MOD_MASS_ITRAQ_8PLEX = 304.2022
# http://www.unimod.org/modifications_view.php?editid1=738
SEARCH_MOD_MASS_TMT_2PLEX = 225.155833
# http://www.unimod.org/modifications_view.php?editid1=737
SEARCH_MOD_MASS_TMT_6PLEX = 229.162932
# http://www.unimod.org/modifications_view.php?editid1=481
SEARCH_MOD_MASS_SILAC_4DA = 4.0246
# http://www.unimod.org/modifications_view.php?editid1=188
SEARCH_MOD_MASS_SILAC_6DA = 6.020129
# http://www.unimod.org/modifications_view.php?editid1=259
SEARCH_MOD_MASS_SILAC_8DA = 8.014199
# http://www.unimod.org/modifications_view.php?editid1=21
SEARCH_MOD_MASS_PHOSPHO = 79.966331



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

    def process_spectrum(self, spectrum, binned_spectrum):
        """
        Handle a spectrum. Calculate precursor mass from m/z and charge, then calculate
        mass of phospho loss and convert back to m/z. Look for ion representing
        that loss, in same charge as precursor. accumulate proportion of total signal 
        contained in those ions. Do the same thing for several control ions
        :param spectrum: 
        :param binned_spectrum: 
        :return: 
        """
        self.n_total_spectra += 1
        # phospho loss peak is DELTA_MASS_PHOSPHO_LOSS lighter than precursor, and in the same charge state
        phospho_loss_mz = spectrum.precursor_mz - (DELTA_MASS_PHOSPHO_LOSS / spectrum.charge)
        phospho_loss_bin = calc_binidx_for_mz_fragment(phospho_loss_mz)
        #print("%d. %d. %d. %d" % (spectrum.scan_number, spectrum.charge, calc_binidx_for_mz_fragment(spectrum.precursor_mz), phospho_loss_bin))
        # increment the control bins for this spectrum
        for offset in PHOSPHO_CONTROLPEAK_PHOSPHOPEAK_OFFSETS:
            self.sums_proportions_per_controlpeak[offset] += binned_spectrum[phospho_loss_bin + offset]
        self.sum_proportions_in_phosopholoss += binned_spectrum[phospho_loss_bin]

    def summarize(self):
        """
        Calculate the average proportion of signal coming from reporter ions across
        all spectra
        :return: 
        """

        control_mean = np.mean(self.sums_proportions_per_controlpeak.values())
        control_sd = np.std(self.sums_proportions_per_controlpeak.values())
        logger.debug("Phospho control peaks (mean=%.03f):" % control_mean)
        for control_peak in self.sums_proportions_per_controlpeak:
            logger.debug("    %d: %.04f" % (control_peak, self.sums_proportions_per_controlpeak[control_peak]))
        proportion_to_control = self.sum_proportions_in_phosopholoss / control_mean
        zscore_to_control = (self.sum_proportions_in_phosopholoss - control_mean) / control_sd
        logger.debug("Phospho-loss peak: %.03f" % self.sum_proportions_in_phosopholoss)
        logger.debug("Phospho: ratio phospho-loss to control peaks: %.05f (z=%.03f)" % (proportion_to_control, zscore_to_control))
        search_modifications = []
        if zscore_to_control > PHOSPHO_ZSCORE_CUTOFF:
            print("Phosphorylation: detected")
            search_modifications.append(util.Modification(util.MOD_TYPE_KEY_CTERM, SEARCH_MOD_MASS_PHOSPHO, True))
            phospho_is_present = True
        else:
            print("Phosphorylation: not detected")
            phospho_is_present = False

        result = util.RunAttributeResult()
        result.name_value_pairs['phospho_present'] = 'T' if phospho_is_present else 'F'
        result.name_value_pairs['phospho_statistic'] = str(zscore_to_control)
        result.search_modifications = search_modifications
        return result


class ReporterIonProportionCalculator(RunAttributeDetector):
    """
    Class that accumulates the proportion of MS/MS fragment signal that's accounted for
    by a list of reporter ion mzs for each of multiple types.
    """

    # minimum number of ions within a given type that must have significant z-scores
    # in order to consider that ion type present.
    # If a nonzero smaller number are present, an info message will be displayed.
    N_SIGNIF_IONS_IN_TYPE_REQUIRED = 2

    def __init__(self):

        self.n_total_spectra = 0
        # map from reporter type to the bins representing that type
        self.reporter_ion_type_bins_map = {}
        # map from reporter type to sum of proportions of fragment ion intensities in bins for that type
        self.reportertype_bin_sum_proportion_map = {}

        # set of all the bins we're considering, for a simple check to see if
        # a peak is in one
        self.all_bins_considered = set()
        for reporter_ion_type in REPORTER_ION_TYPE_MZS_MAP:
            self.reporter_ion_type_bins_map[reporter_ion_type] = []
            self.reportertype_bin_sum_proportion_map[reporter_ion_type] = {}
            for mz in REPORTER_ION_TYPE_MZS_MAP[reporter_ion_type]:
                binidx = calc_binidx_for_mz_fragment(mz)
                self.all_bins_considered.add(binidx)
                self.reporter_ion_type_bins_map[reporter_ion_type].append(binidx)
                self.reportertype_bin_sum_proportion_map[reporter_ion_type][binidx] = 0.0

        logger.debug("Reporter ion type count (including control): %d" % len(self.reporter_ion_type_bins_map))

    def next_file(self):
        """
        Register that a new file is being processed
        :return: 
        """
        return
        
    def process_spectrum(self, spectrum, binned_spectrum):
        """
        Process a single spectrum, checking all fragment mzs against the lists of 
        mzs for each reporter type
        :param spectrum:
        :return:
        """
        # accounting
        self.n_total_spectra += 1
        for reporter_type in self.reporter_ion_type_bins_map:
            for mz_bin in self.reportertype_bin_sum_proportion_map[reporter_type]:
                self.reportertype_bin_sum_proportion_map[reporter_type][mz_bin] += binned_spectrum[mz_bin]

    def summarize(self):
        """
        Calculate the average proportion of signal coming from reporter ions across
        all spectra
        :return: 
        """
        control_bin_sums = self.reportertype_bin_sum_proportion_map['control'].values()
        control_bin_mean = np.mean(control_bin_sums)
        control_bin_sd = np.std(control_bin_sums)
        logger.debug("Reporter ion control bin sums:")
        for control_bin in sorted(self.reportertype_bin_sum_proportion_map['control']):
            # adding 1 to bin number to convert from zero-based index
            logger.debug("  %d: %.02f" % (control_bin + 1, self.reportertype_bin_sum_proportion_map['control'][control_bin]))
        logger.debug("Reporter ion control bin mean: %.02f" % control_bin_mean)
        significant_reporter_types = set()
        reportertype_tstatistic_map = {}
        for reporter_type in self.reportertype_bin_sum_proportion_map:
            if reporter_type == 'control':
                continue
            n_signif_ions_this_type = 0
            reporter_bin_sums = self.reportertype_bin_sum_proportion_map[reporter_type].values()
            reporter_bin_mean = np.mean(reporter_bin_sums)
            logger.debug("%s, individual ions:" % reporter_type)
            ion_zscores = []
            for reporter_bin in sorted(self.reportertype_bin_sum_proportion_map[reporter_type]):
                bin_sum = self.reportertype_bin_sum_proportion_map[reporter_type][reporter_bin]
                zscore = (bin_sum - control_bin_mean) / control_bin_sd
                if reporter_type.startswith('iTRAQ'):
                    cutoff_this_ion = ITRAQ_REPORTER_ION_ZSCORE_CUTOFF
                elif reporter_type.startswith('TMT'):
                    cutoff_this_ion = TMT_REPORTER_ION_ZSCORE_CUTOFF
                else:
                    raise ValueError('Unknown reporter ion type %s' % reporter_type)
                if zscore > cutoff_this_ion:
                    n_signif_ions_this_type += 1
                # adding 1 to bin number to convert from zero-based index
                logger.debug("    %s, mz=%d: ratio=%.02f, zscore=%.02f" % (reporter_type, reporter_bin + 1,
                                                                     bin_sum / control_bin_mean, zscore))
                ion_zscores.append(zscore)
            logger.debug("%s, ion zscores: %s" % (reporter_type, "\t".join([str(x) for x in ion_zscores])))
#            if n_signif_ions_this_type >= ReporterIonProportionCalculator.N_SIGNIF_IONS_IN_TYPE_REQUIRED:
#                significant_reporter_types.add(reporter_type)
#            elif n_signif_ions_this_type > 0:
#                logger.info("%d significant ions for reporter type %s. Not enough to declare present." %
#                            (n_signif_ions_this_type, reporter_type))
            logger.debug("%s bin mean: %.02f" % (reporter_type, reporter_bin_mean))
            t_statistic = ttest_ind(reporter_bin_sums, control_bin_sums, equal_var=False)[0]
            reportertype_tstatistic_map[reporter_type] = t_statistic
            ratio = reporter_bin_mean / control_bin_mean
            logger.debug("%s, overall: reporter/control mean ratio: %.04f. t-statistic: %.04f" %
                         (reporter_type, ratio, t_statistic))
            if t_statistic > REPORTER_ION_TSTAT_THRESHOLDS_MAP[reporter_type]:
                significant_reporter_types.add(reporter_type)
                
        # create result
        search_modifications = []
        result = util.RunAttributeResult()
        
        # handle iTRAQ
        if "iTRAQ_8plex" in significant_reporter_types:
            print("iTRAQ: 8-plex reporter ions detected")
            search_modifications.append(util.Modification("K", SEARCH_MOD_MASS_ITRAQ_8PLEX, True))
            search_modifications.append(util.Modification(util.MOD_TYPE_KEY_NTERM, SEARCH_MOD_MASS_ITRAQ_8PLEX, True))
            if "iTRAQ_4plex" not in significant_reporter_types:
                logger.warn("    No iTRAQ 4-plex reporters detected, only 8-plex.")
            itraq8_is_present = True
            itraq4_is_present = False
        elif "iTRAQ_4plex" in significant_reporter_types:
            print("iTRAQ: 4-plex reporter ions detected")
            # 8plex mass same as 4plex, more or less
            search_modifications.append(util.Modification("K", SEARCH_MOD_MASS_ITRAQ_4PLEX, True))
            search_modifications.append(util.Modification(util.MOD_TYPE_KEY_NTERM, SEARCH_MOD_MASS_ITRAQ_4PLEX, True))
            itraq8_is_present = False
            itraq4_is_present = True
        else:
            print("iTRAQ: no reporter ions detected")
            itraq8_is_present = False
            itraq4_is_present = False

        # handle TMT
        if "TMT_6plex" in significant_reporter_types:
            print("TMT: 6-plex reporter ions detected")
            search_modifications.append(util.Modification("K", SEARCH_MOD_MASS_TMT_6PLEX, True))
            search_modifications.append(util.Modification(util.MOD_TYPE_KEY_NTERM, SEARCH_MOD_MASS_TMT_6PLEX, True))
            if "TMT_2plex" not in significant_reporter_types:
                logger.warn("    No TMT 2-plex reporters detected, only 6-plex")
            tmt6_is_present = True
            tmt2_is_present = False
        elif "TMT_2plex" in significant_reporter_types:
            print("TMT: 2-plex reporter ions detected")
            search_modifications.append(util.Modification("K", SEARCH_MOD_MASS_TMT_2PLEX, True))
            search_modifications.append(util.Modification(util.MOD_TYPE_KEY_NTERM, SEARCH_MOD_MASS_TMT_2PLEX, True))
            tmt6_is_present = False
            tmt2_is_present = True
        else:
            print("TMT: no reporter ions detected")
            tmt6_is_present = False
            tmt2_is_present = False
            
        result.name_value_pairs['iTRAQ_8plex_present'] = 'T' if itraq8_is_present else 'F'
        result.name_value_pairs['iTRAQ_8plex_statistic'] = str(reportertype_tstatistic_map['iTRAQ_8plex'])
        result.name_value_pairs['iTRAQ_4plex_present'] = 'T' if itraq4_is_present else 'F'
        result.name_value_pairs['iTRAQ_4plex_statistic'] = str(reportertype_tstatistic_map['iTRAQ_4plex'])
        result.name_value_pairs['TMT_6plex_present'] = 'T' if tmt6_is_present else 'F'
        result.name_value_pairs['TMT_6plex_statistic'] = str(reportertype_tstatistic_map['TMT_6plex'])
        result.name_value_pairs['TMT_2plex_present'] = 'T' if tmt2_is_present else 'F'
        result.name_value_pairs['TMT_2plex_statistic'] = str(reportertype_tstatistic_map['TMT_2plex'])

        result.search_modifications = search_modifications
        return result


class SILACDetector(RunAttributeDetector):
    """
    Calculate the number of pairs of spectra that are separated by 
    a given set of distances, as a proporation of all pairs of spectra.
    """
    # bin distances to use for comparison. Param-Medic assumes there won't be any
    # excessive pairing of precursors at these mass distances
    CONTROL_BIN_DISTANCES = [11, 14, 15, 21, 23, 27]
    MAX_SCAN_SEPARATION = 50

    def __init__(self):
        self.scan_numbers = []
        self.precursor_mass_bins = []
        self.n_total_spectra = 0

    def next_file(self):
        """
        Register that a new file is being processed
        :return: 
        """
        return

    def process_spectrum(self, spectrum, binned_spectrum):
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
        separations_to_evaluate = set(SILAC_MOD_BIN_DISTANCES + SILACDetector.CONTROL_BIN_DISTANCES)
        if len(separations_to_evaluate) < len(SILAC_MOD_BIN_DISTANCES) + len(SILACDetector.CONTROL_BIN_DISTANCES):
            logger.warn("A specified separation is also a control separation! Specified: %s" % str(SILAC_MOD_BIN_DISTANCES))
        for separation in separations_to_evaluate:
            counts_with_separations[separation] = 0
        
        minidx = 0
        maxidx = 0
        for i in xrange(0, len(self.scan_numbers)):
            scan_number = self.scan_numbers[i]
            min_scan_number = scan_number - SILACDetector.MAX_SCAN_SEPARATION
            max_scan_number = scan_number + SILACDetector.MAX_SCAN_SEPARATION
            while self.scan_numbers[minidx] < min_scan_number:
                minidx += 1
            while self.scan_numbers[maxidx] < max_scan_number and maxidx < len(self.scan_numbers) - 1:
                maxidx += 1
            for j in xrange(minidx, maxidx):
                separation = abs(self.precursor_mass_bins[i] - self.precursor_mass_bins[j])
                if separation in separations_to_evaluate:
                    counts_with_separations[separation] += 1
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
            return []

        control_sd = np.std([counts_with_separations[separation] for separation in SILACDetector.CONTROL_BIN_DISTANCES])

        significant_separations = []
        search_modifications = []
        logger.debug("SILAC: Ratios of mass separations to control separations:")

        result = util.RunAttributeResult()

        for separation in SILAC_MOD_BIN_DISTANCES:
            proportion_to_control = float(counts_with_separations[separation]) / mean_control_count
            zscore_to_control = float(counts_with_separations[separation] - mean_control_count) / control_sd
            if zscore_to_control > SILAC_ZSCORE_CUTOFF:
                significant_separations.append(separation)
                print("SILAC: %dDa separation detected." % separation)
                # figure out the exact appropriate mass for search
                if separation == 4:
                    varmod_mass = SEARCH_MOD_MASS_SILAC_4DA
                elif separation == 6:
                    varmod_mass = SEARCH_MOD_MASS_SILAC_6DA
                elif separation == 8:
                    varmod_mass = SEARCH_MOD_MASS_SILAC_8DA
                else:
                    raise ValueError('Unknown SILAC separation %d' % separation)
                result.name_value_pairs['SILAC_%dDa_present' % separation] = 'T'

                search_modifications.append(util.Modification("K", varmod_mass, True))
                search_modifications.append(util.Modification("R", varmod_mass, True))
            else:
                result.name_value_pairs['SILAC_%dDa_present' % separation] = 'F'
            result.name_value_pairs['SILAC_%dDa_statistic' % separation] = str(zscore_to_control)
            logger.debug("SILAC:     %dDa: %.05f (z=%.03f)" % (separation, proportion_to_control,
                                                               zscore_to_control))

        if not significant_separations:
            print("SILAC: no labeling detected")
        else:
            # 6Da separation is not compatible with 4Da and 8Da
            if 6 in significant_separations and len(significant_separations) > 1:
                logger.warn("Detected incompatible SILAC separations: %s" % str(significant_separations))
        result.search_modifications = search_modifications
        return result

