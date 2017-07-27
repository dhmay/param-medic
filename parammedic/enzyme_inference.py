#!/usr/bin/env python
"""
Code to infer the digestion enzyme used in an experiment.

"""

import bisect
import logging

import numpy as np
from parammedic import util
from scipy.stats import ttest_ind
import binning

from util import RunAttributeDetector, AA_UNMOD_MASSES
from parammedic.binning import calc_binidx_for_mz_fragment

__author__ = "Damon May"
__copyright__ = "Copyright (c) 2016 Damon May"
__license__ = ""
__version__ = ""


logger = logging.getLogger(__name__)


MOD_OXIDATION_MASSDIFF = 15.994915
CTERM_MASS_UNMOD = 2 * util.HYDROGEN_MASS + MOD_OXIDATION_MASSDIFF

TRYPSIN_AAS = {'K', 'R'}
PEPSIN_AAS = {'F', 'L'}
CHYMOTRYPSIN_NOPEPSIN_AAS = {'W', 'Y'}
CHYMOTRYPSIN_ALL_AAS = PEPSIN_AAS.union(CHYMOTRYPSIN_NOPEPSIN_AAS)
THERMOLYSIN_AAS = {'I', 'L', 'V', 'A', 'M', 'F'}
# Full list of possible control AAs.
# Do not use T (120Th) because it has very high signal in a bunch of trypsinized human runs.
ALL_CONTROL_AAS = {"A", "C", "D", "E", "G", "H", "K", "I", "M", "N", "P", "R", "S", "V"}  # note: no T

# minimum proportion of MS/MS signal that must come from fragments 1/3 the m/z of the precursor ion
# in order to use y1 ions for enzyme determination.
# If proportion is less than threshold, b1 ions will be used
PROPORTION_ONETHIRDPRECURSOR_Y1_THRESHOLD = 0.4

# If we don't get this many spectra, warn that we can't be confident in our assessment
MIN_SPECTRA_FOR_CONFIDENCE = 2000

#MIN_CONTROL_MZ = 110.5 * util.HYDROGEN_MASS
#MIN_CONTROL_BINIDX = util.calc_binidx_for_mz_fragment(MIN_CONTROL_MZ)
#MAX_CONTROL_MZ = 206.5 * util.HYDROGEN_MASS
#MAX_CONTROL_BINIDX = util.calc_binidx_for_mz_fragment(MAX_CONTROL_MZ)
#N_CONTROL_BINS = MAX_CONTROL_BINIDX - MIN_CONTROL_BINIDX + 1

# thresholds for considering various statistics significant
MIN_TRYPSIN_ZSCORE_THRESHOLD = 2.0
# Bias is toward Trypsin. If we don't find evidence for anything else, and minimum K or R z-score is above
# this threshold, then report probably trypsin
MIN_TRYPSIN_ZSCORE_LIKELY_THRESHOLD = 1.5 
MIN_ARGC_LYSC_ZSCORE_THRESHOLD = 5.0
MIN_PEPSIN_TSTAT_THRESHOLD = 5.0
MIN_CHYMOTRYPSIN_NOPEPSIN_TSTAT_THRESHOLD = 5.0

ENZYME_STR_LIKELY_TRYPSIN = "Likely Trypsin"
ENZYME_STR_TRYPSIN = "Trypsin"
ENZYME_STR_LYSC = "Lys-C"
ENZYME_STR_ARGC = "Arg-C"
ENZYME_STR_CHYMOTRYPSIN = "Chymotrypsin"
ENZYME_STR_PEPSIN = "Pepsin"
ENZYME_STR_UNKNOWN = "Unknown"


class EnzymeDetector(RunAttributeDetector):
    """
    Accumulates the proportion of MS/MS fragment signal that's accounted for
    by fragments representing a loss of DELTA_MASS_PHOSPHO_LOSS Da from the precursor mass
    
    todo: if the sample is TMT-labeled, account for TMT mass on Lys and N-term. Similarly for other modifications.
    """
    def __init__(self, sample_modifications):
        """
        :param sample_modifications: a list of modifications detected in the sample
        """
        self.n_total_spectra = 0
        # sum of the proportions of signal observed below 1/3 of the precursor m/z
        self.sum_proportion_ms2_signal_onethird_precursor = 0.0
        # these two dictionaries map each amino acid to the proportion of total MS/MS signal
        # represented by a charge-1 ion representing that amino acid in the given ion type.
        self.sum_proportions_y1_dict = {}
        self.sum_proportions_bnminus1_dict = {}

        # map from AA to b1 charge 1 fragment mz for that AA. properly can be static, not sure most Pythonic
        # way to do that
        self.aa_y1_charge1_mz_map = {}
        self.aa_y1_charge1_bin_map = {}

        self.cterm_mass = CTERM_MASS_UNMOD
        self.aa_masses = dict(AA_UNMOD_MASSES)
        
        mod_locations = set()
        for modification in sample_modifications:
            print("Accounting for modification: %s" % modification)
            if modification.location in mod_locations:
                print("WARNING! Multiple modifications on %s" % modification.location)
            if modification.location == util.MOD_TYPE_KEY_CTERM:
                # todo: what if it's variable?
                self.cterm_mass += modification.mass_diff
            elif modification.location == util.MOD_TYPE_KEY_NTERM:
                pass
            else:
                # todo: what if it's variable?
                self.aa_masses[modification.location] += modification.mass_diff

        for aa in self.aa_masses:
            # y1 m/z is the c-terminus mass plus the AA mass plus a hydrogen
            self.aa_y1_charge1_mz_map[aa] = self.aa_masses[aa] + self.cterm_mass + util.HYDROGEN_MASS
            self.aa_y1_charge1_bin_map[aa] = calc_binidx_for_mz_fragment(self.aa_y1_charge1_mz_map[aa])
            self.sum_proportions_y1_dict[aa] = 0.0
            self.sum_proportions_bnminus1_dict[aa] = 0.0
        logger.debug("Y1 ion mzs:")
        logger.debug(self.aa_y1_charge1_mz_map)

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
        :param binned_spectrum
        :return: 
        """
        self.n_total_spectra += 1

        # total signal in this spectrum, for calculating proportions
        # todo: calculate this once per spectrum globally, not in each calculator
        signal_total = sum(spectrum.intensity_array)

        # track how much signal is contained below 1/3 of the precursor m/z.
        # I'll use this to decide which ions to use to determine enzyme
        idx_onethird_precursor_mz = bisect.bisect_right(spectrum.mz_array, spectrum.precursor_mz)
        proportion_onethird_precursor = sum(spectrum.intensity_array[0:idx_onethird_precursor_mz + 1])
        self.sum_proportion_ms2_signal_onethird_precursor += (proportion_onethird_precursor / signal_total)

#        for i in xrange(0, len(binned_spectrum)):
#            if binned_spectrum[i] > MIN_PROPORTION_SIGNAL_FOR_FRAGMENT_COUNT:
#                self.spectrumcounts_withsignal_perbin[i] += 1

        curspectrum_bnminus1_aa_binidx_map = {}
        for aa in self.aa_masses:
            #b(n-1) ion for this amino acid is the precursor mz - the y1 ion, plus the mass of H
            # there are at least three ways to go about calculating this.
            #bnminus1_mz_this_aa = spectrum.precursor_mz - self.aa_y1_charge1_mz_map[aa] + util.HYDROGEN_MASS
            #bnminus1_mz_this_aa = spectrum.precursor_mz - self.aa_masses[aa] - self.cterm_mass
            bnminus1_mz_this_aa = (spectrum.precursor_mz - util.HYDROGEN_MASS) * spectrum.charge - self.aa_masses[aa] - self.cterm_mass + util.HYDROGEN_MASS
            curspectrum_bnminus1_aa_binidx_map[aa] = calc_binidx_for_mz_fragment(bnminus1_mz_this_aa)
        for aa in self.aa_masses:
            self.sum_proportions_y1_dict[aa] += binned_spectrum[self.aa_y1_charge1_bin_map[aa]]
            if curspectrum_bnminus1_aa_binidx_map[aa] >= len(binned_spectrum):
                pass
                # logger.debug("Ignoring b(n-1) ion because precursor is too high! precursor=%f, charge=%d" % (spectrum.precursor_mz, spectrum.charge))
            else:
                self.sum_proportions_bnminus1_dict[aa] += binned_spectrum[curspectrum_bnminus1_aa_binidx_map[aa]]

    def summarize(self):
        """
        Calculate the average proportion of signal coming from reporter ions across
        all spectra
        :return: 
        """
        proportion_lessthan_onethirdprecursor = self.sum_proportion_ms2_signal_onethird_precursor / self.n_total_spectra
        logger.debug("Signal proportion under 1/3 precursor mz: %f" % proportion_lessthan_onethirdprecursor)
        if proportion_lessthan_onethirdprecursor > PROPORTION_ONETHIRDPRECURSOR_Y1_THRESHOLD:
            aa_proportion_sums_for_test_dict = self.sum_proportions_y1_dict
            logger.debug("Using y1 ions for enzyme determination.")
        else:
            aa_proportion_sums_for_test_dict = self.sum_proportions_bnminus1_dict
            logger.debug("Using b(n-1) ions for enzyme determination.")


        # Trypsin, ArgC and LysC.
        # Trypsin is just ArgC + LysC. So test R and K individually. If the lower of the two is significant,
        # Trypsin. If not, then if R or K significant, then that one.
        control_aas = ALL_CONTROL_AAS.difference(TRYPSIN_AAS)
        control_bin_sums = [aa_proportion_sums_for_test_dict[aa] for aa in control_aas]
        control_mean = np.mean(control_bin_sums)
        control_sd = np.std(control_bin_sums)

        aa_zscores = {}
        for aa in aa_proportion_sums_for_test_dict:
            aa_zscores[aa] = (aa_proportion_sums_for_test_dict[aa] - control_mean) / control_sd
            logger.debug("%s: %.4f (z = %.4f)" %
                         (aa, aa_proportion_sums_for_test_dict[aa] / self.n_total_spectra, aa_zscores[aa]))

        logger.debug("  ArgC z-score: %f" % aa_zscores['R'])
        logger.debug("  LysC z-score: %f" % aa_zscores['K'])
        trypsin_min_zscore = min(aa_zscores['R'], aa_zscores['K'])
        logger.debug("  min(ArgC, LysC): %f" % trypsin_min_zscore)

        if trypsin_min_zscore > MIN_TRYPSIN_ZSCORE_THRESHOLD:
            return "trypsin"

        # OK, we don't think it's trypsin. That's the main finding. But can we get more specific?
        if aa_zscores['K'] > MIN_ARGC_LYSC_ZSCORE_THRESHOLD:
            return ENZYME_STR_LYSC
        if aa_zscores['R'] > MIN_ARGC_LYSC_ZSCORE_THRESHOLD:
            return ENZYME_STR_ARGC

        # Try Pepsin. Don't test against Pepsin or Chymotrypsin controls
        control_aas = ALL_CONTROL_AAS.difference(CHYMOTRYPSIN_ALL_AAS)
        pepsin_bin_sums = [aa_proportion_sums_for_test_dict[aa] for aa in PEPSIN_AAS]
        control_bin_sums = [aa_proportion_sums_for_test_dict[aa] for aa in control_aas]
        pepsin_t_statistic = ttest_ind(pepsin_bin_sums, control_bin_sums)[0]
        logger.debug("  Pepsin t-statistic: %f" % pepsin_t_statistic)
        chymo_bin_sums = [aa_proportion_sums_for_test_dict[aa] for aa in CHYMOTRYPSIN_NOPEPSIN_AAS]
        chymotrypsin_t_statistic = ttest_ind(chymo_bin_sums, control_bin_sums)[0]
        logger.debug("  Chymotrypsin (but not Pepsin) t-statistic: %f" % chymotrypsin_t_statistic)
        if pepsin_t_statistic > MIN_PEPSIN_TSTAT_THRESHOLD:
            if chymotrypsin_t_statistic > MIN_CHYMOTRYPSIN_NOPEPSIN_TSTAT_THRESHOLD:
                return ENZYME_STR_CHYMOTRYPSIN
            else:
                return ENZYME_STR_PEPSIN

        if trypsin_min_zscore > MIN_TRYPSIN_ZSCORE_LIKELY_THRESHOLD:
            print("Trypsin test passes bare minimum threshold, and no other enzyme detected.")
            return ENZYME_STR_LIKELY_TRYPSIN
        if self.n_total_spectra < MIN_SPECTRA_FOR_CONFIDENCE:
            print("WARNING: only %d spectra were analyzed. Enzyme determination is suspect." % self.n_total_spectra)




        return ENZYME_STR_UNKNOWN




        return result

