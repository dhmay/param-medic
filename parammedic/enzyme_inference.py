#!/usr/bin/env python
"""
Code to infer the digestion enzyme used in an experiment.

"""

import bisect
import logging

import numpy as np
from parammedic import util
from scipy.stats import ttest_ind

from parammedic.util import RunAttributeDetector, calc_binidx_for_mz_fragment, \
    AA_UNMOD_MASSES

__author__ = "Damon May"
__copyright__ = "Copyright (c) 2016 Damon May"
__license__ = ""
__version__ = ""


logger = logging.getLogger(__name__)


MOD_OXIDATION_MASSDIFF = 15.994915
CTERM_MASS = 2 * util.HYDROGEN_MASS + MOD_OXIDATION_MASSDIFF

TRYPSIN_AAS = ['K', 'R']
CHYMOTRYPSIN_AAS = ['F', 'W', 'Y']
# Control AAs are everything but the Trypsin and Chymotrypsin AAs
# Do not use T (120Th) because it has very high signal in trypsinized human runs.
CONTROL_AAS = ["A", "C", "D", "E", "G", "H", "I", "L", "M", "N", "P", "S", "V"] # note: no T


class EnzymeDetector(RunAttributeDetector):
    """
    Accumulates the proportion of MS/MS fragment signal that's accounted for
    by fragments representing a loss of DELTA_MASS_PHOSPHO_LOSS Da from the precursor mass
    """
    def __init__(self):
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

        for aa in AA_UNMOD_MASSES:
            # y1 m/z is the c-terminus mass plus the AA mass plus a hydrogen
            self.aa_y1_charge1_mz_map[aa] = util.AA_UNMOD_MASSES[aa] + CTERM_MASS + util.HYDROGEN_MASS
            self.aa_y1_charge1_bin_map[aa] = calc_binidx_for_mz_fragment(self.aa_y1_charge1_mz_map[aa])
            self.sum_proportions_y1_dict[aa] = 0.0
            self.sum_proportions_bnminus1_dict[aa] = 0.0
        logger.debug("Y1 ion mzs:")
        logger.debug(self.aa_y1_charge1_mz_map)
#        print("Y1 mzs: ")
#        for aa in AA_UNMOD_MASSES:
#            print("%s: %f   %d" % (aa, self.aa_y1_charge1_mz_map[aa], self.aa_y1_charge1_bin_map[aa]))
#        print(CTERM_MASS)
#        quit()

    def next_file(self):
        """
        Register that a new file is being processed
        :return: 
        """
        return

    def process_spectrum(self, spectrum):
        """
        Handle a spectrum. Calculate precursor mass from m/z and charge, then calculate
        mass of phospho loss and convert back to m/z. Look for ion representing
        that loss, in same charge as precursor. accumulate proportion of total signal 
        contained in those ions. Do the same thing for several control ions
        :param spectrum: 
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

        curspectrum_bnminus1_aa_binidx_map = {}
        for aa in AA_UNMOD_MASSES:
            #b(n-1) ion for this amino acid is the precursor mz - the y1 ion, plus the mass of H
            #bnminus1_mz_this_aa = spectrum.precursor_mz - self.aa_y1_charge1_mz_map[aa] + util.HYDROGEN_MASS
            #bnminus1_mz_this_aa = spectrum.precursor_mz - AA_UNMOD_MASSES[aa] - CTERM_MASS
            bnminus1_mz_this_aa = (spectrum.precursor_mz - util.HYDROGEN_MASS) * spectrum.charge - AA_UNMOD_MASSES[aa] - CTERM_MASS + util.HYDROGEN_MASS
            curspectrum_bnminus1_aa_binidx_map[aa] = calc_binidx_for_mz_fragment(bnminus1_mz_this_aa)
        # loop over all the peaks, incrementing proportion sums in all relevant bins
        for i in xrange(0, len(spectrum.mz_array)):
            frag_binidx = calc_binidx_for_mz_fragment(spectrum.mz_array[i])
            for aa in AA_UNMOD_MASSES:
                if frag_binidx == self.aa_y1_charge1_bin_map[aa]:
                    self.sum_proportions_y1_dict[aa] += (spectrum.intensity_array[i] / signal_total)
                if frag_binidx == curspectrum_bnminus1_aa_binidx_map[aa]:
                    self.sum_proportions_bnminus1_dict[aa] += (spectrum.intensity_array[i] / signal_total)

    def summarize(self):
        """
        Calculate the average proportion of signal coming from reporter ions across
        all spectra
        :return: 
        """
        print("Signal proportion under 1/3 precursor mz: %f" % (self.sum_proportion_ms2_signal_onethird_precursor / self.n_total_spectra))
        print("y1:")
        for aa in self.sum_proportions_y1_dict:
            print("%s: %f" % (aa, self.sum_proportions_y1_dict[aa] / self.n_total_spectra))
        print("b(n-1):")
        for aa in self.sum_proportions_bnminus1_dict:
            print("%s: %f" % (aa, self.sum_proportions_bnminus1_dict[aa] / self.n_total_spectra))
        for curiontype_proportion_map in [self.sum_proportions_y1_dict, self.sum_proportions_bnminus1_dict]:
            print("ion type.")
            control_bin_sums = [curiontype_proportion_map[aa] for aa in CONTROL_AAS]
            control_mean = np.mean(control_bin_sums)
            control_sd = np.std(control_bin_sums)
            argc_zscore = (curiontype_proportion_map['R'] - control_mean) / control_sd
            print("  ArgC z-score: %f" % argc_zscore)
            lysc_zscore = (curiontype_proportion_map['K'] - control_mean) / control_sd
            print("  LysC z-score: %f" % lysc_zscore)
            trypsin_min_zscore = min(argc_zscore, lysc_zscore)
            print("  min(ArgC, LysC): %f" % trypsin_min_zscore)
            chymo_bin_sums = [curiontype_proportion_map[aa] for aa in CHYMOTRYPSIN_AAS]
            chymotrypsin_t_statistic = ttest_ind(chymo_bin_sums, control_bin_sums)[0]
            print("  Chymotrypsin t-statistic: %f" % chymotrypsin_t_statistic)
        print(self.aa_y1_charge1_mz_map['K'])
        print(self.aa_y1_charge1_mz_map['R'])
        print(self.aa_y1_charge1_bin_map['K'])
        print(self.aa_y1_charge1_bin_map['R'])
        return []

