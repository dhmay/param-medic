#!/usr/bin/env python
"""
Code to analyze pairs of closely-eluting spectra with similar precursor and fragments
and infer precursor and fragment error.

This implementation bins the m/z range by Averagine peak-sized bins and, stepping one by one
through the MS/MS spectra, keeps track of the most-recent scan in each bin. Whenever
the next scan in a bin is close enough in precursor to the previous, the pair is evaluated
for adequate overlap of fragments. If there are sufficient fragments in common, the
m/z difference between the precursors and between the pairs of the top fragments are added
to a growing list of such differences.

When all scans are parsed, a mixed Gaussian-Uniform distribution is fit to the precursor
error distribution and the fragment error distribution. Standard deviations of these
distributions are transformed into algorithm parameter settings by multiplying by
a factor determined empirically on 8 training datasets.

Multiple files are handled simply: by blanking out the map that keeps track of scans
in each bin. That way, no pairs are made between files.

A word on efficiency: aside from the mixture-model stuff in mixturemodel.pyx (which I
got from Josh Schreiber -- he deserves all that credit!) none of this code is
particularly efficient. It's written for clarity and simplicity. I don't think the
efficiency matters so much for this use case -- the code runs in under a minute on
the biggest files I throw at it; most of that time is unavoidable I/O, and most of
the rest is Josh's already-nicely-efficient EM implementation.

If efficiency in this part of the code turns out to be a bigger issue, for someone,
I'll consider a rewrite.
"""

import logging
import math
import random
from util import RunAttributeDetector, MIN_SCAN_PEAKS

import numpy as np

from mixturemodel import GeneralMixtureModel, NormalDistribution, UniformDistribution
from util import AVERAGINE_PEAK_SEPARATION

__author__ = "Damon May"
__copyright__ = "Copyright (c) 2016 Damon May"
__license__ = ""
__version__ = ""

logger = logging.getLogger(__name__)

# default values for parameters

# minimum and maximum values for precursor and fragment m/z to consider
DEFAULT_MIN_MZ_FOR_BIN_PRECURSOR = 400.
DEFAULT_MAX_MZ_FOR_BIN_PRECURSOR = 1800.
DEFAULT_MIN_MZ_FOR_BIN_FRAGMENT = 150.
DEFAULT_MAX_MZ_FOR_BIN_FRAGMENT = 1800.

# charge of scan to consider. T
DEFAULT_CHARGE = 2

# Number of most-intense fragment peaks to store per scan
DEFAULT_TOPN_FRAGPEAKS = 30
# Minimum number of fragments two scans must have in common (at a gross level) to
# be considered likely to represent the same peptide
DEFAULT_MIN_FRAGPEAKS_INCOMMON = 20
# Number of fragment peak pairs to use for error estimation
DEFAULT_TOPN_FRAGPEAKS_FOR_ERROR_EST = 5

# maximum scans that can separate two scans for them to be compared.
# This is something of a hack -- ideally, the value would vary based
# on the gradient, or really we'd use retention time, but again the right
# value would depend on the gradient
DEFAULT_MAX_SCANS_BETWEEN_COMPARESCANS = 1000

# maximum PPM difference between two scans for them to be compared
DEFAULT_MAX_PRECURSORDIST_PPM = 50.

# minimum peaks required to try to fit a mixed distribution
DEFAULT_MIN_PEAKPAIRS_FOR_DISTRIBUTION_FIT = 200

#if more than this proportion of mass bins have more than one peak,
#we might be looking at profile-mode data
PROPORTION_MASSBINS_MULTIPEAK_INDICATES_PROFILEMODE = 0.5


# constants


# maximum proportion of precursor delta-masses that can be 0, otherwise we give up
MAX_PROPORTION_PRECURSORDELTAS_0 = 0.5
# maximum peaks to use to fit a mixed distribution
MAX_PEAKPAIRS_FOR_DISTRIBUTION_FIT = 100000

# empirically-derived values for transforming Gaussian error distributions into predictions
FRAG_SIGMA_MULTIPLIER = 0.004274
PRECURSOR_SIGMA_MULTIPLIER = 37.404067

# minimum allowed values for sigma of the estimated normal
MIN_SIGMA_PPM = 0.01
MIN_SIGMA_TH = 0.00001


class ErrorCalculator(RunAttributeDetector):
    """
    Class that accumulates pairs of precursors and fragments and uses them to estimate mass error.
    """
    def __init__(self,
                 min_precursor_mz=DEFAULT_MIN_MZ_FOR_BIN_PRECURSOR,
                 max_precursor_mz=DEFAULT_MAX_MZ_FOR_BIN_PRECURSOR,
                 min_frag_mz=DEFAULT_MIN_MZ_FOR_BIN_FRAGMENT,
                 max_frag_mz=DEFAULT_MAX_MZ_FOR_BIN_FRAGMENT,
                 charge=DEFAULT_CHARGE,
                 min_scan_frag_peaks=MIN_SCAN_PEAKS,
                 topn_frag_peaks=DEFAULT_TOPN_FRAGPEAKS,
                 min_common_frag_peaks=DEFAULT_MIN_FRAGPEAKS_INCOMMON,
                 pair_topn_frag_peaks=DEFAULT_TOPN_FRAGPEAKS_FOR_ERROR_EST,
                 max_scan_separation=DEFAULT_MAX_SCANS_BETWEEN_COMPARESCANS,
                 max_precursor_deltappm=DEFAULT_MAX_PRECURSORDIST_PPM,
                 min_peakpairs=DEFAULT_MIN_PEAKPAIRS_FOR_DISTRIBUTION_FIT
                 ):
        """

        :param min_precursor_mz:
        :param max_precursor_mz:
        :param min_frag_mz:
        :param max_frag_mz:
        :param charge:
        :param min_scan_frag_peaks:
        :param topn_frag_peaks:
        :param min_common_frag_peaks:
        :param pair_topn_frag_peaks:
        :param max_scan_separation:
        :param max_precursor_deltappm:
        :param min_peakpairs:
        """

        # set variables based on parameters
        self.min_precursor_mz = min_precursor_mz
        self.max_precursor_mz = max_precursor_mz
        self.min_frag_mz = min_frag_mz
        self.max_frag_mz = max_frag_mz
        self.charge = charge
        self.min_scan_frag_peaks = min_scan_frag_peaks
        self.topn_frag_peaks = topn_frag_peaks
        self.min_common_frag_peaks = min_common_frag_peaks
        self.pair_topn_frag_peaks = pair_topn_frag_peaks
        self.max_scan_separation = max_scan_separation
        self.max_precursor_deltappm = max_precursor_deltappm
        self.min_peakpairs = min_peakpairs

        logger.debug('min_precursor_mz: %f' % min_precursor_mz)
        logger.debug('max_precursor_mz: %f' % max_precursor_mz)
        logger.debug('min_frag_mz: %f' % min_frag_mz)
        logger.debug('max_frag_mz: %f' % max_frag_mz)
        logger.debug('charge: %f' % charge)
        logger.debug('min_scan_frag_peaks: %f' % min_scan_frag_peaks)
        logger.debug('topn_frag_peaks: %f' % topn_frag_peaks)
        logger.debug('min_common_frag_peaks: %f' % min_common_frag_peaks)
        logger.debug('pair_topn_frag_peaks: %f' % pair_topn_frag_peaks)
        logger.debug('max_scan_separation: %f' % max_scan_separation)
        logger.debug('max_precursor_deltappm: %f' % max_precursor_deltappm)


        # count the spectra that go by
        self.n_total_spectra = 0
        self.n_passing_spectra = 0
        # count the spectra that go by, by charge
        self.charge_spectracount_map = {}
        self.n_spectra_samebin_other_spectrum = 0
        self.n_spectra_withinppm_other_spectrum = 0
        self.n_spectra_withinppm_withinscans_other_spectrum = 0

        # multipliers to transform standord error values into algorithm parameters
        self.precursor_sigma_multiplier = PRECURSOR_SIGMA_MULTIPLIER
        self.frag_sigma_multiplier = FRAG_SIGMA_MULTIPLIER

        # define the number and position of bins
        self.lowest_precursorbin_startmz = self.min_precursor_mz - (self.min_precursor_mz % ( AVERAGINE_PEAK_SEPARATION / self.charge))
        self.lowest_fragmentbin_startmz = self.min_frag_mz - (self.min_frag_mz % AVERAGINE_PEAK_SEPARATION)
        self.n_precursor_bins = self.calc_binidx_for_mz_precursor(self.max_precursor_mz) + 1
        self.n_fragment_bins = self.calc_binidx_for_mz_fragment(self.max_frag_mz) + 1

        # map from bin index to current spectrum. This could also be implemented as a sparse array
        self.binidx_currentspectrum_map = {}

        # the paired peak values that we'll use to estimate mass error
        self.paired_fragment_peaks = []
        self.paired_precursor_mzs = []

        # track the number of candidatebins in which there's just one fragment, and the number in which there
        # are multiple. The ratio of the two can indicate profile-mode data
        self.n_bins_multiple_frags = 0
        self.n_bins_one_frag = 0

    # these utility methods find the right bin for a given mz

    def calc_binidx_for_mz_precursor(self, mz):
        return int(math.floor((mz - self.lowest_precursorbin_startmz) / (AVERAGINE_PEAK_SEPARATION / self.charge)))

    def calc_binidx_for_mz_fragment(self, mz):
        return int(math.floor((mz - self.lowest_fragmentbin_startmz) / AVERAGINE_PEAK_SEPARATION))

    # these utility methods map an mz to its offset from the specified bin center

    def calc_bin_offset_precursor(self, mz, bin_idx):
        return mz - self.calc_bin_startmz_precursor(bin_idx)

    def calc_bin_offset_fragment(self, mz, bin_idx):
        return mz - self.calc_bin_startmz_fragment(bin_idx)

    # these utility methods calculate the start mz values of a specified bin

    def calc_bin_startmz_precursor(self, bin_idx):
        return self.lowest_precursorbin_startmz + bin_idx * (AVERAGINE_PEAK_SEPARATION / self.charge)

    def calc_bin_startmz_fragment(self, bin_idx):
        return self.lowest_fragmentbin_startmz + bin_idx * AVERAGINE_PEAK_SEPARATION

    def clear_all_bins(self):
        """
        Clear all the bins of their most-recent spectra. This is done in order to handle multiple
        files without making pairs between files
        :return:
        """
        self.binidx_currentspectrum_map = {}

    def process_spectrum(self, spectrum, binned_spectrum):
        """
        Handle a spectrum. Check its charge and its number of scans. If passes, find the right
        bin for the precursor. If there's a previous scan in that bin, check to see if we've got
        a pair; if so, record all the peak pair info. Regardless, put the new scan in the bin.
        :param spectrum:
        :return:
        """
        # accounting:
        self.n_total_spectra += 1
        if spectrum.charge not in self.charge_spectracount_map:
            self.charge_spectracount_map[spectrum.charge] = 0
        self.charge_spectracount_map[spectrum.charge] += 1

        if spectrum.charge != self.charge:
            return
        if len(spectrum.mz_array) < self.min_scan_frag_peaks:
            return
        if self.min_precursor_mz <= spectrum.precursor_mz <= self.max_precursor_mz:
            self.n_passing_spectra += 1
            # pull out the top fragments by intensity
            topn_frag_idxs_intensity_desc = np.argsort(spectrum.intensity_array)[::-1][0:self.topn_frag_peaks]
            topn_frags = [(spectrum.mz_array[i], spectrum.intensity_array[i]) for i in topn_frag_idxs_intensity_desc]

            # create a SpectrumObservation object representing this spectrum. This takes up less
            # room than the whole thing.
            current_spec_obs = SpectrumObservation(spectrum.scan_number, spectrum.precursor_mz, topn_frags)

            precursor_bin_idx = self.calc_binidx_for_mz_precursor(spectrum.precursor_mz)
            if precursor_bin_idx in self.binidx_currentspectrum_map:
                # there was a previous spectrum in this bin. Check to see if they're a pair
                prev_spec_obs = self.binidx_currentspectrum_map[precursor_bin_idx]
                precursor_mz_diff = spectrum.precursor_mz - prev_spec_obs.precursor_mz
                precursor_mz_diff_ppm = precursor_mz_diff * 1000000 / spectrum.precursor_mz
                self.n_spectra_samebin_other_spectrum += 1
                # check precursor
                if abs(precursor_mz_diff_ppm) < self.max_precursor_deltappm:
                    # check scan count between the scans
                    self.n_spectra_withinppm_other_spectrum += 1
                    if current_spec_obs.scan_number - prev_spec_obs.scan_number <= self.max_scan_separation:
                        # count the fragment peaks in common
                        self.n_spectra_withinppm_withinscans_other_spectrum += 1
                        paired_fragments_bybin = self.pair_fragments_bybin(prev_spec_obs, current_spec_obs)
                        if len(paired_fragments_bybin) >= self.min_common_frag_peaks:
                            # we've got a pair! record everything
                            minints = [min(x[1], y[1]) for x, y in paired_fragments_bybin]
                            top_minint_idxs = np.argsort(minints)[::1][0:self.pair_topn_frag_peaks]

                            self.paired_fragment_peaks.extend([paired_fragments_bybin[i] for i in top_minint_idxs])
                            self.paired_precursor_mzs.append((prev_spec_obs.precursor_mz, current_spec_obs.precursor_mz))
            # make the new spectrum its bin's representative
            self.binidx_currentspectrum_map[precursor_bin_idx] = current_spec_obs

    def pair_fragments_bybin(self, prev_spec_obs, current_spec_obs):
        """
        given two spectra, pair up their fragments that are in the same bins
        :param prev_spec_obs:
        :param current_spec_obs:
        :return:
        """
        bin_fragment_map_prev = self.bin_topn_fragments(prev_spec_obs)
        bin_fragment_map_current = self.bin_topn_fragments(current_spec_obs)
        result = []
        for bin_idx in bin_fragment_map_prev:
            if bin_idx in bin_fragment_map_current:
                result.append((bin_fragment_map_prev[bin_idx], bin_fragment_map_current[bin_idx]))
        return result

    def bin_topn_fragments(self, spec_obs):
        """
        keep only one fragment per bin. If another fragment wants to be in the bin, toss them *both* out.
        This reduces ambiguity.
        :param spec_obs:
        :return:
        """
        bin_fragment_map = {}
        bins_to_remove = set()

        for mz, intensity in spec_obs.topn_fragments:
            if mz < self.min_frag_mz:
                continue
            bin_idx = self.calc_binidx_for_mz_fragment(mz)
            if bin_idx in bin_fragment_map:
                bins_to_remove.add(bin_idx)
            else:
                bin_fragment_map[bin_idx] = mz, intensity
        for bin_idx in bins_to_remove:
            del bin_fragment_map[bin_idx]
        self.n_bins_multiple_frags += len(bins_to_remove)
        self.n_bins_one_frag += len(bin_fragment_map)
        return bin_fragment_map

    def calc_masserror_dist(self):
        """
        This is to be run after all spectra have been processed. Fit the mixed model to the mixed
        distributions of m/z differences
        :return:
        """
        logger.debug("Processed %d total spectra" % self.n_total_spectra)
        logger.debug("Processed %d qualifying spectra" % self.n_passing_spectra)
        logger.debug("Precursor pairs: %d" % len(self.paired_precursor_mzs))
        logger.debug("Fragment pairs: %d" % len(self.paired_fragment_peaks))

        logger.debug("Total spectra in same bin as another: %d" % self.n_spectra_samebin_other_spectrum)
        logger.debug("Total spectra in same bin as another and within m/z tol: %d" %
                     self.n_spectra_withinppm_other_spectrum)
        logger.debug("Total spectra in same bin as another and within m/z tol and within scan range: %d" %
                     self.n_spectra_withinppm_withinscans_other_spectrum)
        # check the proportion of mass bins, in the whole file, that have multiple fragments.
        # If that's high, we might be looking at profile-mode data.
        if self.n_bins_one_frag + self.n_bins_multiple_frags == 0:
            proportion_bins_multiple_frags = 0
            logger.debug("No values in any bin!")
        else:
            proportion_bins_multiple_frags = float(self.n_bins_multiple_frags) / \
                                             (self.n_bins_one_frag + self.n_bins_multiple_frags)
            logger.debug("Proportion of bins with multiple fragments: %.02f" % (proportion_bins_multiple_frags))

        precursor_distances_ppm = []
        n_zero_precursor_deltas = 0
        if len(self.paired_precursor_mzs) > MAX_PEAKPAIRS_FOR_DISTRIBUTION_FIT:
            logger.debug("Using %d of %d peak pairs for precursor..." %
                         (MAX_PEAKPAIRS_FOR_DISTRIBUTION_FIT, len(self.paired_precursor_mzs)))
            self.paired_precursor_mzs = random.sample(self.paired_precursor_mzs, MAX_PEAKPAIRS_FOR_DISTRIBUTION_FIT)
        for mz1, mz2 in self.paired_precursor_mzs:
            diff_th = mz1 - mz2
            if diff_th == 0.0:
                n_zero_precursor_deltas += 1
            precursor_distances_ppm.append(diff_th * 1000000 / mz1)

        # we need to report precursor if fragment fails, and vice versa.
        # these variables keep track of what failed and why
        failed_precursor = False
        precursor_message = "OK"

        if proportion_bins_multiple_frags > PROPORTION_MASSBINS_MULTIPEAK_INDICATES_PROFILEMODE:
            logger.info("Is this profile-mode data? Proportion of mass bins with multiple peaks is quite high (%.02f)" %
                        proportion_bins_multiple_frags)
            logger.info("Param-Medic will not perform well on profile-mode data.")

        # check for conditions that would cause us to bomb out
        if len(precursor_distances_ppm) < self.min_peakpairs:
            failed_precursor = True
            precursor_message = ("Need >= %d peak pairs to fit mixed distribution. Got only %d.\nDetails:\n" \
                                 "Spectra in same averagine bin as another: %d\n" \
                                 "    ... and also within m/z tolerance: %d\n" \
                                 "    ... and also within scan range: %d\n" \
                                 "    ... and also with sufficient in-common fragments: %d\n" %
                                 (self.min_peakpairs, len(precursor_distances_ppm),
                                  self.n_spectra_samebin_other_spectrum,
                                  self.n_spectra_withinppm_other_spectrum,
                                  self.n_spectra_withinppm_withinscans_other_spectrum,
                                  len(precursor_distances_ppm)))
            if proportion_bins_multiple_frags > PROPORTION_MASSBINS_MULTIPEAK_INDICATES_PROFILEMODE:
                precursor_message += "Is this profile-mode data? Proportion of mass bins with multiple peaks is quite high (%.02f)\n" % proportion_bins_multiple_frags
                precursor_message += "Param-Medic will not perform well on profile-mode data.\n"

        if not failed_precursor:
            proportion_precursor_mzs_zero = float(n_zero_precursor_deltas) / len(self.paired_precursor_mzs)
            logger.debug("proportion zero: %f" % proportion_precursor_mzs_zero)
            if proportion_precursor_mzs_zero > MAX_PROPORTION_PRECURSORDELTAS_0:
                failed_precursor = True
                precursor_message = "Too high a proportion of precursor mass differences (%f) are exactly 0. " \
                                 "Some processing has been done on this run that param-medic can't handle. " \
                                 "You should investigate what that processing might be." % proportion_precursor_mzs_zero

        precursor_mu_ppm_2measures, precursor_sigma_ppm_2measures = None, None
        if not failed_precursor:
            try:
                precursor_mu_ppm_2measures, precursor_sigma_ppm_2measures = estimate_mu_sigma(precursor_distances_ppm, MIN_SIGMA_PPM)
            except Exception as e:
                failed_precursor = True
                precursor_message = "Unknown error estimating mu, sigma: %s" % str(e)

        failed_fragment = False
        fragment_message = "OK"

        frag_distances_ppm = []
        if len(self.paired_fragment_peaks) < self.min_peakpairs:
            failed_fragment = True
            fragment_message = ("Need >= %d peak pairs to fit mixed distribution. Got only %d\nDetails:\n" \
                                "Spectra in same averagine bin as another: %d\n" \
                                "    ... and also within m/z tolerance: %d\n" \
                                "    ... and also within scan range: %d\n" 
                                "    ... and also with sufficient in-common fragments: %d\n"%
                                (self.min_peakpairs, len(self.paired_fragment_peaks),
                                 self.n_spectra_samebin_other_spectrum,
                                 self.n_spectra_withinppm_other_spectrum,
                                 self.n_spectra_withinppm_withinscans_other_spectrum,
                                 len(self.paired_fragment_peaks)))
            if proportion_bins_multiple_frags > PROPORTION_MASSBINS_MULTIPEAK_INDICATES_PROFILEMODE:
                fragment_message += "Is this profile-mode data? Proportion of mass bins with multiple peaks is quite high (%.02f)\n" % proportion_bins_multiple_frags
                fragment_message += "Param-Medic will not perform well on profile-mode data.\n"
        frag_mu_ppm_2measures, frag_sigma_ppm_2measures = None, None
        if not failed_fragment:
            if len(self.paired_fragment_peaks) > MAX_PEAKPAIRS_FOR_DISTRIBUTION_FIT:
                logger.debug("Using %d of %d peak pairs for fragment..." %
                             (MAX_PEAKPAIRS_FOR_DISTRIBUTION_FIT, len(self.paired_fragment_peaks)))
                self.paired_fragment_peaks = random.sample(self.paired_fragment_peaks, MAX_PEAKPAIRS_FOR_DISTRIBUTION_FIT)
            for fragpeak1, fragpeak2 in self.paired_fragment_peaks:
                diff_th = fragpeak1[0] - fragpeak2[0]
                frag_distances_ppm.append(diff_th * 1000000 / fragpeak1[0])
            try:
                # estimate the parameters of the component distributions for each of the mixed distributions.
                frag_mu_ppm_2measures, frag_sigma_ppm_2measures = estimate_mu_sigma(frag_distances_ppm, MIN_SIGMA_PPM)
            except Exception as e:
                failed_fragment = True
                fragment_message = "Unknown error estimating mu, sigma: %s" % str(e)

        if failed_precursor:
            logger.debug("Failed precursor! %s" % precursor_message)
        else:
            logger.debug('precursor_mu_ppm_2measures: %f' % precursor_mu_ppm_2measures)
            logger.debug('precursor_sigma_ppm_2measures: %f' % precursor_sigma_ppm_2measures)

        if failed_fragment:
            logger.debug("Failed fragment! %s" % fragment_message)
        else:
            logger.debug('frag_mu_ppm_2measures: %f' % frag_mu_ppm_2measures)
            logger.debug('frag_sigma_ppm_2measures: %f' % frag_sigma_ppm_2measures)

        # what we have now measured, in the fit Gaussians, is the distribution of the difference
        # of two values drawn from the distribution of error values.
        # Assuming the error values are normally distributed with mean 0 and variance s^2, the
        # differences are normally distributed with mean 0 and variance 2*s^2:
        # http://mathworld.wolfram.com/NormalDifferenceDistribution.html
        # i.e., differences are normally distributed with mean=0 and sd=sqrt(2)*s
        # hence, if differences have sd=diff_sigma, then errors have sd diff_sigma/sqrt(2)
        #
        # incidentally, this transformation doesn't matter one bit, practically, since we're
        # inferring a multiplier for this value empirically. But it lets us report something
        # with an easily-interpretable meaning as an intermediate value
        precursor_sigma_ppm = None
        precursor_prediction_ppm = None
        if not failed_precursor:
            precursor_sigma_ppm = precursor_sigma_ppm_2measures/math.sqrt(2)
            # generate prediction by multiplying by empirically-derived value
            precursor_prediction_ppm = self.precursor_sigma_multiplier * precursor_sigma_ppm
        frag_sigma_ppm = None
        fragment_prediction_th = None
        if not failed_fragment:
            frag_sigma_ppm = frag_sigma_ppm_2measures/math.sqrt(2)
            # generate prediction by multiplying by empirically-derived value
            fragment_prediction_th = self.frag_sigma_multiplier * frag_sigma_ppm

        return (failed_precursor, precursor_message, failed_fragment, fragment_message,
                precursor_sigma_ppm, frag_sigma_ppm,
                precursor_prediction_ppm, fragment_prediction_th)

    def summarize(self):
        (failed_precursor, precursor_message, failed_fragment, fragment_message, precursor_sigma_ppm, frag_sigma_ppm,
         precursor_prediction_ppm, fragment_prediction_th) = \
            self.calc_masserror_dist()
        print("Precursor and fragment error summary:")
        if failed_precursor:
            print("Precursor error calculation failed:")
            print(precursor_message)
        else:
            print('precursor standard deviation: %f ppm' % precursor_sigma_ppm)
        if failed_fragment:
            print("Fragment error calculation failed:")
            print(fragment_message)
        else:
            print('fragment standard deviation: %f ppm' % frag_sigma_ppm)
        logger.debug('')

        search_param_messages = []
        if not failed_precursor:
            search_param_messages.append("Precursor error: %.2f ppm" % precursor_prediction_ppm)
        if not failed_fragment:
            search_param_messages.append("Fragment bin size: %.4f Th" % fragment_prediction_th)
        return search_param_messages, precursor_sigma_ppm, frag_sigma_ppm, precursor_prediction_ppm, fragment_prediction_th

    def next_file(self):
        """
        Register that a new file is being processed.
        clear the bins so we don't end up using pairs across files
        :return: 
        """
        self.clear_all_bins()


def estimate_mu_sigma(data, min_sigma):
    """
    estimate mu and sigma of the mixed distribution, as initial estimate for Gaussian
    :param data: mixed distribution values
    :param min_sigma: minimum value to return
    :return:
    """
    mu_mixed_dist = np.mean(data)
    sigma_mixed_dist = np.std(data)
    logger.debug("mixed distribution: min %f, max %f, mean %f, sd %f" %
                 (min(data), max(data), mu_mixed_dist, sigma_mixed_dist))

    # model the observed distribution as a mixture of Gaussian and uniform
    mixture_model = GeneralMixtureModel([NormalDistribution(mu_mixed_dist, sigma_mixed_dist, min_std=min_sigma),
                                         UniformDistribution(min(data), max(data))])
    frag_deltamzs_ndarray = np.array(data)
    frag_deltamzs_ndarray.shape = (len(data), 1)
    # fit the mixture model with EM
    improvement = mixture_model.fit(frag_deltamzs_ndarray)
    logger.debug("model improvement: %f" % improvement)
    mu_fit = mixture_model.distributions[0].parameters[0]
    sigma_fit = mixture_model.distributions[0].parameters[1]
    logger.debug("fit: mean=%f, sigma=%f" % (mu_fit, sigma_fit))

    return mu_fit, sigma_fit


class SpectrumObservation(object):
    """
    Minimal model of a spectrum, for peak-pairing purposes
    """
    def __init__(self, scan_number, precursor_mz, topn_fragments):
        self.scan_number = scan_number
        self.precursor_mz = precursor_mz
        self.topn_fragments = topn_fragments


