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
import numpy as np
import math
from mixturemodel import GeneralMixtureModel, NormalDistribution, UniformDistribution
import random

__author__ = "Damon May"
__copyright__ = "Copyright (c) 2016 Damon May"
__license__ = ""
__version__ = ""

logger = logging.getLogger(__name__)

# Separation between Averagine peaks. This is used for binning spectra
AVERAGINE_PEAK_SEPARATION = 1.000495

# minimum and maximum values for precursor and fragment m/z to consider
MIN_MZ_FOR_BIN_PRECURSOR = 400.
MAX_MZ_FOR_BIN_PRECURSOR = 1800.
MIN_MZ_FOR_BIN_FRAGMENT = 150.
MAX_MZ_FOR_BIN_FRAGMENT = 1800.

# charge of scan to consider. T
CHARGE = 2

# minimum number of MS2 fragments that a scan must have to be considered
MIN_SCAN_MS2PEAKS = 40
# Number of most-intense fragment peaks to store per scan
TOPN_FRAGPEAKS = 30
# Minimum number of fragments two scans must have in common (at a gross level) to
# be considered likely to represent the same peptide
MIN_FRAGPEAKS_INCOMMON = 20
# Number of fragment peak pairs to use for error estimation
TOPN_FRAGPEAKS_FOR_ERROR_EST = 5

# maximum scans that can separate two scans for them to be compared.
# This is something of a hack -- ideally, the value would vary based
# on the gradient, or really we'd use retention time, but again the right
# value would depend on the gradient
MAX_SCANS_BETWEEN_COMPARESCANS = 1000

# maximum PPM difference between two scans for them to be compared
MAX_PRECURSORDIST_PPM = 50.

# maximum proportion of precursor delta-masses that can be 0, otherwise we give up
MAX_PROPORTION_PRECURSORDELTAS_0 = 0.5

# minimum peaks required to try to fit a mixed distribution
MIN_PEAKPAIRS_FOR_DISTRIBUTION_FIT = 200
# maximum peaks to use to fit a mixed distribution
MAX_PEAKPAIRS_FOR_DISTRIBUTION_FIT = 100000

# empirically-derived values for transforming Gaussian error distributions into predictions
#DEFAULT_FRAG_SIGMA_MULTIPLIER = 4.763766
#DEFAULT_PRECURSOR_SIGMA_MULTIPLIER = 11.130897
DEFAULT_FRAG_SIGMA_MULTIPLIER = 0.003543
DEFAULT_PRECURSOR_SIGMA_MULTIPLIER = 15.720249

# minimum allowed values for sigma of the estimated normal
MIN_SIGMA_PPM = 0.01
MIN_SIGMA_TH = 0.00001


class ErrorCalculator(object):
    """
    Class that accumulates pairs of precursors and fragments and uses them to estimate mass error.
    """
    def __init__(self,
                 precursor_sigma_multiplier=DEFAULT_PRECURSOR_SIGMA_MULTIPLIER,
                 frag_sigma_multiplier=DEFAULT_FRAG_SIGMA_MULTIPLIER,
                 averagine_peak_separation=AVERAGINE_PEAK_SEPARATION):
        """

        :param precursor_sigma_multiplier: multiplier to transform standord error value into algorithm parameters
        :param frag_sigma_multiplier: multiplier to transform standord error value into algorithm parameters
        :param averagine_peak_separation: separation between averagine peaks
        """
        # count the spectra that go by
        self.n_total_spectra = 0
        self.n_passing_spectra = 0

        # multipliers to transform standord error values into algorithm parameters
        self.precursor_sigma_multiplier = precursor_sigma_multiplier
        self.frag_sigma_multiplier = frag_sigma_multiplier

        self.averagine_peak_separation = averagine_peak_separation

        # define the number and position of bins
        self.lowest_precursorbin_startmz = MIN_MZ_FOR_BIN_PRECURSOR - (MIN_MZ_FOR_BIN_PRECURSOR % (averagine_peak_separation / CHARGE))
        self.lowest_fragmentbin_startmz = MIN_MZ_FOR_BIN_FRAGMENT - (MIN_MZ_FOR_BIN_FRAGMENT % averagine_peak_separation)
        self.n_precursor_bins = self.calc_binidx_for_mz_precursor(MAX_MZ_FOR_BIN_PRECURSOR) + 1
        self.n_fragment_bins = self.calc_binidx_for_mz_fragment(MAX_MZ_FOR_BIN_FRAGMENT) + 1

        # map from bin index to current spectrum. This could also be implemented as a sparse array
        self.binidx_currentspectrum_map = {}

        # the paired peak values that we'll use to estimate mass error
        self.paired_fragment_peaks = []
        self.paired_precursor_mzs = []

    # these utility methods find the right bin for a given mz

    def calc_binidx_for_mz_precursor(self, mz):
        return int(math.floor((mz - self.lowest_precursorbin_startmz) / (self.averagine_peak_separation / CHARGE)))

    def calc_binidx_for_mz_fragment(self, mz):
        return int(math.floor((mz - self.lowest_fragmentbin_startmz) / self.averagine_peak_separation))

    # these utility methods map an mz to its offset from the specified bin center

    def calc_bin_offset_precursor(self, mz, bin_idx):
        return mz - self.calc_bin_startmz_precursor(bin_idx)

    def calc_bin_offset_fragment(self, mz, bin_idx):
        return mz - self.calc_bin_startmz_fragment(bin_idx)

    # these utility methods calculate the start mz values of a specified bin

    def calc_bin_startmz_precursor(self, bin_idx):
        return self.lowest_precursorbin_startmz + bin_idx * (self.averagine_peak_separation / CHARGE)

    def calc_bin_startmz_fragment(self, bin_idx):
        return self.lowest_fragmentbin_startmz + bin_idx * self.averagine_peak_separation

    def clear_all_bins(self):
        """
        Clear all the bins of their most-recent spectra. This is done in order to handle multiple
        files without making pairs between files
        :return:
        """
        self.binidx_currentspectrum_map = {}

    def process_spectrum(self, spectrum):
        """
        Handle a spectrum. Check its charge and its number of scans. If passes, find the right
        bin for the precursor. If there's a previous scan in that bin, check to see if we've got
        a pair; if so, record all the peak pair info. Regardless, put the new scan in the bin.
        :param spectrum:
        :return:
        """
        self.n_total_spectra += 1
        if spectrum.charge != CHARGE:
            return
        if len(spectrum.mz_array) < MIN_SCAN_MS2PEAKS:
            return
        if MIN_MZ_FOR_BIN_PRECURSOR <= spectrum.precursor_mz <= MAX_MZ_FOR_BIN_PRECURSOR:
            self.n_passing_spectra += 1
            # pull out the top fragments by intensity
            topn_frag_idxs_intensity_desc = np.argsort(spectrum.intensity_array)[::-1][0:TOPN_FRAGPEAKS]
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
                # check precursor
                if abs(precursor_mz_diff_ppm) < MAX_PRECURSORDIST_PPM:
                    # check scan count between the scans
                    if current_spec_obs.scan_number - prev_spec_obs.scan_number <= MAX_SCANS_BETWEEN_COMPARESCANS:
                        # count the fragment peaks in common
                        paired_fragments_bybin = self.pair_fragments_bybin(prev_spec_obs, current_spec_obs)
                        if len(paired_fragments_bybin) >= MIN_FRAGPEAKS_INCOMMON:
                            # we've got a pair! record everything
                            minints = [min(x[1], y[1]) for x, y in paired_fragments_bybin]
                            top_minint_idxs = np.argsort(minints)[::1][0:TOPN_FRAGPEAKS_FOR_ERROR_EST]

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
        bin_fragment_map_prev = self.bin_fragments(prev_spec_obs)
        bin_fragment_map_current = self.bin_fragments(current_spec_obs)
        result = []
        for bin_idx in bin_fragment_map_prev:
            if bin_idx in bin_fragment_map_current:
                result.append((bin_fragment_map_prev[bin_idx], bin_fragment_map_current[bin_idx]))
        return result

    def bin_fragments(self, spec_obs):
        """
        keep only one fragment per bin. If another fragment wants to be in the bin, toss them *both* out.
        This reduces ambiguity.
        :param spec_obs:
        :return:
        """
        bin_fragment_map = {}
        bins_to_remove = set()

        for mz, intensity in spec_obs.topn_fragments:
            if mz < MIN_MZ_FOR_BIN_FRAGMENT:
                continue
            bin_idx = self.calc_binidx_for_mz_fragment(mz)
            if bin_idx in bin_fragment_map:
                bins_to_remove.add(bin_idx)
            else:
                bin_fragment_map[bin_idx] = mz, intensity
        for bin_idx in bins_to_remove:
            del bin_fragment_map[bin_idx]
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

        # check for conditions that would cause us to bomb out
        if len(precursor_distances_ppm) < MIN_PEAKPAIRS_FOR_DISTRIBUTION_FIT:
            raise ValueError("Need >= %d peak pairs to fit mixed distribution. Got only %d" %
                             (MIN_PEAKPAIRS_FOR_DISTRIBUTION_FIT, len(precursor_distances_ppm)))
        proportion_precursor_mzs_zero = float(n_zero_precursor_deltas) / len(self.paired_precursor_mzs)
        logger.debug("proportion zero: %f" % proportion_precursor_mzs_zero)
        if proportion_precursor_mzs_zero > MAX_PROPORTION_PRECURSORDELTAS_0:
            raise ValueError("Too high a proportion of precursor mass differences (%f) are exactly 0. " \
                             "Some processing has been done on this run that param-medic can't handle. " \
                             "You should investigate what that processing might be." %
                             proportion_precursor_mzs_zero)

        frag_distances_ppm = []
        if len(self.paired_fragment_peaks) > MAX_PEAKPAIRS_FOR_DISTRIBUTION_FIT:
            logger.debug("Using %d of %d peak pairs for fragment..." %
                         (MAX_PEAKPAIRS_FOR_DISTRIBUTION_FIT, len(self.paired_fragment_peaks)))
            self.paired_fragment_peaks = random.sample(self.paired_fragment_peaks, MAX_PEAKPAIRS_FOR_DISTRIBUTION_FIT)
        for fragpeak1, fragpeak2 in self.paired_fragment_peaks:
            diff_th = fragpeak1[0] - fragpeak2[0]
            frag_distances_ppm.append(diff_th * 1000000 / fragpeak1[0])

        # estimate the parameters of the component distributions for each of the mixed distributions.
        precursor_mu_ppm_2measures, precursor_sigma_ppm_2measures = estimate_mu_sigma(precursor_distances_ppm, MIN_SIGMA_PPM)
        frag_mu_ppm_2measures, frag_sigma_ppm_2measures = estimate_mu_sigma(frag_distances_ppm, MIN_SIGMA_PPM)

        logger.debug('precursor_mu_ppm_2measures: %f' % precursor_mu_ppm_2measures)
        logger.debug('precursor_sigma_ppm_2measures: %f' % precursor_sigma_ppm_2measures)
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
        precursor_sigma_ppm = precursor_sigma_ppm_2measures/math.sqrt(2)
        frag_sigma_ppm = frag_sigma_ppm_2measures/math.sqrt(2)

        # generate predictions by multiplying by empirically-derived values
        precursor_prediction_ppm = self.precursor_sigma_multiplier * precursor_sigma_ppm
        fragment_prediction_th = self.frag_sigma_multiplier * frag_sigma_ppm

        return (precursor_sigma_ppm, frag_sigma_ppm,
                precursor_prediction_ppm, fragment_prediction_th)


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