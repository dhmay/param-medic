#!/usr/bin/env python
"""
Analyze all scans in a run to determine precursor mass error
"""

import logging
import numpy as np
from parammedic import mixturemodel
import math
from scipy.optimize import curve_fit

__author__ = "Damon May"
__copyright__ = "Copyright (c) 2016 Damon May"
__license__ = ""
__version__ = ""

logger = logging.getLogger(__name__)


AVERAGINE_PEAK_SEPARATION = 1.000495

MIN_MZ_FOR_BIN_PRECURSOR = 400.
MAX_MZ_FOR_BIN_PRECURSOR = 1800.
MIN_MZ_FOR_BIN_FRAGMENT = 150.
MAX_MZ_FOR_BIN_FRAGMENT = 1800.

CHARGE = 2

MAX_OFFSET_DIST_PPM = 100.

N_DISTANCES_FOR_BACKGROUND_DIST = 1000

# minimum value we're allowed to report for sigma
MIN_PRECURSOR_SIGMA = 1.0

# look this many bins above a given bin to find a fragment to compare for background estimation
N_BINS_ABOVE_TO_COMPARE_BACKGROUND = 7

MIN_SCAN_MS2PEAKS = 40
TOPN_FRAGPEAKS = 30
MIN_FRAGPEAKS_INCOMMON = 20
TOPN_FRAGPEAKS_FOR_ERROR_EST = 5

# maximum scans that can separate two scans for them to be compared
MAX_SCANS_BETWEEN_COMPARESCANS = 1000

MAX_PRECURSORDIST_PPM = 50.

# maximum proportion of precursor delta-masses that can be 0, otherwise we give up
MAX_PROPORTION_PRECURSORDELTAS_0 = 0.5

# empirically-derived values for transforming Gaussian error distributions into predictions
DEFAULT_FRAG_SIGMA_MULTIPLIER = 4.763766
DEFAULT_PRECURSOR_SIGMA_MULTIPLIER = 11.130897

# minimum allowed values for sigma of the estimated normal
MIN_SIGMA_PPM = 0.01
MIN_SIGMA_TH = 0.00001


class ErrorCalculator(object):
    def __init__(self,
                 precursor_sigma_multiplier=DEFAULT_PRECURSOR_SIGMA_MULTIPLIER,
                 frag_sigma_multiplier=DEFAULT_FRAG_SIGMA_MULTIPLIER,
                 averagine_peak_separation=AVERAGINE_PEAK_SEPARATION):
        self.n_total_spectra = 0
        self.n_passing_spectra = 0

        self.precursor_sigma_multiplier = precursor_sigma_multiplier
        self.frag_sigma_multiplier = frag_sigma_multiplier

        self.averagine_peak_separation = averagine_peak_separation
        self.lowest_precursorbin_startmz = MIN_MZ_FOR_BIN_PRECURSOR - (MIN_MZ_FOR_BIN_PRECURSOR % (averagine_peak_separation / CHARGE))
        self.lowest_fragmentbin_startmz = MIN_MZ_FOR_BIN_FRAGMENT - (MIN_MZ_FOR_BIN_FRAGMENT % averagine_peak_separation)
        self.n_precursor_bins = self.calc_binidx_for_mz_precursor(MAX_MZ_FOR_BIN_PRECURSOR) + 1
        self.n_fragment_bins = self.calc_binidx_for_mz_fragment(MAX_MZ_FOR_BIN_FRAGMENT) + 1

        self.chart_list = []

        self.binidx_currentspectrum_map = {}

        self.paired_fragment_peaks = []
        self.paired_precursor_mzs = []

    def calc_binidx_for_mz_precursor(self, mz):
        return int(math.floor((mz - self.lowest_precursorbin_startmz) / (self.averagine_peak_separation / CHARGE)))

    def calc_binidx_for_mz_fragment(self, mz):
        return int(math.floor((mz - self.lowest_fragmentbin_startmz) / self.averagine_peak_separation))

    def calc_bin_offset_precursor(self, mz, bin_idx):
        return mz - self.calc_bin_startmz_precursor(bin_idx)

    def calc_bin_offset_fragment(self, mz, bin_idx):
        return mz - self.calc_bin_startmz_fragment(bin_idx)

    def calc_bin_startmz_precursor(self, bin_idx):
        return self.lowest_precursorbin_startmz + bin_idx * (self.averagine_peak_separation / CHARGE)

    def calc_bin_startmz_fragment(self, bin_idx):
        return self.lowest_fragmentbin_startmz + bin_idx * self.averagine_peak_separation

    def process_spectrum(self, spectrum):
        self.n_total_spectra += 1
        if spectrum.charge != CHARGE:
            return
        if len(spectrum.mz_array) < MIN_SCAN_MS2PEAKS:
            return
        if MIN_MZ_FOR_BIN_PRECURSOR <= spectrum.precursor_mz <= MAX_MZ_FOR_BIN_PRECURSOR:
            self.n_passing_spectra += 1
            topn_frag_idxs_intensity_desc = np.argsort(spectrum.intensity_array)[::-1][0:TOPN_FRAGPEAKS]
            topn_frags = [(spectrum.mz_array[i], spectrum.intensity_array[i]) for i in topn_frag_idxs_intensity_desc]
            current_spec_obs = SpectrumObservation(spectrum.scan_number, spectrum.precursor_mz, topn_frags)

            precursor_bin_idx = self.calc_binidx_for_mz_precursor(spectrum.precursor_mz)
            if precursor_bin_idx in self.binidx_currentspectrum_map:
                prev_spec_obs = self.binidx_currentspectrum_map[precursor_bin_idx]
                precursor_mz_diff = spectrum.precursor_mz - prev_spec_obs.precursor_mz
                precursor_mz_diff_ppm = precursor_mz_diff * 1000000 / spectrum.precursor_mz
                if abs(precursor_mz_diff_ppm) < MAX_PRECURSORDIST_PPM:
                    if current_spec_obs.scan_number - prev_spec_obs.scan_number <= MAX_SCANS_BETWEEN_COMPARESCANS:
                        paired_fragments_bybin = self.pair_fragments_bybin(prev_spec_obs, current_spec_obs)
                        if len(paired_fragments_bybin) >= MIN_FRAGPEAKS_INCOMMON:
                            minints = [min(x[1], y[1]) for x, y in paired_fragments_bybin]
                            top_minint_idxs = np.argsort(minints)[::1][0:TOPN_FRAGPEAKS_FOR_ERROR_EST]

                            self.paired_fragment_peaks.extend([paired_fragments_bybin[i] for i in top_minint_idxs])
                            self.paired_precursor_mzs.append((prev_spec_obs.precursor_mz, current_spec_obs.precursor_mz))
            self.binidx_currentspectrum_map[precursor_bin_idx] = current_spec_obs

    def pair_fragments_bybin(self, prev_spec_obs, current_spec_obs):
        bin_fragment_map_prev = self.bin_fragments(prev_spec_obs)
        bin_fragment_map_current = self.bin_fragments(current_spec_obs)
        result = []
        for bin_idx in bin_fragment_map_prev:
            if bin_idx in bin_fragment_map_current:
                result.append((bin_fragment_map_prev[bin_idx], bin_fragment_map_current[bin_idx]))
        return result

    def bin_fragments(self, spec_obs):
        """
        keep only one fragment per bin. If another fragment wants to be in the bin, toss them both out
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
        print("Processed %d total spectra" % self.n_total_spectra)
        print("Processed %d qualifying spectra" % self.n_passing_spectra)
        print("Precursor pairs: %d" % len(self.paired_precursor_mzs))
        print("Fragment pairs: %d" % len(self.paired_fragment_peaks))

        precursor_distances_th = []
        precursor_distances_ppm = []
        n_zero_precursor_deltas = 0
        for mz1, mz2 in self.paired_precursor_mzs:
            diff_th = mz1 - mz2
            if diff_th == 0.0:
                n_zero_precursor_deltas += 1
            precursor_distances_th.append(diff_th)
            precursor_distances_ppm.append(diff_th * 1000000 / mz1)

        frag_distances_th = []
        frag_distances_ppm = []
        for fragpeak1, fragpeak2 in self.paired_fragment_peaks:
            diff_th = fragpeak1[0] - fragpeak2[0]
            frag_distances_th.append(diff_th)
            frag_distances_ppm.append(diff_th * 1000000 / fragpeak1[0])

        precursor_mu_ppm_2measures, precursor_sigma_ppm_2measures = estimate_mu_sigma(precursor_distances_ppm, MIN_SIGMA_PPM)
        precursor_mu_th_2measures, precursor_sigma_th_2measures = estimate_mu_sigma(precursor_distances_th, MIN_SIGMA_TH)
        frag_mu_ppm_2measures, frag_sigma_ppm_2measures = estimate_mu_sigma(frag_distances_ppm, MIN_SIGMA_PPM)
        frag_mu_th_2measures, frag_sigma_th_2measures = estimate_mu_sigma(frag_distances_th, MIN_SIGMA_TH)

        print('precursor_mu_ppm_2measures: %f' % precursor_mu_ppm_2measures)
        print('precursor_sigma_ppm_2measures: %f' % precursor_sigma_ppm_2measures)
        print('frag_mu_ppm_2measures: %f' % frag_mu_ppm_2measures)
        print('frag_sigma_ppm_2measures: %f' % frag_sigma_ppm_2measures)
        print('frag_mu_th_2measures %f' % frag_mu_th_2measures)
        print('frag_sigma_th_2measures : %f' % frag_sigma_th_2measures)

        # what we have now measured is the sum of two errors.
        # "the sum of two independent normally distributed random variables is normal,
        # with its mean being the sum of the two means, and its variance being the sum of the two variances
        # (i.e., the square of the standard deviation is the sum of the squares of the standard deviations)."
        # https://en.wikipedia.org/wiki/Sum_of_normally_distributed_random_variables
        precursor_mu_ppm = precursor_mu_ppm_2measures / 2
        precursor_mu_th = precursor_mu_th_2measures / 2
        frag_mu_ppm = frag_mu_ppm_2measures / 2
        frag_mu_th = frag_mu_th_2measures / 2

        precursor_sigma_ppm = math.sqrt(math.pow(precursor_sigma_ppm_2measures, 2))
        precursor_sigma_th = math.sqrt(math.pow(precursor_sigma_th_2measures, 2))
        frag_sigma_ppm = math.sqrt(math.pow(frag_sigma_ppm_2measures, 2))
        frag_sigma_th = math.sqrt(math.pow(frag_sigma_th_2measures, 2))

        proportion_precursor_mzs_zero = float(n_zero_precursor_deltas) / len(self.paired_precursor_mzs)
        logger.debug("proportion zero: %f" % proportion_precursor_mzs_zero)
        if proportion_precursor_mzs_zero > MAX_PROPORTION_PRECURSORDELTAS_0:
            print("Too high a proportion of precursor mass differences (%f) are exactly 0. Defaulting to %fppm" %
                  (proportion_precursor_mzs_zero, MAX_PRECURSORDIST_PPM))
            precursor_prediction_ppm = MAX_PRECURSORDIST_PPM
        else:
            precursor_prediction_ppm = self.precursor_sigma_multiplier * precursor_sigma_ppm + abs(precursor_mu_ppm)
        fragment_prediction_th = self.frag_sigma_multiplier * frag_sigma_th + abs(frag_mu_th)
        fragment_prediction_ppm = self.frag_sigma_multiplier * frag_sigma_ppm + abs(frag_mu_ppm)

        return (precursor_sigma_ppm, frag_sigma_ppm, frag_sigma_th,
                precursor_prediction_ppm, fragment_prediction_ppm, fragment_prediction_th)


def estimate_mu_sigma(data, min_sigma):
    # estimate mu and sigma of the mixed distribution, as initial estimate for Gaussian
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
    logger.debug("model:\n%s" % mixture_model)
    logger.debug("model improvement: %f" % improvement)
    mu_fit = mixture_model.distributions[0].parameters[0]
    sigma_fit = mixture_model.distributions[0].parameters[1]
    logger.debug("fit: mean=%f, sigma=%f" % (mu_fit, sigma_fit))


    return mu_fit, sigma_fit


class SpectrumObservation(object):
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