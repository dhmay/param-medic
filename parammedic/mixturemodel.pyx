#cython: boundscheck=False
#cython: cdivision=True

"""
Code for modeling a mixed Gaussian-Uniform distribution.
Code adapted with permission from Jacob Schreiber's Pomegranate:
https://github.com/jmschrei/pomegranate

Pomegranate is far more flexible than what I need for this application
(mixed Gaussian-uniform distribution), and the structure of this code
reflects that -- it's a bit heavyweight for this use case, but that shouldn't
affect performance.

Only notable change from Jacob's code: I had trouble getting nogil functions to work
on my Mac. To avoid cross-platform compatibility issues, I made everything keep gil.
Might be a small performance hit where multiple threads are available. Emprically,
seems to perform fine.
"""

import logging
cimport numpy
import numpy
from libc.math cimport log as clog
from libc.math cimport exp as cexp
from libc.math cimport sqrt as csqrt
from libc.string cimport memset
from libc.stdlib cimport calloc
from libc.stdlib cimport free
from libc.string cimport memcpy

DEF NEGINF = float("-inf")
DEF INF = float("inf")
DEF SQRT_2_PI = 2.50662827463

logger = logging.getLogger(__name__)


## Utility methods

cpdef numpy.ndarray _check_input(X, dict keymap):
    """Check the input to make sure that it is a properly formatted array."""

    cdef numpy.ndarray X_ndarray
    if isinstance(X, numpy.ndarray) and (X.dtype == 'float64'):
        return X

    try:
        X_ndarray = numpy.array(X, dtype='float64')
    except ValueError:
        X_ndarray = numpy.empty(X.shape, dtype='float64')

        if X.ndim == 1:
            for i in range(X.shape[0]):
                X_ndarray[i] = keymap[X[i]]
        else:
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    X_ndarray[i, j] = keymap[X[i][j]]

    return X_ndarray

cdef double _log(double x) nogil:
    """
    A wrapper for the c log function, by returning negative infinity if the
    input is 0.
    """

    return clog(x) if x > 0 else NEGINF

cdef double pair_lse(double x, double y) nogil:
    """
    Perform log-sum-exp on a pair of numbers in log space..  This is calculated
    as z = log( e**x + e**y ). However, this causes underflow sometimes
    when x or y are too negative. A simplification of this is thus
    z = x + log( e**(y-x) + 1 ), where x is the greater number. If either of
    the inputs are infinity, return infinity, and if either of the inputs
    are negative infinity, then simply return the other input.
    """

    if x == INF or y == INF:
        return INF
    if x == NEGINF:
        return y
    if y == NEGINF:
        return x
    if x > y:
        return x + clog(cexp(y - x) + 1)
    return y + clog(cexp(x - y) + 1)

def weight_set(items, weights):
    """Converts both items and weights to appropriate numpy arrays.

    Convert the items into a numpy array with 64-bit floats, and the weight
    array to the same. If no weights are passed in, then return a numpy array
    with uniform weights.
    """
    items = numpy.array(items, dtype=numpy.float64)
    if weights is None:
        # Weight everything 1 if no weights specified
        weights = numpy.ones(items.shape[0], dtype=numpy.float64)
    else:
        # Force whatever we have to be a Numpy array
        weights = numpy.array(weights, dtype=numpy.float64)

    return items, weights

cdef class Model(object):
    """The abstract building block for all distributions."""

    def __cinit__(self):
        self.name = "Model"
        self.frozen = False
        self.d = 0

    def get_params(self, *args, **kwargs):
        return self.__getstate__()

    def set_params(self, state):
        self.__setstate__(state)

    def log_probability(self, double symbol):
        """Return the log probability of the given symbol under this distribution.

        Parameters
        ----------
        symbol : double
                The symbol to calculate the log probability of (overriden for
                DiscreteDistributions)

        Returns
        -------
        logp : double
                The log probability of that point under the distribution.
        """
        return NotImplementedError

    #damonmay removing nogil
    cdef void _v_log_probability(self, double*symbol, double*log_probability, int n):
        pass

    # damonmay removing nogil
    cdef double _summarize(self, double*items, double*weights, SIZE_t n):
        pass

    # damonmay removing nogil
    cdef double _log_probability(self, double symbol):
        return NEGINF

    def summarize(self, items, weights=None):
        """Summarize a batch of data into sufficient statistics for a later update.


        Parameters
        ----------
        items : array-like, shape (n_samples, n_dimensions)
            This is the data to train on. Each row is a sample, and each column
            is a dimension to train on. For univariate distributions an array
            is used, while for multivariate distributions a 2d matrix is used.

        weights : array-like, shape (n_samples,), optional
            The initial weights of each sample in the matrix. If nothing is
            passed in then each sample is assumed to be the same weight.
            Default is None.

        Returns
        -------
        None
        """

        return NotImplementedError

cdef class Distribution(Model):
    """A probability distribution.
    Represents a probability distribution over the defined support. This is
    the base class which must be subclassed to specific probability
    distributions. All distributions have the below methods exposed.
    Parameters
    ----------
    Varies on distribution.
    Attributes
    ----------
    name : str
        The name of the type of distributioon.
    summaries : list
        Sufficient statistics to store the update.
    frozen : bool
        Whether or not the distribution will be updated during training.
    d : int
        The dimensionality of the data. Univariate distributions are all
        1, while multivariate distributions are > 1.
    """

    def __cinit__(self):
        self.name = "Distribution"
        self.frozen = False
        self.summaries = []
        self.d = 1

    def marginal(self, *args, **kwargs):
        """Return the marginal of the distribution.
        Parameters
        ----------
        *args : optional
            Arguments to pass in to specific distributions
        **kwargs : optional
            Keyword arguments to pass in to specific distributions
        Returns
        -------
        distribution : Distribution
            The marginal distribution. If this is a multivariate distribution
            then this method is filled in. Otherwise returns self.
        """

        return self

    def log_probability(self, double symbol):
        """Return the log probability of the given symbol under this distribution.
        Parameters
        ----------
        symbol : double
            The symbol to calculate the log probability of (overriden for
            DiscreteDistributions)
        Returns
        -------
        logp : double
            The log probability of that point under the distribution.
        """

        cdef double logp
        logp = self._log_probability(symbol)
        return logp

cdef class UniformDistribution(Distribution):
    """A uniform distribution between two values."""

    property parameters:
        def __get__(self):
            return [self.start, self.end]
        def __set__(self, parameters):
            self.start, self.end = parameters

    def __cinit__(UniformDistribution self, double start, double end, bint frozen=False):
        """
        Make a new Uniform distribution over floats between start and end,
        inclusive. Start and end must not be equal.
        """

        # Store the parameters
        self.start = start
        self.end = end
        self.summaries = [INF, NEGINF]
        self.name = "UniformDistribution"
        self.frozen = frozen
        self.logp = -_log(end - start)

    #damonmay removing nogil
    cdef double _log_probability(self, double symbol):
        cdef double logp
        self._v_log_probability(&symbol, &logp, 1)
        return logp

    #damonmay removing nogil
    cdef void _v_log_probability(self, double*symbol, double*log_probability, int n):
        cdef int i
        for i in range(n):
            if symbol[i] >= self.start and symbol[i] <= self.end:
                log_probability[i] = self.logp
            else:
                log_probability[i] = NEGINF

    def fit(self, items, weights=None, inertia=0.0):
        """
        Set the parameters of this Distribution to maximize the likelihood of
        the given sample. Items holds some sort of sequence. If weights is
        specified, it holds a sequence of value to weight each item by.
        """

        if self.frozen:
            return

        self.summarize(items, weights)
        self.from_summaries(inertia)

    def summarize(self, items, weights=None):
        """
        Take in a series of items and their weights and reduce it down to a
        summary statistic to be used in training later.
        """

        items, weights = weight_set(items, weights)
        if weights.sum() <= 0:
            return

        cdef double*items_p = <double*> (<numpy.ndarray> items).data
        cdef double*weights_p = <double*> (<numpy.ndarray> weights).data
        cdef SIZE_t n = items.shape[0]

        # damonmay removing with nogil:
        #with nogil:
        self._summarize(items_p, weights_p, n)

    # damonmay removing nogil
    cdef double _summarize(self, double*items, double*weights, SIZE_t n):
        cdef double minimum = INF, maximum = NEGINF
        cdef int i

        for i in range(n):
            if weights[i] > 0:
                if items[i] < minimum:
                    minimum = items[i]
                if items[i] > maximum:
                    maximum = items[i]

        #damonmay removing with gil:
        #with gil:
        if maximum > self.summaries[1]:
            self.summaries[1] = maximum
        if minimum < self.summaries[0]:
            self.summaries[0] = minimum

    def from_summaries(self, inertia=0.0):
        """
        Takes in a series of summaries, consisting of the minimum and maximum
        of a sample, and determine the global minimum and maximum.
        """

        # If the distribution is frozen, don't bother with any calculation
        if self.frozen == True:
            return

        minimum, maximum = self.summaries
        self.start = minimum * (1 - inertia) + self.start * inertia
        self.end = maximum * (1 - inertia) + self.end * inertia
        self.logp = -_log(self.end - self.start)

        self.summaries = [INF, NEGINF]

    def clear_summaries(self):
        """Clear the summary statistics stored in the object."""

        self.summaries = [INF, NEGINF]

cdef class NormalDistribution(Distribution):
    """A normal distribution."""

    property parameters:
        def __get__(self):
            return [self.mu, self.sigma]
        def __set__(self, parameters):
            self.mu, self.sigma = parameters

    def __cinit__(self, mean, std, frozen=False, min_std=None):
        """
        Make a new Normal distribution with the given mean mean and standard
        deviation std.
        """

        self.mu = mean
        self.sigma = std
        self.name = "NormalDistribution"
        self.frozen = frozen
        self.summaries = [0, 0, 0]
        self.log_sigma_sqrt_2_pi = -_log(std * SQRT_2_PI)
        self.two_sigma_squared = 2 * std ** 2
        self.min_std = min_std

    # damonmay removing nogil
    cdef double _log_probability(self, double symbol):
        cdef double logp
        self._v_log_probability(&symbol, &logp, 1)
        return logp

    # damonmay removing nogil
    cdef void _v_log_probability(self, double*symbol, double*log_probability, int n):
        cdef int i
        for i in range(n):
            log_probability[i] = self.log_sigma_sqrt_2_pi - ((symbol[i] - self.mu) ** 2) / \
                                                            self.two_sigma_squared

    def fit(self, items, weights=None, inertia=0.0, min_std=1e-5):
        """
        Set the parameters of this Distribution to maximize the likelihood of
        the given sample. Items holds some sort of sequence. If weights is
        specified, it holds a sequence of value to weight each item by.
        """

        if self.frozen:
            return

        self.summarize(items, weights)
        self.from_summaries(inertia, min_std)

    # damonmay removing nogil
    cdef double _summarize(self, double*items, double*weights, SIZE_t n):
        cdef SIZE_t i
        cdef double x_sum = 0.0, x2_sum = 0.0, w_sum = 0.0

        for i in range(n):
            w_sum += weights[i]
            x_sum += weights[i] * items[i]
            x2_sum += weights[i] * items[i] * items[i]

        #damonmay removing with gil:
        #with gil:
        self.summaries[0] += w_sum
        self.summaries[1] += x_sum
        self.summaries[2] += x2_sum

    def summarize(self, items, weights=None):
        """
        Take in a series of items and their weights and reduce it down to a
        summary statistic to be used in training later.
        """

        items, weights = weight_set(items, weights)
        if weights.sum() <= 0:
            return

        cdef double*items_p = <double*> (<numpy.ndarray> items).data
        cdef double*weights_p = <double*> (<numpy.ndarray> weights).data
        cdef SIZE_t n = items.shape[0]

        #damonmay removing with nogil:
        #with nogil:
        self._summarize(items_p, weights_p, n)

    def from_summaries(self, inertia=0.0, min_std=0.01):
        """
        Takes in a series of summaries, represented as a mean, a variance, and
        a weight, and updates the underlying distribution. Notes on how to do
        this for a Gaussian distribution were taken from here:
        http://math.stackexchange.com/questions/453113/how-to-merge-two-gaussians
        """

        min_std = self.min_std if self.min_std is not None else min_std

        # If no summaries stored or the summary is frozen, don't do anything.
        if self.summaries[0] == 0 or self.frozen == True:
            return

        mu = self.summaries[1] / self.summaries[0]
        var = self.summaries[2] / self.summaries[0] - self.summaries[1] ** 2.0 / self.summaries[0] ** 2.0

        sigma = csqrt(var)
        if sigma < min_std:
            sigma = min_std

        self.mu = self.mu * inertia + mu * (1 - inertia)
        self.sigma = self.sigma * inertia + sigma * (1 - inertia)
        self.summaries = [0, 0, 0]
        self.log_sigma_sqrt_2_pi = -_log(sigma * SQRT_2_PI)
        self.two_sigma_squared = 2 * sigma ** 2

    def clear_summaries(self):
        """Clear the summary statistics stored in the object."""

        self.summaries = [0, 0, 0]

cdef class Kmeans(Model):
    """A kmeans model.

    Kmeans is not a probabilistic model, but it is used in the kmeans++
    initialization for GMMs. In essence, a point is selected as the center
    for one component and then remaining points are selected

    Parameters
    ----------
    k : int
        The number of centroids.

    centroids : numpy.ndarray or None, optional
        The centroids to be used for this kmeans clusterer, if known ahead of
        time. These centroids can either be refined on future data, or used
        for predictions in the future. If None, then it will begin clustering
        on the first batch of data that it sees. Default is None.

    Attributes
    ----------
    k : int
        The number of centroids

    centroids : array-like, shape (k, n_dim)
        The means of the centroid points.
    """

    cdef public int k
    cdef public numpy.ndarray centroids
    cdef double*centroids_ptr
    cdef double*summary_sizes
    cdef double*summary_weights

    def __init__(self, k, centroids=None):
        self.k = k
        self.d = 0

        if centroids is not None:
            self.centroids = numpy.array(centroids, dtype='float64')
            self.centroids_ptr = <double*> self.centroids.data
            self.d = self.centroids.shape[1]

    def __str__(self):
        return self.to_json()

    def __repr__(self):
        return self.to_json()

    def __dealloc__(self):
        free(self.summary_sizes)
        free(self.summary_weights)

    def predict(self, X):
        """Predict nearest centroid for each point.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_dim)
            The data to fit to.

        Returns
        -------
        y : array-like, shape (n_samples,)
            The index of the nearest centroid.
        """

        X = numpy.array(X, dtype='float64')
        cdef double*X_ptr = <double*> (<numpy.ndarray> X).data
        cdef int n = len(X)

        cdef numpy.ndarray y = numpy.zeros(n, dtype='int32')
        cdef int*y_ptr = <int*> y.data

        with nogil:
            self._predict(X_ptr, y_ptr, n)

        return y

    cdef void _predict(self, double*X, int*y, int n) nogil:
        cdef int i, j, l, k = self.k, d = self.d
        cdef double dist, min_dist

        for i in range(n):
            min_dist = INF

            for j in range(k):
                dist = 0.0

                for l in range(d):
                    dist += (self.centroids_ptr[j * d + l] - X[i * d + l]) ** 2.0

                if dist < min_dist:
                    min_dist = dist
                    y[i] = j

    def fit(self, X, weights=None, inertia=0.0, stop_threshold=1e-3, max_iterations=1e3,
            verbose=False):
        """Fit the model to the data using k centroids.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_dim)
            The data to fit to.

        weights : array-like, shape (n_samples,), optional
            The initial weights of each sample in the matrix. If nothing is
            passed in then each sample is assumed to be the same weight.
            Default is None.

        inertia : double, optional
            The weight of the previous parameters of the model. The new
            parameters will roughly be old_param*inertia + new_param*(1-inertia),
            so an inertia of 0 means ignore the old parameters, whereas an
            inertia of 1 means ignore the new parameters. Default is 0.0.

        stop_threshold : double, optional, positive
            The threshold at which EM will terminate for the improvement of
            the model. If the model does not improve its fit of the data by
            a log probability of 0.1 then terminate. Default is 0.1.

        max_iterations : int, optional
            The maximum number of iterations to run for. Default is 1e3.

        verbose : bool, optional
            Whether or not to print out improvement information over iterations.
            Default is False.

        Returns
        -------
        None
        """

        initial_log_probability_sum = NEGINF
        iteration, improvement = 0, INF

        while improvement > stop_threshold and iteration < max_iterations + 1:
            self.from_summaries(inertia)
            log_probability_sum = self.summarize(X, weights)

            if iteration == 0:
                initial_log_probability_sum = log_probability_sum
            else:
                improvement = log_probability_sum - last_log_probability_sum

                if verbose:
                    print("Improvement: {}".format(improvement))

            iteration += 1
            last_log_probability_sum = log_probability_sum

        self.clear_summaries()

        if verbose:
            print("Total Improvement: {}".format(last_log_probability_sum - initial_log_probability_sum))

        return last_log_probability_sum - initial_log_probability_sum

    def summarize(self, X, weights=None):
        """Summarize the points into sufficient statistics for a future update.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_dim)
            The data to fit to.

        weights : array-like, shape (n_samples,), optional
            The initial weights of each sample in the matrix. If nothing is
            passed in then each sample is assumed to be the same weight.
            Default is None.

        Returns
        -------
        dist : double
            The negative total euclidean distance between each point and its nearest
            centroid. This is not a probabilitity, and the negative is returned to
            fit in with the idea of large negative numbers being worse than smaller
            negative numbers, such as with log probabilities.
        """

        cdef numpy.ndarray X_ndarray = numpy.array(X, dtype='float64')
        cdef numpy.ndarray weights_ndarray
        cdef int i, j, n = X_ndarray.shape[0], d = X_ndarray.shape[1]
        cdef double*X_ptr = <double*> X_ndarray.data
        cdef double dist

        if weights is None:
            weights_ndarray = numpy.ones(n, dtype='float64')
        else:
            weights_ndarray = numpy.array(weights, dtype='float64')

        cdef double*weights_ptr = <double*> weights_ndarray.data

        if self.d == 0:
            self.d = d
            self.centroids = numpy.zeros((self.k, d))
            self.centroids_ptr = <double*> self.centroids.data
            self.summary_sizes = <double*> calloc(self.k, sizeof(double))
            self.summary_weights = <double*> calloc(self.k * d, sizeof(double))

            memcpy(self.centroids_ptr, X_ptr, self.k * d * sizeof(double))
            memset(self.summary_sizes, 0, self.k * sizeof(double))
            memset(self.summary_weights, 0, self.k * d * sizeof(double))

        #damonmay removing with nogil:
        #with nogil:
        dist = self._summarize(X_ptr, weights_ptr, n)

        return dist

    # damonmay removing nogil
    cdef double _summarize(self, double*X, double*weights, int n):
        cdef int i, j, l, y, k = self.k, d = self.d
        cdef double min_dist, dist, total_dist = 0.0
        cdef double*summary_sizes = <double*> calloc(k, sizeof(double))
        cdef double*summary_weights = <double*> calloc(k * d, sizeof(double))
        memset(summary_sizes, 0, k * sizeof(double))
        memset(summary_weights, 0, k * d * sizeof(double))

        for i in range(n):
            min_dist = INF

            for j in range(k):
                dist = 0.0

                for l in range(d):
                    dist += (self.centroids_ptr[j * d + l] - X[i * d + l]) ** 2.0

                if dist < min_dist:
                    min_dist = dist
                    y = j

            total_dist -= min_dist
            summary_sizes[y] += weights[i]

            for l in range(d):
                summary_weights[y * d + l] += X[i * d + l] * weights[i]

        #damonmay removing with gil
        #with gil:
        for j in range(k):
            self.summary_sizes[j] += summary_sizes[j]

            for l in range(d):
                self.summary_weights[j * d + l] += summary_weights[j * d + l]

        free(summary_sizes)
        free(summary_weights)
        return total_dist

    def from_summaries(self, double inertia=0.0):
        """Fit the model to the sufficient statistics.

        Parameters
        ----------
        inertia : double, optional
            The weight of the previous parameters of the model. The new
            parameters will roughly be old_param*inertia + new_param*(1-inertia),
            so an inertia of 0 means ignore the old parameters, whereas an
            inertia of 1 means ignore the new parameters. Default is 0.0.

        Returns
        -------
        None
        """

        if self.d == 0:
            return

        cdef int l, j, k = self.k, d = self.d

        with nogil:
            for j in range(k):
                for l in range(d):
                    self.centroids_ptr[j * d + l] = self.centroids_ptr[j * d + l] * inertia + \
                                                    (self.summary_weights[j * d + l] / self.summary_sizes[j] * (
                                                    1 - inertia))

            memset(self.summary_sizes, 0, self.k * sizeof(int))
            memset(self.summary_weights, 0, self.k * self.d * sizeof(int))

    def clear_summaries(self):
        """Clear the stored sufficient statistics.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        memset(self.summary_sizes, 0, self.k * sizeof(int))
        memset(self.summary_weights, 0, self.k * self.d * sizeof(int))


cdef class GeneralMixtureModel(Model):
    """
    General mixture model. This is overkill for my two-distribution use case, but the extra
    complexity doesn't introduce any significant overhead.
    """

    cdef public numpy.ndarray distributions
    cdef object distribution_callable
    cdef public numpy.ndarray weights
    cdef numpy.ndarray summaries_ndarray
    cdef void** distributions_ptr
    cdef double*weights_ptr
    cdef double*summaries_ptr
    cdef dict keymap
    cdef int n

    def __init__(self, distributions, weights=None, n_components=None):
        self.d = 0

        if callable(distributions):
            self.n = n_components
            self.distribution_callable = distributions
        else:
            if len(distributions) < 2:
                raise ValueError("must have at least two distributions for general mixture models")

            for dist in distributions:
                if callable(dist):
                    raise TypeError("must have initialized distributions in list")
                elif self.d == 0:
                    self.d = dist.d
                elif self.d != dist.d:
                    raise TypeError("mis-matching dimensions between distributions in list")

            if weights is None:
                weights = numpy.ones_like(distributions, dtype=float) / len(distributions)
            else:
                weights = numpy.asarray(weights) / weights.sum()

            self.weights = numpy.log(weights)
            self.weights_ptr = <double*> self.weights.data

            self.distributions = numpy.array(distributions)
            self.distributions_ptr = <void**> self.distributions.data

            self.summaries_ndarray = numpy.zeros_like(weights, dtype='float64')
            self.summaries_ptr = <double*> self.summaries_ndarray.data

            self.n = len(distributions)

    def fit(self, X, weights=None, inertia=0.0, stop_threshold=0.1,
            max_iterations=1e8, verbose=False):
        """Fit the model to new data using EM.

        This method fits the components of the model to new data using the EM
        method. It will iterate until either max iterations has been reached,
        or the stop threshold has been passed.

        This is a sklearn wrapper for train method.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_dimensions)
            This is the data to train on. Each row is a sample, and each column
            is a dimension to train on.

        weights : array-like, shape (n_samples,), optional
            The initial weights of each sample in the matrix. If nothing is
            passed in then each sample is assumed to be the same weight.
            Default is None.

        inertia : double, optional
            The weight of the previous parameters of the model. The new
            parameters will roughly be old_param*inertia + new_param*(1-inertia),
            so an inertia of 0 means ignore the old parameters, whereas an
            inertia of 1 means ignore the new parameters. Default is 0.0.

        stop_threshold : double, optional, positive
            The threshold at which EM will terminate for the improvement of
            the model. If the model does not improve its fit of the data by
            a log probability of 0.1 then terminate. Default is 0.1.

        max_iterations : int, optional, positive
            The maximum number of iterations to run EM for. If this limit is
            hit then it will terminate training, regardless of how well the
            model is improving per iteration. Default is 1e8.

        verbose : bool, optional
            Whether or not to print out improvement information over iterations.
            Default is False.

        Returns
        -------
        improvement : double
            The total improvement in log probability P(D|M)
        """

        initial_log_probability_sum = NEGINF
        iteration, improvement = 0, INF

        if weights is None:
            weights = numpy.ones(len(X), dtype='float64')
        else:
            weights = numpy.array(weights, dtype='float64')
        while improvement > stop_threshold and iteration < max_iterations + 1:
            self.from_summaries(inertia)
            log_probability_sum = self.summarize(X, weights)

            if iteration == 0:
                initial_log_probability_sum = log_probability_sum
            else:
                improvement = log_probability_sum - last_log_probability_sum

                if verbose:
                    print("Improvement: {}".format(improvement))

            iteration += 1
            last_log_probability_sum = log_probability_sum


        self.clear_summaries()

        if verbose:
            print("Total Improvement: {}".format(last_log_probability_sum - initial_log_probability_sum))

        return last_log_probability_sum - initial_log_probability_sum

    def summarize(self, X, weights=None):
        """Summarize a batch of data and store sufficient statistics.

        This will run the expectation step of EM and store sufficient
        statistics in the appropriate distribution objects. The summarization
        can be thought of as a chunk of the E step, and the from_summaries
        method as the M step.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_dimensions)
            This is the data to train on. Each row is a sample, and each column
            is a dimension to train on.

        weights : array-like, shape (n_samples,), optional
            The initial weights of each sample in the matrix. If nothing is
            passed in then each sample is assumed to be the same weight.
            Default is None.

        Returns
        -------
        logp : double
            The log probability of the data given the current model. This is
            used to speed up EM.
        """

        cdef int i, n, d
        cdef numpy.ndarray X_ndarray
        cdef numpy.ndarray weights_ndarray
        cdef double log_probability

        if self.d == 1:
            n, d = X.shape[0], 1
        elif self.d > 1 and X.ndim == 1:
            n, d = 1, len(X)
        else:
            n, d = X.shape

        if weights is None:
            weights_ndarray = numpy.ones(n, dtype='float64')
        else:
            weights_ndarray = numpy.array(weights, dtype='float64')

        # If not initialized then we need to do kmeans initialization.
        if self.d == 0:
            X_ndarray = _check_input(X, self.keymap)
            kmeans = Kmeans(self.n)
            kmeans.fit(X_ndarray, max_iterations=1)
            y = kmeans.predict(X_ndarray)

            distributions = [self.distribution_callable.from_samples(X_ndarray[y == i]) for i in range(self.n)]
            self.d = distributions[0].d

            self.distributions = numpy.array(distributions)
            self.distributions_ptr = <void**> self.distributions.data

            self.weights = numpy.log(numpy.ones(self.n, dtype='float64') / self.n)
            self.weights_ptr = <double*> self.weights.data

            self.summaries_ndarray = numpy.zeros_like(self.weights, dtype='float64')
            self.summaries_ptr = <double*> self.summaries_ndarray.data

        cdef double*X_ptr
        cdef double*weights_ptr = <double*> weights_ndarray.data

        X_ndarray = _check_input(X, self.keymap)
        X_ptr = <double*> X_ndarray.data

        # damonmay removing with nogil:
        #with nogil:
        log_probability = self._summarize(X_ptr, weights_ptr, n)

        return log_probability

    # damonmay removing nogil
    cdef double _summarize(self, double*X, double*weights, int n):
        cdef double*r = <double*> calloc(self.n * n, sizeof(double))
        cdef double*summaries = <double*> calloc(self.n, sizeof(double))
        cdef int i, j
        cdef double total, logp, log_probability_sum = 0.0

        memset(summaries, 0, self.n * sizeof(double))
        cdef double tic

        for j in range(self.n):
            (<Model> self.distributions_ptr[j])._v_log_probability(X, r + j * n, n)

        for i in range(n):
            total = NEGINF

            for j in range(self.n):
                r[j * n + i] += self.weights_ptr[j]
                total = pair_lse(total, r[j * n + i])

            for j in range(self.n):
                r[j * n + i] = cexp(r[j * n + i] - total) * weights[i]
                summaries[j] += r[j * n + i]

            log_probability_sum += total * weights[i]

        for j in range(self.n):
            (<Model> self.distributions_ptr[j])._summarize(X, r + j * n, n)

        # damonmay removing with gil:
        #with gil:
        for j in range(self.n):
            self.summaries_ptr[j] += summaries[j]

        free(r)
        free(summaries)
        return log_probability_sum

    def from_summaries(self, inertia=0.0, **kwargs):
        """Fit the model to the collected sufficient statistics.

        Fit the parameters of the model to the sufficient statistics gathered
        during the summarize calls. This should return an exact update.

        Parameters
        ----------
        inertia : double, optional
            The weight of the previous parameters of the model. The new
            parameters will roughly be old_param*inertia + new_param*(1-inertia),
            so an inertia of 0 means ignore the old parameters, whereas an
            inertia of 1 means ignore the new parameters. Default is 0.0.

        Returns
        -------
        None
        """

        if self.d == 0 or self.summaries_ndarray.sum() == 0:
            return

        self.summaries_ndarray /= self.summaries_ndarray.sum()
        for i, distribution in enumerate(self.distributions):
            distribution.from_summaries(inertia, **kwargs)
            self.weights[i] = _log(self.summaries_ndarray[i])
            self.summaries_ndarray[i] = 0.

    def clear_summaries(self):
        """Clear the summary statistics stored in the object.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        self.summaries_ndarray *= 0
        for distribution in self.distributions:
            distribution.clear_summaries()
