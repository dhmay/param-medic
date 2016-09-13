cimport numpy

ctypedef numpy.npy_intp SIZE_t

cdef class Model(object):
    cdef public str name
    cdef public int d
    cdef public bint frozen
    cdef public str model

    # damonmay removing nogil
    cdef double _log_probability(self, double symbol)
    # damonmay removing nogil
    cdef double _summarize(self, double*items, double*weights, SIZE_t n)
    # damonmay removing nogil
    cdef void _v_log_probability( self, double* symbol, double* log_probability, int n )

cdef class Distribution(Model):
    cdef public list summaries

cdef class UniformDistribution(Distribution):
    cdef double start, end, logp

cdef class NormalDistribution(Distribution):
    cdef double mu, sigma, two_sigma_squared, log_sigma_sqrt_2_pi
    cdef object min_std
