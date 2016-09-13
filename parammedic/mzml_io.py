#!/usr/bin/env python

"""
mzML reading
"""

import logging
from pyteomics import mzml
from parammedic import errorcalc


__author__ = "Damon May"
__copyright__ = "Copyright (c) 2016 Damon May"
__license__ = ""
__version__ = ""


log = logging.getLogger(__name__)

def retrieve_scans(mzml_file, scan_numbers):
    """
    retrieve scans from a preindexed mzML file.
    :param mzml_file:
    :param scan_numbers
    :return:
    """
    with mzml.PreIndexedMzML(mzml_file) as reader:
        for scan_number in scan_numbers:
            spectrum = read_scan(reader.get_by_id("controllerType=0 controllerNumber=1 scan=%d" % scan_number))
            yield spectrum


def read_ms2_scans(mzml_file):
    return read_scans(mzml_file, [2])


def read_scans(mzml_file, ms_levels=(1, 2)):
    """
    yields all spectra from an mzML file with level in ms_levels, or
    all processable scans if ms_levels not specified
    :param mzml_file:
    :param ms_levels:
    :param min_pprophet:
    :return:
    """
    with mzml.MzML(mzml_file) as reader:
        for scan in reader:
            if scan['ms level'] in ms_levels:
                yield read_scan(scan)


def read_scan(scan):
    """

    :param scan:
    :return:
    """
    # see below for the byzantine intricacies of the scan object
    ms_level = scan['ms level']
    scan_number = int(scan['id'][scan['id'].index('scan=') + 5:])
    mz_array = scan['m/z array']
    intensity_array = scan['intensity array']
    retention_time = scan['scanList']['scan'][0]['scan start time']
    if ms_level == 1:
        return errorcalc.MSSpectrum(scan_number, retention_time,
                                  mz_array,
                                  intensity_array)
    elif ms_level == 2:
        precursor_selected_ion_map = scan['precursorList']['precursor'][0]['selectedIonList']['selectedIon'][0]
        precursor_mz = precursor_selected_ion_map['selected ion m/z']
        charge = precursor_selected_ion_map['charge state']
        return errorcalc.MS2Spectrum(scan_number, retention_time,
                                   mz_array,
                                   intensity_array,
                                   precursor_mz, charge)
    else:
        raise ValueError("Unhandleable scan level %d" % ms_level)


# example pyteomics.mzml ms2 scan object:
#
#{'MSn spectrum': '',
# 'base peak intensity': 97570.0,
# 'base peak m/z': 288.922,
# 'centroid spectrum': '',
# 'count': 2,
# 'defaultArrayLength': 11,
# 'highest observed m/z': 2325.92,
# 'id': 'controllerType=0 controllerNumber=1 scan=823',
# 'index': 0,
# 'intensity array': array([  8935.11914062,  16606.33789062,   9164.421875  ,   4691.23339844,
#         97570.046875  ,  10380.35546875,   4243.56591797,  26311.13671875,
#          4218.36376953,   4853.49853516,   5048.49267578], dtype=float32),
# 'lowest observed m/z': 208.275,
# 'm/z array': array([  208.27484131,   208.29246521,   208.3107605 ,   220.93244934,
#          288.92218018,   356.91146851,   407.6151123 ,   424.89764404,
#          512.6675415 ,  1729.26879883,  2325.91601562]),
# 'ms level': 2,
# 'positive scan': '',
# 'precursorList': {'count': 1,
#  'precursor': [{'activation': {'beam-type collision-induced dissociation': ''},
#    'selectedIonList': {'count': 1,
#     'selectedIon': [{'charge state': 3.0,
#       'peak intensity': 121445.0,
#       'selected ion m/z': 764.83}]},
#    'spectrumRef': 'controllerType=0 controllerNumber=1 scan=822'}]},
# 'scanList': {'count': 1,
#  'no combination': '',
#  'scan': [{'instrumentConfigurationRef': 'IC1', 'scan start time': 190.209}]},
# 'total ion current': 227645.0}


