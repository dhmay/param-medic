#!/usr/bin/env python

"""
mzML reading
"""

import logging
from pyteomics import mzml

import parammedic.util


__author__ = "Damon May"
__copyright__ = "Copyright (c) 2016 Damon May"
__license__ = ""
__version__ = ""


logger = logging.getLogger(__name__)


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
                # ignore this scan if we get a ValueError.
                # ValueError is only raised if we can't infer charge.
                # If we still have enough scans where we could infer charge, OK to
                # ignore these.
                try:
                    yield read_scan(scan)
                except ValueError as e:
                    logger.debug("Warning! Failed to read scan: %s" % e)


def read_scan(scan):
    """
    Read a single scan into our representation
    :param scan:
    :return:
    """
    # see below for the byzantine intricacies of the scan object
    ms_level = scan['ms level']
    id_field = scan['id']
    if 'scan=' in id_field:
        scan_number = int(id_field[id_field.index('scan=') + len('scan='):])
    elif 'experiment=' in id_field:
        scan_number = int(id_field[id_field.index('experiment=') + len('experiment='):])
    else:
        raise ValueError('cannot parse scan number from id attribute: {}'.format(id_field))

    mz_array = scan['m/z array']
    intensity_array = scan['intensity array']
    retention_time = scan['scanList']['scan'][0]['scan start time']

    if ms_level == 1:
        return parammedic.util.MSSpectrum(scan_number, retention_time,
                                          mz_array,
                                          intensity_array)
    elif ms_level == 2:
        precursor_selected_ion_map = scan['precursorList']['precursor'][0]['selectedIonList']['selectedIon'][0]
        precursor_mz = precursor_selected_ion_map['selected ion m/z']
        if 'charge state' in precursor_selected_ion_map:
            charge = precursor_selected_ion_map['charge state']
        elif 'possible charge state' in precursor_selected_ion_map:
            charge = precursor_selected_ion_map['possible charge state']
        else:
            raise ValueError("Could not find charge for scan {}. Fields available: {}".format(
                scan_number, precursor_selected_ion_map.keys()))

        # damonmay adding for activation type histogram.
        # a scan looks like this:
            # {'count': 2, 'index': 3, 'highest observed m/z': 663.433410644531,
            # 'm/z array': array([ 101.05975342,  105.96199036,  109.10111237,  111.11677551, ...])}
            # , dtype=float32), 'precursorList': {'count': 1, 'precursor':
            # [{'selectedIonList': {'count': 1, 'selectedIon': [{'charge state': 2.0,
            # 'peak intensity': 6005243.5, 'selected ion m/z': 337.715426720255}]},
            # 'activation': {'beam-type collision-induced dissociation': '', 'collision energy': 32.0},
            # 'spectrumRef': 'controllerType=0 controllerNumber=1 scan=3',
        # etc
        activation_type = None
        if 'precursorList' in scan:
            preclist = scan['precursorList']
            if 'count' in preclist and preclist['count'] == 1 and \
                    'precursor' in preclist:
                precursor = preclist['precursor'][0]
                if 'activation' in precursor:
                    # precursor['activation'] is a weird dict that looks like this:
                    # 'activation': {'beam-type collision-induced dissociation': '', 'collision energy': 25.0}
                    activation_type_dict = precursor['activation']
                    # this method if figuring out the activation type seems very brittle.
                    for key in activation_type_dict:
                        if key != 'collision energy':
                            activation_type = key
                            break

        return parammedic.util.MS2Spectrum(scan_number, retention_time,
                                           mz_array,
                                           intensity_array,
                                           precursor_mz, charge,
                                           activation_type=activation_type)
    else:
        logger.debug("Unhandleable scan level %d" % ms_level)


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


