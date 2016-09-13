#!/usr/bin/env python

"""
.ms2 reading.
Some bits grabbed with permission from Alex Hu
"""

import logging
from parammedic import errorcalc

__author__ = "Damon May"
__copyright__ = "Copyright (c) 2016 Damon May"
__license__ = ""
__version__ = ""

logger = logging.getLogger(__name__)

# mass of a hydrogen atom
HYDROGEN_MASS = 1.00794


def retrieve_scans(ms2_file, scan_numbers, precursor_from_zline=True, should_calc_zs_mz_diffs=False):
    """
    retrieve only the scans in the scan_number list
    :param ms2_file:
    :param scan_numbers:
    :param precursor_from_zline:
    :param should_calc_zs_mz_diffs:
    :return:
    """
    for scan in read_scans(ms2_file):
        if scan.scan_number in scan_numbers:
            yield scan


def read_ms2_scans(ms2_file, precursor_from_zline=True, should_calc_zs_mz_diffs=False):
    """
    Silly cover method, because ms2 files only contain ms2 scans. For consistency with mzml_io
    :param ms2_file:
    :param precursor_from_zline:
    :param should_calc_zs_mz_diffs:
    :return:
    """
    return read_scans(ms2_file, precursor_from_zline=precursor_from_zline,
                      should_calc_zs_mz_diffs=should_calc_zs_mz_diffs)


def read_scans(ms2_file, precursor_from_zline=True, should_calc_zs_mz_diffs=False,
               require_rt=False):
    """
    yield all scans in the file at ms2_filepath
    :param ms2_file:
    :param precursor_from_zline:
    :param should_calc_zs_mz_diffs:
    :param require_rt:
    :return:
    """

    # store the values we care about for the current scan
    precursor_mz = None
    scan_number = None
    retention_time = None
    charge = None
    fragment_mzs = []
    fragment_intensities = []

    line_number = 0

    n_yielded = 0

    # differences in m/z between that on the s-line and that calculated from the z-line
    zline_sline_precursor_deltas = []
    zline_sline_masses = []

    for line in ms2_file:
        line_number += 1
        line = line.rstrip()
        chunks = line.split()

        # ignore these types of lines
        if ((chunks[0] == "H") or
            (chunks[0] == "D")):
            continue

        elif chunks[0] == "S":
            logger.debug("S line")
            if len(chunks) != 4:
                raise ValueError("Misformatted line %d.\n%s\n" % (line_number, line))
            # this begins a new scan, so start by writing the old one
            if scan_number and (retention_time or not require_rt) and fragment_mzs and fragment_intensities \
                    and precursor_mz and charge:
                if not retention_time:
                    # if we get here, rt is allowed to be missing, so set it to be 0.0
                    retention_time = 0.0
                # sometimes, a Z line will have a 0 charge. Punt on those
                if charge is not None and charge > 0:
                    logger.debug("0 charge!")
                    yield errorcalc.MS2Spectrum(scan_number,
                                                retention_time,
                                                fragment_mzs,
                                                fragment_intensities,
                                                precursor_mz, charge)
                    n_yielded += 1
                    logger.debug("Yielded #%d" % n_yielded)
                # zero out everything not on this line so that we know if we got it for the next scan
                charge = None
                fragment_mzs = []
                fragment_intensities = []

            else:
                logger.debug("Incomplete scan!")

            in_preamble = False
            precursor_mz = float(chunks[3])
            scan_number = int(chunks[1])

        elif chunks[0] == "I":
            if chunks[1] == "RTime" or chunks[1] == "RetTime":
                retention_time = float(chunks[2])

        elif chunks[0] == "Z":
            logger.debug("Z line")
            if len(chunks) != 3:
                raise ValueError("Misformatted Z line:\n%s\n" % line)
            charge = int(chunks[1])
            z_precursor_mplush = float(chunks[2])
            
            if charge != 0:
                zline_precursor_mz = (z_precursor_mplush - HYDROGEN_MASS) / charge + HYDROGEN_MASS
            if should_calc_zs_mz_diffs:
                if abs(zline_precursor_mz - precursor_mz) > 0.5:
                    pass
                diff_mod = abs((zline_precursor_mz - precursor_mz) * charge) % 1.000495
                while diff_mod > 1.000495 / 2:
                    diff_mod -= 1.000495
                zline_sline_precursor_deltas.append((zline_precursor_mz - precursor_mz) * charge)
                zline_sline_masses.append(precursor_mz * charge)
            if precursor_from_zline:
                precursor_mz = zline_precursor_mz
        # must be a peak line or junk
        elif len(chunks) == 4 or len(chunks) == 2:
            fragment_mzs.append(float(chunks[0]))
            fragment_intensities.append(float(chunks[1]))
        # not a recognized line type. Barf.
        else:
            print("Bad line:\n*\n%s\n*" % line)
            raise ValueError("len(chunks) == %d\n" % len(chunks))

    if scan_number and (retention_time or not require_rt) and fragment_mzs and \
            fragment_intensities and precursor_mz and charge:
        yield errorcalc.MS2Spectrum(scan_number,
                                    retention_time,
                                    fragment_mzs,
                                    fragment_intensities,
                                    precursor_mz, charge)
        n_yielded += 1
    else:
        logger.debug("Tried to write scan with not all values collected")
    logger.debug("Returned %d spectra" % n_yielded)


