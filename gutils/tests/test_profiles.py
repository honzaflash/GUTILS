#!python
# coding=utf-8
import pandas as pd
import pytest
import logging

from gutils.slocum import SlocumReader
from gutils.tests import setup_testing_logger, resource

from gutils.yo import assign_profiles, count_profiles


L = logging.getLogger(__name__)  # noqa


# Configure parameters for the test
# Each record is an ascii file and a set of expected results to assert
#   Expected contains:
#   - count: the number of profiles expected
#   - points: an optional map of profile number to expected number of points/rows
# Lists are per deployment, then get enhanced with deployment specific information
# and concatenated

unit_191_test_files = [
    { 'name': 'unit_191_2025_069_2_355_sbd.dat', 'expected': { 'count': 2, 'points': { 0: 252, 1: 2 } } },
    { 'name': 'unit_191_2025_059_1_17_sbd.dat', 'expected': { 'count': None } },
    { 'name': 'unit_191_2025_098_0_16_sbd.dat', 'expected': { 'count': 1, 'points': { 0: 234 } } },
    { 'name': 'unit_191_2025_069_2_45_sbd.dat', 'expected': { 'count': None } },
    { 'name': 'unit_191_2025_098_0_14_sbd.dat', 'expected': { 'count': 1 } },
    { 'name': 'unit_191_2025_069_2_23_sbd.dat', 'expected': { 'count': None } },
    { 'name': 'unit_191_2025_069_2_198_sbd.dat', 'expected': { 'count': None } },
    { 'name': 'unit_191_2025_063_0_16_sbd.dat', 'expected': { 'count': 1 } },
    { 'name': 'unit_191_2025_063_0_24_sbd.dat', 'expected': { 'count': 2 } },
    { 'name': 'unit_191_2025_098_0_145_sbd.dat', 'expected': { 'count': 1 } },
]

bass_test_files = [
    { 'name': 'usf_bass_2016_252_1_0_sbd.dat', 'expected': { 'count': 6 } },
    { 'name': 'usf_bass_2016_252_1_6_sbd.dat', 'expected': { 'count': None } },
    { 'name': 'usf_bass_2016_252_1_8_sbd.dat', 'expected': { 'count': 8 } },
    { 'name': 'usf_bass_2016_252_1_11_sbd.dat', 'expected': { 'count': None } },
    { 'name': 'usf_bass_2016_252_1_17_sbd.dat', 'expected': { 'count': 7, 'points': { 0: 18, 1: 140, 6: 177 } } },
    { 'name': 'usf_bass_2016_253_0_1_sbd.dat', 'expected': { 'count': 13 } },
    { 'name': 'usf_bass_2016_253_0_4_sbd.dat', 'expected': { 'count': 29 } },
    { 'name': 'usf_bass_2016_253_0_6_sbd.dat', 'expected': { 'count': 32 } },
]

modena_test_files = [
    { 'name': 'modena_2015_175_0_9_dbd.dat', 'expected': { 'count': 12, 'points': { 0: 2718, 11: 43 } } },
]

# Enhance with deployment specific information and concatenate
test_files = [
    *({ 'deployment': 'unit_191-20250226T0000', 'mode': 'rt', **details } for details in unit_191_test_files),
    *({ 'deployment': 'bass-test-ascii', 'mode': 'rt', **details } for details in bass_test_files),
    *({ 'deployment': 'modena-test-ascii', 'mode': 'delayed', **details } for details in modena_test_files),
]

@pytest.mark.long
@pytest.mark.parametrize('ascii_file', test_files, ids=lambda f: f'{f["name"]}')
def test_profile_assignment(ascii_file):
    setup_testing_logger(level=logging.WARNING)
    ascii = resource('slocum', ascii_file['deployment'], ascii_file['mode'], 'ascii', ascii_file['name'])
    df = SlocumReader(ascii).standardize()

    profiles = assign_profiles(df)

    if ascii_file['expected']['count'] is None:
        assert profiles is None
        return
    assert profiles is not None

    assert isinstance(profiles, pd.DataFrame)
    assert 'profile' in profiles.columns
    assert count_profiles(profiles) == ascii_file['expected']['count']

    # Check that the number of rows in each profile matches the expected count if provided
    point_counts = { profile: len(rows) for profile, rows in profiles.groupby('profile') }
    for profile, count in ascii_file['expected'].get('points', {}).items():
        assert profile in point_counts
        assert point_counts[profile] == count

