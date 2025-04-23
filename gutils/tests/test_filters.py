#!python
# coding=utf-8
import pandas as pd
import pytest
import logging

from gutils.filters import default_filter
from gutils.tests import setup_testing_logger, resource

from gutils.yo import count_profiles


L = logging.getLogger(__name__)  # noqa


test_files = [
    { 'name': 'unit_191_2025_069_2_355', 'expected': { 'original_count': 2, 'filtered_count': 1 } },
    { 'name': 'unit_191_2025_098_0_16', 'expected': { 'original_count': 1, 'filtered_count': 1 } },
    { 'name': 'unit_191_2025_098_0_14', 'expected': { 'original_count': 1, 'filtered_count': 1 } },
    { 'name': 'unit_191_2025_063_0_16', 'expected': { 'original_count': 1, 'filtered_count': 1 } },
    { 'name': 'unit_191_2025_063_0_24', 'expected': { 'original_count': 2, 'filtered_count': 1 } },
    { 'name': 'unit_191_2025_098_0_145', 'expected': { 'original_count': 1, 'filtered_count': 1 } },
]

@pytest.mark.long
@pytest.mark.parametrize('profiles', test_files, ids=lambda f: f'{f["name"]}')
def test_profile_filtering(profiles):
    setup_testing_logger()

    L.info(f'Load dataframe with profiles for {profiles["name"]}')
    csv_filename = f'{profiles["name"]}_profiles.csv'
    csv_path = resource('slocum', 'unit_191-20250226T0000', 'profiles', csv_filename)
    df = pd.read_csv(csv_path)
    df['t'] = pd.to_datetime(df['t'], unit='ns')

    original_count = count_profiles(df)
    if original_count != profiles['expected']['original_count']:
        L.warning(
            f'Original profile count {original_count} does not match '
            f'expected {profiles["expected"]["original_count"]}'
        )
        pytest.skip('Test resource did not load as expected!')
    L.info(f'Loaded dataframe with {len(df)} rows and {original_count} profiles')
    
    # Filter data
    filtered, stats = default_filter(df)
    L.info(
        f'Filtered {stats["total_removed"]}/{original_count} profiles from {profiles["name"]} - '
        + ', '.join(
            f'{variable.capitalize()} ({stats[variable]["filter_val"]}{unit}): {stats[variable]["removed"]}'
            for variable, unit in [('depth', 'm'), ('points', ''), ('time', 's'), ('distance', 'm')]
        )
    )
    assert count_profiles(filtered) == profiles['expected']['filtered_count']
    assert stats["total_removed"] == original_count - profiles['expected']['filtered_count']
