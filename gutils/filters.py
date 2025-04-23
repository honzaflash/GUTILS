#!python
# coding=utf-8
import os
import pandas as pd

from gutils.yo import assign_profiles, count_profiles

import logging
L = logging.getLogger(__name__)


def default_filter(dataset, filter_z=None, filter_points=None, filter_time=None, filter_distance=None):
    """Applies profile filters to a dataset

    Filters based on:
    - depth
    - number of points
    - time period
    - distance
    """
    dataset, rm_depth,    did_depth    = filter_profile_depth(dataset, below=filter_z, reindex=False)
    dataset, rm_points,   did_points   = filter_profile_number_of_points(dataset, points_condition=filter_points, reindex=False)
    dataset, rm_time,     did_time     = filter_profile_timeperiod(dataset, timespan_condition=filter_time, reindex=False)
    dataset, rm_distance, did_distance = filter_profile_distance(dataset, distance_condition=filter_distance, reindex=True)
    total_filtered = rm_depth + rm_points + rm_time + rm_distance
    return dataset, {
        'total_removed': total_filtered,
        'depth': { 'removed': rm_depth, 'filter_val': did_depth },
        'points': { 'removed': rm_points, 'filter_val': did_points },
        'time': { 'removed': rm_time, 'filter_val': did_time },
        'distance': { 'removed': rm_distance, 'filter_val': did_distance },
    }


def filter_profiles(dataset, conditional, reindex=True):
    """Filters out profiles that do not meet some criteria

    Returns the filtered set of profiles
    """
    before = count_profiles(dataset)
    filtered = dataset.groupby('profile').filter(conditional).copy()
    after = count_profiles(filtered)
    # Re-index the profiles
    if reindex is True:
        f, _ = pd.factorize(filtered.profile)
        filtered.loc[:, 'profile'] = f.astype('int32')  # Avoid the int64 dtype

    return filtered, before - after


def filter_profile_depth(dataset, below=None, reindex=True):
    """Filters out profiles that are not completely below a certain depth (Default: 1m). This is
    depth positive down.

    Returns a DataFrame with a subset of profiles
    """

    if below is None:
        below = 1

    def conditional(profile):
        return profile.z.max() >= below

    return (*filter_profiles(dataset, conditional, reindex=reindex), below)


def filter_profile_timeperiod(dataset, timespan_condition=None, reindex=True):
    """Filters out profiles that do not span a specified number of seconds
    (Default: 10 seconds)

    Returns a DataFrame with a subset of profiles
    """
    if timespan_condition is None:
        timespan_condition = 10

    def conditional(profile):
        timespan = profile.t.max() - profile.t.min()
        return timespan >= pd.Timedelta(timespan_condition, unit='s')

    return (*filter_profiles(dataset, conditional, reindex=reindex), timespan_condition)


def filter_profile_distance(dataset, distance_condition=None, reindex=True):
    """Filters out profiles that do not span a specified vertical distance
    (Default: 1m)

    Returns a DataFrame with a subset of profiles
    """
    if distance_condition is None:
        distance_condition = 1

    def conditional(profile):
        distance = abs(profile.z.max() - profile.z.min())
        return distance >= distance_condition

    return (*filter_profiles(dataset, conditional, reindex=reindex), distance_condition)


def filter_profile_number_of_points(dataset, points_condition=None, reindex=True):
    """Filters out profiles that do not have a specified number of points
    (Default: 3 points)

    Returns a DataFrame with a subset of profiles
    """
    if points_condition is None:
        points_condition = 3

    def conditional(profile):
        return len(profile) >= points_condition

    return (*filter_profiles(dataset, conditional, reindex=reindex), points_condition)


def process_dataset(file,
                    reader_class,
                    *,
                    tsint=None,
                    filter_z=None,
                    filter_points=None,
                    filter_time=None,
                    filter_distance=None,
                    z_axis_method=1,
                    require_inflection=True,
                    **extra_kwargs):

    # Check filename
    if file is None:
        raise ValueError('Must specify path to combined ASCII file')

    try:
        reader = reader_class(file)
        data = reader.standardize(z_axis_method=z_axis_method)

        # The data section is also passed to extras processing.
        # In certain circumstances, the dimension for certain
        # variables are reassigned with the extras dimension.
        extras, data = reader.extras(data, **extra_kwargs)

        if 'z' not in data.columns:
            L.warning("No Z axis found - Skipping {}".format(file))
            return None, None, None

        if 't' not in data.columns:
            L.warning("No T axis found - Skipping {}".format(file))
            return None, None, None

        # Find profile breaks
        profiles = assign_profiles(data, tsint=tsint, require_inflection=require_inflection)
        # Shortcut for empty dataframes
        if profiles is None:
            return None, None, None

        # Filter data
        original_count = count_profiles(profiles)
        filtered, stats = default_filter(profiles, filter_z, filter_points, filter_time, filter_distance)
        L.info(
            f'Filtered {stats["total_removed"]}/{original_count} profiles from {os.path.basename(file)} - '
            + ', '.join(
                f'{variable.capitalize()} ({stats[variable]["filter_val"]}{unit}): {stats[variable]["removed"]}'
                for variable, unit in [('depth', 'm'), ('points', ''), ('time', 's'), ('distance', 'm')]
            )
        )

        # Downscale profile
        # filtered['profile'] = pd.to_numeric(filtered.profile, downcast='integer')
        filtered['profile'] = filtered.profile.astype('int32')
        # Profiles are 1-indexed, so add one to each
        filtered['profile'] = filtered.profile.values + 1

        # TODO: Backfill U/V?
        # TODO: Backfill X/Y?

        # Combine extra data, which has a time index already
        # This puts the profile information into the extras
        # dataframe
        if not extras.empty:
            try:
                merge = pd.merge_asof(
                    extras,
                    filtered[['t', 'profile']],
                    left_index=True,
                    right_index=False,
                    right_on='t',
                    direction='nearest',
                    tolerance=pd.Timedelta(minutes=10)
                ).set_index(extras.index)
                extras['profile'] = merge.profile.ffill()
            except BaseException as e:
                L.error(f"Could not merge 'extras' data, skipping: {e}")
                extras = pd.DataFrame()

    except ValueError as e:
        L.exception('{} - Skipping'.format(e))
        raise

    return filtered, extras, reader.mode
