"""
Season configuration for Al Karama satellite and shade analyses.

Single source of truth for date ranges, solar dates, and cloud-cover thresholds.
"""

SEASONS = {
    "summer_2020": {
        "id": "summer_2020",
        "label": "Summer 2020",
        "satellite_start": "2020-06-01",
        "satellite_end": "2020-09-30",
        "solar_date": (2020, 7, 15),       # representative date for shade analysis
        "cloud_cover_landsat": 20,
        "cloud_cover_sentinel": 10,
    },
    "winter_2020": {
        "id": "winter_2020",
        "label": "Winter 2020/21",
        "satellite_start": "2020-12-01",
        "satellite_end": "2021-02-28",
        "solar_date": (2021, 1, 15),       # representative date for shade analysis
        "cloud_cover_landsat": 30,
        "cloud_cover_sentinel": 20,
    },
    "summer_2025": {
        "id": "summer_2025",
        "label": "Summer 2025",
        "satellite_start": "2025-06-01",
        "satellite_end": "2025-09-30",
        "solar_date": (2025, 7, 15),       # representative date for shade analysis
        "cloud_cover_landsat": 20,
        "cloud_cover_sentinel": 10,
    },
    "winter_2025": {
        "id": "winter_2025",
        "label": "Winter 2025/26",
        "satellite_start": "2025-12-01",
        "satellite_end": "2026-02-28",
        "solar_date": (2026, 1, 15),       # representative date for shade analysis
        "cloud_cover_landsat": 30,          # slightly higher tolerance for winter
        "cloud_cover_sentinel": 20,
    },
}

DEFAULT_SEASON = "summer_2025"


def get_season_config(season_id=None):
    """Return the config dict for a given season id.

    Parameters
    ----------
    season_id : str or None
        One of the keys in SEASONS.  If None, returns the default season.

    Returns
    -------
    dict
    """
    if season_id is None:
        season_id = DEFAULT_SEASON
    if season_id not in SEASONS:
        raise ValueError(
            f"Unknown season '{season_id}'. "
            f"Available: {', '.join(SEASONS.keys())}"
        )
    return SEASONS[season_id]
