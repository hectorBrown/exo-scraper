import lightkurve as lk
from lightkurve.search import SearchResult
from lightkurve import LightCurveCollection, LightCurve
from astropy.units import Quantity
import numpy as np


def main():
    res = get_exos("Kepler-69", 0.5, 20)
    print(res)


def get_exos(
    star_name: str, min_orbit: float, max_orbit: float
) -> tuple[Quantity, Quantity, Quantity, Quantity] | None:
    """Gets exoplanet candidates for a given star.

    Given a star name, and the minimum and maximum orbital periods, return the
    period, transit time, duration and SNR of the highest peak in the BLS
    periodogram. The star must be evailable in Kepler long-cadence data.

    Args:
        star_name (str): The name of the star to analyze.
        min_orbit (float): The minimum orbital period to search for, in days.
        max_orbit (float): The maximum orbital period to search for, in days.

    Returns:
        Optional[tuple[Quantity, Quantity, Quantity, Quantity]]: The period,
            transit time, duration, and signal-to-noise ratio corresponding to
            the highest peak in the BLS periodogram, or None if no results are
            found.
    """
    # TODO: change this to search SPOC/TESS for bigger data source -- unless the star exists in Kepler in which case we should use their sexy data
    search_res = lk.search_lightcurve(star_name, author="Kepler", cadence="long")

    # TODO: stream from s3 source -- might want to include a switch for this so we can still test locally
    lcs = get_lightcurves(search_res)
    if lcs is None:
        return None

    def _corrector_func(lc: LightCurve) -> LightCurve:
        cadence = get_cadence(lc)
        return (
            lc.remove_nans()
            .remove_outliers()  # pyright:ignore
            .flatten(window_length=get_savgol_window(cadence))  # pyright: ignore
        )

    stitched_lc = lcs.stitch(corrector_func=_corrector_func).remove_nans()
    # TODO: should start search coarse and wide -- refine, mask, restart, until all we pick up is noise
    periodogram = stitched_lc.to_periodogram(
        method="bls",
        period=np.linspace(
            min_orbit, max_orbit, 10000
        ),  # TODO: WTF, is this actually pointless
        frequency_factor=500,
    )
    # TODO: super crude baseline noise estimation
    baseline = np.median(periodogram.power)
    power_at_peak = np.max(periodogram.power)
    snr = power_at_peak / baseline
    # TODO: need to check if more than one planet
    if snr < 6:
        return None
    return (
        periodogram.period_at_max_power,
        periodogram.transit_time_at_max_power,
        periodogram.duration_at_max_power,
        snr,
    )


def get_lightcurves(collection: SearchResult) -> LightCurveCollection | None:
    """Downloads light curves from a SearchResult.

    Given a SearchCollection, download the all light curves and return them as
    a LightCurveCollection.

    Args:
        collection (SearchResult): The SearchResult containing the light curve search results.

    Returns:
        LightCurveCollection: A collection of light curves downloaded from the search results, or None if the download fails.
    """
    # this function is separate in case we need to plug it into AWS somehow to take advantage of s3 buckets

    res = collection.download_all()
    if not isinstance(res, LightCurveCollection):
        return None
    return res


def get_cadence(lc: LightCurve) -> float:
    """Given a light curve, return its cadence in seconds.

    Args:
        lc (LightCurve): The light curve for which to calculate the cadence.

    Returns:
        float: The cadence of the light curve in seconds, calculated as the
            median difference between consecutive time points.
    """
    # bit awkward, we could also try to find this with numpy, but metadata seems like a good source of truth
    if lc.meta is None:
        raise ValueError("Light curve metadata is missing.")
        # TODO: find cadence manually

    unit = lc.meta.get("TIMEUNIT")
    timedel = lc.meta.get("TIMEDEL")

    try:
        timedel = float(timedel)
    except TypeError, ValueError:
        raise ValueError("Could not convert TIMEDEL to float.")

    match unit:
        case "d":
            return timedel * 24 * 60 * 60
        case "s":
            return timedel
        case _:
            raise ValueError(f"Unsupported time unit: {unit}")


def get_savgol_window(cadence: float) -> int:
    """Gets a Savitzky-Golay filter window length.

    Return the optimal window length for the Savitzky-Golay filter based on
    the star's properties.

    Args:
        cadence (float): The cadence of the light curve data, in seconds.

    Returns:
        int: The optimal window length for the Savitzky-Golay filter, calculated based on the star's properties.
    """
    # placeholder function -- literally just returns 15 hours
    # TODO: need to optimise this against stellar variability
    window_length = int(15 * 60 * 60 / cadence)  #
    return (
        window_length if window_length % 2 == 1 else window_length + 1
    )  # ensure the window length is odd


if __name__ == "__main__":
    main()
