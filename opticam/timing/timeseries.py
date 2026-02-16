from astropy.time import Time
from astropy.timeseries import TimeSeries
from astropy.units import Quantity
import numpy as np
from numpy.typing import NDArray




def get_lc(
    light_curves: TimeSeries,
    key: str,
    ) -> TimeSeries:
    """
    Given a table of light curves, extract the light curve for a single key.
    
    Parameters
    ----------
    light_curves : TimeSeries
        The table of light curves.
    key : str
        The camera:filter key (e.g., "1:g" for camera 1 with a g filter).
    
    Returns
    -------
    TimeSeries
        The light curve for the filter.
    """
    
    flux_col = f'{key}_rel_flux'
    err_col = f'{key}_rel_flux_err'
    
    time_arr = light_curves['time']
    flux_arr = light_curves[flux_col]
    err_arr = light_curves[err_col]
    
    mask = np.isfinite(flux_arr) & np.isfinite(err_arr)
    
    new_lc = TimeSeries(time=time_arr[mask])
    new_lc[flux_col] = flux_arr[mask]
    new_lc[err_col] = err_arr[mask]
    
    return new_lc


def split_timeseries_on_gaps(
    ts: TimeSeries,
    threshold: float = 1.5,
    ) -> list[TimeSeries]:
    """
    Split a time series into a list of contiguous time series. Similar to `stingray`'s `split_by_gti()` method.
    
    Parameters
    ----------
    ts : TimeSeries
        The time series, assumed to contain gaps.
    threshold : float, optional
        The gap detection threshold, by default 1.5 times the median time delta.
    
    Returns
    -------
    list[TimeSeries]
        The list of strictly contiguous time series.
    """
    
    ts_list = []
    
    dt = np.diff(ts['time'].mjd)  # convert to MJD to remove units
    gap_indices = np.where(dt > threshold * np.median(dt))[0] + 1  # add 1 to shift index to first point after gap
    edge_indices = np.concat(([0], gap_indices, [len(ts)]))  # include start and end of time series
    
    for i in range(1, len(edge_indices)):
        ts_list.append(ts[edge_indices[i-1]:edge_indices[i]])
    
    return ts_list


def segment_timeseries(
    ts: TimeSeries,
    segment_size: Quantity,
    ) -> list[TimeSeries]:
    """
    Split a time series into equal length, contiguous segments.
    
    Parameters
    ----------
    ts : TimeSeries
        The time series.
    segment_size : Quantity
        The segment size.
    
    Returns
    -------
    list[TimeSeries]
        The time series segments.
    """
    
    segments: list[TimeSeries] = []
    
    # split time series on gaps
    ts_list = split_timeseries_on_gaps(ts)
    
    n: int = get_segment_size(
        ts=ts_list[0],
        segment_size=segment_size,
        )
    
    for ts in ts_list:
        prev = 0
        while prev + n <= len(ts):
            segments.append(TimeSeries(ts[prev:prev + n]))
            prev += n
    
    return segments



def get_segment_size(
    ts: TimeSeries,
    segment_size: Quantity,
    ) -> int:
    """
    Get the number of time series rows per segment.
    
    Parameters
    ----------
    ts : TimeSeries
        The time series.
    segment_size : Quantity
        The segment size.
    
    Returns
    -------
    int
        The number of rows per segment.
    """
    
    dt = np.median(np.diff(ts['time']))
    
    return round((segment_size / dt).decompose().value)


def infer_gtis(
    time: NDArray | Time | Quantity,
    threshold: float = 1.5,
    ) -> NDArray:
    """
    Infer the Good Time Intervals from a time array.
    
    Parameters
    ----------
    time : NDArray | Time | Quantity
        The time array. If this array has units, the resulting GTIs will have the same units.
    threshold : float, optional
        The gap detection threshold, by default 1.5 times the minimum time delta.
    
    Returns
    -------
    NDArray
        The inferred GTIs.
    """
    
    time = np.sort(time)  # ensure time stamps are sorted
    
    # nominal time resolution
    dt = np.median(np.diff(time))
    
    # compute the gap threshold
    gap_threshold = threshold * dt
    
    # define GTI starts and stops
    gti_starts = [time[0] - dt / 2]
    gti_stops = []
    
    # compute GTIs
    for i in range(1, time.size):
        if time[i] - time[i - 1] > gap_threshold:
            gti_stops.append(time[i - 1] + dt / 2)
            gti_starts.append(time[i] - dt / 2)
    
    if gti_starts[-1] == time[-1]:
        gti_starts.pop()
    else:
        gti_stops.append(time[-1] + dt / 2)
    
    # define GTIs in stingray format
    return np.array(list(zip(gti_starts, gti_stops)))







