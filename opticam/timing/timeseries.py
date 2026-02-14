from astropy.timeseries import TimeSeries
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
    
    colnames = light_curves.colnames
    new_colnames = []
    
    for colname in colnames:
        # get filter fluxes
        if 'rel_flux' in colname:
            if f'{key}_rel_flux' in colname:
                new_colnames.append(colname)
        # get filter backgrounds if included
        elif 'bkg' in colname:
            if f'{key}_bkg' in colname:
                new_colnames.append(colname)
        # include all non-flux/non-background columns (time, time_bin_start, etc.)
        else:
            new_colnames.append(colname)
    
    lc = light_curves[*new_colnames]
    
    # remove NaN rows
    f = np.asarray(lc[f'{key}_rel_flux'].value)
    ferr = np.asarray(lc[f'{key}_rel_flux_err'].value)
    mask = np.where(np.isnan(f) | np.isnan(ferr))[0]
    lc.remove_rows(mask)
    
    return TimeSeries(lc)


def split_timeseries_by_gaps(
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
        The gap detection threshold, by default 1.5 times the minimum time delta.
    
    Returns
    -------
    list[TimeSeries]
        The list of strictly contiguous time series.
    """
    
    ts_list = []
    
    dt = np.diff(ts['time'].mjd)  # convert to MJD (or any other format) to remove units
    gap_indices = np.where(dt > threshold * np.median(dt))[0]
    edge_indices = np.concat(([0], gap_indices, [len(ts)]))
    
    for i in range(1, len(edge_indices)):
        ts_list.append(ts[edge_indices[i-1]:edge_indices[i]])
    
    return ts_list


def segment_timeseries(
    ts: TimeSeries,
    segment_size: Quantity,
    ) -> list[TimeSeries]:
    """
    Split a time series into equal length segments.
    
    Parameters
    ----------
    ts : TimeSeries
        The time series, assumed to be contiguous.
    segment_size : Quantity
        The segmen size.
    
    Returns
    -------
    list[TimeSeries]
        The time series segments.
    """
    
    n: int = get_segment_size(
        ts=ts,
        segment_size=segment_size,
        )
    
    ts_segments: list[TimeSeries] = []
    
    prev = 0
    while prev + n < len(ts):
        ts_segments.append(TimeSeries(ts[prev:prev + n]))
        prev += n
    
    return ts_segments



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
    time: NDArray,
    threshold: float = 1.5,
    ) -> NDArray:
    """
    Infer the Good Time Intervals from a light curve.
    
    Parameters
    ----------
    time : ArrayLike
        The time array.
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







