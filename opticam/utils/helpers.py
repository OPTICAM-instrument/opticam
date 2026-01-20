import os
from pathlib import Path
from typing import Any, Dict, List
import re


from astropy.timeseries import TimeSeries
import numpy as np
from numpy.typing import NDArray


from opticam.utils.constants import filter_order




def camel_to_snake(
    string: str,
    ) -> str:
    """
    Convert a camelCase string to snake_case.
    
    Parameters
    ----------
    string : str
        The camelCase string to convert.
    
    Returns
    -------
    str
        The converted snake_case string.
    """
    
    return re.sub(r'(?<!^)(?=[A-Z])', '_', string).lower()


def sort_dict_by_filters(
    d: Dict[str, Any],
    ) -> Dict[str, Any]:
    """
    Attempt to sort a dictionary whose keys are filter names in order of increasing wavelength (e.g., u, g, r, i, z). If
    unrecognised filters are passed, no sorting is performed.
    
    Parameters
    ----------
    d : Dict[str, Any]
        A dictionary with filter names as keys.
    
    Returns
    -------
    Dict[str, Any]
        The sorted dictionary.
    """
    
    for key in d.keys():
        if key not in filter_order.keys():
            # unrecognised filter; cannot sort
            return d
    
    return dict(sorted(d.items(), key=lambda x: filter_order[x[0]]))


def sort_filters(
    filters: List[str],
    ) -> List[str]:
    """
    Attempt to sort a list of filters in order of increasing wavelength (e.g., u, g, r, i, z). If unrecognised filters
    are passed, no sorting is performed.
    
    Parameters
    ----------
    filters : List[str]
        The list of filters.
    
    Returns
    -------
    List[str]
        The sorted list of filters.
    """
    
    for fltr in filters:
        if fltr not in filter_order.keys():
            # unrecognised filter; cannot sort
            return filters
    
    return sorted(filters, key=lambda x: filter_order[x[0]])


def create_file_paths(
    data_directory: Path,
    ) -> List[Path]:
    """
    Given some directories, get the paths to all available FITS files.
    
    Parameters
    ----------
    data_directory : Path
        The directory containing the FITS files.
    
    Returns
    -------
    List[Path]
        The file paths.
    """
    
    file_paths = []
    file_names = os.listdir(data_directory)
    for file_name in file_names:
        if '.fit' in file_name:
            file_paths.append(os.path.join(data_directory, file_name))
    
    return file_paths


def propagate_errors(
    data: NDArray,
    bias_var: float | NDArray[np.float64],
    dark_var: float | NDArray[np.float64],
    flat_var: float | NDArray[np.float64],
    background_rms: float | NDArray,
    read_noise: float,
    ) -> NDArray[np.float64]:
    """
    Compute the propagated error image.
    
    Parameters
    ----------
    data : NDArray
        The calibrated, background-subtracted image.
    bias_var : float | NDArray[np.float64]
        The bias correction variance term.
    dark_var : float | NDArray[np.float64]
        The dark noise correction variance term.
    flat_var : float | NDArray[np.float64]
        The flat-field correction variance term scaled by the square of the calibrated image.
    background_rms : float | NDArray
        The background RMS.
    read_noise : float
        The read noise [electrons/pixel].
    
    Returns
    -------
    NDArray[np.float64]
        The propagated error image.
    """
    
    total_variance = np.clip(data, 0., None)  # source shot noise
    total_variance += background_rms**2
    total_variance += read_noise**2
    total_variance += bias_var
    total_variance += dark_var
    total_variance += flat_var
    
    return np.sqrt(total_variance)


def get_lc(
    light_curves: TimeSeries,
    fltr: str,
    ) -> TimeSeries:
    """
    Given a table of light curves, extract the light curve for a single filter.
    
    Parameters
    ----------
    light_curves : TimeSeries
        The table of light curves.
    fltr : str
        The filter.
    
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
            if f'{fltr}_rel_flux' in colname:
                new_colnames.append(colname)
        # get filter backgrounds if included
        elif 'bkg' in colname:
            if f'{fltr}_bkg' in colname:
                new_colnames.append(colname)
        # include all non-flux/non-background columns (time, time_bin_start, etc.)
        else:
            new_colnames.append(colname)
    
    lc = light_curves[*new_colnames]
    
    # remove NaN rows
    f = np.asarray(lc[f'{fltr}_rel_flux'].value)
    ferr = np.asarray(lc[f'{fltr}_rel_flux_err'].value)
    mask = np.where(np.isnan(f) | np.isnan(ferr))[0]
    lc.remove_rows(mask)
    
    return lc





