import json
import os
from types import FunctionType
from typing import Any, Dict, List

from astropy.io import fits

from opticam.utils.constants import pixel_scales


def log_binnings(
    file_paths: List[str],
    out_directory: str,
    ) -> None:
    """
    Log the binning of each file to out_directory/diag/binnings.json.
    
    Parameters
    ----------
    file_paths : List[str]
        The paths to the files.
    out_directory : str
        The directory to save the log.
    """
    
    file_binnings = {}
    
    for file in file_paths:
        with fits.open(file) as hdul:
            binning = hdul[0].header["BINNING"]
        if binning in file_binnings:
            file_binnings[binning].append(file)
        else:
            file_binnings[binning] = [file]
    
    dir_path = os.path.join(out_directory, 'diag')
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    
    file_path = os.path.join(dir_path, 'binnings.json')
    with open(file_path, "w") as json_file:
        json.dump(file_binnings, json_file, indent=4)


def log_filters(
    file_paths: List[str],
    out_directory: str,
    ) -> None:
    """
    Logs the filters used in each file to out_directory/diag/filters.json.
    
    Parameters
    ----------
    file_paths : List[str]
        The paths to the files.
    out_directory : str
        The directory to save the log.
    """
    
    file_filters = {}
    
    for file in file_paths:
        with fits.open(file) as hdul:
            fltr = hdul[0].header["FILTER"]
        if fltr in file_filters:
            file_filters[fltr].append(file)
        else:
            file_filters[fltr] = [file]
    
    dir_path = os.path.join(out_directory, 'diag')
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    
    file_path = os.path.join(dir_path, 'filters.json')
    with open(file_path, "w") as json_file:
        json.dump(file_filters, json_file, indent=4)


def recursive_log(param: Any, depth: int = 0, max_depth: int = 5) -> Any:
    """
    Recursively log parameters.
    
    Parameters
    ----------
    param : Any
        The parameter to log.
    depth : int, optional
        The parameter depth, by default 0.
    max_depth : int, optional
        The maximum parameter depth, by default 5. This prevents infinite recursion.
    
    Returns
    -------
    Any
        The logged parameter.
    """
    
    if depth > max_depth:
        return f"<Max depth ({max_depth}) reached>"
    
    if isinstance(param, FunctionType):
        # return function name
        return param.__name__
    if isinstance(param, (int, float, str, bool, type(None))):
        return param
    if isinstance(param, (list, tuple, set)):
        return type(param)(recursive_log(item, depth + 1, max_depth) for item in param)
    if isinstance(param, dict):
        return {key: recursive_log(value, depth + 1, max_depth) for key, value in param.items()}
    if hasattr(param, '__dict__'):
        return {key: recursive_log(value, depth + 1, max_depth) for key, value in vars(param).items()}
    return str(param)


def log_psf_params(
    out_directory: str,
    psf_params: Dict[str, Dict[str, float]],
    binning_scale: int,
    rebin_factor: int,
    ) -> None:
    """
    Log the PSF parameters.
    
    Parameters
    ----------
    out_directory : str
        The path to the output directory.
    psf_params : Dict[str, Dict[str, float]]
        The PSF parameters.
    binning_scale : int
        The observation binning scale.
    rebin_factor : int
        The software rebinning factor.
    """
    
    psf_params_full = {}
    
    for fltr in psf_params.keys():
        # convert from pixels to arcsec
        semimajor_sigma_arcsec = psf_params[fltr]['semimajor_sigma'] * binning_scale * rebin_factor * pixel_scales[fltr]
        semiminor_sigma_arcsec = psf_params[fltr]['semiminor_sigma'] * binning_scale * rebin_factor * pixel_scales[fltr]
        
        psf_params_full[fltr] = {
            'semimajor_sigma_arcsec': semimajor_sigma_arcsec,
            'semimajor_sigma_pix': psf_params[fltr]['semimajor_sigma'],
            'semiminor_sigma_arcsec': semiminor_sigma_arcsec,
            'semiminor_sigma_pix': psf_params[fltr]['semiminor_sigma'],
        }
    
    # save PSF params to JSON file
    with open(os.path.join(out_directory, f'misc/psf_params.json'), 'w') as file:
        json.dump(psf_params_full, file, indent=4)























