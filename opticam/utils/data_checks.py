from functools import partial
from logging import Logger
from multiprocessing import cpu_count
from pathlib import Path
import re
from typing import Callable, Dict, List, Tuple
import warnings


import numpy as np
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm


from opticam.utils.constants import bar_format
from opticam.utils.fits_handlers import get_header_info
from opticam.utils.helpers import create_file_paths, sort_dict_by_filters
from opticam.utils.logging import log_file
from opticam.instruments import Instrument




def scan_data(
    out_directory: Path | str,
    data_directory: Path | str,
    instrument: Instrument,
    barycenter: bool = True,
    verbose: bool = True,
    return_output: bool = False,
    logger: Logger | None = None,
    number_of_processors = cpu_count() // 2,
    ) -> None | Tuple[Dict[str, List[Path]], int, Dict[Path, float], List[Path], float]:
    """
    Check that the data are self-consistent.
    
    Parameters
    ----------
    out_directory : Path | str
        The directory path to which any output files will be saved.
    data_directory : Path | str
        The directory path to the data.
    instrument : Instrument
        The instrument.
    barycenter : bool, optional
        Whether to apply a Barycentric correction to the image time stamps, by default True.
    verbose : bool, optional
        Whether to print any output info, by default True.
    return_output : bool, optional
        Whether to return any output, by default False.
    logger : Logger | None, optional
        The logger, by default None.
    number_of_processors : _type_, optional
        The number of processors to use, by default `cpu_count() // 2`.
    
    Returns
    -------
    None | Tuple[Dict[str, List[Path]], int, Dict[Path, float], List[Path], float]
        If `return_output=True`, the file paths, binning scale, Barycentric MJD dates, ignored files, and the reference
        date are returned. Otherwise, nothing is returned.
    """
    
    out_directory = Path(out_directory)
    data_directory = Path(data_directory)
    
    file_paths = create_file_paths(data_directory=data_directory)
    
    # check instrument
    errors = instrument.run_checks(
        file_paths[0],
        return_errors=True,
        )
    if errors == 1:
        raise ValueError(f'[OPTICAM] {errors} Instrument error needs to be resolved.')
    elif errors > 1:
        raise ValueError(f'[OPTICAM] {errors} Instrument errors need to be resolved.')
    
    camera_files: Dict[str, List[Path]] = {}  # {filter : [files]}
    
    # scan files
    chunksize = max(1, len(file_paths) // 100)  # set chunksize to ~1% of the number of files
    results = process_map(
        partial(
            get_header_info,
            instrument=instrument,
            barycenter=barycenter,
            ),
        file_paths,
        max_workers=number_of_processors,
        disable=not verbose,
        desc="[OPTICAM] Scanning data directory",
        chunksize=chunksize,
        bar_format=bar_format,
        tqdm_class=tqdm)
    
    # unpack results
    binning, bmjds, filters, ignored_files = parse_header_results(
        results=results,
        file_paths=file_paths,
        out_directory=out_directory,
        logger=logger,
        )
    
    # for each unique filter
    for fltr in set(filters.values()):
        camera_files.update({fltr: []})  # prepare dictionary entry
        for file in file_paths:
            if file not in ignored_files:
                if filters[file] == fltr:
                    camera_files[fltr].append(file)  # add file name to dict list
    
    # sort camera files so filters match camera order
    camera_files: Dict[str, List[Path]] = sort_dict_by_filters(camera_files)
    
    # sort files by time
    for key in list(camera_files.keys()):
        camera_files[key].sort(key=lambda x: bmjds[x])
    
    t_ref = min(list(bmjds.values()))  # get reference BMJD
    
    output = partial(
        data_checks_output,
        binning=binning,
        camera_files=camera_files)
    if logger:
        output(func=logger.info)
    if verbose:
        output(func=print)
    
    if return_output:
        
        return camera_files, get_binning_scale(binning), bmjds, ignored_files, t_ref


def parse_header_results(
    results: Tuple[List[float], List[float], List[str], List[str], List[float]],
    file_paths: List[Path],
    out_directory: Path,
    logger: Logger | None,
    ) -> Tuple[str, Dict[Path, float], Dict[Path, str], List[Path]]:
    """
    Parse the header info results.
    
    Parameters
    ----------
    results : Tuple[List[float], List[float], List[str], List[str], List[float]]
        The header info results.
    file_paths : List[Path]
        The file paths.
    out_directory : str
        The directory path to which any output files will be saved.
    logger : Logger | None
        The logger.
    
    Returns
    -------
    Tuple[str, Dict[Path, float], Dict[Path, str], List[Path]]
        The binning scale, BMJD dates, filters, and ignored files.
    
    Raises
    ------
    ValueError
        If more than three filters are detected.
    ValueError
        If more than one binning mode is detected.
    """
    
    binnings: Dict[Path, str] = {}
    bmjds: Dict[Path, float] = {}
    exposures: Dict[Path, float] = {}
    filters: Dict[Path, str] = {}
    ignored_files: List[Path] = []
    
    # unpack results
    raw_bmjds, raw_exposures, raw_filters, raw_binnings = zip(*results)
    
    # consolidate results
    for i in range(len(raw_bmjds)):
        if raw_bmjds[i] is not None:
            binnings.update({file_paths[i]: raw_binnings[i]})
            bmjds.update({file_paths[i]: raw_bmjds[i]})
            exposures.update({file_paths[i]: raw_exposures[i]})
            filters.update({file_paths[i]: raw_filters[i]})
        else:
            ignored_files.append(file_paths[i])
    
    # get unique filters
    unique_filters = set(filters.values())
    
    # ensure there is at most one type of binning
    unique_binnings = set(binnings.values())
    if len(unique_binnings) > 1:
        log_file(
            out_directory=out_directory,
            file_name='binnings.json',
            file_contents=binnings,
            )
        string = f"[OPTICAM] Inconsistent binning detected. All images must have the same binning. Image binnings have been logged to {out_directory.joinpath('diag/binnings.json')}."
        if logger is not None:
            logger.error(string)
        raise ValueError(string)
    elif len(unique_binnings) == 0:
        raise ValueError(f'[OPTICAM] No binning values detected.')
    unique_binning = unique_binnings.pop()  # get unique binning
    
    # check for large differences in time
    for fltr in unique_filters:
        fltr_bmjds = np.sort(np.array([bmjds[file] for file in file_paths if file in filters and filters[file] == fltr]))
        files = [file for file in file_paths if file in filters and filters[file] == fltr]
        t = fltr_bmjds - np.min(fltr_bmjds)
        dt = np.diff(t) * 86400
        if np.any(dt > 10 * np.median(dt)):
            indices = np.where(dt > 10 * np.median(dt))[0]
            for index in indices:
                string = f"[OPTICAM] Large time gap detected between {files[index].split('/')[-1]} and {files[index + 1].split('/')[-1]} ({dt[index]:.3f} s compared to the median time difference of {np.median(dt):.3f} s). This may cause alignment issues. If so, consider moving all files after this gap to a separate directory."
                warnings.warn(string)
                if logger:
                    logger.warning(string)
    
    # check all images use the same exposure time
    if len(set(exposures.values())) > 1:
        log_file(
            out_directory=out_directory,
            file_name='exposures.json',
            file_contents=exposures,
            )
        string = f'[OPTICAM] Inconsistent exposure times detected. This is not necessarily a problem, but may cause issues with Fourier transforms later on. Image exposure times have been logged to {out_directory.joinpath('diag/exposures.json')}.'
        if logger:
            logger.warning(string)
        warnings.warn(string)
    
    return unique_binning, bmjds, filters, ignored_files


def data_checks_output(
    binning: str,
    camera_files: Dict[str, List[Path]],
    func: Callable,
    ) -> None:
    """
    Output the results of the data checks.
    
    Parameters
    ----------
    binning : str
        The image binning.
    camera_files : Dict[str, List[Path]]
        The image files separated by filter.
    func : Callable
        The output function (i.e., `print` or `logger.info`)
    """
    
    func(f'[OPTICAM] Binning: {binning}')
    func(f'[OPTICAM] Filters: {", ".join(list(camera_files.keys()))}')
    for fltr in list(camera_files.keys()):
        func(f'[OPTICAM] {len(camera_files[fltr])} {fltr} images.')


def get_binning_scale(binning: str) -> int:
    """
    Given a binning mode string, extract the x and y binning scales as integers.
    
    Parameters
    ----------
    binning : str
        The binning mode string (e.g., "2x2", "1 2", etc.). The first number is assumed to be the binning scale in x,
        while the second number is assumed to be the binning scale in y.
    
    Returns
    -------
    int
        The binning scale.
    """
    
    x, y = map(int, re.findall(r"\d+", binning))
    
    assert(x == y), f'[OPTICAM] Anisotropic binning detected: {binning}. Currently, OPTICAM only supports isotropic binning modes. We apologise for the inconvenience.'
    
    return x
















