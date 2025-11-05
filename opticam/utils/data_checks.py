from functools import partial
from logging import Logger
from multiprocessing import cpu_count
from typing import Callable, Dict, List, Tuple
import warnings
import os.path

import numpy as np
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm

from opticam.utils.constants import bar_format
from opticam.utils.fits_handlers import get_header_info
from opticam.utils.helpers import create_file_paths, sort_filters
from opticam.utils.logging import log_binnings, log_filters


def check_data(
    out_directory: str,
    data_directory: None | str = None,
    c1_directory: None | str = None,
    c2_directory: None | str = None,
    c3_directory: None | str = None,
    barycenter: bool = True,
    verbose: bool = True,
    return_output: bool = False,
    logger: Logger | None = None,
    number_of_processors = cpu_count() // 2,
    ) -> None | Tuple[Dict[str, List[str]], int, Dict[str, float], List[str], Dict[str, float], float]:
    """
    Check that the data are self-consistent.
    
    Parameters
    ----------
    out_directory : str
        The directory path to which any output files will be saved.
    data_directory : None | str, optional
        The directory path to the data for all three cameras, by default None.
    c1_directory : None | str, optional
        The directory path to the data for Camera 1, by default None.
    c2_directory : None | str, optional
        The directory path to the data for Camera 2, by default None
    c3_directory : None | str, optional
        The directory path to the data for Camera 3, by default None
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
    None | Tuple[Dict[str, str], int, Dict[str, float], List[str], Dict[str, float], float]
        If `return_output=True`, the file paths, binning scale, Barycentric MJD dates, ignored files, file gains, and
        the reference date are returned. Otherwise, nothing is returned.
    """
    
    file_paths = create_file_paths(
        data_directory=data_directory,
        c1_directory=c1_directory,
        c2_directory=c2_directory,
        c3_directory=c3_directory,
        )
    
    camera_files = {}  # filter : [files]
    
    # scan files
    chunksize = max(1, len(file_paths) // 100)  # set chunksize to 1% of the number of files
    results = process_map(
        partial(
            get_header_info,
            barycenter=barycenter,
            logger=logger,
            ),
        file_paths,
        max_workers=number_of_processors,
        disable=not verbose,
        desc="[OPTICAM] Scanning data directory",
        chunksize=chunksize,
        bar_format=bar_format,
        tqdm_class=tqdm)
    
    # unpack results
    binning_scale, bmjds, filters, gains, ignored_files = parse_header_results(
        results=results,
        file_paths=file_paths,
        out_directory=out_directory,
        logger=logger,
        )
    
    # for each unique filter
    for fltr in np.unique(list(filters.values())):
        camera_files.update({fltr + '-band': []})  # prepare dictionary entry
        for file in file_paths:
            if file not in ignored_files:
                if filters[file] == fltr:
                    camera_files[fltr + '-band'].append(file)  # add file name to dict list
    
    # sort camera files so filters match camera order
    camera_files = sort_filters(camera_files)
    
    # sort files by time
    for key in list(camera_files.keys()):
        camera_files[key].sort(key=lambda x: bmjds[x])
    
    t_ref = min(list(bmjds.values()))  # get reference BMJD
    
    output = partial(
        data_checks_output,
        binning_scale=binning_scale,
        camera_files=camera_files)
    if logger:
        output(func=logger.info)
    if verbose:
        output(func=print)
    
    if return_output:
        return camera_files, binning_scale, bmjds, ignored_files, gains, t_ref


def parse_header_results(
    results: Tuple[float, float, str, str, float],
    file_paths: List[str],
    out_directory: str,
    logger: Logger | None,
    ) -> Tuple[int, Dict[str, float], Dict[str, str], Dict[str, float], List[str]]:
    """
    Parse the header info results.
    
    Parameters
    ----------
    results : Tuple[float, float, str, str, float]
        The header info results.
    file_paths : List[str]
        The file paths.
    out_directory : str
        The directory path to which any output files will be saved.
    logger : Logger | None
        The logger.
    
    Returns
    -------
    Tuple[int, Dict[str, float], Dict[str, str], Dict[str, float], List[str]]
        The binning scale, BMJD dates, filters, gains, and ignored files.
    
    Raises
    ------
    ValueError
        If more than three filters are detected.
    ValueError
        If more than one binning mode is detected.
    """
    
    bmjds = {}
    filters = {}
    binnings = {}
    gains = {}
    ignored_files = []
    
    # unpack results
    raw_bmjds, raw_filters, raw_binnings, raw_gains = zip(*results)
    
    # consolidate results
    for i in range(len(raw_bmjds)):
        if raw_bmjds[i] is not None:
            bmjds.update({file_paths[i]: raw_bmjds[i]})
            filters.update({file_paths[i]: raw_filters[i]})
            binnings.update({file_paths[i]: raw_binnings[i]})
            gains.update({file_paths[i]: raw_gains[i]})
        else:
            ignored_files.append(file_paths[i])
    
    # ensure there are no more than three filters
    unique_filters = np.unique(list(filters.values()))
    if unique_filters.size > 3:
        log_filters([file_path for file_path in file_paths if file_path not in ignored_files],
                    out_directory)
        raise ValueError(f"[OPTICAM] More than three filters found. Image filters have been logged to {os.path.join(out_directory, 'diag/filters.json')}.")
    
    # ensure there is at most one type of binning
    unique_binning = np.unique(list(binnings.values()))
    if len(unique_binning) > 1:
        log_binnings([file_path for file_path in file_paths if file_path not in ignored_files],
                        out_directory)
        raise ValueError(f"[OPTICAM] Inconsistent binning detected. All images must have the same binning. Image binnings have been logged to {os.path.join(out_directory, 'diag/binnings.json')}.")
    else:
        binning = unique_binning[0]
        binning_scale = int(binning[0])
    
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
    
    return binning_scale, bmjds, filters, gains, ignored_files


def data_checks_output(
    binning_scale: int,
    camera_files: Dict[str, str],
    func: Callable,
    ) -> None:
    """
    Output the results of the data checks.
    
    Parameters
    ----------
    binning_scale : int
        The data binning scale.
    camera_files : Dict[str, str]
        The image files separated by filter.
    func : Callable
        The output function (i.e., `print` or `logger.info`)
    """
    
    func(f'[OPTICAM] Binning: {binning_scale}x{binning_scale}')
    func(f'[OPTICAM] Filters: {", ".join(list(camera_files.keys()))}')
    for fltr in list(camera_files.keys()):
        func(f'[OPTICAM] {len(camera_files[fltr])} {fltr} images.')



















