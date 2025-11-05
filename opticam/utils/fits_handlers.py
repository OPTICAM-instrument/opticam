from logging import Logger
from typing import Dict, Tuple

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy import units as u
from astropy.units import Quantity
from ccdproc import cosmicray_lacosmic  # TODO: replace with astroscrappy to reduce dependencies?
import numpy as np
from numpy.typing import ArrayLike, NDArray
import os.path

from opticam.correctors.flat_field_corrector import FlatFieldCorrector
from opticam.utils.time_helpers import apply_barycentric_correction
from opticam.utils.image_helpers import rebin_image


def get_header_info(
    file: str,
    barycenter: bool,
    logger: Logger | None,
    ) -> Tuple[ArrayLike | None, str | None, str | None, float | None]:
    """
    Get the BMJD, filter, binning, and gain from a file header.
    
    Parameters
    ----------
    file : str
        The file path.
    barycenter : bool
        Whether to apply a Barycentric correction to the image time stamps.
    logger : Logger | None
        The logger.
    
    Returns
    -------
    Tuple[float, str, str, float]
        The BMJD, filter, binning, and gain dictionaries.
    """
    
    try:
        with fits.open(file) as hdul:
            header = hdul[0].header
        
        binning = header["BINNING"]
        gain = header["GAIN"]
        
        try:
            ra = header["RA"]
            dec = header["DEC"]
        except:
            if logger:
                logger.info(f"[OPTICAM] Could not find RA and DEC keys in {file} header.")
            pass
        
        mjd = get_time(header, file)
        fltr = header["FILTER"]
        
        if barycenter:
            try:
                # try to compute barycentric dynamical time
                coords = SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
                bmjd = apply_barycentric_correction(mjd, coords)
                return bmjd, fltr, binning, gain
            except Exception as e:
                if logger:
                    logger.info(f"[OPTICAM] Could not compute BMJD for {file}: {e}. Skipping.")
                return None, None, None, None
    except Exception as e:
        if logger:
            logger.info(f'[OPTICAM] Could not read {file}: {e}. Skipping.')
        return None, None, None, None
    
    return mjd, fltr, binning, gain


def get_time(
    header: Dict,
    file: str,
    ) -> float:
    """
    Parse the time from the header of a FITS file.
    
    Parameters
    ----------
    header
        The FITS file header.
    file : str
        The path to the file.
    
    Returns
    -------
    float
        The time of the observation in MJD.
    
    Raises
    ------
    ValueError
        If the time cannot be parsed from the header.
    KeyError
        If neither 'GPSTIME' nor 'UT' keys are found in the header.
    """
    
    if "GPSTIME" in header.keys():
        gpstime = header["GPSTIME"]
        split_gpstime = gpstime.split(" ")
        date = split_gpstime[0]
        time = split_gpstime[1]
        mjd = Time(date + "T" + time, format="fits").mjd
    elif "UT" in header.keys():
        try:
            mjd = Time(header["UT"].replace(" ", "T"), format="fits").mjd
        except:
            try:
                date = header['DATE-OBS']
                time = header['UT'].split('.')[0]
                mjd = Time(date + 'T' + time, format='fits').mjd
            except:
                raise ValueError('Could not parse time from ' + file + ' header.')
    else:
        raise KeyError(f"[OPTICAM] Could not find GPSTIME or UT key in {file} header.")
    
    return float(mjd)


def get_data(
    file: str,
    flat_corrector: FlatFieldCorrector | None,
    rebin_factor: int,
    remove_cosmic_rays: bool,
    ) -> NDArray:
    """
    Get the image data from a FITS file.
    
    Parameters
    ----------
    file : str
        The file.
    flat_corrector : FlatFieldCorrector | None
        The `FlatFieldCorrector` instance (if specified).
    rebin_factor : int
        The rebin factor.
    remove_cosmic_rays : bool
        Whether to remove cosmic rays from the image.
    
    Returns
    -------
    NDArray
        The data.
    
    Raises
    ------
    ValueError
        If `file` could not be opened.
    """
    
    try:
        with fits.open(file) as hdul:
            data = np.array(hdul[0].data, dtype=np.float64)
            fltr = hdul[0].header["FILTER"] + '-band'
    except Exception as e:
        raise ValueError(f"[OPTICAM] Could not open file {file} due to the following exception: {e}.")
    
    if flat_corrector:
        data = flat_corrector.correct(data, fltr)
    
    # remove cosmic rays if required
    if remove_cosmic_rays:
        data = np.asarray(cosmicray_lacosmic(data, gain_apply=False)[0])
    
    if rebin_factor > 1:
        data = rebin_image(data, rebin_factor)
    
    return data


def save_stacked_images(
    stacked_images: Dict[str, NDArray],
    out_directory: str,
    overwrite: bool,
    ) -> None:
    """
    Save the stacked images to a compressed FITS cube.
    
    Parameters
    ----------
    stacked_images : Dict[str, NDArray]
        The stacked images (filter: stacked image).
    """
    
    hdr = fits.Header()
    hdr['COMMENT'] = 'This FITS file contains stacked images for each filter.'
    empty_primary = fits.PrimaryHDU(header=hdr)
    hdul = fits.HDUList([empty_primary])
    
    for fltr, img in stacked_images.items():
        hdr = fits.Header()
        hdr['FILTER'] = fltr
        hdu = fits.ImageHDU(img, hdr)
        hdul.append(hdu)
    
    file_path = os.path.join(out_directory, f'cat/stacked_images.fits.gz')
    
    if not os.path.isfile(file_path) or overwrite:
        hdul.writeto(file_path, overwrite=overwrite)


def get_stacked_images(
    out_directory: str,
    ) -> Dict[str, NDArray]:
    """
    Unpacked the stacked catalog images from out_directory/cat.
    
    Parameters
    ----------
    out_directory : str
        The directory path to the reduction output.
    
    Returns
    -------
    Dict[str, NDArray]
        The stacked images {filter: image}.
    """
    
    stacked_images = {}
    with fits.open(os.path.join(out_directory, 'cat/stacked_images.fits.gz')) as hdul:
        for hdu in hdul:
            if 'FILTER' not in hdu.header:
                continue
            fltr = hdu.header['FILTER']
            stacked_images[fltr] = np.asarray(hdu.data)
    
    return stacked_images


def get_image_noise_info(
    file_path: str,
    ) -> Tuple[NDArray, Quantity, Quantity, Quantity]:
    """
    Given a FITS file, get the image and corresponding filter, exposure time, dark current, and gain.
    
    Parameters
    ----------
    file_path : str
        The path to the FITS file.
    
    Returns
    -------
    Tuple[NDArray, Quantity, Quantity, Quantity]
        The image, exposure time, dark current, and gain.
    
    Raises
    ------
    ValueError
        If the file could not be parsed.
    """
    
    try:
        with fits.open(file_path) as hdul:
            img = np.array(hdul[0].data, dtype=np.float64)  # type: ignore
            t_exp = float(hdul[0].header['EXPOSURE']) * u.s  # type: ignore
            dark_curr = float(hdul[0].header['DARKCURR']) * u.adu / u.pix / u.s  # type: ignore
            gain = float(hdul[0].header['GAIN']) * u.ph / u.adu  # type: ignore
    except Exception as e:
        raise ValueError(f"[OPTICAM] Could not parse file {file_path} due to the following exception: {e}.")
    
    return img, t_exp, dark_curr, gain























