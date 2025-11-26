from logging import Logger
from typing import Dict, Tuple

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.io.fits import Header
from astropy.time import Time
from astropy import units as u
from ccdproc import cosmicray_lacosmic  # TODO: replace with astroscrappy to reduce dependencies?
import numpy as np
from numpy.typing import NDArray
import os.path

from opticam.correctors.flat_field_corrector import FlatFieldCorrector
from opticam.utils.time_helpers import apply_barycentric_correction
from opticam.utils.image_helpers import rebin_image


def get_header_info(
    file: str,
    barycenter: bool,
    logger: Logger | None,
    ) -> Tuple[float | None, str | None, str | None, float | None, float | None]:
    """
    Get the BMJD, filter, binning, gain, and dark current from a file header.
    
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
    Tuple[float | None, str | None, str | None, float | None, float | None]
        The BMJD, filter, binning, gain, and dark current.
    """
    
    try:
        header = get_header(file)
        
        dark_curr = float(header["DARKCURR"])  # type: ignore
        binning = str(header["BINNING"])
        gain = float(header["GAIN"])  # type: ignore
        
        try:
            ra = header["RA"]
            dec = header["DEC"]
        except:
            if logger:
                logger.info(f"[OPTICAM] Could not find RA and DEC keys in {file} header.")
            pass
        
        mjd = get_time(header, file)
        fltr = str(header["FILTER"])
        
        if barycenter:
            try:
                # try to compute barycentric dynamical time
                coords = SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
                bmjd = float(apply_barycentric_correction(mjd, coords))
                return bmjd, fltr, binning, gain, dark_curr
            except Exception as e:
                if logger:
                    logger.info(f"[OPTICAM] Could not compute BMJD for {file}: {e}. Skipping.")
                return None, None, None, None, None
    except Exception as e:
        if logger:
            logger.info(f'[OPTICAM] Could not read {file}: {e}. Skipping.')
        return None, None, None, None, None
    
    return mjd, fltr, binning, gain, dark_curr


def get_time(
    header: Header,
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
    ) -> Tuple[NDArray, float]:
    """
    Get the image data from a FITS file. The dark current "flux" contribution is automatically computed and subtracted
    from the returned image.
    
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
    Tuple[NDArray, float]
        The image data (after subtracting the dark current "flux" contribution) and the dark current "flux"
        contribution.
    
    Raises
    ------
    ValueError
        If `file` could not be opened.
    """
    
    try:
        with fits.open(file) as hdul:
            data = np.array(hdul[0].data, dtype=np.float64)
            fltr = hdul[0].header["FILTER"] + '-band'
            dark_curr = float(hdul[0].header["DARKCURR"])  # [electrons/pixel/s]
            t_exp = float(hdul[0].header['EXPOSURE'])  # [s]
    except Exception as e:
        raise ValueError(f"[OPTICAM] Could not get image data from {file} due to the following exception: {e}.")
    
    # subtract dark current
    dark_flux = dark_curr * t_exp
    data -= dark_flux
    
    if flat_corrector:
        data = flat_corrector.correct(data, fltr)
    
    if remove_cosmic_rays:
        data = np.asarray(cosmicray_lacosmic(data, gain_apply=False)[0])
    
    if rebin_factor > 1:
        data = rebin_image(data, rebin_factor)
    
    return data, dark_flux


def get_header(
    file: str,
    ) -> fits.Header:
    """
    Get the header of a FITS file.
    
    Parameters
    ----------
    file : str
        _description_

    Returns
    -------
    fits.Header
        _description_
    """
    
    try:
        with fits.open(file) as hdul:
            header = hdul[0].header
    except Exception as e:
        raise ValueError(f"[OPTICAM] Could not get header information from {file} due to the following exception: {e}.")
    
    return header


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
    ) -> Tuple[NDArray, float, float]:
    """
    Given a FITS file, get the image and corresponding filter, exposure time, dark current, and gain.
    
    Parameters
    ----------
    file_path : str
        The path to the FITS file.
    
    Returns
    -------
    Tuple[NDArray, float, Quantity]
        The image, dark flux, and gain.
    """
    
    gain = float(get_header(file=file_path)['GAIN'])  # type: ignore
    
    img, dark_flux = get_data(
        file=file_path,
        flat_corrector=None,
        rebin_factor=1,
        remove_cosmic_rays=False,
        )
    
    return img, dark_flux, gain























