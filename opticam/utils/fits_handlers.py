from logging import Logger
from pathlib import Path
from typing import Dict, Tuple


from astropy.io import fits
from astropy.io.fits import Header
from ccdproc import cosmicray_lacosmic
import numpy as np
from numpy.typing import NDArray
import os.path


from opticam.correctors import BiasCorrector, DarkNoiseCorrector, FlatFieldCorrector
from opticam.utils.image_helpers import rebin_image
from opticam.utils.time_helpers import apply_barycentric_correction
from opticam.instruments import Instrument




def get_header_info(
    file: Path,
    instrument: Instrument,
    barycenter: bool,
    ) -> Tuple[float | None, float | None, str | None, str | None]:
    """
    Get the timestamp, exposure length, filter, binning, and dark current from a file header.
    
    Parameters
    ----------
    file : str
        The file path.
    instrument : Instrument
        The instrument.
    barycenter : bool
        Whether to apply a Barycentric correction to the image's timestamp.
    logger : Logger | None
        The logger.
    
    Returns
    -------
    Tuple[float | None, float | None, str | None, str | None, float | None]
        The timestamp, exposure length, filter, and binning of the image.
    """
    
    binning = str(instrument.get_binning(file))
    
    header: Header = fits.getheader(file)
    
    exposure = float(header[instrument.exptime_kw])
    fltr = instrument.get_filter(header=header)
    
    timestamp = instrument.get_mjd(file)
    
    if barycenter:
        sky_coords = instrument.get_sky_coord(file)
        timestamp = float(apply_barycentric_correction(
            timestamp,
            sky_coords,
            instrument=instrument,
            ))
    
    return timestamp, exposure, fltr, binning


def get_data(
    file_path: Path,
    instrument: Instrument,
    rebin_factor: int,
    remove_cosmic_rays: bool,
    bias_corrector: BiasCorrector | None = None,
    dark_corrector: DarkNoiseCorrector | None = None,
    flat_corrector: FlatFieldCorrector | None = None,
    ) -> Tuple[NDArray[np.float64], float | NDArray[np.float64], float | NDArray[np.float64],
               float | NDArray[np.float64]]:
    """
    Given the path to a FITS file, get the image data and perform and required corrections.
    
    Parameters
    ----------
    file : Path
        The path to the FITS file.
    instrument : Instrument
        The instrument that produced the FITS file.
    rebin_factor : int
        The image rebinning factor.
    remove_cosmic_rays : bool
        Whether to remove cosmic rays from the image.
    bias_corrector : BiasCorrector | None, optional
        The bias corrector, by default `None`. If `None`, no bias corrections are performed.
    dark_corrector : DarkNoiseCorrector | None, optional
        The dark noise corrector, by default `None`. If `None`, no dark noise corrections are performed.
    flat_corrector : FlatFieldCorrector | None, optional
        The flat-field corrector, by default `None`. If `None`, no flat-field corrections are performed.
    
    Returns
    -------
    Tuple[NDArray[np.float64], float | NDArray[np.float64], float | NDArray[np.float64], float | NDArray[np.float64]]
        The corrected image and the master bias, dark, and flat variances. If any of the correctors are undefined,
        the variance of that corrector is set to 0.
    """
    
    try:
        with fits.open(file_path) as hdul:
            header: Header = hdul[0].header
            data: NDArray[np.float64] = np.asarray(hdul[0].data, dtype=np.float64)
    except Exception as e:
        raise ValueError(f"[OPTICAM] Could not open {file_path} due to the following exception: {e}.")
    
    fltr = instrument.get_filter(header=header)
    
    ################################################# bias correction #################################################
    
    if bias_corrector is not None:
        data, bias_var = bias_corrector.correct(
            image=data,
            fltr=fltr,
            )
    else:
        bias_var = 0.
    
    ############################################## dark noise correction ##############################################
    
    if dark_corrector is not None:
        data, dark_var = dark_corrector.correct(
            image=data,
            fltr=fltr,
            bias_corrector=bias_corrector,
            dark_flux=instrument.get_dark_flux(file_path),
            )
    else:
        dark_var = 0.
    
    ############################################## flat-field correction ##############################################
    
    if flat_corrector is not None:
        data, flat_var = flat_corrector.correct(
            image=data,
            fltr=fltr,
            bias_corrector=bias_corrector,
            )
    else:
        flat_var = 0.
    
    ################################################# clip cosmic rays #################################################
    
    if remove_cosmic_rays:
        data = np.asarray(cosmicray_lacosmic(data, gain_apply=False)[0])
    
    ###################################################### rebin ######################################################
    
    if rebin_factor > 1:
        data = rebin_image(data, rebin_factor)
    
    return data, bias_var, dark_var, flat_var


def save_stacked_images(
    stacked_images: Dict[str, NDArray],
    out_directory: Path,
    overwrite: bool,
    ) -> None:
    """
    Save the stacked images to a compressed FITS cube.
    
    Parameters
    ----------
    stacked_images : Dict[str, NDArray]
        The stacked images (filter: stacked image).
    out_directory : Path
        The path to the directory in which the stacked images are saved.
    overwrite : bool
        Whether to overwrite existing stacked images.
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
    out_directory: Path,
    ) -> Dict[str, NDArray]:
    """
    Unpacked the stacked catalog images from out_directory/cat.
    
    Parameters
    ----------
    out_directory : Path
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























