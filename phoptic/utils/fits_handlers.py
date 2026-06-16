import os.path
from pathlib import Path
from typing import Callable


from astropy.io import fits
from astropy.io.fits import Header
from ccdproc import cosmicray_lacosmic
import numpy as np
from numpy.typing import NDArray


from phoptic.correctors import BiasCorrector, DarkNoiseCorrector, FlatFieldCorrector
from phoptic.mef_slice import MEFSlice
from phoptic.timing.helpers import apply_barycentric_correction
from phoptic.utils.helpers import camera_and_filter_key
from phoptic.utils.image_helpers import rebin_image
from phoptic.instruments import Instrument




def get_header_info(
    file: MEFSlice,
    instrument: Instrument,
    barycenter: bool,
    ) -> tuple[float, float, str, str, str]:
    """
    Get the timestamp, exposure length, filter, and binning of the file.
    
    Parameters
    ----------
    file : MEFSlice
        The `MEFSlice` instance representing the file.
    instrument : Instrument
        The instrument that created the file.
    barycenter : bool
        Whether to apply a Barycentric correction to the image's timestamp.
    
    Returns
    -------
    Tuple[float, float, str, str, str, float]
        The timestamp, exposure length, camera, filter, and binning of the image.
    """
    
    header = file.get_header()
    exposure = instrument.get_exptime(header=header)
    camera = instrument.get_camera(header=header)
    fltr = instrument.get_filter(header=header)
    binning = instrument.get_binning(header=header)
    timestamp = instrument.get_mjd(header=header)
    
    if barycenter:
        sky_coords = instrument.get_sky_coord(header=header)
        timestamp = float(apply_barycentric_correction(
            timestamp,
            sky_coords,
            instrument=instrument,
            ))
    
    return timestamp, exposure, camera, fltr, binning


def get_data(
    file: MEFSlice,
    instrument: Instrument,
    rebin_factor: int,
    image_filter: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None,
    remove_cosmic_rays: bool,
    bias_corrector: BiasCorrector | None = None,
    dark_corrector: DarkNoiseCorrector | None = None,
    flat_corrector: FlatFieldCorrector | None = None,
    ) -> tuple[
        NDArray[np.float64],
        Header,
        dict[str, float | NDArray]
        ]:
    """
    Get the (calibrated) image data from a file.
    
    Parameters
    ----------
    file : MEFSlice
        The `MEFSlice` instance representing the file.
    instrument : Instrument
        The instrument that created the file.
    rebin_factor : int
        The image rebinning factor.
    image_filter : Callable[[NDArray[np.float64]], NDArray[np.float64]] | None
        The filter to apply to the image as it is opened.
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
    Tuple[NDArray[np.float64], Header, dict[str, float | NDArray]]
        The corrected image, the image header, and the noise dictionary.
    """
    
    data, header = file.get_data_and_header()
    camera = instrument.get_camera(header=header)
    fltr = instrument.get_filter(header=header)
    key = camera_and_filter_key(camera, fltr)
    
    if image_filter is not None:
        data = image_filter(data)
    
    noise_dict: dict[str, float | NDArray] = {}
    
    ################################################# bias correction #################################################
    
    if bias_corrector is not None:
        data, bias_var = bias_corrector.correct(
            image=data,
            camera=instrument.get_camera(header=header),
            )
    else:
        bias_var = 0.
    
    noise_dict['bias_var'] = bias_var
    
    ############################################## dark noise correction ##############################################
    
    if dark_corrector is not None:
        data, dark_var = dark_corrector.correct(
            image=data,
            key=key,
            dark_flux=instrument.get_dark_flux(header=header),
            )
    else:
        dark_var = 0.
    
    noise_dict['dark_var'] = dark_var
    
    ############################################## flat-field correction ##############################################
    
    if flat_corrector is not None:
        data, flat_var = flat_corrector.correct(
            image=data,
            key=key,
            )
    else:
        flat_var = 0.
    
    noise_dict['flat_var'] = flat_var
    
    ################################################# clip cosmic rays #################################################
    
    if remove_cosmic_rays:
        data = np.asarray(cosmicray_lacosmic(data, gain_apply=False)[0])
    
    ###################################################### rebin ######################################################
    
    if rebin_factor > 1:
        data = rebin_image(image=data, factor=rebin_factor)
    
    noise_dict['rel_scint_noise'] = instrument.get_relative_scintillation_noise(header=header)
    
    return data, header, noise_dict


def save_stacked_images(
    stacked_images: dict[str, NDArray],
    out_directory: Path,
    overwrite: bool,
    ) -> None:
    """
    Save the stacked images to a compressed multi-extension FITS file.
    
    Parameters
    ----------
    stacked_images : dict[str, NDArray]
        The stacked images {filter: stacked image}.
    out_directory : Path
        The path to the directory in which the stacked images will be saved.
    overwrite : bool
        Whether to overwrite the file if it already exists.
    """
    
    hdr = fits.Header()
    hdr['COMMENT'] = 'This FITS file contains stacked images for each filter.'
    hdul = fits.HDUList([fits.PrimaryHDU(header=hdr)])
    
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
    ) -> dict[str, NDArray[np.float64]]:
    """
    Unpacked the stacked catalog images from `out_directory/cat/stacked_images.fits.gz`.
    
    Parameters
    ----------
    out_directory : Path
        The path to the directory containing the stacked images.
    
    Returns
    -------
    Dict[str, NDArray[np.float64]]
        The stacked images {filter: image}.
    """
    
    stacked_images = {}
    with fits.open(os.path.join(out_directory, 'cat/stacked_images.fits.gz')) as hdul:
        for hdu in hdul:
            if 'FILTER' not in hdu.header:
                continue
            fltr = hdu.header['FILTER']
            stacked_images[fltr] = np.asarray(hdu.data, dtype=np.float64)
    
    return stacked_images























