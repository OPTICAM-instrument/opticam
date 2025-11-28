from typing import Callable, Dict, Tuple

from astropy.table import QTable
import numpy as np
from numpy.typing import NDArray

from opticam.background.global_background import BaseBackground
from opticam.photometers import AperturePhotometer
from opticam.utils.constants import counts_to_mag_factor, n_read
from opticam.utils.fits_handlers import get_image_noise_info


def get_source_photons(
    N_source: float | NDArray,
    gain: float,
    ) -> float | NDArray:
    """
    Get the number of source photons.
    
    Parameters
    ----------
    N_source : float | NDArray
        The number of source counts.
    gain : float
        The gain.
    
    Returns
    -------
    float
        The number of source photons.
    """
    
    return N_source * gain

def get_sky_photons_per_pixel(
    n_sky: float,
    gain: float,
    ) -> float:
    """
    Get the number of sky photons per pixel.
    
    Parameters
    ----------
    n_sky : float
        The sky counts per pixel.
    gain : float
        The gain.
    
    Returns
    -------
    float
        The number of sky photons per pixel.
    """
    
    return n_sky * gain


def get_sky_stderr(
    N_source: float,
    N_pix: float,
    n_sky: float,
    gain: float,
    ) -> float:
    """
    Get the standard error (in magnitudes) of the sky noise.
    
    Parameters
    ----------
    N_source : float
        The total number of source counts.
    N_pix : int
        The number of aperture pixels.
    n_sky : float
        The number of sky counts **per pixel**.
    gain: float
        The detector gain.
    
    Returns
    -------
    float
        The standard error (in magnitudes) of the sky noise.
    """
    
    source_photons = get_source_photons(N_source=N_source, gain=gain)
    sky_photons_per_pix = get_sky_photons_per_pixel(n_sky=n_sky, gain=gain)
    
    p_sky = N_pix * sky_photons_per_pix
    
    return counts_to_mag_factor * np.sqrt(p_sky) / source_photons

def get_shot_stderr(
    N_source: float,
    gain: float,
    ) -> float:
    """
    Get the standard error (in magnitudes) of the shot noise.
    
    Parameters
    ----------
    N_source : float
        The total number of source counts.
    gain: float
        The detector gain.
    
    Returns
    -------
    float
        The standard error (in magnitudes) of the shot noise.
    """
    
    source_photons = get_source_photons(N_source=N_source, gain=gain)
    
    return counts_to_mag_factor * np.sqrt(source_photons) / source_photons

def get_dark_stderr(
    N_source: float,
    N_pix: float,
    dark_flux: float,
    gain: float,
    ) -> float:
    """
    Get the standard error (in magnitudes) of the dark current noise.
    
    Parameters
    ----------
    N_source : Quantity
        The total number of source counts.
    N_pix : int
        The number of aperture pixels.
    dark_curr : Quantity
        The number of dark current electrons **per pixel per unit time**. 
    t_exp: Quantity
        The exposure time.
    gain: Quantity
        The detector gain.
    
    Returns
    -------
    float
        The standard error (in magnitudes) of the dark current noise.
    """
    
    source_photons = get_source_photons(N_source=N_source, gain=gain)
    
    p_dark = N_pix * dark_flux
    
    return counts_to_mag_factor * np.sqrt(p_dark) / source_photons

def get_read_stderr(
    N_source: float,
    N_pix: float,
    gain: float,
    ) -> float:
    """
    Get the standard error (in magnitudes) of the readout noise.
    
    Parameters
    ----------
    N_source : float
        The total number of source counts.
    N_pix : float
        The number of aperture pixels.
    gain: float
        The detector gain.
    
    Returns
    -------
    float
        The standard error (in magnitudes) of the readout noise.
    """
    
    source_photons = get_source_photons(N_source=N_source, gain=gain)
    
    p_read = N_pix * n_read**2
    
    return counts_to_mag_factor * np.sqrt(p_read) / source_photons


def snr(
    N_source: float | NDArray,
    N_pix: float,
    n_sky: float,
    dark_flux: float,
    gain: float,
    ) -> float | NDArray:
    """
    The (simplified) S/N ratio equation or CCD Equation (see Chapter 4.4 of Handbook of CCD Astronomy by Howell, 2006).
    
    Parameters
    ----------
    N_source : float | NDArray
        The total number of source counts.
    N_pix : int
        The number of aperture pixels.
    n_sky : float
        The number of sky counts **per pixel**.
    dark_flux : float
        The dark current's "flux" contribution per pixel.
    gain: float
        The detector gain.
    
    Returns
    -------
    float | NDArray
        The S/N ratio.
    """
    
    source_photons = get_source_photons(N_source=N_source, gain=gain)
    sky_photons_per_pix = get_sky_photons_per_pixel(n_sky=n_sky, gain=gain)
    read_counts_per_pixel = n_read
    
    return source_photons / np.sqrt(source_photons + N_pix * (sky_photons_per_pix + dark_flux + read_counts_per_pixel**2))

def snr_stderr(
    N_source: float | NDArray,
    N_pix: float,
    n_sky: float,
    dark_flux: float,
    gain: float,
    ) -> float | NDArray:
    """
    The standard error (in magnitudes) on the CCD Equation (see Chapter 4.4 of Handbook of CCD Astronomy by Howell, 
    2006).
    
    Parameters
    ----------
    N_source : float | NDArray
        The total number of source counts.
    N_pix : float
        The number of aperture pixels.
    n_sky : float
        The number of sky counts **per pixel**.
    dark_flux : float
        The dark current's "flux" contribution per pixel.
    gain: float
        The detector gain.
    
    Returns
    -------
    float | NDArray
        The standard error (in magnitudes) on the S/N ratio.
    """
    
    source_photons = get_source_photons(N_source=N_source, gain=gain)
    sky_photons_per_pix = get_sky_photons_per_pixel(n_sky=n_sky, gain=gain)
    
    p = N_pix * (sky_photons_per_pix + dark_flux + n_read**2)
    
    return counts_to_mag_factor * np.sqrt(source_photons + p) / source_photons


def get_noise_params(
    file: str,
    catalog: QTable,
    background: BaseBackground | Callable,
    psf_params: Dict[str, float],
    ) -> Tuple[NDArray, NDArray, float, float, float, float]:
    """
    Get the noise values of a science image.
    
    Parameters
    ----------
    file : str
        The path to the science image.
    catalog : QTable
        The source catalog corresponding to the science image.
    background : BaseBackground | Callable
        The background estimator.
    psf_params : Dict[str, float]
        The PSF parameters.
    
    Returns
    -------
    Tuple[NDArray, NDArray, float, float, float, float]
        The fluxes, flux errors, number of aperture pixels, backgorund counts/pixel, dark flux, and gain.
    """
    
    coords = np.asarray([catalog['xcentroid'], catalog['ycentroid']]).T
    
    img, dark_flux, gain = get_image_noise_info(file)
    
    # global background
    bkg = background(img)
    n_sky = float(bkg.background_rms_median**2)  # background variance
    
    # subtract background from image
    img_clean = img - bkg.background
    
    # perform photometry
    phot = AperturePhotometer()
    phot_results = phot.compute(
        image=img_clean,
        dark_flux=dark_flux,
        background_rms=np.sqrt(n_sky),
        source_coords=coords,
        image_coords=coords,
        psf_params=psf_params,
        )
    
    # get the number of pixels in the aperture
    N_pix = phot.get_aperture_area(psf_params=psf_params)
    
    fluxes = np.array(phot_results['flux'])
    flux_errs = np.array(phot_results['flux_err'])
    
    return fluxes, flux_errs, N_pix, n_sky, dark_flux, gain

def get_snrs(
    file: str,
    background: BaseBackground | Callable,
    catalog: QTable,
    psf_params: Dict[str, float],
    ) -> NDArray:
    """
    Get the S/N ratios for the cataloged sources in a science image.
    
    Parameters
    ----------
    file : str
        The path to the science image.
    background : BaseBackground | Callable
        The background estimator.
    catalog : QTable
        The source catalog corresponding to the science image.
    psf_params : Dict[str, float]
        The PSF parameters.
    
    Returns
    -------
    NDArray
        The S/N for each source. Sources are ordered as they appear in `catalog`.
    """
    
    fluxes, flux_errs, N_pix, n_sky, dark_flux, gain = get_noise_params(
        file=file,
        catalog=catalog,
        background=background,
        psf_params=psf_params,
    )
    
    return np.asarray(
        snr(
            N_source=fluxes,
            N_pix=N_pix,
            n_sky=n_sky,
            dark_flux=dark_flux,
            gain=gain,
            )
        )

def characterise_noise(
    file: str,
    background: BaseBackground | Callable,
    catalog: QTable,
    psf_params: Dict[str, float],
    ) -> Dict[str, NDArray]:
    """
    Characterise the expected noise from an image and compare it to the measured noise for a number of cataloged 
    sources.
    
    Parameters
    ----------
    file : str
        The file path to the science image.
    background : BaseBackground | Callable
        The background estimator.
    catalog : QTable
        The source catalog corresponding to the science image.
    psf_params : Dict[str, float]
        The PSF parameters.
    
    Returns
    -------
    Dict[str, NDArray]
        The noies properties.
    """
    
    fluxes, flux_errs, N_pix, n_sky, dark_flux, gain = get_noise_params(
        file=file,
        catalog=catalog,
        background=background,
        psf_params=psf_params,
    )
    
    N_source = np.logspace(
        np.log10(np.min(fluxes) / 1.5),
        np.log10(np.max(fluxes) * 1.5),
        100,
        )
    
    results = {}
    
    results['model_mags'] = -2.5 * np.log10(N_source)
    results['effective_noise'] = snr_stderr(N_source, N_pix, n_sky, dark_flux, gain)
    results['sky_noise'] = get_sky_stderr(N_source, N_pix, n_sky, gain)
    results['shot_noise'] = get_shot_stderr(N_source, gain)
    results['dark_noise'] = get_dark_stderr(N_source, N_pix, dark_flux, gain)
    results['read_noise'] = get_read_stderr(N_source, N_pix, gain)
    
    results['measured_mags'] = -2.5 * np.log10(fluxes)
    results['measured_noise'] = counts_to_mag_factor * flux_errs / fluxes
    results['expected_measured_noise'] = snr_stderr(fluxes, N_pix, n_sky, dark_flux, gain)
    
    return results



















