from pathlib import Path
from typing import Callable, Dict, Tuple


from astropy.table import QTable
import numpy as np
from numpy.typing import NDArray


from opticam.background.global_background import BaseBackground
from opticam.correctors import DarkNoiseCorrector
from opticam.instruments import Instrument
from opticam.photometers import AperturePhotometer
from opticam.utils.constants import counts_to_mag_factor
from opticam.utils.fits_handlers import get_data




def get_sky_stderr(
    N_source: float,
    N_pix: float,
    n_sky: float,
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
    
    Returns
    -------
    float
        The standard error (in magnitudes) of the sky noise.
    """
    
    N_sky = N_pix * n_sky
    
    return counts_to_mag_factor * np.sqrt(N_sky) / N_source


def get_shot_stderr(
    N_source: float,
    ) -> float:
    """
    Get the standard error (in magnitudes) of the shot noise.
    
    Parameters
    ----------
    N_source : float
        The total number of source counts.
    
    Returns
    -------
    float
        The standard error (in magnitudes) of the shot noise.
    """
    
    return counts_to_mag_factor * np.sqrt(N_source) / N_source


def get_dark_stderr(
    N_source: float,
    N_pix: float,
    dark_flux: float,
    ) -> float:
    """
    Get the standard error (in magnitudes) of the dark current noise.
    
    Parameters
    ----------
    N_source : Quantity
        The total number of source counts.
    N_pix : int
        The number of aperture pixels.
    dark_flux : Quantity
        The total number of dark current electrons **per pixel**. 
    
    Returns
    -------
    float
        The standard error (in magnitudes) of the dark noise.
    """
    
    N_dark = N_pix * dark_flux
    
    return counts_to_mag_factor * np.sqrt(N_dark) / N_source


def get_read_stderr(
    N_source: float,
    N_pix: float,
    read_noise: float,
    ) -> float:
    """
    Get the standard error (in magnitudes) of the readout noise.
    
    Parameters
    ----------
    N_source : float
        The total number of source counts.
    N_pix : float
        The number of aperture pixels.
    read_noise : float
        The read noise of the detector in electrons/pixel.
    
    Returns
    -------
    float
        The standard error (in magnitudes) of the readout noise.
    """
    
    p_read = N_pix * read_noise**2
    
    return counts_to_mag_factor * np.sqrt(p_read) / N_source


def snr(
    N_source: float | NDArray,
    N_pix: float,
    n_sky: float,
    dark_flux: float,
    read_noise: float,
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
    read_noise : float
        The read noise of the detector in electrons/pixel.
    
    Returns
    -------
    float | NDArray
        The S/N ratio.
    """
    
    return N_source / np.sqrt(N_source + N_pix * (n_sky + dark_flux + read_noise**2))


def snr_stderr(
    N_source: float | NDArray,
    N_pix: float,
    n_sky: float,
    dark_flux: float,
    read_noise: float,
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
        The dark current's "flux" contribution **per pixel**.
    gain: float
        The detector gain.
    read_noise : float
        The read noise of the detector in electrons/pixel.
    
    Returns
    -------
    float | NDArray
        The standard error (in magnitudes) on the S/N ratio.
    """
    
    p = N_pix * (n_sky + dark_flux + read_noise**2)
    
    return counts_to_mag_factor * np.sqrt(N_source + p) / N_source


def get_noise_params(
    file_path: Path,
    catalog: QTable,
    background: BaseBackground | Callable,
    psf_params: Dict[str, float],
    instrument: Instrument,
    dark_corrector: DarkNoiseCorrector,
    ) -> Tuple[NDArray, NDArray, NDArray, float, float, float]:
    """
    Get the noise values of a science image.
    
    Parameters
    ----------
    file_path : Path
        The path to the science image.
    catalog : QTable
        The source catalog corresponding to the science image.
    background : BaseBackground | Callable
        The background estimator.
    psf_params : Dict[str, float]
        The PSF parameters.
    instrument : Instrument
        The instrument.
    dark_corrector : DarkNoiseCorrector
        The dark noise corrector.
    
    Returns
    -------
    Tuple[NDArray, NDArray, float, float, float]
        The source IDs, fluxes, flux errors, number of aperture pixels, backgorund counts/pixel, and dark flux.
    """
    
    coords = np.asarray([catalog['xcentroid'], catalog['ycentroid']]).T
    
    img = get_data(
        file_path=file_path,
        instrument=instrument,
        dark_corrector=dark_corrector,
        flat_corrector=None,
        rebin_factor=1,
        remove_cosmic_rays=False,
        )[0]
    
    # get (median) dark flux
    fltr = instrument.get_filter(file_path)
    if fltr in dark_corrector.master_images.keys():
        dark_var = dark_corrector.master_variances[fltr]
    else:
        dark_var = instrument.get_dark_flux(file_path)
    
    # global background
    bkg = background(img)
    n_sky = float(bkg.background_rms_median**2)  # background variance
    
    # subtract background from image
    img_clean = img - bkg.background
    
    # perform photometry
    phot = AperturePhotometer()
    phot_results = phot.compute(
        image=img_clean,
        bias_var=0.,
        dark_var=dark_var,
        flat_var=0.,
        background_rms=np.sqrt(n_sky),
        source_coords=coords,
        image_coords=coords,
        psf_params=psf_params,
        read_noise=instrument.read_noise,
        )
    
    # get the number of pixels in the aperture
    N_pix = phot.get_aperture_area(psf_params=psf_params)
    
    fluxes = np.array(phot_results['flux'])
    flux_errs = np.array(phot_results['flux_err'])
    source_ids = np.arange(len(catalog)) + 1
    
    # mask unphysical flux values
    mask = fluxes > 1.
    
    return source_ids[mask], fluxes[mask], flux_errs[mask], N_pix, n_sky, np.median(dark_var)


def get_snrs(
    file_path: Path,
    background: BaseBackground | Callable,
    catalog: QTable,
    psf_params: Dict[str, float],
    instrument: Instrument,
    dark_corrector: DarkNoiseCorrector,
    ) -> Tuple[NDArray, NDArray]:
    """
    Get the S/N ratios for the cataloged sources in a science image.
    
    Parameters
    ----------
    file_path : Path
        The path to the science image.
    background : BaseBackground | Callable
        The background estimator.
    catalog : QTable
        The source catalog corresponding to the science image.
    psf_params : Dict[str, float]
        The PSF parameters.
    instrument : Instrument
        The instrument.
    dark_corrector : DarkNoiseCorrector
        The dark noise corrector.
    
    Returns
    -------
    Tuple[NDArray, NDArray]
        The source IDs and S/N for each source.
    """
    
    source_ids, fluxes, flux_errs, N_pix, n_sky, dark_flux = get_noise_params(
        file_path=file_path,
        catalog=catalog,
        background=background,
        psf_params=psf_params,
        instrument=instrument,
        dark_corrector=dark_corrector,
    )
    
    return source_ids, np.asarray(
        snr(
            N_source=fluxes,
            N_pix=N_pix,
            n_sky=n_sky,
            dark_flux=dark_flux,
            read_noise=instrument.read_noise,
            )
        )


def characterise_noise(
    file_path: Path,
    background: BaseBackground | Callable,
    catalog: QTable,
    psf_params: Dict[str, float],
    instrument: Instrument,
    dark_corrector: DarkNoiseCorrector,
    ) -> Dict[str, NDArray]:
    """
    Characterise the expected noise from an image and compare it to the measured noise for a number of cataloged 
    sources.
    
    Parameters
    ----------
    file_path : Path
        The file path to the science image.
    background : BaseBackground | Callable
        The background estimator.
    catalog : QTable
        The source catalog corresponding to the science image.
    psf_params : Dict[str, float]
        The PSF parameters.
    instrument : Instrument
        The instrument.
    dark_corrector : DarkNoiseCorrector
        The dark noise corrector.
    
    Returns
    -------
    Dict[str, NDArray]
        The noies properties.
    """
    
    source_ids, fluxes, flux_errs, N_pix, n_sky, dark_flux = get_noise_params(
        file_path=file_path,
        catalog=catalog,
        background=background,
        psf_params=psf_params,
        instrument=instrument,
        dark_corrector=dark_corrector,
    )
    
    N_source = np.logspace(
        np.log10(np.min(fluxes) / 1.5),
        np.log10(np.max(fluxes) * 1.5),
        100,
        )
    
    results = {}
    
    results['model_mags'] = -2.5 * np.log10(N_source)
    results['effective_noise'] = snr_stderr(N_source, N_pix, n_sky, dark_flux, read_noise=instrument.read_noise)
    results['sky_noise'] = get_sky_stderr(N_source, N_pix, n_sky)
    results['shot_noise'] = get_shot_stderr(N_source)
    results['dark_noise'] = get_dark_stderr(N_source, N_pix, dark_flux)
    results['read_noise'] = get_read_stderr(N_source, N_pix, read_noise=instrument.read_noise)
    
    results['measured_mags'] = -2.5 * np.log10(fluxes)
    results['measured_noise'] = counts_to_mag_factor * flux_errs / fluxes
    results['expected_measured_noise'] = snr_stderr(fluxes, N_pix, n_sky, dark_flux, read_noise=instrument.read_noise)
    
    return results



















