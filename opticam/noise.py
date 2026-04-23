from typing import Callable


from astropy.table import QTable
import numpy as np
from numpy.typing import NDArray


from opticam.background.global_background import BaseBackground
from opticam.correctors import BiasCorrector, DarkNoiseCorrector, FlatFieldCorrector
from opticam.instruments import Instrument
from opticam.photometers import AperturePhotometer
from opticam.utils.constants import counts_to_mag_factor
from opticam.mef_slice import MEFSlice
from opticam.utils.fits_handlers import get_data
from opticam.utils.helpers import camera_and_filter_key




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


def get_bias_stderr(
    N_source: float,
    N_pix: float,
    bias_var: float,
    ) -> float:
    """
    Get the standard error (in magnitudes) of the bias variance.
    
    Parameters
    ----------
    N_source : Quantity
        The total number of source counts.
    N_pix : int
        The number of aperture pixels.
    bias_var : Quantity
        The bias variance.
    
    Returns
    -------
    float
        The standard error (in magnitudes) of the bias variance.
    """
    
    N_bias = N_pix * bias_var
    
    return counts_to_mag_factor * np.sqrt(N_bias) / N_source


def get_dark_stderr(
    N_source: float,
    N_pix: float,
    dark_var: float,
    ) -> float:
    """
    Get the standard error (in magnitudes) of the dark noise variance.
    
    Parameters
    ----------
    N_source : float
        The total number of source counts.
    N_pix : int
        The number of aperture pixels.
    dark_var : float
        The dark noise variance.
    
    Returns
    -------
    float
        The standard error (in magnitudes) of the dark noise.
    """
    
    N_dark = N_pix * dark_var
    
    return counts_to_mag_factor * np.sqrt(N_dark) / N_source


def get_flat_stderr(
    N_source: float,
    N_pix: float,
    flat_var: float,
    ) -> float:
    """
    Get the standard error (in magnitudes) of the flat-field variance.
    
    Parameters
    ----------
    N_source : float
        The total number of source counts.
    N_pix : int
        The number of aperture pixels.
    flat_var : float
        The flat-field variance.
    
    Returns
    -------
    float
        The standard error (in magnitudes) of the flat-field variance.
    """
    
    N_flat = N_pix * flat_var
    
    return counts_to_mag_factor * np.sqrt(N_flat) / N_source


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


def get_scint_stderr(
    N_source: float | NDArray,
    rel_scint_noise: float,
    ) -> float | NDArray:
    """
    Get the standard error (in magnitudes) of the scintillation noise.
    
    Parameters
    ----------
    N_source : float | NDArray
        The total number of source counts.
    rel_scint_noise : float
        The relative scintillation noise.
    
    Returns
    -------
    float | NDArray
        The standard error (in magnitudes) of the scintillation noise.
    """
    
    return counts_to_mag_factor * (np.zeros(len(N_source)) + rel_scint_noise)


def snr(
    N_source: float | NDArray,
    N_pix: float,
    n_sky: float,
    bias_var: float,
    dark_var: float,
    flat_var: float,
    read_noise: float,
    scint_noise: NDArray,
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
    bias_var : float
        The bias variance.
    dark_var : float
        The dark noise variance.
    flat_var : float
        The flat-field variance.
    read_noise : float
        The read noise of the detector in electrons/pixel.
    scint_noise : NDArray
        The scintillation noise.
    
    Returns
    -------
    float | NDArray
        The S/N ratio.
    """
    
    return N_source / np.sqrt(N_source + scint_noise**2 + N_pix * (n_sky + bias_var + dark_var + flat_var + read_noise**2))


def snr_stderr(
    N_source: float | NDArray,
    N_pix: float,
    n_sky: float,
    bias_var: float,
    dark_var: float,
    flat_var: float,
    read_noise: float,
    rel_scint_noise: float,
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
    bias_var : float
        The bias variance.
    dark_var : float
        The dark noise variance.
    flat_var : float
        The flat-field variance.
    read_noise : float
        The read noise of the detector in electrons/pixel.
    rel_scint_noise : float
        The relative scintillation noise.
    
    Returns
    -------
    float | NDArray
        The standard error (in magnitudes) on the S/N ratio.
    """
    
    p = N_pix * (n_sky + bias_var + dark_var + flat_var + read_noise**2)
    
    scint_noise = rel_scint_noise * N_source
    
    return counts_to_mag_factor * np.sqrt(N_source + scint_noise**2 + p) / N_source


def get_noise_params(
    file: MEFSlice,
    catalog: QTable,
    background: BaseBackground | Callable,
    psf_params: dict[str, float],
    instrument: Instrument,
    bias_corrector: BiasCorrector | None,
    dark_corrector: DarkNoiseCorrector | None,
    flat_corrector: FlatFieldCorrector | None,
    ) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64], float, float, float, float, float, NDArray[np.float64]]:
    """
    Get the noise values of a science image.
    
    Parameters
    ----------
    file : MEFSlice
        The science image file.
    catalog : QTable
        The source catalog corresponding to the science image.
    background : BaseBackground | Callable
        The background estimator.
    psf_params : dict[str, float]
        The PSF parameters.
    instrument : Instrument
        The instrument.
    bias_corrector : BiasCorrector | None
        The bias corrector.
    dark_corrector : DarkNoiseCorrector | None
        The dark noise corrector.
    flat_corrector : FlatFieldCorrector | None
        The flat-field corrector.
    
    Returns
    -------
    tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64], float, float, float, float, float, NDArray[np.float64]]
        The source IDs, fluxes, flux errors, number of aperture pixels, backgorund counts/pixel, bias variance, dark
        variance, flat-field variance, and scintillation noise.
    """
    
    coords = np.asarray([catalog['xcentroid'], catalog['ycentroid']]).T
    
    img, bias_var, dark_var, flat_var, rel_scint_noise = get_data(
        file=file,
        instrument=instrument,
        bias_corrector=bias_corrector,
        dark_corrector=dark_corrector,
        flat_corrector=flat_corrector,
        rebin_factor=1,
        remove_cosmic_rays=False,
        )
    
    # global background
    bkg = background(img)
    n_sky = float(bkg.background_rms_median**2)  # background variance
    
    # subtract background from image
    img_clean = img - bkg.background
    
    # perform photometry
    phot = AperturePhotometer()
    phot_results = phot.compute(
        image=img_clean,
        bias_var=bias_var,
        dark_var=dark_var,
        flat_var=flat_var,
        background_rms=np.sqrt(n_sky),
        cat_coords=coords,
        image_coords=coords,
        psf_params=psf_params,
        read_noise=instrument.get_read_noise(file=file),
        rel_scint_noise=rel_scint_noise,
        )
    
    # get the number of pixels in the aperture
    N_pix = phot.get_aperture_area(psf_params=psf_params)
    
    fluxes = np.array(phot_results['flux'])
    flux_errs = np.array(phot_results['flux_err'])
    source_ids = np.arange(len(catalog)) + 1
    scint_noise = rel_scint_noise * fluxes
    
    # mask unphysical flux values
    mask = fluxes > 1.
    
    return source_ids[mask], fluxes[mask], flux_errs[mask], N_pix, n_sky, float(np.median(bias_var)), float(np.median(dark_var)), float(np.median(flat_var)), scint_noise


def get_snrs(
    file: MEFSlice,
    background: BaseBackground | Callable,
    catalog: QTable,
    psf_params: dict[str, float],
    instrument: Instrument,
    bias_corrector: BiasCorrector | None,
    dark_corrector: DarkNoiseCorrector | None,
    flat_corrector: FlatFieldCorrector | None,
    ) -> tuple[NDArray, NDArray]:
    """
    Get the S/N ratios for the cataloged sources in a science image.
    
    Parameters
    ----------
    file : MEFSlice
        The science image file.
    background : BaseBackground | Callable
        The background estimator.
    catalog : QTable
        The source catalog corresponding to the science image.
    psf_params : dict[str, float]
        The PSF parameters.
    instrument : Instrument
        The instrument.
    bias_corrector : BiasCorrector | None
        The bias corrector.
    dark_corrector : DarkNoiseCorrector | None
        The dark noise corrector.
    flat_corrector : FlatFieldCorrector | None
        The flat-field corrector.
    
    Returns
    -------
    tuple[NDArray, NDArray]
        The source IDs and S/N for each source.
    """
    
    source_ids, fluxes, flux_errs, N_pix, n_sky, bias_var, dark_var, flat_var, scint_noise = get_noise_params(
        file=file,
        catalog=catalog,
        background=background,
        psf_params=psf_params,
        instrument=instrument,
        bias_corrector=bias_corrector,
        dark_corrector=dark_corrector,
        flat_corrector=flat_corrector,
    )
    
    return source_ids, np.asarray(
        snr(
            N_source=fluxes,
            N_pix=N_pix,
            n_sky=n_sky,
            bias_var=bias_var,
            dark_var=dark_var,
            flat_var=flat_var,
            read_noise=instrument.get_read_noise(file=file),
            scint_noise=scint_noise,
            )
        )


def characterise_noise(
    file: MEFSlice,
    background: BaseBackground | Callable,
    catalog: QTable,
    psf_params: dict[str, float],
    instrument: Instrument,
    bias_corrector: BiasCorrector | None,
    dark_corrector: DarkNoiseCorrector,
    flat_corrector: FlatFieldCorrector | None,
    ) -> dict[str, NDArray]:
    """
    Characterise the expected noise from an image and compare it to the measured noise for a number of cataloged 
    sources.
    
    Parameters
    ----------
    file : MEFSlice
        The science image file.
    background : BaseBackground | Callable
        The background estimator.
    catalog : QTable
        The source catalog corresponding to the science image.
    psf_params : dict[str, float]
        The PSF parameters.
    instrument : Instrument
        The instrument.
    bias_corrector : BiasCorrector
        The bias corrector.
    dark_corrector : DarkNoiseCorrector
        The dark noise corrector.
    flat_corrector : FlatFieldCorrector
        The flat-field corrector.
    
    Returns
    -------
    dict[str, NDArray]
        The noies properties.
    """
    
    header = file.get_header()
    
    read_noise = instrument.get_read_noise(header=header)
    rel_scint_noise = instrument.get_relative_scintillation_noise(header=header)
    
    source_ids, fluxes, flux_errs, N_pix, n_sky, bias_var, dark_var, flat_var, scint_noise = get_noise_params(
        file=file,
        catalog=catalog,
        background=background,
        psf_params=psf_params,
        instrument=instrument,
        bias_corrector=bias_corrector,
        dark_corrector=dark_corrector,
        flat_corrector=flat_corrector,
    )
    
    N_source = np.logspace(
        np.log10(np.min(fluxes) / 1.5),
        np.log10(np.max(fluxes) * 1.5),
        100,
        )
    
    results = {}
    
    results['model_mags'] = -2.5 * np.log10(N_source)
    results['effective_noise'] = snr_stderr(
        N_source=N_source,
        N_pix=N_pix,
        n_sky=n_sky,
        bias_var=bias_var,
        dark_var=dark_var,
        flat_var=flat_var,
        read_noise=read_noise,
        rel_scint_noise=rel_scint_noise,
        )
    results['sky_noise'] = get_sky_stderr(
        N_source=N_source,
        N_pix=N_pix,
        n_sky=n_sky,
        )
    results['shot_noise'] = get_shot_stderr(N_source=N_source)
    results['bias'] = get_bias_stderr(
        N_source=N_source,
        N_pix=N_pix, bias_var=bias_var)
    results['dark_noise'] = get_dark_stderr(N_source, N_pix, dark_var)
    results['flat'] = get_flat_stderr(N_source, N_pix, flat_var)
    results['read_noise'] = get_read_stderr(N_source, N_pix, read_noise=read_noise)
    results['scint_noise'] = get_scint_stderr(N_source=N_source, rel_scint_noise=rel_scint_noise)
    
    results['measured_mags'] = -2.5 * np.log10(fluxes)
    results['measured_noise'] = counts_to_mag_factor * flux_errs / fluxes
    results['expected_measured_noise'] = snr_stderr(
        N_source=fluxes,
        N_pix=N_pix,
        n_sky=n_sky,
        bias_var=bias_var,
        dark_var=dark_var,
        flat_var=flat_var,
        read_noise=read_noise,
        rel_scint_noise=rel_scint_noise,
        )
    
    return results



















