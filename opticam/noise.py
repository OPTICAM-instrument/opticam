import numpy as np
from numpy.typing import NDArray


from opticam.utils.constants import counts_to_mag_factor




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