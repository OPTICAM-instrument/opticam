from typing import Dict


import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit, minimize  # TODO: replace with astropy modelling?


from opticam.fitting.models import power_law, straight_line
from opticam.fitting.psf_models import gaussian




def fit_rms_vs_flux(
    data: Dict,
    ) -> Dict[str, Dict[str, NDArray]]:
    """
    Iteratively fit a straight line (in log space) to the RMS vs flux plots for each catalog. This can be used to
    identify variable sources and good comparison sources.
    
    Parameters
    ----------
    data : Dict
        The RMS vs flux data.
    
    Returns
    -------
    Dict[str, Dict[str, NDArray]]
        The power law fits for each filter `{filter: {'flux': NDArray, 'rms': NDArray}}`.
    """
    
    pl_fits = {}
    
    for fltr in data.keys():
        flux = []
        rms = []
        ids = []
        for source_id, values in data[fltr].items():
            flux.append(values['flux'])
            rms.append(values['rms'])
            ids.append(int(source_id))
        
        flux = np.array(flux)
        rms = np.array(rms)
        ids = np.array(ids)
        
        # sort data by flux
        order = np.argsort(flux)
        flux = flux[order]
        rms = rms[order]
        ids = ids[order]
        
        popt, pcov = curve_fit(
                straight_line,
                np.log10(flux),
                np.log10(rms),
                )
        
        rms_model = power_law(
            flux,
            10**popt[1],
            popt[0],
            )
        
        pl_fits[fltr] = {
            'ids': ids,
            'flux': flux,
            'rms': rms_model,
            'err': .05 * rms_model,  # 5% error
        }
    
    return pl_fits


def fit_psf(
    image: NDArray,
    x_init: float | int,
    y_init: float | int,
    semimajor_sigma: float,
    semiminor_sigma: float,
    ) -> tuple[float, float, float]:
    """
    Find the location of a source by fitting a Gaussian PSF to an image.
    
    Parameters
    ----------
    image : NDArray
        The image. Should be a small region of a larger image to ensure the correct source is found.
    x_init : float | int
        The initial guess for the x location of the PSF.
    y_init : float | int
        The initial guess for the y location of the PSF.
    semimajor_sigma : float
        The semi-major standard deviation of the PSF.
    semiminor_sigma : float
        The semi-minor standard deviation of the PSF.
    
    Returns
    -------
    tuple[float, float, float]
        The best-fitting x position, y position, and orientation of the PSF.
    """
    
    def residuals(
        params: list[float],
        image: NDArray,
        amplitude: float,
        sigma_major: float,
        sigma_minor: float,
        ) -> float:
        
        x0, y0, theta = params
        model = gaussian(
            shape=image.shape,
            x0=x0,
            y0=y0,
            theta=theta,
            amplitude=amplitude,
            sigma_major=sigma_major,
            sigma_minor=sigma_minor,
            )
        
        return np.sum((image - model)**2)
    
    # normalise image for easier fitting
    normed_image = image / np.max(image)
    
    result = minimize(
        residuals,
        x0=[x_init, y_init, 0.],
        args=(
            normed_image,
            1.,                 # amplitude: normalised to 1
            semimajor_sigma,
            semiminor_sigma,
            ),
        method='L-BFGS-B',
        bounds=[
            (0, image.shape[1]),    # x
            (0, image.shape[0]),    # y
            (0, 2 * np.pi),         # theta
            ]
        )
    
    x_region, y_region, theta = result.x
    
    return x_region, y_region, theta