from typing import Dict

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit

from opticam.fitting.models import power_law, straight_line

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
        rms, flux = [], []
        for values in data[fltr].values():
            rms.append(values['rms'])
            flux.append(values['flux'])
        
        order = np.argsort(flux)
        x = np.array(flux)[order]
        y = np.array(rms)[order]
        
        popt, pcov = curve_fit(
                straight_line,
                np.log10(x),
                np.log10(y),
                )
        
        y_model = power_law(
            x,
            10**popt[1],
            popt[0],
            )
        
        pl_fits[fltr] = {
            'flux': x,
            'rms': y_model,
            'err': .05 * y_model,  # 5% error
        }
    
    return pl_fits
