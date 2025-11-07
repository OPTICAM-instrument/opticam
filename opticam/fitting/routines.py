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
        
        try:
            converged = False
            prev, prev_err = None, None
            while not converged:
                log_x = np.log10(flux)
                log_y = np.log10(rms)
                
                popt, pcov = curve_fit(
                    straight_line,
                    log_x,
                    log_y,
                    )
                perr = np.sqrt(np.diag(pcov))
                
                if prev is not None and prev_err is not None:
                    # assume fit has converged is params are within 20%
                    # do not use perr since it may be very large if only a few sources in the field
                    # or very small if there are many sources in the field
                    converged = np.allclose(popt, prev, rtol=0.2, atol=0.)
                
                # remove largest outliers
                model = straight_line(log_x, *popt)
                r = log_y - model
                i = np.argmax(r)
                
                rms.pop(i)
                flux.pop(i)
                
                prev = popt
                prev_err = perr
        except:
            # if an interative solution cannot be reached
            # fit all the points and move on
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
        
        if len(rms) > 1:
            err = np.std(np.array(rms) / power_law(np.array(flux), 10**popt[1], popt[0]))
        else:
            err = np.std(y / y_model)
        
        pl_fits[fltr] = {
            'flux': x,
            'rms': y_model,
            'err': np.zeros_like(y_model) + 3 * err,  # 3 sigma error
        }
    
    return pl_fits
