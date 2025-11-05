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
                    # do not use perr since it may be very large
                    converged = np.allclose(popt, prev, rtol=0.2, atol=0)
                
                # remove largest outliers
                model = straight_line(log_x, *popt)
                r = log_y - model
                i = np.argmax(r)
                
                rms.pop(i)
                flux.pop(i)
                
                prev = popt
                prev_err = perr
        except:
            popt, pcov = curve_fit(
                    straight_line,
                    np.log10(x),
                    np.log10(y),
                    )
        
        # get model prediction band using Monte Carlo method
        N = 1000
        y_models = np.zeros((N, x.size))
        for i in range(N):
            rng = np.random.default_rng(i)
            a, b = rng.multivariate_normal(popt, pcov)
            y_models[i] += power_law(x, 10**a, b)
        y_model_err = 5 * np.std(y_models, axis=0)  # 5 sigma error
        
        y_model = power_law(
            x,
            10**popt[1],
            popt[0],
            )
        
        pl_fits[fltr] = {
            'flux': x,
            'rms': y_model,
            'err': y_model_err,
        }
    
    return pl_fits
