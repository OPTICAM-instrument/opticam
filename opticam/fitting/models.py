import numpy as np
from numpy.typing import NDArray


def gaussian(
    x: NDArray,
    amp: float,
    mu: float,
    std: float,
    ) -> NDArray:
    """
    Gaussian function.
    
    Parameters
    ----------
    x : NDArray
        The function inputs.
    amplitude : float
        The height of the function peak.
    mu : float
        The mean of the Gaussian.
    std : float
        The standard deviation of the Gaussian.
    
    Returns
    -------
    NDArray
        The Gaussian evaluated at the inputs.
    """
    
    return amp * np.exp(-.5 * ((x - mu) / std)**2)


def power_law(
    x: NDArray,
    norm: float,
    exp: float,
    ) -> NDArray:
    """
    Power law function.
    
    Parameters
    ----------
    x : NDArray
        The function inputs.
    norm : float
        The normalisation.
    exp : float
        The exponent.
    
    Returns
    -------
    NDArray
        The power law evaluated at the inputs.
    """
    
    return norm*x**exp


def straight_line(
    x: NDArray,
    m: float,
    c: float,
    ) -> NDArray:
    """
    Straight line function.
    
    Parameters
    ----------
    x : NDArray
        The function inputs.
    m : float
        The gradient.
    c : float
        The intercept.
    
    Returns
    -------
    NDArray
        The straight line evaluated at the inputs.
    """
    
    return m*x + c