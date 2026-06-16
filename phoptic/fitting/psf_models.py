import numpy as np
from numpy.typing import NDArray


def gaussian(
    shape: tuple[int, int],
    x0: float,
    y0: float,
    theta: float,
    amplitude: float,
    sigma_major: float,
    sigma_minor: float,
    ) -> NDArray[np.float64]:
    """
    Gaussian PSF.
    
    Parameters
    ----------
    shape : tuple[int, int]
        The shape of the array.
    x0 : float
        The x location of the PSF maximum.
    y0 : float
        The y location of the PSF maximum.
    theta : float
        The orientation of the PSF in radians.
    amplitude : float
        The amplitude of the peak of the PSF.
    sigma_major : float
        The semi-major standard deviation.
    sigma_minor : float
        The semi-minor standard deviation.
    
    Returns
    -------
    NDArray[np.float64]
        The PSF model.
    """
    
    ny, nx = shape
    x = np.arange(nx)
    y = np.arange(ny)
    X, Y = np.meshgrid(x, y)
    
    dX = X - x0
    dY = Y - y0
    
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    X_rot =  cos_t * dX + sin_t * dY
    Y_rot = -sin_t * dX + cos_t * dY
    
    return amplitude * np.exp(
        -((X_rot**2) / (2 * sigma_major**2) +
          (Y_rot**2) / (2 * sigma_minor**2))
    )