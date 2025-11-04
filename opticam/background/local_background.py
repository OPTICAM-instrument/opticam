from photutils.aperture import ApertureStats, EllipticalAnnulus
from abc import ABC, abstractmethod
from numpy.typing import NDArray
from typing import Tuple
from astropy.stats import SigmaClip


class BaseLocalBackground(ABC):
    """
    Base class for local background estimators.
    """
    
    def __init__(
        self,
        r_in_scale: float = 5,
        r_out_scale: float = 7.5,
        sigma_clip: None | SigmaClip = SigmaClip(sigma=3, maxiters=10),
        ) -> None:
        """
        Base class for local background estimators.
        
        Parameters
        ----------
        r_in_scale : float, optional
            The inner scale of the annulus in units of aperture semimajor/semiminor axes or radius, by default 5
            (assuming the semimajor axis is in standard deviations for a 2D Gaussian PSF).
        r_out_scale : float, optional
            The outer scale of the annulus in units of aperture semimajor/semiminor axes or radius, by default 7.5
            (assuming the semimajor axis is in standard deviations for a 2D Gaussian PSF).
        sigma_clip : SigmaClip, optional
            The sigma clipper for removing outlier pixels in the annulus, by default `SigmaClip(sigma=3, maxiters=10)`.
        """
        
        self.r_in_scale = r_in_scale
        self.r_out_scale = r_out_scale
        self.sigma_clip = sigma_clip
    
    @abstractmethod
    def __call__(
        self,
        data: NDArray,
        position: NDArray,
        semimajor_axis: float,
        semiminor_axis: float,
        theta: float) -> Tuple[float, float]:
        """
        Compute the local background and its error at a given position (**per pixel**).
        
        Parameters
        ----------
        data : NDArray
            The image data.
        semimajor_axis : float
            The (unscaled) semi-major axis of the aperture.
        semiminor_axis : float
            The (unscaled) semi-minor axis of the aperture.
        theta : float
            The rotation angle of the PSF.
        position : Tuple[float, float]
            The x, y position at which to compute the local background.
        
        Returns
        -------
        Tuple[float, float]
            The local background and its error per pixel.
        """
        
        pass


class DefaultLocalBackground(BaseLocalBackground):
    """
    Default local background estimator using an elliptical annulus.
    """
    
    def __call__(
        self,
        data: NDArray,
        position: NDArray,
        semimajor_axis: float,
        semiminor_axis: float | None = None,
        theta: float = 0.,
        ) -> Tuple[float, float]:
        """
        Compute the local background and its error at a given position (**per pixel**).
        
        Parameters
        ----------
        data : NDArray
            The image data.
        error : NDArray
            The error in the image data.
        position : ArrayLike[float, float]
            The x, y position at which to compute the local background.
        semimajor_axis : float
            The (unscaled) semimajor axis of the aperture.
        semiminor_axis : float | None, optional
            The (unscaled) semiminor axis of the aperture, by default None. If None, it is assumed to be equal to the
            semimajor axis (i.e., the aperture is circular).
        theta : float, optional
            The rotation angle of the PSF, by default 0.0 (i.e., no rotation).
        
        Returns
        -------
        Tuple[float, float]
            The local background and its error **per pixel**.
        """
        
        if semiminor_axis is None:
            semiminor_axis = semimajor_axis
        
        annulus = EllipticalAnnulus(
            position,
            self.r_in_scale * semimajor_axis,
            self.r_out_scale * semimajor_axis,
            self.r_out_scale * semiminor_axis,
            self.r_in_scale * semiminor_axis,
            theta,
            )
        
        stats = ApertureStats(
            data,
            annulus,
            sigma_clip=self.sigma_clip,
            )
        
        return float(stats.mean), float(stats.std)