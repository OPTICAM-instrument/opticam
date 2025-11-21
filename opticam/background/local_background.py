from photutils.aperture import Aperture, ApertureStats, EllipticalAnnulus
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

    @abstractmethod
    def get_annulus(
        self,
        position: NDArray,
        semimajor_axis: float,
        semiminor_axis: float,
        theta: float,
        ) -> Aperture:
        """
        Define an annulus at the given position.
        
        Parameters
        ----------
        position : NDArray
            The centre of the annulus.
        semimajor_axis : float
            The semimajor standard deviation of the PSF.
        semiminor_axis : float
            The semiminor standard deviation of the PSF.
        theta : float
            The orientation of the source **in radians**.
        
        Returns
        -------
        Aperture
            The annulus.
        """
        
        pass

    def get_stats(
        self,
        data: NDArray,
        position: NDArray,
        semimajor_axis: float,
        semiminor_axis: float,
        theta: float,
        ) -> ApertureStats:
        """
        Get the stats of the annulus.
        
        Parameters
        ----------
        data : NDArray
            The image data.
        position : NDArray
            The centre of the annulus.
        semimajor_axis : float
            The semimajor standard deviation of the PSF.
        semiminor_axis : float
            The semiminor standard deviation of the PSF.
        theta : float
            The orientation of the source **in radians**.
        
        Returns
        -------
        ApertureStats
            The stats of the annulus.
        """
        
        annulus = self.get_annulus(
            position,
            semimajor_axis,
            semiminor_axis,
            theta,
            )
        
        return ApertureStats(
            data,
            annulus,
            sigma_clip=self.sigma_clip,
            )


class DefaultLocalBackground(BaseLocalBackground):
    """
    Default local background estimator using an elliptical annulus.
    """

    def get_annulus(
        self,
        position: NDArray,
        semimajor_axis: float,
        semiminor_axis: float,
        theta: float,
        ) -> Aperture:
        """
        Define an annulus at the given position.
        
        Parameters
        ----------
        position : NDArray
            The centre of the annulus.
        semimajor_axis : float
            The semimajor standard deviation of the PSF.
        semiminor_axis : float
            The semiminor standard deviation of the PSF.
        theta : float
            The orientation of the source **in radians**.
        
        Returns
        -------
        Aperture
            The annulus.
        """
        
        return EllipticalAnnulus(
            position,
            self.r_in_scale * semimajor_axis,
            self.r_out_scale * semimajor_axis,
            self.r_out_scale * semiminor_axis,
            self.r_in_scale * semiminor_axis,
            theta,
            )

    def __call__(
        self,
        data: NDArray,
        position: NDArray,
        semimajor_axis: float,
        semiminor_axis: float | None = None,
        theta: float = 0.,
        ) -> Tuple[float, float]:
        """
        Compute the sigma-clipped local background (mean) and its error (standard deviation) at a given position.
        
        Parameters
        ----------
        data : NDArray
            The image data.
        error : NDArray
            The error in the image data.
        position : NDArray
            The x, y position at which to compute the local background.
        semimajor_axis : float
            The (unscaled) semimajor axis of the aperture.
        semiminor_axis : float | None, optional
            The (unscaled) semiminor axis of the aperture, by default `None`. If `None`, it is assumed to be equal to 
            the semimajor axis (i.e., the annulus is circular).
        theta : float, optional
            The rotation angle of the PSF, by default 0 (i.e., no rotation).
        
        Returns
        -------
        Tuple[float, float]
            The local background (mean) and its error (standard deviation).
        """
        
        if semiminor_axis is None:
            semiminor_axis = semimajor_axis
        
        stats = self.get_stats(
            data=data,
            position=position,
            semimajor_axis=semimajor_axis,
            semiminor_axis=semiminor_axis,
            theta=theta,
            )
        
        return float(stats.mean), float(stats.std)