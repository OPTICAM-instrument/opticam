from abc import ABC, abstractmethod
from logging import Logger
from typing import Callable, Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray
from photutils.aperture import aperture_photometry, EllipticalAperture
from photutils.segmentation import detect_threshold

from opticam.background.global_background import BaseBackground
from opticam.background.local_background import BaseLocalBackground
from opticam.correctors.flat_field_corrector import FlatFieldCorrector
from opticam.finders import DefaultFinder
from opticam.utils.constants import fwhm_scale, n_read
from opticam.utils.fits_handlers import get_data
from opticam.utils.helpers import camel_to_snake, propagate_errors


class BasePhotometer(ABC):
    """
    Base class for performing photometry on OPTICAM catalogues.
    """

    def __init__(
        self,
        forced: bool = False,
        source_matching_tolerance: float = 5.,
        local_background_estimator: BaseLocalBackground | Callable | None = None,
        ):
        """
        Initialise a photometer.
        
        Parameters
        ----------
        forced : bool, optional
            Whether to performed "forced" photometry, by default `False`. If `True`, the catalog-aligned coordinates
            are used to perform photometry, even in images where the source is not detected, and the resulting light
            curves will be saved with a 'forced' prefix.
        source_matching_tolerance : float, optional
            The tolerance for source position matching in standard deviations (assuming a Gaussian PSF), by default 5.
            This parameter defines how far from the transformed catalogue position a source can be while still being
            considered the same source.
        local_background_estimator : BaseLocalBackground | Callable | None, optional
            The local background estimator to use, by default `None`. If `None`, the catalogue's 2D background estimator
            is used. If not `None`, this will be used instead of the catalogue's 2D background estimator.
        """
        
        self.forced = forced
        self.source_matching_tolerance = source_matching_tolerance
        
        if local_background_estimator is not None:
            assert callable(local_background_estimator), "[OPTICAM] local_background_estimator must be either None or a callable object."
        
        self.local_background_estimator = local_background_estimator

    @abstractmethod
    def compute(
        self,
        image: NDArray,
        dark_flux: float,
        background_rms: NDArray | None,
        source_coords: NDArray,
        image_coords: NDArray | None,
        psf_params: Dict[str, float],
        ) -> Dict[str, List]:
        """
        Compute the fluxes of the catalogued sources from the given image.
        
        Parameters
        ----------
        image : NDArray
            The image. If `self.local_background_estimator` is undefined, this image will be background subtracted.
        dark_flux : float
            The dark current's "flux" contribution.
        background_rms : NDArray | None
            The background RMS. May be `None` if `self.local_background_estimator` is defined.
        source_coords : NDArray
            The source coordinates in the catalogue.
        image_coords : NDArray | None
            The source coordinates in the image. If `match_sources` is True, this will be used to match sources in the
            image to sources in the catalogue.
        psf_params : Dict[str, float]
            The PSF parameters for the camera used to take the image. This parameter is defined in the catalogue and
            has the following keys: 'semimajor_sigma' (in pixels), 'semiminor_sigma' (in pixels), and 'orientation' (in
            *degrees*).
        
        Returns
        -------
        Dict[str, List]
            The photometry results.
        """
        
        pass

    def get_position(
        self,
        source_coords: NDArray,
        image_coords: NDArray | None,
        source_index: int,
        psf_params: Dict[str, float],
        ) -> NDArray | None:
        """
        Get the position of a source in an image.
        
        Parameters
        ----------
        source_coords : NDArray
            The source coordinates in the catalogue.
        image_coords : NDArray | None
            The source coordinates in the image.
        source_index : int
            The source index.
        psf_params : Dict[str, float]
            The PSF parameters for the camera used to take the image. This parameter is defined in the catalogue and
            has the following keys: 'semimajor_sigma' (in pixels), 'semiminor_sigma' (in pixels), and 'orientation' (in
            *degrees*).
        
        Returns
        -------
        NDArray
            The source coordinates.
        """
        
        if not self.forced:
            return self.get_closest_source(
                source_coords,
                image_coords,
                source_index,
                psf_params,
                )
        else:
            return source_coords[source_index]

    def get_closest_source(
        self,
        source_coords: NDArray,
        image_coords: NDArray | None,
        source_index: int,
        psf_params: Dict[str, float],
        ) -> NDArray | None:
        """
        Given a source, find the closest source in the catalogue.
        
        Parameters
        ----------
        source_coords : NDArray
            The source coordinates in the catalogue.
        image_coords : NDArray | None
            The source coordinates in the image.
        source_index : int
            The source index.
        psf_params : Dict[str, float]
            The PSF parameters for the camera used to take the image. This parameter is defined in the catalogue and
            has the following keys: 'semimajor_sigma' (in pixels), 'semiminor_sigma' (in pixels), and 'orientation' (in
            *degrees*).
        
        Returns
        -------
        NDArray | None
            The coordinates of the closest source.
        """
        
        if image_coords is None:
            return None
        
        # get distances between sources and initial position
        distances = np.sqrt((image_coords[:, 0] - source_coords[source_index][0])**2 + (image_coords[:, 1] - source_coords[source_index][1])**2)
        
        # if the closest source is further than the specified tolerance
        if np.min(distances) > self.source_matching_tolerance * np.sqrt(psf_params['semimajor_sigma']**2 + psf_params['semiminor_sigma']**2):
            return None
        else:
            # get the position of the closest source (assumed to be the source of interest)
            return image_coords[np.argmin(distances)]

    def define_results_dict(
        self,
        ) -> Dict[str, List]:
        """
        Define a results dictionary for the photometer depending on whether `local_background_estimator` is defined.
        
        Returns
        -------
        Dict[str, List]
            The results dictionary with keys 'flux', 'flux_error'. If `local_background_estimator` is defined, the
            dictionary will also contain 'background' and 'background_error'.
        """
        
        results = {
            'flux': [],
            'flux_err': [],
        }
        
        if isinstance(self, AperturePhotometer):
            results['snr'] = []
        
        if self.local_background_estimator is not None:
            results['background'] = []
            results['background_err'] = []
        
        return results

    def pad_results_dict(
        self,
        results: Dict[str, List],
        ) -> Dict[str, List]:
        """
        Pad the results dictionary with None values for flux and flux error, and background and background error if
        `local_background_estimator' is defined. This is used when a source cannot be matched or its position is
        invalid.
        
        Parameters
        ----------
        results : Dict[str, List]
            The results dictionary to pad.
        
        Returns
        -------
        Dict[str, List]
            The padded results dictionary.
        """
        
        results['flux'].append(None)
        results['flux_err'].append(None)
        
        if isinstance(self, AperturePhotometer):
            results['snr'].append(None)
        
        if self.local_background_estimator is not None:
            results['background'].append(None)
            results['background_err'].append(None)
        
        return results

    def populate_results_dict(
        self,
        results: Dict[str, List],
        phot_function: Callable,
        image: NDArray,
        dark_flux: float,
        background_rms: NDArray | None,
        position: NDArray,
        psf_params: Dict[str, float],
        ) -> Dict[str, List]:
        """
        Populate the results dictionary with the computed flux, flux error, and background (if applicable) using the
        provided photometry function.
        
        Parameters
        ----------
        results : Dict[str, List]
            The results dictionary to populate.
        phot_function : Callable
            The photometry function to use for computing the flux and flux error. This function should take the image,
            image error, position, and PSF parameters as arguments and return the flux and flux error, and optionally
            the background and background error if `local_background_estimator` is defined.
        image : NDArray
            The image.
        dark_flux : float
            The dark current's "flux" contribution.
        background_rms : NDArray | None
            The background RMS. May be `None` if `self.local_background_estimator` is defined.
        position : NDArray
            The position of the source in the image.
        psf_params : Dict[str, float]
            The PSF parameters for the camera used to take the image. This parameter is defined in the catalogue and
            has the following keys: 'semimajor_sigma' (in pixels), 'semiminor_sigma' (in pixels), and 'orientation' (in
            *degrees*).
        
        Returns
        -------
        Dict[str, List]
            The updated results dictionary with the computed flux, flux error, and background (if applicable).
        """
        
        if self.local_background_estimator is None:
            if isinstance(self, AperturePhotometer):
                flux, flux_err, snr = phot_function(
                    image=image,
                    dark_flux=dark_flux,
                    background_rms=background_rms,
                    position=position,
                    psf_params=psf_params,
                )
                
                results['flux'].append(flux)
                results['flux_err'].append(flux_err)
                results['snr'].append(snr)
            else:
                flux, flux_err = phot_function(
                    image=image,
                    dark_flux=dark_flux,
                    background_rms=background_rms,
                    position=position,
                    psf_params=psf_params,
                )
                
                results['flux'].append(flux)
                results['flux_err'].append(flux_err)
        else:
            if isinstance(self, AperturePhotometer):
                flux, flux_err, snr, background, background_err = phot_function(
                    image=image,
                    dark_flux=dark_flux,
                    background_rms=background_rms,
                    position=position,
                    psf_params=psf_params,
                )
                
                results['flux'].append(flux)
                results['flux_err'].append(flux_err)
                results['snr'].append(snr)
                results['background'].append(background)
                results['background_err'].append(background_err)
            else:
                flux, flux_err, background, background_err = phot_function(
                    image=image,
                    dark_flux=dark_flux,
                    background_rms=background_rms,
                    position=position,
                    psf_params=psf_params,
                )
                
                results['flux'].append(flux)
                results['flux_err'].append(flux_err)
                results['background'].append(background)
                results['background_err'].append(background_err)
        
        return results

    def get_label(
            self,
            ) -> str:
            """
            Get the label of the photometer for labelling output.
            
            Returns
            -------
            str
                The label.
            """
            
            save_name = camel_to_snake(self.__class__.__name__).replace('_photometer', '')
            
            # change save directory based on photometer settings
            if self.local_background_estimator is not None:
                save_name += '_annulus'
            if self.forced:
                save_name = 'forced_' + save_name
            
            return save_name

class AperturePhotometer(BasePhotometer):
    """
    A photometer for performing aperture photometry.
    """

    def __init__(
        self,
        semimajor_axis: int | None = None,
        semiminor_axis: int | None = None,
        orientation: float | None = None,
        forced: bool = False,
        source_matching_tolerance: float = 5.,
        local_background_estimator: None | BaseLocalBackground = None,
        ):
        """
        Initialise a photometer.
        
        Parameters
        ----------
        semimajor_axis : int | None, optional
            The semi-major axis of the aperture, by default None (set to the FWHM of the PSF).
        semiminor_axis : int | None, optional
            The semi-minor axis of the aperture, by default None (set to the FWHM of the PSF).
        orientation : float, optional
            The orientation of the ellipse, by default None (set based on the averaged PSF orientation).
        forced : bool, optional
            Whether to performed "forced" photometry, by default `False`. If `True`, the catalog-aligned coordinates
            are used to perform photometry, even in images where the source is not detected, and the resulting light
            curves will be saved with a 'forced' prefix.
        source_matching_tolerance : float, optional
            The tolerance for source position matching in standard deviations (assuming a Gaussian PSF), by default 5.
            This parameter defines how far from the transformed catalogue position a source can be while still being
            considered the same source.
        local_background_estimator : None | BaseLocalBackground, optional
            The local background estimator to use, by default `None`. If `None`, the catalogue's 2D background estimator
            is used. If not `None`, this will be used instead of the catalogue's 2D background estimator.
        """
        
        self.semimajor_axis = semimajor_axis
        self.semiminor_axis = semiminor_axis
        self.orientation = orientation
        
        super().__init__(
            forced=forced,
            source_matching_tolerance=source_matching_tolerance,
            local_background_estimator=local_background_estimator,
            )

    def compute(
        self,
        image: NDArray,
        dark_flux: float,
        background_rms: NDArray | None,
        source_coords: NDArray,
        image_coords: NDArray | None,
        psf_params: Dict[str, float],
        ) -> Dict[str, List]:
        """
        Compute the fluxes of the catalogued sources from the given image.
        
        Parameters
        ----------
        image : NDArray
            The image. If `self.local_background_estimator` is undefined, this image will be background subtracted.
        dark_flux : float
            The dark current's "flux" contribution.
        background_rms : NDArray | None
            The background RMS. May be `None` if `self.local_background_estimator` is defined.
        source_coords : NDArray
            The source coordinates in the catalogue.
        image_coords : NDArray | None
            The source coordinates in the image. If `match_sources` is True, this will be used to match sources in the
            image to sources in the catalogue.
        psf_params : Dict[str, float]
            The PSF parameters for the camera used to take the image. This parameter is defined in the catalogue and
            has the following keys: 'semimajor_sigma' (in pixels), 'semiminor_sigma' (in pixels), and 'orientation' (in
            *degrees*).
        
        Returns
        -------
        Dict[str, List]
            The photometry results.
        """
        
        results = self.define_results_dict()
        
        for i in range(len(source_coords)):
            
            # get position of source depending on whether source matching is enabled or not
            position = self.get_position(
                source_coords,
                image_coords,
                i,
                psf_params,
                )
            
            # if position is None, pad the results dictionary and continue to the next source
            if position is None:
                results = self.pad_results_dict(results)
                continue
            
            # populate the results dictionary with the computed flux, flux error, and background (if applicable)
            results = self.populate_results_dict(
                results=results,
                phot_function=self.compute_aperture_flux,
                image=image,
                dark_flux=dark_flux,
                background_rms=background_rms,
                position=position,
                psf_params=psf_params,
                )
        
        return results

    def compute_aperture_flux(
        self,
        image: NDArray,
        dark_flux: float,
        background_rms: NDArray | None,
        position: NDArray,
        psf_params: Dict[str, float],
        ) -> Tuple[float, float, float] | Tuple[float, float, float, float, float]:
        """
        Compute the aperture flux of a source in the image.
        
        Parameters
        ----------
        image : NDArray
            The image.
        dark_flux : float
            The dark current's "flux" contribution.
        background_rms : NDArray | None
            The background RMS. May be `None` if `self.local_background_estimator` is defined.
        position : NDArray
            The position of the source.
        psf_params : Dict[str, float]
            The PSF parameters for the camera used to take the image. This parameter is defined in the catalogue and
            has the following keys: 'semimajor_sigma' (in pixels), 'semiminor_sigma' (in pixels), and 'orientation' (in
            *degrees*).
        
        Returns
        -------
        Tuple[float, float, float] | Tuple[float, float, float, float, float]
            The flux, flux error, and signal-to-noise ratio. If `local_background_estimator` is defined, the background
            and its error are also returned.
        """
        
        aperture = self.get_aperture(
            position=position,
            psf_params=psf_params,
            )
        
        if self.local_background_estimator is None:
            error = propagate_errors(
                data=image,
                dark_flux=dark_flux,
                background_rms=np.asarray(background_rms),
                )
            phot_table = aperture_photometry(image, aperture, error=error)
            
            flux = float(phot_table["aperture_sum"].value[0])
            flux_err = float(phot_table["aperture_sum_err"].value[0])
            snr = float(
                compute_snr(
                    N_source=flux,
                    N_pix=aperture.area,
                    n_sky=float(np.median(np.asarray(background_rms)))**2,
                    dark_flux=dark_flux,
                    gain=1.,
                    )
            )
            
            return flux, flux_err, snr
        else:
            
            local_background, local_background_rms = self.local_background_estimator(
                image,
                position,
                psf_params['semimajor_sigma'],
                psf_params['semiminor_sigma'],
                psf_params['orientation'],
                )
            
            data_clean = image - local_background
            error = propagate_errors(
                data=data_clean,
                dark_flux=dark_flux,
                background_rms=local_background_rms,
            )
            
            phot_table = aperture_photometry(data_clean, aperture, error=error)
            
            flux = float(phot_table["aperture_sum"].value[0])
            flux_err = float(phot_table["aperture_sum_err"].value[0])
            snr = float(
                compute_snr(
                    N_source=flux,
                    N_pix=aperture.area,
                    n_sky=local_background_rms**2,
                    dark_flux=dark_flux,
                    gain=1.,
                    )
            )
            
            return flux, flux_err, snr, local_background, local_background_rms

    def get_aperture(
        self,
        position: NDArray,
        psf_params: Dict[str, float],
        ) -> EllipticalAperture:
        
        if self.semimajor_axis is not None and self.semiminor_axis is not None and self.orientation is not None:
            return EllipticalAperture(
                position,
                self.semimajor_axis,
                self.semiminor_axis,
                self.orientation,
                )
        else:
            return EllipticalAperture(
                position,
                fwhm_scale * psf_params['semimajor_sigma'],
                fwhm_scale * psf_params['semiminor_sigma'],
                psf_params['orientation'],
                )

    def get_aperture_area(
        self,
        psf_params: Dict[str, float],
        ) -> float:
        """
        Get the area of the aperture.
        
        Parameters
        ----------
        psf_params : Dict[str, float],
            The PSF parameters.
        
        Returns
        -------
        float
            The area of the aperture.
        """
        
        return self.get_aperture(
            position=np.zeros(2),  # position does not matter
            psf_params=psf_params,
            ).area


class OptimalPhotometer(BasePhotometer):
    """
    A photometer that implements the optimal photometry method described in Naylor 1998, MNRAS, 296, 339-346.
    """

    def compute(
        self,
        image: NDArray,
        dark_flux: float,
        background_rms: NDArray | None,
        source_coords: NDArray,
        image_coords: NDArray | None,
        psf_params: Dict[str, float],
        ) -> Dict[str, List]:
        """
        Compute the fluxes of the catalogued sources from the given image.
        
        Parameters
        ----------
        image : NDArray
            The image. If `self.local_background_estimator` is undefined, this image will be background subtracted.
        dark_flux : float
            The dark current's "flux" contribution.
        background_rms : NDArray | None
            The background RMS. May be `None` if `self.local_background_estimator` is defined.
        source_coords : NDArray
            The source coordinates in the catalogue.
        image_coords : NDArray | None
            The source coordinates in the image. If `match_sources` is True, this will be used to match sources in the
            image to sources in the catalogue.
        psf_params : Dict[str, float]
            The PSF parameters for the camera used to take the image. This parameter is defined in the catalogue and
            has the following keys: 'semimajor_sigma' (in pixels), 'semiminor_sigma' (in pixels), and 'orientation' (in
            *degrees*).
        
        Returns
        -------
        Dict[str, List]
            The photometry results.
        """
        
        results = self.define_results_dict()
        
        for i in range(len(source_coords)):
            
            position = self.get_position(
                source_coords,
                image_coords,
                i,
                psf_params,
            )
            
            if position is None:
                results = self.pad_results_dict(results)
                continue
            
            results = self.populate_results_dict(
                results=results,
                phot_function=self.compute_optimal_flux,
                image=image,
                dark_flux=dark_flux,
                background_rms=background_rms,
                position=position,
                psf_params=psf_params,
                )
        
        return results

    def compute_optimal_flux(
        self,
        image: NDArray,
        dark_flux: float,
        background_rms: NDArray | None,
        position: NDArray,
        psf_params: Dict[str, float],
        ) -> Tuple[float, float] | Tuple[float, float, float, float]:
        """
        Compute the optimal flux of a source in the image as described in Naylor 1998, MNRAS, 296, 339-346.
        
        Parameters
        ----------
        image : NDArray
            The image.
        dark_flux : float
            The dark current's "flux" contribution.
        background_rms : NDArray | None
            The background RMS. May be `None` if `self.local_background_estimator` is defined.
        position : NDArray
            The position of the source in the image, given as (y, x) coordinates.
        psf_params : Dict[str, float]
            The PSF parameters for the camera used to take the image. This parameter is defined in the catalogue and
            has the following keys: 'semimajor_sigma' (in pixels), 'semiminor_sigma' (in pixels), and 'orientation' (in
            *degrees*).
        
        Returns
        -------
        Tuple[float, float] | Tuple[float, float, float, float]
            The flux and flux error. If `local_background_estimator` is defined, the background and its error are also
            returned.
        """
        
        weights = self.get_weights(
            width=image.shape[1],
            height=image.shape[0],
            position=position,
            psf_params=psf_params,
            )
        
        if self.local_background_estimator is None:
            flux = np.sum(image * weights)
            error = propagate_errors(
                data=image,
                dark_flux=dark_flux,
                background_rms=np.asarray(background_rms),
                )
            flux_error = np.sqrt(np.sum((error * weights)**2))
            
            return flux, flux_error
        else:
            # estimate local background using annulus
            local_background, local_background_rms = self.local_background_estimator(
                image,
                position,
                psf_params['semimajor_sigma'],
                psf_params['semiminor_sigma'],
                psf_params['orientation'],
                )
            
            image_clean = image - local_background
            error = propagate_errors(
                data=image,
                dark_flux=dark_flux,
                background_rms=local_background_rms,
                )
            
            flux = np.sum(image_clean * weights)
            flux_error = np.sqrt(np.sum((error * weights)**2))
            
            return flux, flux_error, local_background, local_background_rms

    @staticmethod
    def get_weights(
        width: int,
        height: int,
        position: NDArray,
        psf_params: Dict[str, float],
        ) -> NDArray:
        """
        Compute the optimal weight for each pixel in an image.
        
        Parameters
        ----------
        width : int
            The width of the image.
        height : int
            The height of the image.
        position : NDArray
            The position of the source.
        psf_params : Dict[str, float]
            The PSF parameters.
        
        Returns
        -------
        NDArray
            The normalised weights.
        """
        
        # define pixel coordinates
        y, x = np.ogrid[:height, :width]
        
        theta = psf_params['orientation'] * np.pi / 180
        
        # offset coordinates to the position of the source and align axes with the orientation of the PSF
        x0, y0 = position
        x_rot = (x - x0) * np.cos(theta) + (y - y0) * np.sin(theta)
        y_rot = -(x - x0) * np.sin(theta) + (y - y0) * np.cos(theta)
        
        weights = np.exp(- .5 * ((x_rot / psf_params['semimajor_sigma'])**2 + (y_rot / psf_params['semiminor_sigma'])**2))
        
        return weights / np.sum(weights)




def perform_photometry(
    file: str,
    photometer: BasePhotometer,
    source_coords: NDArray,
    gains: Dict[str, float],
    bmjds: Dict[str, float],
    barycenter: bool,
    flat_corrector: FlatFieldCorrector | None,
    rebin_factor: int,
    remove_cosmic_rays: bool,
    background: BaseBackground,
    threshold: float,
    finder: DefaultFinder,
    psf_params: Dict[str, Dict[str, float]],
    fltr: str,
    logger: Logger,
    ) -> Dict[str, List]:
    """
    Perform photometry on a file.
    
    Parameters
    ----------
    file : str
        The file path.
    photometer : BasePhotometer
        The photometer to use.
    source_coords : NDArray
        The coordinates of the sources.
    gains : Dict[str, float]
        The image gain.
    bmjds : Dict[str, float]
        The image time stamps.
    barycenter : bool
        Whether to apply a barycentric correction to the image time stamps.
    flat_corrector : FlatFieldCorrector | None
        The flat field corrector.
    rebin_factor : int
        The software pixel rebinning factor.
    remove_cosmic_rays : bool
        Whether to remove cosmic rays from the image.
    background : BaseBackground
        The two-dimensional background estimator.
    threshold : float
        The scalar source detection threshold in units of background RMS.
    finder : DefaultFinder
        The source finder.
    psf_params : Dict[str, Dict[str, float]]
        The PSF parameters.
    fltr : str
        The image filter.
    logger : Logger
        The logger.
    
    Returns
    -------
    Dict[str, List]
        The photometry results.
    """
    
    image, dark_flux = get_data(
        file=file,
        flat_corrector=flat_corrector,
        rebin_factor=rebin_factor,
        remove_cosmic_rays=remove_cosmic_rays,
        )
    
    if photometer.local_background_estimator is None:
        bkg = background(image)  # get 2D background
        image = image - bkg.background  # remove background from image
        threshold = threshold * bkg.background_rms  # define source detection threshold
        background_rms = bkg.background_rms.copy()
    else:
        # estimate source detection threshold from noisy image
        threshold = detect_threshold(image, threshold)  # type: ignore
        background_rms = None
    
    image_coords = None  # assume no image coordinates by default
    if not photometer.forced:
        try:
            tbl = finder(image, threshold)
            image_coords = np.array([tbl["xcentroid"].value,
                                    tbl["ycentroid"].value]).T
        except Exception as e:
            logger.warning(f"[OPTICAM] Could not determine source coordinates in {file}: {e}")
    
    results = photometer.compute(
        image=image,
        dark_flux=dark_flux,
        background_rms=background_rms,
        source_coords=source_coords,
        image_coords=image_coords,
        psf_params=psf_params[fltr],
        )
    
    assert 'flux' in results, f"[OPTICAM] Photometer {photometer.__class__.__name__}'s compute method must return a 'flux' key."
    assert 'flux_err' in results, f"[OPTICAM] Photometer {photometer.__class__.__name__}'s compute method must return a 'flux_err' key."
    
    # add time stamp
    if barycenter:
        results['BMJD'] = bmjds[file]  # type: ignore
    else:
        results['MJD'] = bmjds[file]  # type: ignore
    
    return results


def get_growth_curve(
    image: NDArray,
    x_centroid: float,
    y_centroid: float,
    r_max: int,
    ) -> Tuple[NDArray, NDArray]:
    """
    Compute the growth curve for a point in an image.
    
    Parameters
    ----------
    image : NDArray
        The image.
    x_centroid : float
        The x centroid of the point.
    y_centroid : float
        The y centroid of the point.
    r_max : int
        The maximum radius in pixels.
    
    Returns
    -------
    Tuple[NDArray, NDArray]
        _description_
    """
    
    position = np.array([x_centroid, y_centroid])
    
    radii, fluxes = [], []
    
    for r in range(1, r_max):
        radii.append(r)
        
        photometer = AperturePhotometer(
            semimajor_axis=r,
            semiminor_axis=r,
            orientation=0,
            forced=True,
            )
        
        flux = photometer.compute_aperture_flux(
            image=image,
            dark_flux=0.,  # flux error not needed so dark flux not required
            background_rms=0.,  # flux error not needed so background RMS not required
            position=position,
            psf_params={},  # empty dict since not needed
            )[0]
        
        fluxes.append(flux)
    
    return np.array(radii), np.array(fluxes)


def compute_snr(
    N_source: float | NDArray,
    N_pix: float,
    n_sky: float,
    dark_flux: float,
    gain: float,
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
    dark_flux : float
        The dark current's "flux" contribution per pixel.
    gain: float
        The detector gain.
    
    Returns
    -------
    float | NDArray
        The S/N ratio.
    """
    
    # this function is included to avoid a circular import error between noise.py and photometers.py
    # TODO: resolve this and remove this function
    
    source_photons = N_source * gain
    sky_photons_per_pix = n_sky * gain
    read_counts_per_pixel = n_read
    
    return source_photons / np.sqrt(source_photons + N_pix * (sky_photons_per_pix + dark_flux + read_counts_per_pixel**2))
