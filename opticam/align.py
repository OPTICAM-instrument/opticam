from logging import Logger
from typing import Dict, Literal, List, Tuple

from astroalign import find_transform
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from skimage.transform import SimilarityTransform, warp

from opticam.background.global_background import BaseBackground
from opticam.correctors.flat_field_corrector import FlatFieldCorrector
from opticam.finders import DefaultFinder, get_source_coords_from_image
from opticam.utils.fits_handlers import get_data


def find_translation(
    reference_coords: NDArray,
    coords: NDArray
    ) -> SimilarityTransform:
    """
    Find the translation that maps `reference_coords` onto `coords`. An advantage of this over
    `astroalign.find_transform()` is that it requires fewer sources, however it is therefore more prone to errors. In
    general, `astroalign.find_transform()` should be used, but `find_translation()` is available if there are too few
    sources for `astroalign.find_transform()`.
    
    Parameters
    ----------
    reference_coords : NDArray
        The reference source coordinates.
    coords : NDArray
        The source coordinates.
    
    Returns
    -------
    SimilarityTransform
        The transformation matrix that maps `coords` onto `reference_coords`.
    """
    
    distance_matrix = cdist(reference_coords, coords)
    reference_indices, indices = linear_sum_assignment(distance_matrix)
    
    dx = np.mean(reference_coords[reference_indices, 0] - coords[indices, 0])
    dy = np.mean(reference_coords[reference_indices, 1] - coords[indices, 1])
    
    return SimilarityTransform(translation=[dx, dy])


def align_batch(
    batch: List[str],
    reference_image_shape: Tuple[int],
    reference_coords: NDArray,
    transform_type: Literal['affine', 'translation'],
    rotation_limit: float | None,
    scale_limit: float | None,
    translation_limit: List[float] | None,
    n_alignment_sources: int,
    gains: Dict[str, float],
    flat_corrector: FlatFieldCorrector | None,
    rebin_factor: int,
    remove_cosmic_rays: bool,
    background: BaseBackground,
    finder: DefaultFinder,
    threshold: float | int,
    logger: Logger,
    ) -> Tuple[NDArray, Dict[str, float], Dict[str, Dict[str, float]]]:
    """
    Align an image based on some reference coordinates.
    
    Parameters
    ----------
    file: str
        The file path.
    reference_image : NDArray
        The reference image.
    reference_coords : NDArray
        The source coordinates in the reference image.
    transform_type : Literal['affine', 'translation']
        The type of transform to use for image alignment.
    rotation_limit : float | None
        The maximum rotation limit (in degrees) for image alignment.
    scale_limit : float | None
        The maximum scaling limit for image alignment.
    translation_limit : List[float] | None
        The maximum translation limit for image alignment.
    n_alignment_sources : int
        The (maximum) number of sources to use for image alignment.
    
    Returns
    -------
    Tuple[NDArray, Dict[str, float], Dict[str, Dict[str, float]]]
        The stacked image, transforms, and background results.
    """
    
    stacked_image = np.zeros(reference_image_shape)  # create empty stacked image
    transforms = {}
    bkg_dict = {}
    
    for file in batch:
        data = get_data(
            file,
            flat_corrector=flat_corrector,
            rebin_factor=rebin_factor,
            remove_cosmic_rays=remove_cosmic_rays,
            )[0]
        
        # calculate and subtract background
        bkg = background(data)
        
        # identify sources
        try:
            coords = get_source_coords_from_image(data, finder=finder, threshold=threshold, bkg=bkg)
        except Exception as e:
            logger.info(f'[OPTICAM] No sources detected in {file}: {e}.')
            continue
        
        if len(coords) < n_alignment_sources and transform_type == 'translation':
            logger.info(f'[OPTICAM] {len(coords)} sources detected in {file} but n_alignment_sources={n_alignment_sources} and transform_type="translation". Skipping. To attempt to align images in which fewer than n_alignment_sources are detected, try transform_type="affine".')
            continue
        
        if transform_type == 'translation':
            # find translation
            transform = find_translation(
                coords,
                reference_coords,
                )
        else:
            # find affine transformation using astroalign
            try:
                transform = find_transform(
                    reference_coords,
                    coords,
                    max_control_points=n_alignment_sources,
                    )[0]
            except Exception as e:
                logger.info(f'[OPTICAM] Could not align {file} due to the following exception: {e}. Skipping.')
                continue
        
        # validate transform
        if not valid_transform(
            file=file,
            transform=transform,
            rotation_limit=rotation_limit,
            scale_limit=scale_limit,
            translation_limit=translation_limit,
            logger=logger,
            ):
            continue
        
        transforms[file] = transform.params.tolist()  # type: ignore
        bkg_dict[file] = {
            'Median': bkg.background_median,
            'RMS': bkg.background_rms_median,
            }
        
        # transform and stack image
        stacked_image += warp(
            data - bkg.background,
            transform,
            output_shape=reference_image_shape,
            order=3,
            mode='constant',
            cval=0.,
            clip=True,
            preserve_range=True,
            )
    
    return stacked_image, transforms, bkg_dict


def valid_transform(
    file: str,
    transform: SimilarityTransform,
    rotation_limit: float | None,
    scale_limit: float | None,
    translation_limit: List[float] | None,
    logger: Logger,
    ) -> bool:
    """
    Find whether a transform is valid given some transform limits.
    
    Parameters
    ----------
    file : str
        The file being transformed.
    transform : SimilarityTransform
        The transform.
    rotation_limit : float | None
        The rotation limit.
    scale_limit : float | None
        The scale limit.
    translation_limit : List[float] | None
        The translation limit.
    
    Returns
    -------
    bool
        Whether the transform is valid.
    """
    
    if rotation_limit:
        if abs(transform.rotation) > rotation_limit:
            logger.info(f'[OPTICAM] File {file} transform exceeded rotation limit. Rotation limit is {rotation_limit}, but rotation was {transform.rotation}.')
            return False
    if scale_limit:
        if transform.scale > scale_limit:
            logger.info(f'[OPTICAM] File {file} transform exceeded scale limit. Scale limit is {scale_limit}, but scale was {transform.scale}.')
            return False
    if translation_limit:
        if abs(transform.translation[0]) > translation_limit[0] or abs(transform.translation[1]) > translation_limit[1]:
            logger.info(f'[OPTICAM] File {file} transform exceeded translation limit. Translation limit is {translation_limit}, but translation was {transform.translation}.')
            return False
    
    return True


