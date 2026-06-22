import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from skimage.transform import SimilarityTransform


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


