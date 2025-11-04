from astropy.table import QTable
import numpy as np
from numpy.typing import NDArray
from photutils.background import Background2D
from photutils.segmentation import SourceCatalog, SourceFinder

from opticam.background.global_background import BaseBackground


class DefaultFinder:
    """
    Default source finder. Combines image segmentation with source deblending.
    """
    
    def __init__(
        self,
        npixels: int,
        border_width: int = 0,
        ):
        """
        Default source finder. Combines image segmentation with source deblending.
        
        Parameters
        ----------
        npixels : int
            The minimum number of connected source pixels.
        border_width : int, optional
            Sources within this many pixels of the border will be ignored, by default 0 (no sources are ignored).
        """
        
        assert type(npixels) is int and npixels > 0, '[OPTICAM] npixels must be a positive integer.'
        
        self.border_width = border_width
        self.finder = SourceFinder(npixels=npixels, progress_bar=False)
    
    def __call__(
        self,
        data: NDArray,
        threshold: float | NDArray,
        ) -> QTable:
        
        segment_map = self.finder(data, threshold)
        
        if self.border_width > 0:
            segment_map.remove_border_labels(border_width=self.border_width, relabel=True)
        
        tbl = SourceCatalog(data, segment_map).to_table()
        tbl.sort('segment_flux', reverse=True)
        
        return tbl


def get_source_coords_from_image(
    image: NDArray,
    finder: DefaultFinder,
    threshold: float | int,
    bkg: Background2D | None = None,
    n_sources: int | None = None,
    background: BaseBackground | None = None,
    ) -> NDArray:
    """
    Get an array of source coordinates from an image in descending order of source brightness.
    
    Parameters
    ----------
    image : NDArray
        The **non-background-subtracted** image from which to extract source coordinates.
    bkg : Background2D, optional
        The background of the image, by default None. If None, the background is estimated from the image.
    n_sources : int, optional
        The number of source coordinates to return, by default `None` (all sources will be returned).
    
    Returns
    -------
    NDArray
        The source coordinates in descending order of brightness.
    """
    
    if bkg is None and background is not None:
        bkg = background(image)  # get background
    elif bkg is None and background is None:
        raise ValueError('[OPTICAM] get_source_coords_from_image() requires either bkg or background be specified.')
    
    image_clean = image - bkg.background
    
    tbl = finder(image_clean, threshold*bkg.background_rms)
    
    coords = np.array([tbl["xcentroid"], tbl["ycentroid"]]).T
    
    if n_sources is not None:
        coords = coords[:n_sources]
    
    return coords


