import os
from typing import Dict, List

from astropy.io import fits
import numpy as np
from numpy.typing import NDArray

from opticam.utils.helpers import create_file_paths
from opticam.utils.logging import log_binnings, log_filters

class FlatFieldCorrector:
    """
    Helper class for performing flat-field corrections on OPTICAM images.
    """
    
    def __init__(
        self,
        out_directory: str,
        flats_directory: str | None = None,
        c1_flats_directory: str | None = None,
        c2_flats_directory: str | None = None,
        c3_flats_directory: str | None = None,
        ) -> None:
        """
        Helper class for performing flat-field corrections on OPTICAM images.
        
        Parameters
        ----------
        out_directory : str
            The output directory.
        flats_dir : str, optional
            The directory path to the flat-field images, by default None (no flat-field correction). Note: this
            should point to a directory containing only flat-field images, NOT to a master flat-field image(s). The
            master flat-field image(s) will be created from the flat-field images in this directory and saved in
            out_dir/calibration_images. This parameter will override c1_flats_dir, c2_flats_dir, and c3_flats_dir if
            they are also defined. flats_dir is assumed to contain flat-field images for all cameras being used.
        c1_flats_dir : str, optional
            The directory path to the flat-field images for C1 only, by default None. If flats_dir is also defined, this
            parameter will be ignored.
        c2_flats_dir : str, optional
            The directory path to the flat-field images for C2 only, by default None. If flats_dir is also defined, this
            parameter will be ignored.
        c3_flats_dir : str, optional
            The directory path to the flat-field images for C3 only, by default None. If flats_dir is also defined, this
            parameter will be ignored.
        """
        
        self.out_directory = out_directory
        
        if not os.path.exists(out_directory):
            try:
                os.makedirs(out_directory)
            except:
                raise Exception(f"[OPTICAM] could not create output directory {out_directory}")
        
        # get paths to flats
        flat_paths = create_file_paths(
            data_directory=flats_directory,
            c1_directory=c1_flats_directory,
            c2_directory=c2_flats_directory,
            c3_directory=c3_flats_directory,
        )
        
        # {filter: flat_paths}
        self.flat_paths = validate_flat_files(
            flat_paths=sorted(flat_paths),
            out_directory=self.out_directory,
            )
        
        # load master flats if they already exist
        self.master_flats = {}
        if os.path.isfile(os.path.join(self.out_directory, 'master_flats.fits.gz')):
            self.master_flats.update(
                read_master_flats(
                    self.out_directory,
                    ),
                )
    
    def create_master_flats(self, overwrite: bool = False) -> None:
        """
        Create master flat-field images for each available filter.
        
        Parameters
        ----------
        overwrite : bool, optional
            Whether to overwrite the existing master flat-field image, by default False.
        """
        
        if os.path.isfile(os.path.join(self.out_directory, 'master_flats.fits.gz')) and not overwrite:
            print(f'[OPTICAM] Master flats file already exists. To overwrite existing flats, set overwrite=True.')
            return
        
        for fltr in self.flat_paths.keys():
            
            if len(self.flat_paths[fltr]) == 1:
                raise Exception(f"[OPTICAM] Only one {fltr} flat found. Master flats cannot be created from a single image.")
            
            # read flats
            flats = []
            for flat_path in self.flat_paths[fltr]:
                with fits.open(flat_path) as hdul:
                    flat = np.array(hdul[0].data)
                    flats.append(flat / np.median(flat))  # add normalised flat-field image to list
            
            # create master flat
            master_flat = np.median(flats, axis=0)
            
            # hold master flat in memory (faster than having to read it from disk every time correct() is called)
            self.master_flats[fltr] = master_flat
        
        save_master_flats(
            master_flats=self.master_flats,
            out_directory=self.out_directory,
            overwrite=overwrite,
            )
    
    def correct(self, image: NDArray, fltr: str) -> NDArray:
        """
        Correct an image for flat-fielding.
        
        Parameters
        ----------
        image : np.ndarray
            The image to correct.
        fltr : str
            The image filter.
        
        Returns
        -------
        NDArray
            The corrected image.
        """
        
        if fltr not in self.flat_paths.keys():
            raise ValueError(f"[OPTICAM] No flat-field images found for {fltr} filter.")
        
        if fltr not in self.master_flats.keys() or self.master_flats[fltr] is None:
            print(f'[OPTICAM] {fltr} master flat-field image not found. Attempting to create...')
            try:
                self.create_master_flats()
                print('[OPTICAM] Master flat-field image created.')
            except Exception as e:
                raise Exception(f"[OPTICAM] Could not create master flat-field image(s): {e}")
        
        # correct image for flat-fielding
        return image / self.master_flats[fltr]


def validate_flat_files(
    flat_paths: List[str],
    out_directory: str,
    ) -> Dict[str, List[str]]:
    """
    Ensure that the flat-field images in the specified directory are valid (i.e., contain at most three filters
    and use the same binning).
    
    Parameters
    ----------
    flat_paths : List[str]
        The paths to the flat-field images.
    out_directory : str
        The path to the output directory.
    
    Returns
    -------
    Dict[str, List[str]]
        A dictionary containing the paths to the flat-field images for each filter.
    """
    
    filters, binnings = {}, {}
    
    for flat_path in flat_paths:
        with fits.open(flat_path) as hdul:
            header = hdul[0].header
            filters[flat_path] = (header["FILTER"])
            binnings[flat_path] = (header["BINNING"])
    
    unique_filters = set(filters.values())
    unique_binnings = set(binnings.values())
    
    if len(unique_filters) > 3:
        log_filters(flat_paths, out_directory)
        raise ValueError(f'[OPTICAM] More than three filters found in the flat-field images. Image filters have been logged to {os.path.join(out_directory, 'diag/filters.json')}')
    
    if len(unique_binnings) > 1:
        log_binnings(flat_paths, out_directory)
        raise ValueError(f'[OPTICAM] Inconsistent binning detected in the flat-field images. Image binnings have been logged to {os.path.join(out_directory, 'diag/binnings.json')}')
    
    # get flats for each filter
    flats = {}
    for fltr in unique_filters:
        flats[fltr + '-band'] = []
        for k, v in filters.items():
            if v == fltr:
                flats[fltr + '-band'].append(k)
    
    for k, v in flats.items():
        print(f'[OPTICAM] {len(v)} {k} flat-field images.')
    
    return flats


def save_master_flats(
    master_flats: Dict[str, NDArray],
    out_directory: str,
    overwrite: bool,
    ) -> None:
    """
    Save some master flats to a compressed FITS cube.
    
    Parameters
    ----------
    master_flats : Dict[str, NDArray]
        The master flats {filter: master flat}.
    """
    
    hdr = fits.Header()
    hdr['COMMENT'] = 'This FITS file contains master flat-field images for each filter.'
    empty_primary = fits.PrimaryHDU(header=hdr)
    hdul = fits.HDUList([empty_primary])
    
    for fltr, img in master_flats.items():
        hdr = fits.Header()
        hdr['FILTER'] = fltr
        hdu = fits.ImageHDU(img, hdr)
        hdul.append(hdu)
    
    file_path = os.path.join(out_directory, 'master_flats.fits.gz')
    
    if not os.path.isfile(file_path) or overwrite:
        hdul.writeto(file_path, overwrite=overwrite)


def read_master_flats(
    out_directory: str,
    ) -> Dict[str, NDArray]:
    """
    Read the master flat-field images from the output directory.
    
    Parameters
    ----------
    out_directory : str
        The directory path to the reduction output.
    
    Returns
    -------
    Dict[str, NDArray]
        The master flat-field images {filter: image}.
    """
    
    master_flats = {}
    with fits.open(os.path.join(out_directory, 'master_flats.fits.gz')) as hdul:
        for hdu in hdul:
            if 'FILTER' not in hdu.header:
                continue
            fltr = hdu.header['FILTER']
            master_flats[fltr] = np.asarray(hdu.data)
    
    return master_flats



