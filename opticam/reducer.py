from functools import partial
import json
import logging
from multiprocessing import cpu_count
import os
from pathlib import Path
from typing import Callable, Dict, List, Literal, Tuple


from astroalign import find_transform
from astropy.table import QTable
from matplotlib import pyplot as plt
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from photutils.segmentation import detect_threshold
from skimage.transform import SimilarityTransform, warp
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm


from opticam.utils.transforms import find_translation
from opticam.background.global_background import BaseBackground, DefaultBackground
from opticam.correctors import BiasCorrector, DarkNoiseCorrector, FlatFieldCorrector
from opticam.finders import DefaultFinder, get_source_coords_from_image
from opticam.instruments import Instrument, OPTICAM_MX
from opticam.photometers import AperturePhotometer, BasePhotometer
from opticam.utils.batching import get_batches, get_batch_size
from opticam.utils.constants import bar_format
from opticam.utils.data_checks import scan_data
from opticam.utils.fits_handlers import get_data, get_stacked_images, save_stacked_images
from opticam.plotting.gifs import compile_gif, create_gif_frame
from opticam.utils.logging import recursive_log, log_psf_params
from opticam.plotting.plots import plot_backgrounds, plot_background_meshes, plot_catalogs, plot_growth_curves, \
    plot_time_between_files, plot_psf, plot_rms_vs_median_flux, plot_noise, plot_snrs, plot_apertures




class Reducer:
    """
    Class for reducing astronomical CCD data.
    """


    def __init__(
        self,
        out_directory: Path | str,
        data_directory: Path | str,
        aperture_selector: Callable = np.median,
        background: BaseBackground | None = None,
        barycenter: bool = True,
        bias_corrector: BiasCorrector | None = None,
        dark_corrector: DarkNoiseCorrector = DarkNoiseCorrector(),
        finder: None | Callable = None,
        flat_corrector: FlatFieldCorrector | None = None,
        instrument: Instrument = OPTICAM_MX(),
        number_of_processors: int = cpu_count() // 2,
        rebin_factor: int = 1,
        remove_cosmic_rays: bool = False,
        show_plots: bool = True,
        threshold: float = 5,
        verbose: bool = True
        ) -> None:
        """
        Class for reducing OPTICAM data.
        
        Parameters
        ----------
        out_directory : Path | str
            The path to the directory to save the output files.
        data_directory : Path | str
            The path to the directory containing the data.
        aperture_selector : Callable, optional
            The aperture selector, by default `np.median`. This function is used to select the aperture size for
            photometry. If a callable is provided, it should take a list of source sizes (`List[float]`) as input and
            return a single value.
        background : BaseBackground | None, optional
            The background calculator, by default `None`. If `None`, the default background calculator is used. If a
            callable is provided, it should take an image (`NDArray`) as input and return a `Background2D` object.
        barycenter : bool, optional
            Whether to apply a barycentric correction to the image timestamps, by default `True`.
        bias_corrector : BiasCorrector | None, optional
            The bias corrector, by default `None`. If `None`, no bias corrections are performed. See 
            (TODO: link to corrections docs) for more details.
        dark_corrector : DarkNoiseCorrector, optional,
            The dark noise corrector, by default `DarkNoiseCorrector()`. To perform dark current corrections using
            dark images, a custom `DarkNoiseCorrector` instance must be passed. See (TODO: link to corrections docs)
            for more details.
        finder : Callable, optional
            The source finder, by default `None`. If `None`, the default source finder is used. If a callable is
            provided, it should take an image (`NDArray`) and a threshold (`float | NDArray`) as input and return a
            `QTable` instance. See (TODO: link to finders docs) for details.
        flat_corrector : FlatFieldCorrector | None, optional,
            The flat-field corrector, by default `None`. If `None`, no flat-field corrections are performed. See 
            (TODO: link to corrections docs) for more details.
        instrument : Instrument, optional
            The instrument, by default `OPTICAM_MX()`. To use a custom instrument, see (TODO: link to instruments docs)
            for details.
        number_of_processors : int, optional
            The number of processors to use for parallel processing, by default half the number of available processors.
        rebin_factor: int, optional
            The rebinning factor, by default 1 (no rebinning). The rebinning factor is the factor by which the image is
            rebinned in both dimensions. Rebinning can improve the detectability of faint sources and speed up
            some operations (like cosmic ray removal) at the cost of image resolution.
        remove_cosmic_rays : bool, optional
            Whether to remove cosmic rays from images, by default `False`. Cosmic rays are removed using the LACosmic
            algorithm as implemented in `astroscrappy`. Note: this can be computationally expensive, particularly for
            large images.
        show_plots : bool, optional
            Whether to show plots as they're created, by default `True`. Whether `True` or `False`, plots are always
            saved to `out_directory`.
        threshold : float, optional
            The signal-to-noise ratio threshold for source finding, by default 5. Reduce this value to identify fainter
            sources, though this may lead to the identification of spurious sources.
        verbose : bool, optional
            Whether to print verbose output, by default `True`.
        """
        
        self.verbose = verbose
        
        ########################################### out_directory ###########################################
        
        self.out_directory = Path(out_directory)
        
        # create output directory if it does not exist
        if not self.out_directory.is_dir():
            if self.verbose:
                print(f"[OPTICAM] {self.out_directory} not found, attempting to create ...")
            # create output directory if it does not exist
            try:
                os.makedirs(self.out_directory)
            except Exception as e:
                raise FileNotFoundError(f"[OPTICAM] Could not create directory {self.out_directory} due to the following exception: {e}.")
            if self.verbose:
                print(f"[OPTICAM] {self.out_directory} created.")
        
        ########################################### out_directory ###########################################
        
        self.instrument = instrument
        
        ########################################### logger ###########################################
        
        # configure logger
        self.logger = logging.getLogger('OPTICAM')
        self.logger.setLevel(logging.INFO)
        
        # clear existing handlers
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        
        # create file handler
        file_handler = logging.FileHandler(os.path.join(self.out_directory, 'info.log'))
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        ########################################### sub-directories ###########################################
        
        # create subdirectories
        if not self.out_directory.joinpath("cat").is_dir():
            os.makedirs(self.out_directory.joinpath("cat"))
        if not self.out_directory.joinpath("diag").is_dir():
            os.makedirs(self.out_directory.joinpath("diag"))
        if not self.out_directory.joinpath("misc").is_dir():
            os.makedirs(self.out_directory.joinpath("misc"))
        
        ########################################### input params ###########################################
        
        # set some useful parameters
        self.rebin_factor = rebin_factor
        self.aperture_selector = aperture_selector
        self.threshold = threshold
        self.remove_cosmic_rays = remove_cosmic_rays
        self.barycenter = barycenter
        self.number_of_processors = number_of_processors
        self.show_plots = show_plots
        
        ########################################### check input data ###########################################
        
        self.data_directory = Path(data_directory)
        
        self.camera_files, self.binning_scale, self.bmjds, self.ignored_files, self.t_ref = scan_data(
                data_directory=self.data_directory,
                out_directory=out_directory,
                instrument=instrument,
                barycenter=self.barycenter,
                verbose=verbose,
                return_output=True,
                logger=self.logger,
                number_of_processors=number_of_processors,
                )
        
        ########################################### correctors ###########################################
        
        self.bias_corrector = bias_corrector
        if self.bias_corrector is not None:
            errors = self.bias_corrector.run_checks(
                data_file_paths_by_filter=self.camera_files,
                return_errors=True,
                )
            if errors == 1:
                raise ValueError(f'[OPTICAM] {errors} BiasCorrector error needs to be resolved.')
            elif errors > 1:
                raise ValueError(f'[OPTICAM] {errors} BiasCorrector errors need to be resolved.')
        
        self.dark_corrector = dark_corrector
        if self.dark_corrector is not None:
            errors = self.dark_corrector.run_checks(
                data_file_paths_by_filter=self.camera_files,
                return_errors=True,
                )
            if errors == 1:
                raise ValueError(f'[OPTICAM] {errors} DarkNoiseCorrector error needs to be resolved.')
            elif errors > 1:
                raise ValueError(f'[OPTICAM] {errors} DarkNoiseCorrector errors need to be resolved.')
        
        self.flat_corrector = flat_corrector
        if self.flat_corrector is not None:
            errors = self.flat_corrector.run_checks(
                data_file_paths_by_filter=self.camera_files,
                return_errors=True,
                )
            if errors == 1:
                raise ValueError(f'[OPTICAM] {errors} FlatFieldCorrector error needs to be resolved.')
            elif errors > 1:
                raise ValueError(f'[OPTICAM] {errors} FlatFieldCorrector errors need to be resolved.')
        
        ########################################### plot time between files ###########################################
        
        plot_time_between_files(
            out_directory=self.out_directory,
            camera_files=self.camera_files,
            bmjds=self.bmjds,
            show=self.show_plots,
            save=True,
            )
        
        ########################################### define reference images ###########################################
        
        # define middle image as reference image for each filter
        reference_indices = {}
        self.reference_files = {}
        for key in self.camera_files.keys():
            reference_indices[key] = len(self.camera_files[key]) // 2
            self.reference_files[key] = self.camera_files[key][reference_indices[key]]
        
        ########################################### aperture selector ###########################################
        
        assert callable(aperture_selector), "[OPTICAM] aperture_selector must be callable."
        self.aperture_selector = aperture_selector
        
        ########################################### background ###########################################
        
        if background is None:
            box_size = 2048 // self.binning_scale // self.rebin_factor // 32
            self.background = DefaultBackground(box_size)
            self.logger.info(f'[OPTICAM] Using default background estimator with box_size={box_size}.')
        elif callable(background):
            # use custom background estimator
            self.background = background
            self.logger.info(f'[OPTICAM] Using custom background estimator {background.__class__.__name__} with parameters {background.__dict__}.')
        else:
            raise ValueError('[OPTICAM] background must be a callable or None. If None, the default background estimator is used.')
        
        ########################################### finder ###########################################
        
        if finder is None:
            effective_image_size = 2048 // self.binning_scale // self.rebin_factor
            npixels = 128 // (2048 // effective_image_size)**2
            border_width = 2048 // self.binning_scale // self.rebin_factor // 16
            self.finder = DefaultFinder(npixels, border_width)
            self.logger.info(f'[OPTICAM] Using default source finder with npixels={npixels} and border_width={border_width}.')
        elif callable(finder):
            self.finder = finder
            self.logger.info(f'[OPTICAM] Using custom source finder {finder.__class__.__name__} with parameters {finder.__dict__}.')
        else:
            raise ValueError('[OPTICAM] finder must be a callable or None. If None, the default source finder is used.')
        
        ########################################### log input params ###########################################
        
        # TODO: check if reduction_parameters.json already exists and, if so, that it matches that current parameters
        log_reducer_params(self)
        
        ########################################### misc attributes ###########################################
        
        self.transforms = {}  # define transforms as empty dictionary
        self.unaligned_files = []  # define unaligned files as empty list
        self.catalogs : Dict[str, QTable] = {}  # define catalogs as empty dictionary
        self.psf_params = {}  # define PSF parameters as empty dictionary
        
        ########################################### read transforms ###########################################
        
        if os.path.isfile(os.path.join(self.out_directory, "cat/transforms.json")):
            with open(os.path.join(self.out_directory, "cat/transforms.json"), "r") as file:
                self.transforms.update(json.load(file))
            
            if self.verbose:
                self.logger.info("[OPTICAM] Read transforms from file.")
        
        ########################################### read catalogs ###########################################
        
        for fltr in list(self.camera_files.keys()):
            file_path = os.path.join(self.out_directory, f"cat/{fltr}_catalog.ecsv")
            if os.path.isfile(file_path):
                self.catalogs.update(
                    {
                        fltr: QTable.read(
                            file_path,
                            format="ascii.ecsv",
                            )
                        }
                    )
                self.psf_params[fltr] = set_psf_params(
                    aperture_selector=self.aperture_selector,
                    catalog=self.catalogs[fltr],
                    )
                if self.verbose:
                    print(f"[OPTICAM] Read {fltr} catalog from file.")
        
        ########################################### read unaligned files ###########################################
        
        file_path = os.path.join(self.out_directory, 'diag/unaligned_files.txt')
        if os.path.isfile(file_path):
            with open(file_path, 'r') as file:
                for line in file:
                    self.unaligned_files.append(line)
            
            if self.verbose:
                    print(f"[OPTICAM] Read unaligned files from file.")


    def create_catalogs(
        self,
        max_catalog_sources: int = 15,
        n_alignment_sources: int = 15,
        transform_type: Literal['affine', 'translation'] = 'affine',
        rotation_limit: float | None = None,
        translation_limit: float | int | List[float | int] | None = None,
        scale_limit: float | None = None,
        overwrite: bool = False,
        ) -> None:
        """
        Initialise the source catalogs for each camera. Some aspects of this method are parallelised for speed.
        
        Parameters
        ----------
        max_catalog_sources : int, optional
            The maximum number of sources to include in the catalog, by default 30. Since source IDs are ordered by
            brightness, the brightest `max_catalog_sources` sources are included in the catalog.
        n_alignment_sources : int, optional
            The (maximum) number of sources to use for image alignment, by default 30. If
            `transform_type='translation'`, `n_alignment_sources` must be >= 1, and the brightest `n_alignment_sources`
            sources are used for image alignment. If `transform_type='affine'`, `n_alignment_sources` must be >= 3 and
            represents that *maximum* number of sources that *may* be used for image alignment.
        transform_type : Literal['affine', 'translation'], optional
            The type of transform to use for image alignment, by default 'affine'. 'translation' performs simple
            x, y translations, while 'affine' uses `astroalign.find_transform()`. 'affine' is generally more robust 
            (and is therefore the default) while 'translation' can work with fewer sources.
        rotation_limit : float, optional
            The maximum rotation limit (in degrees) for affine transformations, by default `None` (no limit).
        scale_limit : float, optional
            The maximum scale limit for affine transformations, by default `None` (no limit).
        translation_limit : float | int | List[float | int] | None, optional
            The maximum translation limit for both types of transformations, by default `None` (no limit). Can be a
            scalar value that applies to both x- and y-translations, or an iterable where the first value defines the
            x-translation limit and the second value defines the y-translation limit.
        overwrite : bool, optional
            Whether to overwrite existing catalogs, by default False.
        """
        
        assert transform_type in ['affine', 'translation'], '[OPTICAM] transform_type must be either "affine" or "translation".'
        
        if translation_limit is not None:
            # if a scalar translation limit is specified, convert it to a list
            if isinstance(translation_limit, float) or isinstance(translation_limit, int):
                translation_limit = [translation_limit, translation_limit]
        
        # if catalogs already exist, skip
        if os.path.isfile(os.path.join(self.out_directory, 'cat/catalogs.pdf')) and not overwrite:
            print('[OPTICAM] Catalogs already exist. To overwrite, set overwrite=True.')
            
            plot_catalogs(
                out_directory=self.out_directory,
                stacked_images=get_stacked_images(self.out_directory),
                catalogs=self.catalogs,
                show=self.show_plots,
                save=False,
            )
            
            return
        
        if self.verbose:
            print('[OPTICAM] Creating source catalogs')
        
        stacked_images = {}
        
        # for each camera
        for fltr in self.camera_files.keys():
            
            # if no images found for camera, skip
            if len(self.camera_files[fltr]) == 0:
                continue
            
            # get reference image
            reference_image = get_data(
                file_path=self.reference_files[fltr],
                instrument=self.instrument,
                bias_corrector=self.bias_corrector,
                dark_corrector=self.dark_corrector,
                flat_corrector=self.flat_corrector,
                rebin_factor=self.rebin_factor,
                remove_cosmic_rays=self.remove_cosmic_rays,
                )[0]
            
            try:
                # get source coordinates in descending order of brightness
                reference_coords = get_source_coords_from_image(
                    reference_image,
                    finder=self.finder,  # type: ignore
                    threshold=self.threshold,
                    background=self.background,
                    n_sources=n_alignment_sources,
                    )
            except Exception as e:
                self.logger.info(f'[OPTICAM] No sources detected in {fltr} reference image ({self.reference_files[fltr]}): {e}. Reducing threshold or npixels in the source finder may help.')
                continue
            
            if len(reference_coords) < n_alignment_sources and transform_type == 'translation':
                self.logger.info(f'[OPTICAM] Found {len(reference_coords)} sources in {fltr} reference image ({self.reference_files[fltr]}) but n_alignment_sources={n_alignment_sources}. transform_type="translation" requires at least n_alignment_sources be detected in the reference image to work. Consider reducing n_alignment_sources and/or threshold, or using transform_type="affine".')
                continue
            
            self.logger.info(f'[OPTICAM] {fltr} alignment source coordinates: {reference_coords}')
            
            # align and stack images in batches
            batches = get_batches(self.camera_files[fltr])
            results = process_map(
                partial(
                    self._align_batch,
                    reference_image_shape=reference_image.shape,
                    reference_coords=reference_coords,
                    transform_type=transform_type,
                    rotation_limit=rotation_limit,
                    scale_limit=scale_limit,
                    translation_limit=translation_limit,
                    n_alignment_sources=n_alignment_sources,
                    ),
                batches,
                max_workers=self.number_of_processors,
                disable=not self.verbose,
                desc=f'[OPTICAM] Aligning {fltr} images',
                bar_format=bar_format,
                tqdm_class=tqdm,
                )
            
            self.transforms, self.unaligned_files, stacked_image, background = parse_alignment_results(
                results=results,
                camera_files=self.camera_files[fltr],
                transforms=self.transforms,
                unaligned_files=self.unaligned_files,
                verbose=self.verbose,
                )
            
            try:
                # estimate threshold for source detection
                threshold = detect_threshold(
                    stacked_image,
                    nsigma=self.threshold,
                    )
            except Exception as e:
                self.logger.info(f'[OPTICAM] Unable to estimate source detection threshold for {fltr} stacked image: {e}.')
                continue
            
            try:
                # identify sources in stacked image
                tbl = self.finder(
                    stacked_image,
                    threshold,
                    )
            except Exception as e:
                self.logger.info(f'[OPTICAM] No sources detected in the stacked {fltr} stacked image: {e}. Reducing threshold may help.')
                continue
            
            # save stacked image
            stacked_images[fltr] = stacked_image
            
            # limit catalog to brightest sources
            tbl = tbl[:max_catalog_sources]
            
            # save catalog
            self.catalogs.update({fltr: tbl})  # type: ignore
            self.catalogs[fltr].write(
                os.path.join(self.out_directory, f"cat/{fltr}_catalog.ecsv"),
                format="ascii.ecsv",
                overwrite=True,
                )
            
            self.psf_params[fltr] = set_psf_params(
                aperture_selector=self.aperture_selector,
                catalog=self.catalogs[fltr],
                )
            
            save_background(
                out_directory=self.out_directory,
                background=background,
                fltr=fltr,
                bmjds=self.bmjds,
                )
        
        log_psf_params(
            out_directory=self.out_directory,
            psf_params=self.psf_params,
            binning_scale=self.binning_scale,
            rebin_factor=self.rebin_factor,
            pixel_scales=self.instrument.pixel_scales
            )
        
        plot_catalogs(
            out_directory=self.out_directory,
            stacked_images=stacked_images,
            catalogs=self.catalogs,
            show=self.show_plots,
            save=True,
            )
        
        save_stacked_images(
            stacked_images=stacked_images,
            out_directory=self.out_directory,
            overwrite=overwrite,
            )
        
        plot_backgrounds(
            out_directory=self.out_directory,
            t_ref=self.t_ref,
            show=self.show_plots,
            save=True,
            )
        
        # save transforms to file
        if not os.path.isfile(os.path.join(self.out_directory, "cat/transforms.json")) or overwrite:
            with open(os.path.join(self.out_directory, "cat/transforms.json"), "w") as file:
                json.dump(self.transforms, file, indent=4)
        
        # save unaligned files to file
        save_unaligned_files(
            out_directory=self.out_directory,
            unaligned_files=self.unaligned_files,
            )


    def _align_batch(
        self,
        batch: List[Path],
        reference_image_shape: Tuple[int],
        reference_coords: NDArray,
        transform_type: Literal['affine', 'translation'],
        rotation_limit: float | None,
        scale_limit: float | None,
        translation_limit: List[float] | None,
        n_alignment_sources: int,
        ) -> Tuple[NDArray[np.float64], Dict[Path, List[float]], Dict[Path, Dict[str, float]]]:
        """
        Align a batch of images with respect to some reference coordinates.
        
        Parameters
        ----------
        batch: List[Path]
            The file path.
        reference_image_shape : Tuple[int]
            The reference image's shape.
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
        Tuple[NDArray[np.float64], Dict[Path, List[float]], Dict[Path, Dict[str, float]]]
            The stacked image, transforms, and background results.
        """
        
        stacked_image = np.zeros(reference_image_shape)  # create empty stacked image
        transforms: Dict[Path, List[float]] = {}
        bkg_dict: Dict[Path, Dict[str, float]] = {}
        
        for file_path in batch:
            data = get_data(
                file_path=file_path,
                instrument=self.instrument,
                bias_corrector=self.bias_corrector,
                dark_corrector=self.dark_corrector,
                flat_corrector=self.flat_corrector,
                rebin_factor=self.rebin_factor,
                remove_cosmic_rays=self.remove_cosmic_rays,
                )[0]
            
            # calculate and subtract background
            bkg = self.background(data)
            
            # identify sources
            try:
                coords = get_source_coords_from_image(
                    data,
                    finder=self.finder,  # type: ignore
                    threshold=self.threshold,
                    bkg=bkg,
                    )
            except Exception as e:
                self.logger.info(f'[OPTICAM] No sources detected in {file_path}: {e}.')
                continue
            
            if len(coords) < n_alignment_sources and transform_type == 'translation':
                self.logger.info(f'[OPTICAM] {len(coords)} sources detected in {file_path} but n_alignment_sources={n_alignment_sources} and transform_type="translation". Skipping. To attempt to align images in which fewer than n_alignment_sources are detected, try transform_type="affine".')
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
                    self.logger.info(f'[OPTICAM] Could not align {file_path} due to the following exception: {e}. Skipping.')
                    continue
            
            # validate transform
            if not self._valid_transform(
                file=file_path,
                transform=transform,
                rotation_limit=rotation_limit,
                scale_limit=scale_limit,
                translation_limit=translation_limit,
                ):
                continue
            
            transforms[file_path] = transform.params.tolist()  # type: ignore
            bkg_dict[file_path] = {
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


    def _valid_transform(
        self,
        file: Path,
        transform: SimilarityTransform,
        rotation_limit: float | None,
        scale_limit: float | None,
        translation_limit: List[float] | None,
        ) -> bool:
        """
        Find whether a transform is valid given some transform limits.
        
        Parameters
        ----------
        file : Path
            The path to the file being transformed.
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
                self.logger.info(f'[OPTICAM] File {file} transform exceeded rotation limit. Rotation limit is {rotation_limit}, but rotation was {transform.rotation}.')
                return False
        if scale_limit:
            if transform.scale > scale_limit:
                self.logger.info(f'[OPTICAM] File {file} transform exceeded scale limit. Scale limit is {scale_limit}, but scale was {transform.scale}.')
                return False
        if translation_limit:
            if abs(transform.translation[0]) > translation_limit[0] or abs(transform.translation[1]) > translation_limit[1]:
                self.logger.info(f'[OPTICAM] File {file} transform exceeded translation limit. Translation limit is {translation_limit}, but translation was {transform.translation}.')
                return False
        
        return True


    def plot_background_meshes(
        self,
        save: bool = False,
        ) -> None:
        """
        Plot the background mesh over an image from each filter to verify it's appropriately sized. If stacked catalog
        images exist, those will be used. Otherwise, a random image will be chosen for each filter.
        
        Parameters
        ----------
        save : bool, optional
            Whether to save the plot, by default `False`.
        """
        
        try:
            images = get_stacked_images(self.out_directory)
        except FileNotFoundError:
            images = get_random_image_for_each_filter(
                self.camera_files,
                instrument=self.instrument,
                )
            
            # subtract background
            for label, image in images.items():
                bkg = self.background(image)
                images[label] = image - bkg.background
        
        plot_background_meshes(
            out_directory=self.out_directory,
            images=images,
            background=self.background,
            show=self.show_plots,
            save=save,
            )


    def plot_growth_curves(
        self,
        targets: Dict[str, int | List[int]] | None = None,
        save: bool = False,
        ) -> None:
        """
        Plot the growth curves for the sources identified in the catalog images. The resulting plots are saved to
        out_directory/diag/growth_curves as PDF files.
        
        Parameters
        ----------
        targets : Dict[str, int | List[int]] | None, optional
            The targets for which growth curves will be created, by default `None` (growth curves are created for all
            catalog sources). To create growth curves for specific targets, pass a dictionary with keys listing the
            desired filters and values listing each filter's correpsonding target(s). For example:
            ```
            # plot growth curves for the three brightest sources in each catalog
            plot_growth_curves(
                targets = {
                    'g': [1, 2, 3],
                    'r': [1, 2, 3],
                    'i': [1, 2, 3],
                    },
                )
            ```
        save : bool, optional
            Whether to save the plots, by default `False`.
        """
        
        stacked_images = get_stacked_images(self.out_directory)
        
        # create targets dict if it does not already exist
        if targets is None:
            growth_curve_targets = create_targets_dict(self.catalogs)
        else:
            growth_curve_targets = targets
        
        self.logger.info(f'[OPTICAM] Generating growth curves for targets: {repr(growth_curve_targets)}')
        
        for fltr, cat in self.catalogs.items():
            
            if fltr not in growth_curve_targets.keys():
                self.logger.info(f'[OPTICAM] Filter {fltr} is not in target dictionary. Skipping.')
                continue
            
            fig = plot_growth_curves(
                image=stacked_images[fltr],
                cat=cat,
                targets=growth_curve_targets[fltr],
                psf_params=self.psf_params[fltr],
                read_noise=self.instrument.read_noise,
                )
            
            fig.suptitle(fltr, fontsize='large')
            
            dir_path = os.path.join(self.out_directory, 'diag/growth_curves')
            if not os.path.isdir(dir_path):
                os.makedirs(dir_path, exist_ok=True)
            
            if save:
                fig.savefig(os.path.join(dir_path, f'{fltr}_growth_curves.pdf'))
            
            if self.show_plots:
                plt.show(fig)
            else:
                plt.close(fig)
        
        self.logger.info('[OPTICAM] Growth curves generated.')


    def plot_psfs(
        self,
        ) -> None:
        """
        Plot the PSFs for the catalog sources.
        """
        
        if not os.path.isdir(os.path.join(self.out_directory, 'psfs')):
            os.makedirs(os.path.join(self.out_directory, 'psfs'))
        
        # get stacked images
        stacked_images = get_stacked_images(self.out_directory)
        
        for fltr in self.catalogs.keys():
            
            a = self.psf_params[fltr]['semimajor_sigma']
            b = self.psf_params[fltr]['semiminor_sigma']
            
            for source_indx in tqdm(
                range(len(self.catalogs[fltr])),
                disable=not self.verbose,
                desc=f'[OPTICAM] Plotting {fltr} PSFs',
                bar_format=bar_format,
                ):
                plot_psf(
                    catalog=self.catalogs[fltr],
                    source_indx=source_indx,
                    stacked_image=stacked_images[fltr],
                    fltr=fltr,
                    a=a,
                    b=b,
                    out_directory=self.out_directory,
                )


    def plot_snrs(
        self,
        save: bool = False,
        ) -> None:
        """
        Plot the signal-to-noise ratios for each catalogued source in the reference images.
        
        Parameters
        ----------
        save : bool, optional
            Whether to save the plot, by default `False`.
        """
        
        plot_snrs(
            out_directory=self.out_directory,
            file_paths=self.reference_files,
            background=self.background,
            psf_params=self.psf_params,
            catalogs=self.catalogs,
            instrument=self.instrument,
            dark_corrector=self.dark_corrector,
            show=self.show_plots,
            save=save,
        )


    def plot_noise(
        self,
        save: bool = False,
        ) -> None:
        """
        Plot the noise characterisation for each reference image.
        
        Parameters
        ----------
        save : bool, optional
            Whether to save the plot, by default 'False'.
        """
        
        plot_noise(
            out_directory=self.out_directory,
            file_paths=self.reference_files,
            background=self.background,
            psf_params=self.psf_params,
            catalogs=self.catalogs,
            instrument=self.instrument,
            dark_corrector=self.dark_corrector,
            show=self.show_plots,
            save=save,
            )


    def create_gifs(
        self,
        keep_frames: bool = True,
        overwrite: bool = False,
        ) -> None:
        """
        Create alignment gifs for each camera. Some aspects of this method are parallelised for speed. The frames are 
        saved in out_directory/diag/*_gif_frames and the GIFs are saved in out_directory/cat.
        
        Parameters
        ----------
        keep_frames : bool, optional
            Whether to save the GIF frames in out_directory/diag, by default True. If False, the frames will be deleted
            after the GIF is saved.
        overwrite : bool, optional
            Whether to overwrite existing GIFs, by default False.
        """
        
        # for each camera
        for fltr in list(self.catalogs.keys()):
            
            # skip cameras with no images
            if len(self.camera_files[fltr]) == 0:
                continue
            elif os.path.exists(os.path.join(self.out_directory, f"cat/{fltr}_images.gif")) and not overwrite:
                print(f"[OPTICAM] {fltr} GIF already exists. To overwrite, set overwrite to True.")
                continue
            
            # create gif frames directory if it does not exist
            if not os.path.isdir(os.path.join(self.out_directory, f"diag/{fltr}_gif_frames")):
                os.mkdir(os.path.join(self.out_directory, f"diag/{fltr}_gif_frames"))
            
            chunksize = get_batch_size(len(self.camera_files[fltr]))
            process_map(
                partial(
                    create_gif_frame,
                    out_directory=self.out_directory,
                    aperture_selector=self.aperture_selector,
                    catalog=self.catalogs[fltr],
                    fltr=fltr,
                    transforms=self.transforms,
                    reference_file=self.reference_files[fltr],
                    flat_corrector=self.flat_corrector,
                    rebin_factor=self.rebin_factor,
                    remove_cosmic_rays=self.remove_cosmic_rays,
                    background=self.background,
                    instrument=self.instrument,
                    ),
                self.camera_files[fltr],
                max_workers=self.number_of_processors,
                disable=not self.verbose,
                desc=f"[OPTICAM] Creating {fltr} GIF frames",
                chunksize=chunksize,
                bar_format=bar_format,
                tqdm_class=tqdm,
                )
            
            # save GIF
            compile_gif(
                out_directory=self.out_directory,
                fltr=fltr,
                camera_files=self.camera_files,
                keep_frames=keep_frames,
                )


    def plot_apertures(
        self,
        photometer: AperturePhotometer,
        targets: Dict[str, int] | Dict[str, List[int]] | Dict[str, List[int] | int] | None = None,
        save: bool = False,
        ) -> None:
        """
        Plot the apertures over each source.
        
        Parameters
        ----------
        photometer : AperturePhotometer
            The `AperturePhotometer` instance. If a local background estimator has been defined, this will also be
            plotted.
        targets : Dict[str, int] | Dict[str, List[int]] | Dict[str, List[int] | int] | None
            The targets for which apertures will be plotted, by default `None` (apertures are plotted for all
            sources). To plot apertures for specific targets, pass a dictionary with keys listing the
            desired filters and values listing each filter's correpsonding target(s). For example:
            ```
            # plot apertures for the three brightest sources in each filter
            photometer = opticam.AperturePhotometer()
            plot_apertures(
                photometer=photometer,
                targets = {
                    'g': [1, 2, 3],
                    'r': [1, 2, 3],
                    'i': [1, 2, 3],
                    },
                )
            ```
        save : bool, optional
            Whether to save the plots, by default `False`.
        """
        
        if targets is None:
            targets = create_targets_dict(self.catalogs)
        
        for fltr in self.catalogs.keys():
            if fltr not in targets.keys():
                continue
            
            img = get_data(
                file_path=self.reference_files[fltr],
                instrument=self.instrument,
                bias_corrector=self.bias_corrector,
                dark_corrector=self.dark_corrector,
                flat_corrector=self.flat_corrector,
                rebin_factor=self.rebin_factor,
                remove_cosmic_rays=self.remove_cosmic_rays,
                )[0]
            
            plot_apertures(
                out_directory=self.out_directory,
                data=img,
                cat=self.catalogs[fltr],
                targets=targets[fltr],
                photometer=photometer,
                psf_params=self.psf_params[fltr],
                fltr=fltr,
                show=self.show_plots,
                save=save,
            )


    def photometry(
        self,
        photometer: BasePhotometer,
        overwrite: bool = False,
        ) -> None:
        """
        Perform photometry on the catalogs using the provided photometer.
        
        Parameters
        ----------
        photometer : BasePhotometer
            The photometer. Should be a subclass of `BasePhotometer`, or implement a `compute` method that follows the
            `BasePhotometer` interface.
        overwrite : bool, optional
            Whether to overwrite any existing light curves files computed using the same photometer, by default `False`.
        """
        
        # define save directory using the photometer name
        save_name = photometer.get_label()
        
        print(f'[OPTICAM] Photometry results will be saved to lcs/{save_name} in {self.out_directory}.')
        
        save_dir = self.out_directory.joinpath(f"lcs/{save_name}")
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        
        # for each filter
        for fltr in self.catalogs.keys():
            if os.path.isfile(os.path.join(save_dir, f'{fltr}_source_1.csv')) and not overwrite:
                print(f'[OPTICAM] Skipping {fltr} since existing light curves files were found. To overwrite these files, set overwrite=True.')
                continue
            
            source_coords = np.array([self.catalogs[fltr]["xcentroid"].value,  # type: ignore
                                      self.catalogs[fltr]["ycentroid"].value],  # type:ignore
                                     ).T
            
            file_paths = [file_path for file_path in self.camera_files[fltr] if file_path not in self.unaligned_files]
            batch_size = get_batch_size(len(file_paths))
            results = process_map(
                partial(
                    self._perform_photometry,
                    photometer=photometer,
                    source_coords=source_coords,
                    fltr=fltr,
                ),
                file_paths,
                max_workers=self.number_of_processors,
                disable=not self.verbose,
                desc=f"[OPTICAM] Performing photometry on {fltr} images",
                chunksize=batch_size,
                bar_format=bar_format,
                tqdm_class=tqdm,
            )
            
            save_photometry_results(
                results=results,
                catalogs=self.catalogs,
                barycenter=self.barycenter,
                save_dir=save_dir,
                fltr=fltr
            )
        
        plot_rms_vs_median_flux(
            lc_dir=save_dir,
            save_dir=self.out_directory.joinpath('diag'),
            phot_label=save_name,
            show=self.show_plots,
            )


    def _perform_photometry(
        self,
        file_path: Path,
        photometer: BasePhotometer,
        source_coords: NDArray,
        fltr: str,
        ) -> Dict[str, List]:
        """
        Perform photometry on a file.
        
        Parameters
        ----------
        file_path : Path
            The file path.
        photometer : BasePhotometer
            The photometer to use.
        source_coords : NDArray
            The coordinates of the sources.
        fltr : str
            The image filter.
        
        Returns
        -------
        Dict[str, List]
            The photometry results.
        """
        
        image, bias_var, dark_var, flat_var = get_data(
            file_path=file_path,
            instrument=self.instrument,
            bias_corrector=self.bias_corrector,
            dark_corrector=self.dark_corrector,
            flat_corrector=self.flat_corrector,
            rebin_factor=self.rebin_factor,
            remove_cosmic_rays=self.remove_cosmic_rays,
            )
        
        if photometer.local_background_estimator is None:
            bkg = self.background(image)  # get 2D background
            image -= bkg.background  # remove background from image
            threshold = self.threshold * bkg.background_rms  # define source detection threshold
            background_rms = bkg.background_rms.copy()
        else:
            # estimate source detection threshold from noisy image
            threshold = detect_threshold(image, self.threshold)
            background_rms = None
        
        image_coords = None  # assume no image coordinates by default
        if not photometer.forced:
            try:
                tbl = self.finder(image, threshold)
                image_coords = np.array([tbl["xcentroid"].value,
                                        tbl["ycentroid"].value],
                                        ).T
            except Exception as e:
                self.logger.warning(f"[OPTICAM] Could not determine source coordinates in {file_path}: {e}")
        
        results = photometer.compute(
            image=image,
            bias_var=bias_var,
            dark_var=dark_var,
            flat_var=flat_var,
            background_rms=background_rms,
            source_coords=source_coords,
            image_coords=image_coords,
            psf_params=self.psf_params[fltr],
            read_noise=self.instrument.read_noise,
            )
        
        # add time stamp
        if self.barycenter:
            results['BMJD'] = self.bmjds[file_path]
        else:
            results['MJD'] = self.bmjds[file_path]
        
        return results


    def update_unaligned_files(
        self,
        file_paths: Path | List[Path],
        ) -> None:
        """
        Add one or more files to the list of unaligned files. Unaligned files are skipped when performing photometry.
        
        Parameters
        ----------
        files : Path | List[Path]
            The file or files.
        """
        
        if isinstance(file_paths, Path):
            file_paths = [file_paths]
        
        for file_path in file_paths:
            if os.path.isfile(file_path):
                self.unaligned_files.append(file_path)
            elif os.path.isfile(self.data_directory.joinpath(file_path)):
                self.unaligned_files.append(self.data_directory.joinpath(file_path))
            else:
                raise ValueError(f'[OPTICAM] file {file_path} could not be found.')
        
        save_unaligned_files(
            out_directory=self.out_directory,
            unaligned_files=self.unaligned_files,
        )



################### for a clearner UI, the following functions are intentionally not Reducer methods ###################




def log_reducer_params(
    reducer: Reducer,
    ) -> None:
    """
    Log the input parameters of a `Reducer` instance to file.
    
    Parameters
    ----------
    reducer : Reducer
        The `Reducer` instance.
    """
    
    # get parameters
    params = dict(recursive_log(reducer, max_depth=5))
    
    params.update({'filters': list(reducer.camera_files.keys())})
    
    # remove some parameters that are either already saved elsewhere or are not relevant
    params.pop('logger')
    params.pop('bmjds')
    params.pop('camera_files')
    
    try:
        params.pop('transforms')
    except KeyError:
        pass
    
    try:
        params.pop('unaligned_files')
    except KeyError:
        pass
    
    try:
        params.pop('catalogs')
    except KeyError:
        pass
    
    # sort parameters
    params = dict(sorted(params.items()))
    
    # write parameters to file
    with open(os.path.join(reducer.out_directory, "misc/reduction_parameters.json"), "w") as file:
        json.dump(params, file, indent=4)


def log_dark_current(
    out_directory: str,
    dark_currs: Dict[str, float],
    bmjds: Dict[str, float],
    camera_files: Dict[str, List[str]],
    ) -> None:
    """
    Save the dark currents for each filter.
    
    Parameters
    ----------
    out_directory : str
        The path to the output directory.
    dark_currs : Dict[str, float]
        The dark current for each file {file: dark current}.
    bmjds : Dict[str, float]
        The time stamp for each file {file: time stamp}.
    camera_files : Dict[str, List[str]]
        The files grouped by filter {filter: files}.
    """
    
    dark_curr_df = pd.DataFrame(dark_currs.items(), columns=['file', 'dark_current'])
    bmjds_df = pd.DataFrame(bmjds.items(), columns=['file', 'BMJD'])
    df = pd.merge(dark_curr_df, bmjds_df, on='file')
    df = df[['BMJD', 'dark_current', 'file']]  # change column order
    
    for fltr, files in camera_files.items():
        filter_df = df[df['file'].isin(files)]
        filter_df = filter_df.drop(columns='file')
        filter_df.to_csv(os.path.join(out_directory, f'diag/{fltr}_dark_current.csv'), index=False)


def set_psf_params(
    aperture_selector: Callable,
    catalog: QTable,
    ) -> Dict[str, float]:
    """
    Set the PSF parameters.
    
    Parameters
    ----------
    aperture_selector : Callable
        The aperture selector (e.g., `numpy.median`).
    catalog : QTable
        The source catalog.
    
    Returns
    -------
    Dict[str, float]
        The PSF parameters.
    """
    
    semimajor_sigma_pix = aperture_selector(catalog['semimajor_sigma'].value)  # type: ignore
    semiminor_sigma_pix = aperture_selector(catalog['semiminor_sigma'].value)  # type: ignore
    orientation = aperture_selector(catalog['orientation'].value)  # type: ignore
    
    return {
        'semimajor_sigma': semimajor_sigma_pix,
        'semiminor_sigma': semiminor_sigma_pix,
        'orientation': orientation,
    }


def parse_alignment_results(
    results: Tuple,
    camera_files: List[Path],
    transforms: Dict[Path, List[float]],
    unaligned_files: List[Path],
    verbose: bool,
    ) -> Tuple[Dict[Path, List[float]], List[Path], NDArray[np.float64], Dict[Path, Dict[str, float]]]:
    """
    Parse the alignment results.
    
    Parameters
    ----------
    results : Tuple
        The alignment results.
    camera_files : List[Path]
        The file paths for all files. 
    transforms : Dict[Path, List[float]]
        The image-to-image alignments {file path: transform}.
    unaligned_files : List[Path]
        The paths of the files that could not be aligned.
    verbose : bool
        Whether to include output.
    
    Returns
    -------
    Tuple[Dict[str, List[float]], List[str], NDArray, Dict[str, float], Dict[str, float]]
        The updated transforms, unaligned files, stacked image, median background values and median background RMS
        values.
    """
    
    fltr_transforms: Dict[Path, List[float]] = {}
    fltr_unaligned_files: List[Path] = []
    fltr_background: Dict[Path, Dict[str, float]] = {}
    
    # unpack results
    batch_stacked_images, batch_transforms, batch_backgrounds = zip(*results)
    
    # combine results
    for i in range(len(batch_stacked_images)):
        fltr_transforms.update(batch_transforms[i])
        fltr_background.update(batch_backgrounds[i])
    
    aligned_files = list(fltr_transforms.keys())
    for file in camera_files:
        if file not in aligned_files:
            fltr_unaligned_files.append(file)
    
    stacked_image = np.sum(batch_stacked_images, axis=0)  # stack images
    
    transforms.update(fltr_transforms)  # update transforms to include current filter
    unaligned_files += fltr_unaligned_files  # update unaligned files
    
    if verbose:
        print(f"[OPTICAM] Done.")
        print(f'[OPTICAM] {len(fltr_transforms)} image(s) aligned.')
        print(f'[OPTICAM] {len(fltr_unaligned_files)} image(s) could not be aligned.')
    
    return transforms, unaligned_files, stacked_image, fltr_background


def save_background(
    out_directory: Path,
    background: Dict[Path, Dict[str, float]],
    fltr: str,
    bmjds: Dict[Path, float],
    ) -> None:
    """
    Save the median background and its RMS to a CSV file.
    
    Parameters
    ----------
    out_directory : Path
        The output directory.
    background : Dict[Path, Dict[str, float]]
        The background values for each file.
    fltr : str
        The corresponding filter.
    bmjds : Dict[Path, float]
        The BMJD values for each file {file path}.
    """
    
    df = pd.DataFrame.from_dict(background, orient='index').reset_index()
    df.columns = ['file', 'median', 'rms']
    
    time_df = pd.DataFrame.from_dict(bmjds, orient='index').reset_index()
    time_df.columns = ['file', 'BMJD']
    
    merged_df = pd.merge(df, time_df, on='file', how='inner')  # merge dataframes to get corresponding times
    merged_df = merged_df.drop('file', axis=1)  # delete file column
    merged_df = merged_df.reindex(columns=['BMJD', 'median', 'rms'])  # reorder columns
    
    merged_df.to_csv(os.path.join(out_directory, f'diag/{fltr}_background.csv'), index=False)


def save_unaligned_files(
    out_directory: Path,
    unaligned_files: List[Path],
    ) -> None:
    """
    Save the unaligned files to a text file.
    
    Parameters
    ----------
    out_directory : Path
        The output directory.
    unaligned_files : List[Path]
        The list of unaligned files.
    """
    
    if len(unaligned_files) > 0:
        with open(os.path.join(out_directory, "diag/unaligned_files.txt"), "w") as unaligned_file:
            for file in unaligned_files:
                unaligned_file.write(str(file) + "\n")


def get_random_image_for_each_filter(
    camera_files: Dict[str, List[Path]],
    instrument: Instrument,
    ) -> Dict[str, NDArray]:
    """
    Choose a random image for each filter from a dictionary.
    
    Parameters
    ----------
    camera_files : Dict[str, List[Path]]
        The filters and corresponding files in the data directory {filter: [paths to images]}.
    instrument : Instrument
        The instrument.
    
    Returns
    -------
    Dict[str, NDArray]
        A dictionary containing a random file for each filter
    """
    
    rng = np.random.default_rng()
    images = {}
    
    for file_paths in camera_files.values():
        file_path = file_paths[rng.choice(len(file_paths))]  # choose a random file
        file_name = str(file_path).split('/')[-1]  # get file name (final part of the file path)
        images[file_name] = get_data(
            file_path=Path(file_path),
            instrument=instrument,
            rebin_factor=1,
            remove_cosmic_rays=False,
            )[0]
    
    return images


def create_targets_dict(
    catalogs: Dict[str, QTable],
    ) -> Dict[str, List[int]]:
    """
    Create a dictionary of target IDs for all catalog sources.
    
    Parameters
    ----------
    catalogs : Dict[str, QTable]
        The catalogs.
    
    Returns
    -------
    Dict[str, List[int]]
        The target IDs for all catalog sources.
    """
    
    targets: Dict[str, List[int]] = {}
    
    for fltr, cat in catalogs.items():
        targets[fltr] = []
        for i in range(len(cat)):
            targets[fltr].append(i + 1)
    
    return targets


def save_photometry_results(
    results: Tuple[Dict],
    catalogs: Dict[str, QTable],
    barycenter: bool,
    save_dir: Path,
    fltr: str,
    ):
    """
    Save the photometry results to disk.
    
    Parameters
    ----------
    results : Tuple[Dict]
        The photometry results.
    catalogs : Dict[str, QTable]
        The source catalogs.
    save_dir : Path
        The save directory path.
    fltr : str
        The photometry filter.
    """
    
    photometry_results = parse_photometry_results(results)
    
    time_key = 'BMJD' if barycenter else 'MJD'
    
    # for each source in the catalog
    for i in range(len(catalogs[fltr])):
        
        # unpack results for ith source
        source_results = {}
        for key, values in photometry_results.items():
            
            # time is a special case since it is already a single column
            if key == time_key:
                source_results[key] = np.asarray(values)
            # for other keys, the ith column needs to be extracted
            else:
                col = [value[i] for value in values]
                source_results[key] = np.asarray(col)
        
        # define light curve as a DataFrame
        df = pd.DataFrame(source_results)
        
        # drop NaNs
        df.dropna(inplace=True, ignore_index=True)
        
        # make time column left-most column
        time_col = df.pop(time_key)
        df.insert(0, time_key, time_col)
        
        # save to file
        df.to_csv(
            os.path.join(
                save_dir,
                f'{fltr}_source_{i + 1}.csv',
                ),
            index=False,
            )

def parse_photometry_results(
    results: Tuple[Dict[str, List]],
    ) -> Dict[str, List[List[float]]]:
    """
    Merge the multiprocessed photometry results into a single dictionary.
    
    Parameters
    ----------
    results : Tuple[Dict[str, List]]
        The multiprocessed photometry results.
    
    Returns
    -------
    Dict[str, List[List[float]]]
        The photometry results in a single dictionary.
    """
    
    photometry_results = {}
    for result in results:
        for key, value in result.items():
            if key not in photometry_results:
                photometry_results[key] = []
            photometry_results[key].append(value)
    
    return photometry_results

