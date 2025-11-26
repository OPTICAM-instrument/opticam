from functools import partial
import json
import logging
from multiprocessing import cpu_count
import os
from typing import Callable, Dict, List, Literal, Tuple

from astropy.table import QTable
from matplotlib import pyplot as plt
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from photutils.segmentation import detect_threshold
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm

from opticam.align import align_batch
from opticam.background.global_background import BaseBackground, DefaultBackground
from opticam.correctors.flat_field_corrector import FlatFieldCorrector
from opticam.finders import DefaultFinder, get_source_coords_from_image
from opticam.photometers import AperturePhotometer, BasePhotometer, perform_photometry
from opticam.utils.batching import get_batches, get_batch_size
from opticam.utils.constants import bar_format
from opticam.utils.data_checks import check_data
from opticam.plotting.gifs import compile_gif, create_gif_frame
from opticam.utils.fits_handlers import get_data, get_stacked_images, save_stacked_images
from opticam.utils.logging import recursive_log, log_psf_params
from opticam.plotting.plots import plot_backgrounds, plot_background_meshes, plot_catalogs, plot_growth_curves, \
    plot_time_between_files, plot_psf, plot_rms_vs_median_flux, plot_noise, plot_snrs, plot_apertures


class Reducer:
    """
    Class for reducing OPTICAM data.
    """

    def __init__(
        self,
        out_directory: str,
        data_directory: None | str = None,
        c1_directory: None | str = None,
        c2_directory: None | str = None,
        c3_directory: None | str = None,
        rebin_factor: int = 1,
        flat_corrector: None | FlatFieldCorrector = None,
        background: BaseBackground | None = None,
        finder: None | Callable = None,
        threshold: float = 5,
        aperture_selector: Callable = np.median,
        remove_cosmic_rays: bool = False,
        barycenter: bool = True,
        number_of_processors: int = cpu_count() // 2,
        show_plots: bool = True,
        verbose: bool = True
        ) -> None:
        """
        Class for reducing OPTICAM data.
        
        Parameters
        ----------
        out_directory: str
            The path to the directory to save the output files.
        data_directory: str, optional
            The path to the directory containing the data, by default `None`. If `None`, any of c1_directory, c2_directory,
            or c3_directory must be defined. If data_directory is defined, c1_directory, c2_directory, and c3_directory
            are ignored.
        c1_directory: str, optional
            The path to the directory containing the C1 data, by default `None`. If `None`, any of data_directory,
            c2_directory, or c3_directory must be defined. This parameter is ignored if data_directory is defined.
        c2_directory: str, optional
            The path to the directory containing the C2 data, by default `None`. If `None`, any of data_directory,
            c1_directory, or c3_directory must be defined. This parameter is ignored if data_directory is defined.
        c3_directory: str, optional
            The path to the directory containing the C3 data, by default `None`. If `None`, any of data_directory,
            c1_directory, or c2_directory must be defined. This parameter is ignored if data_directory is defined.
        rebin_factor: int, optional
            The rebinning factor, by default 1 (no rebinning). The rebinning factor is the factor by which the image is
            rebinned in both dimensions. Rebinning can improve the detectability of faint sources and speed up
            some operations (like cosmic ray removal) at the cost of image resolution.
        flat_corrector: FlatFieldCorrector, optional,
            The flat-field corrector, by default `None`. If `None`, no flat-field corrections are applied.
        threshold: float, optional
            The signal-to-noise ratio threshold for source finding, by default 5. Reduce this value to identify fainter
            sources, though this may lead to the identification of spurious sources.
        background: BaseBackground | None, optional
            The background calculator, by default `None`. If `None`, the default background calculator is used. If a
            callable is provided, it should take an image (`NDArray`) as input and return a `Background2D` object.
        finder: Callable, optional
            The source finder, by default `None`. If `None`, the default source finder is used. If a callable is
            provided, it should take an image (`NDArray`) and a threshold (`float | NDArray`) as input and return a
            `QTable` instance.
        aperture_selector: Callable, optional
            The aperture selector, by default `np.median`. This function is used to select the aperture size for
            photometry. If a callable is provided, it should take a list of source sizes (`List[float]`) as input and
            return a single value.
        remove_cosmic_rays: bool, optional
            Whether to remove cosmic rays from images, by default False. Cosmic rays are removed using the LACosmic
            algorithm as implemented in `astroscrappy`. Note: this can be computationally expensive, particularly for
            large images.
        barycenter: bool, optional
            Whether to apply a barycentric correction to the image time stamps, by default True.
        number_of_processors: int, optional
            The number of processors to use for parallel processing, by default half the number of available processors.
        show_plots: bool, optional
            Whether to show plots as they're created, by default `True`. Whether `True` or `False`, plots are always
            saved to `out_directory`.
        verbose: bool, optional
            Whether to print verbose output, by default `True`.
        """
        
        self.verbose = verbose
        
        ########################################### out_directory ###########################################
        
        self.out_directory = out_directory
        
        # create output directory if it does not exist
        if not os.path.isdir(self.out_directory):
            if self.verbose:
                print(f"[OPTICAM] {self.out_directory} not found, attempting to create ...")
            # create output directory if it does not exist
            try:
                os.makedirs(self.out_directory)
            except:
                raise FileNotFoundError(f"[OPTICAM] Could not create directory {self.out_directory}")
            if self.verbose:
                print(f"[OPTICAM] {self.out_directory} created.")
        
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
        if not os.path.isdir(os.path.join(self.out_directory, "cat")):
            os.makedirs(os.path.join(self.out_directory, "cat"))
        if not os.path.isdir(os.path.join(self.out_directory, "diag")):
            os.makedirs(os.path.join(self.out_directory, "diag"))
        if not os.path.isdir(os.path.join(self.out_directory, "misc")):
            os.makedirs(os.path.join(self.out_directory, "misc"))
        
        ########################################### data directories ###########################################
        
        self.data_directory = data_directory
        self.c1_directory = c1_directory
        self.c2_directory = c2_directory
        self.c3_directory = c3_directory
        
        assert self.data_directory is not None or self.c1_directory is not None or self.c2_directory is not None or self.c3_directory is not None, "[OPTICAM] At least one of data_directory, c1_directory, c2_directory, or c3_directory must be defined."
        
        ########################################### input params ###########################################
        
        # set some useful parameters
        self.rebin_factor = rebin_factor
        self.flat_corrector = flat_corrector
        self.aperture_selector = aperture_selector
        self.threshold = threshold
        self.remove_cosmic_rays = remove_cosmic_rays
        self.barycenter = barycenter
        self.number_of_processors = number_of_processors
        self.show_plots = show_plots
        
        ########################################### check input data ###########################################
        
        self.camera_files, self.binning_scale, self.bmjds, self.ignored_files, self.gains, dark_currs, self.t_ref = check_data(
                data_directory=data_directory,
                c1_directory=c1_directory,
                c2_directory=c2_directory,
                c3_directory=c3_directory,
                out_directory=out_directory,
                barycenter=self.barycenter,
                verbose=verbose,
                return_output=True,
                logger=self.logger,
                number_of_processors=number_of_processors,
                )  # type: ignore
        
        ########################################### plot time between files ###########################################
        
        plot_time_between_files(
            out_directory=self.out_directory,
            camera_files=self.camera_files,
            bmjds=self.bmjds,
            show=self.show_plots,
            save=True,
            )
        
        ########################################### save dark currents ###########################################
        
        log_dark_current(self.out_directory, dark_currs, self.bmjds, self.camera_files)
        
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
        
        instance_align_batch = partial(
            align_batch,
            gains=self.gains,
            flat_corrector=self.flat_corrector,
            rebin_factor=self.rebin_factor,
            remove_cosmic_rays=self.remove_cosmic_rays,
            background=self.background,
            finder=self.finder,
            threshold=self.threshold,
            logger=self.logger,
        )
        
        # for each camera
        for fltr in self.camera_files.keys():
            
            # if no images found for camera, skip
            if len(self.camera_files[fltr]) == 0:
                continue
            
            # get reference image
            reference_image = get_data(
                file=self.reference_files[fltr],
                flat_corrector=self.flat_corrector,
                rebin_factor=self.rebin_factor,
                remove_cosmic_rays=self.remove_cosmic_rays,
                )[0]
            
            try:
                # get source coordinates in descending order of brightness
                reference_coords = get_source_coords_from_image(
                    reference_image,
                    finder=self.finder,
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
                    instance_align_batch,
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
            self.catalogs.update({fltr: tbl})
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
        
        # write unaligned files to file
        if len(self.unaligned_files) > 0 and (not os.path.isfile(os.path.join(self.out_directory, "diag/unaligned_files.txt")) or overwrite):
            with open(os.path.join(self.out_directory, "diag/unaligned_files.txt"), "w") as unaligned_file:
                for file in self.unaligned_files:
                    unaligned_file.write(file + "\n")

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
            images = get_random_image_for_each_filter(self.camera_files)
            
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
                    'g-band': [1, 2, 3],
                    'r-band': [1, 2, 3],
                    'i-band': [1, 2, 3],
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
            files=self.reference_files,
            background=self.background,
            psf_params=self.psf_params,
            catalogs=self.catalogs,
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
            files=self.reference_files,
            background=self.background,
            psf_params=self.psf_params,
            catalogs=self.catalogs,
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
        saved in out_directory/diag/*-band_gif_frames and the GIFs are saved in out_directory/cat.
        
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
                    gains=self.gains,
                    transforms=self.transforms,
                    reference_file=self.reference_files[fltr],
                    flat_corrector=self.flat_corrector,
                    rebin_factor=self.rebin_factor,
                    remove_cosmic_rays=self.remove_cosmic_rays,
                    background=self.background,
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
                verbose=self.verbose,
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
                    'g-band': [1, 2, 3],
                    'r-band': [1, 2, 3],
                    'i-band': [1, 2, 3],
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
                self.reference_files[fltr],
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
        
        save_dir = os.path.join(self.out_directory, f"lcs/{save_name}")
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        
        # for each filter
        for fltr in self.catalogs.keys():
            if os.path.isfile(os.path.join(save_dir, f'{fltr}_source_1.csv')) and not overwrite:
                print(f'[OPTICAM] Skipping {fltr} since existing light curves files were found. To overwrite these files, set overwrite=True.')
                continue
            
            source_coords = np.array([self.catalogs[fltr]["xcentroid"].value,
                                      self.catalogs[fltr]["ycentroid"].value]).T
            
            files = [file for file in self.camera_files[fltr] if file not in self.unaligned_files]
            batch_size = get_batch_size(len(files))
            results = process_map(
                partial(
                    perform_photometry,
                    photometer=photometer,
                    source_coords=source_coords,
                    gains=self.gains,
                    bmjds=self.bmjds,
                    barycenter=self.barycenter,
                    flat_corrector=self.flat_corrector,
                    rebin_factor=self.rebin_factor,
                    remove_cosmic_rays=self.remove_cosmic_rays,
                    background=self.background,
                    threshold=self.threshold,
                    finder=self.finder,
                    psf_params=self.psf_params,
                    fltr=fltr,
                    logger=self.logger,
                ),
                files,
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
            save_dir=os.path.join(self.out_directory, 'diag'),
            phot_label=save_name,
            show=self.show_plots,
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
    params.pop('gains')
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
    
    semimajor_sigma_pix = aperture_selector(catalog['semimajor_sigma'].value)
    semiminor_sigma_pix = aperture_selector(catalog['semiminor_sigma'].value)
    orientation = aperture_selector(catalog['orientation'].value)
    
    return {
        'semimajor_sigma': semimajor_sigma_pix,
        'semiminor_sigma': semiminor_sigma_pix,
        'orientation': orientation,
    }


def parse_alignment_results(
    results: Tuple,
    camera_files: List[str],
    transforms: Dict[str, List[float]],
    unaligned_files: List[str],
    verbose: bool,
    ) -> Tuple[Dict[str, List[float]], List[str], NDArray, Dict[str, Dict[str, float]]]:
    """
    Parse the alignment results.
    
    Parameters
    ----------
    results : Tuple
        The alignment results.
    camera_files : List[str]
        The file paths for all files. 
    transforms : Dict[str, List[float]]
        The image-to-image alignments {file path: transform}.
    unaligned_files : List[str]
        The paths of the files that could not be aligned.
    verbose : bool
        Whether to include output.
    
    Returns
    -------
    Tuple[Dict[str, List[float]], List[str], NDArray, Dict[str, float], Dict[str, float]]
        The updated transforms, unaligned files, stacked image, median background values and median background RMS
        values.
    """
    
    fltr_transforms = {}
    fltr_unaligned_files = []
    fltr_background = {}
    
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
    out_directory: str,
    background: Dict[str, Dict[str, float]],
    fltr: str,
    bmjds: Dict[str, float],
    ) -> None:
    """
    Save the median background and its RMS to a CSV file.
    
    Parameters
    ----------
    out_directory : str
        The output directory.
    background : Dict[str, Dict[str, float]]
        The background values for each file.
    fltr : str
        The corresponding filter.
    bmjds : Dict[str, float]
        The BMJD values for each file.
    """
    
    df = pd.DataFrame.from_dict(background, orient='index').reset_index()
    df.columns = ['file', 'median', 'rms']
    
    time_df = pd.DataFrame.from_dict(bmjds, orient='index').reset_index()
    time_df.columns = ['file', 'BMJD']
    
    merged_df = pd.merge(df, time_df, on='file', how='inner')  # merge dataframes to get corresponding times
    merged_df = merged_df.drop('file', axis=1)  # delete file column
    merged_df = merged_df.reindex(columns=['BMJD', 'median', 'rms'])  # reorder columns
    
    merged_df.to_csv(os.path.join(out_directory, f'diag/{fltr}_background.csv'), index=False)


def get_random_image_for_each_filter(
    camera_files: Dict[str, List[str]],
    ) -> Dict[str, NDArray]:
    """
    Choose a random image for each filter from a dictionary.
    
    Parameters
    ----------
    camera_files : Dict[str, List[str]]
        The filters and corresponding files in the data directory.
    
    Returns
    -------
    Dict[str, NDArray]
        A dictionary containing a random file for each filter
    """
    
    rng = np.random.default_rng()
    images = {}
    
    for files in camera_files.values():
        file = files[rng.choice(len(files))]  # choose a random file
        file_name = file.split('/')[-1]  # get file name (final part of the file path)
        images[file_name] = get_data(
            file=file,
            flat_corrector=None,
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
    save_dir: str,
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
    save_dir : str
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





