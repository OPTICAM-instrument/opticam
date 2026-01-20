from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Tuple


from astropy.io import fits
import numpy as np
from numpy.typing import NDArray


from opticam.instruments import Instrument, OPTICAM_MX
from opticam.utils.helpers import create_file_paths
from opticam.utils.image_helpers import rebin_image
from opticam.utils.logging import log_file




class Corrector(ABC):
    """
    Base class for correctors.
    """


    def __init__(
        self,
        out_directory: Path | str | None = None,
        data_directory: Path | str | None = None,
        instrument: Instrument = OPTICAM_MX(),
        rebin_factor: int = 1,
        ) -> None:
        """
        Initialise the corrector.
        
        Parameters
        ----------
        out_directory : Path | str | None, optional
            The path to the output directory, by default `None`. Master correction images will be saved here.
        data_directory : Path | str | None, optional
            The path to the data directory, by default `None`. This must point to a directory containing a series of
            FITS files.
        instrument : Instrument, optional
            The instrument, by default `OPTICAM_MX()`.
        rebin_factor : int, optional
            The factor by which to rebin the data, by default 1. Useful if, for example, calibration images were taken
            in a lower binning mode than the observations.
        """
        
        self.out_directory = Path(out_directory) if out_directory is not None else None
        self.instrument = instrument
        
        assert isinstance(rebin_factor, int), "[OPTICAM] Non-integer rebin factors are not supported!"
        self.rebin_factor = rebin_factor
        
        if self.out_directory is not None:
            if not self.out_directory.is_dir():
                self.out_directory.mkdir(parents=True)
        
        # get the paths to the calibration images
        if data_directory is not None:
            raw_data_paths = create_file_paths(data_directory=Path(data_directory))
            self.data_paths = self._validate_data(raw_data_paths)
        else:
            self.data_paths = None
        
        self.master_images: Dict[str, NDArray[np.float64]] = {}
        self.master_variances: Dict[str, NDArray[np.float64]] = {}
        
        # load master images if they already exist
        if self.master_image_path is not None:
            if self.master_image_path.is_file():
                self._read_master_image()


    @property
    @abstractmethod
    def master_image_path(self) -> Path | None:
        """
        The path to the master calibration image.
        
        Returns
        -------
        Path | None
            The path to the master calibration image.
        """
        pass


    @abstractmethod
    def correct(
        self,
        image: NDArray[np.float64],
        fltr: str,
        *args,
        **kwargs,
        ) -> Tuple[NDArray[np.float64], float | NDArray[np.float64]]:
        """
        Apply the required correction to an image.
        
        Parameters
        ----------
        image : NDArray[np.float64]
            The image.
        fltr : str
            The image filter.
        
        Returns
        -------
        Tuple[NDArray[np.float64], float | NDArray[np.float64]]
            The corrected image and the variance of the correction term. The variance may be a `float` (e.g., if
            the dark noise is calculated from the exposure-integrated dark current) or an `NDArray`.
        """
        
        pass


    @abstractmethod
    def create_master_images(
        self,
        overwrite: bool = False,
        *args,
        **kwargs,
        ) -> None:
        """
        Create the master calibration image(s).
        
        Parameters
        ----------
        bias_corrector : BiasCorrector | None, optional
            The bias corrector, by default `None` (no bias corrections).
        overwrite : bool, optional
            Whether to overwrite any existing master calibration images, by default `False`.
        """
        
        pass


    def _read_master_image(
        self,
        ) -> None:
        """
        Read the master images from the output directory.
        """
        
        with fits.open(self.master_image_path) as hdul:
            # skip primary HDU since it doesn't contain any data
            for hdu in hdul[1:]:
                # filter info is saved using instrument format so it's safe to use the filter keyword instead of the
                # instrument.get_filter() method
                fltr = hdu.header[self.instrument.filter_kw]
                name = hdu.header['EXTNAME']
                data = np.asarray(hdu.data, dtype=np.float64)
                if name == 'DATA':
                    self.master_images[fltr] = data
                elif name == 'VARIANCE':
                    self.master_variances[fltr] = data


    @abstractmethod
    def run_checks(
        self,
        data_file_paths_by_filter: Dict[str, List[Path]],
        return_errors: bool = False,
        ) -> None | int:
        """
        Run a series of checks on the corrector to ensure that it is compatible with the data.
        
        Parameters
        ----------
        data_file_paths_by_filter : Dict[str, List[Path]]
            The paths to all of the science images grouped by filter {filter: list of paths}.
        return_errors : bool, optional
            Whether to return the number of errors raised, by default `False`.
        
        Returns
        -------
        None | int
            If `return_errors=True`, the number of errors raised is returned.
        """
        
        pass


    def _save_master_image(
        self,
        overwrite: bool,
        ) -> None:
        """
        Save the master images and their corresponding variances to a compressed FITS cube.
        
        Parameters
        ----------
        overwrite : bool
            Whether to overwrite an existing master images file.
        """
        
        if self.__class__.__name__ == 'BiasCorrector':
            comment = 'This FITS file contains master bias images and their corresponding variances for each filter.'
        elif self.__class__.__name__ == 'DarkNoiseCorrector':
            comment = 'This FITS file contains master dark images and their corresponding variances for each filter.'
        elif self.__class__.__name__ == 'FlatFieldCorrector':
            comment = 'This FITS file contains master flat-field images and their corresponding variances for each filter.'
        else:
            raise ValueError(f'[OPTICAM] Unrecognised corrector: {self.__class__.__name__}.')
        
        hdr = fits.Header()
        hdr['COMMENT'] = comment
        empty_primary = fits.PrimaryHDU(header=hdr)
        hdul = fits.HDUList([empty_primary])
        
        for fltr in self.master_images:
            # master image
            hdr = fits.Header()
            # filter is already in instrument format so no need to use instrument.get_filter()
            hdr[self.instrument.filter_kw] = fltr
            hdu = fits.ImageHDU(
                data=self.master_images[fltr],
                header=hdr,
                name='DATA',
                )
            hdul.append(hdu)
            
            # master variance
            hdr = fits.Header()
            # filter is already in instrument format so no need to use instrument.get_filter()
            hdr[self.instrument.filter_kw] = fltr
            hdu = fits.ImageHDU(
                data=self.master_variances[fltr],
                header=hdr,
                name='VARIANCE',
                )
            hdul.append(hdu)
        
        if not self.master_image_path.is_file() or overwrite:
            hdul.writeto(self.master_image_path, overwrite=overwrite)


    @abstractmethod
    def _validate_data(
        self,
        file_paths: List[Path],
        ) -> Dict[str, List[Path]]:
        """
        Validate the input data.
        
        Parameters
        ----------
        file_paths : List[Path]
            The file paths to the input data.
        
        Returns
        -------
        Dict[str, List[Path]]
            The file paths to the input data separated by filter.
        """
        
        pass




class BiasCorrector(Corrector):
    """
    Helper clsas for performing bias corrections.
    """


    @property
    def master_image_path(self) -> Path | None:
        """
        The path to the master calibration image.
        
        Returns
        -------
        Path
            The path to the master calibration image.
        """
        
        return self.out_directory.joinpath('master_bias.fits.gz') if self.out_directory is not None else None


    def correct(
        self,
        image: NDArray[np.float64],
        fltr: str,
        ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Subtract the bias from an image.
        
        Parameters
        ----------
        image : NDArray[np.float64]
            The image.
        fltr : str
            The image filter.
        
        Returns
        -------
        Tuple[NDArray[np.float64], NDArray[np.float64]]
            The corrected image and the variance of the master bias image.
        
        Raises
        ------
        ValueError
            If no bias images were found with the given filter.
        """
        
        if fltr not in self.data_paths.keys():
            raise ValueError(f"[OPTICAM] No bias images found for {fltr} filter.")
        if fltr not in self.master_images.keys() or self.master_images[fltr] is None:
            print(f'[OPTICAM] {fltr} master bias image not found. Attempting to create.')
            self.create_master_images()
        
        return image - self.master_images[fltr], self.master_variances[fltr]


    def create_master_images(
        self,
        overwrite: bool = False,
        ) -> None:
        """
        Create master bias images for each filter.
        
        Parameters
        ----------
        overwrite : bool, optional
            Whether to overwrite the existing master bias image, by default `False`.
        """
        
        if self.master_image_path.is_file() and not overwrite:
            print(f'[OPTICAM] Master bias file already exists. To overwrite, set overwrite=True.')
            return
        
        for fltr in self.data_paths.keys():
            
            if len(self.data_paths[fltr]) == 1:
                raise Exception(f"[OPTICAM] Only one {fltr} bias image found. Master bias images cannot be created from a single image.")
            
            biases = []
            for bias_path in self.data_paths[fltr]:
                with fits.open(bias_path) as hdul:
                    bias = np.array(hdul[0].data, dtype=np.float64)
                
                if self.rebin_factor > 1:
                    bias = rebin_image(bias, self.rebin_factor)
                
                biases.append(bias)
            
            # use mean since bias frames shouldn't contain outliers like cosmic rays
            self.master_images[fltr] = np.mean(biases, axis=0)
            self.master_variances[fltr] = np.var(biases, axis=0, ddof=1) / len(biases)
        
        print('[OPTICAM] Master bias image(s) created.')
        
        self._save_master_image(overwrite=overwrite)
        
        print(f'[OPTICAM] Master bias image(s) saved to {self.master_image_path}.')


    def run_checks(
        self,
        data_file_paths_by_filter: Dict[str, List[Path]],
        return_errors: bool = False,
        ) -> None | int:
        """
        Run a series of checks on the corrector to ensure that it is compatible with the data. In this case, check the
        binning of the science images matches those of the bias images (ignoring self.rebin_factor) and that there are
        no missing filters.
        
        Parameters
        ----------
        data_file_paths_by_filter : Dict[str, List[Path]]
            The paths to all of the science images grouped by filter {filter: list of paths}.
        return_errors : bool, optional
            Whether to return the number of errors raised, by default `False`.
        
        Returns
        -------
        None | int
            If `return_errors=True`, the number of errors raised is returned. Otherwise, nothing is returned.
        """
        
        bias_path = next(iter(self.data_paths.values()))[0]  # get the path to a random flat
        bias_header = fits.getheader(bias_path)
        image_path = next(iter(data_file_paths_by_filter.values()))[0]  # get the path to a science image
        science_header = fits.getheader(image_path)
        
        errors = 0
        warnings = 0
        
        #################################################### errors ####################################################
        
        # check filters match
        if self.data_paths.keys() != data_file_paths_by_filter.keys():
            errors += 1
            print(f'[OPTICAM] ERROR: inconsistent filters found between the bias images and the science images. Bias image filters: ({','.join(self.data_paths.keys())}); science image filters: ({','.join(data_file_paths_by_filter.keys())})')
        
        ################################################### warnings ###################################################
        
        # check binnings match
        bias_binning = self.instrument.get_binning(header=bias_header)
        science_binning = self.instrument.get_binning(header=science_header)
        if bias_binning != science_binning:
            warnings += 1
            print(f'[OPTICAM] WARNING: inconsistent binning found between the bias images and the science images. Bias image binning: {bias_binning}; science image binning: {science_binning}. If you have passed a suitable rebin_factor to your BiasCorrector instance, then you can safely ignore this warning.')
        
        ################################################### summary ###################################################
        
        if errors > 0 or warnings > 0:
            print()  # blank line for readibility
        
        if errors == 0:
            print(f'[OPTICAM] BiasCorrector sucessfully passed all checks.')
        else:
            if errors == 1:
                print('[OPTICAM] BiasCorrector failed 1 check.')
            else:
                print(f'[OPTICAM] BiasCorrector failed {errors} checks.')
        
        if warnings == 1:
            print('[OPTICAM] BiasCorrector triggered a warning during 1 check. Warnings may be ignored provided their caveats are satisfied.')
        elif warnings > 1:
            print(f'[OPTICAM] BiasCorrector triggered a warning during {warnings} checks. Warnings may be ignored provided their caveats are satisfied.')
        
        if return_errors:
            return errors


    def _validate_data(
        self,
        file_paths: List[Path],
        ) -> Dict[str, List[Path]]:
        """
        Ensure that the bias images in the specified directory are valid (i.e., all use the same binning and have 
        exposure times of 0 s).
        
        Parameters
        ----------
        file_paths : List[Path]
            The paths to the bias images.
        
        Returns
        -------
        Dict[str, List[Path]]
            A dictionary containing the paths to the bias images for each filter.
        """
        
        filters: Dict[Path, str] = {}
        binnings: Dict[Path, str] = {}
        exptimes: Dict[Path, float] = {}
        
        for file_path in file_paths:
            header = fits.getheader(file_path)
            filters[file_path] = self.instrument.get_filter(header=header)
            binnings[file_path] = self.instrument.get_binning(header=header)
            exptimes[file_path] = float(header[self.instrument.exptime_kw])
        
        unique_filters = set(filters.values())
        unique_binnings = set(binnings.values())
        unique_exptimes = set(exptimes.values())
        zero_second_exposures = all([exptime == 0.0 for exptime in list(unique_exptimes)])
        
        if len(unique_binnings) > 1:
            log_file(
                out_directory=self.out_directory,
                file_name='binnings.json',
                file_contents=binnings,
                )
            raise ValueError(f'[OPTICAM] Inconsistent binning detected in the bias images. Image binnings have been logged to {self.out_directory.joinpath('diag/binnings.json')}.')
        
        if len(unique_exptimes) > 1 or not zero_second_exposures:
            log_file(
                out_directory=self.out_directory,
                file_name='exptimes.json',
                file_contents=exptimes,
                )
            raise ValueError(f'[OPTICAM] Invalid exposure times detected in the bias images. Exposure times have been logged to {self.out_directory.joinpath('diag/exptimes.json')}. All bias images should have an exposure time of 0.0 s.')
        
        # get flats for each filter
        biases = {}
        for fltr in unique_filters:
            biases[fltr] = []
            for k, v in filters.items():
                if v == fltr:
                    biases[fltr].append(k)
        
        for k, v in biases.items():
            print(f'[OPTICAM] {len(v)} {k} bias images.')
        
        return biases





class DarkNoiseCorrector(Corrector):
    """
    Helper class for performing dark noise corrections.
    """


    @property
    def master_image_path(self) -> Path | None:
        """
        The path to the master calibration image.
        
        Returns
        -------
        Path
            The path to the master calibration image.
        """
        
        return self.out_directory.joinpath('master_darks.fits.gz') if self.out_directory is not None else None


    @property
    def median_dark_fluxes(self) -> Dict[str, float]:
        """
        The median dark flux for each master dark image.
        
        Returns
        -------
        Dict[str, float]
            The median dark flux for each master dark image.
        """
        
        median_dark_fluxes: Dict[str, float] = {}
        if len(self.master_images) > 0:
            for fltr in self.master_images:
                median_dark_fluxes[fltr] = float(np.median(self.master_images[fltr]))
        
        return median_dark_fluxes


    def correct(
        self,
        image: NDArray[np.float64],
        fltr: str,
        bias_corrector: BiasCorrector | None = None,
        dark_flux: float | None = None,
        ) -> Tuple[NDArray[np.float64], float | NDArray[np.float64]]:
        """
        Subtract the dark noise from an image.
        
        Parameters
        ----------
        image : NDArray[np.float64]
            The image.
        fltr : str
            The image filter.
        bias_corrector : BiasCorrector | None, optional
            The bias corrector, by default `None` (no bias corrections).
        dark_flux : float | None, optional
            The exposure-integrated dark current, by default `None`. If the instrument provides a measure of the dark
            current in the image header, this obviates the need for master darks.
        
        Returns
        -------
        NDArray[np.float64] | Tuple[NDArray[np.float64], float]
            The corrected image and the variance of the master dark image.
        
        Raises
        ------
        ValueError
            If no dark images were found with the given filter.
        """
        
        if dark_flux is None:
            if fltr not in self.data_paths.keys():
                raise ValueError(f"[OPTICAM] No dark images found for {fltr} filter.")
            if fltr not in self.master_images.keys() or self.master_images[fltr] is None:
                print(f'[OPTICAM] {fltr} master dark image not found. Attempting to create.')
                self.create_master_images(
                    bias_corrector=bias_corrector,
                    )
            
            return image - self.master_images[fltr], self.master_variances[fltr]
        else:
            return image - dark_flux, dark_flux


    def create_master_images(
        self,
        bias_corrector: BiasCorrector | None = None,
        overwrite: bool = False,
        ) -> None:
        """
        Create master dark images for each available filter.
        
        Parameters
        ----------
        overwrite : bool, optional
            Whether to overwrite any existing master dark image, by default `False`.
        """
        
        if self.master_image_path.is_file() and not overwrite:
            print(f'[OPTICAM] Master darks file already exists. To overwrite existing master darks, set overwrite=True.')
            return
        
        for fltr in self.data_paths.keys():
            
            if len(self.data_paths[fltr]) == 1:
                raise Exception(f"[OPTICAM] Only one {fltr} dark image found. Master darks cannot be created from a single image.")
            
            # read darks
            darks = []
            for dark_path in self.data_paths[fltr]:
                with fits.open(dark_path) as hdul:
                    dark = np.array(hdul[0].data, dtype=np.float64)
                
                # apply bias correction
                # TODO: check whether bias should be corrected after rebinning instead?
                if bias_corrector is not None:
                    dark, bias_var = bias_corrector.correct(
                        image=dark,
                        fltr=fltr,
                        )
                else:
                    bias_var = 0.
                
                if self.rebin_factor > 1:
                    dark = rebin_image(dark, self.rebin_factor)
                
                darks.append(dark)
            
            self.master_images[fltr] = np.median(darks, axis=0)
            self.master_variances[fltr] = np.pi / (2 * len(darks)) * (np.var(darks, axis=0, ddof=1) + bias_var)
        
        print('[OPTICAM] Master dark image(s) created.')
        
        self._save_master_image(overwrite=overwrite)
        
        print(f'[OPTICAM] Master dark image(s) saved to {self.master_image_path}.')


    def run_checks(
        self,
        data_file_paths_by_filter: Dict[str, List[Path]],
        return_errors: bool = False,
        ) -> None | int:
        """
        Run a series of checks on the corrector to ensure that it is compatible with the data. In this case, check the
        binning of the science images matches to those of the darks (neglecting self.rebin_factor), there are no missing
        filters, and the exposure times of the science images matches those of the darks.
        
        Parameters
        ----------
        data_file_paths_by_filter : Dict[str, List[Path]]
            The paths to all of the science images grouped by filter {filter: list of paths}.
        return_errors : bool, optional
            Whether to return the number of errors raised, by default `False`.
        
        Returns
        -------
        None | int
            If `return_errors=True`, the number of errors raised is returned. Otherwise, nothing is returned.
        """
        
        # get some random image headers from the dark images and the science images
        if self.data_paths is not None:
            dark_path = next(iter(self.data_paths.values()))[0]  # path to a random flat
            dark_header = fits.getheader(dark_path)
            dark_binning = self.instrument.get_binning(header=dark_header)
        image_path = next(iter(data_file_paths_by_filter.values()))[0]  # path to a science image
        science_header = fits.getheader(image_path)
        science_binning = self.instrument.get_binning(header=science_header)
        
        errors = 0
        warnings = 0
        
        #################################################### errors ####################################################
        
        # check exposure times match
        if self.data_paths is not None:
            dark_exptime = dark_header[self.instrument.exptime_kw]
            science_exptime = science_header[self.instrument.exptime_kw]
            if dark_exptime != science_exptime:
                errors += 1
                print(f'[OPTICAM] ERROR: inconsistent exposure times found between the dark images and the science images. Dark image exposure time: {dark_exptime}; science image exposure time: {science_exptime}')
        
        # check filters match
        if self.data_paths is not None:
            if self.data_paths.keys() != data_file_paths_by_filter.keys():
                errors += 1
                print(f'[OPTICAM] ERROR: inconsistent filters found between the dark images and the science images. Dark image filters: ({','.join(self.data_paths.keys())}); science image filters: ({','.join(data_file_paths_by_filter.keys())})')
        
        if self.data_paths is None:
            if self.instrument.dark_curr_kw not in list(science_header.keys()):
                errors += 1
                print(f'[OPTICAM] ERROR: No dark images passed to DarkCurrentCorrector and the dark current keyword {self.instrument.dark_curr_kw} was not found in the header of the file: {image_path}')
        
        ################################################### warnings ###################################################
        
        # check binnings match
        if self.data_paths is not None:
            if dark_binning != science_binning:
                warnings += 1
                print(f'[OPTICAM] WARNING: inconsistent binning found between the dark images and the science images. Dark image binning: {dark_binning}; science image binning: {science_binning}. If you have passed a suitable rebin_factor to your DarkNoiseCorrector instance, then you can safely ignore this warning.')
        
        ################################################### summary ###################################################
        
        if errors > 0 or warnings > 0:
            print()  # blank line for readibility
        
        if errors == 0:
            print(f'[OPTICAM] DarkNoiseCorrector sucessfully passed all checks.')
        else:
            if errors == 1:
                print('[OPTICAM] DarkNoiseCorrector failed 1 check.')
            else:
                print(f'[OPTICAM] DarkNoiseCorrector failed {errors} checks.')
        
        if warnings == 1:
            print('[OPTICAM] DarkNoiseCorrector triggered a warning during 1 check. Warnings may be ignored provided their caveats are satisfied.')
        elif warnings > 1:
            print(f'[OPTICAM] DarkNoiseCorrector triggered a warning during {warnings} checks. Warnings may be ignored provided their caveats are satisfied.')
        
        if return_errors:
            return errors


    def _validate_data(
        self,
        file_paths: List[Path],
        ) -> Dict[str, List[Path]]:
        """
        Ensure that the dark images in the specified directory are valid (i.e., all use the same binning).
        
        Parameters
        ----------
        file_paths : List[Path]
            The paths to the dark images.
        
        Returns
        -------
        Dict[str, List[Path]]
            A dictionary containing the paths to the dark images for each filter.
        """
        
        filters: Dict[Path, str] = {}
        binnings: Dict[Path, str] = {}
        exptimes: Dict[Path, float] = {}
        
        for dark_path in file_paths:
            header = fits.getheader(dark_path)
            filters[dark_path] = self.instrument.get_filter(header=header)
            binnings[dark_path] = self.instrument.get_binning(header=header)
            exptimes[dark_path] = header[self.instrument.exptime_kw]
        
        unique_filters = set(filters.values())
        unique_binnings = set(binnings.values())
        unique_exptimes = set(exptimes.values())
        
        if len(unique_binnings) > 1:
            log_file(
                out_directory=self.out_directory,
                file_name='binnings.json',
                file_contents=binnings,
                )
            raise ValueError(f'[OPTICAM] Inconsistent binning detected in the dark images. Image binnings have been logged to {self.out_directory.joinpath('diag/binnings.json')}')
        
        if len(unique_exptimes) > 1:
            log_file(
                out_directory=self.out_directory,
                file_name='exptimes.json',
                file_contents=exptimes,
                )
            raise ValueError(f'[OPTICAM] Inconsistent exposure times detected in the dark images. Exposure times have been logged to {self.out_directory.joinpath('diag/exptimes.json')}')
        
        # get dark images for each filter
        darks = {}
        for fltr in unique_filters:
            darks[fltr] = []
            for k, v in filters.items():
                if v == fltr:
                    darks[fltr].append(k)
        
        for k, v in darks.items():
            print(f'[OPTICAM] {len(v)} {k} dark images.')
        
        return darks





class FlatFieldCorrector(Corrector):
    """
    Helper class for performing flat-field corrections.
    """


    @property
    def master_image_path(self) -> Path | None:
        """
        The path to the master flat.
        
        Returns
        -------
        Path
            The path to the master flat.
        """
        
        return self.out_directory.joinpath('master_flats.fits.gz') if self.out_directory is not None else None


    def correct(
        self,
        image: NDArray[np.float64],
        fltr: str,
        bias_corrector : BiasCorrector | None = None,
        ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Correct an image for flat-fielding.
        
        Parameters
        ----------
        image : NDArray[np.float64]
            The image.
        fltr : str
            The image filter.
        
        Returns
        -------
        Tuple[NDArray[np.float64], NDArray[np.float64]]
            The corrected image and the variance of the master flat-field image scaled by the square of the calibrated
            image.
        """
        
        if fltr not in self.data_paths.keys():
            raise ValueError(f"[OPTICAM] Cannot apply flat-field corrections. No flat-field images found for filter: {fltr}.")
        
        if fltr not in self.master_images.keys():
            print(f'[OPTICAM] {fltr} master flat-field image not found. Attempting to create.')
            try:
                self.create_master_images(
                    bias_corrector=bias_corrector,
                    )
            except Exception as e:
                raise Exception(f"[OPTICAM] Could not create master flat-field image(s) due to the following exception: {e}.")
        
        calibrated_image = image / self.master_images[fltr]
        
        # propagate multiplicative variance
        var = self.master_variances[fltr] * calibrated_image**2 / self.master_images[fltr]**2
        
        # correct image for flat-fielding
        return calibrated_image, var


    def create_master_images(
        self,
        bias_corrector: BiasCorrector | None = None,
        overwrite: bool = False,
        ) -> None:
        """
        Create master flat-field images for each filter.
        
        Parameters
        ----------
        overwrite : bool, optional
            Whether to overwrite the existing master flat-field image, by default `False`.
        """
        
        if self.master_image_path.is_file() and not overwrite:
            print(f'[OPTICAM] Master flats file already exists. To overwrite existing flats, set overwrite=True.')
            return
        
        for fltr in self.data_paths.keys():
            
            if len(self.data_paths[fltr]) == 1:
                raise Exception(f"[OPTICAM] Only one {fltr} flat found. Master flats cannot be created from a single image.")
            
            # read flats
            flats = []
            for flat_path in self.data_paths[fltr]:
                with fits.open(flat_path) as hdul:
                    flat = np.array(hdul[0].data, dtype=np.float64)
                
                if bias_corrector is not None:
                    flat, bias_var = bias_corrector.correct(
                        image=flat,
                        fltr=fltr,
                        )
                else:
                    bias_var = 0.
                
                if self.rebin_factor > 1:
                    flat = rebin_image(flat, self.rebin_factor)
                
                flats.append(flat)
            
            # use median to account for outliers
            raw_master_flat = np.median(flats, axis=0)
            norm = np.median(raw_master_flat)
            
            self.master_images[fltr] = raw_master_flat / norm
            self.master_variances[fltr] = np.pi / (2 * len(flats)) * (np.var(flats, axis=0, ddof=1) + bias_var) / norm**2
        
        print('[OPTICAM] Master flat-field image(s) created.')
        
        self._save_master_image(overwrite=overwrite)
        
        print(f'[OPTICAM] Master flat-field image(s) saved to {self.master_image_path}.')


    def run_checks(
        self,
        data_file_paths_by_filter: Dict[str, List[Path]],
        return_errors: bool = False,
        ) -> None | int:
        """
        Run a series of checks on the corrector to ensure that it is compatible with the data. In this case, check the
        binning of the science images can be matched to those of the flats (accounting for self.rebin_factor), and that
        the there are no missing filters.
        
        Parameters
        ----------
        data_file_paths_by_filter : Dict[str, List[Path]]
            The paths to all of the science images grouped by filter {filter: list of paths}.
        return_errors : bool, optional
            Whether to return the number of errors raised, by default `False`.
        
        Returns
        -------
        None | int
            If `return_errors=True`, the number of errors raised is returned. Otherwise, nothing is returned.
        """
        
        errors = 0
        warnings = 0
        
        #################################################### errors ####################################################
        
        # check filters match
        if self.data_paths.keys() != data_file_paths_by_filter.keys():
            errors += 1
            print(f'[OPTICAM] ERROR: inconsistent filters found between the flat-field images and the science images. Flat-field image filters: ({','.join(self.data_paths.keys())}); science image filters: ({','.join(data_file_paths_by_filter.keys())})')
        
        ################################################### warnings ###################################################
        
        # check binnings match
        flat_path = next(iter(self.data_paths.values()))[0]  # get the path to a random flat
        flat_binning = self.instrument.get_binning(file_path=flat_path)
        image_path = next(iter(data_file_paths_by_filter.values()))[0]  # get the path to a science image
        science_binning = self.instrument.get_binning(file_path=image_path)
        if flat_binning != science_binning:
            warnings += 1
            print(f'[OPTICAM] WARNING: inconsistent binning found between the flat-field images and the science images. Flat-field image binning: {flat_binning}; science image binning: {science_binning}. If you have passed a suitable rebin_factor to your FlatFieldCorrector instance, then you can safely ignore this warning.')
        
        ################################################### summary ###################################################
        
        if errors > 0 or warnings > 0:
            print()  # blank line for readibility
        
        if errors == 0:
            print(f'[OPTICAM] FlatFieldCorrector sucessfully passed all checks.')
        else:
            if errors == 1:
                print('[OPTICAM] FlatFieldCorrector failed 1 check.')
            else:
                print(f'[OPTICAM] FlatFieldCorrector failed {errors} checks.')
        
        if warnings == 1:
            print('[OPTICAM] FlatFieldCorrector triggered a warning during 1 check. Warnings may be ignored provided their caveats are satisfied.')
        elif warnings > 1:
            print(f'[OPTICAM] FlatFieldCorrector triggered a warning during {warnings} checks. Warnings may be ignored provided their caveats are satisfied.')
        
        if return_errors:
            return errors


    def _validate_data(
        self,
        file_paths: List[Path],
        ) -> Dict[str, List[Path]]:
        """
        Ensure that the flat-field images in the specified directory are valid (i.e., all use the same binning).
        
        Parameters
        ----------
        file_paths : List[Path]
            The paths to the flat-field images.
        
        Returns
        -------
        Dict[str, List[Path]]
            A dictionary containing the paths to the flat-field images for each filter.
        """
        
        filters, binnings = {}, {}
        
        for file_path in file_paths:
            header = fits.getheader(file_path)
            filters[file_path] = self.instrument.get_filter(header=header)
            binnings[file_path] = self.instrument.get_binning(header=header)
        
        unique_filters = set(filters.values())
        unique_binnings = set(binnings.values())
        
        if len(unique_binnings) > 1:
            log_file(
                out_directory=self.out_directory,
                file_name='binnings.json',
                file_contents=binnings,
                )
            raise ValueError(f'[OPTICAM] Inconsistent binning detected in the flat-field images. Image binnings have been logged to {self.out_directory.joinpath('diag/binnings.json')}')
        
        # get flats for each filter
        flats = {}
        for fltr in unique_filters:
            flats[fltr] = []
            for k, v in filters.items():
                if v == fltr:
                    flats[fltr].append(k)
        
        for k, v in flats.items():
            print(f'[OPTICAM] {len(v)} {k} flat-field images.')
        
        return flats




