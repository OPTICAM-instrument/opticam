from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Any


from astropy.coordinates import EarthLocation, SkyCoord
from astropy.io.fits import Header
from astropy.time import Time
import astropy.units as u
import json
import numpy as np


from opticam.mef_slice import MEFSlice




@dataclass
class Instrument(ABC):
    """
    Base class for instruments.
    
    Parameters
    ----------
    location : EarthLocation
        The location of the observatory as an `astropy.coordinates.EarthLocation` object.
    pixel_scales : dict[str, float]
        The pixel scales for each camera in arcsec/pixel {camera: pixel scale}.
    binning_kw : str, optional
        The binning keyword, by default "BINNING".
    camera_kw : str, optional
        The keyword that uniquely identifies the camera that took the image, by default "INSTRUME". For single-camera
        instruments, this keyword doesn't matter. For multi-camera instruments, however, it is used to apply 
        calibrations like flats correctly.
    dark_curr_kw : str, optional
        The dark current keyword, by default "DARKCURR". Dark current values are assumed to be in 
        electrons/pixel.
    dateobs_kw : str, optional
        The observation date keyword, by default "DATE-OBS". By default, observation dates are assumed to be in
        ISO 8601/FITS format (YYYY-MM-DDTHH:MM:SS[.sss]).
    dec_kw : str, optional
        The DEC keyword, by default "DEC". DEC values are assumed to be in units of degrees.
    exptime_kw : str, optional
        The exposure time keyword, by default "EXPTIME". Exposure times are assumed to be in units of seconds.
    filter_kw : str, optional
        The filter keyword, by default "FILTER".
    gain_kw : str, optional
        The gain keyword, by default "GAIN". Gain values are assumed to be in units of electrons/ADU.
    ra_kw : str, optional
        The RA keyword, by default "RA". RA values are assumed to be in units of hour angle.
    read_noise_kw : str, optional
        The read noise keyword, by default "RDNOISE".
    """


    location: EarthLocation
    pixel_scales: dict[str, float]
    binning_kw: str = 'BINNING'
    camera_kw: str = 'INSTRUME'
    dark_curr_kw: str = 'DARKCURR'
    dateobs_kw: str = 'DATE-OBS'
    dec_kw: str = 'DEC'
    exptime_kw: str = 'EXPTIME'
    filter_kw: str = 'FILTER'
    gain_kw: str = 'GAIN'
    ra_kw: str = 'RA'
    read_noise_kw: str = 'RDNOISE'


    def run_checks(
        self,
        file: MEFSlice | Path,
        return_errors: bool = False,
        ) -> None | int:
        """
        Check that the instrument can be used to parse an image's header.
        
        Parameters
        ----------
        file : MEFSlice | Path
            The file to use for checking the instrument. If a `Path` or `str` instance is specified, the first HDU of
            the corresponding FITS file is used.
        return_errors : bool, optional
            Whether to return the number of errors raised, by default `False`.
        
        Returns
        -------
        None | int
            If `return_errors=True`, returns the number of errors raised. Otherwise, nothing is returned.
        
        Raises
        ------
        ValueError
            If the header of the file could not be read.
        """
        
        print(f'[OPTICAM] Checking instrument {self.__class__.__name__}.')
        
        if isinstance(file, Path) or isinstance(file, str):
            file = MEFSlice(path=Path(file).resolve(), ext=0)
        
        header = file.get_header()
        keys = list(header.keys())
        
        errors = 0
        warnings = 0
        
        #################################################### errors ####################################################
        
        if self.exptime_kw not in keys:
            errors += 1
            print(f'[OPTICAM] ERROR: {self.__class__.__name__}.exptime_kw ({self.exptime_kw}) is not a valid header keyword for file {file.path} extension {file.ext}.')
        
        if self.filter_kw not in keys:
            errors += 1
            print(f'[OPTICAM] ERROR: {self.__class__.__name__}.filter_kw ({self.filter_kw}) is not a valid header keyword for file {file.path} extension {file.ext}.')
        else:
            try:
                fltr = self.get_filter(header=header)
                assert(isinstance(fltr, str))
            except Exception as e:
                errors += 1
                print(f'[OPTICAM] ERROR: {self.__class__.__name__}.get_filter() failed due to the following exception: {e}.')
        
        if self.gain_kw not in keys:
            errors += 1
            print(f'[OPTICAM] ERROR: {self.__class__.__name__}.gain_kw ({self.gain_kw}) is not a valid header keyword for file {file.path} extension {file.ext}.')
        
        if self.dateobs_kw not in keys:
            errors += 1
            print(f'[OPTICAM] ERROR: {self.__class__.__name__}.dateobs_kw ({self.dateobs_kw}) is not a valid header keyword for file {file.path} extension {file.ext}.')
        
        if self.filter_kw in keys:
            try:
                self.pixel_scales[self.get_camera(header=header)]
            except Exception as e:
                errors += 1
                print(f'[OPTICAM] ERROR: {self.__class__.__name__}.pixel_scales does not contain a corresponding value for the camera {self.get_camera(header=header)}.')
        
        try:
            self.get_binning(header=header)
        except Exception as e:
            errors += 1
            print(f"[OPTICAM] ERROR: failed to read image binning for file {file.path} extension {file.ext} due to the exception {e} This is either due to an incorrect keyword being passed to binning_kw and/or your images do not contain a binning keyword. In the latter case, you will need to define a custom instrument with a custom get_binning() method. See https://opticam.readthedocs.io/en/latest/_executed/instruments.html#My-images-don't-contain-a-binning-keyword.-What-should-I-do? for details.")
        
        try:
            Time(self.get_mjd(header=header), format='mjd')
        except Exception as e:
            errors += 1
            print(f"[OPTICAM] ERROR: Failed to parse the MJD of the image due the following exception: {e}. This is likely due to an incorrect keyword being passed to dateobs_kw and/or your images do not give timestamps in FITS format. In the latter case, you will need to define a custom instrument with a custom get_mjd() method. See https://opticam.readthedocs.io/en/latest/_executed/instruments.html#Defining-an-instrument-from-the-opticam.Instrument-base-class for details.")
        
        try:
            float(self.get_read_noise(header=header))
        except Exception as e:
            errors += 1
            print(f"[OPTICAM] ERROR: Failed to parse the read noise in the image due the following exception: {e}. This is likely due to an incorrect keyword being passed to read_noise_kw or your images do not contain read noise information. In the latter case, you will need to define a custom instrument with a custom get_read_noise() method. See https://opticam.readthedocs.io/en/latest/_executed/instruments.html#Defining-an-instrument-from-the-opticam.Instrument-base-class for details.")
        
        ################################################### warnings ###################################################
        
        if self.ra_kw not in keys:
            warnings += 1
            print(f'[OPTICAM] WARNING: {self.__class__.__name__}.ra_kw ({self.ra_kw}) is not a valid header keyword for file {file.path} extension {file.ext}.')
        
        if self.dec_kw not in keys:
            warnings += 1
            print(f'[OPTICAM] WARNING: {self.__class__.__name__}.dec_kw ({self.dec_kw}) is not a valid header keyword for file {file.path} extension {file.ext}.')
        
        try:
            self.get_sky_coord(header=header)
        except Exception as e:
            warnings += 1
            print(f'[OPTICAM] Warning: {self.__class__.__name__}.get_sky_coord() failed due to the following exception: {e}. Barycentric correction will not be possible. If this is a mistake, check the specified RA and DEC keywords ({self.ra_kw} and {self.dec_kw}, respectively) are present in your image headers. If they are present, then they are likely in an unrecognised format. In this case, you will need to define a custom instrument with a custom get_sky_coord() method. See https://opticam.readthedocs.io/en/latest/autoapi/opticam/instruments/index.html#opticam.instruments.Instrument.get_sky_coord for details.')
        
        if self.dark_curr_kw not in keys:
            warnings += 1
            print(f'[OPTICAM] WARNING: {self.__class__.__name__}.dark_curr_kw ({self.dark_curr_kw}) is not a valid header keyword for file {file.path} extension {file.ext}. If no dark current is listed in the image headers, you will need to use a `opticam.DarkNoiseCorrector` instance to correct for dark noise. See https://opticam.readthedocs.io/en/latest/_executed/applying_corrections.html#Dark-noise for details.')
        
        ################################################### summary ###################################################
        
        if errors == 0:
            print(f'[OPTICAM] {self.__class__.__name__} sucessfully passed all checks.')
        else:
            if errors == 1:
                print(f'[OPTICAM] {self.__class__.__name__} failed 1 check.')
            else:
                print(f'[OPTICAM] {self.__class__.__name__} failed {errors} checks.')
        
        if warnings == 1:
            print(f'[OPTICAM] {self.__class__.__name__} triggered a warning during 1 check. Warnings may be ignored provided their caveats are satisfied.')
        elif warnings > 1:
            print(f'[OPTICAM] {self.__class__.__name__} triggered a warning during {warnings} checks. Warnings may be ignored provided their caveats are satisfied.')
        
        if return_errors:
            return errors


    def get_mjd(
        self,
        file: MEFSlice | None = None,
        header: Header | None = None,
        ) -> float:
        """
        Given the path to a FITS file, or its header, parse its observation date into *local* Modified Julian Date (MJD).
        
        Parameters
        ----------
        file : MEFSlice | None, optional
            The `MEFSlice` instance corresponding to the file, by default `None`. If `None`, a `Header` must be passed to
            `header` instead.
        header : Header, optional
            The header of the FITS file, by default `None`. If `None`, a `MEFSlice` must be passed to `file` instead.
        
        Returns
        -------
        float
            The local MJD of the image.
        """
        
        if file is not None:
            header: Header = file.get_header()
        
        timestamp = str(header[self.dateobs_kw])
        mjd = float(np.asarray(Time(timestamp, format="fits").mjd))
        
        return mjd


    def get_camera(
        self,
        file: MEFSlice | None = None,
        header: Header | None = None,
        ) -> str:
        """
        Given the path to a FITS file, get the corresponding camera.
        
        Parameters
        ----------
        file : MEFSlice | None, optional
            The `MEFSlice` instance corresponding to the file, by default `None`. If `None`, a `Header` must be passed
            to `header` instead.
        header : Header, optional
            The header of the FITS file, by default `None`. If `None`, a `MEFSlice` must be passed to `file` instead.
        
        Returns
        -------
        str
            A unique identifier for the camera.
        """
        
        if file is not None:
            header = file.get_header()
        
        return str(header[self.camera_kw])


    def get_sky_coord(
        self,
        file: MEFSlice | None = None,
        header: Header | None = None,
        ) -> SkyCoord:
        """
        Given the path to a FITS file, get the corresponding sky coordinates.
        
        Parameters
        ----------
        file : MEFSlice | None, optional
            The `MEFSlice` instance corresponding to the file, by default `None`. If `None`, a `Header` must be passed to
            `header` instead.
        header : Header, optional
            The header of the FITS file, by default `None`. If `None`, a `MEFSlice` must be passed to `file` instead.
        
        Returns
        -------
        SkyCoord
            The sky coordinates of the image.
        """
        
        if file is not None:
            header = file.get_header()
        
        sky_coord =  SkyCoord(
            header[self.ra_kw],
            header[self.dec_kw],
            unit=(u.hourangle, u.deg),
            )
        
        return sky_coord


    def get_dark_flux(
        self,
        file: MEFSlice | None = None,
        header: Header | None = None,
        ) -> float | None:
        """
        Given the path to a FITS file, get the corresponding dark flux (i.e., the exposure-integrated dark current). If
        the instrument does not list a dark current in the image headers, the returned dark flux can be `None`.
        
        Parameters
        ----------
        file : MEFSlice | None, optional
            The `MEFSlice` instance corresponding to the file, by default `None`. If `None`, a `Header` must be passed to
            `header` instead.
        header : Header, optional
            The header of the FITS file, by default `None`. If `None`, a `MEFSlice` must be passed to `file` instead.
        
        Returns
        -------
        float | None
            The dark flux in the image.
        """
        
        if file is not None:
            header = file.get_header()
        
        try:
            dark_curr = float(header[self.dark_curr_kw])
        except KeyError:
            return
        
        exptime = float(header[self.exptime_kw])
        
        return dark_curr * exptime


    def get_binning(
        self,
        file: MEFSlice | None = None,
        header: Header | None = None,
        ) -> str:
        """
        Get the binning of an image using the instrument's `binning_kw` attribute.
        
        Parameters
        ----------
        file : MEFSlice | None, optional
            The `MEFSlice` instance corresponding to the file, by default `None`. If `None`, a `Header` must be passed to
            `header` instead.
        header : Header, optional
            The header of the FITS file, by default `None`. If `None`, a `MEFSlice` must be passed to `file` instead.
        
        Returns
        -------
        str
            The binning of the image.
        """
        
        if file is not None:
            header: Header = file.get_header()
        
        return header[self.binning_kw]


    def get_filter(
        self,
        file: MEFSlice | None = None,
        header: Header | None = None,
        ) -> str:
        """
        Get the filter of an image using the instrument's `filter_kw` attribute.
        
        Parameters
        ----------
        file : MEFSlice | None, optional
            The `MEFSlice` instance corresponding to the file, by default `None`. If `None`, a `Header` must be passed to
            `header` instead.
        header : Header, optional
            The header of the FITS file, by default `None`. If `None`, a `MEFSlice` must be passed to `file` instead.
        
        Returns
        -------
        str
            The filter of the image.
        """
        
        if file is not None:
            header: Header = file.get_header()
        
        return header[self.filter_kw]


    def get_read_noise(
        self,
        file: MEFSlice | None = None,
        header: Header | None = None,
        ) -> float:
        """
        Get the read noise in an image, in electrons per pixel, using the instrument's `read_noise_kw` attribute.
        
        Parameters
        ----------
        file : MEFSlice | None, optional
            The `MEFSlice` instance corresponding to the file, by default `None`. If `None`, a `Header` must be passed 
            to `header` instead.
        header : Header, optional
            The header of the FITS file, by default `None`. If `None`, a `MEFSlice` must be passed to `file` instead.
        
        Returns
        -------
        float
            The read noise in the image.
        """
        
        if file is not None:
            header: Header = file.get_header()
        
        return float(header[self.read_noise_kw])


    @classmethod
    def from_json(
        cls,
        file_path: Path | str | None = None,
        config: dict[str, Any] | None = None,
        ) -> 'Instrument':
        """
        Create an instrument from a configuration file/dictionary.
        
        Parameters
        ----------
        file_path : Path | str | None, optional
            The path to the configuration file, by default `None`. If `None`, a dictionary must be passed to `config`. 
            If a value is passed to `file_path`, `config` is ignored.
        config : dict[str, Any] | None, optional
            The configuration dictionary, by default `None`. If `None`, a path must be passed to `file_path`. If a value
            is passed to `file_path`, `config` is ignored.
        
        Returns
        -------
        Instrument
            The configured instrument.
        
        Raises
        ------
        AssertionError
            If required keys are missing from the configuration file/dictionary.
        """
        
        assert(file_path is not None or config is not None), "[OPTICAM] Cannot create an instrument if file_path and config are both undefined."
        
        if file_path is not None:
            with open(file_path, 'r') as json_file:
                config = json.load(json_file)
        
        template = create_template()
        template_keys = list(template.keys())
        template_keys = [key for key in template_keys if not key.startswith('_')]  # remove comments
        
        config_keys = list(config.keys())
        config_keys = [key for key in config_keys if not key.startswith('_')]  # remove comments
        
        if not set(template_keys) == set(config_keys):
            missing_keys = [key for key in config_keys if not key in template_keys]
            raise AssertionError(f'[OPTICAM] Cannot create instrument from given config due to the following keys not being present: {','.join(missing_keys)}')
        
        location = EarthLocation.from_geodetic(
            lon=config['longitude'],
            lat=config['latitude'],
            height=config['height'],
            )
        
        return cls(
            location=location,
            pixel_scales=config['pixel_scales'],
            read_noise_kw=config['read_noise_kw'],
            binning_kw=config['binning_kw'],
            dark_curr_kw=config['dark_curr_kw'],
            exptime_kw=config['exptime_kw'],
            filter_kw=config['filter_kw'],
            gain_kw=config['gain_kw'],
            dateobs_kw=config['dateobs_kw'],
            ra_kw=config['ra_kw'],
            dec_kw=config['dec_kw'],
        )


    def to_json(
        self,
        file_path: Path | str,
        ) -> None:
        """
        Export the instrument configuration to a JSON file.
        
        Parameters
        ----------
        file_path : Path | str
            The location to which the file is written. If `save_path` does not include the file name, the file will be
            saved as `instrument_config.json`.
        """
        
        save_path = Path(file_path)
        
        if save_path.suffix:
            directory = save_path.parent
        else:
            directory = save_path
            save_path = save_path.joinpath('instrument_config.json')
        
        if not directory.is_dir():
            directory.mkdir(parents=True)
        
        template = create_template()
        
        template['longitude'] = self.location.lon.to_value(u.deg)
        template['latitude'] = self.location.lat.to_value(u.deg)
        template['height'] = self.location.height.to_value(u.m)
        template['pixel_scales'] = self.pixel_scales
        template['read_noise_kw'] = self.read_noise_kw
        template['binning_kw'] = self.binning_kw
        template['dark_curr_kw'] = self.dark_curr_kw
        template['filter_kw'] = self.filter_kw
        template['gain_kw'] = self.gain_kw
        template['dateobs_kw'] = self.dateobs_kw
        template['ra_kw'] = self.ra_kw
        template['dec_kw'] = self.dec_kw
        
        with open(save_path, 'w') as file:
            json.dump(
                template,
                file,
                indent=4,
                )




def generate_instrument_json_template(out_directory: Path | str) -> None:
    """
    Generate a template JSON file that can be used to define an `Instrument`.
    
    Parameters
    ----------
    out_directory : Path | str
        The path to the directory to which the template is saved.
    """
    
    out_directory = Path(out_directory)
    
    template = create_template()
    
    with open(out_directory.joinpath('instrument_template.json'), 'w') as file:
        json.dump(
            template,
            file,
            indent=4,
            )


def create_template() -> dict[str, Any]:
    """
    Create an instrument configuration template.
    
    Returns
    -------
    dict[str, Any]
        The instrument configuration template.
    """
    
    return {
        'longitude': 0.0,
        '_longitude_description': 'The East longitude of the observatory in degrees.',
        'latitude': 0.0,
        '_latitude_description': 'The latitude of the observatory in degrees.',
        'height': 0.0,
        '_height_description': 'The height of the observatory in meters.',
        'pixel_scales': {
            'filter_1': 0.0,
            'filter_2': 0.0,
            },
        '_pixel_scales_description': 'The pixel-scale in arcsec/pixel for each filter.',
        'read_noise_kw': 'RDNOISE',
        '_read_noise_description': "The header keyword that corresponds to the detector's readout noise in electrons/pixel.",
        'binning_kw': 'BINNING',
        '_binning_kw_description': 'The header keyword that corresponds to the binning mode.',
        'dark_curr_kw': 'DARKCURR',
        '_dark_curr_kw_description': "The header keyword that corresponds to the detector's dark current in electrons/pixel/s.",
        'exptime_kw': 'EXPTIME',
        '_exptime_kw_description': 'The header keyword that corresponds to the exposure time in seconds.',
        'filter_kw': 'FILTER',
        '_filter_kw_description': 'The header keyword that corresponds to the image filter.',
        'gain_kw': 'GAIN',
        '_gain_kw_description': "The header keyword that corresponds to the detector's gain value in electrons/ADU.",
        'dateobs_kw': 'DATE-OBS',
        '_dateobs_kw_description': "The header keyword that corresponds to the image's timestamp in ISO 8601/FITS format (i.e., YYYY-MM-DDTHH:MM:SS[.sss]). If your instrument does not give timestamps in this format, you will need to define the instrument with a custom get_mjd() method. See https://opticam.readthedocs.io/en/latest/_executed/instruments.html#Defining-an-instrument-from-the-opticam.Instrument-base-class for details.",
        'ra_kw': 'RA',
        '_ra_kw_description': "The header keyword that corresponds to the image's RA in units of hour angle. If your instrument does not give the RA in units of hour angle, you will need to define the instrument with a custom get_sky_coord() method. See https://opticam.readthedocs.io/en/latest/autoapi/opticam/instruments/index.html#opticam.instruments.Instrument.get_sky_coord for details.",
        'dec_kw': 'DEC',
        '_dec_kw_description': "The header keyword that corresponds to the image's DEC in units of degrees. If your instrument does not give the DEC in units of degrees, you will need to define the instrument with a custom get_sky_coord() method. See https://opticam.readthedocs.io/en/latest/autoapi/opticam/instruments/index.html#opticam.instruments.Instrument.get_sky_coord for details.",
        }




class OPTICAM_MX(Instrument):
    """
    OAN-SPM OPTICAM-MX instrument.
    """


    def __init__(
        self, 
        location = EarthLocation.from_geodetic(
            lon=-115.463611 * u.deg,
            lat=31.044167 * u.deg,
            height=2790 * u.m,
            ),
        pixel_scales = {
            '1': 0.1397,
            '2': 0.1406,
            '3': 0.1661,
            },
        dateobs_kw='UT',
        exptime_kw='EXPOSURE',
        ) -> None:
        """
        Interface for the OAN-SPM OPTICAM instrument.
        
        Parameters
        ----------
        location : EarthLocation
            The location of the observatory.
        pixel_scales : dict
            The instrument's pixel scales.
        dateobs_kw : str
            The observation date keyword.
        exptime_kw : str
            The exposure time keyword.
        """
        
        return super().__init__(
            location=location,
            pixel_scales=pixel_scales,
            exptime_kw=exptime_kw,
            dateobs_kw=dateobs_kw,
            )


    def get_mjd(
        self,
        file: MEFSlice | None = None,
        header: Header | None = None,
        ) -> float:
        """
        Get the timestamp of the image in MJD. OPTICAM uses a "UT" keyword to represent an image's timestamp in ISO
        format.
        
        Parameters
        ----------
        file : MEFSlice | None, optional
            The `MEFSlice` instance corresponding to the file, by default `None`. If `None`, a `Header` must be passed 
            to `header` instead.
        header : Header, optional
            The header of the FITS file, by default `None`. If `None`, a `MEFSlice` must be passed to `file` instead.
        
        Returns
        -------
        float
            The timestamp of the image in MJD format.
        """
        
        if file is not None:
            header = file.get_header()
        
        obs_time = Time(str(header[self.dateobs_kw]), format="iso")
        exposure = float(header[self.exptime_kw]) * u.s
        
        mjd = np.asarray((obs_time + exposure / 2).mjd)
        
        return float(mjd)


    def get_camera(
        self,
        file: MEFSlice | None = None,
        header: Header | None = None,
        ) -> str:
        
        if file is not None:
            header = file.get_header()
        
        fltr = self.get_filter(header=header)
        
        if fltr in ["u", "u'", "g", "g'"]:
            return '1'
        elif fltr in ["r", "r'"]:
            return '2'
        elif fltr in ["i", "z'"]:
            return '3'
        
        return 'None'


    def get_read_noise(
        self,
        file: MEFSlice | None = None,
        header: Header | None = None,
        ) -> float:
        
        return 1.1



# TODO: check when this is needed
class OPTICAM_MX_UNKNOWN(OPTICAM_MX):


    def __init__(
        self,
        ):
        super().__init__(dateobs_kw='GPSTIME')




class MEXMAN(Instrument):
    """
    OAN-SPM MEXMAN instrument.
    """
    
    def __init__(
        self,
        location: EarthLocation = EarthLocation.from_geodetic(
            lon=-115.463611 * u.deg,
            lat=31.044167 * u.deg,
            height=2790 * u.m,
            ),
        pixel_scales: dict[str, float] = {
            'MEXMAN': 0.24645,
            },
        dateobs_kw: str = 'JD',
        binning_kw: str = 'CCDSUM',
        ) -> None:
        """
        Interface for the OAN-SPM MEXMAN instrument.
        
        Parameters
        ----------
        location : EarthLocation
            The location of the observatory.
        pixel_scales : dict
            The instrument's pixel scale.
        dateobs_kw : str, optional
            The observation date keyword.
        binning_kw : str, optional
            The binning keyword.
        """
        
        return super().__init__(
            location=location,
            pixel_scales=pixel_scales,
            dateobs_kw=dateobs_kw,
            binning_kw=binning_kw,
            )


    def get_camera(
        self,
        file: MEFSlice | None = None,
        header: Header | None = None,
        ) -> str:
        """
        Get the camera used to create the file. Since MEXMAN is a single-camera instrument, the name of the instrument
        is returned.
        
        Parameters
        ----------
        file : MEFSlice | None, optional
            The `MEFSlice` instance corresponding to the file, by default `None`. If `None`, a `Header` must be passed 
            to `header` instead.
        header : Header, optional
            The header of the FITS file, by default `None`. If `None`, a `MEFSlice` must be passed to `file` instead.
        
        Returns
        -------
        str
            The camera used to create the file.
        """
        
        return 'MEXMAN'


    def get_mjd(
        self,
        file: MEFSlice | None = None,
        header: Header | None = None,
        ) -> float:
        """
        Get the timestamp of the image in MJD. MEXMAN uses a "JD" keyword to represent an image's timestamp in Julian 
        Date format.
        
        Parameters
        ----------
        file : MEFSlice | None, optional
            The `MEFSlice` instance corresponding to the file, by default `None`. If `None`, a `Header` must be passed 
            to `header` instead.
        header : Header, optional
            The header of the FITS file, by default `None`. If `None`, a `MEFSlice` must be passed to `file` instead.
        
        Returns
        -------
        float
            The timestamp of the image in MJD format.
        """
        
        if file is not None:
            header = file.get_header()
        
        jd = header[self.dateobs_kw]
        
        return Time(jd, format='jd').mjd









