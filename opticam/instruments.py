from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


from astropy.coordinates import EarthLocation, SkyCoord
from astropy.io import fits
from astropy.io.fits import Header
from astropy.time import Time
import astropy.units as u
import json
import numpy as np




@dataclass
class Instrument(ABC):
    """
    Base class for instruments.
    
    Parameters
    ----------
    location : EarthLocation
        The location of the observatory as an `astropy.coordinates.EarthLocation` object.
    pixel_scales : Dict[str, float]
        The pixel scales for each filter in arcsec/pixel {filter: pixel scale}.
    read_noise : float
        The detector's read noise in electrons/pixel.
    binning_kw : str, optional
        The binning keyword, by default "BINNING".
    dark_curr_kw : str, optional
        The dark current keyword, by default "DARKCURR". Dark current values are assumed to be in 
        electrons/pixel.
    exptime_kw : str, optional
        The exposure time keyword, by default "EXPTIME". Exposure times are assumed to be in units of seconds.
    filter_kw : str, optional
        The filter keyword, by default "FILTER".
    gain_kw : str, optional
        The gain keyword, by default "GAIN". Gain values are assumed to be in units of electrons/ADU.
    dateobs_kw : str, optional
        The observation date keyword, by default "DATE-OBS". By default, observation dates are assumed to be in
        ISO 8601/FITS format (YYYY-MM-DDTHH:MM:SS[.sss]).
    ra_kw : str, optional
        The RA keyword, by default RA. RA values are assumed to be in units of hour angle.
    dec_kw : str, optional
        The DEC keyword, by default DEC. DEC values are assumed to be in units of degrees.
    """


    location: EarthLocation
    pixel_scales: Dict[str, float]
    read_noise: float
    binning_kw: str = 'BINNING'
    dark_curr_kw: str = 'DARKCURR'
    exptime_kw: str = 'EXPTIME'
    filter_kw: str = 'FILTER'
    gain_kw: str = 'GAIN'
    dateobs_kw: str = 'DATE-OBS'
    ra_kw: str = 'RA'
    dec_kw: str = 'DEC'


    def run_checks(
        self,
        file_path: Path | str,
        return_errors: bool = False,
        ) -> None | int:
        """
        Check that the instrument can be used to parse an image's header.
        
        Parameters
        ----------
        file_path : Path | str
            The path to the FITS file.
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
        
        try:
            header = fits.getheader(file_path)
        except Exception as e:
            raise ValueError(f'[OPTICAM] Could not get header of {file_path} due to the following exception: {e}.')
        
        keys = list(header.keys())
        errors = 0
        warnings = 0
        
        #################################################### errors ####################################################
        
        if self.exptime_kw not in keys:
            errors += 1
            print(f'[OPTICAM] ERROR: instrument.exptime_kw ({self.exptime_kw}) is not a valid header key for file {file_path}.')
        
        if self.filter_kw not in keys:
            errors += 1
            print(f'[OPTICAM] ERROR: instrument.filter_kw ({self.filter_kw}) is not a valid header key for file {file_path}.')
        else:
            try:
                fltr = self.get_filter(header=header)
                assert(isinstance(fltr, str))
            except Exception as e:
                errors += 1
                print(f'[OPTICAM] ERROR: instrument.get_filter() failed due to the following exception: {e}.')
        
        if self.gain_kw not in keys:
            errors += 1
            print(f'[OPTICAM] ERROR: instrument.gain_kw ({self.gain_kw}) is not a valid header key for file {file_path}.')
        
        if self.dateobs_kw not in keys:
            errors += 1
            print(f'[OPTICAM] ERROR: instrument.dateobs_kw ({self.dateobs_kw}) is not a valid header key for file {file_path}.')
        
        if self.filter_kw in keys:
            try:
                self.pixel_scales[self.get_filter(header=header)]
            except Exception as e:
                errors += 1
                print(f'[OPTICAM] ERROR: instrument.pixel_scales does not contain a corresponding value for the filter {self.get_filter(header=header)}.')
        
        try:
            self.get_binning(file_path=file_path)
        except Exception as e:
            errors += 1
            print(f'[OPTICAM] ERROR: failed to read image binning for file {file_path} due to the exception {e}. This is either due to an incorrect value being passed to binning_kw, or your images do not contain a binning keyword. In the latter case, you will need to define a custom instrument with a custom get_binning() method. See (TODO: link) for more details.')
        
        try:
            Time(self.get_mjd(file_path=file_path), format='mjd')
        except Exception as e:
            errors += 1
            print(f'[OPTICAM] ERROR: Failed to parse the MJD of the image due the following exception: {e}')
        
        try:
            self.get_sky_coord(file_path=file_path)
        except Exception as e:
            errors += 1
            print(f'[OPTICAM] ERROR: instrument.get_sky_coord() failed due to the following exception: {e}')
        
        ################################################### warnings ###################################################
        
        if errors > 0:
            print()  # blank line for readibility
        
        if self.ra_kw not in keys:
            warnings += 1
            print(f'[OPTICAM] WARNING: instrument.ra_kw ({self.ra_kw}) is not a valid header key for file {file_path}. Barycentric corrections will not be possible. This is either due to an inccorect value being passed to ra_kw, or your images do not contain coordinate information. In the latter case, you can ignore this message but must pass barycenter=False when using this instrument to create an opticam.Reducer instance.')
        if self.dec_kw not in keys:
            warnings += 1
            print(f'[OPTICAM] WARNING: instrument.dec_kw ({self.dec_kw}) is not a valid header key for file {file_path}. Barycentric corrections will not be possible. This is either due to an inccorect value being passed to dec_kw, or your images do not contain coordinate information. In the latter case, you can ignore this message but must pass barycenter=False when using this instrument to create an opticam.Reducer instance.')
        
        if self.dark_curr_kw not in keys:
            warnings += 1
            print(f'[OPTICAM] WARNING: instrument.dark_curr_kw ({self.dark_curr_kw}) is not a valid header key for file {file_path}. If no dark current is listed in the image headers, dark images can be used alongside a `opticam.DarkNoiseCorrector` instance to correct for the dark noise.')
        
        ################################################### summary ###################################################
        
        if errors > 0 or warnings > 0:
            print()  # blank line for readibility
        
        if errors == 0:
            print(f'[OPTICAM] Instrument sucessfully passed all checks.')
        else:
            if errors == 1:
                print('[OPTICAM] Instrument failed 1 check.')
            else:
                print(f'[OPTICAM] Instrument failed {errors} checks.')
        
        if warnings == 1:
            print('[OPTICAM] Instrument triggered a warning during 1 check. Warnings may be ignored provided their caveats are satisfied.')
        elif warnings > 1:
            print(f'[OPTICAM] Instrument triggered a warning during {warnings} checks. Warnings may be ignored provided their caveats are satisfied.')
        
        if return_errors:
            return errors


    def get_mjd(
        self,
        file_path: Path | str | None = None,
        header: Header | None = None,
        ) -> float:
        """
        Given the path to a FITS file, or its header, parse its observation date into *local* Modified Julian Date (MJD).
        
        Parameters
        ----------
        file_path : Path | str | None, optional
            The path to the FITS file, by default `None`. If `None`, a header must be passed to `header` instead.
        header : Header, optional
            The header of the FITS file, by default `None`. If `None`, a file path must be passed to `file_path` instead.
        
        Returns
        -------
        float
            The local MJD of the image.
        """
        
        if header is None:
            header: Header = fits.getheader(file_path)
        
        try:
            timestamp = str(header[self.dateobs_kw])
            mjd = float(np.asarray(Time(timestamp, format="fits").mjd))
        except Exception as e:
            raise ValueError(f'[OPTICAM] Unable to get MJD of file {file_path} due to the following exception: {e}. If using a custom instrument, it is likely that either the dateobs_kw does not exist in the image header or, if it does, the timestamp is not given in FITS format; in this case, you will need to define your instrument as a class, inheriting from opticam.Instrument, and implement a custom get_mjd() method that parses the timestamp into an MJD. See [TODO: link to instruments tutorial] for more details.')
        
        return mjd


    def get_sky_coord(
        self,
        file_path: Path | str | None = None,
        header: Header | None = None,
        ) -> SkyCoord:
        """
        Given the path to a FITS file, get the corresponding sky coordinates.
        
        Parameters
        ----------
        file_path : Path | str | None, optional
            The path to the FITS file, by default `None`. If `None`, a header must be passed to `header` instead.
        header : Header, optional
            The header of the FITS file, by default `None`. If `None`, a file path must be passed to `file_path` instead.
        
        Returns
        -------
        SkyCoord
            The sky coordinates of the image.
        """
        
        if header is None:
            header: Header = fits.getheader(file_path)
        
        try:
            sky_coord =  SkyCoord(
                header[self.ra_kw],
                header[self.dec_kw],
                unit=(u.hourangle, u.deg),
                )
        except Exception as e:
            raise ValueError(f'[OPTICAM] Unable to get sky coord of file {file_path} due to the following exception: {e}. If using a custom instrument, it is likely that the ra_kw and/or dec_kw rows do not exist in the image header or, if they do, the values are not given in the expected units (hour angle for RA and degrees for DEC). If your instrument uses different units, you will need to define your instrument as a class, inheriting from opticam.Instrument, and implement a custom get_sky_coord() method that parses the sky coordinates into an astropy.coordinates.SkyCoord object. See [TODO: link to instruments tutorial] for more details.')
        
        return sky_coord


    def get_dark_flux(
        self,
        file_path: Path | str | None = None,
        header: Header | None = None,
        ) -> float | None:
        """
        Given the path to a FITS file, get the corresponding dark flux (i.e., the exposure-integrated dark current). If
        the instrument does not list a dark current in the image headers, the returned dark flux can be `None`.
        
        Parameters
        ----------
        file_path : Path | str | None, optional
            The path to the FITS file, by default `None`. If `None`, a header must be passed to `header` instead.
        header : Header, optional
            The header of the FITS file, by default `None`. If `None`, a file path must be passed to `file_path` instead.
        
        Returns
        -------
        float | None
            The dark flux in the image.
        """
        
        if header is None:
            header: Header = fits.getheader(file_path)
        
        try:
            dark_curr = float(header[self.dark_curr_kw])
        except KeyError:
            return
        
        exptime = float(header[self.exptime_kw])
        
        return dark_curr * exptime


    def get_binning(
        self,
        file_path: Path | str | None = None,
        header: Header | None = None,
        ) -> str:
        """
        Get the binning of an image using the instrument's `binning_kw` attribute.
        
        Parameters
        ----------
        file_path : Path | str | None, optional
            The path to the FITS file, by default `None`. If `None`, a header must be passed to `header` instead.
        header : Header, optional
            The header of the FITS file, by default `None`. If `None`, a file path must be passed to `file_path` instead.
        
        Returns
        -------
        str
            The binning of the image.
        """
        
        if header is None:
            header: Header = fits.getheader(file_path)
        
        return header[self.binning_kw]


    def get_filter(
        self,
        file_path: Path | str | None = None,
        header: Header | None = None,
        ) -> str:
        """
        Get the filter of an image using the instrument's `filter_kw` attribute.
        
        Parameters
        ----------
        file_path : Path | str | None, optional
            The path to the FITS file, by default `None`. If `None`, a header must be passed to `header` instead.
        header : Header, optional
            The header of the FITS file, by default `None`. If `None`, a file path must be passed to `file_path` instead.
        
        Returns
        -------
        str
            The filter of the image.
        """
        
        if header is None:
            header: Header = fits.getheader(file_path)
        
        return header[self.filter_kw]


    @classmethod
    def from_json(
        cls,
        file_path: Path | str | None = None,
        config: Dict[str, Any] | None = None,
        ) -> 'Instrument':
        """
        Create an instrument from a configuration file/dictionary.
        
        Parameters
        ----------
        file_path : Path | str | None, optional
            The path to the configuration file, by default `None`. If `None`, a dictionary must be passed to `config`. 
            If a value is passed to `file_path`, `config` is ignored.
        config : Dict[str, Any] | None, optional
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
            read_noise=config['read_noise'],
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
        template['read_noise'] = self.read_noise
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


def create_template() -> Dict[str, Any]:
    """
    Create an instrument configuration template.
    
    Returns
    -------
    Dict[str, Any]
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
        'read_noise': 0.0,
        '_read_noise_description': "The detector's readout noise in electrons/pixel.",
        'binning_kw': 'BINNING',
        '_binning_kw_description': 'The header keyword that corresponds to the binning mode.',
        'dark_curr_kw': 'DARKCURR',
        '_dark_curr_kw_description': "The header keyword that corresponds to the detector's dark current in electrons/pixel.",
        'exptime_kw': 'EXPTIME',
        '_exptime_kw_description': 'The header keyword that corresponds to the exposure time in seconds.',
        'filter_kw': 'FILTER',
        '_filter_kw_description': 'The header keyword that corresponds to the image filter.',
        'gain_kw': 'GAIN',
        '_gain_kw_description': "The header keyword that corresponds to the detector's gain value in electrons/ADU.",
        'dateobs_kw': 'DATE-OBS',
        '_dateobs_kw_description': "The header keyword that corresponds to the image's timestamp in ISO 8601/FITS format (i.e., YYYY-MM-DDTHH:MM:SS[.sss]). If your instrument does not give timestamps in this format, you will need to define the instrument with a custom get_mjd() method. See [TODO: link to docs] for details.",
        'ra_kw': 'RA',
        '_ra_kw_description': "The header keyword that corresponds to the image's RA in units of hour angle. If your instrument does not give the RA in units of hour angle, you will need to define the instrument with a custom get_sky_coord() method. See [TODO: link to docs] for details.",
        'dec_kw': 'DEC',
        '_dec_kw_description': "The header keyword that corresponds to the image's DEC in units of degrees. If your instrument does not give the DEC in units of degrees, you will need to define the instrument with a custom get_sky_coord() method. See [TODO: link to docs] for details.",
        }



class OPTICAM_MX(Instrument):
    """
    OPTICAM-MX instrument. For use with OPTICAM-MX data taken after 2022. For OPTICAM-MX observations taken in 2022,
    use `OPTICAMMX2022` instead.
    """


    def __init__(
        self, 
        location = EarthLocation.from_geodetic(
            lon=-115.463611 * u.deg,
            lat=31.044167 * u.deg,
            height=2790 * u.m,
            ),
        pixel_scales = {
            'u': 0.1397,
            "u'": 0.1397,
            'g': 0.1397,
            "g'": 0.1397,
            'r': 0.1406,
            "r'": 0.1406,
            'i': 0.1661,
            "i'": 0.1661,
            'z': 0.1661,
            "z'": 0.1661,
            },
        read_noise = 1.1,
        exptime_kw='EXPOSURE',
        dateobs_kw='UT',
        ):
        
        return super().__init__(
            location=location,
            pixel_scales=pixel_scales,
            read_noise=read_noise,
            exptime_kw=exptime_kw,
            dateobs_kw=dateobs_kw,
            )


    def get_mjd(
        self,
        file_path: Path | str | None = None,
        header: Header | None = None,
        ) -> float:
        
        if header is None:
            header: Header = fits.getheader(file_path)
        
        instrument_time = str(header[self.dateobs_kw])
        date, time = instrument_time.split(' ')
        fits_time = f'{date}T{time}'
        mjd = float(np.asarray(Time(fits_time, format="fits").mjd))
        
        return mjd





# TODO: check when this is needed
class OPTICAM_MX_UNKNOWN(OPTICAM_MX):


    def __init__(
        self,
        ):
        super().__init__(dateobs_kw='GPSTIME')




# TODO: fix this instrument
class OPTICAM_MX_2022(OPTICAM_MX):
    """
    Legacy OPTICAM-MX instrument. For use with OPTICAM-MX data taken in 2022. For OPTICAM-MX observations taken after
    2022, use `OPTICAM_MX` instead.
    """


    def get_mjd(
        self,
        file_path: Path | str | None = None,
        header: Header | None = None,
        ) -> float:
        
        if header is None:
            header: Header = fits.getheader(file_path)
        
        date = str(header['DATE-OBS'])
        time = str(header['UT'])
        mjd = float(np.asarray(Time(f'{date}T{time}', format='fits').mjd))
        
        return mjd