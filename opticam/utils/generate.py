from astropy.io import fits
import os
from tqdm import tqdm
import numpy as np
from numpy.typing import NDArray
from typing import Tuple

from opticam.utils.constants import bar_format


FILTERS = ["g", "r", "i"]
N_SOURCES = 6

RMS = 0.02  # fractional RMS of variability
FREQ = 0.135  # frequency of variability

# variability phase lags
PHASE_LAGS = {
    'g': 0,
    'r': np.pi / 2,
    'i': np.pi,
}


def two_dimensional_gaussian(
    image: NDArray,
    x_centroid: float,
    y_centroid: float,
    flux: float,
    a: float,
    b: float,
    theta: float,
    ) -> NDArray:
    """
    Add a source to an image.
    
    Parameters
    ----------
    image : NDArray
        The image.
    x_centroid : float
        The x-coordinate of the source.
    y_centroid : float
        The y-coordinate of the source.
    flux : float
        The total flux of the source.
    a : float
        The semi-major standard deviation.
    b : float
        The semi-minor standard deviation.
    theta : float
        The rotation angle of the source.
    
    Returns
    -------
    NDArray
        The image with the source added.
    """
    
    theta_rad = theta * np.pi / 180
    
    y, x = np.mgrid[0:image.shape[0], 0:image.shape[1]]
    
    dx = x - x_centroid
    dy = y - y_centroid
    
    x_rot = dx * np.cos(theta_rad) + dy * np.sin(theta_rad)
    y_rot = -dx * np.sin(theta_rad) + dy * np.cos(theta_rad)
    
    amp = flux / (2 * np.pi * a * b)
    
    return amp * np.exp(-.5 * (np.square(x_rot / a) + np.square(y_rot / b)))

def variable_function(
    i: float,
    fltr: str,
    ) -> float:
    """
    Variable flux to be added to a source.
    
    Parameters
    ----------
    i : float
        The image index (equivalent to time).
    fltr : str
        The filter of the image, used to introduce a lag between filters.
    
    Returns
    -------
    float
        The flux.
    """
    
    amp = RMS * np.sqrt(2)  # convert RMS amplitude to peak amplitude
    
    return 1 + amp * np.sin(2 * np.pi * i * FREQ + PHASE_LAGS[fltr])

def create_image(
    binning_scale: int,
    ) -> NDArray:
    """
    Create a base image with Poisson noise.
    
    Parameters
    ----------
    binning_scale : int
        The binning scale of the image.
    
    Returns
    -------
    NDArray
        The noisy image.
    """
    
    return np.zeros((int(2048 / binning_scale), int(2048 / binning_scale))) + 100

def add_noise(
    image: NDArray,
    i: int,
    ) -> NDArray:
    """
    Add Poisson noise to an image.
    
    Parameters
    ----------
    image : NDArray
        The image to add noise to.
    i : int
        The image index, used to seed the random number generator.
    
    Returns
    -------
    NDArray
        The noisy image.
    """
    
    rng = np.random.default_rng(i)
    
    return rng.normal(image, np.sqrt(image))

def create_images(
    out_dir: str,
    variable_source: int,
    source_positions: NDArray,
    fluxes: NDArray,
    i: int,
    binning_scale: int,
    circular_aperture: bool,
    overwrite: bool,
    ) -> None:
    """
    Create the ith image for each filter.
    
    Parameters
    ----------
    filters : List[str]
        The filters.
    N_sources : int
        The number of sources.
    variable_source : int
        The index of the variable source.
    source_positions : NDArray
        The positions of the sources.
    fluxes : NDArray
        The fluxes of the sources.
    i : int
        The image index (equivalent to time).
    binning_scale : int
        The binning scale of the image.
    circular_aperture : bool
        Whether to apply a circular aperture shadow to the image.
    overwrite : bool
        Whether to overwrite the image if it already exists.
    """
    
    for fltr in FILTERS:
        if os.path.isfile(f"{out_dir}/240101{fltr}{200000000 + i}o.fits.gz") and not overwrite:
            continue
        
        # generate image
        image = create_image(binning_scale)
        if circular_aperture:
            image = apply_flat_field(image)  # apply circular aperture shadow
        
        # PSF parameters (typical PSF stdev of ~6pix at 1x1 binning for good seeing)
        semimajor_sigma = 6 / (2048 / image.shape[0])
        semiminor_sigma = 6 / (2048 / image.shape[0])
        orientation = 0
        
        # (x, y) translations
        rng = np.random.default_rng(i)
        dx = 2048 // 512 * rng.normal() / binning_scale
        dy = 2048 // 512 * rng.normal() / binning_scale
        x_positions = source_positions[:, 0] + dx
        y_positions = source_positions[:, 1] + dy
        
        # put sources in the image
        for j in range(N_SOURCES):
            if j == variable_source:
                image += two_dimensional_gaussian(
                    image,
                    x_positions[j],
                    y_positions[j],
                    fluxes[j] * variable_function(i, fltr),  # add variable flux to the source
                    semimajor_sigma,
                    semiminor_sigma,
                    orientation,
                    )
            else:
                image += two_dimensional_gaussian(
                    image,
                    x_positions[j],
                    y_positions[j],
                    fluxes[j],
                    semimajor_sigma,
                    semiminor_sigma,
                    orientation,
                    )
        
        noisy_image = add_noise(image, i)  # add Poisson noise
        
        # create fits file
        hdu = fits.PrimaryHDU(noisy_image)
        hdu.header["FILTER"] = fltr
        hdu.header["BINNING"] = f'{binning_scale}x{binning_scale}'
        hdu.header["GAIN"] = 1.
        hdu.header["EXPOSURE"] = 1.
        hdu.header["DARKCURR"] = .1
        
        # create observation pointing
        hdu.header['RA'] = 0.
        hdu.header['DEC'] = 0.
        
        # create observation time
        hh = str(i // 3600).zfill(2)
        mm = str(i % 3600 // 60).zfill(2)
        ss = str(i % 60).zfill(2)
        hdu.header["UT"] = f"2024-01-01 {hh}:{mm}:{ss}"
        
        # save fits file
        hdu.writeto(
            f"{out_dir}/240101{fltr}{200000000 + i}o.fits.gz",
            overwrite=overwrite,
            )

def apply_flat_field(
    image: NDArray,
    ) -> NDArray:
    """
    Apply a circular aperture shadow to an image.
    
    Parameters
    ----------
    image : NDArray
        The image.
    
    Returns
    -------
    NDArray
        The image with a circular aperture shadow.
    """
    
    # define mask to apply circular aperture
    x_mid, y_mid = image.shape[1] // 2, image.shape[0] // 2
    distance_from_centre = np.sqrt((x_mid - np.arange(image.shape[1]))**2 +
                                   (y_mid - np.arange(image.shape[0]))[:, np.newaxis]**2)
    radius = image.shape[0] // 2
    mask = distance_from_centre >= radius
    
    # create circular aperture shadow
    falloff = 1 / (distance_from_centre[mask] / radius)**2
    
    # apply circular aperture shadow
    image[mask] *= falloff
    
    return image

def create_flats(
    out_dir: str,
    filters: list,
    i: int,
    binning_scale: int,
    overwrite: bool,
    ) -> None:
    """
    Create the ith flat-field image for each filter. 
    
    Parameters
    ----------
    out_dir : str
        The directory to save the flat-field images.
    filters : list
        The filters to create flat-field images for.
    i : int
        The index of the flat-field image (equivalent to time).
    binning_scale : int
        The binning scale of the flat-field image.
    overwrite : bool
        Whether to overwrite the flat-field image if it already exists.
    """
    
    for fltr in filters:
        
        if os.path.isfile(f"{out_dir}/{fltr}-band_image_{i}.fits.gz") and not overwrite:
            continue
        
        # create flat-field image
        image = create_image(binning_scale) * 100  # mean count rate = 100 * 100 = 10,000
        image = apply_flat_field(image)  # apply circular aperture shadow
        noisy_image = add_noise(image, 123 * (i + 123))  # ensure different noise from observation images
        
        # create fits file
        hdu = fits.PrimaryHDU(noisy_image)
        hdu.header["FILTER"] = fltr
        hdu.header["BINNING"] = f'{binning_scale}x{binning_scale}'
        hdu.header["GAIN"] = 1.
        
        # create observation time
        hh = str(i // 3600).zfill(2)
        mm = str(i % 3600 // 60).zfill(2)
        ss = str(i % 60).zfill(2)
        hdu.header["UT"] = f"2024-01-01 {hh}:{mm}:{ss}"
        
        # save fits file
        hdu.writeto(f"{out_dir}/{fltr}-band_flat_{i}.fits.gz", overwrite=overwrite)

def generate_flats(
    out_dir: str,
    n_flats: int = 5,
    binning_scale: int = 4,
    overwrite: bool = False,
    ) -> None:
    """
    Create synthetic flat-field images.
    
    Parameters
    ----------
    out_dir : str
        The directory to save the data.
    n_flats : int, optional
        The number of flats per camera, by default 5.
    binning_scale : int, optional
        The binning scale of the flat-field images, by default 4 (512x512).
    overwrite : bool, optional
        Whether to overwrite data if they currently exist, by default False.
    """
    
    # create directory if it does not exist
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    
    filters = ["g", "r", "i"]
    
    for i in tqdm(range(n_flats), desc="Generating flats", bar_format=bar_format):
        create_flats(
            out_dir,
            filters,
            i,
            binning_scale,
            overwrite,
            )

def setup_obs(
    out_dir: str,
    binning_scale: int,
    ) -> Tuple[int, NDArray, NDArray]:
    """
    Configure the dummy observation parameters.
    
    Parameters
    ----------
    out_dir : str
        The output directory.
    binning_scale : int
        The image binning scale.
    
    Returns
    -------
    Tuple[List[str], int, int, NDArray, NDArray]
        The variable source index, source positions, and peak fluxes.
    """
    
    # create directory if it does not exist
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    
    rng = np.random.default_rng(123)
    
    border = 2048 // (16 * binning_scale)
    source_positions = rng.uniform(border, 2048 // binning_scale - border, (N_SOURCES, 2))  # generate random source positions away from the edges
    
    random_fluxes = np.round(10**(3 + rng.random(N_SOURCES)))  # generate random fluxes (uniform in logarithm)
    fluxes = -np.sort(-random_fluxes)  # sort fluxes in descending order
    
    variable_source = 1  # index of the variable source
    
    print(f'[OPTICAM] variable source is at ({source_positions[variable_source][0]:.0f}, {source_positions[variable_source][1]:.0f})')
    print(f'[OPTICAM] variability RMS: {RMS} %')
    print(f'[OPTICAM] variability frequency: {FREQ} Hz')
    print('[OPTICAM] variability phase lags:')
    for fltr, lag in PHASE_LAGS.items():
        print(f'    [OPTICAM] {fltr}-band: {lag:.3f} radians')
    
    # print(f'[OPTICAM] source fluxes: {fluxes}')
    
    return variable_source, source_positions, fluxes

def generate_observations(
    out_dir: str,
    n_images: int = 100,
    circular_aperture: bool = True,
    binning_scale: int = 4,
    overwrite: bool = False,
    ) -> None:
    """
    Create synthetic observation data for testing and following the tutorials.
    
    Parameters
    ----------
    out_dir : str
        The directory to save the data.
    n_images : int, optional
        The number of images to create, by default 100.
    circular_aperture : bool, optional
        Whether to apply a circular aperture shadow to the images, by default True.
    binning_scale : int, optional
        The binning scale of the images, by default 4 (512x512).
    overwrite : bool, optional
        Whether to overwrite data if they currently exist, by default False.
    """
    
    variable_source, source_positions, fluxes = setup_obs(
        out_dir=out_dir,
        binning_scale=binning_scale,
        )
    
    for i in tqdm(range(n_images), desc="Generating observations", bar_format=bar_format):
        create_images(
            out_dir=out_dir,
            variable_source=variable_source,
            source_positions=source_positions,
            fluxes=fluxes,
            i=i,
            binning_scale=binning_scale,
            circular_aperture=circular_aperture,
            overwrite=overwrite,
            )


def generate_gappy_observations(
    out_dir: str,
    n_images: int = 1000,
    circular_aperture: bool = True,
    binning_scale: int = 4,
    overwrite: bool = False,
    ) -> None:
    """
    Create synthetic observation data for testing and following the tutorials.
    
    Parameters
    ----------
    out_dir : str
        The directory to save the data.
    n_images : int, optional
        The number of images to create, by default 100.
    circular_aperture : bool, optional
        Whether to apply a circular aperture shadow to the images, by default True.
    binning_scale : int, optional
        The binning scale of the images, by default 4 (512x512).
    overwrite : bool, optional
        Whether to overwrite data if they currently exist, by default False.
    """
    
    variable_source, source_positions, fluxes = setup_obs(
        out_dir=out_dir,
        binning_scale=binning_scale,
        )
    
    rng = np.random.default_rng(42)
    gap_probability = .02  # probability of skipping an image
    
    for i in tqdm(range(n_images), desc="Generating observations", bar_format=bar_format):
        
        # randomly skip some images to create gaps
        if rng.random() < gap_probability:
            # if an image is skipped, increase the probability of skipping the next one to create larger gaps
            gap_probability = .95
            continue
        else:
            gap_probability = .02  # reset the probability of skipping the next image
        
        create_images(
            out_dir=out_dir,
            variable_source=variable_source,
            source_positions=source_positions,
            fluxes=fluxes,
            i=i,
            binning_scale=binning_scale,
            circular_aperture=circular_aperture,
            overwrite=overwrite,
            )