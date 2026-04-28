from typing import Literal

import numpy as np
from numpy.typing import NDArray




def reshape_image(
    image: NDArray,
    factor: int,
    ) -> NDArray:
    
    shape = (image.shape[0] // factor, factor, image.shape[1] // factor, factor)
    
    return image.reshape(shape)


def pad_image(
    image: NDArray,
    factor: int,
    ) -> NDArray:
    """
    Pad an image by treating the boundaries as mirrors.
    
    Parameters
    ----------
    image : NDArray
        The image.
    factor : int
        The binning factor. The image is padded until it can be divided by `factor` with a remainder of zero.
    
    Returns
    -------
    NDArray
        The padded image.
    """
    
    index = -2
    while image.shape[0] % factor != 0:
        image = np.hstack((image, image[index, :]))
        index -= 1
    
    index = -2
    while image.shape[1] % factor != 0:
        image = np.vstack((image, image[:, index]))
        index -= 1
    
    return image


def rebin_image(
    image: NDArray,
    factor: int,
    method: Literal['median', 'sum'] = 'sum',
    ) -> NDArray:
    """
    Rebin an image in both dimensions.
    
    Parameters
    ----------
    image : NDArray
        The image to rebin.
    factor : int
        The factor to rebin by.
    method : Literal["median", "sum"], optional
        The rebinning method, by default "sum".
    
    Returns
    -------
    NDArray
        The rebinned image.
    
    Raises
    ------
    ValueError
        If the value of `method` is not recognised.
    """
    
    if method == 'sum':
        return rebin_sum(image=image, factor=factor)
    elif method == 'median':
        return median_filter(image=image, factor=factor)
    else:
        raise ValueError(f'[OPTICAM] Rebinning method {method} is not supported. Try method="median" or method="sum" instead.')


def rebin_sum(
    image: NDArray,
    factor: int,
    ) -> NDArray:
    """
    Rebin an image in both dimensions by summing.
    
    Parameters
    ----------
    image : NDArray
        The image to rebin.
    factor : int
        The factor to rebin by.
    
    Returns
    -------
    NDArray
        The rebinned image.
    
    Raises
    ------
    ValueError
        If the image cannot be downsampled by the desired factor.
    """
    
    if image.shape[0] % factor != 0 or image.shape[1] % factor != 0:
        raise ValueError(f'[OPTICAM] The dimensions of the input data must be divisible by the rebinning factor. Got shape {image.shape} and factor {factor}.')
    
    reshaped_data = reshape_image(image=image, factor=factor)
    
    return reshaped_data.sum(axis=(1, 3))


def median_filter(
    image: NDArray,
    factor: int,
    ) -> NDArray:
    """
    Rebin an image in both dimensions by taking the median. If the image cannot be downsampled by the desired factor,
    the boundaries will be treated as mirrors to pad the image. Inspired by Paez+2026:
    https://ui.adsabs.harvard.edu/abs/2026RASTI...5ag021P/abstract.
    
    Parameters
    ----------
    image : NDArray
        The image to rebin.
    factor : int
        The factor to rebin by.
    
    Returns
    -------
    NDArray
        The rebinned image.
    """
    
    if image.shape[0] % factor != 0 or image.shape[1] % factor != 0:
        image = pad_image(
            image=image,
            factor=factor,
            )
    
    reshaped_data = reshape_image(image=image, factor=factor)
    
    return np.median(reshaped_data, axis=(1, 3))


