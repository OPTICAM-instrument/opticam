from numpy.typing import NDArray


def rebin_image(
    image: NDArray,
    factor: int,
    ) -> NDArray:
    """
    Rebin an image by a given factor in both dimensions.
    
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
        raise ValueError(f'[OPTICAM] The dimensions of the input data must be divisible by the rebinning factor. Got shape {image.shape} and factor {factor}.')
    
    # reshape the array to efficiently rebin
    shape = (image.shape[0] // factor, factor, image.shape[1] // factor, factor)
    reshaped_data = image.reshape(shape)
    
    # rebin image by summing over the new axes
    return reshaped_data.sum(axis=(1, 3))