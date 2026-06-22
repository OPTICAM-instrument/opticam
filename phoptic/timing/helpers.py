from astropy.coordinates import SkyCoord
from astropy.time import Time
import numpy as np
from numpy.typing import NDArray


from phoptic.instruments import Instrument


def apply_barycentric_correction(
    original_times: float | NDArray,
    sky_coords: SkyCoord,
    instrument: Instrument,
    ) -> NDArray:
    """
    Apply barycentric corrections to a time array.
    
    Parameters
    ----------
    times : float | NDArray
        The time(s) to correct.
    sky_coords : SkyCoord
        The coordinates corresponding to the image.
    
    Returns
    -------
    NDArray
        The corrected time(s).
    """
    
    # format the times
    times = Time(original_times, format='mjd', scale='utc', location=instrument.location)
    
    # compute light travel time to barycentre
    ltt_bary = times.light_travel_time(sky_coords)
    
    return np.asarray((times.tdb + ltt_bary).value)