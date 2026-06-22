from pathlib import Path
from typing import Iterable

from astropy.io import fits
import numpy as np



def prep_hcam(
    rdata: Iterable,
    out_directory: Path | str,
    overwrite = False,
    RA: str | None = None,
    DEC: str | None = None,
    ) -> None:
    """
    Convert HiPERCAM data to a format that can be understood by `opticam`.
    
    Parameters
    ----------
    rdata : Iterable
        A `hipercam.hcam.Rdata` instance containing HiPERCAM data
    """
    
    out_directory = Path(out_directory)
    if not out_directory.is_dir():
        print(f'Creating {out_directory}')
        out_directory.mkdir(parents=True)
    
    for exposure, mccd in enumerate(rdata):
        for ccd_num, ccd in mccd.items():
            ccd_hdul = ccd.whdul()
            
            # skip bad times (i.e., every other u-band image)
            if not ccd_hdul[0].header['GOODTIME']:
                continue
            
            header = fits.Header()
            header['MJDINT'] = ccd_hdul[0].header['MJDINT']
            header['MJDFRAC'] = ccd_hdul[0].header['MJDFRAC']
            header['EXPTIME'] = ccd_hdul[0].header['EXPTIME']
            header['GAIN'] = '1'
            header['CCDSUM'] = ccd_hdul[0].header['CCDSUM']  # binning
            header['CCD'] = ccd_num  # identify corresponding CCD
            
            if RA is not None:
                header['RA'] = RA
            if DEC is not None:
                header['DEC'] = DEC
            
            # get image filter
            if ccd_num == '1':
                header['FILTER'] = 'u'
            elif ccd_num == '2':
                header['FILTER'] = 'g'
            elif ccd_num == '3':
                header['FILTER'] = 'r'
            elif ccd_num == '4':
                header['FILTER'] = 'i'
            elif ccd_num == '5':
                header['FILTER'] = 'z'
            
            windows = {}
            for ccd_hdu in ccd_hdul:
                windows[ccd_hdu.header['WINDOW']] = ccd_hdu.data.astype(np.uint16)
            
            if ccd_hdul[0].header['REFLECT']:
                u = np.hstack((windows['F1'], windows['E1']))
                l = np.hstack((windows['G1'], windows['H1']))
            else:
                u = np.hstack((windows['E1'], windows['F1']))
                l = np.hstack((windows['H1'], windows['G1']))
            
            data = np.vstack((u, l))
            data = np.flip(data, axis=1)
            
            fits.PrimaryHDU(data=data, header=header).writeto(
                out_directory / f'CCD_{ccd_num}_exposure_{exposure}.fits',
                overwrite=overwrite,
                )