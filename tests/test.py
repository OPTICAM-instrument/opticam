import unittest
import numpy as np
import os
import tempfile
from astropy.io import fits

from opticam.background.global_background import DefaultBackground
from opticam.background.local_background import DefaultLocalBackground
from opticam.finders import DefaultFinder
from opticam.utils.generate import generate_observations


class TestBackground(unittest.TestCase):
    
    def test_background(self):
        """
        Test the default background estimator.
        """
        
        binning_scale = 4
        
        with tempfile.TemporaryDirectory() as temp_dir:
            
            dir_path = os.path.join(temp_dir, 'observations')
            
            if not os.path.isdir(dir_path):
                generate_observations(dir_path, n_images=5, circular_aperture=False, binning_scale=binning_scale)
            
            bkg_estimator = DefaultBackground(2048 // binning_scale // 16)
            
            for im in os.listdir(dir_path):
                    with fits.open(os.path.join(dir_path, im)) as hdul:
                        image = np.array(hdul[0].data)
                    
                    bkg = bkg_estimator(image)
                    
                    self.assertTrue(np.allclose(bkg.background, 100, rtol = .05))
                    self.assertTrue(np.allclose(bkg.background_rms, 10, rtol = .1))


class TestFinder(unittest.TestCase):
    
    def test_finder(self):
        """
        Test the source finder.
        """
        
        binning_scale = 4
        
        with tempfile.TemporaryDirectory() as temp_dir:
            
            dir_path = os.path.join(temp_dir, 'observations')
            
            if not os.path.isdir(dir_path):
                generate_observations(dir_path, n_images=5, circular_aperture=False, binning_scale=binning_scale)
            
            bkg_estimator = DefaultBackground(2048 // binning_scale // 16)
            finder = DefaultFinder(128 // binning_scale**2)
            
            for im in os.listdir(dir_path):
                with fits.open(os.path.join(dir_path, im)) as hdul:
                    image = np.array(hdul[0].data)
                
                bkg = bkg_estimator(image)
                tbl = finder(image - bkg.background, 5 * bkg.background_rms)
                
                self.assertTrue(len(tbl) == 6)


class TestLocalBackground(unittest.TestCase):
    
    def test_elliptical_local_background(self):
        """
        Test the local background estimator.
        """
        
        binning_scale = 4
        
        with tempfile.TemporaryDirectory() as temp_dir:
            
            dir_path = os.path.join(temp_dir, 'observations')
            
            if not os.path.isdir(dir_path):
                generate_observations(dir_path, n_images=5, circular_aperture=False)
            
            bkg_estimator = DefaultBackground(2048 // binning_scale // 16)
            finder = DefaultFinder(128 // binning_scale**2)
            
            for im in os.listdir(dir_path):
                with fits.open(os.path.join(dir_path, im)) as hdul:
                    image = np.array(hdul[0].data)
                
                bkg = bkg_estimator(image)
                tbl = finder(image - bkg.background, 5 * bkg.background_rms)
                coords = np.array([tbl["xcentroid"], tbl["ycentroid"]]).T
                local_bkg_estimator = DefaultLocalBackground()
                
                for i in range(len(coords)):
                    local_bkg, local_bkg_error = local_bkg_estimator(
                        image,
                        coords[i],
                        tbl['semimajor_sigma'][i].value,
                        tbl['semiminor_sigma'][i].value,
                        tbl['orientation'][i],
                        )
                    
                    self.assertTrue(np.allclose(local_bkg, 100, rtol = 0.05))
                    self.assertTrue(np.allclose(local_bkg_error, 10, rtol = 0.2))