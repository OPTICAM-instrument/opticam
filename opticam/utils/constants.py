import matplotlib.colors as mcolors
import numpy as np

# custom tqdm progress bar format
bar_format= '{l_bar}{bar}|[{elapsed}<{remaining}]'

# camera pixel scales
pixel_scales = {
    'u-band': 0.1397,  # camera 1
    "u'-band": 0.1397,  # camera 1
    'g-band': 0.1397,  # camera 1
    "g'-band": 0.1397,  # camera 1
    'r-band': 0.1406,  # camera 2
    "r'-band": 0.1406,  # camera 2
    'i-band': 0.1661,  # camera 3
    "i'-band": 0.1661,  # camera 3
    'z-band': 0.1661,  # camera 3
    "z'-band": 0.1661,  # camera 3
    }

# plotting colours for each filter
colors = {
    'u-band': 'tab:purple',  # camera 1
    "u'-band": 'tab:purple',  # camera 1
    'g-band': 'tab:green',  # camera 1
    "g'-band": 'tab:green',  # camera 1
    'r-band': 'tab:orange',  # camera 2
    "r'-band": 'tab:orange',  # camera 2
    'i-band': 'tab:olive',  # camera 3
    "i'-band": 'tab:olive',  # camera 3
    'z-band': 'tab:brown',  # camera 3
    "z'-band": 'tab:brown',  # camera 3
}

# stdev -> FWHM scale factor
fwhm_scale = 2 * np.sqrt(2 * np.log(2))

# factor for converting counts to magnitudes (~ 1.0857)
counts_to_mag_factor = 2.5 / np.log(10)

# colors for catalog source markers
catalog_colors = list(mcolors.TABLEAU_COLORS.keys())
catalog_colors.pop(catalog_colors.index("tab:brown"))
catalog_colors.pop(catalog_colors.index("tab:gray"))
catalog_colors.pop(catalog_colors.index("tab:purple"))
catalog_colors.pop(catalog_colors.index("tab:blue"))

n_read = 1.1  # [electrons/pixel] Andor Zyla 4.2 PLUS read noise @ 216 MHz