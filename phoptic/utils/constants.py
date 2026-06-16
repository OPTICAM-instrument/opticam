import numpy as np




# custom tqdm progress bar format
bar_format= '{l_bar}{bar}|[{elapsed}<{remaining}]'


# stdev -> FWHM scale factor
fwhm_scale = 2 * np.sqrt(2 * np.log(2))


# factor for converting counts to magnitudes (~ 1.0857)
counts_to_mag_factor = 2.5 / np.log(10)


# colors for catalog source markers
catalog_colors = [
    'tab:blue',
    'tab:orange',
    'tab:green',
    'tab:red',
    'tab:brown',
    ]


# order for sorting filters in order of increasing wavelength
# TODO: include more filters
filter_order = {
    'u': 0,
    "u'": 0,
    'g': 1,
    "g'": 1,
    "r": 2,
    "r'": 2,
    'i': 3,
    "i'": 3,
    'z': 4,
    "z'": 4,
    }