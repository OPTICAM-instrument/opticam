from typing import Any, Dict, List
import re
import os


def camel_to_snake(
    string: str,
    ) -> str:
    """
    Convert a camelCase string to snake_case.
    
    Parameters
    ----------
    string : str
        The camelCase string to convert.
    
    Returns
    -------
    str
        The converted snake_case string.
    """
    
    return re.sub(r'(?<!^)(?=[A-Z])', '_', string).lower()


def sort_filters(
    d: Dict[str, Any],
    ) -> Dict[str, Any]:
    """
    Sort a dictionary whose keys are filter names in the order of the camera filters (e.g., u/g, r, i/z).
    
    Parameters
    ----------
    d : Dict[str, Any]
        A dictionary with filter names as keys.
    
    Returns
    -------
    Dict[str, Any]
        The sorted dictionary.
    """
    
    key_order = {
        'u-band': 0,
        "u'-band": 0,
        'g-band': 0,
        "g'-band": 0,
        "r-band": 1,
        "r'-band": 1,
        'i-band': 2,
        "i'-band": 2,
        'z-band': 2,
        "z'-band": 2,
        }
    
    return dict(sorted(d.items(), key=lambda x: key_order[x[0]]))


def create_file_paths(
    data_directory: None | str = None,
    c1_directory: None | str = None,
    c2_directory: None | str = None,
    c3_directory: None | str = None,
    ) -> List[str]:
    """
    Given some directories, get the paths to all available FITS files.
    
    Parameters
    ----------
    data_directory : None | str, optional
        The directory containing the FITS files of all three cameras, by default None.
    c1_directory : None | str, optional
        The directory containing the FITS files of Camera 1, by default None.
    c2_directory : None | str, optional
        The directory containing the FITS files of Camera 2, by default None.
    c3_directory : None | str, optional
        The directory containing the FITS files of Camera 3, by default None.
    
    Returns
    -------
    List[str]
        The file paths.
    """
    
    file_paths = []
    
    for directory in [data_directory, c1_directory, c2_directory, c3_directory]:
        if directory is not None:
            file_names = os.listdir(directory)
            for file_name in file_names:
                if '.fit' in file_name:
                    file_paths.append(os.path.join(directory, file_name))
    
    return file_paths











