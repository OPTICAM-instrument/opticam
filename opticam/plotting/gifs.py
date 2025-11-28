import os
from typing import Callable, Dict, List

from astropy.table import QTable
from astropy.visualization.mpl_normalize import simple_norm
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from PIL import Image
from skimage.transform import matrix_transform
from tqdm import tqdm

from opticam.background.global_background import BaseBackground
from opticam.correctors.flat_field_corrector import FlatFieldCorrector
from opticam.utils.constants import catalog_colors
from opticam.utils.fits_handlers import get_data


def create_gif_frame(
    file: str,
    out_directory: str,
    aperture_selector: Callable,
    catalog: QTable,
    fltr: str,
    gains: Dict[str, float],
    transforms: Dict[str, List[float]],
    reference_file: str,
    flat_corrector: FlatFieldCorrector | None,
    rebin_factor: int,
    remove_cosmic_rays: bool,
    background: BaseBackground,
    ) -> None:
    
    data = np.asarray(
        get_data(
            file=file,
            flat_corrector=flat_corrector,
            rebin_factor=rebin_factor,
            remove_cosmic_rays=remove_cosmic_rays,
            )[0]
        )
    
    file_name = file.split('/')[-1].split(".")[0]
    
    bkg = background(data)
    clean_data = data - bkg.background
    
    # clip negative values to zero for better visualisation
    plot_image = np.clip(clean_data, 0, None)
    
    fig, ax = plt.subplots(num=1, clear=True, tight_layout=True)
    
    ax.imshow(
        plot_image,
        origin="lower",
        cmap="Greys",
        interpolation="nearest",
        norm=simple_norm(plot_image, stretch="log"),
        )  # type: ignore
    
    # for each source
    for i in range(len(catalog)):
        
        source_position = (catalog["xcentroid"][i], catalog["ycentroid"][i])
        
        if file == reference_file:
            aperture_position = source_position
            ax.set_title(f'{file_name} (reference)', color='blue', fontsize='large')
        elif file in transforms:
            aperture_position = matrix_transform(source_position, transforms[file])[0]
            ax.set_title(f'{file_name} (aligned)', color='black', fontsize='large')
        else:
            aperture_position = source_position
            ax.set_title(f'{file_name} (unaligned)', color='red', fontsize='large')
        
        radius = 5 * aperture_selector(catalog["semimajor_sigma"].value)
        
        ax.add_patch(
            Circle(
                xy=(aperture_position),
                radius=radius,
                edgecolor=catalog_colors[i % len(catalog_colors)],
                facecolor="none",
                lw=1,
                ),
            )
        ax.text(
            aperture_position[0] + 1.05 * radius,
            aperture_position[1] + 1.05 * radius,
            str(i + 1),
            color=catalog_colors[i % len(catalog_colors)],
            )
        
        ax.set_xlabel('X', fontsize='large')
        ax.set_ylabel('Y', fontsize='large')
    
    fig.savefig(os.path.join(out_directory, f'diag/{fltr}_gif_frames/{file_name}.png'), bbox_inches='tight')

def compile_gif(
    out_directory: str,
    fltr: str,
    camera_files: Dict[str, List[str]],
    keep_frames: bool,
    verbose: bool,
    ) -> None:
    """
    Create a gif from the frames saved in out_directory.

    Parameters
    ----------
    fltr : str
        The filter.
    keep_frames : bool
        Whether to keep the frames after the gif is saved.
    """
    
    # load frames
    frames = []
    for file in camera_files[fltr]:
        try:
            frames.append(Image.open(os.path.join(out_directory, f'diag/{fltr}_gif_frames/{file.split('/')[-1].split(".")[0]}.png')))
        except:
            pass
    
    # save gif
    frames[0].save(
        os.path.join(
            out_directory,
            f'cat/{fltr}_images.gif',
            ),
        format='GIF',
        append_images=frames[1:],
        save_all=True,
        duration=200,
        loop=0,
        )
    
    # close images
    for frame in frames:
        frame.close()
    del frames
    
    if not keep_frames:
        # delete frames after gif is saved
        for file in tqdm(os.listdir(os.path.join(out_directory, f"diag/{fltr}_gif_frames")), disable=not verbose,
                            desc=f"[OPTICAM] Deleting {fltr} GIF frames"):
            os.remove(os.path.join(out_directory, f"diag/{fltr}_gif_frames/{file}"))