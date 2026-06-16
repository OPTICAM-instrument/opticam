import os
from pathlib import Path
from typing import Callable, Dict, List


from astropy.table import QTable
from astropy.visualization.mpl_normalize import simple_norm
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from numpy.typing import NDArray
from PIL import Image
from skimage.transform import matrix_transform


from phoptic.background.global_background import BaseBackground
from phoptic.utils.constants import catalog_colors
from phoptic.mef_slice import MEFSlice
from phoptic.utils.fits_handlers import get_data
from phoptic.instruments import Instrument




def create_gif_frame(
    file: MEFSlice,
    out_directory: Path,
    aperture_selector: Callable,
    catalog: QTable,
    key: str,
    instrument: Instrument,
    transforms: Dict[str, List[float]],
    reference_file: MEFSlice,
    rebin_factor: int,
    image_filter: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None,
    background: BaseBackground,
    ) -> None:
    """
    Create an alignment GIF frame.
    
    Parameters
    ----------
    file : MEFSlice
        The `MEFSlice` instance representing the image.
    out_directory : Path
        The output directory.
    aperture_selector : Callable
        The aperture selector function.
    catalog : QTable
        The catalogue.
    key : str
        The camera:filter key.
    instrument : Instrument
        The instrument that produced the image.
    transforms : Dict[str, List[float]]
        The alignment transformations.
    reference_file : MEFSlice
        The `MEFSlice` instance representing the reference file.
    rebin_factor : int
        The image rebin factor.
    image_filter : Callable[[NDArray[np.float64]], NDArray[np.float64]] | None
        The kernel/filter to apply to the image.
    background : BaseBackground
        The background estimator.
    """
    
    data = get_data(
        file=file,
        instrument=instrument,
        dark_corrector=None,
        rebin_factor=rebin_factor,
        image_filter=image_filter,
        remove_cosmic_rays=False,  # not required, disable for improved performance
        )[0]
    
    file_name = f'{file.path.name} ext {file.ext}'
    
    bkg = background(data)
    clean_data = data - bkg.background
    
    # clip negative values for better visualisation
    plot_image = np.clip(clean_data, 0, None)
    
    fig, ax = plt.subplots(clear=True, tight_layout=True)
    
    ax.imshow(
        plot_image,
        origin="lower",
        cmap="Greys",
        interpolation="nearest",
        norm=simple_norm(plot_image, stretch="log"),  # type: ignore
        )
    
    # for each source
    for i in range(len(catalog)):
        source_position = (catalog["x_centroid"][i], catalog["y_centroid"][i])
        
        if file == reference_file:
            aperture_position = source_position
            ax.set_title(f'{file_name} (reference)', color='blue', fontsize='large')
        elif file.key in transforms:
            aperture_position = matrix_transform(source_position, transforms[file.key])[0]
            ax.set_title(f'{file_name} (aligned)', color='black', fontsize='large')
        else:
            aperture_position = source_position
            ax.set_title(f'{file_name} (unaligned)', color='red', fontsize='large')
        
        radius = 5 * aperture_selector(catalog["semimajor_axis"].value)  # type: ignore
        
        ax.add_patch(
            Circle(
                xy=(aperture_position),  # type: ignore
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
    
    fig.savefig(os.path.join(out_directory, f'diag/{key}_gif_frames/{file.path.name.split(".")[0]}_{file.ext}.png'), bbox_inches='tight')
    
    fig.clear()
    plt.close(fig)


def compile_gif(
    out_directory: Path,
    key: str,
    camera_files: Dict[str, List[MEFSlice]],
    keep_frames: bool,
    ) -> None:
    """
    Create a GIF file from the frames saved in `out_directory`.
    
    Parameters
    ----------
    out_directory : Path
        The output directory.
    key : str
        The camera:filter key.
    camera_files : Dict[str, List[MEFSlice]]
        The image files grouped by filter.
    keep_frames : bool
        Whether to keep the frames after the GIT file is saved.
    """
    
    # load frames
    frames = []
    for file in camera_files[key]:
        try:
            frames.append(Image.open(os.path.join(out_directory, f'diag/{key}_gif_frames/{file.path.name.split(".")[0]}_{file.ext}.png')))
        except:
            pass
    
    # save gif
    frames[0].save(
        os.path.join(
            out_directory,
            f'cat/{key}_images.gif',
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
        for file in os.listdir(os.path.join(out_directory, f"diag/{key}_gif_frames")):
            os.remove(os.path.join(out_directory, f"diag/{key}_gif_frames/{file}"))