import os.path
from typing import Callable, Dict, List

from astropy.table import QTable
from astropy.visualization import simple_norm
from matplotlib.patches import Circle, Ellipse
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from numpy.typing import NDArray
import pandas as pd

from opticam.background.global_background import BaseBackground
from opticam.noise import characterise_noise, get_snrs
from opticam.photometers import get_growth_curve
from opticam.fitting.models import gaussian
from opticam.fitting.routines import fit_rms_vs_flux
from opticam.utils.constants import catalog_colors, fwhm_scale
from opticam.utils.helpers import sort_filters


def plot_catalogs(
    out_directory: str,
    stacked_images: Dict[str, NDArray],
    catalogs: Dict[str, QTable],
    show: bool,
    save: bool,
    ) -> None:
    """
    Plot the source catalogs.
    
    Parameters
    ----------
    out_directory : str
        The directory path to which the resulting plot will be saved.
    filters : List[str]
        The catalog filters.
    stacked_images : Dict[str, NDArray]
        The stacked images for each filter {filter: image}.
    show : bool
        Whether to show the plot.
    save : bool
        Whether to save the plot.
    """
    
    fig, axes = plt.subplots(
        ncols=len(stacked_images),
        tight_layout=True,
        figsize=(
            len(stacked_images) * 5,
            5,
            ),
        )
    
    if len(stacked_images) == 1:
        axes = [axes]
    
    for i, fltr in enumerate(stacked_images):
        
        plot_image = np.clip(stacked_images[fltr], 0, None)  # clip negative values to zero for better visualisation
        
        # plot stacked image
        axes[i].imshow(
            plot_image,
            origin="lower",
            cmap="Greys",
            interpolation="nearest",
            norm=simple_norm(
                plot_image,
                stretch="log",
                ),
            )
        
        # get aperture radius
        radius = 5 * np.median(catalogs[fltr]["semimajor_sigma"].value)  # type: ignore
        
        for j in range(len(catalogs[fltr])):
            # label sources
            axes[i].add_patch(
                Circle(
                    xy=(
                        catalogs[fltr]["xcentroid"][j],
                        catalogs[fltr]["ycentroid"][j],
                        ),  # type: ignore
                    radius=radius,
                    edgecolor=catalog_colors[j % len(catalog_colors)],
                    facecolor="none",
                    lw=1,
                    ),
                )
            axes[i].text(
                catalogs[fltr]["xcentroid"][j] + 1.05 * radius,
                catalogs[fltr]["ycentroid"][j] + 1.05 * radius,
                j + 1,  # source number
                color=catalog_colors[j % len(catalog_colors)],
                )
            
            # label plot
            axes[i].set_title(fltr, fontsize='large')
            axes[i].set_xlabel("X", fontsize='large')
            axes[i].set_ylabel("Y", fontsize='large')
    
    if save:
        fig.savefig(os.path.join(out_directory, "cat/catalogs.pdf"))
    
    if show:
        plt.show(fig)
    else:
        fig.clear()
        plt.close(fig)


def plot_time_between_files(
    out_directory: str,
    camera_files: Dict[str, List[str]],
    bmjds: Dict[str, float],
    show: bool,
    save: bool,
    ) -> None:
    """
    Plot the times between files. Useful for identifying gaps.
    
    Parameters
    ----------
    out_directory : str
        The directory path to which the resulting plot will be saved.
    camera_files : Dict[str, List[str]]
        The file paths separated by camera {filter: file paths}.
    bmjds : Dict[str, float]
        The file time stamps {file path: time stamp}.
    show : bool
        Whether to show the plot.
    save : bool
        Whether to save the plot.
    """
    
    fig, axes = plt.subplots(nrows=3, ncols=len(camera_files), tight_layout=True,
                                figsize=((2 * len(camera_files) / 3) * 6.4, 2 * 4.8), sharey='row')
    
    for fltr in list(camera_files.keys()):
        times = np.array([bmjds[file] for file in camera_files[fltr]])
        times -= times.min()
        times *= 86400  # convert to seconds from first observation
        dt = np.diff(times)  # get time between files
        file_numbers = np.arange(2, len(times) + 1, 1)  # start from 2 because we are plotting the time between files
        
        bin_edges = np.arange(int(dt.min()), np.ceil(dt.max() + .2), .1)  # define bins with width 0.1 s
        
        if len(camera_files) == 1:
            axes[0].set_title(fltr)
            
            # cumulative plot of time between files
            axes[0].plot(file_numbers, np.cumsum(dt), "k-", lw=1)
            
            # time between each file
            axes[1].plot(file_numbers, dt, "k-", lw=1)
            
            axes[2].hist(dt, bins=bin_edges, histtype="step", color="black", lw=1)
            axes[2].set_yscale("log")
            
            axes[0].set_ylabel("Cumulative time between files [s]")
            axes[0].set_xlabel("File number")
            
            axes[1].set_ylabel("Time between files [s]")
            axes[1].set_xlabel("File number")
            
            axes[2].set_xlabel("Time between files [s]")
        else:
            axes[0, list(camera_files.keys()).index(fltr)].set_title(fltr)
            
            # cumulative plot of time between files
            axes[0, list(camera_files.keys()).index(fltr)].plot(file_numbers, np.cumsum(dt), "k-", lw=1)
            
            # time between each file
            axes[1, list(camera_files.keys()).index(fltr)].plot(file_numbers, dt, "k-", lw=1)
            
            # histogram of time between files
            axes[2, list(camera_files.keys()).index(fltr)].hist(dt, bins=bin_edges, histtype="step", color="black", lw=1)
            axes[2, list(camera_files.keys()).index(fltr)].set_yscale("log")
            
            axes[0, 0].set_ylabel("Cumulative time between files [s]")
            axes[1, 0].set_ylabel("Time between files [s]")
            
            for col in range(len(camera_files)):
                axes[0, col].set_xlabel("File number")
                axes[1, col].set_xlabel("File number")
                axes[2, col].set_xlabel("Time between files [s]")
    
    for ax in axes.flatten():
        ax.minorticks_on()
        ax.tick_params(which="both", direction="in", top=True, right=True)
    
    if save:
        fig.savefig(os.path.join(out_directory, "diag/header_times.png"))
    
    if show:
        plt.show(fig)
    else:
        plt.close(fig)


def plot_backgrounds(
    out_directory: str,
    t_ref: float,
    show: bool,
    save: bool,
    ) -> None:
    """
    Plot the time-varying background for each camera.
    
    Parameters
    ----------
    out_directory : str
        The directory to which the background files, and where the resulting plot will be saved if `save=True`.
    t_ref : float
        The reference BMJD.
    show: bool
        Whether to display the plot.
    save : bool
        Whether to save the plot.
    """
    
    diag_files = os.listdir(os.path.join(out_directory, 'diag'))
    
    background_files = {}
    for file in diag_files:
        if file.endswith('_background.csv'):
            fltr = file.split('_')[0]
            background_files[fltr] = os.path.join(out_directory, f'diag/{file}')
    background_files = sort_filters(background_files)
    
    fig, axs = plt.subplots(
        nrows=2,
        ncols=len(background_files),
        tight_layout=True,
        figsize=(len(background_files) * 6.4, 1.5 * 4.8),
        sharex='col',
        gridspec_kw={
            'hspace': 0,
            },
        )
    
    # for each camera
    for i, (fltr, file) in enumerate(background_files.items()):
        df = pd.read_csv(file)
        
        # match times to background_median and background_rms keys
        t = np.asarray(df['BMJD'].values)
        plot_times = (t - t_ref) * 86400  # convert time to seconds from first observation
        
        if len(background_files) == 1:
            axs[0].set_title(fltr)
            axs[0].plot(plot_times, df['median'].values, "k.", ms=2)
            axs[1].plot(plot_times, df['rms'].values, "k.", ms=2)
            
            axs[1].set_xlabel(f"Time from BMJD {t_ref:.4f} [s]", fontsize='large')
            axs[0].set_ylabel("Median background RMS", fontsize='large')
            axs[1].set_ylabel("Median background", fontsize='large')
        else:
            # plot background
            axs[0, i].set_title(fltr, fontsize='large')
            axs[0, i].plot(plot_times, df['median'].values, "k.", ms=2)
            axs[1, i].plot(plot_times, df['rms'].values, "k.", ms=2)
            
            for col in range(len(background_files)):
                axs[1, col].set_xlabel(f"Time from BMJD {t_ref:.4f} [s]", fontsize='large')
            
            axs[0, 0].set_ylabel("Median background", fontsize='large')
            axs[1, 0].set_ylabel("Median background RMS", fontsize='large')
    
    for ax in axs.flatten():
        ax.minorticks_on()
        ax.tick_params(which="both", direction="in", top=True, right=True)
    
    if save:
        fig.savefig(os.path.join(out_directory, "diag/background.pdf"))
    
    if show:
        plt.show()
    else:
        fig.clear()
        plt.close(fig)


def plot_background_meshes(
    out_directory: str,
    images: Dict[str, NDArray],
    background: BaseBackground,
    show: bool,
    save: bool,
    ) -> None:
    """
    Plot the background mesh on top of some images.
    
    Parameters
    ----------
    out_directory : str
        The path to the output directory.
    images : Dict[str, NDArray]
        The images. The keys should give the filters (or file names) and the values should be the images.
    background: BaseBackground
        The background estimator.
    show : bool
        Whether to show the plot.
    save : bool
        Whether to save the plot.
    """
    
    ncols = len(images)
    fig, axes = plt.subplots(ncols=ncols, tight_layout=True, figsize=(ncols * 5, 5))
    
    if ncols == 1:
        # convert axes to list
        axes = [axes]
    
    for i, (label, image) in enumerate(images.items()):
        
        # clip negative values
        plot_image = np.clip(image, 0., None)
        
        bkg = background(image)
        
        # plot background mesh
        axes[i].imshow(
            plot_image,
            origin="lower",
            cmap="Greys",
            interpolation="nearest",
            norm=simple_norm(plot_image, stretch="log"),
            )
        bkg.plot_meshes(
            ax=axes[i],
            outlines=True,
            marker='.',
            color='red',
            alpha=0.3,
            )
        
        #label plot
        axes[i].set_title(label)
        axes[i].set_xlabel("X")
        axes[i].set_ylabel("Y")
    
    if save:
        fig.savefig(os.path.join(out_directory, 'diag/background_meshes.pdf'))
    
    if show:
        plt.show(fig)
    else:
        fig.clear()
        plt.close(fig)


def plot_growth_curves(
    image: NDArray,
    cat: QTable,
    targets: int | List[int],
    psf_params: Dict,
    ) -> Figure:
    """
    Plot the growth curves given a (stacked) image and a source catalog.
    
    Parameters
    ----------
    image : NDArray
        The image.
    cat : QTable
        The catalog corresponding to `image`.
    targets : int | List[int]
        The target(s) for which growth curves are to be computed.
    psf_params : Dict
        The PSF parameters.
    
    Returns
    -------
    Figure
        The growth curve plots.
    """
    
    def pix2sigma(x):
        return x / (psf_params['semimajor_sigma'] * fwhm_scale)
    
    def sigma2pix(x):
        return x * (psf_params['semimajor_sigma'] / fwhm_scale)
    
    if isinstance(targets, int):
        targets = [targets]
    
    n = len(targets)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    
    fig, axes = plt.subplots(
        nrows=rows,
        ncols=cols,
        figsize=(cols * 3, rows * 3),
        tight_layout=True,
        sharey='row',
    )
    
    if n > 1:
        for row in axes:
            row[0].set_ylabel('Flux [%]', fontsize='large')
    else:
        axes.set_ylabel('Flux [%]', fontsize='large')
    
    axes = np.asarray([axes]).flatten()
    
    for target in targets:
        i = target - 1  # index is just target ID - 1
        
        radii, fluxes = get_growth_curve(
            image=image,
            x_centroid=cat['xcentroid'][i],
            y_centroid=cat['ycentroid'][i],
            r_max = round(10 * psf_params['semimajor_sigma']),
        )
        
        axes[i].step(
            radii,
            100 * fluxes / np.max(fluxes),
            c='k',
            lw=1,
            where='mid',
            )
        
        secax = axes[i].secondary_xaxis('top', functions=(pix2sigma, sigma2pix))
        secax.set_xlabel('Radius [FWHM]', fontsize='large')
        secax.minorticks_on()
        secax.tick_params(which='both', direction='in')
        
        axes[i].set_title(f'Source {i + 1}', fontsize='large')
        axes[i].set_xlabel('Radius [pixels]', fontsize='large')
        
        axes[i].minorticks_on()
        axes[i].tick_params(which='both', direction='in', right=True)
    
    # delete empty subplots
    m = axes.size - n
    for i in range(1, m + 1):
        fig.delaxes(axes[-i])
    
    return fig


def plot_psf(
    catalog: QTable,
    source_indx: int,
    stacked_image: NDArray,
    fltr: str,
    a: float,
    b: float,
    out_directory: str,
    ) -> None:
    """
    Plot the PSF for given source.
    
    Parameters
    ----------
    catalog : QTable
        The source catalog.
    source_indx : int
        The index of the source in the catalog.
    stacked_image : NDArray
        The catalog image.
    fltr : str
        The filter.
    a : float
        The semimajor standard deviation of the PSF.
    b : float
        The semiminor standard deviation of the PSF.
    out_directory : str,
        The save path.
    """
    
    x_lo, x_hi = 0, stacked_image.shape[1]
    y_lo, y_hi = 0, stacked_image.shape[0]
    
    w = a * 10  # region width
    
    xc = catalog['xcentroid'][source_indx]
    yc = catalog['ycentroid'][source_indx]
    x_range = np.arange(max(x_lo, round(xc - w)), min(x_hi, round(xc + w)))  # x range
    y_range = np.arange(max(y_lo, round(yc - w)), min(y_hi, round(yc + w)))  # y range
    x_smooth = np.linspace(x_range[0], x_range[-1], 100)
    y_smooth = np.linspace(y_range[0], y_range[-1], 100)
    
    theta = catalog['orientation'].value[source_indx]
    theta_rad = theta * np.pi / 180
    
    # create mask
    mask = np.zeros_like(stacked_image, dtype=bool)
    for x_ in x_range:
        for y_ in y_range:
            mask[y_, x_] = True
    
    # isolate source
    rows_to_keep = np.any(mask, axis=1)
    region = stacked_image[rows_to_keep, :]
    cols_to_keep = np.any(mask, axis=0)
    region = region[:, cols_to_keep]
    
    fig, axes = plt.subplots(
        ncols=2,
        nrows=2,
        tight_layout=True,
        figsize=(6, 6),
        sharex='col',
        sharey='row',
        gridspec_kw={
            'hspace': 0,
            'wspace': 0,
            },
        )
    fig.delaxes(axes[0, 1])
    
    x, y = np.meshgrid(x_range, y_range)
    axes[1, 0].contour(
        x,
        y,
        region,
        5,
        colors='black',
        linewidths=1,
        zorder=1,
        linestyles='dashdot',
        )
    axes[1, 0].set_xlabel('X', fontsize='large')
    axes[1, 0].set_ylabel('Y', fontsize='large')
    axes[1, 0].add_patch(
        Ellipse(
            xy=(xc, yc),
            width=2 * fwhm_scale * a,  # in this parameterisation, the width is the semimajor axis
            height=2 * fwhm_scale * b,  # in this parameterisation, the height is the semiminor axis
            angle=theta,  # in this parameterisation, the angle is the orientation of the PSF
            facecolor='none',
            edgecolor='r',
            lw=1,
            ls='-',
            zorder=2,
            ),
        )
    
    # project PSF onto x, y axes
    xstd = np.sqrt(a**2 * np.cos(theta_rad)**2 + b**2 * np.sin(theta_rad)**2)
    ystd = np.sqrt(a**2 * np.sin(theta_rad)**2 + b**2 * np.cos(theta_rad)**2)
    
    axes[0, 0].step(
        x_range,
        100 * region[region.shape[0] // 2, :] / np.max(region[region.shape[0] // 2, :]),
        color='k',
        lw=1,
        where='mid',
        zorder=1,
        )
    axes[0, 0].plot(
        x_smooth,
        gaussian(x_smooth, 100, xc, xstd),
        'r-',
        lw=1,
        zorder=2,
    )
    axes[0, 0].set_ylabel('Peak flux [%]', fontsize='large')
    
    axes[1, 1].step(
        100 * region[:, region.shape[1] // 2] / np.max(region[:, region.shape[1] // 2]),
        y_range,
        color='k',
        lw=1,
        where='mid',
        )
    axes[1, 1].plot(
        gaussian(y_smooth, 100, yc, ystd),
        y_smooth,
        'r-',
        lw=1,
    )
    axes[1, 1].set_xlabel('Peak flux [%]', fontsize='large')
    
    for ax in axes.flatten():
        ax.minorticks_on()
        ax.tick_params(
            which='both',
            direction='in',
            right=True,
            top=True,
            )
    
    fig.suptitle(f'{fltr} Source {source_indx + 1}', fontsize='large')
    fig.savefig(
        os.path.join(
            out_directory,
            f'psfs/{fltr}_source_{source_indx + 1}.pdf',
            ),
        )
    plt.close(fig)


def plot_rms_vs_median_flux(
    lc_dir: str,
    save_dir: str,
    phot_label: str,
    show: bool = True,
    ) -> None:
    """
    Plot the RMS as a function of the median flux for all catalog sources.
    
    Parameters
    ----------
    lc_dir : str
        The light curve directory path.
    save_dir : str
        The output directory path.
    phot_label : str
        The photometry label.
    show : bool, optional
        Whether to show the plot, by default True.
    """
    
    data = get_lc_rms_and_flux_dict(
        lc_dir=lc_dir,
        )
    pl_fits = fit_rms_vs_flux(data)
    
    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        tight_layout=True,
        figsize=(15, 5),
        sharex='col',
        gridspec_kw={
            'hspace': 0,
            'height_ratios': [4, 1],
            },
        )
    
    for fltr in data.keys():
        if fltr in ['u-band', 'g-band']:
            ax1 = axes[0][0]
            ax2 = axes[1][0]
        elif fltr in ['r-band']:
            ax1 = axes[0][1]
            ax2 = axes[1][1]
        elif fltr in ['i-band', 'z-band']:
            ax1 = axes[0][2]
            ax2 = axes[1][2]
        else:
            raise ValueError(f'[OPTICAM] Unrecognised filter: {fltr}.')
        
        ax1.set_title(
            fltr,
            fontsize='large',
            )
        
        # plot model
        ax1.plot(
            pl_fits[fltr]['flux'],
            pl_fits[fltr]['rms'],
            color='blue',
            lw=1,
            )
        ax1.fill_between(
            pl_fits[fltr]['flux'],
            pl_fits[fltr]['rms'] - pl_fits[fltr]['err'],
            pl_fits[fltr]['rms'] + pl_fits[fltr]['err'],
            color='grey',
            edgecolor='none',
            alpha=.5,
            )
        
        # highlight potentially variable sources
        for source_number, values in data[fltr].items():
            i = np.where(pl_fits[fltr]['flux'] == values['flux'])[0]
            r = values['rms'] / pl_fits[fltr]['rms'][i]
            
            if r - 1 >= pl_fits[fltr]['err'][i] / pl_fits[fltr]['rms'][i]:
                color = 'red'
            else:
                color = 'black'
            
            ax1.scatter(
                values['flux'],
                values['rms'],
                marker='.',
                color=color,
                )
            ax1.text(
                values['flux'] * 1.03,
                values['rms'] * 1.03,
                str(source_number),
                color=color,
                )
            
            ax2.scatter(
                values['flux'],
                r,
                marker='.',
                color=color,
                )
            ax2.text(
                values['flux'] * 1.015,
                r * 1.015,
                str(source_number),
                fontsize='large',
                color=color,
                )
        
        ax1.set_yscale('log')
        ax1.set_ylabel(
            'Flux RMS [counts]',
            fontsize='large',
            )
        
        ax2.plot(
            pl_fits[fltr]['flux'],
            np.ones_like(pl_fits[fltr]['flux']),
            color='blue',
            lw=1,
            )
        ax2.fill_between(
            pl_fits[fltr]['flux'],
            1 - pl_fits[fltr]['err'] / pl_fits[fltr]['rms'],
            1 + pl_fits[fltr]['err'] / pl_fits[fltr]['rms'],
            color='grey',
            edgecolor='none',
            alpha=.5,
            )
        
        lo, hi = ax2.get_ylim()
        ax2.set_ylim(lo * 0.95, hi * 1.05)
        ax2.set_xlabel(
            'Median flux [counts]',
            fontsize='large',
            )
        ax2.set_ylabel(
            'RMS / model',
            fontsize='large',
            )
    
    for ax in axes.flatten():
        ax.set_xscale('log')
        ax.minorticks_on()
        ax.tick_params(which='both', direction='in', top=True, right=True)
    
    fig.savefig(os.path.join(save_dir, f'{phot_label}_rms_vs_median.pdf'))
    
    if show:
        plt.show(fig)
    else:
        plt.close(fig)

def get_lc_rms_and_flux_dict(
    lc_dir: str,
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Get the RMS and median flux for a series of light curves.
    
    Parameters
    ----------
    lc_dir : str
        The directory path to the light curves.
    
    Returns
    -------
    Dict[str, Dict[str, Dict[str, float]]]
        The median and RMS flux values for each light curve grouped by filter.
    """
    
    lcs = os.listdir(lc_dir)
    
    data = {}
    
    for lc in lcs:
        
        file_name, extension = lc.split('.')
        fltr, _, source_number = file_name.split('_')
        
        df = pd.read_csv(os.path.join(lc_dir, lc))
        
        flux = np.array(df['flux'].values, dtype=np.float64)
        flux = flux
        
        median = np.median(flux)
        rms = np.sqrt(np.mean(np.square(flux - np.mean(flux))))
        
        if not np.isnan(median) and not np.isnan(rms):
            if fltr not in data.keys():
                data[fltr] = {}
            source_info = {
                'rms': rms,
                'flux': median,
                }
            data[fltr][source_number] = source_info
    
    return data


def plot_snrs(
    out_directory: str,
    files: Dict[str, str],
    background: BaseBackground | Callable,
    psf_params: Dict[str, Dict[str, float]],
    catalogs: Dict[str, QTable],
    show: bool = False,
    ):
    """
    Plot the S/N for each source.
    
    Parameters
    ----------
    out_directory : str
        The output directory.
    files : Dict[str, str]
        The reference files for each filter {filter: path to image}.
    background : BaseBackground | Callable
        The global background estimator.
    psf_params : Dict[str, Dict[str, float]]
        The PSF parameters for each filter {filter: psf parameters}.
    catalogs : Dict[str, QTable]
        The catalogs for each filter {filter: catalog}.
    photometer : BasePhotometer
        The photometer to use for measuring noise.
    show : bool, optional
        Whether to show the plot, by default `False`.
    """
    
    fig, axes = plt.subplots(
        ncols=3,
        tight_layout=True,
        figsize=(15, 5),
        )
    
    for i, (fltr, file) in enumerate(files.items()):
        
        source_ids = np.arange(len(catalogs[fltr])) + 1  # source IDs
        snrs = np.round(
            get_snrs(
                file=file,
                background=background,
                catalog=catalogs[fltr],
                psf_params=psf_params[fltr],
                ),
            1,
            )
        
        axes[i].set_title(
            fltr,
            fontsize='large',
            )
        axes[i].set_xlabel(
            'Source ID',
            fontsize='large',
            )
        axes[i].set_ylabel(
            'S/N',
            fontsize='large',
            )
        
        p = axes[i].bar(
            source_ids,
            snrs,
            facecolor='none',
            edgecolor='k',
            lw=1,
            )
        axes[i].bar_label(
            p,
            padding=0.02 * axes[i].get_ylim()[1],
            fontsize='large',
            rotation=90,
            )
    
    for ax in axes:
        ax.set_ylim(ax.get_ylim()[0], 1.2 * ax.get_ylim()[1])
        ax.minorticks_on()
        ax.tick_params(which='both', direction='in', right=True, top=True)
    
    fig.savefig(
        os.path.join(out_directory, 'diag/snrs.pdf'),
        bbox_inches='tight',
        )
    
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_noise(
    out_directory: str,
    files: Dict[str, str],
    background: BaseBackground | Callable,
    psf_params: Dict[str, Dict[str, float]],
    catalogs: Dict[str, QTable],
    show: bool = False,
    ):
    """
    Plot the various noise contributions and compare them to the measured noise for a series of images.
    
    Parameters
    ----------
    out_directory : str
        The output directory.
    files : Dict[str, str]
        The reference files for each filter {filter: path to image}.
    background : BaseBackground | Callable
        The global background estimator.
    psf_params : Dict[str, Dict[str, float]]
        The PSF parameters for each filter {filter: psf parameters}.
    catalogs : Dict[str, QTable]
        The catalogs for each filter {filter: catalog}.
    photometer : BasePhotometer
        The photometer to use for measuring noise.
    show : bool, optional
        Whether to show the plot, by default `False`.
    """
    
    fig, axes = plt.subplots(
        ncols=3,
        nrows=2,
        tight_layout=True,
        sharex='col',
        gridspec_kw={
            'hspace': 0,
            'height_ratios': [4, 1],
            },
        figsize=(15, 5),
        )
    
    for i, (fltr, file) in enumerate(files.items()):
        
        results = characterise_noise(
            file=file,
            background=background,
            catalog=catalogs[fltr],
            psf_params=psf_params[fltr],
            )
        
        axes[0][i].plot(results['model_mags'], results['effective_noise'], label='Effective noise', c='k', lw=1, zorder=3)
        axes[0][i].plot(results['model_mags'], results['sky_noise'], ls='--', lw=1, label='Sky noise')
        axes[0][i].plot(results['model_mags'], results['shot_noise'], ls='--', lw=1, label='Shot noise')
        axes[0][i].plot(results['model_mags'], results['dark_noise'], ls='--', lw=1, label='Dark noise')
        axes[0][i].plot(results['model_mags'], results['read_noise'], ls='--', lw=1, label='Read noise')
        
        axes[0][i].scatter(
            results['measured_mags'],
            results['measured_noise'],
            label='Measured'
            )
        
        axes[1][i].axhline(
            1,
            c='k',
            lw=1,
            )
        axes[1][i].scatter(
            results['measured_mags'],
            results['measured_noise'] / results['expected_measured_noise'],
        )
        axes[1][i].fill_between(
            axes[1][i].set_xlim(),
            [1.05, 1.05],
            [.95, .95],
            color='grey',
            edgecolor='none',
            alpha=.5,
            )
        
        axes[0][i].set_yscale('log')
        axes[0][i].set_ylabel('$\\sigma_{\\rm mag}$', fontsize='large')
        axes[0][i].set_title(fltr, fontsize='large')
        
        axes[1][i].set_xlabel('-2.5 log(counts)', fontsize='large')
        axes[1][i].set_ylabel('$\\frac{\\sigma_{\\rm measured}}{\\sigma_{\\rm expected}}$', fontsize='xx-large')
    
    for ax in axes.flatten():
        ax.minorticks_on()
        ax.tick_params(which='both', direction='in', right=True, top=True)
    
    for ax in axes[0, :]:
        ax.invert_xaxis()
    
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(),
        bbox_to_anchor=(.5, .97),
        loc='lower center',
        ncol=6,
        bbox_transform=fig.transFigure,
        )
    
    fig.savefig(
        os.path.join(out_directory, 'diag/noise_characterisation.pdf'),
        bbox_inches='tight',
        )
    
    if show:
        plt.show()
    else:
        plt.close(fig)








