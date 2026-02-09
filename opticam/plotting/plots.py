import os.path
from pathlib import Path
from typing import Any, Callable


from astropy import units as u
from astropy.units import Quantity
from astropy.table import QTable
from astropy.timeseries import TimeSeries
from astropy.visualization import simple_norm
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Ellipse, Rectangle
from matplotlib.figure import Figure
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from photutils.aperture import ApertureStats, BoundingBox


from opticam.background.global_background import BaseBackground
from opticam.correctors import BiasCorrector, DarkNoiseCorrector, FlatFieldCorrector
from opticam.instruments import Instrument
from opticam.noise import characterise_noise, get_snrs
from opticam.photometers import AperturePhotometer, get_growth_curve
from opticam.fitting.models import gaussian
from opticam.fitting.routines import fit_rms_vs_flux
from opticam.utils.constants import catalog_colors, fwhm_scale
from opticam.mef_slice import MEFSlice
from opticam.utils.helpers import get_lc, sort_dict_by_filters




def plot_catalogs(
    out_directory: Path,
    stacked_images: dict[str, NDArray],
    catalogs: dict[str, QTable],
    show: bool,
    save: bool,
    ) -> None:
    """
    Plot the source catalogs.
    
    Parameters
    ----------
    out_directory : Path
        The path to the directory in which the resulting plot will be saved.
    stacked_images : dict[str, NDArray]
        The stacked images for each filter {filter: image}.
    catalogs : dict[str, QTable]
        The source catalogs for each filter {filter: catalog}.
    show : bool
        Whether to show the plot.
    save : bool
        Whether to save the plot.
    """
    
    ncols: int = len(stacked_images)
    
    fig, axes = plt.subplots(
        ncols=ncols,
        tight_layout=True,
        figsize=(ncols * 5, 5),
        )
    
    if ncols == 1:
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
                fontsize='large',
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
    out_directory: Path,
    camera_files: dict[str, list[MEFSlice]],
    bmjds: dict[str, float],
    show: bool,
    save: bool,
    ) -> None:
    """
    Plot the times between files. Useful for identifying gaps.
    
    Parameters
    ----------
    out_directory : Path
        The directory path to which the resulting plot will be saved.
    camera_files : dict[str, list[MEFSlice]]
        The files separated by camera.
    bmjds : dict[str, float]
        The file time stamps {file path + extension: time stamp}.
    show : bool
        Whether to show the plot.
    save : bool
        Whether to save the plot.
    """
    
    ncols: int = len(camera_files)
    
    fig, axes = plt.subplots(
        nrows=3,
        ncols=ncols,
        tight_layout=True,
        figsize=((2 * ncols / 3) * 6.4, 2 * 4.8),
        sharey='row',
        gridspec_kw={
            'wspace': 0,
            },
        )
    
    for fltr in list(camera_files.keys()):
        times = np.array([bmjds[file.key] for file in camera_files[fltr]])
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
    out_directory: Path,
    t_ref: float,
    show: bool,
    save: bool,
    ) -> None:
    """
    Plot the time-varying background for each camera.
    
    Parameters
    ----------
    out_directory : Path
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
    background_files = sort_dict_by_filters(background_files)
    
    fig, axes = plt.subplots(
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
            axes[0].set_title(fltr)
            axes[0].plot(plot_times, df['median'].values, "k.", ms=2)
            axes[1].plot(plot_times, df['rms'].values, "k.", ms=2)
            
            axes[1].set_xlabel(f"Time from BMJD {t_ref:.4f} [s]", fontsize='large')
            axes[0].set_ylabel("Median background RMS", fontsize='large')
            axes[1].set_ylabel("Median background", fontsize='large')
        else:
            # plot background
            axes[0, i].set_title(fltr, fontsize='large')
            axes[0, i].plot(plot_times, df['median'].values, "k.", ms=2)
            axes[1, i].plot(plot_times, df['rms'].values, "k.", ms=2)
            
            for col in range(len(background_files)):
                axes[1, col].set_xlabel(f"Time from BMJD {t_ref:.4f} [s]", fontsize='large')
            
            axes[0, 0].set_ylabel("Median background", fontsize='large')
            axes[1, 0].set_ylabel("Median background RMS", fontsize='large')
    
    for ax in axes.flatten():
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
    out_directory: Path,
    images: dict[str, NDArray[np.float64]],
    background: BaseBackground,
    show: bool,
    save: bool,
    ) -> None:
    """
    Plot the background mesh on top a series of images.
    
    Parameters
    ----------
    out_directory : Path
        The path to the output directory.
    images : dict[str, NDArray[np.float64]]
        The images {string: image}
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
    targets: int | list[int],
    psf_params: dict,
    ) -> Figure:
    """
    Plot the growth curves given a (stacked) image and corresponding source catalog.
    
    Parameters
    ----------
    image : NDArray
        The image.
    cat : QTable
        The catalog corresponding to `image`.
    targets : int | list[int]
        The target(s) for which growth curves are to be computed.
    psf_params : dict
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
    
    if rows > 1 and cols > 1:
        for row in axes:
            row[0].set_ylabel('Flux [%]', fontsize='large')
    elif cols > 1:
        for col in axes:
            col.set_ylabel('Flux [%]', fontsize='large')
    else:
        axes.set_ylabel('Flux [%]', fontsize='large')
    
    axes = np.asarray([axes]).flatten()
    
    for target in targets:
        i = targets.index(target)
        
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
        
        axes[i].set_title(f'Source {target}', fontsize='large')
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
    key: str,
    a: float,
    b: float,
    out_directory: Path,
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
    key : str
        The camera:filter key.
    a : float
        The semimajor standard deviation of the PSF.
    b : float
        The semiminor standard deviation of the PSF.
    out_directory : Path,
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
    
    fig.suptitle(f'{key} Source {source_indx + 1}', fontsize='large')
    fig.savefig(
        os.path.join(
            out_directory,
            f'psfs/{key}_source_{source_indx + 1}.pdf',
            ),
        )
    plt.close(fig)


def plot_rms_vs_median_flux(
    lc_dir: Path,
    save_dir: Path,
    phot_label: str,
    show: bool = True,
    ) -> None:
    """
    Plot the RMS as a function of the median flux for all catalog sources.
    
    Parameters
    ----------
    lc_dir : Path
        The light curve directory path.
    save_dir : Path
        The output directory path.
    phot_label : str
        The photometry label.
    show : bool, optional
        Whether to show the plot, by default True.
    """
    
    data: dict[str, dict[str, dict[str, float]]] = get_lc_rms_and_flux_dict(lc_dir=lc_dir)
    pl_fits: dict[str, dict[str, NDArray[np.float64]]] = fit_rms_vs_flux(data)
    
    ncols: int = len(pl_fits)
    assert ncols > 0, f"[OPTICAM] No valid light curve files found in {lc_dir}."
    
    fig, axes = plt.subplots(
        nrows=2,
        ncols=ncols,
        tight_layout=True,
        figsize=(2 / 3 * ncols * 6.4, 4.8),
        sharex='col',
        sharey='row',
        squeeze=False,
        gridspec_kw={
            'hspace': 0,
            'wspace': 0,
            'height_ratios': [4, 1],
            },
        )
    
    for i, fltr in enumerate(data.keys()):
        ax1 = axes[0][i]
        ax2 = axes[1][i]
        
        if i == 0:
            ax1.set_ylabel(
                'Flux RMS [counts]',
                fontsize='large',
                )
            
            ax2.set_ylabel(
                '$\\frac{\\rm RMS}{\\rm model}$',
                fontsize='xx-large',
                )
        
        ax2.set_xlabel(
            'Median flux [counts]',
            fontsize='large',
            )
        
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
                fontsize='large',
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
    
    for ax in axes.flatten():
        ax.set_xscale('log')
        ax.minorticks_on()
        ax.tick_params(which='both', direction='in', top=True, right=True)
    
    fig.savefig(os.path.join(save_dir, f'{phot_label}_rms_vs_median.pdf'), bbox_inches='tight')
    
    if show:
        plt.show(fig)
    else:
        plt.close(fig)


def get_lc_rms_and_flux_dict(
    lc_dir: Path,
    ) -> dict[str, dict[str, dict[str, float]]]:
    """
    Get the RMS and median flux for a series of light curves.
    
    Parameters
    ----------
    lc_dir : Path
        The directory path to the light curves.
    
    Returns
    -------
    dict[str, dict[str, dict[str, float]]]
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
        if not np.isfinite(np.log10(median)):
            continue
        
        rms = np.std(flux)
        if not np.isfinite(np.log10(rms)):
            continue
        
        if fltr not in data.keys():
            data[fltr] = {}
        source_info = {
            'rms': rms,
            'flux': median,
            }
        data[fltr][source_number] = source_info
    
    return sort_dict_by_filters(data)


def plot_snrs(
    out_directory: Path,
    files: dict[str, MEFSlice],
    background: BaseBackground | Callable,
    psf_params: dict[str, dict[str, float]],
    catalogs: dict[str, QTable],
    instrument: Instrument,
    bias_corrector: BiasCorrector | None,
    dark_corrector: DarkNoiseCorrector | None,
    flat_corrector: FlatFieldCorrector | None,
    show: bool,
    save: bool,
    ):
    """
    Plot the S/N for each source.
    
    Parameters
    ----------
    out_directory : Path
        The output directory.
    files : dict[str, MEFSlice]
        The reference file for each filter.
    background : BaseBackground | Callable
        The global background estimator.
    psf_params : dict[str, dict[str, float]]
        The PSF parameters for each filter {filter: psf parameters}.
    catalogs : dict[str, QTable]
        The catalogs for each filter {filter: catalog}.
    instrument : Instrument
        The instrument that produced the data.
    bias_corrector : BiasCorrector | None
        The bias corrector.
    dark_corrector : DarkNoiseCorrector | None
        The dark noise corrector.
    flat_corrector : FlatFieldCorrector | None
        The flat-field corrector.
    show : bool
        Whether to show the plot.
    save : bool
        Whether to save the plot.
    """
    
    ncols: int = len(files)
    
    fig, axes = plt.subplots(
        ncols=ncols,
        tight_layout=True,
        figsize=(2 / 3 * ncols * 6.4, 5),
        )
    
    # in event of a single column, make axes subscriptable
    if ncols == 1:
        axes = [axes]
    
    for i, (fltr, file) in enumerate(files.items()):
        
        source_ids, snrs = np.round(
            get_snrs(
                file=file,
                background=background,
                catalog=catalogs[fltr],
                psf_params=psf_params[fltr],
                instrument=instrument,
                bias_corrector=bias_corrector,
                dark_corrector=dark_corrector,
                flat_corrector=flat_corrector,
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
    
    if save:
        fig.savefig(
            os.path.join(out_directory, 'diag/snrs.pdf'),
            bbox_inches='tight',
            )
    
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_noise(
    out_directory: Path,
    files: dict[str, MEFSlice],
    background: BaseBackground | Callable,
    psf_params: dict[str, dict[str, float]],
    catalogs: dict[str, QTable],
    instrument: Instrument,
    bias_corrector: BiasCorrector | None,
    dark_corrector: DarkNoiseCorrector,
    flat_corrector: FlatFieldCorrector | None,
    show: bool,
    save: bool,
    ):
    """
    Plot the various noise contributions and compare them to the measured noise for a series of images.
    
    Parameters
    ----------
    out_directory : Path
        The output directory.
    files : dict[str, MEFSlice]
        The reference files for each filter.
    background : BaseBackground | Callable
        The global background estimator.
    psf_params : dict[str, dict[str, float]]
        The PSF parameters for each filter {filter: psf parameters}.
    catalogs : dict[str, QTable]
        The catalogs for each filter {filter: catalog}.
    instrument : Instrument
        The instrument that produced the data.
    bias_corrector : BiasCorrector | None
        The bias corrector.
    dark_corrector : DarkNoiseCorrector
        The dark noise corrector.
    flat_corrector : FlatFieldCorrector | None
        The flat-field corrector.
    show : bool
        Whether to show the plot.
    save : bool
        Whether to save the plot.
    """
    
    ncols: int = len(files)
    
    fig, axes = plt.subplots(
        ncols=ncols,
        nrows=2,
        squeeze=False,
        tight_layout=True,
        sharex='col',
        sharey='row',
        gridspec_kw={
            'hspace': 0,
            'wspace': 0,
            'height_ratios': [4, 1],
            },
        figsize=(2 / 3 * ncols * 6.4, 5),
        )
    
    for i, (fltr, file) in enumerate(files.items()):
        
        results = characterise_noise(
            file=file,
            background=background,
            catalog=catalogs[fltr],
            psf_params=psf_params[fltr],
            instrument=instrument,
            bias_corrector=bias_corrector,
            dark_corrector=dark_corrector,
            flat_corrector=flat_corrector,
            )
        
        axes[0][i].plot(results['model_mags'], results['effective_noise'], label='Effective noise', c='k', lw=1, zorder=3)
        
        axes[0][i].plot(results['model_mags'], results['sky_noise'], ls=(5, (10, 3)), lw=1, label='Sky noise')
        axes[0][i].plot(results['model_mags'], results['shot_noise'], ls=(0, (5, 5)), lw=1, label='Shot noise')
        
        if np.any(results['bias'] > 0):
            axes[0][i].plot(results['model_mags'], results['bias'], ls=(0, (5, 1)), lw=1, label='Bias')
        
        if np.any(results['dark_noise'] > 0):
            axes[0][i].plot(results['model_mags'], results['dark_noise'], ls=(0, (3, 5, 1, 5)), lw=1, label='Dark noise')
        
        if np.any(results['flat'] > 0):
            axes[0][i].plot(results['model_mags'], results['flat'], ls=(0, (3, 1, 1, 1)), lw=1, label='Flat')
        
        axes[0][i].plot(results['model_mags'], results['read_noise'], ls=(0, (3, 5, 1, 5, 1, 5)), lw=1, label='Read noise')
        
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
        
        for j in range(len(results['measured_mags'])):
            axes[0][i].text(
                results['measured_mags'][j],
                results['measured_noise'][j] * 1.2,
                f'{j + 1}',
                ha='center',
                va='bottom',
                fontsize='large',
                )
            
            r = results['measured_noise'][j] / results['expected_measured_noise'][j]
            
            if r >= 1:
                axes[1][i].text(
                results['measured_mags'][j],
                r * 1.01,
                f'{j + 1}',
                ha='center',
                va='bottom',
                fontsize='large',
                )
            else:
                axes[1][i].text(
                results['measured_mags'][j],
                r * .99,
                f'{j + 1}',
                ha='center',
                va='top',
                fontsize='large',
                )
        
        axes[0][i].set_yscale('log')
        axes[0][i].set_title(fltr, fontsize='large')
        
        axes[1][i].set_xlabel('-2.5 log(counts)', fontsize='large')
    
    for ax in axes.flatten():
        ax.minorticks_on()
        ax.tick_params(which='both', direction='in', right=True, top=True)
    
    for ax in axes[0, :]:
        ax.invert_xaxis()
    
    axes[0, 0].set_ylabel('$\\sigma_{\\rm mag}$', fontsize='large')
    axes[1, 0].set_ylabel('$\\frac{\\sigma_{\\rm measured}}{\\sigma_{\\rm expected}}$', fontsize='xx-large')
    
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(),
        bbox_to_anchor=(.5, .97),
        loc='lower center',
        ncol=len(results),
        bbox_transform=fig.transFigure,
        fontsize='large',
        )
    
    if save:
        fig.savefig(
            os.path.join(out_directory, 'diag/noise_characterisation.pdf'),
            bbox_inches='tight',
            )
    
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_apertures(
    out_directory: Path,
    data: NDArray,
    cat: QTable,
    targets: list[int] | int,
    photometer: AperturePhotometer,
    psf_params: dict[str, float],
    key: str,
    show: bool,
    save: bool,
    ):
    """
    Plot the specified aperture over each target source.
    
    Parameters
    ----------
    out_directory : Path
        The output directory. Used to save the plot if `save=True`.
    data : NDArray
        The image data.
    cat : QTable
        The source catalog.
    targets : list[int] | int
        The target IDs to plot apertures for.
    photometer : AperturePhotometer
        The `AperturePhotometer` instance.
    psf_params : dict[str, float]
        The PSF parameters.
    key : str
        The camera:filter key.
    show : bool
        Whether to show the plot.
    save : bool
        Whether to save the plot. If true, the plot is saved to `out_directory/diag/apertures/fltr_apertures.pdf`.
    """
    
    if isinstance(targets, int):
        targets = [targets]
    
    n = len(targets)
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))
    
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(ncols * 3, nrows * 3),
        tight_layout=True,
    )
    
    axes = np.asarray([axes]).flatten()
    
    # delete axes that will not be used
    excess_axes = axes.size - n
    for i in range(1, 1 + excess_axes):
        fig.delaxes(axes[-i])
    
    region_size = get_max_region_size(
        targets=targets,
        photometer=photometer,
        data=data,
        cat=cat,
        psf_params=psf_params,
    )
    
    for i, target in enumerate(targets):
        cat_indx = target - 1
        position = [cat['xcentroid'][cat_indx], cat['ycentroid'][cat_indx]]
        theta = cat['orientation'][cat_indx].value
        
        aperture = photometer.get_aperture(
            position=position,
            psf_params=psf_params,
            )
        
        if photometer.local_background_estimator is not None:
            annulus_stats = photometer.local_background_estimator.get_stats(
                data=data,
                position=position,
                semimajor_axis=psf_params['semimajor_sigma'],
                semiminor_axis=psf_params['semiminor_sigma'],
                theta=theta * np.pi / 180,  # radians
                )
            bbox = annulus_stats.bbox
            
            padding = region_size // 10
        else:
            aperture_stats = ApertureStats(
                data=data,
                aperture=aperture,
            )
            bbox = aperture_stats.bbox
            
            padding = region_size // 4
        
        ixmin = max(0, bbox.ixmin - padding)
        ixmax = min(data.shape[1], bbox.ixmin + region_size + padding)
        dx = bbox.ixmin - ixmin
        
        iymin = max(0, bbox.iymin - padding)
        iymax = min(data.shape[0], bbox.iymin + region_size + padding)
        dy = bbox.iymin - iymin
        
        bbox = BoundingBox(
            ixmin=ixmin,
            ixmax=ixmax,
            iymin=iymin,
            iymax=iymax,
            )
        
        # get region of interest
        region = data[bbox.iymin:bbox.iymax, bbox.ixmin:bbox.ixmax]
        centre = (position[0] - bbox.ixmin, position[1] - bbox.iymin)  # centre of region
        
        axes[i].imshow(region, origin='lower', cmap='Greys', norm=simple_norm(region, stretch='log'))
        
        if photometer.local_background_estimator is not None:
            
            annulus_mask = np.asarray(annulus_stats.data_cutout).astype(bool)
            
            # factor of 2 since matplotlib assumes diameter
            annulus_inner_width = 2 * photometer.local_background_estimator.r_in_scale * psf_params['semimajor_sigma']
            annulus_outer_width = 2 * photometer.local_background_estimator.r_out_scale * psf_params['semimajor_sigma']
            annulus_inner_height = 2 * photometer.local_background_estimator.r_in_scale * psf_params['semiminor_sigma']
            annulus_outer_height = 2 * photometer.local_background_estimator.r_out_scale * psf_params['semiminor_sigma']
            
            for coord in np.argwhere(annulus_mask):
                row, col = coord
                
                # offset coords to fit within bbox region
                row += dy
                col += dx
                
                rect = Rectangle((col - 0.5, row - 0.5), 1, 1, linewidth=1, edgecolor='red', facecolor='none')
                circ = Circle((col, row), .1, linewidth=1, edgecolor='red', facecolor='none')
                axes[i].add_patch(rect)
                axes[i].add_patch(circ)
            
            inner_ellipse = Ellipse(centre,
                                    width=annulus_inner_width,
                                    height=annulus_inner_height,
                                    angle=theta,
                                    facecolor='none',
                                    edgecolor='blue',
                                    lw=1,
                                    ls='--',
                                    )
            axes[i].add_patch(inner_ellipse)
            
            outer_ellipse = Ellipse(centre,
                                    width=annulus_outer_width,
                                    height=annulus_outer_height,
                                    angle=theta,
                                    facecolor='none',
                                    edgecolor='blue',
                                    lw=1,
                                    ls='--',
                                    )
            axes[i].add_patch(outer_ellipse)
        
        aperture_ellipse = Ellipse(
            centre,
            width=2 * aperture.a,
            height=2 * aperture.b,
            angle=theta,
            facecolor='none',
            edgecolor='blue',
            lw=1,
            ls='-',
            )
        axes[i].add_patch(aperture_ellipse)
        
        axes[i].set_xlabel('X', fontsize='large')
        axes[i].set_ylabel('Y', fontsize='large')
        axes[i].set_title(f'Source {target}', fontsize='large')
    
    fig.suptitle(key)
    
    if save:
        save_path = os.path.join(out_directory, 'diag/apertures')
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        fig.savefig(os.path.join(save_path, f'{key}_apertures.pdf'))
    
    if show:
        plt.show(fig)
    else:
        fig.clear()
        plt.close(fig)


def get_max_region_size(
    targets: list[int],
    photometer: AperturePhotometer,
    data: NDArray[np.float64],
    cat: QTable,
    psf_params: dict[str, float],
    ) -> int:
    """
    Get the maximum region size for plotting apertures.
    
    Parameters
    ----------
    targets : list[int]
        The target source IDs.
    photometer : AperturePhotometer
        The `AperturePhotometer` instance.
    data : NDArray[np.float64]
        The image data.
    cat : QTable
        The source catalog.
    psf_params : dict[str, float]
        The PSF parameters.
    
    Returns
    -------
    int
        The maximum region size.
    """
    
    region_sizes = []
    
    for target in targets:
        i = targets.index(target)
        position = [cat['xcentroid'][i], cat['ycentroid'][i]]
        
        aperture = photometer.get_aperture(
            position=position,
            psf_params=psf_params,
            )
        
        if photometer.local_background_estimator is not None:
            annulus_stats = photometer.local_background_estimator.get_stats(
                data=data,
                position=position,
                semimajor_axis=psf_params['semimajor_sigma'],
                semiminor_axis=psf_params['semiminor_sigma'],
                theta=psf_params['orientation'],
                )
            
            bbox = annulus_stats.bbox
        else:
            aperture_stats = ApertureStats(
                data=data,
                aperture=aperture,
                )
            
            bbox = aperture_stats.bbox
        
        width = bbox.ixmax - bbox.ixmin
        height = bbox.iymax - bbox.iymin
        region_sizes.append(max(width, height))
    
    return max(region_sizes)


def plot_light_curves(
    keys: list[str],
    light_curves: TimeSeries,
    t_ref: Quantity | None,
    y_label: Any = None,
    ) -> Figure:
    """
    Plot a table of light curves using a dedicated subplot for each filter.
    
    Parameters
    ----------
    keys : list[str]
        The light curve camera:filter keys.
    light_curves : TimeSeries
        The light curves.
    t_ref : Quantity
        The reference time. Light curves are plotted in seconds from this reference time.
    y_label : Any, optional
        The y-axis label, by default `None`.
    
    Returns
    -------
    Figure
        The resulting figure.
    """
    
    nrows: int = len(keys)
    
    fig, axes = plt.subplots(
        nrows=nrows,
        figsize=(2 * 6.4, .5 * nrows * 4.8),
        tight_layout=True,
        sharex=True,
        gridspec_kw={
            "hspace": 0,
            },
        )
    
    if nrows == 1:
        axes = [axes]
    
    if t_ref is None:
        t_ref = light_curves.time.min()
    
    for i, key in enumerate(keys):
        
        lc = get_lc(light_curves, key=key)
        
        time = (lc['time'] - t_ref).to_value(u.s)
        flux = lc[f'{key}_rel_flux'].value
        flux_err = lc[f'{key}_rel_flux_err'].value
        
        axes[i].errorbar(
            time,
            flux,
            flux_err,
            marker='none',
            linestyle='none',
            ecolor='grey',
            elinewidth=1,
            alpha=.5,
            )
        axes[i].step(
            time,
            flux,
            where='mid',
            lw=1,
            color='k',
            )
        
        axes[i].plot(
                [],
                [],
                marker='none',
                linestyle='none',
                label=key,
            )
        
        axes[i].legend(
            handlelength=0,
            fontsize='x-large',
            frameon=False,
        )
    
    axes[-1].set_xlabel(f'Time from BMJD {t_ref.value:.4f} [s]', fontsize='large')
    
    if y_label is not None:
        axes[nrows // 2].set_ylabel(f'{y_label}', fontsize='large')
    
    for ax in axes:
        ax.minorticks_on()
        ax.tick_params(which='both', direction='in', top=True, right=True)
    
    return fig


