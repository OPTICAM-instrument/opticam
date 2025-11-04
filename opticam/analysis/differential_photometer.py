from typing import List, Tuple
import os
import pandas as pd
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from astropy.table import QTable
import json
from stingray import Lightcurve
from matplotlib.axes import Axes
from astropy.io import fits
from astroalign import find_transform

from opticam.analysis.analyzer import Analyzer
from opticam.utils.fits_handlers import get_stacked_images
from opticam.plotting.plots import plot_catalogs
from opticam.utils.time_helpers import infer_gtis


class DifferentialPhotometer:
    """
    Helper class for creating relative light curves from OPTICAM data.
    """
    
    def __init__(
        self,
        out_directory: str,
        show_plots: bool = True,
        ) -> None:
        """
        Helper class for creating relative light curves from OPTICAM data.
        
        Parameters
        ----------
        out_directory : str
            The path to the directory where output will be saved.
        show_plots : bool, optional
            Whether plots should be shown as they're generated, by default True.
        
        Raises
        ------
        FileNotFoundError
            If out_directory cannot be found.
        """
        
        ########################################### input params ###########################################
        
        self.out_directory = out_directory
        if not os.path.isdir(self.out_directory):
            raise FileNotFoundError(f'[OPTICAM] {self.out_directory} not found.')
        
        self.show_plots = show_plots
        
        ########################################### attributes ###########################################
        
        with open(os.path.join(self.out_directory, 'misc/reduction_parameters.json'), 'r') as file:
            input_parameters = json.load(file)
        self.filters = input_parameters['filters']
        self.t_ref = float(input_parameters['t_ref'])
        self.time_key = 'BMJD' if input_parameters['barycenter'] else 'MJD'
        
        # output filters
        print('[OPTICAM] Filters: ' + ', '.join(list(self.filters)))
        
        ########################################### read catalogs ###########################################
        
        self.catalogs = {}
        for fltr in self.filters:
            try:
                self.catalogs.update(
                    {
                        fltr: QTable.read(
                            os.path.join(self.out_directory, f'cat/{fltr}_catalog.ecsv'),
                            format='ascii.ecsv'),
                        },
                    )
            except:
                print(f'[OPTICAM] Could not load {os.path.join(self.out_directory, f'cat/{fltr}_catalog.ecsv')}, skipping ...')
                self.filters.remove(fltr)
                continue
        
        ########################################### plot catalogs ###########################################
        
        if show_plots:
            stacked_images = get_stacked_images(self.out_directory)
            
            plot_catalogs(
                out_directory=self.out_directory,
                stacked_images=stacked_images,
                catalogs=self.catalogs,
                show=show_plots,
                save=False,
                )
            
            plt.show()
    
    def get_relative_light_curve(
        self,
        fltr: str,
        target: int,
        comparisons: int | List[int],
        phot_label: str,
        prefix: str | None = None,
        match_other_cameras: bool = False,
        show_diagnostics: bool = True,
        ) -> Analyzer:
        """
        Compute the relative light curve for a target source with respect to one or more comparison sources. By default,
        the relative light curve is computed for a single filter. The relative light curve is saved to
        out_directory/relative_light_curves. To automatically match the target and comparison sources across the other
        two filters, set match_other_cameras to True. Note that this can incorrectly match sources, so it is recommended
        to manually check the results.
        
        Parameters
        ----------
        fltr : str
            The filter to compute the relative light curve for.
        target : int
            The catalog ID of the target source.
        comparisons : int | List[int]
            The catalog ID(s) of the comparison source(s).
        phot_label : str
            The photometry label, used for file reading and labelling.
        prefix : str, optional
            The prefix to use when saving the relative light curve (e.g., the target star's name), by default None.
        match_other_cameras : bool, optional
            Whether to match the target and comparison(s) IDs to the remaining catalog filters, by default `False`. If
            `True`, astroalign must be installed.
        show_diagnostics : bool, optional
            Whether to show diagnostic plots, by default True.
        
        Returns
        -------
        Analyzer
            An Analyzer object containing the relative light curve(s).
        """
        
        if not os.path.isdir(os.path.join(self.out_directory, 'relative_light_curves')):
            os.mkdir(os.path.join(self.out_directory, 'relative_light_curves'))
        
        # validate filter
        if fltr not in self.filters:
            raise ValueError('[OPTICAM] ' + fltr + ' is not a valid filter.')
        
        if isinstance(comparisons, int):
            # if a single comparison source is given, convert to list
            comparisons = [comparisons]
        
        relative_light_curves = {}
        
        if not match_other_cameras:
            # compute and plot relative light curve for single filter
            relative_light_curves[fltr] = self._compute_relative_light_curve(
                fltr,
                target,
                comparisons,
                prefix,
                phot_label,
                show_diagnostics,
                )
        else:
            # catalog of reference filter
            ref_cat = QTable.read(
                os.path.join(self.out_directory, f"cat/{fltr}_catalog.ecsv"),
                format="ascii.ecsv",
                )
            
            # source coords in reference filter catalog
            ref_coords = np.asarray([ref_cat["xcentroid"].value, ref_cat["ycentroid"].value]).T
            # subtract 1 to account for zero-indexing
            ref_target_coords = ref_coords[target - 1]
            ref_comparison_coords = [ref_coords[comp - 1] for comp in comparisons]
            
            for new_fltr in self.filters:
                if new_fltr == fltr:
                    # no matching necessary
                    relative_light_curves[fltr] = self._compute_relative_light_curve(
                        fltr,
                        target,
                        comparisons,
                        prefix,
                        phot_label,
                        show_diagnostics,
                        )
                else:
                    try:
                        new_target, new_comparisons = transform_IDs(
                            self.out_directory,
                            ref_coords,
                            ref_target_coords,
                            ref_comparison_coords,
                            new_fltr,
                            )
                        
                        print(f'[OPTICAM] {fltr} target ID {target} was matched to {new_fltr} target ID {new_target}')
                        for i in range(len(comparisons)):
                            print(f'[OPTICAM] {fltr} comparison ID {comparisons[i]} was matched to {new_fltr} comparison ID {new_comparisons[i]}')
                    except:
                        print(f'[OPTICAM] Could not match {new_fltr} sources to {fltr} sources. This can happen if many stars are not identified across all catalogs. Sometimes simply trying again can help (RNG is involved), but often increasing max_catalog_sources in Catalog.create_catalogs() will more reliably solve the issue.')
                    
                    relative_light_curves[new_fltr] = self._compute_relative_light_curve(
                        new_fltr,
                        new_target,
                        new_comparisons,
                        prefix,
                        phot_label,
                        show_diagnostics,
                        )
        
        return Analyzer(
            self.out_directory,
            light_curves=relative_light_curves,
            prefix=prefix,
            phot_label=phot_label,
            )
    
    def _compute_relative_light_curve(
        self,
        fltr: str,
        target: int,
        comparisons: List[int],
        prefix: str | None,
        phot_label: str,
        show_diagnostics: bool,
        ) -> Lightcurve | None:
        """
        Compute the relative light curve for a target source with respect to one or more comparison sources for a given
        filter.
        
        Parameters
        ----------
        fltr : str
            The filter to compute the relative light curve for.
        target : int
            The catalog ID of the target source.
        comparisons : List[int]
            The catalog ID(s) of the comparison source(s).
        prefix : str | None
            The prefix to use when saving the relative light curve (e.g., the target star's name), by default None.
        phot_label : str
            The photometry label, used for file reading and labelling.
        show_diagnostics : bool
            Whether to show diagnostic plots, by default True.
        
        Returns
        -------
        Lightcurve | None
            The relative light curve for the target source with respect to the comparison sources, or None if the light
            curve could not be computed.
        """
        
        # TODO: create functions to clean up this code
        
        # subdirectory where results will be saved
        light_curve_dir = f'lcs/{phot_label}'
        
        # get target data frame
        try:
            target_df = pd.read_csv(os.path.join(self.out_directory, f'{light_curve_dir}/{fltr}_source_{target}.csv'))
        except:
            print(
                f'[OPTICAM] Could not load {os.path.join(self.out_directory,
                f'{light_curve_dir}/{fltr}_source_{target}.csv')}, skipping ...',
                )
            return None
        
        # get comparison data frame(s)
        comp_dfs = []
        for comp in comparisons:
            path = os.path.join(self.out_directory, f'{light_curve_dir}/{fltr}_source_{comp}.csv')
            try:
                comparison_df = pd.read_csv(path)
            except:
                print(f'[OPTICAM] Could not load {path}, skipping ...')
                continue
            comp_dfs.append(comparison_df)
        
        # ensure all DataFrames have the same time column
        filtered_target_df, filtered_comp_dfs = filter_dataframes_to_common_time_column(
            target_df=target_df,
            comp_dfs=comp_dfs,
            time_key=self.time_key,
            )
        
        # plot diagnostic light curves for target and comparison source(s)
        for i, df in enumerate(filtered_comp_dfs):
            self._plot_diag(
                fltr,
                target,
                comparisons[i],
                filtered_target_df,
                df,
                phot_label,
                show_diagnostics,
            )
        
        if len(filtered_comp_dfs) > 1:
            # plot diagnostic light curves between comparison sources
            for i, df in enumerate(filtered_comp_dfs):
                for j, df2 in enumerate(filtered_comp_dfs):
                    if i != j:
                        self._plot_diag(
                            fltr,
                            comparisons[i],
                            comparisons[j],
                            df,
                            df2,
                            phot_label,
                            show_diagnostics,
                            )
        
        time = filtered_target_df[self.time_key].values
        
        # get total flux and error of comparison sources
        comp_fluxes = np.sum([df["flux"].values for df in filtered_comp_dfs], axis=0)
        comp_flux_errors = np.sqrt(np.sum([np.square(df["flux_err"].values) for df in filtered_comp_dfs], axis=0))
        
        # compute relative flux and error
        relative_flux = filtered_target_df["flux"].values / comp_fluxes
        relative_flux_error = relative_flux * np.abs(np.sqrt(np.square(filtered_target_df["flux_err"].values / filtered_target_df["flux"].values) + np.square(comp_flux_errors / comp_fluxes)))
        
        # mask non-finite values
        valid_mask = np.isfinite(relative_flux) & np.isfinite(relative_flux_error)
        time = time[valid_mask]
        relative_flux = relative_flux[valid_mask]
        relative_flux_error = relative_flux_error[valid_mask]
        
        # infer light curve GTIs
        gtis = infer_gtis(time, threshold=1.5)
        
        # save relative light curve to CSV
        df = pd.DataFrame({
            self.time_key: time,
            'rel_flux': relative_flux,
            'rel_flux_err': relative_flux_error,
        })
        df.to_csv(
            os.path.join(self.out_directory, f'relative_light_curves/{phot_label}/{prefix}_{fltr}_light_curve.csv'),
            index=False,
        )
        
        lc = Lightcurve(
            time,
            relative_flux,
            err=relative_flux_error,
            gti=gtis,
        )
        
        self._plot_relative_light_curve(
            lc,
            target,
            comparisons,
            prefix,
            fltr,
            phot_label,
        )
        
        return lc
    
    def _plot_relative_light_curve(
        self,
        relative_light_curve: Lightcurve,
        target: int,
        comparisons: List[int],
        prefix: str | None,
        fltr: str,
        phot_label: str,
        ax: Axes | None = None,
        ) -> None:
        """
        Plot the relative light curve for a target source with respect to one or more comparison sources for a given
        filter.
        
        Parameters
        ----------
        relative_light_curve : Lightcurve
            The relative light curve to plot.
        target : int
            The catalog ID of the target source.
        comparisons : List[int]
            The catalog ID(s) of the comparison source(s).
        prefix : str | None
            The prefix to use when saving the relative light curve (e.g., the target star's name), by default None.
        fltr : str
            The filter to plot the relative light curve for.
        phot_label : str
            The photometry label, used for file reading and labelling.
        ax : Axes, optional
            The axes to plot the relative light curve on, by default None. If None, a new figure and axes will be 
            created.
        """
        
        # convert time to seconds from t_ref
        time = np.asarray(relative_light_curve.time).copy()
        time -= self.t_ref
        time *= 86400
        
        if ax is None:
            fig, ax = plt.subplots(tight_layout=True)
            standalone_plot = True
        else:
            standalone_plot = False
        
        ax.errorbar(
            time,
            relative_light_curve.counts,
            np.abs(relative_light_curve.counts_err),
            marker='none',
            linestyle='none',
            ecolor="grey",
            elinewidth=1,
            )
        ax.step(
            time,
            relative_light_curve.counts,
            where='mid',
            color='k',
            lw=1,
            )
        
        if standalone_plot:
            ax.set_title(f'{fltr} (Source ID: {target}, Comparison ID(s): {', '.join([str(comp) for comp in comparisons])})')
            ax.set_xlabel(f"Time from {self.time_key} {self.t_ref:.4f} [s]")
            ax.set_ylabel("Relative flux")
            
            ax.minorticks_on()
            ax.tick_params(which='both', direction='in', top=True, right=True)
            
            fig.savefig(os.path.join(self.out_directory, f'relative_light_curves/{phot_label}/{prefix}_{fltr}_light_curve.png'))
            
            if self.show_plots:
                plt.show()
            else:
                fig.clear()
                plt.close(fig)
    
    def _plot_diag(
        self,
        fltr: str,
        comparison1: int,
        comparison2: int,
        comparison1_df: pd.DataFrame,
        comparison2_df: pd.DataFrame,
        phot_label: str,
        show: bool,
        ) -> None:
        """
        Plot the relative diagnostic light curve for two comparison sources for a given filter.
        
        Parameters
        ----------
        fltr : str
            The filter to compute the relative light curve.
        comparison1 : int
            The catalog ID of the first comparison source.
        comparison2 : int
            The catalog ID of the second comparison source.
        comparison1_df : pd.DataFrame
            The data frame of the first comparison source.
        comparison2_df : pd.DataFrame
            The data frame of the second comparison source.
        pho_label : str
            The photometry label.
        t_ref : float
            The time of the earliest observation (used for plotting the relative light curve in seconds from t_ref).
        show : bool
            Whether to show the diagnostic plot.
        """
        
        # convert time to seconds from t_ref
        time = (comparison1_df[self.time_key].values - self.t_ref) * 86400
        
        fig, axes = plt.subplots(
            nrows=3,
            tight_layout=True,
            sharex=True,
            figsize=(6.4, 2 * 4.8),
            gridspec_kw={
                "hspace": 0,
                },
            )
        
        # compute relative flux and error
        relative_flux = comparison1_df["flux"].values / comparison2_df["flux"].values
        relative_flux_error = relative_flux * np.abs(np.sqrt(np.square(comparison1_df["flux_err"].values / comparison1_df["flux"].values) + np.square(comparison2_df["flux_err"].values / comparison2_df["flux"].values)))
        
        ########################################### normalised light curves ###########################################
        
        axes[0].set_title(f'{fltr} Comparison ID: {comparison1}, Comparison ID: {comparison2}')
        axes[0].errorbar(
            time,
            comparison1_df["flux"] / comparison1_df["flux"].median(),
            np.abs(comparison1_df["flux_err"] / comparison1_df["flux"].median()),
            fmt="kx-",
            ms=5,
            elinewidth=1,
            label=f'Source {comparison1}',
            alpha=.5,
            )
        axes[0].errorbar(
            time,
            comparison2_df["flux"] / comparison2_df["flux"].median(),
            np.abs(comparison2_df["flux_err"] / comparison2_df["flux"].median()),
            fmt="r+-",
            ms=5,
            elinewidth=1,
            label=f'Source {comparison2}',
            alpha=.5,
            )
        axes[0].legend()
        axes[0].set_ylabel("Normalised raw flux [counts]")
        
        ########################################### residuals ###########################################
        
        axes[1].errorbar(
            time,
            comparison1_df["flux"] / comparison1_df["flux"].median() - comparison2_df["flux"] / comparison2_df["flux"].median(),
            np.sqrt(np.sum(
                [
                    np.square(comparison1_df["flux_err"].values / comparison1_df["flux"].values),
                    np.square(comparison2_df["flux_err"].values / comparison2_df["flux"].values),
                ],
                axis=0,
                ),
                    ),
            fmt='k.',
            ms=2,
            ecolor='grey',
            elinewidth=1,
            )
        axes[1].axhline(
            0,
            color='r',
            lw=1,
        )
        axes[1].set_ylabel('Residuals')
        
        ########################################### relative light curve ###########################################
        
        axes[2].errorbar(
            time,
            relative_flux / np.median(relative_flux),
            np.abs(relative_flux_error) / np.median(relative_flux),
            fmt="k.",
            ms=2,
            ecolor="grey",
            elinewidth=1,
            )
        axes[2].axhline(
            1,
            color='r',
            lw=1,
        )
        axes[2].set_xlabel(f"Time from {self.time_key} {self.t_ref} [s]")
        axes[2].set_ylabel("Normalised relative flux")
        
        ########################################### format plot ###########################################
        
        for ax in axes:
            ax.minorticks_on()
            ax.tick_params(which="both", direction="in", top=True, right=True)
        
        ########################################### save plot ###########################################
        
        save_dir = os.path.join(self.out_directory, f'relative_light_curves/{phot_label}/diag')
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        
        fig.savefig(os.path.join(save_dir, f'{fltr}_{comparison1}_{comparison2}_diag_light_curve.png'))
        
        ########################################### optionally show plot ###########################################
        
        if not show:
            fig.clear()
            plt.close(fig)


def filter_dataframes_to_common_time_column(
    target_df: pd.DataFrame,
    comp_dfs: List[pd.DataFrame],
    time_key: str,
    ) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
    """
    Get the matching times between a target data frame (light curve) and a list of comparison data frames (light 
    curves).
    
    Parameters
    ----------
    target_df : pd.DataFrame
        The data frame of the target source.
    comp_dfs : List[pd.DataFrame]
        The list of data frames of the comparison sources.
    time_key : str,
        The time key (either BMJD or MJD depending on whether Barycentric corrections were applied).
    
    Returns
    -------
    Tuple[pd.DataFrame, List[pd.DataFrame]]
        The filtered target data frame and the list of filtered comparison data frames.
    """
    
    # get time columns from all data frames
    time_columns = [target_df[time_key].values]
    time_columns.extend([df[time_key].values for df in comp_dfs])
    
    # get matching times between all data frames
    common_times = set(time_columns[0])
    for time_col in time_columns[1:]:
        common_times.intersection_update(time_col)
    common_times = sorted(common_times)
    
    # get matching times for target
    filtered_target_df = target_df[target_df[time_key].isin(common_times)]
    filtered_target_df.reset_index(drop=True, inplace=True)
    
    # get matching times for comparisons
    filtered_comp_dfs = [df[df[time_key].isin(common_times)] for df in comp_dfs]
    filtered_comp_dfs = [df.reset_index(drop=True) for df in filtered_comp_dfs]
    
    return filtered_target_df, filtered_comp_dfs


def transform_IDs(out_directory: str, ref_coords: NDArray, ref_target_coords: NDArray,
                  ref_comparison_coords: List[NDArray], fltr: str) -> Tuple[int, List[int]]:
    
    # get source positions in new filter
    cat = QTable.read(
        os.path.join(out_directory, f"cat/{fltr}_catalog.ecsv"),
        format="ascii.ecsv",
        )
    coords = np.asarray([cat["xcentroid"].value, cat["ycentroid"].value]).T
    
    # get star-to-star correspondence
    source_arr, ref_arr = find_transform(coords, ref_coords)[1]
    
    # get transformed coordinates for target and comparison(s)
    target_coords = source_arr[np.where(ref_arr == ref_target_coords)]
    comparison_coords = [source_arr[np.where(ref_arr == comp_coords)] for comp_coords in ref_comparison_coords]
    
    # get transformed IDs for target and comparison(s)
    new_target = int(np.where(coords == target_coords)[0][0]) + 1
    new_comparisons = [int(np.where(coords == comp_coords)[0][0]) + 1 for comp_coords in comparison_coords]
    
    return new_target, new_comparisons






