from pathlib import Path
from typing import Dict, List, Literal, Tuple


from astropy.table import MaskedColumn, Table, QTable, vstack
from astropy.time import Time
from astropy.timeseries import aggregate_downsample, BinnedTimeSeries, LombScargle, LombScargleMultiband, TimeSeries
import astropy.units as u
from astropy.units.quantity import Quantity
import numpy as np
from numpy.typing import NDArray
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from stingray import Lightcurve


from opticam.utils.helpers import get_lc, sort_filters
from opticam.plotting.plots import plot_light_curves
from opticam.utils.time_helpers import infer_gtis




class Analyzer:
    """
    Helper class for analyzing OPTICAM light curves.
    """


    def __init__(
        self,
        out_directory: Path | str,
        light_curves: TimeSeries | None = None,
        norm: Literal['max', 'mean', 'none'] = 'mean',
        prefix: str | None = None,
        phot_label: str | None = None,
        show_plots: bool = True,
        ) -> None:
        """
        Helper class for analyzing OPTICAM light curves.
        
        Parameters
        ----------
        out_directory : Path
            The directory to save the output files (i.e., the same directory as `out_directory` used by
            `opticam_new.Reducer` when creating the light curves).
        light_curves : TimeSeries | None, optional
            The light curves to analyze, by default `None`.
        norm : Literal['max', 'mean', 'none'], optional
            The light curve normalisation, by default 'mean'. 'max' normalises the fluxes to a maximum flux of 1, 'mean'
            normalises the fluxes to a mean flux of 1, and 'none' applies no normalisation.
        prefix : str | None, optional
            The prefix to use for the output files (e.g., the name of the target source).
        phot_label : str, optional
            The label for the photometry routine used to generate the light curves, used in the output file names.
        show_plots : bool, optional
            Whether to render and show plots, by default `True`.
        """
        
        if light_curves:
            lc_cols = light_curves.colnames
            filter_cols: List[str] = [col for col in lc_cols if '_rel_flux_err' in col]
            filters = [col.replace('_rel_flux_err', '') for col in filter_cols]
            self.filters = sort_filters(list(set(filters)))
        else:
            self.filters = []
        
        self.norm = norm
        self.light_curves = validate_light_curves(
            light_curves,
            norm=self.norm,
            filters=self.filters,
            )
        self.t_ref = self.light_curves['time'].min()
        
        self.out_directory = Path(out_directory)
        if not self.out_directory.is_dir():
            self.out_directory.mkdir(parents=True)
        
        self.prefix = prefix
        self.phot_label = phot_label
        self.show_plots = show_plots
        
        if not self.out_directory.joinpath('plots').is_dir():
            self.out_directory.joinpath('plots').mkdir(parents=True)


    def join(
        self,
        analyzer: 'Analyzer',
        ) -> 'Analyzer':
        """
        Combine another `Analyzer` instance with the current one. If the new `Analyzer` has light curves with filters
        that are not present in the current `Analyzer`, those filters will be added. If the new `Analyzer` has light
        curves with filters that are already present in the current `Analyzer`, those light curves will be merged.
        
        Parameters
        ----------
        analyzer : Analyzer
            The analyzer instance being combined with the current one.
        
        Returns
        -------
        Analyzer
            A new `Analyzer` instance with the combined light curves.
        """
        
        assert analyzer.light_curves, f'[OPTICAM] cannot join an empty analyzer.'
        
        new_light_curves = vstack([self.light_curves, analyzer.light_curves])
        
        return Analyzer(
            out_directory=self.out_directory,
            light_curves=new_light_curves,
            norm=self.norm,
            prefix=self.prefix,
            phot_label=self.phot_label,
            show_plots=self.show_plots,
            )


    def rebin(
        self,
        time_bin_size: Quantity,
        method: Literal['mean', 'sum'] = 'mean',
        ) -> 'Analyzer':
        """
        Rebin the light curves, propagating errors accordingly. Returns a new `Analyzer` instance containing the binned
        light curves.
        
        Parameters
        ----------
        time_bin_size : Quantity
            The time bin size.
        method : Literal['mean', 'sum'], optional
            The type of binning, by default `'mean'`.
        
        Returns
        -------
        Analyzer
            A new `Analyzer` instance containing the binned light curves.
        """
        
        new_lcs = rebin(
            method=method,
            light_curves=self.light_curves,
            time_bin_size=time_bin_size,
            )
        
        return Analyzer(
            out_directory=self.out_directory,
            light_curves=new_lcs,
            norm=self.norm,
            prefix=self.prefix,
            phot_label=self.phot_label,
            show_plots=self.show_plots,
            )


    def plot(
        self,
        save: bool = True,
        return_fig: bool = False,
        ) -> Figure | None:
        """
        Plot the light curves.
        
        Parameters
        ----------
        save : bool, optional
            Whether to save the plot, by default `True`.
        return_fig : bool, optional
            Whether to return the resulting `Figure` instance, by default `False`. This can be used to make edits to the
            plot.
        
        Returns
        -------
        Figure | None
            The figure containing the light curves.
        """
        
        fig = plot_light_curves(
            filters=self.filters,
            light_curves=self.light_curves,
            t_ref=self.t_ref,
            y_label='Normalized flux',
            )
        
        if save:
            save_figure(
                fig=fig,
                path=self.out_directory.joinpath(f'plots/{self.prefix}_{self.phot_label}_light_curves.pdf'),
                )
        
        if self.show_plots:
            plt.show(fig)
        
        if return_fig:
            return fig
        else:
            fig.clear()
            plt.close(fig)


    def lomb_scargle(
        self,
        frequency: Quantity | None = None,
        scale: Literal['linear', 'semilogx', 'semilogy', 'loglog'] = 'linear',
        save: bool = True,
        return_fig: bool = False,
        ) -> Dict[str, LombScargle] | Tuple[Dict[str, LombScargle], Figure]:
        """
        Compute the Lomb-Scargle periodogram for each light curve.
        
        Parameters
        ----------
        frequency : Quantity | None, optional
            The frequency grid, by default `None`. If `None`, the `autofrequency()` method of `astropy`'s `LombScargle`
            class is used to generate a frequency grid.
        scale : Literal[&#39;linear&#39;, &#39;semilogx&#39;, &#39;semilogy&#39;, &#39;loglog&#39;], optional
            The scale for the resulting plot, by default `'linear'`.
        save : bool, optional
            Whether to save the resulting plot, by default `True`.
        return_fig : bool, optional
            Whether to return the figure, by default `False`. Useful if you want to edit the figure before saving.
        
        Returns
        -------
        Dict[str, LombScargle] | Tuple[Dict[str, LombScargle], Figure]
            If `return_fig=True`, the Lomb-Scargle periodograms and figure are returned. Otherwise, only the
            Lomb-Scargle periodograms are returned.
        """
        
        nrows: int = len(self.filters)
        
        fig, axes = plt.subplots(
            nrows=nrows,
            sharex=True,
            gridspec_kw={
                'hspace': 0,
                },
            figsize=(6.4, nrows * .5 * 4.8),
            tight_layout=True,
            )
        
        if nrows == 1:
            axes = [axes]
        
        lsps: Dict[str, LombScargle] = {}
        
        for i, fltr in enumerate(self.filters):
            
            lc = get_lc(self.light_curves, fltr)
            
            t = lc.time
            dt = np.min(np.diff(t))
            
            lsps[fltr] = LombScargle.from_timeseries(
                lc,
                signal_column_name=f'{fltr}_rel_flux',
                uncertainty=f'{fltr}_rel_flux_err',
                )
            
            if frequency is None:
                f = lsps[fltr].autofrequency(
                    maximum_frequency=.5 / dt,  # (pseudo-)Nyquist frequency
                    )
            else:
                f = frequency
            
            power = lsps[fltr].power(f)
            
            axes[i].step(
                f.to_value(u.Hz),
                power,
                c='k',
                lw=1,
                label=fltr,
                )
            
            axes[i].legend(
                handlelength=0,
                fontsize='x-large',
                frameon=False,
            )
            
            scale_ax(axes[i], scale)
        
        axes[-1].set_xlabel('Frequency [Hz]')
        
        if save:
            save_figure(
                fig=fig,
                path=self.out_directory.joinpath(f'plots/{self.prefix}_LSP.pdf'),
                )
        
        if self.show_plots:
            plt.show()
        
        if not return_fig:
            fig.clear()
            plt.close(fig)
            
            return lsps
        else:
            return lsps, fig


    def multiband_lomb_scargle(
        self,
        frequency: Quantity | None = None,
        scale: Literal['linear', 'semilogx', 'semilogy', 'loglog'] = 'linear',
        save: bool = True,
        return_fig: bool = False,
        ) -> LombScargleMultiband | Tuple[LombScargleMultiband, Figure]:
        """
        Compute the multiband Lomb-Scargle periodogram from all light curves.
        
        Parameters
        ----------
        frequency : Quantity | None, optional
            The frequency grid, by default `None`. If `None`, the `autofrequency()` method of `astropy`'s
            `LombScargleMultiband` class is used to generate a frequency grid.
        scale : Literal[&#39;linear&#39;, &#39;semilogx&#39;, &#39;semilogy&#39;, &#39;loglog&#39;], optional
            The scale for the resulting plot, by default `'linear'`.
        save : bool, optional
            Whether to save the resulting plot, by default `True`.
        return_fig : bool, optional
            Whether to return the figure, by default `False`. Useful if you want to edit the figure before saving.
        
        Returns
        -------
        LombScargleMultiband | Tuple[LombScargleMultiband, Figure]
            If `return_fig=True`, the multiband Lomb-Scargle periodogram and figure are returned. Otherwise, only the
            multiband Lomb-Scargle periodogram is returned.
        """
        
        fig, ax = plt.subplots(
            tight_layout=True,
            )
        
        # set maximum and minimum frequencies
        minimum_frequency = np.inf
        maximum_frequency = 0
        for fltr in self.filters:
            lc = get_lc(self.light_curves, fltr)
            lo = 1 / (lc.time.max() - lc.time.min())
            hi = .5 / np.min(np.diff(lc.time))
            if lo < minimum_frequency:
                minimum_frequency = lo
            if hi > maximum_frequency:
                maximum_frequency = hi
        
        lc_cols = self.light_curves.colnames
        signal_column = [col for col in lc_cols if 'flux' in col and not 'err' in col]
        uncertainty_column = [col for col in lc_cols if 'flux_err' in col]
        
        lsp = LombScargleMultiband.from_timeseries(
            timeseries=self.light_curves,
            signal_column=signal_column,
            uncertainty_column=uncertainty_column,
            band_labels=self.filters,
        )
        
        if frequency is None:
            f = lsp.autofrequency(
                minimum_frequency=minimum_frequency,
                maximum_frequency=maximum_frequency,
                )
        else:
            f = frequency
        
        power = lsp.power(f)
        
        ax.step(
            f.to_value(u.Hz),
            power,
            lw=1,
            c='k',
            )
        
        scale_ax(ax, scale)
        
        ax.set_xlabel('Frequency [Hz]')
        
        if save:
            save_figure(
                fig=fig,
                path=self.out_directory.joinpath(f'plots/{self.prefix}_multiband_LSP.pdf'),
                )
        
        if self.show_plots:
            plt.show()
        
        if not return_fig:
            fig.clear()
            plt.close(fig)
            
            return lsp
        else:
            return lsp, fig


    def fold(
        self,
        period: Quantity,
        epoch_time: Time | None = None,
        nbins: int | None = None,
        sharey: bool = False,
        save: bool = True,
        return_fig: bool = False,
        ) -> Table | Tuple[Table, Figure]:
        """
        Fold the light curves on the given period.
        
        Parameters
        ----------
        period : Quantity
            The period used to fold the light curves. Must have units of time.
        epoch_time : Time | None, optional
            The reference time that defines zero phase, by default `None`. If `None`, the first time light curve time
            value is used.
        nbins : int | None, optional
            Bin the folded light curve into this many bins, by default `None`. If `None`, no binning is performed.
        sharey : bool, optional
            Whether to share y axes, by default `False`.
        save : bool, optional
            Whether to save the figure, by default `True`.
        return_fig : bool, optional
            Whether to return the figure, by default `False`. Useful if you want to edit the figure before saving.
        
        Returns
        -------
        Table | Tuple[Table, Figure]
            If `return_fig=True`, the folded light curve and resulting figure are returned. Otherwise, just the folded
            time series is returned. The folded light curve is converted from a `TimeSeries` to a `Table` since the
            `fold()` method of `TimeSeries` replaces the time column with phase values, causing time formatting errors.
        """
        
        phase_bin = isinstance(nbins, int)
        
        nrows: int = len(self.filters)
        
        fig, axes = plt.subplots(
            nrows=nrows,
            sharex=True,
            sharey=sharey,
            gridspec_kw={
                'hspace': 0,
                },
            figsize=(6.4, nrows * .5 * 4.8),
            tight_layout=True,
            )
        
        if nrows == 1:
            axes = [axes]
        
        if epoch_time is None:
            epoch_time = self.t_ref
        
        folded_lcs = Table()
        
        for i, fltr in enumerate(self.filters):
            
            lc = get_lc(self.light_curves, fltr)
            folded_lc = Table(lc.fold(
                period=period,
                epoch_time=epoch_time - (period / 2),  # shift epoch time by half a period to account for 0.5 phase offset
                normalize_phase=True,
                ))
            
            if phase_bin:
                folded_lc = rebin(
                    method='mean',
                    light_curves=folded_lc,
                    nbins=nbins,
                    )
            
            folded_lc.rename_column(name='time', new_name='phase')
            folded_lcs = vstack([folded_lcs, folded_lc])
            
            # plot two periods for clarity
            phase = np.append(0.5 + folded_lc['phase'].value, 1.5 + folded_lc['phase'].value)
            flux = np.append(folded_lc[f'{fltr}_rel_flux'].value, folded_lc[f'{fltr}_rel_flux'].value)
            flux_err = np.append(folded_lc[f'{fltr}_rel_flux_err'].value, folded_lc[f'{fltr}_rel_flux_err'].value)
            
            axes[i].errorbar(
                phase,
                flux,
                flux_err,
                color='black',
                linestyle='none',
                marker='.' if not phase_bin else 'none',
                ms=2,
                ecolor='grey',
                elinewidth=1,
                label=fltr,
                )
            
            if phase_bin:
                axes[i].step(
                phase,
                flux,
                where='mid',
                color='k',
                lw=1,
                )
            
            leg = axes[i].legend(
                handlelength=0,
                fontsize='x-large',
                frameon=False,
                )
            for handle in leg.legend_handles:
                handle.set_visible(False)
        
        axes[-1].set_xlabel('Phase')
        axes[nrows // 2].set_ylabel('Normalized flux')
        
        if save:
            save_figure(
                fig=fig,
                path=self.out_directory.joinpath(f'plots/{self.prefix}_folded_P={period}.pdf'),
                )
        
        if self.show_plots:
            plt.show()
        
        folded_lcs = folded_lcs.group_by('phase').groups.aggregate(np.nanmax)
        
        if not return_fig:
            fig.clear()
            plt.close(fig)
            
            return folded_lcs
        else:
            return folded_lcs, fig


    def export_light_curves_to_stingray(self) -> Dict[str, Lightcurve]:
        """
        Export the light curves from an `astropy.timeseries.TimeSeries` table to a dictionary of `stingray.Lightcurve`
        instances.
        
        Returns
        -------
        Dict[str, Lightcurve]
            The light curves {filter: Lightcurve}.
        """
        
        lcs: Dict[str, Lightcurve] = {}
        
        for fltr in self.filters:
            lc = get_lc(
                light_curves=self.light_curves,
                fltr=fltr,
                )
            lcs[fltr] = convert_lc_to_stingray(
                lc=lc,
                fltr=fltr,
                t_ref=self.t_ref,
                )
        
        return lcs




def rebin(
    method: Literal['mean', 'sum'],
    light_curves: Table | TimeSeries | QTable,
    time_bin_size: Quantity | None = None,
    nbins: int | None = None,
    ) -> Table | TimeSeries | QTable:
    """
    Rebin an `astropy` `Table` or similar to a lower time resolution while propagating errors correctly.
    
    Parameters
    ----------
    method : Literal[&#39;mean&#39;, &#39;sum&#39;]
        The rebinning method. 
    light_curves : Table | TimeSeries | QTable
        The light curves being rebinned.
    time_bin_size : Quantity | None, optional
        The time resolution of the binned light curve, by default `None`. If a value it passed, it must have units of
        time.
    nbins : int | None, optional
        The desired number of light curve bins, by default `None`. This parameter does nothing if `time_bin_size` is
        defined.
    
    Returns
    -------
    Table | TimeSeries | QTable
        The rebinned light curve.
    
    Raises
    ------
    NotImplementedError
        If the value passed to `method` is not recognised.
    ValueError
        If neither `time_bin_size` or `nbins` are defined.
    """
    
    if method == 'mean':
        aggregate_func = aggregate_mean
    else:
        raise NotImplementedError(f'[OPTICAM] Rebinning light curves using method="sum" is not supported yet. We apologise for the inconvenience.')
    
    if time_bin_size is not None:
        binned_lcs: BinnedTimeSeries =  aggregate_downsample(
            time_series=light_curves,
            aggregate_func=aggregate_func,
            time_bin_size=time_bin_size,
            )
        new_lcs = convert_binned_timeseries_to_timeseries(binned_lc=binned_lcs)
    elif nbins is not None:
        # if binning a light curve into a specified number of bins, convert the light curve to a Table to prevent time
        # column issues
        lc_table = Table(light_curves)
        lc_table['bin'] = np.floor(lc_table['time'] * nbins).astype(int)
        new_lcs = lc_table.group_by('bin').groups.aggregate(aggregate_func)
        new_lcs.remove_column('bin')
    else:
        raise ValueError('[OPTICAM] Cannot rebin a light curve unless time_bin_size or nbins is defined.')
    
    return new_lcs


def convert_binned_timeseries_to_timeseries(
    binned_lc: BinnedTimeSeries,
    ) -> TimeSeries:
    """
    Convert an `astropy.timeseries.BinnedTimeSeries` into an `astropy.timeseries.TimeSeries`.
    
    Parameters
    ----------
    binned_lc : BinnedTimeSeries
        The binned light curve.
    
    Returns
    -------
    TimeSeries
        The binned light curve as an `astropy.timeseries.TimeSeries`.
    """
    
    # time values of new TimeSeries are in the middle of the bins
    new_lc = TimeSeries(
        time=binned_lc['time_bin_start'] + binned_lc['time_bin_size'] / 2,
        )
    for col in binned_lc.colnames:
        if col not in ['time_bin_start', 'time_bin_end', 'time_bin_size']:
            new_lc.add_column(binned_lc[col], name=col)
    
    return new_lc


def aggregate_mean(
    col: MaskedColumn,
    ) -> Quantity | float:
    """
    Aggregate a column of values using the mean. If the column represents error values, then the propagated error
    is returned.
    
    Parameters
    ----------
    col : MaskedColumn
        The column values.
    
    Returns
    -------
    Quantity | float
        The aggregated column values.
    """
    
    if 'err' in str(col.name):
        valid = np.isfinite(col)
        n = valid.sum()
        
        if n == 0:
            return np.nan
        else:
            return np.sqrt((col[valid]**2).sum()) / n
    else:
        return np.nanmean(col)


def tidy_light_curves(
    light_curves: TimeSeries,
    ) -> TimeSeries:
    """
    Tidy a light curve table. Groups rows that have the same time value and sorts by time.
    
    Parameters
    ----------
    light_curves : TimeSeries
        The light curve table.
    
    Returns
    -------
    TimeSeries
        The tidied light curve table.
    """
    
    return light_curves.group_by('time').groups.aggregate(np.nanmax)


def validate_light_curves(
    light_curves: TimeSeries | None,
    norm: Literal['max', 'mean', 'none'],
    filters: List[str],
    ) -> TimeSeries:
    
    validated_light_curves = TimeSeries()
    
    if light_curves:
        for fltr in filters:
            
            lc = get_lc(
                light_curves=light_curves,
                fltr=fltr,
                )
            
            # apply normalisation
            factor = get_norm_factor(norm, lc[f'{fltr}_rel_flux'].value)
            lc[f'{fltr}_rel_flux'] /= factor
            lc[f'{fltr}_rel_flux_err'] /= factor
            
            validated_light_curves = vstack([validated_light_curves, lc])
    
    return tidy_light_curves(validated_light_curves)


def get_norm_factor(
    norm: Literal['max', 'mean', 'none'],
    fluxes: NDArray,
    ) -> float:
    """
    Compute the specified normalisation factor for the given fluxes.
    
    Parameters
    ----------
    norm : Literal['max', 'mean', 'none'], optional
        The light curve normalisation. 'max' normalises the fluxes to a maximum flux of 1, 'mean' normalises the fluxes 
        to a mean flux of 1, and 'none' applies no normalisation.
    fluxes : NDArray
        The light curve fluxes.
    
    Returns
    -------
    float
        The normalisation factor.
    """
    
    if norm == 'none':
        return 1.
    elif norm == 'max':
        return float(np.nanmax(fluxes))
    elif norm == 'mean':
        return float(np.nanmean(fluxes))
    else:
        raise ValueError(f'[OPTICAM] norm={norm} is not supported.')


def scale_ax(
    ax: Axes,
    scale: Literal['linear', 'semilogx', 'semilogy', 'loglog'],
    ) -> None:
    """
    Set the scale(s) of an `Axes` based on `scale`.

    Parameters
    ----------
    ax : Axes
        The axis to be scaled.
    scale : Literal[&#39;linear&#39;, &#39;semilogx&#39;, &#39;semilogy&#39;, &#39;loglog&#39;]
        The desired scale.
    """
    
    if scale == 'linear':
        return
    if scale == 'semilogx' or scale == 'loglog':
        ax.set_xscale('log')
    if scale == 'semilogy' or scale == 'loglog':
        ax.set_yscale('log')


def save_figure(
    fig: Figure,
    path: Path,
    ) -> None:
    """
    Save a figure to the specified path.
    
    Parameters
    ----------
    fig : Figure
        The figure.
    path : Path
        The path.
    """
    
    fig.savefig(
        path,
        bbox_inches='tight',
        )
    print(f'[OPTICAM] Plot saved to {path}.')


def convert_lc_to_stingray(
    lc: TimeSeries,
    fltr: str,
    t_ref: Time,
    ) -> Lightcurve:
    """
    Convert a light curve from an `astropy.timeseries.TimeSeries` instance to a `stingray.Lightcurve` instance.
    
    Parameters
    ----------
    lc : TimeSeries
        The light curve.
    fltr : str
        The filter.
    t_ref : Time
        The reference time.
    
    Returns
    -------
    Lightcurve
        The light curve as a `stingray.Lightcurve` instance.
    """
    
    time = np.asarray((lc.time - t_ref).to_value(u.s))
    counts = np.asarray(lc[f'{fltr}_rel_flux'].value)
    err = np.asarray(lc[f'{fltr}_rel_flux_err'].value)
    
    gti = infer_gtis(time)
    
    return Lightcurve(
        time=time,
        counts=counts,
        err=err,
        gti=gti,
        err_dist='gauss',
        ).sort()



