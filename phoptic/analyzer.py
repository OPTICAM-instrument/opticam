from pathlib import Path
from typing import Callable, Literal


from astropy.table import MaskedColumn, Table, QTable, vstack
from astropy.time import Time
from astropy.timeseries import aggregate_downsample, BinnedTimeSeries, LombScargle, LombScargleMultiband, TimeSeries
import astropy.units as u
from astropy.units.quantity import Quantity
import numpy as np
from numpy.typing import NDArray
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from stingray import Lightcurve


from phoptic.plotting.helpers import scale_ax
from phoptic.plotting.plots import plot_light_curves
from phoptic.timing.timeseries import get_lc, infer_gtis
from phoptic.utils.helpers import sort_filters
from phoptic.utils.helpers import save_figure




class Analyzer:
    """
    Helper class for analyzing OPTICAM light curves.
    """


    def __init__(
        self,
        out_directory: Path | str,
        light_curves: TimeSeries | None = None,
        norm: Literal['max', 'mean', 'none'] = 'none',
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
        
        self.norm = norm
        
        if light_curves is not None:
            lc_cols = light_curves.colnames
            filter_cols: list[str] = [col for col in lc_cols if '_rel_flux_err' in col]
            keys = [col.replace('_rel_flux_err', '') for col in filter_cols]
            self.keys = sorted(keys)
            
            self.light_curves = validate_light_curves(
            light_curves,
            norm=self.norm,
            keys=self.keys,
            )
            self.t_ref = self.light_curves['time'].min()
        else:
            self.keys = []
        
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
        
        assert analyzer.light_curves is not None, f'[OPTICAM] cannot join an empty analyzer.'
        
        if hasattr(self, 'light_curves'):
            new_light_curves = vstack([self.light_curves, analyzer.light_curves])
        else:
            new_light_curves = analyzer.light_curves
        
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
        method: Literal['mean'] = 'mean',
        ) -> 'Analyzer':
        """
        Rebin the light curves, propagating errors accordingly. Returns a new `Analyzer` instance containing the binned
        light curves. Rebinning uses a common reference time to ensure simultaneity between multiple light curves.
        
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
        
        binned_lcs = TimeSeries()
        
        for key in self.keys:
            # rebin light curves one at a time to ensure correct GTI handling
            lc = self.get_lc(key)
            binned_lc = rebin(
                method=method,
                light_curves=lc,
                time_bin_size=time_bin_size,
                time_bin_start=self.t_ref,
                )
            
            binned_lcs = vstack([binned_lcs, binned_lc])
        
        return Analyzer(
            out_directory=self.out_directory,
            light_curves=binned_lcs,
            norm=self.norm,
            prefix=self.prefix,
            phot_label=self.phot_label,
            show_plots=self.show_plots,
            )


    def get_lc(
        self,
        key: str,
        ) -> TimeSeries:
        """
        Return the light curve for a single key.
        
        Parameters
        ----------
        key : str
            The camera:filter key (e.g., "1:g" for camera 1 with a g filter).
        
        Returns
        -------
        TimeSeries
            The light curve for the key.
        """
        
        return get_lc(light_curves=self.light_curves, key=key)


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
            keys=self.keys,
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


    def fold(
        self,
        period: Quantity,
        epoch_time: Time | None = None,
        nbins: int | None = None,
        sharey: bool = False,
        save: bool = True,
        return_fig: bool = False,
        ) -> Table | tuple[Table, Figure]:
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
        Table | tuple[Table, Figure]
            If `return_fig=True`, the folded light curve and resulting figure are returned. Otherwise, just the folded
            time series is returned. The folded light curve is converted from a `TimeSeries` to a `Table` since the
            `fold()` method of `TimeSeries` replaces the time column with phase values, causing time formatting errors.
        """
        
        phase_bin = isinstance(nbins, int)
        
        nrows: int = len(self.keys)
        
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
        
        for i, key in enumerate(self.keys):
            lc = self.get_lc(key=key)
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
            flux = np.append(folded_lc[f'{key}_rel_flux'].value, folded_lc[f'{key}_rel_flux'].value)
            flux_err = np.abs(np.append(folded_lc[f'{key}_rel_flux_err'].value, folded_lc[f'{key}_rel_flux_err'].value))
            
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
                )
            
            if phase_bin:
                axes[i].step(
                phase,
                flux,
                where='mid',
                color='k',
                lw=1,
                )
            
            axes[i].plot(
                [],
                [],
                marker='none',
                linestyle='none',
                label=key,
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


    def lomb_scargle(
        self,
        frequency: Quantity | None = None,
        scale: Literal['linear', 'semilogx', 'semilogy', 'loglog'] = 'linear',
        save: bool = True,
        return_fig: bool = False,
        ) -> dict[str, LombScargle] | tuple[dict[str, LombScargle], Figure]:
        """
        Compute the Lomb-Scargle periodogram for each light curve.
        
        Parameters
        ----------
        frequency : Quantity | None, optional
            The frequency grid, by default `None`. If `None`, the `autofrequency()` method of `astropy`'s `LombScargle`
            class is used to generate a frequency grid.
        scale : Literal["linear", "semilogx", "semilogy", "loglog"], optional
            The scale for the resulting plot, by default `'linear'`.
        save : bool, optional
            Whether to save the resulting plot, by default `True`.
        return_fig : bool, optional
            Whether to return the figure, by default `False`. Useful if you want to edit the figure before saving.
        
        Returns
        -------
        dict[str, LombScargle] | tuple[dict[str, LombScargle], Figure]
            If `return_fig=True`, the Lomb-Scargle periodograms and figure are returned. Otherwise, only the
            Lomb-Scargle periodograms are returned.
        """
        
        nrows: int = len(self.keys)
        
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
        
        lsps: dict[str, LombScargle] = {}
        
        for i, key in enumerate(self.keys):
            lc = self.get_lc(key=key)
            
            t = lc.time
            dt = np.min(np.diff(t))
            
            lsps[key] = LombScargle.from_timeseries(
                lc,
                signal_column_name=f'{key}_rel_flux',
                uncertainty=f'{key}_rel_flux_err',
                )
            
            if frequency is None:
                f = lsps[key].autofrequency(
                    maximum_frequency=.5 / dt,  # (pseudo-)Nyquist frequency
                    )
            else:
                f = frequency
            
            power = lsps[key].power(f)
            
            axes[i].step(
                f.to_value(u.Hz),
                power,
                c='k',
                lw=1,
                where='mid',
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
        ) -> LombScargleMultiband | tuple[LombScargleMultiband, Figure]:
        """
        Compute the multiband Lomb-Scargle periodogram from all light curves.
        
        Parameters
        ----------
        frequency : Quantity | None, optional
            The frequency grid, by default `None`. If `None`, the `autofrequency()` method of `astropy`'s
            `LombScargleMultiband` class is used to generate a frequency grid.
        scale : Literal["linear", "semilogx", "semilogy", "loglog"], optional
            The scale for the resulting plot, by default `'linear'`.
        save : bool, optional
            Whether to save the resulting plot, by default `True`.
        return_fig : bool, optional
            Whether to return the figure, by default `False`. Useful if you want to edit the figure before saving.
        
        Returns
        -------
        LombScargleMultiband | tuple[LombScargleMultiband, Figure]
            If `return_fig=True`, the multiband Lomb-Scargle periodogram and figure are returned. Otherwise, only the
            multiband Lomb-Scargle periodogram is returned.
        """
        
        fig, ax = plt.subplots(
            tight_layout=True,
            )
        
        # set maximum and minimum frequencies
        minimum_frequency = np.inf
        maximum_frequency = 0
        for key in self.keys:
            lc = self.get_lc(key=key)
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
            band_labels=self.keys,
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
            where='mid',
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


    def export_light_curves_to_stingray(self) -> dict[str, Lightcurve]:
        """
        Export the light curves from an `astropy.timeseries.TimeSeries` table to a dictionary of `stingray.Lightcurve`
        instances.
        
        Returns
        -------
        dict[str, Lightcurve]
            The light curves {filter: Lightcurve}.
        """
        
        lcs: dict[str, Lightcurve] = {}
        
        for key in self.keys:
            lc = self.get_lc(key=key)
            lcs[key] = convert_lc_to_stingray(
                lc=lc,
                key=key,
                t_ref=self.t_ref,
                )
        
        return lcs




def rebin(
    method: Literal['mean'],
    light_curves: Table | TimeSeries | QTable,
    time_bin_size: Quantity | None = None,
    time_bin_start: Time | None = None,
    nbins: int | None = None,
    ) -> Table | TimeSeries | QTable:
    """
    Rebin an `astropy` `Table` or similar to a lower time resolution while propagating errors correctly.
    
    Parameters
    ----------
    method : Literal["mean", "sum"]
        The rebinning method. 
    light_curves : Table | TimeSeries | QTable
        The light curves being rebinned.
    time_bin_size : Quantity | None, optional
        The time resolution of the binned light curve, by default `None`.
    time_bin_start : Time | None
        The time bin start (useful for ensuring simultaneity between binned time series), by default `None`.
    nbins : int | None, optional
        The desired number of light curve bins, by default `None`. This parameter does nothing if `time_bin_size` is
        defined.
    
    Returns
    -------
    Table | TimeSeries | QTable
        The rebinned light curve.
    
    Raises
    ------
    ValueError
        If the value passed to `method` is not recognised.
    ValueError
        If neither `time_bin_size` or `nbins` are defined.
    """
    
    if method == 'mean':
        aggregate_func = aggregate_mean
    else:
        raise ValueError(f'[OPTICAM] Only method="mean" is currently supported.')
    
    if time_bin_size is not None:
        return bin_timeseries(
            ts=TimeSeries(light_curves),
            aggregate_func=aggregate_func,
            time_bin_size=time_bin_size,
            time_bin_start=time_bin_start,
        )
    elif nbins is not None:
        return bin_table(
            tbl=Table(light_curves),
            aggregate_func=aggregate_func,
            nbins=nbins,
            )
    else:
        raise ValueError('[OPTICAM] Cannot rebin a light curve unless time_bin_size or nbins is defined.')


def bin_timeseries(
    ts: TimeSeries,
    aggregate_func: Callable,
    time_bin_size: Quantity,
    time_bin_start: Time | None,
    ) -> TimeSeries:
    """
    Bin an `astropy.timeseries.TimeSeries`. The resulting `astropy.timeseries.BinnedTimeSeries` is converted into an
    `astropy.timeseries.TimeSeries` using the time bin centers as the new times. This allows for gaps to be removed
    from the time series instead of being padded with zeros.
    
    Parameters
    ----------
    ts : TimeSeries
        The time series.
    aggregate_func : Callable
        The aggregate function - should propagate error columns correctly.
    time_bin_size : Quantity
        The time bin size.
    time_bin_start : Time | None
        The time bin start - useful for ensuring simultaneity between binned time series.
    
    Returns
    -------
    TimeSeries
        The binned time series.
    """
    
    gtis = infer_gtis(ts.time)  # GTIs are time-aware
    
    binned_ts: BinnedTimeSeries =  aggregate_downsample(
        time_series=ts,
        time_bin_start=time_bin_start,
        time_bin_size=time_bin_size,
        aggregate_func=aggregate_func,
        )
    
    return convert_binned_timeseries_to_timeseries(binned_ts=binned_ts, gtis=gtis)


def bin_table(
    tbl: Table,
    aggregate_func: Callable,
    nbins: int,
    ) -> Table:
    """
    Bin an `astropy.table.Table` into a specified number of bins.
    
    Parameters
    ----------
    tbl : Table
        The table.
    aggregate_func : Callable
        The aggregate function - should propagate error columns correctly.
    nbins : int
        The number of bins.
    
    Returns
    -------
    Table
        The binned table.
    """
    
    tbl['bin'] = np.floor(tbl['time'] * nbins).astype(int)  # get bin numbers
    new_tbl = tbl.group_by('bin').groups.aggregate(aggregate_func)  # aggregate bins
    new_tbl.remove_column('bin')  # remove bin numbers
    
    return new_tbl


def convert_binned_timeseries_to_timeseries(
    binned_ts: BinnedTimeSeries,
    gtis: Quantity,
    ) -> TimeSeries:
    """
    Convert an `astropy.timeseries.BinnedTimeSeries` into an `astropy.timeseries.TimeSeries` and apply the "Good Time
    Intervals".
    
    Parameters
    ----------
    binned_lc : BinnedTimeSeries
        The binned time series.
    gtis : Quantity
        The "Good Time Intervals" of the time series. Used to mask gaps.
    
    Returns
    -------
    TimeSeries
        The binned time series as an `astropy.timeseries.TimeSeries`.
    """
    
    new_ts = TimeSeries()
    
    # get names of all non-time columns
    columns = binned_ts.colnames
    ignored_columns = ['time_bin_start', 'time_bin_end', 'time_bin_size', 'time_bin_center']
    columns = [col for col in columns if col not in ignored_columns]
    
    for row in binned_ts:
        # convert row to a QTable to remove requirement for time columns
        row_copy = QTable(row)
        
        # get bin start and end times
        t_start = row_copy['time_bin_start']
        t_stop = row_copy['time_bin_start'] + row_copy['time_bin_size']
        
        # check if the bin falls within any of the GTIs
        valid = any([(t_start > gti[0]) & (t_stop < gti[1]) for gti in gtis])
        if valid:
            # create new row and add it to our new time series
            row_tbl = TimeSeries(time=row_copy['time_bin_start'] + row_copy['time_bin_size'] / 2)
            for col in columns:
                row_tbl.add_column(row_copy[col], name=col)
            
            new_ts = vstack([new_ts, row_tbl])
    
    return new_ts


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
    
    vals = col.copy()
    valid = np.isfinite(vals)
    
    if 'err' in str(vals.name):
        n = valid.sum()
        
        if n == 0:
            return np.nan
        else:
            return np.sqrt(np.sum(vals[valid]**2)) / n
    else:
        return np.mean(vals[valid])


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
    keys: list[str],
    ) -> TimeSeries:
    """
    Validate light curves. This ensures light curves are normalised and groups redundant rows.
    
    Parameters
    ----------
    light_curves : TimeSeries | None
        The light curves.
    norm : Literal["max", "mean", "none"]
        The normalisation.
    keys : list[str]
        The light curve keys.
    
    Returns
    -------
    TimeSeries
        The validated light curves.
    """
    
    validated_light_curves = TimeSeries()
    
    if light_curves:
        for key in keys:
            
            lc = get_lc(
                light_curves=light_curves,
                key=key,
                )
            
            # apply normalisation
            factor = get_norm_factor(norm, lc[f'{key}_rel_flux'].value)
            lc[f'{key}_rel_flux'] /= factor
            lc[f'{key}_rel_flux_err'] = np.abs(lc[f'{key}_rel_flux_err'] / factor)
            
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


def convert_lc_to_stingray(
    lc: TimeSeries,
    key: str,
    t_ref: Time,
    ) -> Lightcurve:
    """
    Convert a light curve from an `astropy.timeseries.TimeSeries` instance to a `stingray.Lightcurve` instance.
    
    Parameters
    ----------
    lc : TimeSeries
        The light curve.
    key : str
        The camera:filter key.
    t_ref : Time
        The reference time.
    
    Returns
    -------
    Lightcurve
        The light curve as a `stingray.Lightcurve` instance.
    """
    
    time = np.asarray((lc.time - t_ref).to_value(u.s))
    counts = np.asarray(lc[f'{key}_rel_flux'].value)
    err = np.asarray(lc[f'{key}_rel_flux_err'].value)
    
    gti = infer_gtis(time)
    
    return Lightcurve(
        time=time,
        counts=counts,
        err=err,
        gti=gti,
        err_dist='gauss',
        ).sort()



