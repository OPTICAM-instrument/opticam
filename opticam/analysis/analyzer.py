import copy
import os
from typing import Dict, Literal, Tuple

import astropy.units as u
from astropy.units.quantity import Quantity
import numpy as np
from numpy.typing import NDArray
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from pandas import DataFrame
from stingray import AveragedPowerspectrum, Powerspectrum
from stingray import AveragedCrossspectrum, Crossspectrum
from stingray import CrossCorrelation
from stingray import Lightcurve
from stingray.lombscargle import LombScarglePowerspectrum

from opticam.utils.helpers import sort_filters
from opticam.utils.time_helpers import infer_gtis
from opticam.utils.constants import colors


class Analyzer:
    """
    Helper class for analyzing OPTICAM light curves.
    """

    def __init__(
        self,
        out_directory: str,
        light_curves: Dict[str, Lightcurve | DataFrame] | None = None,
        prefix: str | None = None,
        phot_label: str | None = None,
        show_plots: bool = True,
        ) -> None:
        """
        Helper class for analyzing OPTICAM light curves.
        
        Parameters
        ----------
        out_directory : str
            The directory to save the output files (i.e., the same directory as `out_directory` used by
            `opticam_new.Photometer` when creating the light curves).
        light_curves : Dict[str, Lightcurve | DataFrame] | None, optional
            The light curves to analyze, where the keys are the filter names and the values are either Lightcurve
            objects or DataFrames containing 'BMJD', 'rel_flux', and 'rel_flux_err' columns. Leave as `None` to create an
            empty analyzer that can be populated later using the `join()` method.
        prefix : str | None, optional
            The prefix to use for the output files (e.g., the name of the target source).
        phot_label : str, optional
            The label for the photometry routine used to generate the light curves, used in the output file names.
        show_plots : bool, optional
            Whether to render and show plots, by default `True`.
        """
        
        self.light_curves = validate_light_curves(light_curves)
        
        self.out_directory = out_directory
        if not os.path.isdir(out_directory):
            try:
                os.makedirs(out_directory)
            except Exception as e:
                raise Exception(f'[OPTICAM] could not create output directory {out_directory}: {e}')
        
        self.prefix = prefix
        self.phot_label = phot_label
        self.show_plots = show_plots
        
        if self.show_plots:
            if not os.path.isdir(os.path.join(self.out_directory, 'plots')):
                os.makedirs(os.path.join(self.out_directory, 'plots'))
        
        if len(self.light_curves) > 0:
            self.t_ref = float(min([np.min(np.asarray(lc.time)) for lc in self.light_curves.values()]))


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
        
        new_light_curves = copy.copy(self.light_curves)
        
        for fltr in analyzer.light_curves.keys():
            if fltr not in self.light_curves.keys():
                # if a new filter is being added, copy the light curve
                new_light_curves[fltr] = copy.copy(analyzer.light_curves[fltr])
            else:
                # if an existing filter is being added, merge the light curves
                new_light_curves[fltr] = self.light_curves[fltr].join(analyzer.light_curves[fltr])
        
        return Analyzer(
            out_directory=self.out_directory,
            light_curves=new_light_curves,  # type: ignore
            prefix=self.prefix,
            phot_label=self.phot_label,
            show_plots=self.show_plots,
        )


    def rebin_light_curves(
        self,
        dt: Quantity,
        method: Literal['mean', 'sum'] = 'mean',
        ) -> None:
        """
        Rebin the light curves to a desired time resolution using `stingray.Lightcurve.rebin()`.
        
        Parameters
        ----------
        dt : Quantity
            The desired time resolution for the rebinned light curves. This must be an astropy `Quantity` with units of
            time (e.g., `astropy.units.s`) to ensure correct handling of the time resolution.
        method : Literal['mean', 'sum'], optional
            The rebinning method, by default `'mean'`.
        """
        
        # convert dt to days
        dt = dt.to_value(u.day)  # type: ignore
        
        for fltr, lc in self.light_curves.items():
            self.light_curves[fltr] = lc.rebin(dt, method=method)

    def plot_light_curves(
        self,
        show_gtis: bool = False,
        ) -> Figure:
        """
        Plot the light curves.
        
        Parameters
        ----------
        show_gtis : bool, optional
            Whether to highlight the Good Time Intervals on the light curve plot, by default `False`.
        
        Returns
        -------
        Figure
            The figure containing the light curves.
        """
        
        fig, axes = plt.subplots(
            nrows=len(self.light_curves),
            figsize=(2 * 6.4, .5 * len(self.light_curves) * 4.8),
            tight_layout=True,
            sharex=True,
            gridspec_kw={
                "hspace": 0,
                },
            )
        
        if len(self.light_curves) == 1:
            axes = [axes]
        
        for i, (fltr, lc) in enumerate(self.light_curves.items()):
            for lc_segment in lc.split_by_gti(min_points=1):
                t = (np.asarray(lc_segment.time) - self.t_ref) * 86400
                
                axes[i].errorbar(
                    t,
                    lc_segment.counts,
                    lc_segment.counts_err,
                    marker='none',
                    linestyle='none',
                    ecolor='grey',
                    elinewidth=1,
                    alpha=.5,
                    )
                axes[i].step(
                    t,
                    lc_segment.counts,
                    where='mid',
                    color=colors[fltr],
                    lw=1,
                    label=fltr,
                )
            
            axes[i].text(
                .95,
                .9,
                fltr,
                fontsize='large',
                transform=axes[i].transAxes,
                va='top',
                ha='right',
            )
            
            if show_gtis:
                gti = 86400 * (np.asarray(lc.gti) - self.t_ref)
                for j in range(gti.shape[0] - 1):
                    stop = gti[j][1]
                    start = gti[j + 1][0]
                    
                    axes[i].fill_betweenx(
                        axes[i].set_ylim(),
                        stop,
                        start,
                        color='grey',
                        edgecolor='none',
                        alpha=.5,
                    )
        
        axes[-1].set_xlabel(f'Time from BMJD {self.t_ref:.4f} [s]', fontsize='large')
        axes[len(self.light_curves) // 2].set_ylabel('Normalized flux', fontsize='large')
        
        for ax in axes:
            ax.minorticks_on()
            ax.tick_params(which='both', direction='in', top=True, right=True)
        
        fig.savefig(f'{self.out_directory}/plots/{self.prefix}_{self.phot_label}_light_curves.pdf')
        
        if self.show_plots:
            plt.show()
        
        return fig


    def phase_fold_light_curves(
        self,
        period: Quantity,
        ) -> Dict[str, NDArray]:
        """
        Phase fold each light curve using the given period.
        
        Parameters
        ----------
        period : Quantity
            The period to use for phase folding. This must be an astropy `Quantity` with units of time (e.g.,
            `astropy.units.s`) to ensure correct handling of the period.
        
        Returns
        -------
        Dict[str, NDArray]
            The phase folded light curves.
        """
        
        save_period = period.to_value(u.s)  # type: ignore
        period = period.to_value(u.day)  # type: ignore
        
        results = {}
        
        for fltr, lc in self.light_curves.items():
            phase = (lc.time % period) / period
            results[fltr] = phase
        
        if self.show_plots:
            fig, axes = plt.subplots(
                tight_layout=True,
                nrows=len(self.light_curves),
                sharex=True,
                figsize=(6.4, .5 * len(self.light_curves) * 4.8),
                gridspec_kw={'hspace': 0}
                )
            
            for i, (fltr, lc) in enumerate(self.light_curves.items()):
                axes[i].errorbar(
                    np.append(results[fltr], results[fltr] + 1),
                    np.append(np.asarray(lc.counts), np.asarray(lc.counts)),
                    np.append(lc.counts_err, lc.counts_err),
                    marker='none',
                    linestyle='none',
                    color=colors[fltr],
                    ecolor='grey',
                    elinewidth=1,
                    alpha=.5,
                    zorder=0,
                    )
                axes[i].plot(
                    np.append(results[fltr], results[fltr] + 1),
                    np.append(np.asarray(lc.counts), np.asarray(lc.counts)),
                    marker='.',
                    ms=2,
                    linestyle='none',
                    color=colors[fltr],
                    zorder=2,
                    )
                axes[i].text(
                    .95,
                    .9,
                    fltr,
                    fontsize='large',
                    va='top',
                    ha='right',
                    transform=axes[i].transAxes,
                )
            
            for ax in axes.flatten():
                ax.minorticks_on()
                ax.tick_params(which='both', direction='in', top=True, right=True)
            
            axes[-1].set_xlabel('Phase')
            axes[len(self.light_curves) // 2].set_ylabel('Normalized Flux')
            
            fig.savefig(f'{self.out_directory}/plots/{self.prefix}_{self.phot_label}_P={save_period:.4f}s_phase_fold.pdf')
            plt.show()
        
        return results

    def phase_bin_light_curves(
        self,
        period: Quantity,
        t0: float | None = None,
        n_bins: int = 10,
        plot: bool = True,
        subplot : bool = True,
        sharey: bool = False,
        ) -> Dict[str, Dict[str, NDArray]]:
        """
        Phase bin each light curve using the given period.
        
        Parameters
        ----------
        period : Quantity
            The period to use for phase binning. This must be an astropy `Quantity` with units of time (e.g.,
            `astropy.units.s`) to ensure correct handling of the period.
        t0 : float | None, optional
            Time of zero phase, by default `None`. If `None`, the first time value in the light curve will be used.
        n_bins : int, optional
            The number of phase bins, by default 10.
        plot : bool, optional
            Whether to plot the phase binned light curves, by default True.
        subplot : bool, optional
            Whether to plot filters in separate subplots, by default True.
        sharey : bool, optional
            Whether to render the plot with a common y-axis (useful for directly comparing amplitudes), by default
            False. Only used if `plot=True` and `subplot=True`.
        
        Returns
        -------
        Dict[str, Dict[str, NDArray]]
            The phase binned light curves.
        """
        
        save_period = period.to_value(u.s)  # type: ignore
        period = period.to_value(u.day)  # type: ignore
        
        results = {}
        
        for fltr, lc in self.light_curves.items():
            
            lc_time = np.asarray(lc.time)
            
            if t0:
                t = lc_time - t0
            else:
                t = lc_time - lc_time[0]
            
            phase = (t % period) / period
            bins = [[] for i in range(n_bins + 1)]
            bin_errs = [[] for i in range(n_bins + 1)]
            
            for i in range(len(phase)):
                bin_num = int(phase[i] * (n_bins + 1))
                bins[bin_num].append(np.asarray(lc.counts)[i])
                bin_errs[bin_num].append(np.asarray(lc.counts_err)[i])
            
            # remove final bin (it will be the same as the first)
            bins.pop()
            bin_errs.pop()
            
            fluxes = np.array([np.mean(b) for b in bins])
            flux_errs = np.array(np.sqrt([np.sum(np.asarray(b)**2) for b in bin_errs]) / len(bins[0]))
            
            results[fltr] = {
                'phase': np.linspace(0, 1, n_bins + 1)[:-1],  # remove final phase value (it will be the same as the first)
                'flux': fluxes,
                'flux error': flux_errs
            }
        
        if plot:
            if subplot:
                fig, axes = plt.subplots(
                    tight_layout=True,
                    nrows=len(self.light_curves),
                    sharex=True,
                    sharey=sharey,
                    figsize=(6.4, (0.5 * len(self.light_curves) * 4.8)),
                    gridspec_kw={'hspace': 0.},
                    )
                
                for i, (fltr, lc) in enumerate(self.light_curves.items()):
                    axes[i].errorbar(
                        np.append(results[fltr]['phase'], results[fltr]['phase'] + 1),
                        np.append(results[fltr]['flux'], results[fltr]['flux']),
                        np.append(results[fltr]['flux error'], results[fltr]['flux error']),
                        marker='none',
                        linestyle='none',
                        color=colors[fltr],
                        ecolor='grey',
                        elinewidth=1,
                        )
                    axes[i].step(
                        np.append(results[fltr]['phase'], results[fltr]['phase'] + 1),
                        np.append(results[fltr]['flux'], results[fltr]['flux']),
                        where='mid',
                        color=colors[fltr],
                        lw=1,
                    )
                    axes[i].text(
                        .95,
                        .9,
                        fltr,
                        fontsize='large',
                        va='top',
                        ha='right',
                        transform=axes[i].transAxes,
                    )
                
                axes[-1].set_xlabel('Phase', fontsize='large')
                axes[len(self.light_curves) // 2].set_ylabel('Normalized flux', fontsize='large')
                
                for ax in axes.flatten():
                    ax.minorticks_on()
                    ax.tick_params(which='both', direction='in', top=True, right=True)
            else:
                fig, ax = plt.subplots(
                    tight_layout=True,
                    )
                
                for i, (fltr, lc) in enumerate(self.light_curves.items()):
                    ax.errorbar(
                        np.append(results[fltr]['phase'], results[fltr]['phase'] + 1),
                        np.append(results[fltr]['flux'], results[fltr]['flux']),
                        np.append(results[fltr]['flux error'], results[fltr]['flux error']),
                        marker='none',
                        linestyle='none',
                        color=colors[fltr],
                        ecolor='grey',
                        elinewidth=1,
                        )
                    ax.step(
                        np.append(results[fltr]['phase'], results[fltr]['phase'] + 1),
                        np.append(results[fltr]['flux'], results[fltr]['flux']),
                        where='mid',
                        color=colors[fltr],
                        label=fltr,
                    )
                
                ax.set_xlabel('Phase', fontsize='large')
                ax.set_ylabel('Normalized flux', fontsize='large')
                ax.legend(fontsize='large')
                ax.minorticks_on()
                ax.tick_params(which='both', direction='in', top=True, right=True)
            
            fig.savefig(f'{self.out_directory}/plots/{self.prefix}_{self.phot_label}_P={save_period:.4f}s_phase_bin.pdf')
            plt.show()
        
        return results


    def compute_power_spectra(
        self,
        norm: Literal['frac', 'abs'] = 'frac',
        scale: Literal['linear', 'log', 'loglog'] = 'linear',
        ) -> Dict[str, Powerspectrum]:
        """
        Compute the power spectrum for each light curve using `stingray.Powerspectrum`. It's usually a good idea to call 
        the rebin() method to rebin your light curves to a regular time grid before calling this method.
        
        Parameters
        ----------
        norm : Literal['frac', 'abs'], optional
            The normalisation to use for the power spectrum, by default 'frac'. If 'frac', the power spectrum is
            normalised to fractional rms. If 'abs', the power spectrum is normalised to absolute power.
        scale : Literal['linear', 'log', 'loglog'], optional
            The scale to use for the plot, by default 'linear'. If 'linear', all axes are linear. If 'log', the
            frequency axis is logarithmic. If 'loglog', both the frequency and power axes are logarithmic.
        
        Returns
        -------
        Dict[str, Powerspectrum]
            A dictionary containing the power spectrum for each light curve, where the keys are the filter names and the
            values are the power spectra.
        """
        
        results = {}
        
        for fltr, lc in self.light_curves.items():
            
            dt = np.diff(np.asarray(lc.time))
            if not np.allclose(dt, dt[0]):
                print(f'[OPTICAM] Unable to compute periodogram for {fltr} light curve due to gaps. Consider using either the compute_lomb_scargle_periodograms() or compute_averaged_power_spectra() methods instead.')
                continue
            
            results[fltr] = Powerspectrum.from_lightcurve(
                convert_lc_time_to_seconds(lc, self.t_ref),
                norm=norm,
                silent=True,
                )
        
        if self.show_plots and len(results) > 0:
            fig = plot(results, scale)
            fig.savefig(f'{self.out_directory}/plots/{self.prefix}_{self.phot_label}_periodograms.pdf')
            plt.show()
        
        return results

    def compute_averaged_power_spectra(
        self,
        segment_size: Quantity,
        rebin_factor: float | None = None,
        norm: Literal['frac', 'abs'] = 'frac',
        scale: Literal['linear', 'log', 'loglog'] = 'linear',
        ) -> Dict[str, AveragedPowerspectrum]:
        """
        Compute the averaged power spectrum for each light curve using `stingray.AveragedPowerSpectrum`. It's usually a
        good idea to call the rebin() method to rebin your light curves to a regular time grid before calling this 
        method.
        
        Parameters
        ----------
        segment_size : Quantity
            The size of the segments to use for averaging the power spectra. This must be an astropy `Quantity` with
            units of time (e.g., `astropy.units.s`) to ensure correct handling of the segment size.
        rebin_factor : float | None, optional
            The factor by which to rebin the power spectrum in frequency. If 'None', no rebinning will be performed.
            If a float, the power spectrum will be geometrically/logarithmically rebinned with each bin being a factor
            `1 + rebin_factor` larger than the previous one.
        norm : Literal['frac', 'abs'], optional
            The normalisation to use for the power spectrum, by default 'frac'. If 'frac', the power spectrum is
            normalised to the fractional rms. If 'abs', the power spectrum is normalised to the absolute rms.
        scale : Literal['linear', 'log', 'loglog'], optional
            The scale to use for the plot, by default 'linear'. If 'linear', all axes are linear. If 'log', the
            frequency axis is logarithmic. If 'loglog', both the frequency and power axes are logarithmic.
        
        Returns
        -------
        Dict[str, AveragedPowerspectrum]
            The averaged power spectrum for each light curve, where the keys are the filter names and the values are
            the averaged power spectra.
        """
        
        segment_size = segment_size.to_value(u.s)  # type: ignore
        
        results = {}
        for fltr, lc in self.light_curves.items():
            
            ps = AveragedPowerspectrum.from_lightcurve(
                convert_lc_time_to_seconds(lc, self.t_ref),
                segment_size,
                norm=norm,
                silent=True,
            )
            
            print(f'[OPTICAM] {ps.m} {fltr} segments averaged.')
            
            if rebin_factor:
                ps = ps.rebin_log(rebin_factor)
            
            results[fltr] = ps
        
        if self.show_plots:
            fig = plot(results, scale)
            fig.savefig(f'{self.out_directory}/plots/{self.prefix}_{self.phot_label}_averaged_power_spectra.pdf')
            plt.show()
        
        return results


    def compute_lomb_scargle_periodograms(
        self,
        norm: Literal['abs', 'frac'] = 'frac',
        scale: Literal['linear', 'log', 'loglog'] = 'linear',
        ) -> Dict[str, LombScarglePowerspectrum]:
        """
        Compute the Lomb-Scargle periodogram for each light curve using `stingray.LombScarglePowerspectrum`.
        
        Parameters
        ----------
        norm : Literal['abs', 'frac'], optional
            The normalisation to use for the Lomb-Scargle periodogram, by default 'frac'. If 'abs', the periodogram is
            normalised to absolute power. If 'frac', the periodogram is normalised to fractional rms.
        scale : Literal['linear', 'log', 'loglog'], optional
            The scale to use for the inferred frequencies, by default 'linear'. If 'linear', the frequency grid is
            linearly spaced. If 'log', the frequency grid is logarithmically spaced. If 'loglog', both the frequency
            and power axes will be in logarithm. The upper and lower bounds of the frequencies are the same in all
            cases.
        Returns
        -------
        Tuple[NDArray, Dict[str, NDArray]] | Dict[str, NDArray]
            If no frequencies are provided, returns a tuple containing the frequencies and a dictionary of periodograms
            for each light curve. If frequencies are provided, returns a dictionary of periodogram powers for each light
            curve.
        """
        
        results = {}
        
        for fltr, lc in self.light_curves.items():
            
            # from_lightcurve() causes an error?
            results[fltr] = LombScarglePowerspectrum(
                convert_lc_time_to_seconds(lc, self.t_ref),
                norm=norm,
                power_type='absolute',
            )
        
        if self.show_plots:
            fig = plot(results, scale)
            fig.savefig(f'{self.out_directory}/plots/{self.prefix}_{self.phot_label}_L-S_periodograms.pdf')
            plt.show()
        
        return results


    def compute_cross_correlations(
        self,
        mode: Literal['same', 'valid', 'full'] = 'same',
        norm: Literal['none', 'variance'] = 'variance',
        force_match: bool = True,
        ) -> Dict[str, CrossCorrelation]:
        """
        Compute the cross-correlations for each pair of light curves using `stingray.CrossCorrelation`.
        
        Parameters
        ----------
        mode : Literal['same', 'valid', 'full'], optional
            The mode to use for the cross-correlation, by default 'same'. See `stingray.CrossCorrelation` for details on
            the different modes.
        norm : Literal['none', 'variance'], optional
            The normalisation to use for the cross-correlation, by default 'variance'. See `stingray.CrossCorrelation`
            for details on the different normalisations.
        force_match : bool, optional
            Whether to force the light curves to have the same time columns before computing the cross-correlation,
            by default `True`. If `False`, cross-correlation calculations may fail if the light curves have different
            time columns.
        
        Returns
        -------
        Dict[str, CrossCorrelation]
            A dictionary containing the cross-correlations for each pair of light curves, where the keys are tuples of
            filter names and the values are the cross-correlations.
        """
        
        results = {}
        
        for fltr1, lc1 in self.light_curves.items():
            for fltr2, lc2 in self.light_curves.items():
                # skip if the filters are the same or if the cross-correlation has already been computed
                if fltr1 == fltr2 or f'{fltr2} x {fltr1}' in results.keys():
                    continue
                
                if force_match:
                    # force the light curves to have the same time columns
                    lc1, lc2 = match_light_curve_times(lc1, lc2)
                
                cc = CrossCorrelation(
                    lc1,
                    lc2,
                    mode=mode,
                    norm=norm,
                    )
                
                # convert times to seconds
                cc.dt *= 86400  # type: ignore
                cc.time_lags *= 86400  # type: ignore
                cc.time_shift *= 86400  # type: ignore
                
                results[f'{fltr1} x {fltr2}'] = cc
        
        if self.show_plots:
            fig = plot(results)
            fig.savefig(f'{self.out_directory}/plots/{self.prefix}_{self.phot_label}_cross_correlations.pdf')
            plt.show()
        
        return results




def validate_light_curves(
    light_curves: Dict[str, Lightcurve | DataFrame] | None,
    ) -> Dict[str, Lightcurve]:
    """
    Validate the light curves by converting DataFrames to Lightcurve objects and inferring GTIs.
    
    Parameters
    ----------
    light_curves : Dict[str, Lightcurve | DataFrame] | None
        The light curves to validate, where the keys are the filter names and the values are either Lightcurve
        objects or DataFrames containing 'BMJD', 'rel_flux', and 'rel_flux_err' columns. If `None`, an empty
        dictionary will be returned.
    
    Returns
    -------
    Dict[str, Lightcurve]
        If `light_curves` is `None`, returns an empty dictionary. Otherwise, returns a dictionary containing the 
        validated light curves, where the keys are the filter names and the values are Lightcurve objects.
    """
    
    validated_light_curves = {}
    
    if light_curves:
        for fltr in light_curves.keys():
            if isinstance(light_curves[fltr], DataFrame):
                time = np.asarray(light_curves[fltr]['BMJD'].values)
                counts = np.asarray(light_curves[fltr]['rel_flux'].values)
                counts_err = np.asarray(light_curves[fltr]['rel_flux_err'].values)
                
                # infer GTIs
                gtis = infer_gtis(time, threshold=1.5)
            elif isinstance(light_curves[fltr], Lightcurve):
                time = np.asarray(light_curves[fltr].time)
                counts = np.asarray(light_curves[fltr].counts)
                counts_err = np.asarray(light_curves[fltr].counts_err)
                gtis = light_curves[fltr].gti
            else:
                raise TypeError(f'[OPTICAM] Light curve for filter {fltr} must be either a DataFrame or a Lightcurve object, but got {type(light_curves[fltr])}.')
            
            # normalise flux
            mean_flux = np.mean(counts)
            counts /= mean_flux
            counts_err /= mean_flux
            
            validated_light_curves[fltr] = Lightcurve(
                time,
                counts,
                err=counts_err,
                gti=gtis,
                err_dist='gauss',
                ).sort()
    
    return sort_filters(validated_light_curves)

def convert_lc_time_to_seconds(
    lc: Lightcurve,
    t_ref: float,
    ) -> Lightcurve:
    """
    Convert the time of a light curve from days to seconds from some reference time.
    
    Parameters
    ----------
    lc : Lightcurve
        The light curve to convert.
    t_ref : float
        The reference time.
    
    Returns
    -------
    Lightcurve
        The light curve with time converted to seconds from `t_ref`.
    """
    
    t = (np.asarray(lc.time) - t_ref) * 86400
    gti = (np.asarray(lc.gti) - t_ref) * 86400
    
    return Lightcurve(
        t,
        lc.counts,
        err=lc.counts_err,
        gti=gti,
        err_dist=lc.err_dist,
    )


def plot(
    results: Dict[str,
                  AveragedPowerspectrum | 
                  Powerspectrum | 
                  Crossspectrum | 
                  AveragedCrossspectrum | 
                  LombScarglePowerspectrum |
                  CrossCorrelation
                  ],
    scale: Literal['linear', 'log', 'loglog'] = 'linear',
    ) -> Figure:
    """
    Plot the results of some timing analysis. Not intended to be called directly by the user, but rather via
    various timing methods.
    
    Parameters
    ----------
    results : Dict[str, AveragedPowerspectrum  |  Powerspectrum  |  Crossspectrum  |  AveragedCrossspectrum  |  
    LombScarglePowerspectrum | CrossCorrelation]
        The timing analysis results, where the keys are the filter names and the values are the results.
    scale : Literal['linear', 'log', 'loglog'], optional
        The scale to use for the plot, by default 'linear'. If 'linear', all axes are linear. If 'log', the
        x-axis is logarithmic. If 'loglog', both the x- and y-axes are logarithmic.
    
    Returns
    -------
    Figure
        The figure containing the plot.
    """
    
    fig, axes = plt.subplots(
        figsize=(6.4, 4.8 * .5 * len(results)),
        tight_layout=True,
        nrows=len(results),
        sharex=True,
        gridspec_kw={'hspace': 0.},
    )
    
    if len(results) == 1:
        axes = [axes]
    
    for i, key in enumerate(results.keys()):
        if isinstance(results[key], 
                      AveragedPowerspectrum | 
                      Powerspectrum | 
                      LombScarglePowerspectrum
                      ):
            x = results[key].freq
            y = results[key].power
            yerr = define_yerr(results[key])
            
            x_label = 'Frequency [Hz]'
            y_label = f'Power [{get_normalisation_units(results[key].norm)}]'
        elif isinstance(results[key], CrossCorrelation):
            x = results[key].time_lags
            y = results[key].corr
            yerr = None
            
            x_label = 'Time lag [s]'
            y_label = 'Correlation'
        
        if key in colors:
            color = colors[key]
        else:
            color = 'k'
        
        axes[i].step(
            x,
            y,
            color=color,
            lw=1,
            where='mid',
        )
        
        if yerr is not None:
            axes[i].errorbar(
                x,
                y,
                yerr,
                linestyle='none',
                marker='none',
                ecolor='grey',
                elinewidth=1,
                zorder=0,
                )
        
        axes[i].text(
            .95,
            .9,
            key,
            transform=axes[i].transAxes,
            fontsize='large',
            ha='right',
            va='top',
        )
        
        if isinstance(results[key], CrossCorrelation):
            axes[i].text(
                .05,
                .9,
                f'time shift: {results[key].time_shift:.1f} s',
                transform=axes[i].transAxes,
                fontsize='large',
                ha='left',
                va='top',
            )
    
    if scale == 'log':
        for ax in axes:
            ax.set_xscale('log')
    elif scale == 'loglog':
        for ax in axes:
            ax.set_xscale('log')
            ax.set_yscale('log')
    
    for ax in axes:
        ax.minorticks_on()
        ax.tick_params(which='both', direction='in', top=True, right=True)
    
    axes[-1].set_xlabel(x_label)
    axes[len(results) // 2].set_ylabel(y_label)
    
    return fig

def define_yerr(
    obj: AveragedPowerspectrum |
    Powerspectrum |
    Crossspectrum |
    AveragedCrossspectrum |
    LombScarglePowerspectrum,
    ) -> NDArray | None:
    """
    Determine the y-error for the given object.
    
    Parameters
    ----------
    obj : AveragedPowerspectrum | Powerspectrum | Crossspectrum | AveragedCrossspectrum | LombScarglePowerspectrum
        The object for which to determine the y-error.
    
    Returns
    -------
    NDArray | None
        The y-error array if it exists (and a sufficient number of segments has been averaged), otherwise `None`.
    """
    
    if isinstance(obj, (AveragedPowerspectrum, AveragedCrossspectrum)):
        return obj.power_err
    
    return None

def get_normalisation_units(
    norm: str,
    ) -> str:
    """
    Get the units of the normalisation based on the specified normalisation type.
    
    Parameters
    ----------
    norm : str
        The normalisation type.
    
    Returns
    -------
    str
        The units of the normalisation.
    
    Raises
    ------
    ValueError
        If the specified normalisation type is invalid.
    """
    
    if norm == 'frac':
        return '(fractional rms)$^2$ Hz$^{{-1}}$'
    elif norm == 'abs':
        return 'rms$^2$ Hz$^{{-1}}$'
    else:
        raise ValueError(f'[OPTICAM] Normalization {norm} is not supported. Use either "frac" (recommended) or "abs".')


def match_light_curve_times(
    lc1: Lightcurve,
    lc2: Lightcurve,
    ) -> Tuple[Lightcurve, Lightcurve]:
    """
    Match the time columns of two light curves.
    
    Parameters
    ----------
    lc1 : Lightcurve
        The first light curve.
    lc2 : Lightcurve
        The second light curve.
    
    Returns
    -------
    Tuple[Lightcurve, Lightcurve]
        The two light curves with matched time columns.
    """
    
    # get intersecting GTIs
    gti = intersect_gtis(lc1.gti, lc2.gti)
    lc1_restricted = restrict_to_gti(lc1, gti)
    lc2_restricted = restrict_to_gti(lc2, gti)
    
    # interpolate lc2 onto lc1's time grid
    interp_counts = np.interp(
        np.asarray(lc1_restricted.time),
        np.asarray(lc2_restricted.time),
        np.asarray(lc2_restricted.counts),
    )
    interp_err = np.interp(
        np.asarray(lc1_restricted.time),
        np.asarray(lc2_restricted.time),
        np.asarray(lc2_restricted.counts_err),
    )
    lc2_interp = Lightcurve(
        lc1_restricted.time,
        interp_counts,
        err=interp_err,
        gti=gti,
        err_dist=lc2_restricted.err_dist,
    )
    
    return lc1_restricted, lc2_interp

def intersect_gtis(gti1, gti2):

    result = []
    for start1, stop1 in gti1:
        for start2, stop2 in gti2:
            start = max(start1, start2)
            stop = min(stop1, stop2)
            if start < stop:
                result.append([start, stop])
    
    return np.array(result)

def restrict_to_gti(lc, gti):
    
    mask = np.zeros_like(lc.time, dtype=bool)
    for start, stop in gti:
        mask |= (lc.time >= start) & (lc.time <= stop)
        
    return Lightcurve(
        lc.time[mask],
        lc.counts[mask],
        err=lc.counts_err[mask],
        gti=gti,
        err_dist=lc.err_dist,
    )


