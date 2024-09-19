import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
import matplotlib.colors as clr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import run_experiments as rexp
import metrics
import copy

# TODO: add some extremes plots

# define some colors and cmaps -- csu and nuggets
c_flowertrial = '#E56A54'
c_horsetooth = '#008FB3'
c_sunshine = '#CA9500'
c_skyline = '#1D428A'
c_skyline_l = '#295dc3'
c_skyline_ll = '#5583db'
c_midnight = '#0E2240'
red_skyline = ['#FFC72C', '#FF8200', '#FC4C02', '#DC4405', '#862633']
red_cmap = clr.LinearSegmentedColormap.from_list('red skyline', red_skyline, N=256)
blue_skyline = ['#0C2340', '#003057', '#1D4289', '#0057B7', '#418FDE', '#FFFFFF']
blue_cmap = clr.LinearSegmentedColormap.from_list('blue skyline', blue_skyline[::-1], N=256)
# set up an alpha colorbar
alpha_colors = []
for ii in range(0, 11):
    alpha = np.sin(ii/10*np.pi/2)
    alpha_colors.append((1.0, 0.6445674740484428, 0.08593617839292579, alpha))
alpha_cmap = clr.LinearSegmentedColormap.from_list('alpha cmap', alpha_colors, N=11)

# set some plotting defaults
plt.rc('xtick',labelsize=16)
plt.rc('ytick',labelsize=16)
plt.rcParams['xtick.major.size'] = 8
plt.rcParams['xtick.major.width'] = 1.75
plt.rcParams['xtick.minor.size'] = 5
plt.rcParams['xtick.minor.width'] = 1.25
plt.rcParams['ytick.major.size'] = 8
plt.rcParams['ytick.major.width'] = 1.75
plt.rcParams['ytick.minor.size'] = 5
plt.rcParams['ytick.minor.width'] = 1.25
plt.rcParams['axes.linewidth'] = 2

# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------

class PlotMetrics():
    """
    Class for plotting analog metrics for a given SOI.
    """
    def __init__(self, experiment_name, soi_model, soi_member, variable, soi_index=None, fig_path=None):
        if fig_path is None:
            fig_path = 'figures/'+experiment_name+'/'
        self.fig_path = fig_path

        # save the experiment_name
        self.experiment_name = experiment_name
        self.soi_model = soi_model
        self.soi_member = soi_member
        self.variable = variable
        self.soi_index = soi_index
        # get the settings so the region name can be used in the figure
        self.settings = rexp.Experiments().get_experiment(self.experiment_name)
        self.leadtime = self.settings['leadtime']
        self.metrics = metrics.AnalogMetrics(self.experiment_name, self.soi_model, self.soi_member, self.variable, self.soi_index)
        self.predictions = ["analog", "model", "perfect"]

# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------

    def nana_mse(self, n_analogs, savename=None, time_range=None):
        """plot the spread over ensemble versus number of analogs and compare to model mean"""
        mses = []
        nana = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 1000, 2000, 5000]
        for ii in nana:
            mse = self.metrics.prediction_mse(ii, prediction="analog", map=False, time_range=time_range)
            mses.append(np.nanmean(mse))
        mse_allmem = np.nanmean(self.metrics.prediction_mse(ii, prediction="model", map=False, time_range=time_range))
        mse_perfect = np.nanmean(self.metrics.prediction_mse(ii, prediction="perfect", map=False, time_range=time_range))

        # now make the plot
        fig, ax = plt.subplots(1, 1, figsize=(12,5))
        # plot the analog prediction mse
        ax.semilogx(nana, mses, '-o', label='analog ('+str(n_analogs)+')', lw=2.66, c=c_sunshine,
                    path_effects=[patheffects.withStroke(linewidth=4, foreground="k")])
        # plot the same for the all model mean
        ax.axhline(mse_allmem, label='model mean', ls='--', lw=1.66, c=c_skyline_l,
                path_effects=[patheffects.withStroke(linewidth=2.66, foreground="k")])
        # and for the perfect analogs
        ax.axhline(mse_perfect, label='perfect analogs', ls='--', lw=1.66, c=c_flowertrial,
                path_effects=[patheffects.withStroke(linewidth=2.66, foreground="k")])

        # legend, labels, etc.
        handles, labels = ax.get_legend_handles_labels()
        handles = [copy.copy(ha) for ha in handles]
        leg = plt.legend(handles=handles, labels=labels, loc='upper left', ncols=3, frameon=False, prop={'weight':'bold', 'size':'15'})
        ax.set_xlabel('number of analogs', fontsize=18)
        ax.set_ylabel('mean squared error', fontsize=18)
        ax.set_title(str(self.settings['leadtime'])+' year lead, '+str(self.settings['window'])+' year window', fontsize=18)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')    
        ax.patch.set_facecolor('none')
        ax.tick_params(which='both', direction='inout', right=False, labelright=False, labelleft=True, length=10)
        if savename is None:
            savename = self.experiment_name+'_nana_mse.pdf'
            plt.savefig(self.fig_path+savename, dpi=400, bbox_inches='tight', pad_inches=0.1)
        elif savename is False:
            pass
        else:
            plt.savefig(self.fig_path+savename, dpi=400, bbox_inches='tight', pad_inches=0.1)
        plt.show()

# -------------------------------------------------------------------------------

    def time_mse(self, n_analogs, savename=None, time_range=None):
        # restrict to specific time range, if requested
        if time_range is not None:
            time_slice = slice(time_range[0], time_range[1])
        else:
            time_slice = slice("0000", "9999")

        # get the metrics for the analog, model, and perfect predictions
        mse_analog = self.metrics.prediction_mse(n_analogs, prediction="analog", map=False, time_range=time_range)
        mse_model = self.metrics.prediction_mse(n_analogs, prediction="model", map=False, time_range=time_range)
        mse_perfect = self.metrics.prediction_mse(n_analogs, prediction="perfect", map=False, time_range=time_range)
        # corresponding times
        time = self.metrics.truth.sel(time=time_slice)['time.year']

        # now make the plot
        fig, ax = plt.subplots(1, 1, figsize=(14,5))
        # plot the analog prediction
        ax.semilogy(time, mse_analog, '-', label='analog ('+str(n_analogs)+')', lw=2.66, c=c_sunshine,
                    path_effects=[patheffects.withStroke(linewidth=4, foreground="k")])
        # the mean across time
        ax.axhline(np.nanmean(mse_analog), ls='--', color=c_sunshine, lw=1.66, zorder=0, label='analog mean',
                path_effects=[patheffects.withStroke(linewidth=2.66, foreground="k")])
        # plot the same for the all model mean
        ax.semilogy(time, mse_model, '-', label='model', lw=2.66, c=c_skyline_l,
                path_effects=[patheffects.withStroke(linewidth=4, foreground="k")])
        ax.axhline(np.nanmean(mse_model), ls='--', label='model mean', lw=1.66, c=c_skyline_l, zorder=0,
                path_effects=[patheffects.withStroke(linewidth=2.66, foreground="k")])
        # and for the perfect analogs
        ax.semilogy(time, mse_perfect, '-', label='perfect analogs', lw=2.66, c=c_flowertrial,
                path_effects=[patheffects.withStroke(linewidth=4, foreground="k")])
        ax.axhline(np.nanmean(mse_perfect), ls='--', label='perfect mean', lw=1.66, c=c_flowertrial, zorder=0,
                path_effects=[patheffects.withStroke(linewidth=2.66, foreground="k")])

        # legend, labels, etc.
        handles, labels = ax.get_legend_handles_labels()
        handles = [copy.copy(ha) for ha in handles]
        leg = plt.legend(handles=handles, labels=labels, loc='upper left', ncols=3, frameon=False,
                        prop={'weight':'bold', 'size':'15'}, bbox_to_anchor=(0.1, 1.15))
        ax.set_xlabel('year', fontsize=18)
        ax.set_ylabel('mean squared error', fontsize=18)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')    
        ax.patch.set_facecolor('none')
        ax.tick_params(which='both', direction='inout', right=False, labelright=False, labelleft=True, length=10)
        if savename is None:
            savename = self.experiment_name+'_time_mse.pdf'
            plt.savefig(self.fig_path+savename, dpi=400, bbox_inches='tight', pad_inches=0.1)
        elif savename is False:
            pass
        else:
            plt.savefig(self.fig_path+savename, dpi=400, bbox_inches='tight', pad_inches=0.1)
        plt.show()

# -------------------------------------------------------------------------------

    def map_mse(self, n_analogs, prediction="analog", savename=None, time_range=None, align=False):
        # get the metrics for the requested prediction type
        mse_map = self.metrics.prediction_mse(n_analogs, prediction=prediction, map=True, time_range=time_range)
        
        # region of target
        lats, lons = self.metrics.lats, self.metrics.lons
        lat_over_lon = (lats.max() - lats.min()) / (lons.max() - lons.min())
        lon_size, lat_size = 9, 9 * lat_over_lon / 2
        # get the coastlines
        land_feature = cfeature.NaturalEarthFeature(
            category='physical',
            name='land',
            scale='110m',
            facecolor='none',
            edgecolor='k',
            linewidth=2.5,
        )
        data_crs = ccrs.PlateCarree()

        # make the range of colorbar values consistent between prediction types?
        if align:
            ptypes = np.setdiff1d(self.predictions, prediction)
            vmin = np.min([mse_map]+[self.metrics.prediction_mse(n_analogs, prediction=ptype, map=True, time_range=time_range) for ptype in ptypes])
            vmax = np.max([mse_map]+[self.metrics.prediction_mse(n_analogs, prediction=ptype, map=True, time_range=time_range) for ptype in ptypes])
        else:
            vmin, vmax = None, None

        # make the plot
        fig = plt.figure(figsize=(lon_size, lat_size))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=215.))
        ax.add_feature(land_feature)
        image = ax.pcolormesh(lons, lats, mse_map, transform=data_crs, vmin=vmin, vmax=vmax, cmap=blue_cmap)
        # add a color bar for the figure
        cax = fig.add_axes([0.3, -0.05, 0.4, 0.05])
        cbar = fig.colorbar(image, cax=cax, orientation='horizontal')
        cax.set_xlabel('mse ('+prediction+')', fontsize=20)
        cax.tick_params(which='major', direction='inout', length=10, labelsize=16)
        plt.tight_layout()
        if savename is None:
            savename = self.experiment_name+'_map_mse.pdf'
            plt.savefig(self.fig_path+savename, dpi=400, bbox_inches='tight', pad_inches=0.1)
        elif savename is False:
            pass
        else:
            plt.savefig(self.fig_path+savename, dpi=400, bbox_inches='tight', pad_inches=0.1)
        plt.show()

# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------

    def nana_soe(self, n_analogs, savename=None, time_range=None):
        """plot the spread over ensemble versus number of analogs and compare to model mean"""
        soes = []
        nana = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 1000, 2000, 5000]
        for ii in nana:
            soe = self.metrics.spread_ensemble(ii, prediction="analog", map=False, time_range=time_range)
            soes.append(np.nanmean(soe))
        soe_allmem = np.nanmean(self.metrics.spread_ensemble(ii, prediction="model", map=False, time_range=time_range))

        # now make the plot
        fig, ax = plt.subplots(1, 1, figsize=(12,5))
        # plot the analog prediction spread over ensemble error
        ax.semilogx(nana, soes, '-o', label='analog ('+str(n_analogs)+')', lw=2.66, c=c_sunshine,
                    path_effects=[patheffects.withStroke(linewidth=4, foreground="k")])
        # plot the same for the all model mean
        ax.axhline(soe_allmem, label='model mean', ls='--', lw=1.66, c=c_skyline_l,
                path_effects=[patheffects.withStroke(linewidth=2.66, foreground="k")])

        # legend, labels, etc.
        handles, labels = ax.get_legend_handles_labels()
        handles = [copy.copy(ha) for ha in handles]
        leg = plt.legend(handles=handles, labels=labels, loc='upper left', ncols=3, frameon=False, prop={'weight':'bold', 'size':'15'})
        ax.set_xlabel('number of analogs', fontsize=18)
        ax.set_ylabel('spread over ensemble error', fontsize=18)
        ax.set_title(str(self.settings['leadtime'])+' year lead, '+str(self.settings['window'])+' year window', fontsize=18)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')    
        ax.patch.set_facecolor('none')
        ax.tick_params(which='both', direction='inout', right=False, labelright=False, labelleft=True, length=10)
        if savename is None:
            savename = self.experiment_name+'_nana_soe.pdf'
            plt.savefig(self.fig_path+savename, dpi=400, bbox_inches='tight', pad_inches=0.1)
        elif savename is False:
            pass
        else:
            plt.savefig(self.fig_path+savename, dpi=400, bbox_inches='tight', pad_inches=0.1)
        plt.show()

# -------------------------------------------------------------------------------

    def time_soe(self, n_analogs, savename=None, time_range=None):
        # restrict to specific time range, if requested
        if time_range is not None:
            time_slice = slice(time_range[0], time_range[1])
        else:
            time_slice = slice("0000", "9999")

        # get the metrics for the analog, model, and perfect predictions
        soe_analog = self.metrics.spread_ensemble(n_analogs, prediction="analog", map=False, time_range=time_range)
        soe_model = self.metrics.spread_ensemble(n_analogs, prediction="model", map=False, time_range=time_range)
        # corresponding times
        time = self.metrics.truth.sel(time=time_slice)['time.year']

        # now make the plot
        fig, ax = plt.subplots(1, 1, figsize=(14,5))
        # plot the analog prediction
        ax.plot(time, soe_analog, '-', label='analog ('+str(n_analogs)+')', lw=2.66, c=c_sunshine,
                    path_effects=[patheffects.withStroke(linewidth=4, foreground="k")])
        # the mean across time
        ax.axhline(np.nanmean(soe_analog), ls='--', color=c_sunshine, lw=1.66, zorder=0, label='analog mean',
                path_effects=[patheffects.withStroke(linewidth=2.66, foreground="k")])
        # plot the same for the all model mean
        ax.plot(time, soe_model, '-', label='model', lw=2.66, c=c_skyline_l,
                path_effects=[patheffects.withStroke(linewidth=4, foreground="k")])
        ax.axhline(np.nanmean(soe_model), ls='--', label='model mean', lw=1.66, c=c_skyline_l, zorder=0,
                path_effects=[patheffects.withStroke(linewidth=2.66, foreground="k")])

        # legend, labels, etc.
        handles, labels = ax.get_legend_handles_labels()
        handles = [copy.copy(ha) for ha in handles]
        leg = plt.legend(handles=handles, labels=labels, loc='upper left', ncols=4, frameon=False,
                        prop={'weight':'bold', 'size':'15'}, bbox_to_anchor=(0.0, 1.1))
        ax.set_xlabel('year', fontsize=18)
        ax.set_ylabel('spread over ensemble error', fontsize=18)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')    
        ax.patch.set_facecolor('none')
        ax.tick_params(which='both', direction='inout', right=False, labelright=False, labelleft=True, length=10)
        if savename is None:
            savename = self.experiment_name+'_time_soe.pdf'
            plt.savefig(self.fig_path+savename, dpi=400, bbox_inches='tight', pad_inches=0.1)
        elif savename is False:
            pass
        else:
            plt.savefig(self.fig_path+savename, dpi=400, bbox_inches='tight', pad_inches=0.1)
        plt.show()


# -------------------------------------------------------------------------------

    def map_soe(self, n_analogs, prediction="analog", savename=None, time_range=None, align=False):
        # get the metrics for the requested prediction type
        soe_map = self.metrics.spread_ensemble(n_analogs, prediction=prediction, map=True, time_range=time_range)
        
        # region of target
        lats, lons = self.metrics.lats, self.metrics.lons
        lat_over_lon = (lats.max() - lats.min()) / (lons.max() - lons.min())
        lon_size, lat_size = 9, 9 * lat_over_lon / 2
        # get the coastlines
        land_feature = cfeature.NaturalEarthFeature(
            category='physical',
            name='land',
            scale='110m',
            facecolor='none',
            edgecolor='k',
            linewidth=2.5,
        )
        data_crs = ccrs.PlateCarree()

        # make the range of colorbar values consistent between prediction types?
        if align:
            ptypes = np.setdiff1d(self.predictions, [prediction, "perfect"])
            vmin = np.min([soe_map]+[self.metrics.spread_ensemble(n_analogs, prediction=ptype, map=True, time_range=time_range) for ptype in ptypes])
            vmax = np.max([soe_map]+[self.metrics.spread_ensemble(n_analogs, prediction=ptype, map=True, time_range=time_range) for ptype in ptypes])
        else:
            vmin, vmax = None, None

        # make the plot
        fig = plt.figure(figsize=(lon_size, lat_size))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=215.))
        ax.add_feature(land_feature)
        image = ax.pcolormesh(lons, lats, soe_map, transform=data_crs, vmin=vmin, vmax=vmax, cmap=blue_cmap)
        # add a color bar for the figure
        cax = fig.add_axes([0.3, -0.05, 0.4, 0.05])
        cbar = fig.colorbar(image, cax=cax, orientation='horizontal')
        cax.set_xlabel('soe ('+prediction+')', fontsize=20)
        cax.tick_params(which='major', direction='inout', length=10, labelsize=16)
        plt.tight_layout()
        if savename is None:
            savename = self.experiment_name+'_map_soe.pdf'
            plt.savefig(self.fig_path+savename, dpi=400, bbox_inches='tight', pad_inches=0.1)
        elif savename is False:
            pass
        else:
            plt.savefig(self.fig_path+savename, dpi=400, bbox_inches='tight', pad_inches=0.1)
        plt.show()

# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------

    def nana_emd(self, n_analogs, penalty=1, savename=None, time_range=None):
        """plot the earth mover's distance versus number of analogs and compare to model mean and perfect analogs"""
        emds = []
        nana = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 1000, 2000, 5000]
        for ii in nana:
            emd = self.metrics.earth_movers_distance(ii, prediction="analog", time_range=time_range, extra_mass_penalty=penalty)
            emds.append(np.nanmean(emd))
        emd_allmem = np.nanmean(self.metrics.earth_movers_distance(ii, prediction="model", time_range=time_range, extra_mass_penalty=penalty))
        emd_perfect = np.nanmean(self.metrics.earth_movers_distance(ii, prediction="perfect", time_range=time_range, extra_mass_penalty=penalty))

        # now make the plot
        fig, ax = plt.subplots(1, 1, figsize=(12,5))
        # plot the analog prediction earth mover's distance
        ax.semilogx(nana, emds, '-o', label='analog ('+str(n_analogs)+')', lw=2.66, c=c_sunshine,
                    path_effects=[patheffects.withStroke(linewidth=4, foreground="k")])
        # plot the same for the all model mean
        ax.axhline(emd_allmem, label='model mean', ls='--', lw=1.66, c=c_skyline_l,
                path_effects=[patheffects.withStroke(linewidth=2.66, foreground="k")])
        # and for the perfect analogs
        ax.axhline(emd_perfect, label='perfect analogs', ls='--', lw=1.66, c=c_flowertrial,
                path_effects=[patheffects.withStroke(linewidth=2.66, foreground="k")])

        # legend, labels, etc.
        handles, labels = ax.get_legend_handles_labels()
        handles = [copy.copy(ha) for ha in handles]
        leg = plt.legend(handles=handles, labels=labels, loc='upper left', ncols=3, frameon=False, prop={'weight':'bold', 'size':'15'})
        ax.set_xlabel('number of analogs', fontsize=18)
        ax.set_ylabel("earth mover's distance", fontsize=18)
        ax.set_title(str(self.settings['leadtime'])+' year lead, '+str(self.settings['window'])+' year window', fontsize=18)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')    
        ax.patch.set_facecolor('none')
        ax.tick_params(which='both', direction='inout', right=False, labelright=False, labelleft=True, length=10)
        if savename is None:
            savename = self.experiment_name+'_nana_emd.pdf'
            plt.savefig(self.fig_path+savename, dpi=400, bbox_inches='tight', pad_inches=0.1)
        elif savename is False:
            pass
        else:
            plt.savefig(self.fig_path+savename, dpi=400, bbox_inches='tight', pad_inches=0.1)
        plt.show()

# -------------------------------------------------------------------------------

    def time_emd(self, n_analogs, penalty=1, savename=None, time_range=None):
        # restrict to specific time range, if requested
        if time_range is not None:
            time_slice = slice(time_range[0], time_range[1])
        else:
            time_slice = slice("0000", "9999")

        # get the metrics for the analog, model, and perfect predictions
        emd_analog = self.metrics.earth_movers_distance(n_analogs, prediction="analog", time_range=time_range, extra_mass_penalty=penalty)
        emd_model = self.metrics.earth_movers_distance(n_analogs, prediction="model", time_range=time_range, extra_mass_penalty=penalty)
        emd_perfect = self.metrics.earth_movers_distance(n_analogs, prediction="perfect", time_range=time_range, extra_mass_penalty=penalty)
        # corresponding times
        time = self.metrics.truth.sel(time=time_slice)['time.year']

        # now make the plot
        fig, ax = plt.subplots(1, 1, figsize=(14,5))
        # plot the analog prediction
        ax.semilogy(time, emd_analog, '-', label='analog ('+str(n_analogs)+')', lw=2.66, c=c_sunshine,
                    path_effects=[patheffects.withStroke(linewidth=4, foreground="k")])
        # the mean across time
        ax.axhline(np.nanmean(emd_analog), ls='--', color=c_sunshine, lw=1.66, zorder=0, label='analog mean',
                path_effects=[patheffects.withStroke(linewidth=2.66, foreground="k")])
        # plot the same for the all model mean
        ax.semilogy(time, emd_model, '-', label='model', lw=2.66, c=c_skyline_l,
                path_effects=[patheffects.withStroke(linewidth=4, foreground="k")])
        ax.axhline(np.nanmean(emd_model), ls='--', label='model mean', lw=1.66, c=c_skyline_l, zorder=0,
                path_effects=[patheffects.withStroke(linewidth=2.66, foreground="k")])
        # and for the perfect analogs
        ax.semilogy(time, emd_perfect, '-', label='perfect analogs', lw=2.66, c=c_flowertrial,
                path_effects=[patheffects.withStroke(linewidth=4, foreground="k")])
        ax.axhline(np.nanmean(emd_perfect), ls='--', label='perfect mean', lw=1.66, c=c_flowertrial, zorder=0,
                path_effects=[patheffects.withStroke(linewidth=2.66, foreground="k")])

        # legend, labels, etc.
        handles, labels = ax.get_legend_handles_labels()
        handles = [copy.copy(ha) for ha in handles]
        leg = plt.legend(handles=handles, labels=labels, loc='upper left', ncols=3, frameon=False,
                        prop={'weight':'bold', 'size':'15'}, bbox_to_anchor=(0.1, 1.15))
        ax.set_xlabel('year', fontsize=18)
        ax.set_ylabel("earth mover's distance", fontsize=18)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')    
        ax.patch.set_facecolor('none')
        ax.tick_params(which='both', direction='inout', right=False, labelright=False, labelleft=True, length=10)
        if savename is None:
            savename = self.experiment_name+'_time_emd.pdf'
            plt.savefig(self.fig_path+savename, dpi=400, bbox_inches='tight', pad_inches=0.1)
        elif savename is False:
            pass
        else:
            plt.savefig(self.fig_path+savename, dpi=400, bbox_inches='tight', pad_inches=0.1)
        plt.show()

# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------

    def nana_var(self, n_analogs, savename=None, time_range=None):
        """plot the variance difference versus number of analogs and compare to model mean and perfect analogs"""
        vars = []
        nana = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 1000, 2000, 5000]
        for ii in nana:
            var = self.metrics.variance_difference(ii, prediction="analog", time_range=time_range)
            vars.append(np.nanmean(var))
        var_allmem = np.nanmean(self.metrics.variance_difference(ii, prediction="model", time_range=time_range))
        var_perfect = np.nanmean(self.metrics.variance_difference(ii, prediction="perfect", time_range=time_range))

        # now make the plot
        fig, ax = plt.subplots(1, 1, figsize=(12,5))
        # plot the analog prediction class accuracy
        ax.semilogx(nana, vars, '-o', label='analog ('+str(n_analogs)+')', lw=2.66, c=c_sunshine,
                    path_effects=[patheffects.withStroke(linewidth=4, foreground="k")])
        # plot the same for the all model mean
        ax.axhline(var_allmem, label='model mean', ls='--', lw=1.66, c=c_skyline_l,
                path_effects=[patheffects.withStroke(linewidth=2.66, foreground="k")])
        # and for the perfect analogs
        ax.axhline(var_perfect, label='perfect analogs', ls='--', lw=1.66, c=c_flowertrial,
                path_effects=[patheffects.withStroke(linewidth=2.66, foreground="k")])

        # legend, labels, etc.
        handles, labels = ax.get_legend_handles_labels()
        handles = [copy.copy(ha) for ha in handles]
        leg = plt.legend(handles=handles, labels=labels, loc='upper left', ncols=3, frameon=False, prop={'weight':'bold', 'size':'15'})
        ax.set_xlabel('number of analogs', fontsize=18)
        ax.set_ylabel(u'fractional difference\n in variance', fontsize=18)
        ax.set_title(str(self.settings['leadtime'])+' year lead, '+str(self.settings['window'])+' year window', fontsize=18)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')    
        ax.patch.set_facecolor('none')
        ax.tick_params(which='both', direction='inout', right=False, labelright=False, labelleft=True, length=10)
        if savename is None:
            savename = self.experiment_name+'_nana_var.pdf'
            plt.savefig(self.fig_path+savename, dpi=400, bbox_inches='tight', pad_inches=0.1)
        elif savename is False:
            pass
        else:
            plt.savefig(self.fig_path+savename, dpi=400, bbox_inches='tight', pad_inches=0.1)
        plt.show()

# -------------------------------------------------------------------------------

    def map_var(self, n_analogs, prediction="analog", savename=None, time_range=None, align=False):
        # get the metrics for the requested prediction type
        var_map = self.metrics.variance_difference(n_analogs, prediction=prediction, time_range=time_range)
        
        # region of target
        lats, lons = self.metrics.lats, self.metrics.lons
        lat_over_lon = (lats.max() - lats.min()) / (lons.max() - lons.min())
        lon_size, lat_size = 9, 9 * lat_over_lon / 2
        # get the coastlines
        land_feature = cfeature.NaturalEarthFeature(
            category='physical',
            name='land',
            scale='110m',
            facecolor='none',
            edgecolor='k',
            linewidth=2.5,
        )
        data_crs = ccrs.PlateCarree()

        # make the range of colorbar values consistent between prediction types?
        if align:
            ptypes = np.setdiff1d(self.predictions, prediction)
            vmin = np.min([var_map]+[self.metrics.variance_difference(n_analogs, prediction=ptype, time_range=time_range) for ptype in ptypes])
            vmax = np.max([var_map]+[self.metrics.variance_difference(n_analogs, prediction=ptype, time_range=time_range) for ptype in ptypes])
        else:
            vmin, vmax = None, None

        # make the plot
        fig = plt.figure(figsize=(lon_size, lat_size))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=215.))
        ax.add_feature(land_feature)
        image = ax.pcolormesh(lons, lats, var_map, transform=data_crs, vmin=vmin, vmax=vmax, cmap=blue_cmap)
        # add a color bar for the figure
        cax = fig.add_axes([0.3, -0.05, 0.4, 0.05])
        cbar = fig.colorbar(image, cax=cax, orientation='horizontal')
        cax.set_xlabel('var ('+prediction+')', fontsize=20)
        cax.tick_params(which='major', direction='inout', length=10, labelsize=16)
        plt.tight_layout()
        if savename is None:
            savename = self.experiment_name+'_map_var.pdf'
            plt.savefig(self.fig_path+savename, dpi=400, bbox_inches='tight', pad_inches=0.1)
        elif savename is False:
            pass
        else:
            plt.savefig(self.fig_path+savename, dpi=400, bbox_inches='tight', pad_inches=0.1)
        plt.show()

# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------

    def nana_crps(self, n_analogs, savename=None, time_range=None):
        """plot the spread over ensemble versus number of analogs and compare to model mean and perfect analogs"""
        crpss = []
        nana = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 1000, 2000, 5000]
        for ii in nana:
            crps = self.metrics.crps(ii, prediction="analog", map=False, time_range=time_range)
            crpss.append(np.nanmean(crps))
        crps_allmem = np.nanmean(self.metrics.crps(ii, prediction="model", map=False, time_range=time_range))
        crps_perfect = np.nanmean(self.metrics.crps(ii, prediction="perfect", map=False, time_range=time_range))

        # now make the plot
        fig, ax = plt.subplots(1, 1, figsize=(12,5))
        # plot the analog prediction class accuracy
        ax.semilogx(nana, crpss, '-o', label='analog ('+str(n_analogs)+')', lw=2.66, c=c_sunshine,
                    path_effects=[patheffects.withStroke(linewidth=4, foreground="k")])
        # plot the same for the all model mean
        ax.axhline(crps_allmem, label='model mean', ls='--', lw=1.66, c=c_skyline_l,
                path_effects=[patheffects.withStroke(linewidth=2.66, foreground="k")])
        # and for the perfect analogs
        ax.axhline(crps_perfect, label='perfect analogs', ls='--', lw=1.66, c=c_flowertrial,
                path_effects=[patheffects.withStroke(linewidth=2.66, foreground="k")])

        # legend, labels, etc.
        handles, labels = ax.get_legend_handles_labels()
        handles = [copy.copy(ha) for ha in handles]
        leg = plt.legend(handles=handles, labels=labels, loc='upper left', ncols=3, frameon=False, prop={'weight':'bold', 'size':'15'})
        ax.set_xlabel('number of analogs', fontsize=18)
        ax.set_ylabel('crps', fontsize=18)
        ax.set_title(str(self.settings['leadtime'])+' year lead, '+str(self.settings['window'])+' year window', fontsize=18)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')    
        ax.patch.set_facecolor('none')
        ax.tick_params(which='both', direction='inout', right=False, labelright=False, labelleft=True, length=10)
        if savename is None:
            savename = self.experiment_name+'_nana_crps.pdf'
            plt.savefig(self.fig_path+savename, dpi=400, bbox_inches='tight', pad_inches=0.1)
        elif savename is False:
            pass
        else:
            plt.savefig(self.fig_path+savename, dpi=400, bbox_inches='tight', pad_inches=0.1)
        plt.show()


# -------------------------------------------------------------------------------

    def time_crps(self, n_analogs, savename=None, time_range=None):
        # restrict to specific time range, if requested
        if time_range is not None:
            time_slice = slice(time_range[0], time_range[1])
        else:
            time_slice = slice("0000", "9999")

        # get the metrics for the analog, model, and perfect predictions
        crps_analog = self.metrics.crps(n_analogs, prediction="analog", map=False, time_range=time_range)
        crps_model = self.metrics.crps(n_analogs, prediction="model", map=False, time_range=time_range)
        crps_perfect = self.metrics.crps(n_analogs, prediction="perfect", map=False, time_range=time_range)
        # corresponding times
        time = self.metrics.truth.sel(time=time_slice)['time.year']

        # now make the plot
        fig, ax = plt.subplots(1, 1, figsize=(12,5))
        # plot the analog prediction mse
        ax.plot(time, crps_analog, '-', label='analog ('+str(n_analogs)+')', lw=2.66, c=c_sunshine,
                    path_effects=[patheffects.withStroke(linewidth=4, foreground="k")])
        # the mean across time
        ax.axhline(np.nanmean(crps_analog), ls='--', color=c_sunshine, lw=1.66, zorder=0, label='analog mean',
                path_effects=[patheffects.withStroke(linewidth=2.66, foreground="k")])
        # plot the same for the all model mean
        ax.plot(time, crps_model, '-', label='model', lw=2.66, c=c_skyline_l,
                path_effects=[patheffects.withStroke(linewidth=4, foreground="k")])
        ax.axhline(np.nanmean(crps_model), ls='--', label='model mean', lw=1.66, c=c_skyline_l, zorder=0,
                path_effects=[patheffects.withStroke(linewidth=2.66, foreground="k")])
        # and for the perfect analogs
        ax.plot(time, crps_perfect, '-', label='perfect analogs', lw=2.66, c=c_flowertrial,
                path_effects=[patheffects.withStroke(linewidth=4, foreground="k")])
        ax.axhline(np.nanmean(crps_perfect), ls='--', label='perfect mean', lw=1.66, c=c_flowertrial, zorder=0,
                path_effects=[patheffects.withStroke(linewidth=2.66, foreground="k")])

        # legend, labels, etc.
        handles, labels = ax.get_legend_handles_labels()
        handles = [copy.copy(ha) for ha in handles]
        leg = plt.legend(handles=handles, labels=labels, loc='upper left', ncols=3, frameon=False,
                        prop={'weight':'bold', 'size':'15'}, bbox_to_anchor=(0.1, 1.15))
        ax.set_xlabel('year', fontsize=18)
        ax.set_ylabel('crps', fontsize=18)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')    
        ax.patch.set_facecolor('none')
        ax.tick_params(which='both', direction='inout', right=False, labelright=False, labelleft=True, length=10)
        if savename is None:
            savename = self.experiment_name+'_time_crps.pdf'
            plt.savefig(self.fig_path+savename, dpi=400, bbox_inches='tight', pad_inches=0.1)
        elif savename is False:
            pass
        else:
            plt.savefig(self.fig_path+savename, dpi=400, bbox_inches='tight', pad_inches=0.1)
        plt.show()

# -------------------------------------------------------------------------------

    def map_crps(self, n_analogs, prediction="analog", savename=None, time_range=None, align=False):
        # get the metrics for the requested prediction type
        crps_map = self.metrics.crps(n_analogs, prediction=prediction, map=True, time_range=time_range)
        
        # region of target
        lats, lons = self.metrics.lats, self.metrics.lons
        lat_over_lon = (lats.max() - lats.min()) / (lons.max() - lons.min())
        lon_size, lat_size = 9, 9 * lat_over_lon / 2
        # get the coastlines
        land_feature = cfeature.NaturalEarthFeature(
            category='physical',
            name='land',
            scale='110m',
            facecolor='none',
            edgecolor='k',
            linewidth=2.5,
        )
        data_crs = ccrs.PlateCarree()

        # make the range of colorbar values consistent between prediction types?
        if align:
            ptypes = np.setdiff1d(self.predictions, prediction)
            vmin = np.min([crps_map]+[self.metrics.crps(n_analogs, prediction=ptype, map=True, time_range=time_range) for ptype in ptypes])
            vmax = np.max([crps_map]+[self.metrics.crps(n_analogs, prediction=ptype, map=True, time_range=time_range) for ptype in ptypes])
        else:
            vmin, vmax = None, None

        # make the plot
        fig = plt.figure(figsize=(lon_size, lat_size))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=215.))
        ax.add_feature(land_feature)
        image = ax.pcolormesh(lons, lats, crps_map, transform=data_crs, vmin=vmin, vmax=vmax, cmap=blue_cmap)
        # add a color bar for the figure
        cax = fig.add_axes([0.3, -0.05, 0.4, 0.05])
        cbar = fig.colorbar(image, cax=cax, orientation='horizontal')
        cax.set_xlabel('crps ('+prediction+')', fontsize=20)
        cax.tick_params(which='major', direction='inout', length=10, labelsize=16)
        plt.tight_layout()
        if savename is None:
            savename = self.experiment_name+'_map_crps.pdf'
            plt.savefig(self.fig_path+savename, dpi=400, bbox_inches='tight', pad_inches=0.1)
        elif savename is False:
            pass
        else:
            plt.savefig(self.fig_path+savename, dpi=400, bbox_inches='tight', pad_inches=0.1)
        plt.show()

# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------

    def nana_class(self, n_analogs, n_cells, coarsen_time=1, savename=None, time_range=None):
        """plot the class accuracy versus number of analogs and compare to model mean and perfect analogs"""
        classes = []
        nana = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 1000, 2000, 5000]
        for ii in nana:
            class_acc = self.metrics.class_accuracy(ii, n_cells, prediction="analog", map=False, coarsen_time=coarsen_time, time_range=time_range)
            classes.append(np.nanmean(class_acc))
        class_allmem = np.nanmean(self.metrics.class_accuracy(ii, n_cells, prediction="model", map=False, coarsen_time=coarsen_time, time_range=time_range))
        class_perfect = np.nanmean(self.metrics.class_accuracy(ii, n_cells, prediction="perfect", map=False, coarsen_time=coarsen_time, time_range=time_range))

        # now make the plot
        fig, ax = plt.subplots(1, 1, figsize=(12,5))
        # plot the analog prediction class accuracy
        ax.semilogx(nana, classes, '-o', label='analog ('+str(n_analogs)+')', lw=2.66, c=c_sunshine,
                    path_effects=[patheffects.withStroke(linewidth=4, foreground="k")])
        # plot the same for the all model mean
        ax.axhline(class_allmem, label='model mean', ls='--', lw=1.66, c=c_skyline_l,
                path_effects=[patheffects.withStroke(linewidth=2.66, foreground="k")])
        # and for the perfect analogs
        ax.axhline(class_perfect, label='perfect analogs', ls='--', lw=1.66, c=c_flowertrial,
                path_effects=[patheffects.withStroke(linewidth=2.66, foreground="k")])

        # legend, labels, etc.
        handles, labels = ax.get_legend_handles_labels()
        handles = [copy.copy(ha) for ha in handles]
        leg = plt.legend(handles=handles, labels=labels, loc='upper left', ncols=3, frameon=False, prop={'weight':'bold', 'size':'15'})
        ax.set_xlabel('number of analogs', fontsize=18)
        ax.set_ylabel('class accuracy ('+str(n_cells)+' classes)', fontsize=18)
        ax.set_title(str(self.settings['leadtime'])+' year lead, '+str(self.settings['window'])+' year window', fontsize=18)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')    
        ax.patch.set_facecolor('none')
        ax.tick_params(which='both', direction='inout', right=False, labelright=False, labelleft=True, length=10)
        ax.set_ylim(top=1)
        if savename is None:
            savename = self.experiment_name+'_nana_class.pdf'
            plt.savefig(self.fig_path+savename, dpi=400, bbox_inches='tight', pad_inches=0.1)
        elif savename is False:
            pass
        else:
            plt.savefig(self.fig_path+savename, dpi=400, bbox_inches='tight', pad_inches=0.1)
        plt.show()

# -------------------------------------------------------------------------------

    def time_class(self, n_analogs, n_cells, coarsen_time=1, savename=None, time_range=None):
        # restrict to specific time range, if requested
        if time_range is not None:
            time_slice = slice(time_range[0], time_range[1])
        else:
            time_slice = slice("0000", "9999")

        # get the metrics for the analog, model, and perfect predictions
        class_analog = self.metrics.class_accuracy(n_analogs, n_cells, prediction="analog", map=False, time_range=time_range, coarsen_time=coarsen_time)
        class_model = self.metrics.class_accuracy(n_analogs, n_cells, prediction="model", map=False, time_range=time_range, coarsen_time=coarsen_time)
        class_perfect = self.metrics.class_accuracy(n_analogs, n_cells, prediction="perfect", map=False, time_range=time_range, coarsen_time=coarsen_time)
        # corresponding times
        time = self.metrics.truth.sel(time=time_slice)['time.year']

        # now make the plot
        fig, ax = plt.subplots(1, 1, figsize=(12,5))
        # plot the analog prediction mse
        ax.plot(time, class_analog, '-', label='analog ('+str(n_analogs)+')', lw=2.66, c=c_sunshine,
                    path_effects=[patheffects.withStroke(linewidth=4, foreground="k")])
        # the mean across time
        ax.axhline(np.nanmean(class_analog), ls='--', color=c_sunshine, lw=1.66, zorder=0, label='analog mean',
                path_effects=[patheffects.withStroke(linewidth=2.66, foreground="k")])
        # plot the same for the all model mean
        ax.plot(time, class_model, '-', label='model', lw=2.66, c=c_skyline_l,
                path_effects=[patheffects.withStroke(linewidth=4, foreground="k")])
        ax.axhline(np.nanmean(class_model), ls='--', label='model mean', lw=1.66, c=c_skyline_l, zorder=0,
                path_effects=[patheffects.withStroke(linewidth=2.66, foreground="k")])
        # and for the perfect analogs
        ax.plot(time, class_perfect, '-', label='perfect analogs', lw=2.66, c=c_flowertrial,
                path_effects=[patheffects.withStroke(linewidth=4, foreground="k")])
        ax.axhline(np.nanmean(class_perfect), ls='--', label='perfect mean', lw=1.66, c=c_flowertrial, zorder=0,
                path_effects=[patheffects.withStroke(linewidth=2.66, foreground="k")])

        # legend, labels, etc.
        handles, labels = ax.get_legend_handles_labels()
        handles = [copy.copy(ha) for ha in handles]
        leg = plt.legend(handles=handles, labels=labels, loc='upper left', ncols=3, frameon=False,
                        prop={'weight':'bold', 'size':'15'}, bbox_to_anchor=(0.1, 1.15))
        ax.set_xlabel('year', fontsize=18)
        ax.set_ylabel('class accuracy', fontsize=18)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')    
        ax.patch.set_facecolor('none')
        ax.tick_params(which='both', direction='inout', right=False, labelright=False, labelleft=True, length=10)
        if savename is None:
            savename = self.experiment_name+'_time_class.pdf'
            plt.savefig(self.fig_path+savename, dpi=400, bbox_inches='tight', pad_inches=0.1)
        elif savename is False:
            pass
        else:
            plt.savefig(self.fig_path+savename, dpi=400, bbox_inches='tight', pad_inches=0.1)
        plt.show()

# -------------------------------------------------------------------------------

    def map_class(self, n_analogs, n_cells, coarsen_time=1, prediction="analog", savename=None, time_range=None, align=False):
        # get the metrics for the requested prediction type
        class_map = self.metrics.class_accuracy(n_analogs, n_cells=n_cells, coarsen_time=coarsen_time, prediction=prediction, time_range=time_range, map=True)
        
        # region of target
        lats, lons = self.metrics.lats, self.metrics.lons
        lat_over_lon = (lats.max() - lats.min()) / (lons.max() - lons.min())
        lon_size, lat_size = 9, 9 * lat_over_lon / 2
        # get the coastlines
        land_feature = cfeature.NaturalEarthFeature(
            category='physical',
            name='land',
            scale='110m',
            facecolor='none',
            edgecolor='k',
            linewidth=2.5,
        )
        data_crs = ccrs.PlateCarree()

        # make the range of colorbar values consistent between prediction types?
        if align:
            ptypes = np.setdiff1d(self.predictions, prediction)
            vmin = np.min([class_map]+[self.metrics.class_accuracy(n_analogs, n_cells=n_cells, coarsen_time=coarsen_time, prediction=ptype, time_range=time_range, map=True) for ptype in ptypes])
            vmax = np.max([class_map]+[self.metrics.class_accuracy(n_analogs, n_cells=n_cells, coarsen_time=coarsen_time, prediction=ptype, time_range=time_range, map=True) for ptype in ptypes])
        else:
            vmin, vmax = None, None

        # make the plot
        fig = plt.figure(figsize=(lon_size, lat_size))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=215.))
        ax.add_feature(land_feature)
        image = ax.pcolormesh(lons, lats, class_map, transform=data_crs, vmin=vmin, vmax=vmax, cmap=blue_cmap)
        # add a color bar for the figure
        cax = fig.add_axes([0.3, -0.05, 0.4, 0.05])
        cbar = fig.colorbar(image, cax=cax, orientation='horizontal')
        cax.set_xlabel('class ('+prediction+')', fontsize=20)
        cax.tick_params(which='major', direction='inout', length=10, labelsize=16)
        plt.tight_layout()
        if savename is None:
            savename = self.experiment_name+'_map_class.pdf'
            plt.savefig(self.fig_path+savename, dpi=400, bbox_inches='tight', pad_inches=0.1)
        elif savename is False:
            pass
        else:
            plt.savefig(self.fig_path+savename, dpi=400, bbox_inches='tight', pad_inches=0.1)
        plt.show()

# -------------------------------------------------------------------------------



# -------------------------------------------------------------------------------



# -------------------------------------------------------------------------------



# -------------------------------------------------------------------------------



# -------------------------------------------------------------------------------



# -------------------------------------------------------------------------------



# -------------------------------------------------------------------------------



# -------------------------------------------------------------------------------



# -------------------------------------------------------------------------------



# -------------------------------------------------------------------------------
