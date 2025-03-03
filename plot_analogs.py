import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
import matplotlib.colors as clr
import run_experiments as rexp
import best_analogs as ba
import metrics
import copy

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

class PlotAnalogs():
    """
    Class for plotting analog predictions for a given SOI.
    """
    def __init__(self, experiment_name, soi_model, soi_member, tether, variable, library_name=None, fig_path=None):
        if fig_path is None:
            fig_path = 'figures/'+experiment_name+'/'
        self.fig_path = fig_path

        # save the experiment_name
        self.experiment_name = experiment_name
        self.soi_model = soi_model
        self.soi_member = soi_member
        self.variable = variable
        self.tether = tether
        self.library_name = library_name
        # get the settings so the region name can be used in the figure
        self.settings = rexp.Experiments().get_experiment(self.experiment_name)
        self.leadtime = self.settings['leadtime']

        try:
            self.prediction_dict = ba.BestAnalogs(experiment_name, library_name=library_name).prediction_dict
            # get the soi savename corresponding to the requested soi_model, soi_member, and tether
            soi_signature = soi_model + '_mem' + str(soi_member) + '_teth' + str(tether)
            if library_name is not None:
                soi_signature = soi_signature + '_library_' + library_name
            # check the selected soi exists
            self.prediction_dict['analogs'][soi_signature]
            print('loaded analogs for', soi_signature)
        except:
            print('no previouly saved analog library or requested analogs does not exist!')

        # grab perfect analogs, analog library, and truth
        self.perfect_analogs = metrics.get_perfect_analogs(self.prediction_dict, self.experiment_name, soi_model, soi_member, variable,
                                                           global_average=False, library_name=library_name)
        self.perfect_mean = metrics.get_perfect_analogs(self.prediction_dict, self.experiment_name, soi_model, soi_member, variable,
                                                        global_average=True, library_name=library_name)
        self.analog_library = metrics.get_non_soi_analogs(self.prediction_dict, soi_model, soi_member, variable, global_average=False)
        self.library_mean = metrics.get_non_soi_analogs(self.prediction_dict, soi_model, soi_member, variable, global_average=True)
        self.truth, _, _, _ = ba.get_targets(self.experiment_name, self.soi_model, self.soi_member, 1, self.variable, 1,
                                             prediction_dict=self.prediction_dict, library_name=library_name)
        self.truth_mean, _, _, _ = ba.get_targets(self.experiment_name, self.soi_model, self.soi_member, 1, self.variable, 1,
                                                  prediction_dict=self.prediction_dict, global_mean=True, library_name=library_name)

# -------------------------------------------------------------------------------

    def plot_prediction(self, n_analogs, savename=None, time_range=None):
        # restrict to specific time range, if requested
        if time_range is not None:
            time_slice = slice(time_range[0], time_range[1])
            query_text = "time >= " + time_range[0] + " & time <= " + time_range[1]
        else:
            time_slice = slice("0000", "9999")
            query_text = "time >= 0000 & time <= 9999"

        # get the analog prediction
        _, analog, _, _ = ba.get_targets(self.experiment_name, self.soi_model, self.soi_member, self.tether, self.variable, n_analogs,
                                         prediction_dict=self.prediction_dict, global_mean=True, library_name=self.library_name)
        # now make the plot
        fig, ax = plt.subplots(1, 1, figsize=(14,5))
        # plot the true values (i.e. the soi target)
        ax.plot(self.truth_mean.sel(time=time_slice)['time.year'], self.truth_mean.sel(time=time_slice), lw=0.75, c='white', label='truth',
                path_effects=[patheffects.withStroke(linewidth=4, foreground=c_midnight)])
        # plot the prediction
        ax.plot(analog.query(time=query_text)['time'], analog.query(time=query_text), '-', label='analog prediction ('+str(n_analogs)+')',
                lw=2.66, c=c_sunshine, path_effects=[patheffects.withStroke(linewidth=4, foreground="k")])

        # legend, labels, etc.
        handles, labels = ax.get_legend_handles_labels()
        handles = [copy.copy(ha) for ha in handles]
        # [ha.set_alpha(0.7) for ha in handles[:1]]
        leg = plt.legend(handles=handles, labels=labels, loc='upper left', ncols=2, frameon=False, prop={'weight':'bold', 'size':'15'})

        ax.set_xlabel('prediction year', fontsize=18)
        ax.set_ylabel(self.settings['region']+'\n'+self.variable, fontsize=18)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')    
        ax.patch.set_facecolor('none')
        ax.tick_params(which='both', direction='inout', right=False, labelright=False, labelleft=True, length=10)
        if savename is None:
            savename = self.experiment_name+'_prediction.pdf'
            plt.savefig(self.fig_path+savename, dpi=400, bbox_inches='tight', pad_inches=0.1)
        elif savename is False:
            pass
        else:
            plt.savefig(self.fig_path+savename, dpi=400, bbox_inches='tight', pad_inches=0.1)
        plt.show()

# -------------------------------------------------------------------------------

    def plot_range(self, n_analogs, savename=None, time_range=None):
        # restrict to specific time range, if requested
        if time_range is not None:
            time_slice = slice(time_range[0], time_range[1])
            query_text = "time >= " + time_range[0] + " & time <= " + time_range[1]
        else:
            time_slice = slice("0000", "9999")
            query_text = "time >= 0000 & time <= 9999"

        # get the targets for the requested soi and variable
        _, _, prediction_min, prediction_max = ba.get_targets(self.experiment_name, self.soi_model, self.soi_member, self.tether, self.variable, n_analogs,
                                                              prediction_dict=self.prediction_dict, global_mean=True, library_name=self.library_name)
        # get the full range of analog predictions
        all_min, all_max = ba.get_full_analog_range(self.experiment_name, self.prediction_dict, self.soi_model, self.soi_member, self.variable, global_mean=True)

        # now make the plot
        fig, ax = plt.subplots(1, 1, figsize=(14,5))
        # plot the range of library
        ax.fill_between(all_min['time.year'].sel(time=time_slice), all_min.sel(time=time_slice), all_max.sel(time=time_slice), color=c_skyline_ll,
                         alpha=0.33, label='model range', path_effects=[patheffects.withStroke(linewidth=1, foreground=c_skyline_l)])
        # plot the range of all the analogs
        ax.fill_between(prediction_min.query(time=query_text)['time'], prediction_min.query(time=query_text), prediction_max.query(time=query_text), color=c_sunshine,
                        alpha=0.3, label='analog range', path_effects=[patheffects.withStroke(linewidth=1, foreground=c_sunshine)])
        # plot the true values (i.e. the soi target)
        ax.plot(self.truth_mean.sel(time=time_slice)['time.year'], self.truth_mean.sel(time=time_slice), lw=0.75, c='white', label='truth',
                path_effects=[patheffects.withStroke(linewidth=4, foreground=c_midnight)])

        # legend, labels, etc.
        handles, labels = ax.get_legend_handles_labels()
        handles = [copy.copy(ha) for ha in handles]
        [ha.set_alpha(0.7) for ha in handles[:2]]
        leg = plt.legend(handles=handles, labels=labels, loc='upper left', ncols=2, frameon=False, prop={'weight':'bold', 'size':'15'})

        ax.set_xlabel('prediction year', fontsize=18)
        ax.set_ylabel(self.settings['region']+'\n'+self.variable, fontsize=18)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')    
        ax.patch.set_facecolor('none')
        ax.tick_params(which='both', direction='inout', right=False, labelright=False, labelleft=True, length=10)
        if savename is None:
            savename = self.experiment_name+'_range.pdf'
            plt.savefig(self.fig_path+savename, dpi=400, bbox_inches='tight', pad_inches=0.1)
        elif savename is False:
            pass
        else:
            plt.savefig(self.fig_path+savename, dpi=400, bbox_inches='tight', pad_inches=0.1)
        plt.show()

# -------------------------------------------------------------------------------

    def plot_model_mean(self, time_range=None, savename=None):
        # restrict to specific time range, if requested
        if time_range is not None:
            time_slice = slice(time_range[0], time_range[1])
            query_text = "time >= " + time_range[0] + " & time <= " + time_range[1]
        else:
            time_slice = slice("0000", "9999")
            query_text = "time >= 0000 & time <= 9999"

        # get the all model mean
        all_model_mean = ba.get_full_analog_range(self.experiment_name, self.prediction_dict, self.soi_model, self.soi_member,
                                                  self.variable, global_mean=True, minmax=False)[:-self.leadtime]

        # now make the plot
        fig, ax = plt.subplots(1, 1, figsize=(14,5))
        # plot the true values (i.e. the soi target)
        ax.plot(self.truth_mean.sel(time=time_slice)["time.year"], self.truth_mean.sel(time=time_slice), lw=0.75, c='white', label='truth',
                path_effects=[patheffects.withStroke(linewidth=4, foreground=c_midnight)])
        # plot the model mean of library
        ax.plot(all_model_mean.sel(time=time_slice)["time.year"], all_model_mean.sel(time=time_slice), label='model mean',
                color=c_skyline_l, lw=3, path_effects=[patheffects.withStroke(linewidth=4.66, foreground="k")])

        # legend, labels, etc.
        handles, labels = ax.get_legend_handles_labels()
        handles = [copy.copy(ha) for ha in handles]
        # [ha.set_alpha(0.7) for ha in handles[:1]]
        leg = plt.legend(handles=handles, labels=labels, loc='upper left', ncols=2, frameon=False, prop={'weight':'bold', 'size':'15'})

        ax.set_xlabel('prediction year', fontsize=18)
        ax.set_ylabel(self.settings['region']+'\n'+self.variable, fontsize=18)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')    
        ax.patch.set_facecolor('none')
        ax.tick_params(which='both', direction='inout', right=False, labelright=False, labelleft=True, length=10)
        if savename is None:
            savename = self.experiment_name+'_model_mean.pdf'
            plt.savefig(self.fig_path+savename, dpi=400, bbox_inches='tight', pad_inches=0.1)
        elif savename is False:
            pass
        else:
            plt.savefig(self.fig_path+savename, dpi=400, bbox_inches='tight', pad_inches=0.1)
        plt.show()

# -------------------------------------------------------------------------------
        
    def plot_prediction_suite(self, n_analogs, time_range=None, savename=None):
        # restrict to specific time range, if requested
        if time_range is not None:
            time_slice = slice(time_range[0], time_range[1])
            query_text = "time >= " + time_range[0] + " & time <= " + time_range[1]
        else:
            time_slice = slice("0000", "9999")
            query_text = "time >= 0000 & time <= 9999"

        # get the analog prediction
        _, analog, analog_min, analog_max = ba.get_targets(self.experiment_name, self.soi_model, self.soi_member, self.tether, self.variable, n_analogs,
                                                           prediction_dict=self.prediction_dict, global_mean=True, library_name=self.library_name)
        # get the full range of all models and the model mean
        all_min, all_max = ba.get_full_analog_range(self.experiment_name, self.prediction_dict, self.soi_model, self.soi_member, self.variable, global_mean=True)
        all_model_mean = ba.get_full_analog_range(self.experiment_name, self.prediction_dict, self.soi_model, self.soi_member,
                                                  self.variable, global_mean=True, minmax=False)[:-self.leadtime]
        # get the perfect analog prediction
        perfect_analogs = self.perfect_mean

        # now make the figure
        fig, ax = plt.subplots(4, 1, figsize=(14,5*4))
        # --------------- plot the analog prediction and truth --------------- #
        # plot the true values (i.e. the soi target)
        ax[0].plot(self.truth_mean.sel(time=time_slice)['time.year'], self.truth_mean.sel(time=time_slice), lw=0.75, c='white', label='truth',
                   path_effects=[patheffects.withStroke(linewidth=4, foreground=c_midnight)])
        # plot the prediction
        ax[0].plot(analog.query(time=query_text)['time'], analog.query(time=query_text), '-', label='analog prediction ('+str(n_analogs)+')',
                   lw=2.66, c=c_sunshine, path_effects=[patheffects.withStroke(linewidth=4, foreground="k")])
        # --------------- plot the ranges and truth (refining spread) --------------- #
        # plot the range of library
        ax[1].fill_between(all_min.sel(time=time_slice)['time.year'], all_min.sel(time=time_slice), all_max.sel(time=time_slice), color=c_skyline_ll,
                           alpha=0.33, label='model range', path_effects=[patheffects.withStroke(linewidth=1, foreground=c_skyline_l)])
        # plot the range of all the analogs
        ax[1].fill_between(analog_min.query(time=query_text)['time'], analog_min.query(time=query_text), analog_max.query(time=query_text), color=c_sunshine,
                           alpha=0.3, label='analog range', path_effects=[patheffects.withStroke(linewidth=1, foreground=c_sunshine)])
        # plot the true values (i.e. the soi target)
        ax[1].plot(self.truth_mean.sel(time=time_slice)['time.year'], self.truth_mean.sel(time=time_slice), lw=0.75, c='white', label='truth',
                   path_effects=[patheffects.withStroke(linewidth=4, foreground=c_midnight)])
        # --------------- plot the model mean and truth --------------- #
        # plot the true values (i.e. the soi target)
        ax[2].plot(self.truth_mean.sel(time=time_slice)['time.year'], self.truth_mean.sel(time=time_slice), lw=0.75, c='white', label='truth',
                   path_effects=[patheffects.withStroke(linewidth=4, foreground=c_midnight)])
        # plot the model mean of library
        ax[2].plot(all_model_mean.sel(time=time_slice)['time.year'], all_model_mean.sel(time=time_slice), label='model mean',
                   color=c_skyline_l, lw=3, path_effects=[patheffects.withStroke(linewidth=4.66, foreground="k")])
        # --------------- plot the perfect analogs and truth --------------- #
        # plot the true values (i.e. the soi target)
        ax[3].plot(self.truth_mean.sel(time=time_slice)['time.year'], self.truth_mean.sel(time=time_slice), lw=0.75, c='white', label='truth',
                   path_effects=[patheffects.withStroke(linewidth=4, foreground=c_midnight)])
        ax[3].plot(perfect_analogs.sel(soi_time=time_slice)['soi_time.year'], perfect_analogs.sel(soi_time=time_slice), label='perfect analogs',
                   lw=2.66, c=c_flowertrial, alpha=0.5)

        # set up the legend and axes for each subplot
        for ii in range(4):
            # legend, labels, etc.
            handles, labels = ax[ii].get_legend_handles_labels()
            handles = [copy.copy(ha) for ha in handles]
            if ii == 1:
                [ha.set_alpha(0.7) for ha in handles[:2]]
            leg = ax[ii].legend(handles=handles, labels=labels, loc='upper left', ncols=2, frameon=False, prop={'size':'15', 'style':'oblique'})
            ax[ii].spines['right'].set_visible(False)
            ax[ii].spines['top'].set_visible(False)
            ax[ii].yaxis.set_ticks_position('left')
            ax[ii].xaxis.set_ticks_position('bottom')    
            ax[ii].patch.set_facecolor('none')
            ax[ii].tick_params(which='both', direction='inout', right=False, labelright=False, labelleft=True, length=10, labelsize=16)
        # x and y labels, title
        ax[3].set_xlabel('prediction year', fontsize=18)
        # add y label just once in the middle
        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.grid(False)
        plt.ylabel(self.variable, fontsize=22, labelpad=15)

        plt.subplots_adjust(hspace=0.17)
        if savename is None:
            savename = self.experiment_name+'_suite.pdf'
            plt.savefig(self.fig_path+savename, dpi=400, bbox_inches='tight', pad_inches=0.1)
        elif savename is False:
            pass
        else:
            plt.savefig(self.fig_path+savename, dpi=400, bbox_inches='tight', pad_inches=0.1)
        plt.show()
        
# -------------------------------------------------------------------------------

    def plot_model_distribution(self, n_analogs, time_range=None, savename=None):
        """plot the number of times each model was selected as a top n_analogs analog"""
        # restrict to specific time range, if requested
        if time_range is not None:
            time_slice = slice(time_range[0], time_range[1])
            query_text = "time >= " + time_range[0] + " & time <= " + time_range[1]
        else:
            time_slice = slice("0000", "9999")
            query_text = "time >= 0000 & time <= 9999"

        # get the analog predictions to obtain the models selected
        _, analog, _, _ = ba.get_targets(self.experiment_name, self.soi_model, self.soi_member, self.tether, self.variable, n_analogs,
                                         global_mean=True, return_average=False, prediction_dict=self.prediction_dict, library_name=self.library_name)
        analog = analog.query(time=query_text)
        
        # get the models that were selected for the analogs of this soi
        selected_model = analog.model
        # get the names of the models that were selected for this soi across all times
        models, model_counts = np.unique(selected_model, return_counts=True)
        # sort by total number of selections for each model (high to low) -- for keeping it to top ten
        sorted = np.argsort(model_counts)[::-1]
        models = models[sorted]
        model_counts = model_counts[sorted]

        # convert selected models to integers
        selected_model_int = np.empty(selected_model.shape, dtype=int)
        model_member_size = np.empty(models.size, dtype=int)
        for ii, model in enumerate(models):
            selected_model_int[np.where(selected_model == model)] = int(ii)
            # number of each model in the full library
            model_member_size[ii] = np.unique(self.prediction_dict['inputs'].member[
                np.where(self.prediction_dict['inputs'].model == model)[0]
                ]).size

        # make the plot
        fig, ax = plt.subplots(1, 1, figsize=(np.min([models.size, 10]), 5))
        # histogram, stacked across ranking of analog (best, second best, etc.)
        n, bins, patches = ax.hist(selected_model_int, stacked=True, bins=np.arange(models.size+1), density=True,)
        # get the bin centers for placing the labels and coloring
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        # the relevant bins are those with non-zero entries
        relevant_bins = np.where(np.any(n != 0, axis=0))[0]
        for c, p in zip(np.linspace(0, 1, n_analogs), patches):
            plt.setp(p, 'facecolor', red_cmap(c))
            plt.setp(p, 'edgecolor', 'k')
        # wrap the model name on the first dash appearing in the name (nicer printing)
        for ii, model in enumerate(models):
            model_parts = model.replace('-', '\n', 1)
            models[ii] = model_parts
        for ii in range(np.min([models.size, 10])):
            ax.text(bin_centers[relevant_bins[ii]], n[-1, ii]+0.01, str(model_member_size[ii]) + '/' + str(model_member_size.sum()) +
                    '\n(' + str(np.round(model_member_size[ii]/model_member_size.sum(), 2)) + ')', ha='center', fontsize=8)
        ax.set_xticks(bin_centers[relevant_bins][:10], models[:10], fontsize=10)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')    
        ax.patch.set_facecolor('none')
        ax.tick_params(which='both', direction='inout', right=False, labelright=False, labelleft=True, length=10)
        ax.tick_params(which='both', axis='x', bottom=False)
        ax.set_xlim([bins[0], bins[np.min([models.size, 10])]])
        if savename is None:
            savename = self.experiment_name+'_model_dist.pdf'
            plt.savefig(self.fig_path+savename, dpi=400, bbox_inches='tight', pad_inches=0.1)
        elif savename is False:
            pass
        else:
            plt.savefig(self.fig_path+savename, dpi=400, bbox_inches='tight', pad_inches=0.1)
        plt.show()

# -------------------------------------------------------------------------------

    def plot_model_times(self, n_analogs, time_range=None, savename=None):
        # restrict to specific time range, if requested
        if time_range is not None:
            time_slice = slice(time_range[0], time_range[1])
            query_text = "time >= " + time_range[0] + " & time <= " + time_range[1]
        else:
            time_slice = slice("0000", "9999")
            query_text = "time >= 0000 & time <= 9999"
        
        # get the targets for the requested soi and variable to restrict the times
        _, analog, _, _ = ba.get_targets(self.experiment_name, self.soi_model, self.soi_member, self.tether, self.variable, n_analogs,
                                         global_mean=True, return_average=False, prediction_dict=self.prediction_dict, library_name=self.library_name)
        
        xtimes = analog.sel(time=time_slice)['time']
        # get the models that were selected for the analogs of this soi
        selected_model = analog.sel(time=time_slice).model
        # get the names of the models that were selected for this soi across all times
        models = np.unique(selected_model)

        # sort by total number of selections for each model (high to low) -- for keeping it to top ten
        model_counts = np.unique(selected_model, return_counts=True)[1]
        sorted = np.argsort(model_counts)[::-1]
        models = models[sorted]
        model_counts = model_counts[sorted]

        # convert selected models to integers
        selected_model_int = np.zeros([selected_model.shape[0], models.size], dtype=int)
        for ii, model in enumerate(models):
            selected_model_int[:, ii] = np.sum(selected_model.to_numpy() == model, axis=1)
        # normalize to maximum count -- this just makes the relative color intensity look better (spans 0 to 1)
        selected_model_int = selected_model_int/selected_model_int.max()

        # make the plot
        fig, ax = plt.subplots(1, 1, figsize=(10, np.min([models.size/2, 3.5])))
        ax.pcolormesh(xtimes, models[:10], selected_model_int[:, :10].T, cmap=alpha_cmap)
        ax.set_yticks(np.arange(models[:10].size), models[:10], fontsize=11)
        ax.set_xlim([xtimes.min()-2, xtimes.max()+2])
        ax.set_ylim([0-0.5, np.min([models.size, 10])-0.5])  
        ax.set_xlabel('prediction year', fontsize=14)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')    
        ax.patch.set_facecolor('none')
        ax.tick_params(which='both', direction='inout', right=False, labelright=False, labelleft=True, length=0)
        if savename is None:
            savename = self.experiment_name+'_model_times.pdf'
            plt.savefig(self.fig_path+savename, dpi=400, bbox_inches='tight', pad_inches=0.1)
        elif savename is False:
            pass
        else:
            plt.savefig(self.fig_path+savename, dpi=400, bbox_inches='tight', pad_inches=0.1)
        plt.show()
        
# -------------------------------------------------------------------------------

    def plot_selected_times(self, n_analogs, roll_years, time_range=None, savename=None):
        # restrict to specific time range, if requested
        if time_range is not None:
            time_slice = slice(time_range[0], time_range[1])
            query_text = "time >= " + time_range[0] + " & time <= " + time_range[1]
        else:
            time_slice = slice("0000", "9999")
            query_text = "time >= 0000 & time <= 9999"

        # roll overs some years (centered, so roll_years needs to be odd)
        if roll_years % 2 == 0:
            roll_years = roll_years - 1
            print("roll years should be odd, using", roll_years)
        spread = int((roll_years - 1)/2)
        
        # get the indices for the predicted year (indices where truth is not nan + leadtime)
        _, analog, _, _ = ba.get_targets(self.experiment_name, self.soi_model, self.soi_member, self.tether, self.variable, n_analogs,
                                         global_mean=True, return_average=False, prediction_dict=self.prediction_dict, library_name=self.library_name)
        # get the x and y axes (year predicted -- 1 per year, and year of analogs -- n_analogs per year)
        pred_years = analog.query(time=query_text)['time']
        analog_years = analog.query(time=query_text)['analog_time']

        # getting the rolling mean, min, max
        best_analog = np.zeros(pred_years.size - roll_years + 1)
        min_analog = np.zeros(pred_years.size - roll_years + 1)
        max_analog = np.zeros(pred_years.size - roll_years + 1)
        for ii in range(spread, best_analog.size+spread):
            best_analog[ii-spread] = analog_years[ii-spread:ii+spread+1].mean()
            # min_analog[ii-spread] = np.quantile(analog_years[ii], 0.2)
            # max_analog[ii-spread] = np.quantile(analog_years[ii], 0.8)
            min_analog[ii-spread] = np.nanmin(analog_years[ii])
            max_analog[ii-spread] = np.nanmax(analog_years[ii])
        # roll the soi_times over the same period
        pred_years = pred_years.rolling(time=roll_years, center=False).mean().dropna("time", how="all")

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.errorbar(pred_years, best_analog, yerr=np.array([abs(best_analog-min_analog), abs(max_analog-best_analog)]), fmt='o', 
                    elinewidth=1.1, markersize=4, c=alpha_colors[-1])
        ax.plot(pred_years, pred_years, c=c_midnight)

        ax.set_xlabel('prediction year', fontsize=14)
        ax.set_ylabel('analog year', fontsize=14)
        ax.set_xlim([np.min(pred_years)-5, np.max(pred_years)+5])
        ax.set_ylim([np.min(analog_years)-5, np.max(analog_years)+5])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')    
        ax.patch.set_facecolor('none')
        ax.tick_params(which='both', direction='inout', right=False, labelright=False, labelleft=True, length=10)
        if savename is None:
            savename = self.experiment_name+'_timecomp.pdf'
            plt.savefig(self.fig_path+savename, dpi=400, bbox_inches='tight', pad_inches=0.1)
        elif savename is False:
            pass
        else:
            plt.savefig(self.fig_path+savename, dpi=400, bbox_inches='tight', pad_inches=0.1)
        plt.show()

# -------------------------------------------------------------------------------

    def plot_constrained_prediction(self, n_analogs, max_year, time_range=None, savename=None):
        # restrict to specific time range, if requested
        if time_range is not None:
            time_slice = slice(time_range[0], time_range[1])
            query_text = "time >= " + time_range[0] + " & time <= " + time_range[1]
        else:
            time_slice = slice("0000", "9999")
            query_text = "time >= 0000 & time <= 9999"

        # get the targets for the requested soi and variable
        _, analog, analog_min, analog_max = ba.get_targets(self.experiment_name, self.soi_model, self.soi_member, self.tether, self.variable, n_analogs,
                                                           return_average=True, global_mean=True, prediction_dict=self.prediction_dict, library_name=self.library_name)
        _, c_analog, c_analog_min, c_analog_max = ba.get_targets(self.experiment_name, self.soi_model, self.soi_member, self.tether, self.variable, n_analogs,
                                                                 return_average=True, global_mean=True, prediction_dict=self.prediction_dict,max_year=max_year,
                                                                 library_name=self.library_name)
        # now make the plot
        fig, ax = plt.subplots(1, 1, figsize=(14,5))
        # plot the range of all the best analogs (unconstrained)
        ax.fill_between(analog_min.query(time=query_text)['time'], analog_min.query(time=query_text), analog_max.query(time=query_text), color=c_flowertrial,
                        alpha=0.3, path_effects=[patheffects.withStroke(linewidth=1, foreground=c_flowertrial, alpha=0.3)])
        # plot the range of all the best analogs (constrained)
        ax.fill_between(c_analog_min.query(time=query_text)['time'], c_analog_min.query(time=query_text), c_analog_max.query(time=query_text), color=c_horsetooth,
                        alpha=0.3, path_effects=[patheffects.withStroke(linewidth=1, foreground=c_horsetooth, alpha=0.3)])
        # plot the true values (i.e. the soi target)
        ax.plot(self.truth_mean.sel(time=time_slice)['time.year'], self.truth_mean.sel(time=time_slice), lw=3.33, c=c_midnight, label='truth',
                path_effects=[patheffects.withStroke(linewidth=5, foreground='white')])
        # plot the prediction (unconstrained)
        ax.plot(analog.query(time=query_text)['time'], analog.query(time=query_text), label='analogs ('+str(n_analogs)+')',
                lw=3.33, c=c_flowertrial, path_effects=[patheffects.withStroke(linewidth=5, foreground="k")])
        # plot the prediction (unconstrained)
        ax.plot(c_analog.query(time=query_text)['time'], c_analog.query(time=query_text), label='constrained analogs ('+str(n_analogs)+')',
                lw=3.33, c=c_horsetooth, path_effects=[patheffects.withStroke(linewidth=5, foreground="k")])

        # legend, labels, etc.
        handles, labels = ax.get_legend_handles_labels()
        handles = [copy.copy(ha) for ha in handles]
        leg = plt.legend(handles=handles, labels=labels, loc='upper left', ncols=2, frameon=False, prop={'weight':'bold', 'size':'15'})

        ax.set_xlabel('prediction year', fontsize=18)
        ax.set_ylabel(self.settings['region']+'\n'+self.variable, fontsize=18)
        ax.set_title(self.experiment_name+r', $\leq$'+str(max_year), fontsize=18)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')    
        ax.patch.set_facecolor('none')
        ax.tick_params(which='both', direction='inout', right=False, labelright=False, labelleft=True, length=10)
        if savename is None:
            savename = self.experiment_name+'_constrained_prediction.pdf'
            plt.savefig(self.fig_path+savename, dpi=400, bbox_inches='tight', pad_inches=0.1)
        elif savename is False:
            pass
        else:
            plt.savefig(self.fig_path+savename, dpi=400, bbox_inches='tight', pad_inches=0.1)
        plt.show()

# -------------------------------------------------------------------------------

# -------------------------------------------------------------------------------
