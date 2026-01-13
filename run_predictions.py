"""for the specified regions, lead times, and variables: learn masks and pick analogs"""

##########################################################################################
######################## import statements, helper function ##############################
##########################################################################################
import run_experiments as rexp
# create an Experiments object which allows you to access previous experiments and define new ones (and train a mask)
exp_object = rexp.Experiments('experiments.json')
import best_analogs as ba
# hide some warnings that are not helpful
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
def abbrev_regionname(rname):
    vowels = ['a', 'e', 'i', 'o', 'u']
    final_name = ''
    rparts = rname.split(' ')
    for rpart in rparts:
        final_name += rpart[:1]
        for rchar in rpart[1:]:
            if rchar not in vowels:
                final_name += rchar
    return final_name

##########################################################################################
####################### set regions, leads, variables ####################################
##########################################################################################
# pick a region or several -- these can be from regionmask or from regions.json
# to see a list of available regions, run this code with regionnames empty
regionnames = []
if regionnames == []:
    exp_object.list_regions
else:
    # if the region is not in the library (regionmask), and is not a simple shape (can be added to regions.json), include a shapefile (path to shapefile)
    # NOTE: this overrides the region selection, but not the naming, so include a name for the region in regionnames still!
    shapefile = None
    # pick a variable
    variable = ['tas']
    # pick a lead time or several
    leadtimes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # state of interest, i.e., the ground truth and what you want to match against
    soi_model = 'BEST' # 'ERA5'
    # member, in case you want to use a model as truth
    soi_member = 0
    # tether length (matching period) -- needs further exploration, but runs fastest if set to 1
    tethers = 1
    # which models to train on -- this is the full set, but can be pared down if desired
    all_models = ['GISS-E2-2-G', 'NorESM2-LM', 'ACCESS-ESM1-5', 'CanESM5-1', 'CMCC-CM2-HR4', 'CESM2', 'AWI-ESM-1-1', 'CAMS-CSM1', 'CMCC-ESM2', 'CESM2-WACCM', 'MPI-ESM1-2-HR', 'AWI-CM-1-1', 'MIROC', 'MPI-ESM1-2-LR', 'GISS-E2-1-G', 'CNRM-CM6-1', 'CMCC-CM2-SR5', 'MRI-ESM2-0', 'MIROC-ES2H', 'CanESM5-CanOE', 'ACCESS-CM2', 'BCC-ESM1', 'UKESM1-1-LL', 'UKESM1-0-LL', 'CanESM5', 'BCC-CSM2', 'TaiESM', 'NorESM2-MM', 'MIROC-ES2L']


    ##########################################################################################
    ####################### learn masks, pick analogs, and save ##############################
    ##########################################################################################
    for rr, region in enumerate(regionnames): # for each region
        # get region short name
        region_shortname = abbrev_regionname(region)
        for ii, lt in enumerate(leadtimes): # for each lead time
            # experiment name is region short name + lead time
            experiment_name = region_shortname[rr]+'_lt'+str(lt)
            try:
                # define the experiment -- these variables can be played with, but for now, best to leave as the defaults
                new_exp = exp_object.make_exp_dictionary(experiment_name, variable_list=variable, analog_models=all_models, soi_models=all_models, leadtime=lt, split=[5, 3, 2], region=region, shapefile=shapefile, val_models=all_models, target_var=variable, window=False, trajectory=1, resolutions=1, time_range=False, norm_data=True, baseline=[1961, 1990], remove_trend=False, norm_weights=True, transfer_exp=None, mask=False, prediction_nodes=[], prediction_activation='elu', learning_rate=0.001, loss='mse', patience=50, min_delta=0.005, batch_size=64, val_batch_size=2500, max_epochs=5000, mask_activation='relu', mask_init="ones", mask_l1=0.0, mask_l2=0.0, seed=20)
                # add experiment to the json file
                exp_object.add_experiment(new_exp)
            except:
                print('experiment already defined')

            # learn the mask
            exp_object.make_mask(experiment_name)
            # plot the mask
            exp_object.plot_mask(experiment_name, colorbar=False)
            
            # make the library for this lead time and region
            if lt == 1:
                try:
                    library = ba.BestAnalogs(experiment_name)
                except:
                    library = ba.BestAnalogs(experiment_name, model_list=all_models+[soi_model], n_members=-1)
            else:
                library = ba.BestAnalogs(experiment_name, library_name=region_shortname[rr]+'_lt1')
            # get analogs for learned mask -- saves in the saved_analogs directory
            library.get_all_soi_analogs(soi_model, soi_member, tether=tethers)

            ##########################################################################################
            ################ OPTIONAL: get analogs for global and regional mask ######################
            ##########################################################################################
            # library.get_all_soi_analogs(soi_model, soi_member, tether=tethers, savename='global_t'+str(tethers), mask_type='global')
            # library.get_all_soi_analogs(soi_model, soi_member, tether=tethers, savename='regional_t'+str(tethers), mask_type='regional')
