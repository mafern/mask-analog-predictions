<h1 align="center">analog forecasting with an ml mask</h1>

*code base for paper xxx:
regional climate predictions on multi-year to decadal time scales using the analog forecasting approach, with a twist
-- a machine learning derived mask of weights highlights the most important regions for matching analogs to prediction states,
leading to improved analog selection*

***
<p align="center">
  <img src="https://github.com/mafern/mask-analog-projections/assets/39971310/ba8f8c2d-ab25-4a17-a462-be84aa1b3c72" width="70%" height="70%" border="30%"/>
</p>

**the above schematic shows the machine learning component of the full prediction framework:**
- two maps are taken (not necessarily from the same model, member, or time) and their difference is scaled by a randomly initialized mask of weights
- the mean squared value of this weighted difference map is passed to a neural network tasked with predicting the mean-squared-error between the two maps at a future time
- the randomly initialized mask of weights is updated through training until the network predicts future mse from weighted map difference well
- the final mask of weights can then be used in an analog forecasting framework

> [!NOTE]
> this code base has been written to primarily use monthly or yearly data, which can come from either earth system models or observations (though the requisite number of samples makes observationally based masks impractical)

<h2 align="center">code usage</h2>

***
to create a machine learnt mask: define a new experiment, add it to the experiment json file, then (make_mask) preprocess data, build and train the network
```python
import run_experiments as rexp
# create an experiment with desired variable, region, lead time and window, data,
# preprocessing steps, and network architecture -- see documentation for all the options
experiments = rexp.Experiments('/path/to/experiments.json')
new_exp = experiments.make_exp_dictionary('exp_name', leadtime=1, window=1, variable_list=['tas',], analog_models=['ACCESS-CM2', 'CESM2'], soi_models=['ACCESS-CM2', 'CESM2'], split=[5,2,2,1], prediction_nodes=[5,5,5], prediction_activation='elu', learning_rate=0.001, output_period='1M', region='western us', baseline=[1961, 1990])
experiments.add_experiment(new_exp)

# train the neural network using the architecture and data for the given experiment
experiments.make_mask('exp_name')
```

once the make_mask function is finished the preprocessed data (processed_data), the trained network (saved_models), and a mask file (saved_masks) will all be saved

> [!TIP]
> if the training dataset is large, running the make_mask step in terminal can help avoid kernel crashes

<p align="center">
<ins>example mask of weights</ins>
</p>
<p align="center">
<img src="https://github.com/mafern/mask-analog-projections/assets/39971310/eb9ab270-e23d-4c3d-a666-be28d0eaabb9" width="70%" height="70%" border="30%"/>
</p>

you can then make a prediction by selecting a state of interest (SOI, the state from which you want to predict) and finding the best analogs
```python
import best_analogs as ba
# first, generate a library of analogs -- this should go through many (but not all!) of the same preprocessing
# steps as the training data (this will remove members that were used in the training splits automatically)
library = ba.BestAnalogs('exp_name', model_list=["ACCESS-ESM1-5", "CESM2", "MIROC-ES2L", "MPI-ESM1-2-LR"], n_members=-1)
# there is no distinction between SOIs and analogs at this point, so observations can also be added to this library
library = ba.BestAnalogs('exp_name', model_list=["BEST"], n_members=-1, data_period='yearly')
# finally, select an SOI and rank all of the remaining library from best to worst match (saved in saved_analogs)
library.get_all_soi_analogs("BEST", 0);
```

> [!NOTE]
> preprocessing steps used for the training data that should **not** be part of the library generation are automatically removed -- for example, the data does not need to be normalized or synced by time for the library

> [!TIP]
> if the library is large, run all of the above in terminal to avoid kernel crashes

various plots can be made automatically from the library object, for example
```python
library.plot_projection("BEST", 0, 'tas', 2)
```
<p align="center">
<img src="https://github.com/mafern/mask-analog-projections/assets/39971310/00811ca5-4762-4e06-b635-d54e1d0e9099" width="80%" height="80%" border="20%"/>
</p>


<h2 align="center">python packages</h2>

***
> [!IMPORTANT]
> most of the following packages require the conda-forge channel, versions given below are sufficient, but not necessarily required
```bash
conda create --name analog-env python=3.11.4
conda activate analog-env

conda install cartopy=0.22.0 esmf=8.4.2 hdf5=1.14.1 json5=0.9.14 matplotlib=3.7.2 regionmask=0.10.0 xarray=2023.7.0 xesmf=0.7.1 pip=23.2.1

pip install numpy==1.24.3
pip install tensorflow==2.13.0
pip install tensorflow-macos==2.13.0
pip install silence-tensorflow==1.2.1
pip install keras==2.13.1
pip install ipykernel
```

<h2 align="center">references</h2>

***
[Rader, Jamin K., and Elizabeth A. Barnes. 2023. “Optimizing Seasonal-To-Decadal Analog Forecasts With a Learned Spatially-Weighted Mask.” Geophysical Research Letters 50 (23): e2023GL104983.](https://doi.org/10.1029/2023GL104983)


