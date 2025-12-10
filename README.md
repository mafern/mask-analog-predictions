<h1 align="center">analog forecasting with a machine learning mask</h1>

*code base for paper xxx:
regional climate predictions on multi-year to decadal time scales using an analog forecasting approach with a machine learning derived mask of weights, which highlights the most important precursor regions for matching analogs to prediction states*

***
<p align="center">
  <img src="https://github.com/mafern/mask-analog-predictions/blob/main/figures/main_schematic.png" width="70%" height="70%" border="30%"/>
</p>


**the above schematic shows the machine learning component of the full prediction framework:**
- two maps are taken (not necessarily from the same model, member, or time) and scaled by a randomly initialized mask of weights
- the mean-squared-difference between the weighted maps is passed to a linear node that predicts the target region mean-squared-difference at a future time
- the randomly initialized mask of weights is updated through training until there is a strong correspondence between input (global) mse and output (target region) future mse
- the final mask of weights can then be used in an analog forecasting framework

> [!NOTE]
> this code base has been written to use yearly data, which can come from either earth system models or observations

> [!IMPORTANT]
> ensure that you have all the same models/members when matching on multiple variables

<h2 align="center">code usage</h2>

steps for running the full framework (from defining a new experiment, to learning the mask, to identifying the best analogs and plotting predictions) can be see in the new_experiment.ipynb notebook

> [!TIP]
> if the training dataset is large, training the mask and finding the best analogs can be done in terminal to avoid kernel crashes

<p align="center">
<ins>example mask of weights</ins>
</p>
<p align="center">
<img src="https://github.com/mafern/mask-analog-projections/assets/39971310/eb9ab270-e23d-4c3d-a666-be28d0eaabb9" width="70%" height="70%" border="30%"/>
</p>

<h2 align="center">python packages</h2>

> [!IMPORTANT]
> most of the following packages require the conda-forge channel, versions given below are sufficient, but not necessarily required
```bash
conda create --name analog-env python=3.11.4
conda config --add channels conda-forge
conda activate analog-env

conda install cartopy=0.22.0 esmf=8.4.2 hdf5=1.14.1 json5=0.9.14 matplotlib=3.7.2 regionmask=0.10.0 xarray=2023.7.0 xesmf=0.7.1 pip=23.2.1

pip install numpy==1.24.3 tensorflow==2.13.0 tensorflow-macos==2.13.0 silence-tensorflow==1.2.1 keras==2.13.1 ipykernel cftime
```

<h2 align="center">references</h2>

***
[Rader, Jamin K., and Elizabeth A. Barnes. 2023. “Optimizing Seasonal-To-Decadal Analog Forecasts With a Learned Spatially-Weighted Mask.” Geophysical Research Letters 50 (23): e2023GL104983.](https://doi.org/10.1029/2023GL104983)


