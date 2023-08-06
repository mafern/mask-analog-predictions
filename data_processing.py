# call the _data_processing functions in a modular way and return the final processed data

# this should take in experiment_settings and get all the data

# for analog versus training/validation/testing split either specify data sets to use for each, or list of values to be used across all given models
# for example, the ideal situation is to use models for the analog reservoir, and observations for the training/validation/testing

# input and output processing steps may differ, these should be two different calls, so it probably makes sense to load the data once, then pass that 
# to the processing steps as copies

# print line between operations print(-------------------------------------------------------)