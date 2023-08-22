# return the best N analogs, along with any relevant information (which model, which member, etc.)

# unclear whether the members used in the mask training should be discarded as potential analogs for the projections.
# leave it open to them at first, then can check a few ways: 'tag' them and see if they are selected an unusual amount of the time,
# or simply see how it performs with test cases with and without them.

# so, we have an analog library (reservoir?) with all available datasets (models and observations) -- possibly without the data used
# to train the network for the mask (see above) -- and from this we can pull an SOI (model or observation) and then the relevant
# analogs (relevant meaning model members that have enough years in front of them to make the projection and aren't the SOI).
# Also note, we can apply this timing restriction to the SOI as well for testing (analog projection versus truth)

# this probably should be a class (subclass of _data_processing?) that reads in the pickle files for the processed data,
# reads in additional datasets (observations or models that were not included in the mask portion)
# and can process these new potential inputs/targets to match what should then be multiplied by the mask to determine
# the best analogs (which may require regridding and other steps to match the mask)

# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------