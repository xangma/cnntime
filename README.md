# cnntime

Keras CNN to take in L-PICOLA sims and classify them (only binary so far).

### File list
+ cnntime.py - main program
+ progstart.sh - starts program on sciama (must be on login3)
+ pythonstart.sh	- starts python (with tensorflow ability) on sciama (must be on login3)
+ qsubRUN_dgp_500.sh	Generate 500 dgp sims
+ qsubRUN_lcdm_500.sh	Generate 500 lcdm sims
+ voxeltime_dpg.py - histogram dpg sims
+ voxeltime_lcdm.py - histogram lcdm sims

### cnntime.py code layout
Settings at the top of the file. remake_data will cause the program to read in the histogrammed sims, preprocess them, and save them to a large hdf5 ready for on the fly data generation. By preprocess, I mean gain the 24 unique cube rotations for each sim to increase the training and test set.

The code also ensures the data has been randomised, with a set random seed in the code.
The CNN settings are pretty much as in the Ravanbakhsh 2016 paper.

### References
This work mimics a lot of the settings in this paper.
http://proceedings.mlr.press/v48/ravanbakhshb16.pdf
