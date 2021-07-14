AMBIENT NOISE PROCESSING TOOLS
February 2021
This a set of Python functions that can be used to calculate ambient noise cross correlations and to extract phase velocities. Initial Fortran scripts developed by Kees Weemstra (Delft) and modified and extended by Emanuel Kaestle (UPMC Paris, FU Berlin).
Please contact Emanuel Kaestle if you have any questions.

This work is documented by three articles. At least the more recent one should be cited by anyone who uses this code:

Kaestle, E., Molinari, I., Boschi, L., Kissling, E., 2021. Azimuthal anisotropy from Eikonal Tomography: example from ambient-noise measurements in the AlpArray network. Submitted to GJI.

Kaestle, E., R. Soomro, C. Weemstra, L. Boschi, and T. Meier, 2016. Two-receiver measurements of phase velocity: cross-validation of ambient-noise and earthquake-based observations. Geophys. J. Int., 207, 1493--1512, doi:10.1093/gji/ggw341. 

Boschi L., C. Weemstra, J. Verbeke, G. Ekstrom, A. Zunino and D. Giardini, 2013. On measuring surface-wave phase velocity from station-station cross-correlation of ambient signal. Geophys. J. Int., 192, 346--358, DOI: 10.1093/gji/ggs023. 

In order to run these scripts you need a recent (as of February 2018) version of Python and the following packages

numpy
scipy
matplotlib
obspy

I recommend to use the Anaconda python package that can be installed on all types of machines.
You can find a pre-packed installer, including obspy, on the Obspy webpage. With Anaconda, missing
packages can be installed by typing in a console window
conda install package_name

The following functions can be found in noise.py

adapt_timespan : compares traces in two streams and cuts them to one or several overlapping time ranges
noisecorr : correlates two traces, optional whitening or 1bit normalization
velocity_filter : filters cross correlation to show only signals that arrive within a certain
range of velocities.
get_zero_crossings : extracts zero crossings from a cross correlation trace. Can also be used
to get zero crossings from horizontal component CCs (RR,TT). Does not support cross-correlations
of non-identical components (RT,ZR,...)
extract_phase_velocity : Extracts the phase velocity by picking zero crossings.
get_smooth_pv : Extracts the phase velocity by smoothing the zero crossings prior to picking.

You can find a short description and the necessary input parameters for each of these functions in the noise.py script itself. An additional explanation for the get_smooth_pv function is provided in this folder.

The folder also includes two example scripts which can be used to check whether everything works as expected (with 4 days of data, not much but enough to get an impression).
In example2 the FFT is calculated parallel on several cores to make it faster, however this parallel computing might cause some problems on Windows machines the way it is currently written.
Either use a python interpreter such as Spyder or run it from your console by typing:

python example1.py
for ZZ cross-correlation and phase-velocity extraction.
or

python example2.py
for TT cross-correlation and phase-velocity extraction.

Other scripts:
create_ccs.py is a convencience script to create and maintain a sqlite database of daily records (in mseed or SAC format). All records in the database are cross-correlated and the results are saved. Adapt parameters as needed in the header of the script

process_spectra.py is a script to automatically process all cross-correlation spectra that result from create_ccs.py and extract the phase-velocity curves. Also some experimental signal-to-noise ratio filter can be activated to discard measurements that are below a certain threshold. Adapt parameters as needed in the header of the script. By default, it will pick the dispersion curve from the positive lag time correlation and the negative lag time correlation separately and discard the result if the difference is too large.

manual_picking.py is a script that can be used to pick the dispersion curves manually and creates some informative plots.

create_synthetic_noise_seismograms.py can be used to create some simple synthetic data for testing

All scripts can be run with the example data given in the folder. I would recommend you start with the example1.py and example2.py scripts. Then, you can try to run create_ccs.py which automatically creates a stationlist file, a database of all pre-processed input data files and does the cross correlation. With the manual_picking.py script, you can pick dispersion curves from the just created cross correlations. Alternatively, you can run process_spectra.py which does the work automatically.




