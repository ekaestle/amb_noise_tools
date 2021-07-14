# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 14:12:58 2016

@author: emanuel

This example uses the classical picking method, picking directly the
zero crossings. You can also try replacing noise.extract_phase_velocity
with noise.get_smooth_pv. In this case, adapt the input parameters
according to the documentation of the functions in noise.py.
"""

import numpy as np
import noise
from obspy import read
from obspy.geodetics.base import gps2dist_azimuth
import matplotlib.pyplot as plt

ref_curve = np.loadtxt("Average_phase_velocity_rayleigh")

tr1 = read("preprocessed_data/SULZ.LHZ.CH.2013.219.processed.SAC")[0]
tr2 = read("preprocessed_data/VDL.LHZ.CH.2013.219.processed.SAC")[0]
# bad example with only one day of correlation

dist,az,baz = gps2dist_azimuth(tr1.stats.sac.stla,tr1.stats.sac.stlo,
                              tr2.stats.sac.stla,tr2.stats.sac.stlo)

freq,xcorr,n_corr_wins = noise.noisecorr(tr1,tr2,window_length=3600.,overlap=0.5)
 
smoothed = noise.velocity_filter(freq,xcorr,dist/1000.,cmin=1.5,cmax=5.0,return_all=False)
                                    
crossings,phase_vel = noise.extract_phase_velocity(freq,smoothed,dist/1000.,ref_curve,\
                         freqmin=0.004,freqmax=0.25, min_vel=1.5, max_vel=5.0,min_amp=0.0,\
                         horizontal_polarization=False, smooth_spectrum=False,plotting=True)


plt.figure(figsize=(16,10))
plt.subplot(2,2,1)
plt.plot(freq,np.real(xcorr),label='original')
plt.plot(freq,np.real(smoothed),'r',lw=2,label='smoothed')
plt.title("Cross-correlation spectrum")
plt.xlabel("Frequency")
plt.legend(numpoints=1)
plt.subplot(2,2,2)
plt.plot(crossings[:,0],crossings[:,1],'o',ms=8)
plt.plot(ref_curve[:,0],ref_curve[:,1],label='reference curve')
plt.plot(phase_vel[:,0],phase_vel[:,1],'o',ms=4,label='picks')
plt.xlabel("Frequency")
plt.ylabel("Velocity [km/s]")
plt.title("Zero crossings")
plt.legend()
plt.subplot(2,2,3)
plt.plot(1./phase_vel[:,0],phase_vel[:,1])
plt.xlabel("Period")
plt.ylabel("Velocity [km/s]")
plt.title("Phase velocity curve")
plt.show()
