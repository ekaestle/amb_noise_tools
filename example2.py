# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 14:35:18 2016

@author: emanuel

This script uses the alternative picking method, where the picks are smoothed
with a sliding window before picking.
You can also try replacing noise.get_smooth_pv with
noise.extract_phase_velocity. In this case, adapt the input parameters
according to the documentation of the functions in noise.py.
"""

import numpy as np
from obspy import read,Stream
from obspy.geodetics.base import gps2dist_azimuth
from multiprocessing import Pool
import matplotlib.pyplot as plt
import noise


##############################################
ref_curve = np.loadtxt("Average_phase_velocity_love")
no_cores = 4


def multiprocess_fft(inlist):
    tr1=inlist[0]
    tr2=inlist[1]
    window_length=inlist[2]
    overlap = inlist[3]
    try:
        return noise.noisecorr(tr1,tr2,window_length,overlap)
    except:
        return None


if __name__ == '__main__':
    
    joblist = []
    for jday in np.arange(365):
        
        stream1 = Stream()
        stream2 = Stream()
        
        try:
            stream1 += read("preprocessed_data/SULZ*.%d.*" %jday)
            stream2 += read("preprocessed_data/VDL*.%d.*" %jday)
        except:
            continue

            
        dist,az,baz = gps2dist_azimuth(stream1[0].stats.sac.stla,
                                       stream1[0].stats.sac.stlo,
                                       stream2[0].stats.sac.stla,
                                       stream2[0].stats.sac.stlo)
        
        # cut the horizontal components to the same time range
        # this is necessary to rotate the components (NE->RT)
        # and to make sure that only overlapping time ranges are correlated
        st1 = (stream1.select(component='N') + stream1.select(component='E'))
        st2 = (stream2.select(component='N') + stream2.select(component='E'))
        st1,st2 = noise.adapt_timespan(st1,st2)
        
        # we add the Z component again (not really necessary)
        stream1 = stream1.select(component='Z') + st1 
        stream2 = stream2.select(component='Z') + st2
    
        # az = azimuth from station1 -> station2
        # baz = azimuth from station2 -> station1
        # for stream2 the back azimuth points in direction of station1
        # for stream1 the azimuth points in direction of station2
        # BUT 180. degree shift is needed so that the radial components point in the same direction!
        # otherwise they point towards each other => transverse comp would be also opposed
        stream1.rotate('NE->RT',back_azimuth=(az+180.)%360.)
        stream2.rotate('NE->RT',back_azimuth=baz)
        
        tr1 = stream1.select(component='T')[0]
        tr2 = stream2.select(component='T')[0]
        
        print("next correlation job (T-components, Love waves):")
        print(tr1)
        print(tr2)
        print("\n")

        # joblist contains the arguments for the cross correlation, i.e.
        # trace1, trace2, correlation window (1hr=3600s) and the window overlap (0.5=50%)
        joblist.append([tr1,tr2,3600.,0.5])
        
    print("starting FFT with",no_cores,"cores")
    p = Pool(no_cores)
    corr_list = p.map(multiprocess_fft, joblist)
    p.close()
    corr_spectrum = []
    no_spectra = 0
    for corrday in corr_list:
        if corrday == None:
            continue
        if len(corr_spectrum) == 0:
            freq = corrday[0]
            corr_spectrum = corrday[1]
        else:
            corr_spectrum += corrday[1]
        no_spectra += 1
    corr_spectrum/=no_spectra
    
    
    smoothed = noise.velocity_filter(freq,corr_spectrum,dist/1000.,cmin=1.5,
                                     cmax=5.5,return_all=False)
                                        
    crossings,phase_vel = noise.get_smooth_pv(
        freq,smoothed,dist/1000.,ref_curve,freqmin=0.004,freqmax=0.25, 
        min_vel=1.5, max_vel=5.5,filt_width=3,filt_height=1.0,
        pick_threshold=2.0,horizontal_polarization=True,
        smooth_spectrum=False,plotting=True)
    
    plt.figure(figsize=(16,10))
    plt.subplot(2,2,1)
    plt.plot(freq,np.real(corr_spectrum))
    plt.plot(freq,np.real(smoothed))
    plt.title("Cross-correlation spectrum Love")
    plt.xlabel("Frequency")
    plt.subplot(2,2,2)
    for branchidx in np.unique(crossings[:,2]):
        plt.plot(crossings[crossings[:,2]==branchidx,0],
                 crossings[crossings[:,2]==branchidx,1],'o',
                 markeredgecolor='black',ms=6,linewidth=0.1)
    plt.plot(ref_curve[:,0],ref_curve[:,1],label='reference curve')
    plt.plot(phase_vel[:,0],phase_vel[:,1],'o',ms=5,label='picks')
    plt.xlabel("Frequency")
    plt.ylabel("Velocity [km/s]")
    plt.title("Zero crossings")
    plt.legend()
    plt.subplot(2,2,3)
    plt.plot(1./phase_vel[:,0],phase_vel[:,1])
    plt.xlabel("Period")
    plt.ylabel("Velocity [km/s]")
    plt.title("Phase velocity curve Love")
    plt.show()
