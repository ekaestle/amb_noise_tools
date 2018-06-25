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

def multiprocess_fft(inlist):
    tr1=inlist[0]
    tr2=inlist[1]
    window_length=inlist[2]
    overlap = inlist[3]
    try:
        return noise.noisecorr(tr1,tr2,window_length,overlap)
    except:
        return None
        
##############################################
ref_curve = np.loadtxt("Average_phase_velocity_love")
no_cores = 4 # multiprocessing might cause probems on windows machines, because of missing "if __name__ == '__main__':"
joblist = []
for jday in np.arange(365):       
    stream1 = Stream()
    stream2 = Stream()
    
    try:
        stream1 += read("preprocessed_data/SULZ*.%d.*" %jday)
        stream2 += read("preprocessed_data/VDL*.%d.*" %jday)
    except:
        continue
        
    dist,az,baz = gps2dist_azimuth(stream1[0].stats.sac.stla,stream1[0].stats.sac.stlo,
                                  stream2[0].stats.sac.stla,stream2[0].stats.sac.stlo)

    try:
        stream1,stream2 = noise.adapt_timespan(stream1,stream2)
    except:
        continue
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
    
    joblist.append([tr1,tr2,3600.,0.6])
    
print "starting FFT with",no_cores,"cores"
p = Pool(no_cores)
corr_list = p.map(multiprocess_fft, joblist)
p.close()
freq = corr_list[0][0]
corr_spectrum = np.zeros(len(corr_list[0][0]),dtype=complex)
for corrday in corr_list:
    if corrday == None:
        continue
    corr_spectrum += corrday[1]
corr_spectrum/=float(len(corr_list))


#smoothed = noise.velocity_filter(freq,corr_spectrum,dist/1000.,cmin=1.5,cmax=5.5,return_all=False)
smoothed = corr_spectrum 
#%%
from obspy.signal.filter import bandpass
def snr_cc(symmetric_cc,df,distance,cmin,cmax,intervals,plotting=False):
    snr_filt = np.zeros(len(intervals)-1)
    if plotting:
        plt.figure()
        plt.subplot(len(intervals)+1,1,1)
        plt.plot(symmetric_cc)
        ax = plt.gca()
        ax.set_yticklabels([])
    for i in range(len(intervals)-1):
        lim1 = intervals[i]
        lim2 = intervals[i+1]
        signal = bandpass(symmetric_cc,lim1,lim2,df)
        idx1 = int(dist/cmax*df)
        idx2 = int(dist/cmin*df)
        snr_filt[i] = np.mean(np.abs(signal[idx1:idx2])) / np.mean(np.abs(np.append(signal[0:idx1],signal[idx2:int(len(signal)/2)])))
        if plotting:
            plt.subplot(len(intervals)+1,1,i+2)
            plt.plot(signal)
            plt.plot([idx1,idx1],[np.min(signal),np.max(signal)],'r')
            plt.plot([idx2,idx2],[np.min(signal),np.max(signal)],'r')
            if lim1==0.:
                lim1 = 1./999
            plt.text(idx2,np.max(signal)/2.,"%d - %ds SNR: %.1f" %(1./lim2,1./lim1,snr_filt[i]))
            ax = plt.gca()
            ax.set_yticklabels([])
    if plotting:
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    return snr_filt

intervals = freq[1:int(len(freq)/2)][::3]
gaussian = signal.gaussian(len(freq),5)
smoothed_freq = []
smoothed_spec = []
phase_spec = []
greens = np.fft.irfft(np.real(corr_spectrum))
#snr_filt = snr_cc(greens,1,dist/1000.,1.0,6.0,intervals,plotting=False)
for i in np.arange(8,len(freq)-7):
    if i<100:
        continue
    plt.plot(bandpass(greens,freq[i-15],freq[i+15],1.0,corners=4,zerophase=True))
    break
    #filt = np.roll(gaussian,i-900)
    #filt_spec = np.real(corr_spectrum)*filt
    #phase_spec.append(np.angle(filt_spec))
plt.figure()
plt.plot(freq,np.array(phase_spec))
plt.figure()
phase_spec = np.angle(corr_spectrum)
amp_spec = np.abs(corr_spectrum)
plt.figure()
plt.plot(freq,np.cos(phase_spec)*amp_spec)
plt.plot(freq,np.real(corr_spectrum))
plt.plot(freq,np.angle(np.real(corr_spectrum)))
plt.plot(freq,corr_spectrum)
#%%                                   
crossings,phase_vel = noise.get_smooth_pv(freq,smoothed,dist/1000.,ref_curve,\
                         freqmin=0.004,freqmax=0.25, min_vel=1.5, max_vel=5.5,\
                        filt_width=3,filt_height=0.2,x_overlap=0.75,y_overlap=0.75,pick_threshold=1.7,\
                       horizontal_polarization=False, smooth_spectrum=False,plotting=True)

plt.figure(figsize=(16,10))
plt.subplot(2,2,1)
plt.plot(freq,np.real(corr_spectrum))
plt.plot(freq,np.real(smoothed))
plt.title("Cross-correlation spectrum Love")
plt.xlabel("Frequency")
plt.subplot(2,2,2)
plt.plot(crossings[:,0],crossings[:,1],'o',ms=10)
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
