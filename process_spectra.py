# -*- coding: utf-8 -*-
"""
Created in July 2015

@author: emanuel

This script will process all previously created cross correlation files in 'spectra_path'.
It is assumed that the naming scheme for the CC files is according to the create_cc.py script.
Otherwise please adapt command below, so that station names are recognized correctly from
the file name and can be found in the station list file.
If you encounter any difficulties, do not hesitate to contact me: Emanuel Kaestle @ FU Berlin
"""

import glob,os,sys
import numpy as np
import noise
import datetime
import matplotlib.pyplot as plt
from obspy.geodetics import gps2dist_azimuth
from obspy.signal.invsim import cosine_taper
from obspy.signal.filter import bandpass

""" you can run this program also by calling
python process_spectra.py stat1 stat2
it will then only process stat1 and stat2 and create several plots """

""" USER PARAMETERS"""
max_period = 999. # don't use data above that period
min_period = 1./0.3 # don't use data below that period
min_wavelength_criterion = 1 # don't pick data at less than one wavelength inter-station distance
min_vel=1.5 # minimum velocity for the desired wave type (Rayleigh/Love)
max_vel=5.0 # maximum velocity ...
#min_amplitude = 0.05 # minimum allowed amplitude of the cc spectrum with respect to its maximum (should be tested, otherwise set to 0)
pick_thresh = 1.8
horizontal_polarization=False # If True, TT or RR cross-correlations are calculated (additional J2 term). If False, only ZZ
spectra_path = "./CROSS_SPECTRA/" # path where spectra files are stored (filenames are expected to have both station names, as the ones created by create_ccs.py. Otherwise adapt stat1 = fname.split("/")[-1].split("_")[0] below.)
phase_vel_path='./PHASE_VEL_CURVES/' # path where to save the cross_correlation spectra
ref_curve = np.loadtxt("Average_phase_velocity_rayleigh") # reference curve for picking
statlist_file = "./statlist.txt" # location of station list file (3 columns, statname, lat, lon)
velocity_filter = True # do a velocity filter between min and max velocity, recommended
plotresult = False
# SECTION ABOUNT SNR FILTERS. This is a filter applied only at the very end, discarding parts of the phase-velocity curves where the SNR is lower than a certain threshold
# this has not been tested a lot yet. I recommend to keep the filters on False for most applications.
# snr filters are only active in combination with the velocity filter
# snr filter in the time domain:
snr_filter_time_domain = False
snr_filter_threshold_td = 2
#snr filter in the freq domain
snr_filter_frequency_domain = False
snr_filter_threshold_fd = 1 # see Sadeghisorkhani 2017 (computers and geosciences), appendix; only important when snr_filter_frequency_domain=True
# used for both freq and time domain snr filters. Checks the SNR in certain frequency intervals.
# define intervals where to check the snr ratio, e.g.: 1./np.array([200,100,50,35,20,10,6,3])
intervals = np.append(0,np.logspace(np.log10(0.01),np.log10(0.3),10))
"""END OF USER DEFINED PARAMETERS """

""" ADDITIONAL PARAMETERS AND NOTES
- It may be interesting to modify also other parameters that influence the result
of the picking procedure. For example the filter width and height for the 
get_smooth_pv function.
- If the velocity filter is switched off, smoothing will be applied to the
spectrum before picking. If smoothing should be switched off entirely, please
modify values in the picking funtion below.
- You can try also the classical zero-crossing picking procedure. In order to 
do so, please replace get_smooth_pv with extract_phase_velocity. In this case,
remove also the internal try/except loop and remove input values filt_width, 
filt_height, x_overlap, y_overlap, pick_threshold. See documentation of noise.py.
- the file name recognition pattern may have to be changed if your cross
correlation files have a different naming scheme. Please check if you encounter
any errors.
- feel free to contact me if you have questions under emanuel.kaestle@upmc.fr or
under my mail address at the Freie Universitaet Berlin
"""

if (velocity_filter==False and snr_filter_time_domain) or (velocity_filter==False and snr_filter_time_domain):
    print "snr filter only works with velocity filter activated!"
    sys.exit()

def running_mean(x, N):
    if N%2 == 0:
        N+=1
    x = np.insert(x,0,np.ones(np.int(N/2))*x[0])
    x = np.append(x,np.ones(np.int(N/2))*x[-1])
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / N 
    
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

        
if np.mean(ref_curve[:,1])>1000:
    ref_curve[:,1]/=1000.
stat_dict = {}
statlist = np.loadtxt(statlist_file,dtype='str')
for line in statlist:
    stat_dict[line[0]] = (float(line[1]),float(line[2]))
    
statpair = False
try:
    station1=sys.argv[1]
    station2=sys.argv[2]
    statpair = True
    plotresult=True
except:
    pass

#""" for testing """
#station1 = 'HKWD'
#station2 = 'ARSA'
#statpair = True
#plotresul=True
#""" """

#crossings_path='./CROSSINGS_RADIAL/'
#if not os.path.exists(crossings_path):
#    os.makedirs(crossings_path)
if not os.path.exists(phase_vel_path):
    os.makedirs(phase_vel_path)

spectralist = glob.glob(os.path.join(spectra_path,"*")) 
phase_vel_curves = glob.glob(os.path.join(phase_vel_path,"*"))
print len(phase_vel_curves),"already calculated phase-velocity curves have been found."

if not statpair:
    for fpath in phase_vel_curves:
        fname = fpath.split("/")[-1].replace("_curve","")
        fname_remove = os.path.join(spectra_path,fname)
        spectralist.remove(fname_remove)
    
time_start=datetime.datetime.now()
print "Processing",len(spectralist),"cross-correlation spectra."
m=1
fails=0
for i,fname in enumerate(spectralist):
    if statpair:
        if not (station1 in fname and station2 in fname):
            continue
        else:
            print fname
    if plotresult:
        print fname
    if float(i)/len(spectralist) > m*0.01:
        time_passed = (datetime.datetime.now()-time_start).total_seconds()
        time_left = (100.-m)/m * time_passed
        print m*1,"percent processed, estimated time left:",np.round(time_left/60.),"minutes"
        print fails,"/",i,"cross-correlations could not be processed."
        m+=1
        
    spectrum = np.loadtxt(fname)
    stat1 = fname.split("/")[-1].split("_")[0]
    stat2 = fname.split("/")[-1].split("_")[2]
    #dist = float(fname.split("/")[-1].split("_")[4])/1000.
    try:
        dist,az,baz = gps2dist_azimuth(stat_dict[stat1][0],stat_dict[stat1][1],stat_dict[stat2][0],stat_dict[stat2][1])
    except:
        print "station information not found in statlist! aborting..."
        continue
    dist/=1000.
    dt = 1./spectrum[-1,0]/2.

    """ applying velocity filter and SNR filter"""
    if velocity_filter:
        idx_tmin = int((dist/max_vel)/dt*0.95) # 5percent extra for taper
        idx_tmax = int((dist/min_vel)/dt*1.05) # 5% extra for taper
        vel_filt_window = cosine_taper(idx_tmax-idx_tmin,p=0.1)
        win_samples=int((len(spectrum)-1)*2)
        vel_filt = np.zeros(win_samples)
        vel_filt[idx_tmin:idx_tmax] = vel_filt_window
        vel_filt[-idx_tmax:-idx_tmin] = vel_filt_window
        TCORR = np.fft.irfft(spectrum[:,1])#+1j*spectrum[:,2])
        if snr_filter_time_domain:
            snr_filt_td = snr_cc(TCORR,1./dt,dist,min_vel,max_vel,intervals,plotting=statpair) > snr_filter_threshold_td
        overall_snr = np.mean(np.mean(np.abs(TCORR[idx_tmin:idx_tmax]))/np.mean(np.abs(np.append(TCORR[0:idx_tmin],TCORR[idx_tmax:int(len(TCORR)/2)]))))
        TCORR *= vel_filt
        CORR = np.fft.rfft(TCORR)
        if snr_filter_frequency_domain:
            a_thresh = np.sqrt((1+snr_filter_threshold_fd**2)/((len(spectrum)*2-2)/(0.5*dist*(1./min_vel-1./max_vel))+snr_filter_threshold_fd**2))
            Psignal = np.abs(spectrum[:,1]+1j*spectrum[:,2])**2
            Psignal = running_mean(Psignal,21)
            Pnoise = np.abs(CORR)**2
            Pnoise = running_mean(Pnoise,21)
            Pnoise[Pnoise>Psignal] = Psignal[Pnoise>Psignal]
            a = np.sqrt(Pnoise/Psignal)
            snr_filt_fd=[]
            for i in range(len(intervals)-1):
                lim1 = intervals[i]
                lim2 = intervals[i+1]
                if lim1==0.:
                    lim1=1./999.
                if np.sum(a[(lim1<=spectrum[:,0])*(spectrum[:,0]<lim2)]<a_thresh)>1:
                    snr_filt_fd.append(False)
                else:
                    snr_filt_fd.append(True)      
            if statpair:
                plt.figure()
                plt.plot(spectrum[:,0],Psignal/np.max(Psignal),label='PSD real cc-spectrum with velocity filter')
                plt.plot(spectrum[:,0],Pnoise/np.max(Pnoise),label='PSD real cc-spectrum')
                plt.plot(spectrum[:,0],a,label='PSDnoise/PSDsignal')
                #plt.plot(spectrum[:,0],running_mean(a,31))
                plt.plot(spectrum[:,0],np.ones(len(spectrum))*a_thresh,'--',label='SNR=%.1f threshold' %snr_filter_threshold_fd)
                for i in range(len(intervals)-1):
                    lim1 = intervals[i]
                    lim2 = intervals[i+1]
                    plt.plot([lim1,lim1],[0,1],'k',lw=0.2)
                    plt.plot([lim2,lim2],[0,1],'k',lw=0.2)
                    if snr_filt_fd[i]:
                        plt.text(lim1,0.05*i,'accepted')
                    else:
                        plt.text(lim1,0.05*i,'rejected')
                plt.legend()                
    
    snr_filt = np.zeros(len(intervals)-1)
    if snr_filter_frequency_domain and snr_filter_time_domain:
        for i in range(len(snr_filt_fd)):
            snr_filt[i] = snr_filt_fd[i] * snr_filt_td[i]
    elif snr_filter_frequency_domain:
        snr_filt = snr_filt_fd
    elif snr_filter_time_domain:
        snr_filt = snr_filt_td
 
    """ getting zero crossings and extracting phase velocity """   
    min_freq = 1./(dist/3.5)
    if min_freq < 1./max_period:
        min_freq = 1./max_period
    if velocity_filter:
        ccspec = CORR
    else:
        ccspec = spectrum[:,1]+1j*spectrum[:,2]

    try:
        try:
            crossings,phase_vel = noise.get_smooth_pv(spectrum[:,0],ccspec,dist,ref_curve,
                         freqmin=min_freq,freqmax=1./min_period, min_vel=min_vel, max_vel=max_vel,
                         filt_width=3,filt_height=0.2,x_overlap=0.75,y_overlap=0.75,pick_threshold=pick_thresh,
                       horizontal_polarization=horizontal_polarization, smooth_spectrum=not(velocity_filter),plotting=plotresult)
        except:
            crossings,phase_vel = noise.get_smooth_pv(spectrum[:,0],ccspec,dist,ref_curve,
                         freqmin=min_freq,freqmax=1./min_period, min_vel=min_vel, max_vel=max_vel,
                         filt_width=5,filt_height=0.2,x_overlap=0.85,y_overlap=0.85,pick_threshold=pick_thresh*1.2,
                       horizontal_polarization=horizontal_polarization, smooth_spectrum=not(velocity_filter),plotting=plotresult)            

        if snr_filter_frequency_domain or snr_filter_time_domain:
            freq_filtered = np.array([])
            phase_vel_filtered = np.array([])
            for i in range(len(intervals)-1):
                if snr_filt[i]:
                    lim1 = intervals[i]
                    lim2 = intervals[i+1]
                    phase_vel_filtered = np.append(phase_vel_filtered,phase_vel[:,1][(lim1<=phase_vel[:,0])*(phase_vel[:,0]<lim2)])
                    freq_filtered = np.append(freq_filtered,phase_vel[:,0][(lim1<=phase_vel[:,0])*(phase_vel[:,0]<lim2)])
            if statpair:
                plt.figure()
                plt.plot(1./phase_vel[:,0],phase_vel[:,1],'o',ms=6,label='without snr filter')
                plt.plot(1./freq_filtered,phase_vel_filtered,'o',ms=4,label='with snr filter')
                plt.legend()
            phase_vel = np.column_stack((freq_filtered,phase_vel_filtered))
    except:
        fails+=1
        print "    not successful for station pair",stat1,stat2
        continue
    if statpair or plotresult:
        plt.figure(figsize=(16,8))
        plt.suptitle(stat1+"_"+stat2+"   dist: %dkm   SNR: %.1f" %(dist,overall_snr))
        plt.subplot(221)
        plt.plot((np.arange(len(spectrum)*2-2)-len(spectrum)+1)*dt,np.fft.fftshift(np.fft.irfft(spectrum[:,1]+1j*spectrum[:,2])),label='raw cc')
        if velocity_filter:
            plt.plot((np.arange(len(spectrum)*2-2)-len(spectrum)+1)*dt,np.fft.fftshift(TCORR),'--',label='cc filtered and symmetric')
        plt.legend(numpoints=1)
        plt.xlim(dist/min_vel*-2.,dist/min_vel*2)
        plt.title("CC")
        plt.xlabel('time lag [s]')
        plt.subplot(222)
        plt.plot(spectrum[:,0],spectrum[:,1],label='cc raw')
        if velocity_filter:
            plt.plot(spectrum[:,0],np.real(CORR),'--',label='cc filtered')
        plt.legend(numpoints=1)
        plt.title("real CC spectrum")
        plt.xlabel('frequency [Hz]')
        plt.subplot(223)
        plt.plot(1./phase_vel[:,0],phase_vel[:,1],'ro',ms=4,label='picked phase vels')
        plt.plot(1./crossings[:,0],crossings[:,1],'o',ms=2,label='crossings')
        plt.plot(1./ref_curve[:,0],ref_curve[:,1],label='reference curve')
        plt.legend(numpoints=1)
        plt.xlim(min_period,1./min_freq)
        plt.xlabel('period [s]')
        plt.ylabel('velocity [km/s]')
        plt.subplot(224)
        plt.plot(1./phase_vel[:,0],phase_vel[:,1])
        plt.xlabel('period [s]')
        plt.ylabel('velocity [km/s]')
        if snr_filter_frequency_domain or snr_filter_time_domain:
            y0=np.min(phase_vel[:,1])-0.2
            for i in range(len(intervals)-1):
                lim1 = intervals[i]
                lim2 = intervals[i+1]
                if lim1==0:
                    lim1 = 1./300.
                if snr_filt[i]:
                    plt.plot([1./lim2,1./lim1],[y0,y0],color='green',lw=3)
                else:
                    plt.plot([1./lim2,1./lim1],[y0,y0],color='red',lw=3)
            plt.plot([],[],lw=3,color='red',label='snr filter: discarded')
            plt.plot([],[],lw=3,color='green',label='snr filter: accepted')
            plt.legend(numpoints=1)
        plt.xlim(3,100)
        plt.show()
    np.savetxt(phase_vel_path+fname.split("/")[-1]+"_curve",phase_vel,fmt="%.5f %.3f")
