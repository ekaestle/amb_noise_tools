# -*- coding: utf-8 -*-
"""
Last update: 14.05.2018
Adapted for Python3 now

@author: emanuel

Set of functions for the Ambient Noise correlation. Procedure originally 
developed by Kees Weemstra.
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.special import jn_zeros,jv
from scipy.stats import linregress
from scipy.signal import detrend
#from scipy.signal import argrelextrema
#from scipy.interpolate import griddata
#from scipy.ndimage.filters import gaussian_filter
from obspy.signal.invsim import cosine_taper
#from obspy.signal.filter import lowpass
from scipy.signal import find_peaks
from obspy.core import UTCDateTime, Stream
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.interpolate import LSQUnivariateSpline
from scipy.optimize import curve_fit

########################
def running_mean(x, N):
    if N%2 == 0:
        N+=1
    idx0 = int((N-1)/2)
    runmean = np.zeros(len(x))
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    runmean[idx0:-idx0] = (cumsum[N:] - cumsum[:-N]) / N
    for i in range(idx0):
        runmean[i] = np.mean(x[:2*i+1])
        runmean[-i-1] = np.mean(x[-2*i-1:])
    return runmean

##############################################################################
def adapt_timespan(st1,st2):
    """
    Slices all traces from the two input streams to the same timerange.
    Returns sliced streams.\n
    If there is no common time range an exception is raised.\n  
    
    .. rubric:: Basic Usage
    >>>  st1_sliced, st2_sliced = adapt_timespan(st1,st2)
    
    :rtype: :class: `~obspy.core.stream`
    :returns: stream1 and stream2 sliced    
    """
    # this has to be done twice, because otherwise there sometimes occurs a 1s timeshift
    for adapt in range(2):
        start=UTCDateTime(0)
        end=UTCDateTime(9e9)
        for tr in st1:
            if tr.stats.starttime>start:
                start=tr.stats.starttime
            if tr.stats.endtime<end:
                end=tr.stats.endtime
        for tr in st2:
            if tr.stats.starttime>start:
                start=tr.stats.starttime
            if tr.stats.endtime<end:
                end=tr.stats.endtime
        
        if end<start:
            raise Exception('No common time range')
        
        st1=st1.slice(start,end)
        st2=st2.slice(start,end)
        for tr in st1:
            tr.stats.starttime=start
        for tr in st2:
            tr.stats.starttime=start
            
    return st1,st2


def freq_to_time_domain(spectrum,f):
    """
    Converts the (cross-correlation) spectrum back to the time domain.\n
    Uses the numpy fft.irfft function (assuming real valued time signal).\n
    :type spectrum: :class:`~numpy.ndarray`
    :param spectrum: Some spectrum in the frequency domain (real or complex).
    :type f: :class:`~numpy.ndarray` OR float
    :param f: Frequency axis corresponding to spectrum OR Nyquist frequency.

    """

    if isinstance(f,np.ndarray):
        dt = 1./f[-1]/2.
    else:
        dt = 1./f/2.

    tcorr = np.fft.irfft(spectrum)   
    tcorr = np.fft.fftshift(tcorr)
    # should make no difference whether len(spectrum) is pair or impair
    # tcorr should always be pair (check?)
    timeax = (np.arange(len(tcorr))-len(tcorr)/2.)*dt

    return timeax,tcorr



##############################################################################
def noisecorr(trace1, trace2, window_length=3600., overlap=0.5,\
              onebit=False,whiten=True, waterlevel=1e-10,cos_taper=True,\
              taper_width=0.05):
    """
    Correlates trace1 and trace2 and returns the summed correlation function
    in the frequency domain.\n
    The streams are cut to a common time range and sliced into windows of
    length 'window_length' [s]. The windows are correlated in the frequency 
    domain with an overlap of 'overlap'.\n
    The traces can be tapered and normalized beforehand. The cos taper is
    applied to the beginning and ending 'taper_width' percent of each window
    in the time domain. Spectral whitening divides each spectrum by its
    absolute value with a ``waterlevel``.\n
    1 BIT NORMALIZATION NOT WELL IMPLEMENTED YET\n
    
    :type trace1: :class:`~obspy.core.trace.Trace`
    :param trace1: Trace1
    :type trace2: :class:`~obspy.core.trace.Trace`
    :param trace2: Trace2  
    :type window_length: float
    :param window_length: Window length in seconds.
    :type overlap: float
    :param overlap: Overlap of successive time windows, must be between 0 
        (no overlap) and < 1 (complete overlap).
    :type onebit: bool
    :param onebit: If ``True`` a one bit normalization will be applied.
    :type whiten: bool
    :param whiten: If ``True`` the spectrum will be whitened before correlation. 
    :type waterlevel: float
    :param waterlevel: Waterlevel for the whitening.
    :type cos_taper: bool
    :param cos_taper: If ``True`` the windowed traces will be tapered before
        the FFT.
    :type taper_width: float
    :param taper_width: Decimal percentage of cosine taper (ranging from 0 to
        1). Default is 0.05 (5%) which tapers 2.5% from the beginning and 2.5%
        from the end.
    
    :rtype: :class:`~numpy.ndarray`
    :return: **freq, corr_spectrum** - The frequency axis and the stacked
        cross-correlation spectrum
    """
    st1,st2=adapt_timespan(Stream(trace1),Stream(trace2))
    tr1=st1[0]
    tr2=st2[0]
    if (tr1.stats.endtime - tr1.stats.starttime) < window_length:
        raise Exception('Common time range of traces shorter than window \
                         length.')
    if tr1.stats.sampling_rate==tr2.stats.sampling_rate:
        dt=1./tr1.stats.sampling_rate
    else:
        raise Exception('Both input streams should have the same sampling \
                         rate!')
    win_samples=int(window_length/dt)
    data1 = tr1.data
    data2 = tr2.data
    no_windows=int((len(data1)-win_samples)/((1-overlap)*win_samples))+1
    freq=np.fft.rfftfreq(win_samples,dt)
    
    if cos_taper:
        taper = cosine_taper(win_samples,p=taper_width)     

    # loop for all time windows
    for i in range(no_windows):
        window0=int(i*(1-overlap)*win_samples)
        window1=window0+win_samples
        d1=data1[window0:window1]
        d2=data2[window0:window1]
        d1 = detrend(d1,type='constant')
        d2 = detrend(d2,type='constant')
        
        if onebit:
            # 1 bit normalization - doesn't work very convincingly like that(?)
            d1=np.sign(d1)
            d2=np.sign(d2)
        
        if cos_taper: # tapering in the time domain
            d1*=taper
            d2*=taper
        
        # time -> freqency domain
        D1=np.fft.rfft(d1)
        D2=np.fft.rfft(d2)
        
        if whiten:
            #D1=np.exp(1j * np.angle(D1)) # too slow
            #D2=np.exp(1j * np.angle(D2))
            D1/=np.abs(D1)+waterlevel
            D2/=np.abs(D2)+waterlevel
            
        # actual correlation in the frequency domain
        CORR=np.conj(D1)*D2

        # summing all time windows
        if i==0:
            SUMCORR=CORR
        else:
            SUMCORR+=CORR
    
    #freq=freq[(freq>freqmin) & (freq<=freqmax)]
    SUMCORR /= float(no_windows)
    return freq, SUMCORR


def velocity_filter(freq,corr_spectrum,interstation_distance,cmin=1.0,cmax=5.0,
                    return_all=False):
    """
    Returns the velocity-filtered cross-correlation spectrum (idea from Sadeghisorkhani).
    Filter is applied in the time domain after inverse FFT. Signals slower than cmin and faster than
    cmax are set to zero. The filter window is cosine-tapered. This filter can improve the SNR
    significantly.

    :type freq: :class:`~numpy.ndarray`
    :param freq: Frequency axis corresponding to corr_spectrum.
    :type corr_spectrum: :class:`~numpy.ndarray`
    :param corr_spectrum: Array containing the cross-correlation spectrum. Can be complex
        valued or real.
    :type interstation_distance: float
    :param interstation_distance: Distance between station pair in km.
    :type cmin: float
    :param cmin: Minimum velocity of the signals of interest in km/s.
    :type cmax: float
    :param cmax: Maximum velocity of the signals of interest in km/s.
    :type return_all: bool
    :param return_all: If ``True``, the function returns for arrays: the frequency axis, the filtered
        cc-spectrum, the time-shift axis and the filtered time-domain cross-correlation. Otherwise,
        only the filtered cc-spectrum is returned.
        
    :rtype: :class:`~numpy.ndarray`
    :return: Returns an 1-D array containing the filtered cross-correlation spectrum.
        If return_all=``True``, it also returns the frequency axis, the time-shift axis and the filtered
        time-domain cross-correlation.
    """
    dt = 1./(2.*freq[-1])
    idx_tmin = int((interstation_distance/cmax)/dt*0.95) # 5percent extra for taper
    idx_tmax = int((interstation_distance/cmin)/dt*1.05) # 5% extra for taper
    vel_filt_window = cosine_taper(idx_tmax-idx_tmin,p=0.1)
    tcorr = np.fft.irfft(corr_spectrum)
    vel_filt = np.zeros(len(tcorr))
    vel_filt[idx_tmin:idx_tmax] = vel_filt_window
    vel_filt[-idx_tmax+1:-idx_tmin+1] = vel_filt_window #+1 is just for symmetry reasons
    tcorr *= vel_filt
    corr = np.fft.rfft(tcorr)
    if return_all:
        time = (np.arange(len(tcorr))-len(tcorr)/2.)*dt
        tcorr = np.fft.ifftshift(tcorr)
        return freq,corr,time,tcorr
    else:
        return corr
   
        
##############################################################################
def get_zero_crossings(freq,corr_spectrum, interstation_distance, freqmin=0.0,\
                       freqmax=99.0, min_vel=1.0, max_vel=5.0,\
                       horizontal_polarization=False, smooth_spectrum=False,
                       return_smoothed_spectrum=False):
    """
    Returns the zero crossings from the smoothed complex cross-correlation
    spectrum.
    
    :type freq: :class:`~numpy.ndarray`
    :param freq: Frequency axis corresponding to corr_spectrum.
    :type corr_spectrum: :class:`~numpy.ndarray`
    :param corr_spectrum: Complex valued array containing the cross-correlation
        spectrum.
    :type freqmin, freqmax: float
    :param freqmin, freqmax: Restrict to a range between ``freqmin`` and
        ``freqmax``. Values outside the bounds of ``freq`` are ignored.
    :type interstation_distance: float
    :param interstation distance: Interstation distance in km.
    :type min_vel, max_vel: float
    :param min_vel, max_vel: Min and max surface wave velocities for the
        region, only zero crossings corresponding to velocities within these
        boundaries are considered.
    :type horizontal_polarization: bool
    :param horizontal_polarization: If ``True``, the zero crossings from the spectrum are
        compared to the difference function J0 - J2 (Bessel functions of the
        first kind of order 0 and 2 respectively, see Aki 1957). Appropriate for Love- and the
        radial component of Rayleighwaves. Otherwise only J0 is used.
    :param smooth_spectrum: If ``True``, the spectrum is smoothed by a filter prior
        to zero-crossing extraction. Normally not necessary if velocity filter has been applied.
        The number of knots is determined by 0.5*min_vel
        (knots = zero crossings should not be closer than min_vel/interstation_distance apart)
    :param return_smoothed_spectrum: If ``True``, the smoothed spectrum  and the freqency axis
        are returned as additonal return variables.
        
    :rtype: :class:`~numpy.ndarray`
    :return: Returns an 2-D array containing frequencies and corresponding
        velocities in km/s from the Bessel function zero crossings.
        Returns additionally the smoothed spectrum if smooth_spectrum is ``True``.
    """  
    #Returns two arrays: pos_crossings, neg_crossings, each of shape (x,2).
    #The first column contains the frequencies the second velocities in km/s
    #corresponding to all possible zero crossings.     
       
    # Smoothing by lowpassfiltering. The corner frequency is defined by the 
    # "Period" of a Bessel function, which is 2pi,scaled by w*R/vel
    # with w=2pi*f and the minimum velocity assumed 1km/s this gives
    # 2pi = f*2pi*R/1 -> f = 1/R
    # (as we're in the freq domain, f is the "Period", and 1/f gives the corner freqency)
    delta = interstation_distance    
    if smooth_spectrum:
    #smoothed = lowpass(np.real(corr_spectrum),delta,df=1./(freq[1]-freq[0]),zerophase=True)
        # lowpass has slightly worse boundary effects compared to LSQUnivariateSpline
    # smooth spline function fitting the data - same results as lowpassing
    # t = knots, should be half the sampling rate
        usp = LSQUnivariateSpline(freq,np.real(corr_spectrum),t=np.arange(freq[1],freq[-1],0.5*min_vel/delta))    
        ccspec = usp(freq)
    else:
        ccspec = np.real(corr_spectrum)
    # Limit to the desired frequency range
    w = freq[(freq>=freqmin) & (freq<=freqmax)]
    #minf=w[0]
    maxf=w[-1]
    ccspec = ccspec[(freq>=freqmin) & (freq<=freqmax)]
    
    # get the positive (from negative to positive) and negative zero crossings
    # from the smoothed spectrum
    # Splitting into positive and negative crossings cancels out a lot of 
    # unusable crossings as it would give a bad fit if we attribute a positive
    # Bessel function zero crossing to a negative crossing in the spectrum
    cross_idx=np.where((ccspec[:-1]*ccspec[1:] < 0))[0]
    crossings = -ccspec[cross_idx]/(ccspec[cross_idx+1]-ccspec[cross_idx])*(w[cross_idx+1]-w[cross_idx]) + w[cross_idx]

    values = ccspec[cross_idx]
    pos_crossings = crossings[values<0]
    neg_crossings = crossings[values>0]

#    plt.figure()
#    plt.plot(freq,corr_spectrum)
#    plt.plot(w,ccspec)
#    plt.plot(w,jv(0,w*2*np.pi*delta/3.8)*2000.)
#    plt.plot(pos_crossings,np.zeros(len(pos_crossings)),'.')
#    plt.plot(neg_crossings,np.zeros(len(neg_crossings)),'.')
    
    
    # maximum number of zero crossings for a Bessel function from 0 to maxf,
    # distance delta and minimum velocity min_vel
    no_bessel_zeros=int(maxf*2*np.pi*delta/min_vel/np.pi)
    if horizontal_polarization: # when treating Love and Rayleigh (radial) waves (see aki 1957)
        # calculate zeros for the difference J0(x)-J2(x)
        # x = 2*pi*delta*freqency / velocity
        j02_axis = np.linspace(0,2*np.pi*maxf*delta/min_vel,no_bessel_zeros*5)
        J02=jv(0,j02_axis)-jv(2,j02_axis)
        cross_idx = np.where((J02[:-1]*J02[1:]) < 0)[0]
        bessel_zeros = -J02[cross_idx]/(J02[cross_idx+1]-J02[cross_idx])*(j02_axis[cross_idx+1]-j02_axis[cross_idx]) + j02_axis[cross_idx]
    else:
        bessel_zeros=jn_zeros(0,no_bessel_zeros)
        
    pos_bessel_zeros=bessel_zeros[1::2]
    neg_bessel_zeros=bessel_zeros[0::2]
    
    # All found zero crossings in the smoothed spectrum are compared to all
    # possible zero crossings in the Bessel function. Resulting velocities
    # that are within the min_vel - max_vel range are stored and returned.    
    pos_crossings_vel=np.array(()) #from negative to positive
    pos_crossings_freqs=np.array(())
    neg_crossings_vel=np.array(())
    neg_crossings_freqs=np.array(())    
    for j,pcross in enumerate(pos_crossings):
        velocities=pcross*2*np.pi*delta/pos_bessel_zeros
        velocities=velocities[(velocities>min_vel)*(velocities<max_vel)]        
        pos_crossings_freqs=np.append(pos_crossings_freqs,np.ones(len(velocities))*pcross)
        pos_crossings_vel=np.append(pos_crossings_vel,velocities[::-1])        
    for ncross in neg_crossings:
        velocities=ncross*2*np.pi*delta/neg_bessel_zeros
        velocities=velocities[(velocities>min_vel)*(velocities<max_vel)]
        neg_crossings_freqs=np.append(neg_crossings_freqs,np.ones(len(velocities))*ncross)
        neg_crossings_vel=np.append(neg_crossings_vel,velocities[::-1])
    
    all_crossings_vel = np.hstack((pos_crossings_vel,neg_crossings_vel))
    all_crossings_freqs = np.hstack((pos_crossings_freqs,neg_crossings_freqs))

    if return_smoothed_spectrum:
        return np.column_stack((w,ccspec)),np.column_stack((all_crossings_freqs,all_crossings_vel))
    else:
        return np.column_stack((all_crossings_freqs,all_crossings_vel))
    #return np.column_stack((pos_crossings_freqs,pos_crossings_vel)), \
    #       np.column_stack((neg_crossings_freqs,neg_crossings_vel))
##############################################################################
def extract_phase_velocity(frequencies,corr_spectrum,interstation_distance,ref_curve,\
                         freqmin=0.0,freqmax=99.0, min_vel=1.0, max_vel=5.0,min_amp=0.0,\
                       horizontal_polarization=False, smooth_spectrum=False,plotting=False):
    """
    Function for picking the phase velocity curve from the zero crossings. A 
    reference dispersion curve must be given.
    
    :type frequencies: :class:`~numpy.ndarray`
    :param frequencies: 1-D array containing frequency samples of the CC spectrum.
    :type corr_spectrum: :class: `~numpy.ndarray`
    :param corr_spectrum: 1-D or 2-D array containing real or complex CC spectrum.
    :type ref_curve: :class:`~numpy.ndarray`
    :param ref_curve: Reference phase velocity curve, column 1 must contain
        frequencies, column 2 velocities in km/s. The phase velocity curve is
        only picked within the frequency limits of the reference curve.
    :type interstation_distance: float
    :param interstation_distance: Interstation_distance in km.
    :type freqmin, freqmax: float
    :param freqmin, freqmax: Restrict to a range between ``freqmin`` and
        ``freqmax``. Values outside the bounds of ``freq`` are ignored.
    :type min_vel, max_vel: float
    :param min_vel, max_vel: Min and max surface wave velocities for the
        region, only zero crossings corresponding to velocities within these
        boundaries are considered.
    :type min_amp: float
    :param min_amp: minimum amplitude of CC spectrum w.r.t. its max amplitude.
        If the peak amplitude next to a zero crossing is lower than this threshold,
        the zero crossing is ignored. Beware, amplitude is not necessarily the best
        indicator of low SNR.
    :type horizontal_polarization: bool
    :param horizontal_polarization: When ``True``, the zero crossings from the spectrum are
        compared to the difference function J0 - J2 (Bessel functions of the
        first kind of order 0 and 2 respectively, see Aki 1957). Appropriate for Love- and the
        radial component of Rayleighwaves. Otherwise only J0 is used.
    :param smooth_spectrum: When ``True``, the spectrum is smoothed by a filter prior
        to zero-crossing extraction. Normally not necessary if velocity filter has been applied.    
    :type plotting: bool        
    :param plotting: When ``True``, a control plot is created.
    
    :rtype: :class:`~numpy.ndarray`
    :return: Returns two 2-D arrays containing frequencies and corresponding zero crossings
        as well as frequencies and corresponding velocities in km/s.
    """

    crossings = get_zero_crossings(frequencies,corr_spectrum,interstation_distance,
                                   freqmin=freqmin,freqmax=freqmax,min_vel=min_vel,max_vel=max_vel,
                                   horizontal_polarization=horizontal_polarization,smooth_spectrum=smooth_spectrum)
    if plotting:
        print("Extracting phase velocity...")
        plt.figure(figsize=(10,7))
        bound1=[]
        bound2=[]
    delta=interstation_distance
    w_axis=np.unique(crossings[:,0])
    w_axis=w_axis[(np.min(ref_curve[:,0])<w_axis)*(w_axis<np.max(ref_curve[:,0]))]
    
    cross={}
    for w in w_axis:
        cross[w] = crossings[:,1][crossings[:,0] == w]
    
    func=interp1d(ref_curve[:,0],ref_curve[:,1])
    ref_velocities=func(w_axis)
    #%%    
    pick=[]
    old_picks = []
    #ref_vel_picks=[]
    picking_errors=0
    i = 0
    while i < len(w_axis):
        freq = w_axis[i]
        ref_vel = ref_velocities[i]

        # picking criterium 1
        # close to reference curve
        velocities = cross[freq]
        idx_closest_vel = np.abs(velocities - ref_vel).argsort()
        closest_vel =  velocities[idx_closest_vel[0]]
        
        # amplitude criterion: amplitude of nearby maxima in the cc spectrum should be larger
        # than min_amp*max amplitude of the entire cc spectrum
        if min_amp != 0.0:
            if i==0:
                j=1
            elif i==len(w_axis)-1:
                j=len(w_axis)-2
            else:
                j=i
            closest_amp = np.max(np.abs(np.real(corr_spectrum[(frequencies>w_axis[j-1])*(frequencies<w_axis[j+1])])))
            if closest_amp < min_amp*np.max(np.abs(np.real(corr_spectrum))):
                if plotting:
                    print("   freq: %.3f - amplitude of cc spectrum too low!" %freq)
                    plt.plot(np.ones(len(velocities))*freq,velocities,'kx')
                if len(pick)>0:
                    picking_errors += 1
                i+=1
                continue
            
        # next cycle of zero crossings/2pi shift ambiguity; expressed as velocity difference:
        dv_cycle = np.abs(ref_vel - 1./(1./(delta*freq) + 1./ref_vel)) # velocity jump corresponding to 2pi cycle jump
        if len(pick)==0:
            # if more than 3 cycles (denominator) are within the velocity range between min and max vel of the reference curve
            # an unique choice of a first pick is probably not possible anymore
            if dv_cycle/(np.max(ref_curve[:,1]) - np.min(ref_curve[:,1]) + 0.01) < 1/3.:
                if plotting:
                    print("   freq: %.3f - cycles are too close, no first pick can be made. aborting" %freq)
                    print("                dv_cycle: %f" %dv_cycle)
                    print("                test value:",dv_cycle/(np.max(ref_curve[:,1]) - np.min(ref_curve[:,1] + 0.01)))
                break
            if np.abs(closest_vel - ref_vel) > dv_cycle/2.:
                if plotting:
                    print("   freq: %.3f - first zero crossing cannot be uniquely identified." %freq)
                i+=1
                continue
            
            #check if the cycles are not too close to each other (should be bigger than the estimated data error, 0.2?):
            if len(idx_closest_vel)>=2:
                if np.min(np.abs(velocities[idx_closest_vel[1:3]]-closest_vel)) < 0.2:
                    if plotting:
                        print(np.min(np.abs(velocities[idx_closest_vel[1:3]]-closest_vel)))
                        print("cycles are too close!")
                    i+=1
                    if i>20:
                        if plotting:
                            print("Break criterium: Cycles are too close. Cannot find a pick to start.")
                        break
                    continue
            
            #check the first few picks to estimate the data quality
            start_picks = np.array([])
            idxmax = np.int(np.ceil(((np.max(w_axis)-np.min(w_axis))*0.05)/(ref_vel/(2*delta))))
            if idxmax<3:
                idxmax=3
            if idxmax>6:
                idxmax=6
            if len(w_axis)-i < idxmax:
                idxmax=len(w_axis)-i
            for j in range(idxmax):
                idx0 = np.abs(cross[w_axis[i+j]] - ref_velocities[i+j]).argsort()
                start_picks = np.append(start_picks,cross[w_axis[i+j]][idx0[0]])
            if len(start_picks)<2:
                i+=1
                continue
            if np.var(start_picks)/(np.var(ref_velocities[i:i+j+1])+0.00001) > 20:
                if plotting:
                    print(start_picks)
                    print(ref_velocities[i:i+j+1])
                    print("   freq:",freq,"- data variance too high:",np.var(start_picks)/(np.var(ref_velocities[i:i+j+1])+0.00001))
                i+=1
                continue                    

        # picking criterium 2:
        # zero crossings of a bessel function are always in a pi distance
        # scaled by the argument: frequency*distance/velocity
        # with that information we can calculate the freqency of an
        # expected next crossing
        # (assuming the velocity doesn't change dramatically = smoothness criterium)

        # picking criterium 3:
        # the next cycle of zero crossings is associated to a 2pi step, which we want to avoid
        # J(n*2pi*freq*delta/velocity) which results in n possible phase velocity curves
        # picking crit 2 is controlling the jump from one freqency to the next, this controls the jump in velocity
        # if the routine proposes a pick that is around 2pi (more than 3/2 (7/5) pi) from the previous pick
        # it will try to pick another crossing with a smaller velocity jump  
          
        # choose the following picks
        if len(pick)>1:
            freq_step = pick[-1][1]/(2*delta)
            #if freq>0.240:
             #   break                         

           #check for strong increases in velocity (should not be the case for first picks)            
            if len(pick)==2:
                if pick[-1][1]-pick[-2][1] > 0.1:
                    if plotting:
                        print("   freq: %.3f - first two picks go up in velocity more than 0.1km/s, restarting" %freq)
                    pick = []
                    i+=1
                    continue

            if freq - previous_freq < freq_step*0.5:
                if len(pick)<2:
                    if plotting:
                        print("   freq: %.3f - small freq step, probably noisy data, restarting" %freq)
                    pick=[]
                    picking_errors=0
                    continue
                if len(pick)<4 and (pick[-1][0]-pick[0][0])/(np.max(w_axis)-np.min(w_axis)) < 0.10:
                    if plotting:
                        print("   freq: %.3f - small freq step, probably noisy data, restarting" %freq)
                    if len(pick) > len(old_picks):
                        old_picks = pick
                    pick=[]
                    picking_errors = 0
                    if len(pick)>2:
                        i-=2
                    continue
                picking_errors += 1
                if picking_errors > 2: # if the cycle jumps appeared before 10percent were picked, restart
                    if ((pick[-1][0]-pick[0][0])/(np.max(w_axis)-np.min(w_axis)) < 0.10 and len(pick) < 9 and 
                            dv_cycle/(np.max(ref_curve[:,1]) - np.min(ref_curve[:,1])) > 1/2.):
                        if plotting:
                            print("   freq: %.3f - too many jumps, noisy data, restarting" %freq)
                            plt.plot([freq,freq],[1,5],'y',lw=3)
                        if len(pick) > len(old_picks):
                            old_picks = pick
                        pick = []
                        picking_errors=0
                        if len(pick)>2:
                            i-=2
                        continue
                if picking_errors>6:
                    if plotting:
                        print("Break criterium: Too many jumped crossings.")
                    break
# # # # # # predict the next crossing's velocity from linear extrapolation # # # # # # # # # # # # # #
            npicks = np.int(np.ceil(((np.max(w_axis)-np.min(w_axis))*0.05)/freq_step)) #no of picks for slope determination
            # no of picks corresponds to 5% of the freq axis
            if npicks<3:
                npicks = 3
            if npicks>8:
                npicks = 8
            if len(pick)<npicks:
                if i+1<len(ref_velocities):
                    slope = (ref_velocities[i+1]-ref_velocities[i-1])/(w_axis[i+1]-w_axis[i-1])
                else:
                    slope = (ref_velocities[i]-ref_velocities[i-2])/(w_axis[i]-w_axis[i-2])     
                intercept = pick[-1][1] - slope*pick[-1][0]
            # if there are enough picks, take last picks and predict from linear extrapolation
            else:
                if pick[-1][0] - pick[-2][0] < freq_step*2/3. and npicks>3:
                #if there was a small step, the last value is probably shifted, so exclude it from slope determination
                    slope, intercept, r_value, p_value, std_err = \
                    linregress(np.array(pick)[-npicks:-1,0],np.array(pick)[-npicks:-1,1])
                else:
                    slope, intercept, r_value, p_value, std_err = \
                    linregress(np.array(pick)[-npicks:,0],np.array(pick)[-npicks:,1])
                j=1 # only accept negative slopes
                while slope*freq+intercept-pick[-1][1]>0:
                    slope, intercept, r_value, p_value, std_err = \
                    linregress(np.array(pick)[-npicks-j:,0],np.array(pick)[-npicks-j:,1])
                    j+=1
                    if len(pick) < npicks+j:
                        break
                # check whether the picked slope is totally off:
                slope_ref, intercept_ref, r_value, p_value, std_err_ref = \
                  linregress(w_axis[i-npicks:i+1],ref_velocities[i-npicks:i+1])
                if np.abs(np.arctan(slope) - np.arctan(slope_ref)) > np.pi*0.7: #value is chosen after testing
                    if (pick[-1][0]-pick[0][0])/(np.max(w_axis)-np.min(w_axis)) < 0.10:
                        if plotting:
                            plt.plot([freq,freq],[1,5],'g',lw=2)
                            print("restart picking: slope differs too much from ref curve.")
                        if len(pick) > len(old_picks):
                            old_picks = pick
                        pick=[]
                        picking_errors=0
                        i = i-len(pick)+1
                        continue
                    else:
                        if plotting:
                            print("Break criterium: Slope differs too much from reference curve.")
                        break
# # # # # # find all the potential zero crossings that are within a resonable step width # # # # # # # 
            best_next_crossings = []
            predicted_vels = []
            j=0
            while i+j < len(w_axis) and w_axis[i+j] - previous_freq < 2.5*freq_step:
                predicted_vel = slope*w_axis[i+j]+intercept
                idx_potential_next_crossing = np.abs(cross[w_axis[i+j]] - predicted_vel).argmin()
                potential_next_vel = cross[w_axis[i+j]][idx_potential_next_crossing]
                # picking criterium 3:
                 #alternative: abs(2*pi*delta*(freq/preferred_pick - pick[-1][0]/pick[-1][1])) ??? 
                if np.abs(2*np.pi*delta*w_axis[i+j]*(1./potential_next_vel - 1./predicted_vel)) > 0.6*np.pi:
                    best_next_crossings.append([w_axis[i+j],999])
                    predicted_vels.append(predicted_vel)
                    j+=1
                    continue
                # if there has been a complete cycle jump over the last 3 picks, don't take this pick
                if len(pick) > 3:
                    if np.abs(2*np.pi*delta*w_axis[i+j]*(1./potential_next_vel - 1./np.mean(np.array(pick)[-4:-1,1]))) > 1.5*np.pi:
                        best_next_crossings.append([w_axis[i+j],999])
                        predicted_vels.append(predicted_vel)
                        j+=1
                        continue
                best_next_crossings.append([w_axis[i+j],potential_next_vel])
                predicted_vels.append(predicted_vel)
                j+=1
            # test and add also the zero crossing closest to the previous pick
            reference_pick = 0
            if (freq - previous_freq < (4./3.)*freq_step and freq-previous_freq > (2./3)*freq_step and
               np.abs(2*np.pi*delta*freq*(1./velocities[np.abs(velocities-pick[-1][1]).argmin()] - 1./pick[-1][1])) < 0.5*np.pi):
               reference_pick = velocities[np.abs(velocities-pick[-1][1]).argmin()]

            best_next_crossings = np.array(best_next_crossings)
            if reference_pick == 0:
                no_suitable_picks=False
                if len(best_next_crossings) == 0:
                     no_suitable_picks = True
                elif (best_next_crossings[:,1]==999).all():
                    no_suitable_picks = True
                if no_suitable_picks:
                    if len(pick)<8 or (pick[-1][0]-pick[0][0])/(np.max(w_axis)-np.min(w_axis)) < 0.10:
                        if len(pick) > len(old_picks):
                            old_picks = pick
                        pick=[]
                        if len(pick)>2:
                            i-=2
                        if plotting:
                            print("   freq: %.3f - Restarting: No reasonable pick within velocity- and frequency-jump criteria." %freq)
                        continue
                    else:
                        if plotting:
                            print("Break criterium: No reasonable pick within velocity- and frequency-jump criteria.")
                            print("freq:",freq,"previous freq:",previous_freq,"freq step:",freq - previous_freq,"expected step:",freq_step)
                        break

            if len(best_next_crossings) == 0:
                preferred_pick = reference_pick
                if plotting:
                    print("   freq: %.3f - choosing zero crossing closest to the previous pick" %freq)
            elif (best_next_crossings[:,1]==999).all():
                preferred_pick = reference_pick
                if plotting:
                    print("   freq: %.3f - chossing zero crossing closest to the previous pick" %freq)
            else:
                if plotting:
                    v1=1./(7/5./(2*freq*delta) + 1./pick[-1][1])
                    v2=1./(-7/5./(2*freq*delta) + 1./pick[-1][1])
                    bound1.append([freq,v1])
                    bound2.append([freq,v2])
                    for element in best_next_crossings:
                        if element[1]<900:
                            plt.plot(element[0],element[1],'yo')
                # from all picks that fit criteria 2 and 3, the one closest to the prediction is chosen
                # if the very next crossing is relatively good, better not jump any crossing
                if (best_next_crossings[0,0] - previous_freq < (5./3.)*freq_step and 
                    np.abs(2*np.pi*delta*best_next_crossings[0,0]*(1./best_next_crossings[0,1] - 1./predicted_vels[0])) < 0.75*np.pi):
                    idx_preferred_pick = 0
                else:
                    idx_preferred_pick = np.abs(best_next_crossings[:,1] - np.array(predicted_vels)).argmin()
                    if idx_preferred_pick > len(pick):
                        if plotting:
                            print("   freq: %.3f - too noisy data. starting with another pick" %freq)
                        i+=1
                        pick=[]
                        continue
                    if plotting:
                        print("   freq: %.3f - jumping",idx_preferred_pick,"crossings" %freq)
                # check whether the pick closest to the reference curve might be the best
                if np.abs(best_next_crossings[:,1][idx_preferred_pick] - predicted_vels[idx_preferred_pick]) < np.abs(closest_vel - predicted_vels[0]):
                    preferred_pick = best_next_crossings[:,1][idx_preferred_pick]
                    i += idx_preferred_pick 
                    if idx_preferred_pick !=0:
                        picking_errors += 1
                else:
                    #print(freq,"took closest velocity pick")
                    preferred_pick = closest_vel

        else:
            preferred_pick = closest_vel
        freq = w_axis[i]
        pick.append([freq,preferred_pick])
        previous_freq = freq
                
        #ref_vel_picks.append([freq,ref_vel])
        if plotting:
            plt.plot(freq,preferred_pick,'ro',ms=2)
        i+=1
        # if no picks are found and the loop has already passed half the freq axis: break
        if len(pick)<3 and freq>((np.max(w_axis)-np.min(w_axis))/2.+np.min(w_axis)):
            pick=[]
            if plotting:
                print("Break criterium: No picks found over half the freqency axis.")
            break          
    #%%
    # picking done, appending to pick list
    if len(old_picks)>1 and len(pick)==0:
        pick = old_picks
    elif len(old_picks)>1 and len(pick)>1:
        if old_picks[-1][0]-old_picks[0][0] > pick[-1][0]-pick[0][0]:
            pick = old_picks
    pick=np.array(pick)    
    if plotting:
        plt.plot([0,0],[0,0],'kx',label='low amp of cc')
        plt.plot([0,0],[0,0],'yo',label='potential pick')
        if len(pick)>0:
            plt.plot(pick[:,0],pick[:,1],'ro',label='final picks')        
        plt.plot(crossings[:,0],crossings[:,1],'ko',ms=3)
        try:
            plt.plot(np.array(bound1)[:,0],np.array(bound1)[:,1],'g')
            plt.plot(np.array(bound2)[:,0],np.array(bound2)[:,1],'g')
        except:
            pass
        plt.plot(ref_curve[:,0],ref_curve[:,1],lw=2,label='ref_curve')
        plt.xlabel("frequency [Hz]")
        plt.ylabel("velocity [km/s]")
        plt.ylim(1,5)
        plt.legend(numpoints=1)
        plt.show()
    if len(pick)==0:
        raise Exception('Picking phase velocities from zero crossings was not successful.')
    if (pick[-1][0]-pick[0][0])/(np.max(w_axis)-np.min(w_axis)) > 0.10 and len(pick)>3:
        return crossings,pick
    else:
        if plotting:
            print("less than 10% of the frequency axis were picked. discarding...")
        raise Exception('Picking phase velocities from zero crossings was not successful.')
        
        
 #%%###################################################################
def get_smooth_pv(frequencies,corr_spectrum,interstation_distance,ref_curve,\
                        freqmin=0.0,freqmax=99.0, min_vel=1.0, max_vel=5.0,\
                        filt_width=4,filt_height=0.5,x_step=0.5,\
                        pick_threshold=1.7,distortion = 0.0001,\
                        horizontal_polarization=False, smooth_spectrum=False,\
                        plotting=False):
    """
    Function for picking the phase velocity curve from the zero crossings of
    the frequency-domain representation of the stacked cross correlation.\n
    The picking procedure is based on drawing an ellipse around each zero
    crossings and assigning a weight according to the distance from the center
    of the zero crossing to the ellipse boundary. The weights are then stacked
    in the phase-velocity - frequency plot and a smoothed version of the zero-
    crossing branches is obtained. This procedure reduces the influence of
    spurious zero crossings due to data noise, makes it easier to identify the
    well constrained parts of the phase-velocity curve and to obtain a smooth 
    dispersion curve. A reference dispersion curve must be given to guide the
    algorithm in finding the correct phase-velocity branch to start the picking,
    because parallel branches are subject to a 2 pi ambiguity.
    
    :type frequencies: :class:`~numpy.ndarray`
    :param frequencies: 1-D array containing frequency samples of the CC spectrum.
    :type corr_spectrum: :class: `~numpy.ndarray`
    :param corr_spectrum: 1-D or 2-D array containing real or complex CC spectrum.
    :type ref_curve: :class:`~numpy.ndarray`
    :param ref_curve: Reference phase velocity curve, column 1 must contain
        frequencies, column 2 velocities in km/s. The phase velocity curve is
        only picked within the frequency limits of the reference curve.
    :type interstation_distance: float
    :param interstation_distance: Interstation_distance in km.
    :type freqmin, freqmax: float
    :param freqmin, freqmax: Restrict to a range between ``freqmin`` and
        ``freqmax``. Values outside the bounds of ``freq`` are ignored.
    :type min_vel, max_vel: float
    :param min_vel, max_vel: Min and max surface wave velocities for the
        region, only zero crossings corresponding to velocities within these
        boundaries are considered.
    :type filt_width: integer
    :param filt_width: Controls the width of the smoothing window. Corresponds
        to the number of zero crossings that should be within one window.
    :type filt_height: float
    :param filt_height: Controls the height of the smoothing window. Corresponds
        to the portion of a cycle jump. Should never exceed 1, otherwise it will
        smooth over more than one cycle.
    :type x_step: float
    :param x_step: Controls the step width for the picking routine along the
        x (frequency) axis, in fractions of the expected step width between
        two zero crossings. Should be smaller than 1.   
    :type distortion: float
    :param distortion: The ellipses around each zero crossing are drawn in a
        strongly distorted coordinate system that overcompensates the effect of
        the very different units of x (frequency) and y (phase velocity) axis.
        The distortion stabilizes the shape of the ellipses. 
    :type horizontal_polarization: bool
    :param horizontal_polarization: When ``True``, the zero crossings from the spectrum are
        compared to the difference function J0 - J2 (Bessel functions of the
        first kind of order 0 and 2 respectively, see Aki 1957). Appropriate for Love- and the
        radial component of Rayleighwaves. Otherwise only J0 is used.
    :type smooth_spectrum: bool        
    :param smooth_spectrum: When ``True``, the spectrum is smoothed by a filter prior
        to zero-crossing extraction. Normally not necessary if velocity filter has been applied.    
    :type plotting: bool        
    :param plotting: When ``True``, a control plot is created.
        
    :rtype: :class:`~numpy.ndarray`
    :return: Returns two 2-D arrays containing frequencies and corresponding zero crossings
        as well as frequencies and corresponding velocities in km/s.
    """
#%%
    """ for debugging
    frequencies=freqax
    corr_spectrum=ccspec
    interstation_distance=dist
    ref_curve=ref_curve
    freqmin=min_freq
    freqmax=1./min_period
    min_vel=min_vel
    max_vel=max_vel
    filt_width=3
    filt_height=0.5
    x_step=0.5
    pick_threshold=pick_thresh
    horizontal_polarization=horizontal_polarization
    smooth_spectrum=False
    plotting=True
    distortion = 0.0001
    """  

    crossings = get_zero_crossings(frequencies,corr_spectrum,interstation_distance,
                                   freqmin=freqmin,freqmax=freqmax,min_vel=min_vel,max_vel=max_vel,
                                   horizontal_polarization=horizontal_polarization,smooth_spectrum=smooth_spectrum)
    copycrossings = crossings.copy()
    delta=interstation_distance
    w_axis=np.unique(crossings[:,0])
    w_axis=w_axis[(np.min(ref_curve[:,0])<w_axis)*(w_axis<np.max(ref_curve[:,0]))]    
    cross={}
    for w in w_axis:
        cross[w] = crossings[:,1][crossings[:,0] == w]    
    try:
        func=interp1d(ref_curve[:,0],ref_curve[:,1],bounds_error=False,fill_value='extrapolate')
    except:
        raise Exception("Error: please make sure that you are using the latest version of SciPy (>1.0).")

    three_freq_step = np.int(3/x_step)
    picks = []
    picks_backup=[]

#%%
    picks = []
    freq = np.min(w_axis)
    slope_history = np.array([])
    
    vel_plot_vertice = 0.

    X=np.array([])
    Y=np.array([])
    intensity=np.array([])
    idx_intens=np.array([])

    cross_freqs,uniqueidx = np.unique(crossings[:,0],return_inverse=True)
    if plotting:
        ellipse_paths = []    
    
    # PICKING LOOP OVER THE FREQ AXIS
    pick_to_the_end=True
    pick_info = []
    crossings = copycrossings.copy()
    col_no = 0
    while True:
        
        ref_vel = func(freq)
        slope_ref = (func(freq+0.01)-func(freq-0.01))/0.02
        intercept_ref = ref_vel-slope_ref*freq
        
        freq_step = ref_vel/(2*delta)
        
        if len(picks)<3:
            dv_cycle=np.abs(ref_vel-1./(1./(delta*freq) + 1./ref_vel)) # velocity jump corresponding to 2pi cycle jump
        else:
            dv_cycle=np.abs(picks[-1][1]-1./(1./(delta*freq) + 1./picks[-1][1]))
           
        if len(picks)>3:
            Ypoints = np.arange(np.max((picks[-1][1]-5*dv_cycle,min_vel)),np.min((picks[-1][1]+2*dv_cycle,max_vel)),dv_cycle/20.)
        else:
            Ypoints = np.arange(min_vel,max_vel,dv_cycle/20.)
        Xpoints = np.ones(len(Ypoints))*freq
        
        X = np.append(X,Xpoints)
        Y = np.append(Y,Ypoints)
        intensity = np.append(intensity,np.zeros(len(Xpoints)))
        idx_intens = np.append(idx_intens,np.ones(len(Xpoints))*col_no)
                      
        
        pick_to_the_end = False
        if freq>= np.max(w_axis):
            if len(pick_info)>0:
                pick_to_the_end = True
            else:
                break
        elif freq+x_step*freq_step > np.max(w_axis):
            freq += np.max([np.max(w_axis)-freq,0.001])
        else:
            pick_info.append([col_no,freq,freq_step,dv_cycle])
            col_no += 1
            freq+=x_step*freq_step
        
        # if we have added enough points so that the ellipse will fit completely, start creating ellipses
        if freq - pick_info[0][1] > 2*filt_width*pick_info[0][2] or pick_to_the_end:           
            
            cno,fpick,fstep,dvcycle = pick_info.pop(0)
            yaxis = Y[idx_intens==cno]
            
            cross_idx = np.where(crossings[:,0]<=fpick+fstep*filt_width)[0]
        
            values=crossings[cross_idx].copy()
            
            # slope measurement # # #
            if len(picks)>int(three_freq_step/2): #1.5 freq steps for prediction  
                curveslope, intercept, r_value, p_value, std_err = \
                        linregress(np.array(picks)[-int(three_freq_step/2):,0],np.array(picks)[-int(three_freq_step/2):,1])  
                if len(slope_history)>1:
                    slope = 0.5*curveslope + 0.5*np.mean(slope_history[-three_freq_step:])
                    slope_y = picks[-1][1]
            else:
                curveslope = slope_ref
                intercept = intercept_ref
                slope_y = func(fpick)
            if len(picks)>int(three_freq_step):
                accepted_slope = 0.9
            else:
                accepted_slope = 0.7
            if len(picks)>5 and np.abs(np.arctan(slope) - np.arctan(slope_ref)) > np.pi*accepted_slope: #value is chosen after testing
                slope = slope_ref
                if plotting:
                    print("    freq: %.3f - removing picks because of inconsistent slope (accepted slope: %f)" %(picks[-1][0],accepted_slope))
                for j in range(int(three_freq_step/4)+1):
                    picks.remove(picks[-1])
            else:
                slope = 0.1*slope_ref + 0.9*curveslope # for window determination
            if slope>0 or np.isnan(slope):
                slope=slope_ref
            if len(slope_history)>0:
                if slope<np.min(slope_history[-3*three_freq_step:]):
                    slope = np.min(slope_history[-three_freq_step:])
            slope_history = np.append(slope_history,slope)             
            
            #dvcycle_values1 = np.abs(values[:,1]-1./(delta*(values[:,0]-0.01)) + 1./values[:,1])
            #dvcycle_values2 = np.abs(values[:,1]-1./(delta*(values[:,0]+0.01)) + 1./values[:,1])
            dvcycle_before = np.abs(slope_y-1./(1./(delta*(fpick-0.001)) + 1./slope_y))
            dvcycle_after  = np.abs(slope_y-1./(1./(delta*(fpick+0.001)) + 1./slope_y))
            cyclecount = (values[:,1]-slope_y)/dvcycle
            y_before = (values[:,1]-0.001*slope)+cyclecount*dvcycle_before            
            y_after  = (values[:,1]+0.001*slope)+cyclecount*dvcycle_after
            relslopes = (y_after-y_before)/0.002
            relslopes = (relslopes+slope)/2. # downweight the relative slope
                        
            crossings = np.delete(crossings,cross_idx,axis=0)
        
            if plotting and len(values)>0:
                if vel_plot_vertice != 0.:
                    idx_plot = np.abs(values[:,1]-vel_plot_vertice).argmin()
                elif len(picks)>0:
                    idx_plot = np.abs(values[:,1]-picks[-1][1]).argmin()
                else:
                    idx_plot = np.abs(values[:,1]-ref_vel).argmin()
                vel_plot_vertice = values[idx_plot,1]
                idx_plot = [idx_plot-1,idx_plot+1]
                
            # create a ellipse around each zero crossing and check which points are inside
            for j,cross in enumerate(values):
                
                dv_cycle_cross = np.abs(cross[1]-1./(1./(delta*fpick) + 1./cross[1]))
                #distortion = cross[1]/(2*delta)/dv_cycle_cross # because x (frequency) and y (velocity) have very different units

                # shape of the ellipse around each zero crossing
                width = fstep*filt_width
                height = dv_cycle_cross*filt_height
                if np.abs(relslopes[j])*width/2.>height:
                    width = height/np.abs(relslopes[j])*2 
                    
                #if np.abs(np.tan(np.arctan(relslopes[j])-np.pi/2))*height < fstep*filt_width:
                #    height = fstep*filt_width/np.abs(np.tan(np.arctan(relslopes[j])-np.pi/2))
                height *= distortion                                   
                
                # The ellipse
                angle = np.degrees(np.arctan(relslopes[j]*distortion))
                cos_angle = np.cos(np.radians(angle))
                sin_angle = np.sin(np.radians(angle))                                
                
                g_ell_center = (cross[0],cross[1]*distortion)
                g_ell_width = np.abs(cos_angle)*width + np.abs(sin_angle)*height
                g_ell_height = np.abs(cos_angle)*height# + 0.1*np.abs(sin_angle)*width
                    
                cos_angle = np.cos(np.radians(180.-angle))
                sin_angle = np.sin(np.radians(180.-angle))
                
                xc = X - g_ell_center[0]
                yc = Y*distortion - g_ell_center[1]
                
                xct = xc * cos_angle - yc * sin_angle
                yct = xc * sin_angle + yc * cos_angle 
                
                rad_cc = (xct**2/(g_ell_width/2.)**2) + (yct**2/(g_ell_height/2.)**2)
                
                intensity[rad_cc <= 1] += np.cos(np.pi/2.*rad_cc[rad_cc<=1])                

                if plotting and j in idx_plot:
                    # create ellipse
                    ellipse = patches.Ellipse(g_ell_center, g_ell_width, g_ell_height, angle=angle, fill=False, edgecolor='white', linewidth=0.5)
                    # Get the path
                    path = ellipse.get_path()
                    # Get the list of path vertices
                    vertices = path.vertices.copy()
                    # Transform the vertices so that they have the correct coordinates
                    vertices = ellipse.get_patch_transform().transform(vertices)
                    vertices[:,1]/=distortion
                    ellipse_paths.append(vertices)

            # bit of smoothing
            #intensity[cno==idx_intens] -= np.min(intensity[cno==idx_intens])
            #intensity[cno==idx_intens] = gaussian_filter(intensity[cno==idx_intens],dvcycle*15.)
            intensity[cno==idx_intens] = running_mean(intensity[cno==idx_intens],7)
            column = intensity[cno==idx_intens]
    
            # QUALITY CHECKING PREVIOUS PICKS
            # CHECK 1: IS THERE A CYCLE JUMP BETWEEN THE LAST PICK AND THE ONE TWO CYCLES BEFORE?
            if len(picks) > 5:
                idx_pick = np.abs(np.array(picks)[:,0]-(fpick-2*fstep)).argmin()
                if idx_pick>0:                     
                    # cycle jumps that go to higher velocities are more likely to get punished
                    if (picks[-1][1]-picks[idx_pick][1] < -0.25*dvcycle - np.max([np.abs(ref_vel-func(picks[-1][0])),np.abs(slope*(picks[-1][0]-picks[idx_pick][0]))]) or
                        picks[-1][1]-picks[idx_pick][1] >  0.4*dvcycle):
                        if plotting:
                            print("    cycle jump detected")
                            print("    last pick:",picks[-1])
                            print("    pick two cycles before:",picks[idx_pick])
                            print("    freq two cycles before:",fpick-2*fstep)
                        while picks[-1][0]>fpick-1.5*fstep:
                            if plotting:
                                print("    removing",picks[-1])
                            picks.remove(picks[-1])

            # CHECK 2: DID WE JUMP MORE THAN 2 ZERO CROSSINGS?
            if len(picks) > 1:
                if fpick-picks[-1][0] > 2*fstep:
                    if plotting:
                        print("    %.3f: data quality too poor." %fpick)
                    if dvcycle/(np.max(ref_curve[:,1]) - np.min(ref_curve[:,1])) < 1/3.:
                        if plotting:
                            print("    stopping")
                        pick_to_the_end=False
                        break
                    else:
                        if plotting:
                            print("    restarting")
                        if len(picks_backup)<len(picks):
                            picks_backup = picks # save a copy of "bad" picks
                        picks = []

            # IF NO PREVIOUS PICKS HAVE BEEN TAKEN: CHOOSING FIRST PICKS 
            if len(picks)<5:
                if dvcycle/(ref_curve[0,1] - ref_vel) < 0.33:
                    if plotting:
                        print("    dvcycle:",dvcycle,"ref_curve[0,1]-ref_vel :",(ref_curve[0,1] - ref_vel))
                        print("    %.2f: cycles are too close, stopping" %fpick)
                    break

                idxmax = find_peaks(column)[0] #all indices of maxima 
                if len(idxmax)==0:
                    continue
                idxmin = find_peaks(column*-1)[0]
                idxpick = np.abs(yaxis[idxmax]-func(fpick)).argmin() #index of closest maximum
                maxamp = column[idxmax[idxpick]]
                try:
                    if yaxis[idxmin[idxpick]]>yaxis[idxmax[idxpick]]:
                        minamp = np.mean(column[idxmin[idxpick-1:idxpick+1]])
                    else:
                        minamp = np.mean(column[idxmin[idxpick:idxpick+2]]) 
                except:
                    minamp = 0.0001
                    if idxmax[idxpick]==idxmax[-1] or idxmax[idxpick]==idxmax[0]: #there should be one faster and one slower maximum nearby.
                        # otherwise, use the amplitude at +- one cycle from the pick
                        minamp = np.min(column[((yaxis>yaxis[idxmax[idxpick]]-dvcycle)&(yaxis<yaxis[idxmax[idxpick]]+dvcycle))])
                        #minamp = np.min(smoothed[(np.abs(yaxis-(yaxis[idxmax[idxpick]]+dvcycle/2.)).argmin(),
                        #                np.abs(yaxis-(yaxis[idxmax[idxpick]]-dvcycle/2.)).argmin()),j])
                if maxamp/(minamp+1e-5)>pick_threshold*1.25: # higher threshold for first picks
                    closest_vel = yaxis[idxmax[idxpick]]
                    if len(picks)>0:
                        if (np.abs(closest_vel-picks[-1][1])>0.1*dvcycle+np.abs(func(fpick)-func(picks[-1][0])) or 
                            closest_vel>picks[-1][1]+0.1*dvcycle):
                            if plotting:
                                print("   %.3f: pick (%.2f) too far off last pick (%.2f), skipping" %(fpick,closest_vel,picks[-1][1]))
                            picks = [[fpick,closest_vel]]
                            continue
                    picks.append([fpick,closest_vel])
                else:
                    if plotting:
                        print("   %.3f: amplitude of maximum too low, no first pick taken" %fpick)  

                if len(picks)==5:
                    if np.abs(np.mean(np.array(picks)[:,1])-func(fpick))>dvcycle/2.:
                        picks = []
                        if plotting:
                            print("    discarding first picks: too far from reference curve")
                    elif picks[-1][1]-picks[0][1]>0:
                        if plotting:
                            print("   %.3f: discarding picks: increasing velocities" %(fpick))
                        picks = []


            # CHOOSING FOLLOWING PICKS
            else:
                intercept=picks[-1][1]-slope*picks[-1][0]
                v_predicted = slope*fpick+intercept
                idxmax = find_peaks(column)[0] #all indices of maxima 
                if len(idxmax)==0:
                    continue
                idxmin = find_peaks(column*-1)[0]
                idxpick = np.abs(yaxis[idxmax]-v_predicted).argmin() #index of closest maximum
                idxpick2 = np.abs(yaxis[idxmax]-v_predicted).argsort()[1] # index of 2nd closest maximum
                if (np.abs((yaxis[idxmax[idxpick2]]-v_predicted)/(yaxis[idxmax[idxpick]]-v_predicted)) < 1.2 or
                    np.abs(yaxis[idxmax[idxpick2]]-yaxis[idxmax[idxpick]]) < 0.5*dvcycle):
                    if plotting:
                        print("    %.3f: branches are too close to get a good next pick" %(fpick))
                    picks.remove(picks[-1]) #punish
                    continue
                #idxpick = np.abs(yaxis[idxmax]-picks[-1][1]).argmin() #index of closest maximum
                maxamp = column[idxmax[idxpick]]
                #print("\n",fpick,maxamp)
                try:
                    if yaxis[idxmin[idxpick]]>yaxis[idxmax[idxpick]]:
                        minamp = np.mean(column[idxmin[idxpick-1:idxpick+1]]) 
                        #dv_minima = np.diff(yaxis[idxmin[idxpick-1:idxpick+1]])
                    else:
                        minamp = np.mean(column[idxmin[idxpick:idxpick+2]]) 
                        #dv_minima = np.diff(yaxis[idxmin[idxpick:idxpick+2]])                        
                except:
                    minamp = 0.0001

                if idxpick>0 and idxpick<(len(idxmax)-2):
                    dv_maxima = np.mean(np.diff(yaxis[idxmax[idxpick-1:idxpick+2]]))
                    if dv_maxima<0.4*dvcycle or dv_maxima>2.5*dvcycle:
                        picks.remove(picks[-1]) #punish
                        if plotting:
                            print("   %.3f: adjacent maxima are too close/too far away. %.3f %.3f" %(fpick,dv_maxima,dvcycle))
                        continue  
                if maxamp/(minamp+1e-5)>pick_threshold:
                    closest_vel = yaxis[idxmax[idxpick]]
                    if closest_vel-picks[-1][1] < - 0.25*dvcycle - np.max([np.abs(ref_vel-func(picks[-1][0])),np.abs(slope*(picks[-1][0]-fpick))]):
                        if plotting:
                            print("   %.3f: velocityjump to lower velocities too large" %fpick)
                            print("   veljump: %.2f; 0.2*dvcycle: %.2f; reference curve jump: %.2f; predicted jump: %.2f" %(closest_vel-picks[-1][1],-0.2*dvcycle,-np.abs(ref_vel-func(picks[-1][0])),-np.abs(slope*(picks[-1][0]-fpick))))
                            print("slope:",slope)
                            print("last pick:",picks[-1])
                        continue
                        if fpick<w_axis[-1]/10.:
                            picks = []
                    if closest_vel-picks[-1][1] >   np.max([0.3*dvcycle,ref_vel-func(picks[-1][0])]):
                        if plotting:
                            print("   %.3f: velocityjump to higher velocities too large" %fpick)
                            print("   veljump: %.2f; 0.3*dvcycle: %.2f; reference curve jump: %.2f; predicted jump: %.2f" %(closest_vel-picks[-1][1],0.2*dvcycle,np.abs(ref_vel-func(picks[-1][0])),np.abs(slope*(picks[-1][0]-fpick))))
                        continue
                        if fpick<w_axis[-1]/10.:
                            picks = []
                    if np.abs(closest_vel-picks[-1][1]) > np.max([np.abs(func(fpick)-func(picks[-1][0])),10*np.mean(np.abs(np.diff(np.array(picks)[-three_freq_step:,1])))]):
                        if plotting:
                            print("   %.3f: irregular velocity jump (previous: %.2f, current: %.2f" %(fpick,picks[-1][1],closest_vel))
                        continue
                        if fpick<w_axis[-1]/10.:
                            picks = []                        
                    picks.append([fpick,closest_vel])
                else: # if it is currently not possible to take a pick, then remove the last slope in the collection, otherwise the algorithm will append several times the same slope, overweighting the last picks
                    slope_history = slope_history[:-1]
                    if plotting:
                        print("    %.3f: no pick taken" %fpick)


    # END OF PICKING LOOP -------------


#    # PICKING TO THE END OF THE ALREADY SMOOTHED WINDOW
#    if len(picks)>0 and pick_to_the_end:
#        for j in range(int(three_freq_step/2)+1,three_freq_step):
#            idxmax = argrelextrema(smoothed[:,j],np.greater)[0] #all indices of maxima
#            idxmin = argrelextrema(smoothed[:,j],np.less)[0]
#            idxpick = np.abs(yaxis[idxmax]-picks[-1][1]).argmin() #index of closest maximum
#            maxamp = smoothed[:,j][idxmax[idxpick]]
#            try:
#                if yaxis[idxmin[idxpick]]>yaxis[idxmax[idxpick]]:
#                    minamp = np.mean(smoothed[:,j][idxmin[idxpick-1:idxpick+1]]) 
#                else:
#                    minamp = np.mean(smoothed[:,j][idxmin[idxpick:idxpick+2]]) 
#            except:
#                minamp = 0.0001
#            if maxamp/minamp>pick_threshold:
#                closest_vel = yaxis[idxmax[idxpick]]
#                if np.abs(closest_vel-picks[-1][1])>0.3*dv_cycle:
#                    continue
#                picks.append([X[0][-three_freq_step+j],closest_vel])
#
#            # QUALITY CHECKING
#            if freq-picks[-1][0] > 3*freq_step:
#                if plotting:
#                    print("data quality too poor.")
#                if dv_cycle/(np.max(ref_curve[:,1]) - np.min(ref_curve[:,1])) < 1/3.:
#                    if plotting:
#                        print("stopping")
#                    break
#        if len(picks)>0:
#            idx_pick = np.abs(np.array(picks)[:,0]-(freq-2*freq_step)).argmin()
#            if idx_pick>0:
#                if np.abs(picks[-1][1]-picks[idx_pick][1])>0.5*dv_cycle or np.abs(picks[-1][1]-picks[-2][1])>0.5*dv_cycle:
#                    if plotting:
#                        print("cycle jump detected")
#                    while picks[-1][0]>freq-2*freq_step:
#                        if plotting:
#                            print("removing",picks[-1])
#                        picks.remove(picks[-1])
#                            
#    Z_smooth = np.column_stack((Z_smooth,smoothed[:,int(three_freq_step/2):]))
    
    # CHECK IF THE BACKUP PICKS ("BAD" ONES DISCARDED BEFORE) HAVE THE BETTER COVERAGE
    if len(picks_backup)>len(picks):
        picks = picks_backup

    # SMOOTH PICKED PHASE VELOCITY CURVE
    picks=np.array(picks)
    if len(picks)>three_freq_step:
        picksfu = interp1d(picks[:,0],picks[:,1],bounds_error=False)
        picks_interpolated = picksfu(np.unique(X))
        smoothingpicks = picks_interpolated[~np.isnan(picks_interpolated)]
        smooth_picks = running_mean(smoothingpicks,int((1-x_step)*20)+1)
        smooth_picks_x=np.unique(X)[~np.isnan(picks_interpolated)]
    else:
        smooth_picks = []
        smooth_picks_x = [] 
    

    if plotting:   
        plt.ioff()
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111)
        ax2 = ax.twiny()
        fig.subplots_adjust(bottom=0.15)
        #circles = [plt.Circle((xi,yi), radius=0.5, linewidth=0,color=ci) for xi,yi,ci in zip(X.flatten(),y.flatten(),intensity.flatten())]
        #c = matplotlib.collections.PatchCollection(circles)
        #ax.add_collection(c)
        ax.scatter(X,Y,c=intensity)
        #ax.tricontourf(X,Y,intensity)
        x,y = (np.linspace(0,0.5,500),np.linspace(1,6,500))
        xplot,yplot = np.meshgrid(x,y)
        #plt.pcolormesh(xplot,yplot,griddata((X,Y),intensity,(xplot,yplot)))
        #ax.pcolormesh(X,Y,intensity)
        ax.plot(copycrossings[:,0],copycrossings[:,1],'wo',markeredgecolor='black',ms=2,linewidth=0.3,label='zero crossings')
        ax.plot(ref_curve[:,0],ref_curve[:,1],linewidth=2,color='lightblue',label='reference curve')
        #plt.plot(tipx,tipy,'o')
        #for vertices in ellipse_paths[::1]:
        #    ax.plot(vertices[:,0],vertices[:,1],color='white',linewidth=0.5)
        for pick in picks:
            ax.plot(pick[0],pick[1],'o',color='red',ms=2)
        ax.plot([],[],'o',color='red',ms=2,label='picks')
        ax.plot(smooth_picks_x,smooth_picks,'r--',label='smoothed picks')
        ax.legend(loc='upper right',framealpha=0.85)
        ax.set_xlim(0,np.max([fpick+0.1,w_axis[int(len(w_axis)/2)]]))
        ax2.set_xlim(ax.get_xlim())
        new_tick_locations = ax.get_xticks()[ax.get_xticks()<np.max(ax.get_xlim())]
        ax2.xaxis.set_ticks_position("bottom")
        ax2.xaxis.set_label_position("bottom")
        ax2.spines["bottom"].set_position(("axes", -0.1))
        ax2.set_frame_on(True)
        ax2.patch.set_visible(False)
        #for sp in ax2.spines.itervalues():
        #    sp.set_visible(False)
        ax2.spines["bottom"].set_visible(True)
        ax2.set_xticks(new_tick_locations[new_tick_locations>0.])
        ax2.set_xticklabels(np.around(1./new_tick_locations[new_tick_locations>0.],2))
        ax2.set_xlabel("Period [s]")
        ax.set_ylim(min_vel,max_vel)
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Phase Velocity [km/s]")
        ax.annotate("Distance: %d km" %interstation_distance,xycoords='figure fraction',xy=(0.7,0.18),fontsize=12,bbox=dict(boxstyle="square,pad=0.2",fc='white',ec='None', alpha=0.85))
        plt.savefig("example_picking.pdf")
        plt.show()
        

 #   plt.figure()
#    plt.tricontourf(cross_img[:,0],cross_img[:,1],cross_img[:,2])
#    plt.scatter(cross_img[:,0],cross_img[:,1],c=cross_img[:,2])
#    #plt.plot(picks[:,0],picks[:,1],'r')
#    plt.plot(picks[:,0],picks[:,1],'r.')
#    plt.plot(crossings[:,0],crossings[:,1],'ko',ms=3)
#%%
    if len(smooth_picks)> 7*1/x_step or (smooth_picks_x[-1]-smooth_picks_x[0]) > (w_axis[-1]-w_axis[0])/5.: 
        return copycrossings,np.column_stack((smooth_picks_x,smooth_picks))
    else:
        if plotting:
            print("    picked freqency range too short")
            print("   ",len(smooth_picks))
            print("   ",7*1/x_step)
            print(smooth_picks)
        raise Exception('Picking phase velocities from zero crossings was not successful.')

#%%
##############################################################################
def bessel_curve_fitting(freq,corr_spectrum, ref_curve, interstation_distance, freqmin=0.0,\
                       freqmax=99.0, polydegree=4, min_vel=1.0, max_vel=5.0, horizontal_polarization=False,\
                       smooth_spectrum=False,plotting=False):
    """
    still in development
    """
                       
    def besselfu(freq,m4,m3,m2,m1,m0):
        p = np.poly1d([m4,m3,m2,m1,m0])
        return jv(0,freq*2.*np.pi*interstation_distance/p(freq))

    def running_mean(x, N):
        if N%2 == 0:
            N+=1
        x = np.insert(x,0,np.ones(np.int(N/2))*x[0])
        x = np.append(x,np.ones(np.int(N/2))*x[-1])
        cumsum = np.cumsum(np.insert(x, 0, 0)) 
        return (cumsum[N:] - cumsum[:-N]) / N 
        
    spectrum = np.real(corr_spectrum)
    polydegree = 4
    nfree = polydegree +1
    pref = np.polyfit(ref_curve[:,0],ref_curve[:,1],polydegree)
    
    refbessel = (besselfu(freq,pref[0],pref[1],pref[2],pref[3],pref[4]))
    env_ref = np.ones(len(freq))*2.
    env_ref[1:-1] = np.sign(np.abs(refbessel)[1:-1]-np.abs(refbessel)[0:-2])+np.sign(np.abs(refbessel)[1:-1]-np.abs(refbessel)[2:])
    envfu_ref = interp1d(freq[env_ref==2],np.abs(refbessel)[env_ref==2],kind='cubic')
    env_ref = envfu_ref(freq)
    env_sig = np.ones(len(freq))*2.
    env_sig[1:-1] = np.sign(np.abs(spectrum)[1:-1]-np.abs(spectrum)[0:-2])+np.sign(np.abs(spectrum)[1:-1]-np.abs(spectrum)[2:])
    #smooth_envelope = np.abs(spectrum)[env_sig==2]
    #smooth_envelope[1:-1] = running_mean(np.abs(spectrum)[env_sig==2],3)
    smooth_envelope = running_mean(np.abs(spectrum)[env_sig==2],5)
    #extra_smooth_envelope = np.copy(smooth_envelope)
    #extra_smooth_envelope = running_mean(np.abs(spectrum)[env_sig==2],9)
    #freq2 = freq[env_sig==2]
    envfu_sig = interp1d(freq[env_sig==2],smooth_envelope,kind='cubic')
    #envfu_sig = interp1d(freq[env_sig==2],extra_smooth_envelope,kind='cubic')    
    env_sig = envfu_sig(freq)
    #env_sig_smooth = envfu_sig(freq)    
    env_sig_smooth = running_mean(env_sig,51)
    np.ones(len(freq))*np.mean(env_sig)
    
    long_win_avg = running_mean(np.abs(spectrum),int(len(freq)/5))
    #short_win_avg = running_mean(np.abs(spectrum),int(len(freq)/30))
    long_win_env = running_mean(env_sig,int(len(freq))/20)
    test = np.sign(long_win_avg - long_win_env)
    
    #simplemod = np.polyfit(freq,env_sig,2)
    #triglist = []
    #for i in range(100):
   #     triglist.append(int(var(test[i:i+200])/var(refbessel[i:i+200])))
    if plotting:
        plt.figure()
        plt.plot(freq,spectrum,label='real spectrum')
        #plt.plot(freq,refbessel,'--',label='reference curve')
        plt.plot(freq,env_sig)
        plt.plot(freq,env_sig_smooth,':')
        plt.plot(freq,long_win_env)
        plt.plot(freq,long_win_avg)
        plt.plot(freq,test*0.1)
        #plt.plot(freq,np.abs(refbessel*spectrum))        
        #plt.plot(np.arange(len(triglist))*(freq[1]-freq[0]),np.array(triglist)*0.1)
        #plt.plot(freq,running_mean(env_sig,9),':')
        #plt.plot(freq,np.ones(len(freq))*np.mean(env_sig),'k')
        #plt.plot(freq,np.ones(len(freq))*np.mean(np.abs(spectrum)))
        #plt.plot(freq,0.5*np.poly1d(simplemod)(freq),'--')
        plt.legend()
       # print("average envelope:",np.mean(env_sig))
    
#    ascent = np.where(np.abs(np.diff(test))>0)[0]
#    if test[0] == -1:
#        freqmin=freqmin
#        freqmax = np.min([freqmax,freq[ascent[0]]])
#    else:
#        freqmin = np.max([freqmin,freq[ascent[0]]])
#        freqmax = np.min([freqmax,freq[ascent[1]]])
#    ref_curve_idx = np.where((ref_curve[:,0]>=freqmin)&(ref_curve[:,0]<=freqmax))[0]
#    pref = np.polyfit(ref_curve[ref_curve_idx,0],ref_curve[ref_curve_idx,1],polydegree)
    prefmin = pref-0.5*np.abs(pref)
    prefmax = pref+0.5*np.abs(pref)#bounds=(prefmin,prefmax)
    popt,pcov = curve_fit(besselfu,freq[(freq>=freqmin) & (freq<=freqmax)],
                                        (spectrum*(env_ref/env_sig))[(freq>=freqmin) & (freq<=freqmax)],
                                        p0=pref,sigma=None,absolute_sigma=None,check_finite=True,method='trf')

    polyfu = np.poly1d(popt)    
    if plotting:
        plt.plot()
        plt.plot(freq[(freq>=freqmin) & (freq<=freqmax)],(spectrum*(env_ref/env_sig))[(freq>=freqmin) & (freq<=freqmax)],)
        plt.plot(freq[(freq>=freqmin) & (freq<=freqmax)],besselfu(freq[(freq>=freqmin) & (freq<=freqmax)],popt[0],popt[1],popt[2],popt[3],popt[4]),'--',label='best fit')
        plt.plot(freq[(freq>=freqmin) & (freq<=freqmax)],besselfu(freq[(freq>=freqmin) & (freq<=freqmax)],pref[0],pref[1],pref[2],pref[3],pref[4]),'--',label='reference')
        plt.show()
    return freq[(freq>=freqmin) & (freq<=freqmax)],\
                polyfu(freq[(freq>=freqmin) & (freq<=freqmax)]),\
                besselfu(freq[(freq>=freqmin) & (freq<=freqmax)],popt[0],popt[1],popt[2],popt[3],popt[4])


    
