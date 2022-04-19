# -*- coding: utf-8 -*-
"""
Set of functions for the Ambient Noise correlation. Procedure originally 
developed by Kees Weemstra.

Autor: Emanuel Kaestle (emanuel.kaestle@fu-berlin.de)

# # # Updates # # #
Updated April 2022
- The function get_smooth_pv is now able to handle the input of several cross-
  correlation spectra. A kernel density estimate/heatmap is then drawn around 
  all available zero crossings. This can be used in combination with monthly
  spectra for the same station pair.

Updated February 2022
- bugfix in adapt_timespan: the previous function did no always detect if
  one of the streams was empty.
- The noisecorr function does now the correlation with the scipy correlate
  function. This requires additional Fourier transforms and is thus slower.
  However, the signal outside the data window is now zero padded which is 
  the expected behaviour of a correlation. The previous approach implicitly
  assumed that the signal was periodicly repeating (differences are very small).
- The velocity_filter function has been changed to work like a bandpass filter.
  The user has to choose the passband velocities now and the velocity band in
  which the signal is tapered as velband=(6.0,5.0,1.5,0.5), i.e. passband 
  between 5km/s and 1.5km/s, taper between 6-5km/s and 1.5-0.5km/s.
  The previous version was working with a percentage taper which gave some-
  times weird results if the available timespan was too short to fit the
  taper.
  
Updated December 2021
- The get_zero_crossings function now returns also a branch index for each
  zero crossing
- The get_smooth_pv function does a quality check on the zero crossings based
  on the real part of the cross correlation spectrum. The elliptical kernels
  close to the picked dispersion curve are not rotated anymore to avoid biased
  picks.

Updated September 2021
- The adapt timespan function has been rewritten to work in more general cases.
  The function performs more checks and provides more options to correct for
  subsample timeshifts.
- get_smooth_pv has been slightly improved to yield more stable results.
- the waterlevel used in the whitening operation is now calculated with the
  obspy.invsim.waterlevel tool.

Updated July 2021
- The time shift correction has been removed from the cross correlation function.
  The problem of potential time shifts is now solved with the updated 
  adapt_timespan function
- The get_smooth_pv function has been rewritten. The results seem to be
  slightly improved. The documentation and structure of the function should
  be much clearer now.

Updated June 2021
the cross correlation function checks for time shifts and shifts the 
correlation function automatically

Updated May 2018
Adapted for Python3

"""

import numpy as np
from scipy.interpolate import interp1d,griddata
from scipy.special import jn_zeros,jv
from scipy.stats import linregress
from scipy.signal import detrend
from obspy.signal.invsim import cosine_taper, waterlevel
from scipy.signal import find_peaks
from obspy.core import Stream, Trace
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.interpolate import LSQUnivariateSpline
from scipy.optimize import curve_fit

########################
def running_mean(x, N):
    if N>len(x):
        print("Warning: Length of the array is shorter than the number of samples to apply the running mean. Setting N=%d." %len(x))
        N=len(x)
    if N<=1:
        return x
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
def adapt_timespan(st1,st2=None,min_overlap=0.,correct_timeshift=True,
                   interpolate=False,copystreams=True):
    """
    Slices traces from the input to the time ranges where all traces of
    different ids overlap. Traces with identical ids do not need to overlap. 
    Returns the sliced streams.\n
    If only one input stream is provided, the output is also just one stream.
    The result of one joint input stream or two input streams is in general
    identical, unless st1 and st2 contain traces with identical ids.
    Overlapping traces of identical ids are not allowed if they are in the same
    input stream and are automatically cut. This does not happen if they are
    in separate input streams.\n
    If the starttime of the sliced traces do not fit exactly (because of sub-
    sample time shifts), the traces can be shifted or interpolated to correct
    for this time shift. Simple shifting will introduce a subsample timing 
    error.\n
    
    Example:\n
    st1, 4 Traces in Stream, first two traces have an overlap:\n
        NE.STA1..Z | 2000-01-01T00:00:00.000000Z - 2000-01-01T00:59:59.000000Z
        | 1.0 Hz, 3600 samples\n
        NE.STA1..Z | 2000-01-01T00:51:39.000000Z - 2000-01-01T01:30:00.000000Z
        | 1.0 Hz, 2302 samples\n
        NE.STA1..N | 2000-01-01T00:00:00.000000Z - 2000-01-01T00:59:59.000000Z 
        | 1.0 Hz, 3600 samples\n
        NE.STA1..E | 2000-01-01T00:00:00.000000Z - 2000-01-01T00:59:59.000000Z
        | 1.0 Hz, 3600 samples\n
    
    st2, 2 Traces in Stream, quarter sample shifted from the full second:\n
        NE.STA1..Z | 2000-01-01T00:10:03.250000Z - 2000-01-01T00:40:02.250000Z 
        | 1.0 Hz, 1800 samples\n
        NE.STA1..Z | 2000-01-01T00:43:22.250000Z - 2000-01-01T01:10:01.250000Z 
        | 1.0 Hz, 1600 samples\n
    
    There are two common time windows (00:10:03.25 - 00:40:01.25 and 
    00:43:22.25 - 00:59:58.25):\n
    
    st1_out, 6 Traces in Stream:\n
        NE.STA1..Z | 2000-01-01T00:10:03.250000Z - 2000-01-01T00:40:01.250000Z 
        | 1.0 Hz, 1799 samples\n
        NE.STA1..N | 2000-01-01T00:10:03.250000Z - 2000-01-01T00:40:01.250000Z
        | 1.0 Hz, 1799 samples\n
        NE.STA1..E | 2000-01-01T00:10:03.250000Z - 2000-01-01T00:40:01.250000Z
        | 1.0 Hz, 1799 samples\n
        NE.STA1..Z | 2000-01-01T00:43:22.250000Z - 2000-01-01T00:59:58.250000Z
        | 1.0 Hz, 997 samples\n
        NE.STA1..N | 2000-01-01T00:43:22.250000Z - 2000-01-01T00:59:58.250000Z
        | 1.0 Hz, 997 samples\n
        NE.STA1..E | 2000-01-01T00:43:22.250000Z - 2000-01-01T00:59:58.250000Z
        | 1.0 Hz, 997 samples\n
    
    st2_out, 2 Traces in Stream:\n
        NE.STA1..Z | 2000-01-01T00:10:03.250000Z - 2000-01-01T00:40:01.250000Z
        | 1.0 Hz, 1799 samples\n
        NE.STA1..Z | 2000-01-01T00:43:22.250000Z - 2000-01-01T00:59:58.250000Z
        | 1.0 Hz, 997 samples\n
    
    Please note that the result would be different if all the traces were put
    into the same stream and st2=None. This is because of the identical trace
    ids of the Z component:\n
    
    st1, 6 Traces in Stream (st2=None):\n
        NE.STA1..Z | 2000-01-01T00:00:00.000000Z - 2000-01-01T00:59:59.000000Z
        | 1.0 Hz, 3600 samples\n
        NE.STA1..Z | 2000-01-01T00:10:03.250000Z - 2000-01-01T00:40:02.250000Z
        | 1.0 Hz, 1800 samples\n
        NE.STA1..Z | 2000-01-01T00:43:22.250000Z - 2000-01-01T01:10:01.250000Z
        | 1.0 Hz, 1600 samples\n
        NE.STA1..Z | 2000-01-01T01:00:01.000000Z - 2000-01-01T01:30:00.000000Z
        | 1.0 Hz, 1800 samples\n
        NE.STA1..N | 2000-01-01T00:00:00.000000Z - 2000-01-01T00:59:59.000000Z
        | 1.0 Hz, 3600 samples\n
        NE.STA1..E | 2000-01-01T00:00:00.000000Z - 2000-01-01T00:59:59.000000Z
        | 1.0 Hz, 3600 samples\n
    
    Output:\n
    st1_out, 3 Traces in Stream:\n
        NE.STA1..E | 2000-01-01T00:00:00.000000Z - 2000-01-01T00:59:58.000000Z
        | 1.0 Hz, 3599 samples\n
        NE.STA1..N | 2000-01-01T00:00:00.000000Z - 2000-01-01T00:59:58.000000Z
        | 1.0 Hz, 3599 samples\n
        NE.STA1..Z | 2000-01-01T00:00:00.000000Z - 2000-01-01T00:59:58.000000Z
        | 1.0 Hz, 3599 samples\n

    Parameters
    ----------
    st1 : `~obspy.core.stream` or `~obspy.core.trace`
        Obspy stream or trace object containing the first stream whose traces
        should be sliced to the common time range(s).
    st2 : `~obspy.core.stream` or `~obspy.core.trace`
        Obspy stream or trace object containing the second stream whose traces
        should be sliced to the common time range(s).
    min_overlap : float, optional
        The minimum overlap in seconds for two traces. Shorter overlapping
        traces are ignored. The default is 0.
    correct_timeshift : BOOL, optional
        If set to 'True', the starttimes among all traces are being adapted
        so that subsample time shifts are removed. May not work properly if
        the sampling rates differ between traces. The default is True.
    interpolate : BOOL, optional
        If set to 'True', subsample time shifts are corrected by interpolating
        the traces. Otherwise, the traces are just shifted by a subsample time
        shift, if necessary. The default is False.
    copystreams : BOOL, optional
        If set to 'True', a copy of the input streams is returned, otherwise,
        the function acts on the streams directly. The default is True.

    Raises
    ------
    Warning
        A warning is raised if the sampling rates are different between traces.

    Returns
    -------
    st1_out : `~obspy.core.stream`
        Obspy stream object containing the overlapping, sliced
        traces of st1.
    st2_out : `~obspy.core.stream`
        If two input streams were provided, returns the overlapping, sliced
        traces of st2.
    """
    
    if interpolate and not correct_timeshift:
        print("Warning: the interpolate option works only with correct_timeshift=True")
    
    st1 = Stream(st1) if isinstance(st1, Trace) else st1

    if len(st1) == 0:
        if st2 is None:
            return Stream()
        else:
            return Stream(),Stream()
    
    if st2 is None:
        # create a dummy trace
        starttime = min([tr.stats.starttime for tr in st1])
        endtime = max([tr.stats.endtime for tr in st1])        
        st2 = Trace()
        st2.stats.starttime = starttime
        st2.stats.sampling_rate = st1[0].stats.sampling_rate
        st2.data = np.zeros(int(st2.stats.sampling_rate * (endtime-starttime)),
                            dtype=int)
        single_stream = True
    else:
        single_stream = False
    
    st2 = Stream(st2) if isinstance(st2, Trace) else st2
    
    if single_stream and len(st1)==0:
        return Stream()
    elif len(st2)==0:
        return Stream(),Stream()

    # if the sampling rate is not identical, it may not be possible to cut
    # the streams to exactly the same time ranges
    min_sampling_rate = st1[0].stats.sampling_rate
    for trace in (st1+st2):
        if trace.stats.sampling_rate != min_sampling_rate:
            raise Warning("Not all traces have the same sampling rate, this "+
                          "may lead to unexpected results!")
            min_sampling_rate = np.min([min_sampling_rate,
                                        trace.stats.sampling_rate])
    # we require at least 3 samples overlap to avoid some unwanted effects
    min_overlap = np.max([min_overlap+1./min_sampling_rate,
                          3*1./min_sampling_rate])
            
    if copystreams:
        stream1 = st1.copy()
        stream2 = st2.copy()
    else:
        stream1 = st1
        stream2 = st2
      
    # merge overlapping traces of same ids if they have identical 
    # data and sampling
    stream1._cleanup()  
    stream2._cleanup()
      
    # get the different trace ids and remove too short traces
    ids1 = []
    for trace in stream1:
        if trace.stats.endtime-trace.stats.starttime < min_overlap:
            stream1.remove(trace)
        elif not trace.id in ids1:
            ids1.append(trace.id)
    ids2 = []
    for trace in stream2:
        if trace.stats.endtime-trace.stats.starttime < min_overlap:
            stream2.remove(trace)
        elif not trace.id in ids2:
            ids2.append(trace.id)            
    
    ids = ids1+ids2
    
    if single_stream and len(st1)==0:
        return Stream()
    elif len(st2)==0:
        return Stream(),Stream()
    
    # remove overlapping traces of same trace-id within each stream
    for traceid in ids1:
        traces = stream1.select(id=traceid)
        traces = traces.sort()
        endtime = traces[0].stats.starttime
        for trace in traces:
            if trace.stats.starttime < endtime: # overlap
                trace.trim(starttime=endtime,nearest_sample=False)
                if len(trace)==0:
                    stream1.remove(trace)
                    continue
            endtime = trace.stats.endtime + 2./trace.stats.sampling_rate
    for traceid in ids2:
        traces = stream2.select(id=traceid)
        traces = traces.sort()
        endtime = traces[0].stats.starttime
        for trace in traces:
            if trace.stats.starttime < endtime: # overlap
                trace.trim(starttime=endtime,nearest_sample=False)
                if len(trace)==0:
                    stream2.remove(trace)
                    continue
            endtime = trace.stats.endtime + 2./trace.stats.sampling_rate
                
    # get the gaps in all streams, these will be cut out later
    gaps1 = stream1.get_gaps()
    for gap in gaps1:
        if gap[-1] < 0:
            raise Exception("there should be no overlaps in stream1")
    gaps2 = stream2.get_gaps()
    for gap in gaps2:
        if gap[-1] < 0:
            raise Exception("there should be no overlaps in stream2")
    gaps = gaps1+gaps2
            
    # find the overall common time range (latest starttime from all ids to
    # earliest endtime from all ids)
    starttimes = []
    endtimes = []
    for traceid in ids:
        traces = (stream1+stream2).select(id=traceid)
        starttimes.append(min([tr.stats.starttime for tr in traces]))
        endtimes.append(max([tr.stats.endtime for tr in traces]))
            
    starttime = max(starttimes)
    endtime = min(endtimes)
    
    if starttime+min_overlap >= endtime:
        if single_stream:
            return Stream()
        else:
            return Stream(),Stream()
    
    # find timeranges that overlap
    overlapping_timeranges = [(starttime,endtime)]
    for gap in gaps:
        timeranges_updated = []
        gapstart = gap[4]
        gapend = gap[5]
        for timerange in overlapping_timeranges:
            # gap is larger than timerange:
            if gapstart<=timerange[0] and gapend>=timerange[1]:
                continue
            # gap is inside timerange
            if gapstart>timerange[0] and gapend<timerange[1]:
                timeranges_updated += [(timerange[0],gapstart),
                                       (gapend,timerange[1])]
            # gap overlaps with the beginning of the timerange
            elif gapstart<=timerange[0] and gapend>timerange[0]:
                timeranges_updated += [(gapend,timerange[1])]
            # gap overlaps with the end of the timerange
            elif gapstart<timerange[1] and gapend>=timerange[1]:
                timeranges_updated += [(timerange[0],gapstart)]
            # gap does not overlap at all
            else:
                timeranges_updated += [timerange]
                         
        # make sure that the timeranges are at least min_overlap long
        overlapping_timeranges = []
        for timerange in timeranges_updated:
            if timerange[1]-timerange[0] >= min_overlap:
                overlapping_timeranges.append(timerange)
        

    # cut out the timeranges from the stream objects
    st1_out = Stream()
    st2_out = Stream()
    for timerange in overlapping_timeranges:
        
            slice1 = stream1.slice(timerange[0],timerange[1])
            slice2 = stream2.slice(timerange[0],timerange[1])
        
            if correct_timeshift:
                
                if interpolate:
                    
                    starttime = max([tr.stats.starttime for tr in slice1]+
                                    [tr.stats.starttime for tr in slice2])
                    for trace in (slice1+slice2):
                        if ( np.std(trace.data)!=0 and 
                             trace.stats.starttime!=starttime ):
                            trace.interpolate(trace.stats.sampling_rate,
                                              starttime=starttime)
                        else:
                            trace.stats.starttime = starttime
                            
                else:
                    
                    for trace in (slice1+slice2):
                        trace.stats.starttime = timerange[0]
                        
                
                endtime = min([tr.stats.endtime for tr in slice1]+
                              [tr.stats.endtime for tr in slice2])                
                # if correct_timeshift, slice again to make sure that the
                # end sample is also identical
                slice1 = slice1.slice(timerange[0],endtime)
                slice2 = slice2.slice(timerange[0],endtime)
                
                
            st1_out += slice1
            st2_out += slice2
            
            
    if single_stream:
        return st1_out
    else:
        return st1_out,st2_out


##############################################################################
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
              onebit=False,whiten=True, water_level=60,cos_taper=True,\
              taper_width=0.05,subsample_timeshift_interpolation=True):
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
    :type water_level: float
    :param water_level: Waterlevel used for whitening (in dB, 
                                                      see obspy.signal.invsim).
    :type cos_taper: bool
    :param cos_taper: If ``True`` the windowed traces will be tapered before
        the FFT.
    :type taper_width: float
    :param taper_width: Decimal percentage of cosine taper (ranging from 0 to
        1). Default is 0.05 (5%) which tapers 2.5% from the beginning and 2.5%
        from the end.
    :type subsample_timeshift_interpolation: bool
    :param subsample_timeshift_interpolation: If ``True`` traces will be 
        interpolated to correct for subsample time shifts, if necessary.
        
    :rtype: :class:`~numpy.ndarray`
    :return: **freq, corr_spectrum, nwins** - The frequency axis, the stacked
        cross-correlation spectrum and the number of windows used for stacking.
    """
    if ( np.isnan(trace1.data).any() or np.isnan(trace2.data).any() or
         np.isinf(trace1.data).any() or np.isinf(trace2.data).any() ):
        raise Exception("input traces for noisecorr must not contain nan/inf values!")
    if np.std(trace1.data)==0.:
        print("Warning: trace1 is flat!")
    if np.std(trace2.data)==0.:
        print("Warning: trace2 is flat!")
    
    st1,st2=adapt_timespan(trace1,trace2,correct_timeshift=True,
                           interpolate=subsample_timeshift_interpolation)
    tr1=st1[0]
    tr2=st2[0]
    if ((tr1.stats.endtime - tr1.stats.starttime) < window_length or
        (tr2.stats.endtime - tr2.stats.starttime) < window_length):
        raise Exception('Common time range of traces shorter than window length.')
    if tr1.stats.sampling_rate==tr2.stats.sampling_rate:
        dt=1./tr1.stats.sampling_rate
    else:
        raise Exception('Both input streams should have the same sampling rate!')
                         
    win_samples=int(window_length/dt)
    data1 = tr1.data
    data2 = tr2.data
    no_windows=int((len(data1)-win_samples)/((1-overlap)*win_samples))+1
    freq=np.fft.rfftfreq(win_samples,dt)
    
    if cos_taper:
        taper = cosine_taper(win_samples,p=taper_width)     

    # loop for all time windows
    cnt = 0
    for i in range(no_windows):
        window0=int(i*(1-overlap)*win_samples)
        window1=window0+win_samples
        d1=data1[window0:window1]
        d2=data2[window0:window1]
        if len(d1)!=len(d2):
            if i==0:
                raise Exception("common time range shorter than window length!")
            else:
                continue
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
            # water level calculated depending on the spectral amplitudes
            D1/=np.abs(D1)+waterlevel(D1,water_level)
            D2/=np.abs(D2)+waterlevel(D2,water_level)
            
        # actual correlation in the frequency domain
        CORR=np.conj(D1)*D2
        
        if np.isnan(CORR).any() or np.isinf(CORR).any():
            raise Exception("nan/inf value in correlation!")
            
        if np.std(CORR)==0:
            continue

        # summing all time windows
        if i==0:
            SUMCORR=CORR
        else:
            SUMCORR+=CORR
            
        cnt += 1
    
    if cnt > 0:
        #freq=freq[(freq>freqmin) & (freq<=freqmax)]
        return freq, SUMCORR/cnt, cnt
    else:
        raise Exception("cross correlation calculation failed.")


def velocity_filter(freq,corr_spectrum,interstation_distance,
                    velband=(6.0,5.0,1.5,0.5),return_all=False):
    """
    Returns the velocity-filtered cross-correlation spectrum (idea from 
    Sadeghisorkhani). Filter is applied in the time domain after inverse FFT. 
    Signals arriving with velocities outside the velband are set to zero.
    The velband works like a bandpass: Outside the velocity limits the signal
    is set to zero, between the corners there is a cosine taper and between
    the two middle velocities the signal is unaltered.
    This filter can improve the SNR significantly.
    This filter is zero-phase because it is symmetric in the time domain.

    :type freq: :class:`~numpy.ndarray`
    :param freq: Frequency axis corresponding to corr_spectrum.
    :type corr_spectrum: :class:`~numpy.ndarray`
    :param corr_spectrum: Array containing the cross-correlation spectrum. Can be complex
        valued or real.
    :type interstation_distance: float
    :param interstation_distance: Distance between station pair in km.
    :type velband: tuple of floats
    :param velband: Gives the range of allowed velocities similar to a bandpass
        filter in km/s. The upper/lower limit can be deactivated by setting very 
        high/low values.
    :type return_all: bool
    :param return_all: If ``True``, the function returns for arrays: the frequency axis, the filtered
        cc-spectrum, the time-shift axis and the filtered time-domain cross-correlation. Otherwise,
        only the filtered cc-spectrum is returned.
        
    :rtype: :class:`~numpy.ndarray`
    :return: Returns an 1-D array containing the filtered cross-correlation spectrum.
        If return_all=``True``, it also returns the frequency axis, the time-shift axis and the filtered
        time-domain cross-correlation.
    """
    
    dt = 1 / (2 * freq[-1])
    tcorr = np.fft.irfft(corr_spectrum)
    
    c1,c2,c3,c4 = velband
    idx1 = int(interstation_distance/c1/dt)
    idx2 = int(interstation_distance/c2/dt)
    idx3 = int(interstation_distance/c3/dt)
    idx4 = int(interstation_distance/c4/dt)
    
    vel_filt = np.zeros(int(len(tcorr)/2))
    idx1 = np.min([idx1,idx2-1])
    if idx2>=len(vel_filt):
        raise Exception("correlation trace too short. Signals arriving with {c2} are already beyond the maximum lag time.")
    idx3 = np.min([idx3,len(vel_filt)-2])
    idx4 = np.min([idx4,len(vel_filt)-1])
    vel_filt[idx1:idx2] = np.cos(np.linspace(0,np.pi/2.,idx2-idx1))[::-1]
    vel_filt[idx2:idx3] = 1.
    vel_filt[idx3:idx4] = np.cos(np.linspace(0,np.pi/2.,idx4-idx3))
    
    vel_filt = np.hstack((vel_filt,np.roll(vel_filt[::-1],1)))
               
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
    
    if len(pos_crossings)==0 or len(neg_crossings)==0:
        if return_smoothed_spectrum:
            return np.column_stack((w,ccspec)),np.zeros((0,3))
        else:
            return np.zeros((0,3))
        

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
    crossings1 = []
    crossings2 = []
    for j,pcross in enumerate(pos_crossings):
        velocities=pcross*2*np.pi*delta/pos_bessel_zeros
        branch_indices = np.arange(len(velocities))-j
        idx_valid = (velocities>min_vel)*(velocities<max_vel)
        velocities=velocities[idx_valid]
        branch_indices = branch_indices[idx_valid]
        crossings1.append(np.column_stack((np.ones(len(velocities))*pcross,
                                          velocities,branch_indices)))
    for j,ncross in enumerate(neg_crossings):
        velocities=ncross*2*np.pi*delta/neg_bessel_zeros
        branch_indices = np.arange(len(velocities))-j
        idx_valid = (velocities>min_vel)*(velocities<max_vel)
        velocities=velocities[idx_valid]
        branch_indices = branch_indices[idx_valid]
        crossings2.append(np.column_stack((np.ones(len(velocities))*ncross,
                                          velocities,branch_indices))) 
    # check for branch mixups
    crossings1 = np.vstack(crossings1)
    crossings2 = np.vstack(crossings2)
    teststd = []
    for j in range(-1,2):
        testcross = np.vstack((crossings1[crossings1[:,2]==0],
                                crossings2[crossings2[:,2]+j==0]))
        testcross = testcross[testcross[:,0].argsort(),1]
        teststd.append(np.std(np.diff(testcross)))
    shift_index = np.argmin(teststd)-1
    crossings2[:,2] += shift_index
           
    crossings = np.vstack((crossings1,crossings2))
    if len(crossings)>0:
        idxmin = np.abs(np.min(crossings[:,2]))
        crossings[:,2] += idxmin
        crossings = crossings[crossings[:,0].argsort()]

    if return_smoothed_spectrum:
        return np.column_stack((w,ccspec)),crossings
    else:
        return crossings
    

   
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

    crossings = get_zero_crossings(
        frequencies,corr_spectrum,interstation_distance,
        freqmin=freqmin,freqmax=freqmax,min_vel=min_vel,max_vel=max_vel,
        horizontal_polarization=horizontal_polarization,
        smooth_spectrum=smooth_spectrum)
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
    previous_freq = None
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
            idxmax = int(np.ceil(((np.max(w_axis)-np.min(w_axis))*0.05)/(ref_vel/(2*delta))))
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
            npicks = int(np.ceil(((np.max(w_axis)-np.min(w_axis))*0.05)/freq_step)) #no of picks for slope determination
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
def get_smooth_pv(frequencies,corr_spectrum,interstation_distance,ref_curve,
                  freqmin=0.0,freqmax=99.0, min_vel=1.0, max_vel=5.0,
                  filt_width=3,filt_height=1.0,x_step=None,
                  pick_threshold=2.,
                  horizontal_polarization=False, smooth_spectrum=False,
                  check_cross_quality=True,
                  plotting=False,plotitem=None):
    """
    Function for picking the phase velocity curve from the zero crossings of
    the frequency-domain representation of the stacked cross correlation.\n
    The picking procedure is based on drawing an ellipse around each zero
    crossings and assigning a weight according to the distance from the center
    of the zero crossing to the ellipse boundary. The weights are then stacked
    in the phase-velocity - frequency plot and a smoothed version of the zero-
    crossing branches is obtained. Because of the similarity to a kernel-
    density estimate, the elliptical shapes are called kernels in this code.
    This procedure reduces the influence of spurious zero crossings due to data
    noise, makes it easier to identify the well constrained parts of the phase-
    velocity curve and to obtain a smooth dispersion curve. A reference 
    dispersion curve must be given to guide the algorithm in finding the 
    correct phase-velocity branch to start the picking, because parallel 
    branches are subject to a 2 pi ambiguity.
    It is also possible to process several spectra at the same time (e.g.,
    if you have monthly spectra for a certain station pair, it is possible to
    process these together. This can help to reduce effects from single bad
    zero crossings).
    
    :type frequencies: :class:`~numpy.ndarray` or list of arrays
    :param frequencies: 1-D array containing frequency samples of the CC spectrum.
        Alternatively, provide a list of 1-D arrays for several cross spectra
        to be processed jointly.
    :type corr_spectrum: :class: `~numpy.ndarray` or list of arrays
    :param corr_spectrum: 1-D or 2-D array containing real or complex CC spectrum.
        Alternatively, provide a list of 1-D arrays for several cross spectra
        to be processed jointly.
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
        to the number of zero crossings that should be within one window. The
        recommended values is 3-4. Too large values may lead to oversmoothing.
    :type filt_height: float
    :param filt_height: Controls the height of the smoothing window. Corresponds
        to the portion of a cycle jump. Should not be much larger than 1, 
        otherwise it will smooth over more than one cycle.
    :type x_step: float
    :param x_step: Controls the step width for the picking routine along the
        x (frequency) axis, in fractions of the expected step width between
        two zero crossings. Is chosen automatically if not set.   
    :type horizontal_polarization: bool
    :param horizontal_polarization: When ``True``, the zero crossings from the spectrum are
        compared to the difference function J0 - J2 (Bessel functions of the
        first kind of order 0 and 2 respectively, see Aki 1957). Appropriate for Love- and the
        radial component of Rayleighwaves. Otherwise only J0 is used.
    :type smooth_spectrum: bool        
    :param smooth_spectrum: When ``True``, the spectrum is smoothed by a filter prior
        to zero-crossing extraction. Normally not necessary if velocity filter has been applied.
    :type check_cross_quality: bool        
    :param check_cross_quality: When ``True``, the spectrum is analysed and
        zero crossings that coincide with low amplitudes of the real part of 
        the spectrum are discarded. Also crossings which are too close to each
        other are ignored.
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
    filt_height=1.2
    pick_threshold=pick_thresh
    horizontal_polarization=horizontal_polarization
    smooth_spectrum=False
    plotting=True
    """
    
    
    def dv_cycle_jump(frequency,velocity,interstation_distance):
        
        return np.abs(velocity-1./(1./(interstation_distance*frequency) + 
                                   1./velocity))
    
     
    def get_slope(picks,freq, x_step,
                  reference_curve_func, verbose=False):
        """
        function to determine a slope estimate for the picked dispersion curve
        it tries to automatically weight between the slope of the reference
        curve and that of the previous picks. The slope is used to make a
        prediction where the next zero crossing is expected.
        
        freq: frequency (x-axis) where the slope shall be determined
        picks: previous picks
        reference_curve_function: interpolation function for the ref curve
        """
        
        # number of picks to use for prediction
        npicks = np.max([3,int(3/x_step)])
        
        # slope of the reference curve
        slope_ref = ((reference_curve_func(freq+0.02) - 
                      reference_curve_func(freq-0.02))/0.04)        
                 
        # slope and intercept of previous picks
        if len(picks)>npicks: # picks for prediction  
            curveslope, intercept, r_value, p_value, std_err = \
                linregress(np.array(picks)[-npicks:,0],
                            np.array(picks)[-npicks:,1])
        else:
            curveslope = slope_ref
           
        # compare to the slopes between the previous picks
        if len(picks) > 3:
            slope_history = np.diff(np.array(picks),axis=0)
            slope_history = slope_history[:,1]/slope_history[:,0]           
            # trying to predict how the slope is changing from previous steps
            average_slope_change = np.mean(np.diff(slope_history[-10:]))
            slope_pred = np.mean(slope_history[-10:]) + average_slope_change
        else: 
            slope_history = []
            slope_pred = slope_ref
            
        # the final slope is a weighted average between the slope predicted
        # from the previous slopes, the slope from the linear fit to the
        # previous picks and the slope of the reference curve
        slope = 0.25*slope_pred + 0.25*curveslope + 0.5*slope_ref
        
        if len(slope_history) > 3:
            # check whether slope is getting steeper with increasing frequency
            # if yes, make it closer to the reference slope
            if np.abs(slope) > np.mean(np.abs(slope_history[-3:])):
                slope = np.mean(slope_history[-3:])

        # make sure that the slope is negative (lower velocities with
        # increasing frequency
        if slope > 0 or np.isnan(slope):
            slope=0.

        return slope
        

    def get_zero_crossing_slopes(crossings,freq,reference_velocity,slope,width,
                                 reference_curve_func,interstation_distance,
                                 bad_freqs):

        """
        This function attributes a slope to each zero crossing.
        
        This has been simplified so that the slope next to the reference
        curve is always zero. The original version with varying slopes some-
        times led to a bias in the picked phase velocities.
        
        Function needs to be cleaned up from previous code...
        """
           
        crossfreq,uniqueidx = np.unique(crossings[:,0],return_inverse=True)
        freq_idx = np.where(crossfreq==freq)[0]
        cross_idx = np.where(uniqueidx==freq_idx)[0]
        cross = crossings[cross_idx]
                   
        fstep = reference_velocity/(2*interstation_distance)
        dvcycle = dv_cycle_jump(freq,reference_velocity,interstation_distance)

        # first, find the best-fitting slope for the zero crossing wich is
        # closest to the reference velocity
        closest_idx = np.abs(cross[:,1] - reference_velocity).argmin()
        vel = cross[closest_idx,1]
        if np.abs(vel-reference_velocity)>1.:
              return np.zeros(len(cross_idx)), cross_idx, reference_velocity, 0.      
        
        
        """
        # slope of the reference curve
        slope_ref = ((reference_curve_func(freq+0.01) - 
                      reference_curve_func(freq-0.01))/0.02)
        if slope is None:
            slope = slope_ref

        # test a couple of different slopes that may vary between the last
        # slope minus 0.1 cycle jumps and the reference curve slope 
        dslope = np.min([0.1*dvcycle,0.01])/(width/2.*fstep)
        test_slopes = np.arange(slope-dslope,0,1)
    
        if len(test_slopes)==0:# or freq in bad_freqs:
            test_slopes = [slope_ref]
        test_slopes = [0.]
        # loop through all the test slopes and check the clostest distances
        # from a line with the given slope to the next crossings
        dists = np.zeros_like(test_slopes)
        # putting more weight on the next/future crossings
        freqax = np.unique(crossfreq[(crossfreq<=freq+width/2.*fstep)*(crossfreq>=freq)])
        for i,test_slope in enumerate(test_slopes):
            
            v_predicted = vel+test_slope*(freqax-freq)
            
            v_distance = 0.
            for f,v in np.column_stack((freqax,v_predicted)):
                if f in bad_freqs:
                    continue
                cross_v = crossings[crossings[:,0]==f,1]
                if (cross_v>v).all() or (cross_v<v).all():
                    continue
                v_distance += np.min(np.abs(v-cross_v))
            
            dists[i] = v_distance

        best_slope = test_slopes[dists.argmin()]
        
        if best_slope>0:
            best_slope=0.
        """
        # simply use the reference curve slope for the closest zero crossing
        best_slope = ((reference_curve_func(freq+0.01) - 
                       reference_curve_func(freq-0.01))/0.02)
        
        # for all other zero crossings, we model the slope from 'best_slope'.
        cycle_count = (cross[:,1]-vel)/dvcycle

        dvcycle1 = dv_cycle_jump(freq-width/2.*fstep,reference_velocity,
                                  interstation_distance)
        dvcycle2 = dv_cycle_jump(freq+width/2.*fstep,reference_velocity,
                                  interstation_distance)
        
        dv = ((vel+cycle_count*dvcycle2) - 
              (vel+cycle_count*dvcycle1))
        cross_slopes = dv/(width*fstep) + best_slope
        
        # make sure the slopes are not too large (more than one cycle jump
        # over 3 frequency steps)
        maxslope = (dv_cycle_jump(freq,cross[:,1],interstation_distance) / 
                    (3.*fstep))
        mod_idx = np.abs(cross_slopes) > np.abs(maxslope)
        cross_slopes[mod_idx] = maxslope[mod_idx]*np.sign(cross_slopes[mod_idx])
        
        # reduce positive slopes that are probably wrong
        cross_slopes[cross_slopes>0] *= 0.5
        
        # at the main branch, set the previously determined optimal slope
        cross_slopes[closest_idx] = best_slope

        return cross_slopes, cross_idx, vel, best_slope
 
    
    def get_kernel(X,Y,freq,vel,slope,interstation_distance,
                   fstep,filt_width,filt_height,
                   return_poly_coords=False,return_weights=False):
        """
        X,Y: background grid (not regular)
        freq: central_frequency (x-axis) around which the kernel is drawn
        vel: central velocity (y-axis) around which the kernel is drawn
        slope: slope of the elliptically shaped kernel
        
        Description
        function to determine the elliptical kernels around each zero crossing.
        similar to a kernel density estimate (KDE) method, an ellipse is
        drawn around each zero crossing having a weight of 1 at the location
        of the zero crossing which decreases to 0 at its border.
        the phase velocity pick is taken where the weight of the overlapping
        elliptical kernels is maximum which gives smoother picks as compared
        to picking the (noisy) zero crossings directly.
        Shape and size of the elliptical kernels is user determined
        
        """
        # the width of the elliptical kernel
        width = fstep*filt_width
        # sample the ellipse at the frequencies where X has sample points
        freqax = np.unique(X[(X<=freq+width/2.)*(X>=freq-width/2.)])
        # theoretical width of the kernel relative to the actual width
        factor = width/(freqax[-1]-freqax[0])
        
        if freqax[-1]<=freqax[0]:
            print(freq,fstep,filt_width,freqax,np.unique(X))
            raise Exception("here")
        if len(freqax)<2:
            print(fstep,filt_width,freqax,np.unique(X))
            raise Exception("test")
        # along the frequency axis, the weights are 1 at the zero crossing
        # and decrease to 0 towards the edges of the freqax
        xweights = np.abs(np.cos((freq-freqax)/(width/2.)*np.pi/2.))
        # predicted velocities along the frequency axis
        v_predicted = vel+slope*(freqax-freq)
        # predicted distance between cycles along the frequency axis
        dv_cycle = dv_cycle_jump(freqax,vel,interstation_distance)
        dv_cycle[dv_cycle>1] = 1.
        # the height of the ellipse is maximum around the zero crossing
        # and decreases to zero towards the edges of the freqax
        heights = dv_cycle*filt_height*xweights
        
        boundary_coords = None
        if return_poly_coords:
            # boundary points of the polygon patch
            fax_bound = np.linspace(freq-width/2.,freq+width/2.,20)
            xweights_bound = np.sqrt(np.abs(np.cos((freq-fax_bound)/(width/2.)*np.pi/2.)))
            v_predicted_bound = vel+slope*(fax_bound-freq)
            dv_cycle_bound = dv_cycle_jump(fax_bound,v_predicted_bound,
                                           interstation_distance)
            dv_cycle_bound[dv_cycle_bound>.5] = 0.5
            heights_bound = dv_cycle_bound*filt_height*xweights_bound
            boundary_coords = np.vstack((
                np.column_stack((fax_bound,v_predicted_bound+heights_bound/2.)),
                np.column_stack((fax_bound,v_predicted_bound-heights_bound/2.))[::-1],
                np.array([fax_bound[0],v_predicted_bound[0]+heights_bound[0]/2.])))

                    
        poly_weight_ind = np.empty((0,),dtype=int)
        poly_weights = np.empty((0,))
        if return_weights:
            # loop over all frequencies along the freqax and get the weights
            for f_ind,f in enumerate(freqax):
                x_ind = np.where(X==f)[0]
                y = (Y[x_ind]-v_predicted[f_ind])/(heights[f_ind]/2.)
                y_ind = np.where(np.abs(y)<1.)[0]
                yweights = np.zeros(len(y))
                yweights[y_ind] = np.cos(y*np.pi/2.)[y_ind] * xweights[f_ind]
                poly_weight_ind = np.append(poly_weight_ind,x_ind)
                poly_weights = np.append(poly_weights,yweights)
           
        # compensate for boundary effects
        # this increases the weights at the boundaries and reduces the bias
        poly_weights *= (1+factor)/2.
                
        return boundary_coords,poly_weight_ind,poly_weights
        
   
        
    def update_density_field(X,Y,density,crossings,ref_vel,
                             filt_width,filt_height,
                             interstation_distance,
                             distortion=None,idx_plot=[]):
        """
        Adds new kernels to the density field. The kernels are weighted (gauss)
        according to their distance from the reference curve/previous picks.
        This has currently no significant effect since the standard deviation
        is set to a large value. Could be changed but I would have to test
        whether this helps the picking procedure.

        """
        
        ellipse_paths = []
        
        fstep = ref_vel/(2*interstation_distance)
        
        # add the weights to the density field for every zero crossing
        for j,(freq,vel,slope) in enumerate(crossings):
                  
            if j in idx_plot:
                return_poly_coords = True
            else:
                return_poly_coords = False
                
            poly_coords, poly_weight_ind, poly_weights = get_kernel(
                    X,Y,freq,vel,slope,interstation_distance,
                    fstep,filt_width,filt_height,
                    return_poly_coords=return_poly_coords,
                    return_weights=True)
            sig = 1.5 # standard deviation in km/s
            # normalized (max=1) gaussian function that downweights the 
            # contribution from far away branches.
            gauss = (1. / (np.sqrt(2 * np.pi) * sig) * np.exp(
                        -0.5 * np.square(vel - ref_vel) / np.square(sig)) *
                    np.sqrt(2*np.pi)*sig)
            gauss = np.max([gauss,0.1])
            poly_weights *= gauss
            
            density[poly_weight_ind] += poly_weights
                
            if j in idx_plot:
                ellipse_paths.append(poly_coords)

        return density, ellipse_paths

            # Alternative approach, using a rotated ellipse
            # needs a coordinate distortion
            # if np.abs(slope)*width/2.>height:
            #     width = height/np.abs(slope)*2 
                
            # height *= distortion                                
            
            # # The ellipse
            # angle = np.degrees(np.arctan(slope*distortion))
            # #angle=0.
            # cos_angle = np.cos(np.radians(angle))
            # sin_angle = np.sin(np.radians(angle))                                
            
            # g_ell_center = (freq,vel*distortion)
            # g_ell_width = np.abs(cos_angle)*width + np.abs(sin_angle)*height
            # g_ell_height = np.abs(cos_angle)*height# + 0.1*np.abs(sin_angle)*width
                
            # cos_angle = np.cos(np.radians(180.-angle))
            # sin_angle = np.sin(np.radians(180.-angle))
            
            # xc = X - g_ell_center[0]
            # yc = Y*distortion - g_ell_center[1]
            
            # xct = xc * cos_angle - yc * sin_angle
            # yct = xc * sin_angle + yc * cos_angle 
            
            #rad_cc = (xct**2/(g_ell_width/2.)**2) + (yct**2/(g_ell_height/2.)**2)
    
            #weights[rad_cc <= 1] += np.cos(np.pi/2.*rad_cc[rad_cc<=1])
            
                # # get the polygon vertices for plotting
                # # create polygon
                # poly = patches.Polygon(poly_coords,closed=True)
                # # Get the path
                # path = poly.get_path()
                # # Get the list of path vertices
                # vertices = path.vertices.copy()
                # # Transform the vertices so that they have the correct coordinates
                # vertices = poly.get_patch_transform().transform(vertices)
                # #vertices[:,1]/=distortion
                # ellipse_paths.append(vertices)            
                

    def check_previous_picks(picks,picks_backup,frequency,slope,minvel,maxvel,
                             reference_curve_func,freqax_picks,
                             interstation_distance,verbose=False):
        
        # QUALITY CHECKING PREVIOUS PICKS        
        total_picks_jumped = 0
        
        # number of picks to use for prediction
        no_test_picks = np.max([5,int(4/x_step)])
        
        if len(picks_backup)>1 and len(picks)>=1:
            if (np.abs(picks_backup[-1][1] - picks[0][1]) > 
                dv_cycle_jump(picks[0][0], picks[0][1], interstation_distance)):
                picks = []
                return picks,picks_backup,slope,total_picks_jumped
            
        # CHECK 1: IS THERE A CYCLE JUMP BETWEEN THE LAST PICK AND THE ONE THREE CYCLES BEFORE?
        if len(picks) > no_test_picks:
            
            ref_dv = reference_curve_func(picks[-1][0]) - reference_curve_func(picks[-no_test_picks][0])
            
            pickarray = np.array(picks)
            slope_history = np.diff(pickarray,axis=0)
            slope_history = slope_history[:,1]/slope_history[:,0]
            testslope = np.mean(slope_history[-no_test_picks-5:-no_test_picks+1])
                        
            testpicks = pickarray[-no_test_picks:]
            dfreq,dv,_ = testpicks[-1] - testpicks[0]
            dv_cycle = np.mean([dv_cycle_jump(testpicks[-1,0],testpicks[-1,1],interstation_distance),
                                dv_cycle_jump(testpicks[0,0],testpicks[0,1],interstation_distance)])
            
            allowed_vel_reduction = -0.6*dv_cycle + np.min([
                testslope*(picks[-1][0]-picks[-no_test_picks][0]),
                ref_dv])
            
            allowed_vel_increase = 0.3*dv_cycle + slope*(picks[-1][0]-picks[-no_test_picks][0])
            dv_max = np.max([np.max(pickarray[:,1])-np.min(pickarray[:,1]),
                             np.abs(ref_dv),dv_cycle,(maxvel-minvel)/4.])
            
            # jumps to lower velocities
            if dv < allowed_vel_reduction and dv < 0.:
                if verbose:
                    print("    %.3f: cycle jump to lower velocities detected" 
                          %frequency)
                    print("      last pick:",picks[-1])
                    print("      pick before:",picks[-no_test_picks])
                    print("      removing the last picks")
                for i in range(int(no_test_picks/3)):
                    picks.remove(picks[-1])
            
            # jumps to higher velocities are more likely to get punished   
            elif (dv > allowed_vel_increase or dv > 0.2*dv_max) and dv > 0.:
                if verbose:
                    print("    %.3f: cycle jump to higher velocities detected" 
                          %frequency)
                    print("      last pick:",picks[-1])
                    print("      pick before:",picks[-no_test_picks])
                    print("      removing the last picks")  
                for i in range(int(no_test_picks/3)):
                    picks.remove(picks[-1])
          
        # CHECK 2: COUNT HOW MANY PICKS WERE NOT MADE.
        if len(picks) > 0:
            idx0 = np.where(picks[0][0]==freqax_picks)[0][0]
            idxpick = np.where(frequency==freqax_picks)[0][0]
            missing_indices = np.in1d(freqax_picks[idx0:idxpick],
                                      np.array(picks)[:,0],assume_unique=True,
                                      invert=True)
            total_picks_jumped = np.sum(missing_indices)
            # if the total number of jumped picks is too large, abort
            if ((total_picks_jumped > no_test_picks) or 
                (total_picks_jumped >= no_test_picks/2 and len(picks) < no_test_picks) or
                (missing_indices[-int(np.ceil(no_test_picks/1.5)):]).all()):
                if verbose:
                    print("    %.3f: %d picks were not made, data quality too poor." 
                          %(frequency,total_picks_jumped))
                    print("    restarting")
                if len(picks_backup)*2<len(picks):
                    picks_backup = picks[:-1] # save a copy of "bad" picks, without last one
                picks = []
                return picks,picks_backup,slope,total_picks_jumped      

        return picks,picks_backup,slope,total_picks_jumped
        
    
    
    def pick_velocity(picks,frequency,densities,slope,x_step,minvel,maxvel,
                      dvcycle,reference_curve_func,pick_threshold,
                      no_start_picks=3,verbose=False):
        """
        Function to add a new phase-velocity pick to the picks list.       
        
        Parameters
        ----------
        picks : TYPE
            List of picked phase velocities, new pick will be appended to
            that list.
        frequency : TYPE
            Frequency at which the next pick will be made.
        densities : TYPE
            Array containing the result of the kernel densities.
        slope : TYPE
            Slope is used to predict the velocity of the next pick.
        x_step : TYPE
            x_step gives the stepsize between two adjacent picks relative to
            the step between two zero crossings.
        dvcycle : TYPE
            Expected velocity difference between two adjacent cycles.
        reference_curve_func : TYPE
            Function that returns the velocity of the reference curve at a
            given frequency.
        pick_threshold : TYPE
            Picks are only made at density maxima if the maximum is larger
            than pick_threshold times the adjacent minimum.
        no_start_picks : int, optional
            Number of 'first picks' for which more strict picking criteria
            apply. The default is 3.
        verbose : TYPE, optional
            Switch verbose to True or False. The default is False.

        Returns
        -------
        picks : TYPE
            List of picks.

        """
        
        if len(picks)<=no_start_picks:
            # higher pick threshold for the first few picks
            thresh_factor = 1.25
            pick_threshold *= thresh_factor
        else:
            thresh_factor = 1.
        
        velocities, weights = densities
        
        if len(weights)<4:
            return picks
        
        # bit of smoothing 
        # to avoid that there are small density maxima very close to each other
        weights = running_mean(weights,7)
        
        # picks are made where there is a maximum in the density array
        idxmax = find_peaks(weights)[0] #all indices of maxima
        idxmax = idxmax[weights[idxmax]>0.5*np.max(weights)]
        if len(idxmax)==0:
            if verbose:
                print("    %.3f: no maximum" %frequency)
            return picks
        # don't pick anything, if there are only maxima below the reference 
        # curve, unless the velocity difference is less than 20%
        if np.max(velocities[idxmax]) < 0.8*reference_curve_func(frequency):
            if verbose:
                print("    %.3f: too far from reference curve" %frequency)
            return picks
        
        no_start_picks /= x_step
        
        # if there are previous picks, try to predict the next pick
        if len(picks)>=no_start_picks:
            dv_ref = reference_curve_func(frequency)-reference_curve_func(picks[-1][0])
            dv_predicted = slope*(frequency-picks[-1][0])
            v_predicted = picks[-1][1]+dv_predicted
        # otherwise take the reference velocity as prediction
        elif len(picks)>=1:
            dv_ref = reference_curve_func(frequency)-reference_curve_func(picks[-1][0])
            dv_predicted = 0.
            #if len(picks)<no_start_picks:
            #    v_predicted = reference_curve_func(frequency)
            #else:
            v_predicted = picks[-1][1]+dv_ref
        else:
            v_predicted = reference_curve_func(frequency)
                
        idxpick1 = idxmax[np.abs(velocities[idxmax]-v_predicted).argmin()] #index of closest maximum
        vpick = velocities[idxpick1]
        
        # check also the second closest maximum and make sure that the two 
        # maxima are well separated so that the pick is not ambiguous
        if len(idxmax)>1:
            idxpick2 = idxmax[np.abs(velocities[idxmax]-v_predicted).argsort()[1]] # index of 2nd closest maximum
            vpick2 = velocities[idxpick2]
            
            if (np.abs((vpick2-v_predicted)/(vpick-v_predicted+1e-5)) < 1.5 or
                np.abs(vpick2-vpick) < 0.4*dvcycle or
                np.abs(vpick2-vpick) > 2.5*dvcycle):
                if verbose:
                    if np.abs(vpick2-vpick) > 2.5*dvcycle:
                        print("    %.3f: branches are too far apart to get a good pick" %(frequency))
                    else:
                        print("    %.3f: branches are too close to get a good pick" %(frequency))
                        print("        pick1=%.3f pick2=%.3f pick_predicted=%.3f slope=%.3f" 
                              %(vpick,vpick2,v_predicted,slope))
                    return picks
        
        # check the weights at the maximum
        maxamp = weights[idxpick1]
        
        # and the weights at the adjacent minima
        idxmin = find_peaks(weights*-1)[0]
        # if there are no minima
        minamp1 = minamp2 = np.min(weights[(velocities>vpick-dvcycle)*
                                           (velocities<vpick+dvcycle)])
        # if there are minima overwrite minamp1 and minamp2
        if len(idxmin)>0:
            # if there is no minimum above
            if (idxpick1>idxmin).all():
                minamp2 = weights[idxmin[-1]]
            # if there is no minimum below
            elif (idxpick1<idxmin).all():
                minamp1 = weights[idxmin[0]]
            else:
                minamp1 = weights[idxmin[idxmin>idxpick1][0]]
                minamp2 = weights[idxmin[idxmin<idxpick1][-1]]
        minamp = np.mean([minamp1,minamp2])
    
        
        if maxamp > pick_threshold * minamp and maxamp > 0.5*np.max(weights):
            
            if len(picks)>0:
            
                # quality checking current pick
                if (maxamp/np.mean(np.array(picks)[-int(4/x_step):,2]) < 0.5 and 
                    len(picks)>no_start_picks):
                    if verbose:
                        print("    %.3f: amplitude of pick too low" %frequency)
                    return picks
                
                maximum_allowed_velocity_reduction = (
                    -0.3*dvcycle + np.min([dv_ref, dv_predicted]) )
                maximum_allowed_velocity_increase = ( 
                     0.05*dvcycle + np.max([dv_ref, dv_predicted]) )
                                    
                maximum_allowed_velocity_reduction = np.min([
                    -(maxvel-minvel)/100.,maximum_allowed_velocity_reduction])
                maximum_allowed_velocity_increase = np.max([
                    (maxvel-minvel)/100.,maximum_allowed_velocity_increase,0.01])
                
                if len(picks) <= no_start_picks and frequency < 1./30:
                    # at long periods, increasing velocities are very unlikely
                    maximum_allowed_velocity_increase = np.max([
                        0.05*maximum_allowed_velocity_increase,0.01])
                                
                if vpick-picks[-1][1] < maximum_allowed_velocity_reduction:
                    if verbose:
                        print("    %.3f: velocityjump to lower velocities too large" %frequency)
                        print("      veljump: %.2f; allowed jump: %.2f" %(vpick-picks[-1][1],
                                                    maximum_allowed_velocity_reduction))
                        print("      slope:",slope)
                        print("      last pick:",picks[-1])
                    return picks
                
                if vpick-picks[-1][1] > maximum_allowed_velocity_increase:
                    if verbose:
                        print("    %.3f: velocityjump to higher velocities too large" %frequency)
                        print("      veljump: %.2f; allowed jump: %.2f" %(vpick-picks[-1][1],
                                                    maximum_allowed_velocity_increase))
                    return picks
                
                
            else:
                # if the maximum is more than half a cycle away from the reference
                # we cannot be sure that it is actually the correct branch
                if np.abs(v_predicted-vpick)/dvcycle > 0.45:
                    if verbose:
                        print("    %.3f: too far from reference curve, no first pick taken" %frequency)
                        print("           velocity difference: %.2f maximum allowed difference: %.2f" %(np.abs(v_predicted-vpick),dvcycle*0.45))
                    return picks
            
            picks.append([frequency,vpick,maxamp])
       
        else:
            if verbose:
                print("    %.3f: amplitude of maximum too low, no pick taken" %frequency)
                print("      maxamp=%.2f minamp=%.2f vpick=%.2f" %(maxamp,minamp,vpick))
            
        return picks
    
    
    def check_zero_crossing_quality(frequencies,spectrum,zero_crossings,
                                    interstation_distance,
                                    reference_curve_func):
        
        # identify crossings that have a bad quality based on
        # (1) amplitude ratio of spectral amplitudes of the bessel function
        # (2) absolute amplitudes of the bessel function
        # (3) spacing between zero crossings along the frequency axis
        cross_freqs = np.unique(zero_crossings[:,0])
        bad_quality = np.zeros(len(cross_freqs),dtype=bool)
        crossamps = np.zeros(len(cross_freqs))
        peakamps = np.zeros(len(cross_freqs))
        cross_selection = zero_crossings[zero_crossings[:,0]==cross_freqs[0]]
        closest_vel = cross_selection[np.abs(cross_selection[:,1]-reference_curve_func(cross_freqs[0])).argmin(),1]
        for i in range(1,len(cross_freqs)-1):
            idx = np.where((frequencies>cross_freqs[i-1])*(frequencies<cross_freqs[i+1]))[0]
            maxpeak = np.abs(np.max(spectrum.real[idx]))
            minpeak = np.abs(np.min(spectrum.real[idx]))
            crossamps[i] = np.mean([maxpeak,minpeak])
            peakamps[i] = np.max([maxpeak,minpeak])
            if maxpeak>minpeak:
                peak_ratio = maxpeak/minpeak
            else:
                peak_ratio = minpeak/maxpeak
            if peak_ratio>3.:
                bad_quality[i] = True
            if 1./cross_freqs[i] > 30.:
                cross_selection = zero_crossings[zero_crossings[:,0]==cross_freqs[i]]
                new_closest_vel = cross_selection[np.abs(cross_selection[:,1]-closest_vel).argmin(),1]
                if new_closest_vel > closest_vel+0.05:
                    bad_quality[i-1] = True
                closest_vel = new_closest_vel               
        crossamps[0] = crossamps[1]
        crossamps[-1] = crossamps[-2]
        peakamps[0] = peakamps[1]
        peakamps[-1] = peakamps[-2]
        expected_fstep = reference_curve_func(cross_freqs[:-1]+np.diff(cross_freqs))/(2*interstation_distance)
        peakamps = np.interp(np.append(cross_freqs[0],cross_freqs[0]+np.cumsum(expected_fstep)),cross_freqs,peakamps)
        peakamps = running_mean(peakamps,7)
        peakamps = np.interp(cross_freqs,np.append(cross_freqs[0],cross_freqs[0]+np.cumsum(expected_fstep)),peakamps)
        bad_quality[np.append(False,np.diff(cross_freqs)>1.5*expected_fstep)] = True
        bad_quality[crossamps < 0.5*peakamps] = True
        for i in range(1,len(bad_quality)-1):
            if bad_quality[i-1] and bad_quality[i+1]:
                bad_quality[i] = True
                
        return bad_quality
        
        
    #%% # # # # # # # # # # 
    # Main function
    
    # interpolation function for the reference curve
    try:
        reference_curve_func = interp1d(ref_curve[:,0],ref_curve[:,1],
                                        bounds_error=False,
                                        fill_value='extrapolate')
    except:
        raise Exception("Error: please make sure that you are using the "+
                        "latest version of SciPy (>1.0).")
        
    # get the zero crossings of the cross correlation spectrum
    if type(corr_spectrum)!=type([]):
        if len(np.shape(corr_spectrum))==2:
            raise Exception("If you provide more than one spectrum, please join the spectra in a list.")
    if len(frequencies)!=len(corr_spectrum):
        raise Exception("The frequency axis and the correlation spectrum have to be of the same shape (or lists of the same length)")
    if not type(corr_spectrum)==type([]):
        corr_spectrum = [corr_spectrum]
        frequencies = [frequencies]
        
    crossings = np.empty((0,3))
    w_axis_total = np.empty(0)
    bad_quality_total = np.empty(0,dtype=bool)
    smoothed_spectra = []
    for i in range(len(corr_spectrum)):
        
        freqs = frequencies[i]
        spectrum = corr_spectrum[i]

        spectrum_smoothed, zero_crossings = get_zero_crossings(
                freqs,spectrum,interstation_distance,
                freqmin=0,freqmax=freqmax,min_vel=min_vel,max_vel=max_vel,
                horizontal_polarization=horizontal_polarization,
                smooth_spectrum=smooth_spectrum,return_smoothed_spectrum=True)

        if len(np.unique(zero_crossings[:,0]))<7:
            continue
        smoothed_spectra.append(spectrum_smoothed)
    
        w_axis = np.unique(zero_crossings[:,0])
        crossings = np.vstack((crossings,np.column_stack((
            zero_crossings[:,:2],np.zeros(len(zero_crossings))))))
        
        if check_cross_quality:
            bad_quality = check_zero_crossing_quality(freqs,spectrum,
                                                      zero_crossings,
                                                      interstation_distance,
                                                      reference_curve_func)
        else:
            bad_quality = np.zeros(len(freqs),dtype=bool)
        
        w_axis_total = np.hstack((w_axis_total,w_axis))
        bad_quality_total = np.hstack((bad_quality_total,bad_quality))
        
    w_axis,uidx = np.unique(w_axis_total,return_index=True)
    bad_quality = bad_quality_total[uidx]

    if len(w_axis)<7:
        if plotting:
            print("    not enough zero crossings, cannot extract a dispersion curve")
        return zero_crossings,[]
    valid_range = (np.min(ref_curve[:,0])<w_axis)*(w_axis<np.max(ref_curve[:,0]))
    w_axis=w_axis[valid_range] 
    bad_quality=bad_quality[valid_range]
    maxfreq = np.max(w_axis)
    
    # create a logarithmically spaced frequency axis. Final picks are inter-
    # polated and smoothed on this axis.
    npts = 2
    f_max = np.max(ref_curve[:,0])
    f_min = np.min(ref_curve[:,0])
    while True:
        if 1.05**npts * f_min > f_max:
            break
        npts += 1
    logspaced_frequencies = np.logspace(np.log10(f_min),np.log10(f_max),npts)
    df = np.abs(logspaced_frequencies[1]-logspaced_frequencies[0])
    if df>1:
        n_round = 1
    else:
        n_round = int(np.abs(np.log10(df))) + 2
    logspaced_frequencies = np.around(logspaced_frequencies,n_round)
    
    # cut the logspaced_frequencies to the range where there are crossings
    valid_range = np.where((logspaced_frequencies > np.min(w_axis))*
                           (logspaced_frequencies < np.max(w_axis)))[0]
    idx0 = valid_range[0]
    idx1 = np.min([valid_range[-1]+2,len(logspaced_frequencies)])
    logspaced_frequencies = logspaced_frequencies[idx0:idx1]
    
    if x_step is None:
        refbessel = jv(0,2*np.pi*ref_curve[:,0]*interstation_distance/
                       ref_curve[:,1])
        no_crossings = np.sum(refbessel[1:]*refbessel[:-1] < 0)
        x_step = np.around(no_crossings/len(ref_curve[:,0]),1)
        #x_step = np.around(len(w_axis)/np.sum((ref_curve[:,0]>np.min(w_axis))*
        #                                      (ref_curve[:,0]<np.max(w_axis))),1)
        x_step = np.min([x_step,0.9])
        x_step = np.max([x_step,0.3])

    if plotting:
        print("  starting picking (x-step = %.1f)" %x_step)
    picks = []
    picks_backup=[]
    
    X=np.array([])
    Y=np.array([])
    density=np.array([])
    
    sampling = 20
    slope = None
    cross_vel = None
    cross_slope = None
    cross_idx = None
    if plotting:
        ellipse_paths = []
      
    gridaxis = []
    freq = np.min(w_axis)
    while freq <= np.max(w_axis):
        gridaxis.append(freq)
        freq += x_step*reference_curve_func(freq)/(2*interstation_distance)
    gridaxis = np.array(gridaxis)   
    
    # kernel: approximately elliptical shape drawn around each zero crossing
    # that assigns a weight which is maximum at the center of the ellipse
    # (where the zero crossing is) and decreases to 0 towards the border of
    # the ellipse. The elliptical kernels are overlapping. Picks are made
    # where the weight density is maximum.

    # the procedure works with 3 nested loops
    # INNERMOST LOOP creates new gridpoints at which the kernel density is
    # evaluated. This has to be done first, before the kernels are calculated
    # and before the picks are made. This loop advances the fastest along
    # the frequency axis
    #freq_gridpoints = np.min(w_axis)
    idx_gridpoints = 0
    # INNER LOOP creates the (approximately) elliptical kernels around each
    # zero crossing. The values of all kernels are evaluated at the gridpoints
    # and summed to give the density field. This loop runs behind the inner-
    # most loop
    idx_kernel = 0
    # OUTER LOOP runs behind both the inner and the innermost loops. Once the
    # kernels are evaluated it will pick the phase velocities where the kernel
    # density field is maximum.
    #freqax_picks = [np.min(w_axis)]
    #freq_pick = np.min(w_axis)
    #idx_pick = 0
    idx_pick = 0
    
    idx_plot = []
    
    ##########################################################################
    # OUTER LOOP FOR PICKING THE VELOCITIES
    while idx_pick <= idx_gridpoints and idx_pick < len(gridaxis):
        
        freq_pick = gridaxis[idx_pick]
        
        if freq_pick > maxfreq:
            if plotting:
                print("    %.3f: Distance between the next zero crossings " +
                      "seems to be wrong. Stopping." %freq_pick)
            break
        

        ######################################################################
        # INNER LOOP TO UPDATE THE KERNEL DENSITY (HEATMAP) FIELD
        # the density field must be advanced with respect to the picking loop
        while idx_kernel < len(w_axis):
                        
            freq_kernel = w_axis[idx_kernel]
            fstep_kernel = reference_curve_func(freq_kernel)/(2*interstation_distance) 
            
            if freq_kernel > freq_pick + 0.6*filt_width*fstep_kernel or freq_kernel>gridaxis[-1]:
                break
            
            # if there is more than one zero-crossing gap, abort.
            if idx_kernel>1 and maxfreq > w_axis[idx_kernel]:
                if ((w_axis[idx_kernel]-w_axis[idx_kernel-1])/fstep_kernel > 1.5 and
                    (w_axis[idx_kernel-1]-w_axis[idx_kernel-2])/fstep_kernel > 1.5):
                    maxfreq = w_axis[idx_kernel]
            
            kernel_upper_boundary = freq_kernel + filt_width*fstep_kernel
            
            ##################################################################
            # INNERMOST LOOP THAT ADDS NEW GRIDPOINTS
            # the distance between the gridpoints is dependent on the distance
            # between zero crossings (along x-axis) and on the distance between
            # adjacent phase-velocity cyles/branches (along y-axis)
            # new gridpoints at which the density field is computed
            # the gridpoints have again to be created before the density
            # field can be computed
            while (idx_gridpoints < len(gridaxis)):
                
                freq_gridpoints = gridaxis[idx_gridpoints]
                if freq_gridpoints > kernel_upper_boundary:
                    break
                                
                # determine the sampling for the weight field in y-direction (vel)
                # we do not care about the branches that are very far away from the
                # last pick, only take those into account that are close
                if len(picks)>3:
                    dv = dv_cycle_jump(freq_gridpoints,picks[-1][1],
                                       interstation_distance)
                    dy = np.min([0.05,dv/sampling])
                    lower_vlim = np.min([picks[-1][1]-8*dv,
                                         reference_curve_func(freq_gridpoints)-dv])
                    upper_vlim = np.max([picks[-1][1]+2*dv,
                                         reference_curve_func(freq_gridpoints)+dv])
                    Ypoints = np.arange(np.max((lower_vlim,min_vel)),
                                        np.min((upper_vlim,max_vel)),dy)
                else:
                    dv = dv_cycle_jump(freq_gridpoints,
                                       reference_curve_func(freq_gridpoints),
                                       interstation_distance)
                    dy = np.min([0.05,dv/sampling])
                    Ypoints = np.arange(min_vel,max_vel,dy)
                Xpoints = np.ones(len(Ypoints))*freq_gridpoints
                
                X = np.append(X,Xpoints)
                Y = np.append(Y,Ypoints)
                density = np.append(density,np.zeros(len(Xpoints)))
                        
                idx_gridpoints += 1
                # END OF INNERMOST LOOP
            ##################################################################
                
            # the reference velocity is needed to find the optimal rotation
            # angles for the elliptical kernels
            if len(picks)<5:
                reference_velocity = reference_curve_func(freq_kernel)
                #if cross_vel is not None and idx_kernel>2:
                #    reference_velocity = cross_vel
            else:
                reference_velocity = (picks[-1][1] + np.mean([
                     slope*(freq_kernel-picks[-1][0]),
                     reference_curve_func(freq_kernel)-
                     reference_curve_func(picks[-1][0])]) )
            # find the most likely slope associated to each zero crossing
            # these slopes are used to rotate the elliptical kernels
            cross_slopes,cross_idx,cross_vel,cross_slope = (
                get_zero_crossing_slopes(
                    crossings,freq_kernel,reference_velocity,cross_slope,
                filt_width,reference_curve_func,interstation_distance,
                w_axis[bad_quality]))
            crossings[cross_idx,2] = cross_slopes
      
            # add the kernel weights
            # if the plotting option is True, some of the ellipses will be drawn
            # the idx_plot list controls for which zero crossings this will be done
            if plotting:
                idx_plot = []
                if idx_kernel%1==0:
                    #idx_plot = np.arange(len(cross_idx),dtype=int)
                    idx_plot = np.abs(crossings[cross_idx,1]-reference_velocity).argmin()
                    idx_plot = [idx_plot]#[idx_plot-2,idx_plot-1,idx_plot+1]
            
            # get the elliptical kernels and update the weight field
            if not bad_quality[idx_kernel]:
                density, ell_paths = update_density_field(
                    X,Y,density,crossings[cross_idx],
                    reference_velocity,
                    filt_width,filt_height,
                    interstation_distance,idx_plot=idx_plot)
                
                if plotting:
                    ellipse_paths += ell_paths
                
            idx_kernel += 1
    
            # END OF INNER LOOP
        ######################################################################             
        # back to picking loop
        
        # take either the velocity of the last pick or that of the ref curve
        if len(picks)<3:
            reference_velocity = reference_curve_func(freq_pick)
        else:
            reference_velocity = picks[-1][1]
            
        # dv_cycle: velocity jump corresponding to 2pi cycle jump    
        dv_cycle = dv_cycle_jump(freq_pick,reference_velocity,
                                 interstation_distance)

        # estimate the slope of the picked phase velocity curve
        # if there are no picks yet, use the reference curve
        slope = get_slope(picks,freq_pick,x_step,
            reference_curve_func, verbose=plotting)
    
        # check if there are already picks made, if yes, do a quality check
        picks,picks_backup,slope,picks_jumped = check_previous_picks(
            picks,picks_backup,freq_pick,slope,min_vel,max_vel,
            reference_curve_func,gridaxis,interstation_distance,
            verbose=plotting)
        if len(picks)==0 and picks_jumped>0:
            idx_pick -= picks_jumped
            continue
                           
        # check that the parallel phase velocity branches are well separated
        # for taking the first picks. otherwise do not take a first pick
        if len(picks)<3:
            cross_freq = w_axis[np.abs(w_axis-freq_pick).argmin()]
            if (np.sum(crossings[:,0]==cross_freq) > 20 or 
                (dv_cycle < np.max([reference_curve_func(w_axis[0]) - reference_velocity,
                                    (max_vel-min_vel)/10.]))):
                if plotting:
                    print("    dvcycle: %.2f  ref_curve[0,1]-ref_vel : %.2f"
                          %(dv_cycle, ref_curve[0,1]-reference_velocity))
                    print("    %.3f: cycles are too close, stopping" %freq_pick)
                break # terminate picking
        
        # weights at the current frequency where the next pick will be taken
        # Y array contains the velocities along the y axis and weights the
        # corresponding weights. Picks will be taken where the weights are maximum
        pick_density = (Y[freq_pick==X],density[freq_pick==X])

        # pick next velocity, skip if the three last zero crossings had a bad quality
        if freq_pick>freqmin:
            if not (bad_quality[np.abs(w_axis-freq_pick).argsort()][:2+int(len(corr_spectrum)/2)]).all():
                picks = pick_velocity(picks,freq_pick,pick_density,slope,x_step,
                                      min_vel,max_vel,dv_cycle,
                                      reference_curve_func,
                                      pick_threshold,verbose=plotting)
            elif plotting:
                print("    %.3f: bad quality crossings, skipping." %freq_pick)
        elif freq_pick>freqmin and plotting:
            print("    %.3f: bad crossing quality, skipping" %freq_pick)
        if len(picks)>1:
            if np.sum(bad_quality[(w_axis<=freq_pick)*(w_axis>=picks[int(len(picks)/2)][0])]) > 6*len(corr_spectrum):
                if plotting:
                    print("    %.3f: terminating picking, too many zero crossings with bad quality." %freq_pick)
                break
        
        idx_pick += 1
        # END OF PICKING LOOP
    ##########################################################################


    # CHECK IF THE BACKUP PICKS ("BAD" ONES DISCARDED BEFORE) HAVE THE BETTER COVERAGE
    if len(picks_backup)*2>len(picks):
        picks = picks_backup
    
    # SMOOTH PICKED PHASE VELOCITY CURVE
    picks=np.array(picks)    
    if (len(picks) > 3 and np.max(picks[:,0])-np.min(picks[:,0]) > 
                          (np.max(w_axis)-np.min(w_axis))/10.):
        # remove picks that are above the highest/fastest zero crossing
        bad_pick_idx = []
        maxv = 0.
        for freq in w_axis:
            cross = crossings[freq==crossings[:,0]]
            maxv = np.max(np.append(cross[:,1],maxv))
            if maxv > np.max(picks[:,1]):
                break
            bad_pick_idx.append(np.where((picks[:,0]<freq)*(picks[:,1]>maxv))[0])
        if len(bad_pick_idx)>0:
            bad_pick_idx = np.unique(np.hstack(bad_pick_idx))
        picks_smoothed = np.delete(picks,bad_pick_idx,axis=0)
        # smooth and interpolate picks
        if len(picks_smoothed)>2:
            picks_smoothed[:,1] = running_mean(
                picks_smoothed[:,1],np.min([len(picks_smoothed),int(3/x_step)]))
            picksfu = interp1d(picks_smoothed[:,0],picks_smoothed[:,1],
                               bounds_error=False,fill_value=np.nan)
            picks_interpolated = picksfu(logspaced_frequencies)
            smoothingpicks = picks_interpolated[~np.isnan(picks_interpolated)]
            smooth_picks = running_mean(smoothingpicks,7)[2:]
            smooth_picks_x = logspaced_frequencies[~np.isnan(picks_interpolated)][2:]
        else:
            smooth_picks = []
            smooth_picks_x = []
    else:
        if plotting and len(picks)>1:
            print("  picked freq range: %.3f required: %.3f" %(
                np.max(picks[:,0])-np.min(picks[:,0]),
                (np.max(w_axis)-np.min(w_axis))/10.))
        smooth_picks = []
        smooth_picks_x = []
        
    if len(smooth_picks) < len(logspaced_frequencies)/6.:
        if plotting:
            print("  picked range too short")
            print("  picked crossings:",len(picks),"of a total of",len(gridaxis))
        smooth_picks = []
        smooth_picks_x = []         
        
    # PLOT THE RESULTS
    if plotting: 
        plt.ioff()
        fig = plt.figure(figsize=(8,8))
        gs = GridSpec(2,1,figure=fig,hspace=0.15,height_ratios=(1,2))
        ax0 = fig.add_subplot(gs[0])
        for i in range(len(corr_spectrum)):
            ax0.plot(frequencies[i],corr_spectrum[i].real)
        if smooth_spectrum:
            for spectrum_smoothed in smoothed_spectra:
                ax0.plot(spectrum_smoothed[:,0],spectrum_smoothed[:,1].real)
        #ax0.plot(w_axis,peakamps,'k')
        ax0.plot(frequencies[0],np.zeros(len(frequencies[0])),'k',linewidth=0.3)
        if check_cross_quality:
            ax0.plot(w_axis[bad_quality],np.zeros(np.sum(bad_quality)),'ro',
                     markersize=2,label='low quality crossing')
        if len(corr_spectrum)>1:
            ax0.plot([],[],'k-',label='%d correlation spectra' %len(corr_spectrum))
        ax0.legend(loc='upper right',framealpha=0.85)
        ax = fig.add_subplot(gs[1])
        #ax.set_xscale('log')
        ax2 = ax.twiny()
        fig.subplots_adjust(bottom=0.1)
        #ax.scatter(X,Y,c=density)
        distortion = 0.0001
        dy = np.diff(Y)
        dy = np.min(dy[dy>0])
        dx = np.diff(X)
        dx = np.min(dx[dx>0])
        x = np.linspace(np.min(X),np.max(X),int(np.max([100,(np.max(X)-np.min(X))/dx])))
        y = np.linspace(np.min(Y),np.max(Y),int(np.max([100,(np.max(Y)-np.min(Y))/dy])))
        xplot,yplot = np.meshgrid(x,y)
        density_interpolated = griddata((X,Y*distortion),density,(xplot,yplot*distortion))
        for xi in range(len(x)):
            xtest = X[np.abs(x[xi]-X).argmin()]
            testvels = Y[X==xtest]
            density_interpolated[yplot[:,xi]>np.max(testvels),xi] = np.nan
            density_interpolated[yplot[:,xi]<np.min(testvels),xi] = np.nan
        #ax.tricontourf(X,Y,density)
        try:
            vmax=np.nanmax(density_interpolated[xplot>picks[0,0]])
        except:
            vmax=np.nanmax(density_interpolated)
        cbar = ax.pcolormesh(xplot,yplot,density_interpolated,vmin=0,vmax=vmax,
                             shading='nearest')
        #length = np.mean(np.diff(w_axis))/2.
        #for cross in crossings:
        #    ax.plot([cross[0]-length,cross[0]+length],[cross[1]+cross[2]*-length,cross[1]+cross[2]*length],'y-')        
        if len(corr_spectrum)>1:
            ax.plot(crossings[:,0],crossings[:,1],'wo',markeredgecolor='black',
                    ms=4,linewidth=0.1)
        else:
            for branchidx in np.unique(zero_crossings[:,2]):
                ax.plot(zero_crossings[zero_crossings[:,2]==branchidx,0],
                        zero_crossings[zero_crossings[:,2]==branchidx,1],'o',
                        markeredgecolor='black',ms=4,linewidth=0.1)
        ax.plot(ref_curve[:,0],ref_curve[:,1],linewidth=2,color='lightblue',label='reference curve')
        #plt.plot(tipx,tipy,'o')
        if False:
            for vertices in ellipse_paths[::1]:
                ax.plot(vertices[:,0],vertices[:,1],color='white',linewidth=0.5)
        ax.axvline(freq_pick,linestyle='dashed',color='black')
        ax.axvline(freqmin,linestyle='dashed',color='white')
        #for testpnt in test_points:
        #    ax.plot(testpnt[:,0],testpnt[:,1],'r--',linewidth=0.9)
        for pick in picks:
            ax.plot(pick[0],pick[1],'o',color='red',ms=2)
        ax.plot([],[],'o',color='red',ms=2,label='picks')
        ax.plot(smooth_picks_x,smooth_picks,'r--',label='smoothed picks')
        if plotitem is not None:
            ax.plot(plotitem[:,0],plotitem[:,1],'--',color='lightgrey',
                    linewidth=3,label='plotitem')
        ax.legend(loc='upper right',framealpha=0.85)
        ax0.set_xlim(0,np.max([freq_pick+0.1,w_axis[int(len(w_axis)/2)]]))
        ax.set_xlim(0,np.max([freq_pick+0.1,w_axis[int(len(w_axis)/2)]]))
        ax2.set_xlim(ax.get_xlim())
        new_tick_locations = ax.get_xticks()[ax.get_xticks()<np.max(ax.get_xlim())]
        ax2.xaxis.set_ticks_position("bottom")
        ax2.xaxis.set_label_position("bottom")
        ax2.spines["bottom"].set_position(("axes", -0.1))
        ax2.set_frame_on(True)
        ax2.patch.set_visible(False)
        ax2.spines["bottom"].set_visible(True)
        ax2.set_xticks(new_tick_locations[new_tick_locations>0.])
        ax2.set_xticklabels(np.around(1./new_tick_locations[new_tick_locations>0.],2))
        ax2.set_xlabel("Period [s]")
        ax.set_ylim(min_vel,max_vel)
        #ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Phase Velocity [km/s]")
        ax.annotate("Distance: %d km" %interstation_distance,xycoords='figure fraction',
                    xy=(0.7,0.12),fontsize=12,bbox=dict(boxstyle="square,pad=0.2",fc='white',ec='None', alpha=0.85))
        #ax.set_aspect('equal')
        #plt.colorbar(cbar,shrink=0.5)
        plt.savefig("example_picking.png",bbox_inches='tight',dpi=200)
        plt.show()
    #%%
    if False: # additional instructive plot, should not be used
        #%%
        from matplotlib.patches import ConnectionPatch
        fig = plt.figure(figsize=(12,8))
        gs = GridSpec(2,2,figure=fig,hspace=0.35,height_ratios=(1,2),width_ratios=(2.5,1))

        ## RIGHT HAND SIDE OF THE FIGURE
        # cc spectrum zoom in
        ax0 = fig.add_subplot(gs[1])
        ax0.plot(frequencies,corr_spectrum.real)
        if smooth_spectrum:
            ax0.plot(spectrum_smoothed[:,0],spectrum_smoothed[:,1].real)
        ax0.plot(frequencies,np.zeros(len(frequencies)),'k',linewidth=0.3)
        ax0.scatter(w_axis,np.zeros(len(w_axis)),c='None',edgecolors='grey',label='zero crossings')
        ax0.legend(loc='upper right',framealpha=0.85)
        
        # zero crossings zoom in
        ax = fig.add_subplot(gs[3])
        ax2 = ax.twiny()
        fig.subplots_adjust(bottom=0.1)
        distortion = 0.0001
        dy = np.diff(Y)
        dy = np.min(dy[dy>0])
        dx = np.diff(X)
        dx = np.min(dx[dx>0])
        x = np.linspace(np.min(X),np.max(X),int(np.max([100,(np.max(X)-np.min(X))/dx])))
        y = np.linspace(np.min(Y),np.max(Y),int(np.max([100,(np.max(Y)-np.min(Y))/dy])))
        xplot,yplot = np.meshgrid(x,y)
        heatmap = np.zeros_like(xplot).flatten()
        """ DEFINE THE WINDOW TO ZOOM IN HERE:"""
        freqwin = [0.09,0.13]
        velwin = [2.5,4.1]
        testfreq = zero_crossings[np.abs(zero_crossings[:,0]-np.mean(freqwin)).argmin(),0]
        cross_sub = zero_crossings[zero_crossings[:,0]==testfreq]
        testvel = cross_sub[np.abs(cross_sub[:,1]-np.mean(velwin)).argmin(),1]
        boundary_coords,poly_weight_ind,poly_weights = get_kernel(
            xplot.flatten(),yplot.flatten(),testfreq,testvel,0.,interstation_distance,
            reference_curve_func(testfreq)/(2*interstation_distance),
            filt_width,filt_height,return_poly_coords=True,return_weights=True)
        heatmap[poly_weight_ind] = poly_weights
        heatmap = np.reshape(heatmap,xplot.shape)
        cbar = ax.contourf(xplot,yplot,heatmap,vmin=0,vmax=1,
                             levels=20)
        for c in cbar.collections:
            c.set_rasterized(True)
        for freq in w_axis[(w_axis>=freqwin[0])*(w_axis<=freqwin[1])]:
            xyA = (freq,0)
            xyB = (freq,velwin[0])
            con = ConnectionPatch(xyA=xyA, xyB=xyB, coordsA="data", coordsB="data",
                                  axesA=ax0, axesB=ax, color="grey",linestyle='dashed')
            ax.add_artist(con)
        ax.plot(testfreq,testvel,'ko',ms=8)
        for branchidx in np.unique(zero_crossings[:,2]):
            ax.plot(zero_crossings[zero_crossings[:,2]==branchidx,0],
                    zero_crossings[zero_crossings[:,2]==branchidx,1],'o',
                    markeredgecolor='black',ms=6,linewidth=0.1)
        ax.plot(boundary_coords[:,0],boundary_coords[:,1],color='white',linewidth=1)
        xline = np.max(boundary_coords[:,0])+(np.max(boundary_coords[:,0])-np.min(boundary_coords[:,0]))*0.1
        yline = [np.min(boundary_coords[:,1]),np.max(boundary_coords[:,1])]
        ax.plot([xline,xline],yline,'w-')
        ax.text(xline+0.002,np.mean(yline),"kernel height",color='white',rotation=-90,
                verticalalignment='center')
        xline = [np.min(boundary_coords[:,0]),np.max(boundary_coords[:,0])]
        yline = np.max(boundary_coords[:,1])+(np.max(boundary_coords[:,1])-np.min(boundary_coords[:,1]))*0.1
        ax.plot(xline,[yline,yline],'w-')
        ax.text(np.mean(xline),yline+0.05,"kernel width",color='white',rotation=0,
                horizontalalignment='center')
        
        if False:
            for vertices in ellipse_paths[::1]:
                ax.plot(vertices[:,0],vertices[:,1],color='white',linewidth=0.3)

        ax0.set_xlim(0.09,0.13)
        ax.set_xlim(0.09,0.13)
        ax2.set_xlim(ax.get_xlim())
        new_tick_locations = ax.get_xticks()[ax.get_xticks()<np.max(ax.get_xlim())]
        ax2.xaxis.set_ticks_position("bottom")
        ax2.xaxis.set_label_position("bottom")
        ax2.spines["bottom"].set_position(("axes", -0.1))
        ax2.set_frame_on(True)
        ax2.patch.set_visible(False)
        ax2.spines["bottom"].set_visible(True)
        ax2.set_xticks(new_tick_locations[new_tick_locations>0.])
        ax2.set_xticklabels(np.around(1./new_tick_locations[new_tick_locations>0.],2))
        ax2.set_xlabel("Period [s]")
        ax.set_ylim(2.5,4.1)
        ax0.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Phase Velocity [km/s]")
        #ax0.set_title("real part of cross-correlation spectrum",loc='left')
        ax.set_title("zoom in (single kernel)",loc='left')
        cax1 = ax.inset_axes([0.4,0.01,0.55,0.18])
        cax1.set_xticks([])
        cax1.set_yticks([])
        cax1.set_facecolor([1,1,1,0.7])
        cax = cax1.inset_axes([0.1,0.7,0.8,0.2])
        cb = plt.colorbar(cbar,cax=cax,orientation='horizontal',label='kernel weight')
        cb.ax.set_xticks([0,0.5,1])

        ## LEFT HAND SIDE OF THE PLOT
        # cross correlation plot
        ax0 = fig.add_subplot(gs[0])
        ax0.plot(frequencies,corr_spectrum.real)
        if smooth_spectrum:
            ax0.plot(spectrum_smoothed[:,0],spectrum_smoothed[:,1].real)
        ax0.plot(w_axis,peakamps,'k')
        ax0.plot(frequencies,np.zeros(len(frequencies)),'k',linewidth=0.3)
        ax0.plot(w_axis[bad_quality],np.zeros(np.sum(bad_quality)),'ro',
                 markersize=2,label='low quality crossing')
        ax0.legend(loc='upper right',framealpha=0.85)
        # zero crossing plot
        ax = fig.add_subplot(gs[2])
        ax2 = ax.twiny()
        fig.subplots_adjust(bottom=0.1)
        distortion = 0.0001
        dy = np.diff(Y)
        dy = np.min(dy[dy>0])
        dx = np.diff(X)
        dx = np.min(dx[dx>0])
        x = np.linspace(np.min(X),np.max(X),int(np.max([100,(np.max(X)-np.min(X))/dx])))
        y = np.linspace(np.min(Y),np.max(Y),int(np.max([100,(np.max(Y)-np.min(Y))/dy])))
        xplot,yplot = np.meshgrid(x,y)
        density_interpolated = griddata((X,Y*distortion),density,(xplot,yplot*distortion))
        for xi in range(len(x)):
            xtest = X[np.abs(x[xi]-X).argmin()]
            testvels = Y[X==xtest]
            density_interpolated[yplot[:,xi]>np.max(testvels),xi] = np.nan
            density_interpolated[yplot[:,xi]<np.min(testvels),xi] = np.nan
        #ax.tricontourf(X,Y,density)
        try:
            vmax=np.nanmax(density_interpolated[xplot>picks[0,0]])
        except:
            vmax=np.nanmax(density_interpolated)
        cbar = ax.contourf(xplot,yplot,density_interpolated,vmin=0,vmax=vmax,
                             levels=20)
        for c in cbar.collections:
            c.set_rasterized(True)
        for branchidx in np.unique(zero_crossings[:,2]):
            ax.plot(zero_crossings[zero_crossings[:,2]==branchidx,0],
                    zero_crossings[zero_crossings[:,2]==branchidx,1],'o',
                    markeredgecolor='black',ms=4,linewidth=0.1)
        ax.plot(ref_curve[:,0],ref_curve[:,1],linewidth=2,color='lightblue',label='reference curve')
        #plt.plot(tipx,tipy,'o')
        if False:
            for vertices in ellipse_paths[::1]:
                ax.plot(vertices[:,0],vertices[:,1],color='white',linewidth=0.5)
        ax.axvline(freq_pick,linestyle='dashed',color='black')
        ax.axvline(freqmin,linestyle='dashed',color='white')

        for pick in picks:
            ax.plot(pick[0],pick[1],'x',color='black',ms=5)
        ax.plot([],[],'x',color='black',ms=5,label='picks')
        ax.plot(smooth_picks_x,smooth_picks,'r--',label='smoothed picks')
        if plotitem is not None:
            ax.plot(plotitem[:,0],plotitem[:,1],'--',color='lightgrey',
                    linewidth=3,label='plotitem')
        ax.legend(loc='upper right',framealpha=0.9)
        ax0.set_xlim(0,np.max([freq_pick+0.1,w_axis[int(len(w_axis)/2)]]))
        ax.set_xlim(0,np.max([freq_pick+0.1,w_axis[int(len(w_axis)/2)]]))
        ax2.set_xlim(ax.get_xlim())
        new_tick_locations = ax.get_xticks()[ax.get_xticks()<np.max(ax.get_xlim())]
        ax2.xaxis.set_ticks_position("bottom")
        ax2.xaxis.set_label_position("bottom")
        ax2.spines["bottom"].set_position(("axes", -0.1))
        ax2.set_frame_on(True)
        ax2.patch.set_visible(False)
        ax2.spines["bottom"].set_visible(True)
        ax2.set_xticks(new_tick_locations[new_tick_locations>0.])
        ax2.set_xticklabels(np.around(1./new_tick_locations[new_tick_locations>0.],2))
        ax2.set_xlabel("Period [s]")
        ax.set_ylim(min_vel,max_vel)
        ax0.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Phase Velocity [km/s]")
        #ax.annotate("Distance: %d km" %interstation_distance,xycoords='figure fraction',
        #            xy=(0.7,0.12),fontsize=12,bbox=dict(boxstyle="square,pad=0.2",fc='white',ec='None', alpha=0.85))
        ax0.set_title("real part of cross-correlation spectrum",loc='left')
        ax.set_title("zero crossing diagram",loc='left')
        #ax.set_aspect('equal')
        plt.savefig("example_kernel.pdf",bbox_inches='tight',dpi=200)
        plt.show()
        #%%
        

#%%
    if len(smooth_picks) > 0: 
        return crossings,np.column_stack((smooth_picks_x,smooth_picks))
    else:
        return crossings,[]

#%%
"""
##############################################################################
def bessel_curve_fitting(freq,corr_spectrum, ref_curve, interstation_distance, freqmin=0.0,\
                       freqmax=99.0, polydegree=4, min_vel=1.0, max_vel=5.0, horizontal_polarization=False,\
                       smooth_spectrum=False,plotting=False):
    
    # # still in development
    
                       
    def besselfu(freq,m4,m3,m2,m1,m0):
        p = np.poly1d([m4,m3,m2,m1,m0])
        return jv(0,freq*2.*np.pi*interstation_distance/p(freq))

    def running_mean(x, N):
        if N%2 == 0:
            N+=1
        x = np.insert(x,0,np.ones(int(N/2))*x[0])
        x = np.append(x,np.ones(int(N/2))*x[-1])
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

"""
    
