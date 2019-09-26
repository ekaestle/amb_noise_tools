# -*- coding: utf-8 -*-
"""
Created September 2019

@author: emanuel

If you have any questions or suggestions feel free to contact Emanuel Kaestle
(emanuel.kaestle@fu-berlin.de)
"""

###########################################
"""
USER PARAMETERS
Try changing all of these to get a feeling of how the synthetics and the re-
constructed Green's function differ depending on the model, noise source
distribution, whithening, etc..
Later you can try to modify the code in the script itself to create more com-
plex source distributions (for example one sided) or have sources that are
acting simultaneously (currently, sources are active one after another).

"""
# VELOCITY MODEL
# just ONE of these should be True
split_model = False
gradual_model = True
random_model = False
circular_model = False
constant_model = False
dispersion_curve = False
# dispersion curve means that you have a frequency- (i.e. depth-) dependent velocity variation
# all other model only have lateral variations but are not dispersive
# a combination of both (lateral variation and depth variation) is currently not implemented

# STATIONS
interstation_distance = 150 # in km
station_station_azimuth = 30 # in degree

# SOURCES
number_of_sources = 500
# just one source model should be TRUE
source_circle = True # sources on a circle
random_sources = False # sources randomly distributed

circle_radius = 1800 # radius of the sources on a circle in km, has no effect for random sources
source_type = 'random' # you can choose between 'random' (white noise) or 'spike' (single impulse)

# AMBIENT NOISE SIGNAL
# this adds additional random noise to the signal that simulates instrument noise
# this type of noise is different for the same source at station 1 and station 2
random_noise = False

# calculating the cross correlation (OPTIONAL, not necessary for the synthetic data)
# whiten the signals before correlation:
whitening = True

output_folder = 'synthetic_seismograms'

##########################################
""" END OF USER PARAMETERS """
##########################################

if sum([split_model,gradual_model,random_model,circular_model,constant_model,dispersion_curve]) != 1:
      raise Exception("Please just choose one velocity model. Set all to False but one.")
if source_circle and random_sources:
    raise Exception("Please just choose one source model. Either source_circle=True or random_sources=True. Not both.")
if not(source_circle or random_sources):  
    raise Exception("Please choose one source model. Either source_circle=True or random_sources=True. Not both False.")


import numpy as np
from scipy.special import hankel2
from obspy.signal.invsim import cosine_taper
from scipy.interpolate import RectBivariateSpline
import os
import scipy.ndimage as snd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import cm

if not os.path.isdir(os.path.join(os.getcwd(),output_folder)):
    os.makedirs(os.path.join(os.getcwd(),output_folder))

def source(t,sourcetype='random'):
    """
    returns the frequency domain representation of the source time function.
    It is possible to introduce also other types of source, for example a
    wavelet.
    """
    n = len(t)
#    damp=0.05
    if sourcetype=='random':
        signal = (np.random.rand(n)-0.5)  
    #    signal*= t**0.3*exp(-damp*t)
        signal*= cosine_taper(len(signal),0.01)
    elif sourcetype=='spike':
        signal = np.zeros(len(t))
        signal[0] = 1.
    else:
        raise Exception("Did not understand the source type. Choose either 'random' or 'spike'.")
    return np.fft.rfft(signal)
    
def G2Dconvolve_freq(w,dispcurve,distance,source,farfield=False,noise=False):
    """
    Function for convolving the 2D Green's function between a source and a
    receiver with the source signal. Everything is calculated in the frequency
    domain. The calculation is done on a flat 2D surface, so the distance is
    assumed to be a straight line distance.
    
    :type w: :class: `~numpy.ndarray`
    :param w: 1-D array containing frequency samples for the source function
    :type dispcurve: float or :class: `~numpy.ndarray`
    :param dispcurve: Either a float of a single velocity [km/s] or a
        dispersion curve in the frequency domain with the same number of
        samples as ``w``.    
    :type distance: float
    :param distance: distance between source and receiver in km
    :type source: :class: `~numpy.ndarray`
    :param source: Frequency domain representation of the source time function.
        Must have the same number of samples as ``w``.
    :type farfield: bool        
    :param farfield: When ``True``, the farfield approximation of the Green's
        function is used. See for example Bosch and Weemstra, 2015, 
        'Stationary-phase integrals in the cross correlation of ambient noise',
        Equation E16 (exact formula) vs. E17 (farfield approximation)   
    :type noise: bool        
    :param noise: When ``True``, some random noise that does not propagate
        (for example instrument/local noise) is added
        
    :rtype: :class:`~numpy.ndarray`
    :return: Returns two 2-D arrays containing the Green's function between
        source and receiver and recorded signal, i.e. the convolution of the
        Green's function with the source signal. Both in the frequency domain.  
    """
    
    # distance/np.min(dispcurve) is the maximum traveltime from source to station
    # 1./(w[1]-w[0]) is the length of the time axis
    if distance/np.min(dispcurve) > 1./(w[1]-w[0]):
        print("WARNING: the travel time between source and station (%.1fs) " %(distance/np.min(distance))+
              "is longer than the time axis (%.1fs). Ignoring this source. " %(1./(w[1]-w[0]))+
              "Choose sources that are closer to the station or increase the "+
              "length of the time axis to fix this error.")
        return np.nan*np.ones(len(w)),np.nan*np.ones(len(w))
    if farfield:
        greens = 1./(4.*1j*np.pi*dispcurve**(3/2)*np.sqrt(2*np.pi*w*distance))*np.exp(-1j*(2*np.pi*w*distance/dispcurve - np.pi/4.))
    else:
        greens = 1./(4.*1j*np.sqrt(2*np.pi)*dispcurve**2)*hankel2(0,2*np.pi*w*distance/dispcurve)
    greens[0]=3*greens[1] # the first sample is always bad. With this fix the results are nicer
    signal = greens*source # convolution in the time domain is a multiplication in the frequency domain
    if noise:
        signal = signal+(np.random.randn(len(w))*0.01*np.max(signal) + 1j*(np.random.randn(len(w)))*0.01*np.max(signal))
    return greens,signal
    

#%%
# create time and frequency axis
# time axis
t=np.linspace(0,2**10-0.5,2**11)
# frequency axis calculated from the time axis.
w=np.fft.rfftfreq(len(t),t[1]-t[0])

#%%

# define model space
xmap = np.linspace(-2000,2000,1001)
ymap = np.linspace(-2000,2000,1001)
vel = np.zeros((len(xmap),len(ymap)))

# define model
def splitmod(x,y):
        if x<-30.:
            return 3.6
        elif -30.<= x <= 30.:
            return 3.8 + 0.02/3.*x
        elif x > 30.:
            return 4.0
        else:
            print("error in the creation of the split model. Please check the source code")

if split_model:
    # split model
    for i in range(len(xmap)):
        for j in range(len(ymap)):
            vel[i,j] = splitmod(xmap[i],ymap[j])
    #        if xmap[i]<0:
    #        #if xmap[i]*ymap[j]<0:
    #            vel[i,j] = 3.6        
    #        elif xmap[i] == 0:# or ymap[j]==0:
    #            vel[i,j] = 3.8
    #        else:
    #            vel[i,j] = 4.0


if gradual_model:
    # gradual model
    for j in range(len(ymap)):
        vel[:,j] = np.linspace(3.6,4.0,len(xmap))
            
if random_model:
    # random model
    vel = np.random.randint(-1,1,(len(xmap),len(ymap)))
    struct_size = 1000000
    A = 10 #2
    for i in range(struct_size):
        a = np.random.randint(1,len(xmap))
        b = np.random.randint(1,len(ymap))
        vel[a-(A-1):a+A,b-(A-1):b+A] = vel[a,b]
    vel = vel * 0.2
    vel += 3.8

if circular_model:
    # circular model
    vel = np.zeros((len(xmap),len(ymap)))
    x_grid, y_grid = np.meshgrid(xmap,ymap)
    vel[:,:] = 3.6
    vel[(x_grid-0.)**2 + (y_grid-0.)**2 < 250.**2] = 3.8

if constant_model:
    # same velocity everywhere
    vel = np.zeros((len(xmap),len(ymap)))
    x_grid, y_grid = np.meshgrid(xmap,ymap)
    vel[:,:] = 3.8    


if dispersion_curve:
    xmap = xmap[::10]
    ymap = ymap[::10]
    # dummy velocity model, not used
    vel = np.ones((len(xmap),len(ymap)))*3.8
    # dispersion curve
    dispcurve=257.*w**3 - 93.*w**2 + 2.5*w + 3.8
    dispcurve[dispcurve.argmin():]=min(dispcurve)
    dispcurve[:dispcurve.argmax()]=max(dispcurve)
    vel_model = None

# plotting
if not dispersion_curve:   

    # smooth model
    vel = snd.gaussian_filter(vel,sigma=4)

    # testplot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cbar = ax.pcolormesh(xmap,ymap,vel.T,cmap=cm.GMT_haxby_r,rasterized=True)
    plt.colorbar(cbar,label='velocity [km/s]')
    ax.set_xlabel('distance [km]')
    ax.set_ylabel('distance [km]')
    ax.set_aspect('equal')
    ax.set_title("velocity model")
    vel_model = RectBivariateSpline(xmap,ymap,vel,kx=1,ky=1)


#%%
# stations
az = station_station_azimuth/180.*np.pi    
statx=[-interstation_distance*np.cos(az),interstation_distance*np.cos(az)]
staty=[-interstation_distance*np.sin(az),interstation_distance*np.sin(az)]

if not dispersion_curve:
    x_samples = np.arange(300)*np.cos(az) + statx[0]
    y_samples = np.arange(300)*np.sin(az) + staty[0] 
    interstation_mean = np.mean(np.diag(vel_model(np.sort(x_samples),np.sort(y_samples))))
    print("mean interstation velocity:",interstation_mean)


#%%
# sources

if source_circle:
    sphi = np.linspace(0,360,number_of_sources,endpoint=False)
    #sr = []
    #for i in intensity:
    #    sr.append(abs(i-1)*5000.+1000.)
    # # 
    sr = np.ones(len(sphi))*circle_radius
    sphi = sphi/180. * np.pi
    source_x=sr*np.cos(sphi)
    source_y=sr*np.sin(sphi)
  
if random_sources:
    source_x = np.random.uniform(np.min(xmap),np.max(xmap),number_of_sources)
    source_y = np.random.uniform(np.min(ymap),np.max(ymap),number_of_sources)


fig = plt.figure(figsize=(7,5.5))
ax = fig.add_subplot(111)
cbar = ax.pcolormesh(xmap,ymap,vel.T,cmap=cm.GMT_haxby_r,rasterized=True)
ax.scatter(source_x,source_y,s=20,edgecolors='black',label='sources')    
ax.plot(statx[0],staty[0],'rv',label='station 1')
ax.plot(statx[1],staty[1],'bv',label='station 2')
ax.set_xlabel('distance [km]')
ax.set_ylabel('distance [km]')
plt.legend(loc='upper right')
plt.colorbar(cbar,label='velocity [km/s]')
ax.set_aspect('equal') 
plt.savefig("synthetic_example_sources_and_stations.png",bbox_inches='tight')




#%%
# this calculates the synthetic seismograms

fig = plt.figure(figsize=(12,10))

signal = np.zeros((2,len(w)),dtype=complex)
greens = np.zeros((2,len(w)),dtype=complex)
signal_collection = {}
counter = 0
# choose some example sources for plotting
examplesourcenumbers = np.random.randint(0,len(source_x),6)
plotcounter = 0

# calculate synthetics for every source for both stations
for srcx,srcy in np.column_stack((source_x,source_y)):
    # azimuth of the source
    sourceaz = np.arctan2(srcx,srcy)
    # create a different random source signal for every source:
    sourcefreq = source(t,sourcetype=source_type)
    # for the two stations:
    for j in range(2):
        # distance to the station
        r = np.sqrt((srcx-statx[j])**2+(srcy-staty[j])**2)
        # the velocity model should be a function that can be interpolated
        if str(type(vel_model)) == "<class 'scipy.interpolate.fitpack2.RectBivariateSpline'>":
            # average dispersion curve for complex velocity models:
            stationaz = np.arctan2((srcy-staty[j]),(srcx-statx[j]))
            # create a densely sampled line between the source and the station
            r_samples,stepwidth=np.linspace(0,r,int(r/1.),endpoint=False,retstep=True)
            r_samples += stepwidth/2.
            x_samples = r_samples*np.cos(stationaz) + statx[j]
            y_samples = r_samples*np.sin(stationaz) + staty[j]
            # get the average velocity between the source and the station.
            # this is our "dispersion curve for a simple 2D model (no variation with depth)
            dispcurve = r/sum(stepwidth/vel_model.ev(x_samples,y_samples))
            greens[j],signal[j]=G2Dconvolve_freq(w,dispcurve,r,sourcefreq,farfield=False,noise=random_noise)  
        else:
            greens[j],signal[j]=G2Dconvolve_freq(w,dispcurve,r,sourcefreq,farfield=False,noise=random_noise)  
            
    if np.isnan(signal[0]).any() or np.isnan(signal[1]).any():
        continue
        
    # storing the result
    signal_collection[sourceaz] = signal.copy() 
    
    signal1_tdomain = np.fft.irfft(signal[0])
    signal1_tdomain /= np.max(signal1_tdomain)
    greens1_tdomain = np.fft.irfft(greens[0])
    greens1_tdomain /= np.max(greens1_tdomain)
    
    signal2_tdomain = np.fft.irfft(signal[1])
    signal2_tdomain /= np.max(signal2_tdomain)
    greens2_tdomain = np.fft.irfft(greens[1])
    greens2_tdomain /= np.max(greens2_tdomain)    
    
    # plotting
    if counter in examplesourcenumbers:
        ax0 = fig.add_subplot(6,3,plotcounter*3+1)
        ax0.plot(t,np.fft.irfft(sourcefreq),label='signal source %d' %(counter+1))
        ax0.legend(loc='upper right')
        ax0.set_yticks([])
        if plotcounter != 5:
            ax0.set_xticks([])
            
        ax01 = fig.add_subplot(6,3,plotcounter*3+2)
        ax01.plot(t,greens1_tdomain,label='Greens function src stat1')
        ax01.plot(t,greens2_tdomain,label='Greens function src stat2')
        ax01.legend(loc='upper right')
        ax01.set_yticks([])
        if plotcounter != 5:
            ax01.set_xticks([])
            
        ax1 = fig.add_subplot(6,3,plotcounter*3+3)
        ax1.plot(t,signal1_tdomain,label='signal stat 1')
        ax1.plot(t,signal2_tdomain,label='signal stat 2')
        ax1.legend(loc='upper right')
        ax1.set_yticks([])
        if plotcounter != 5:
            ax1.set_xticks([])
  
        plotcounter += 1
          
        if plotcounter == 5:
            ax0.set_xlabel("time [s]")
            ax01.set_xlabel("time [s]")
            ax1.set_xlabel("time [s]")
            plt.suptitle("source signal convolved with Green's function gives the signal at the station\n"+
                         "note the time shift between the signals at station 1 and 2.")
            plt.savefig("synthetic_example_synthetic_seismograms.png",bbox_inches='tight')

    counter += 1
        
    # writing to file
    header="Station x,y: %.5f %.5f\nSource x,y: %.5f %.5f\ntime[s]\tsignal\n" %(statx[j],staty[j],srcx,srcy)
    np.savetxt(os.path.join(os.getcwd(),output_folder,"syndata_source_%d_station_1.txt" %(counter)),
               np.column_stack((t,signal1_tdomain)),
               header=header,fmt='%.2f\t%.5f')
    
    header="Station x,y: %.5f %.5f\nSource x,y: %.5f %.5f\ntime[s]\tsignal\n" %(statx[j],staty[j],srcx,srcy)
    np.savetxt(os.path.join(os.getcwd(),output_folder,"syndata_source_%d_station_2.txt" %(counter)),
               np.column_stack((t,signal2_tdomain)),
               header=header,fmt='%.2f\t%.5f')
        
    

#%%
# let's do the correlation of all the synthetic seismograms and stack them

# create an axis of shift times for the correlation functions
tshift_axis = np.append(t[1:int(len(t)/2)+1][::-1]*-1,t[0:int(len(t)/2)])


fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(111)

stack = np.zeros(len(w),dtype='complex')
for sourceazimuth in signal_collection:
    signal = signal_collection[sourceazimuth]
    if whitening:
        sig1 = signal[0]/np.abs(signal[0])
        sig2 = signal[1]/np.abs(signal[0])
    else:
        sig1 = signal[0]
        sig2 = signal[1]
    # the signal is in the frequency domain. So a a correlation is a multiplication with the complex conjugate
    corr = np.conj(sig1) * sig2
    stack += corr 
    
    corr_tdomain = np.fft.irfft(corr)
    corr_tdomain = np.fft.fftshift(corr_tdomain)
    corr_tdomain /= np.max(np.abs(corr_tdomain))
    ax.plot(tshift_axis,6*corr_tdomain+sourceazimuth/np.pi*180.,'k',linewidth=0.1)
    
stack /= len(signal_collection)


stack_tdomain = np.fft.irfft(stack)
stack_tdomain /= np.max(np.abs(stack_tdomain))
# shift the time domain stacked cross correlations to reverse the effects of the FFT shift
stack_tdomain = np.fft.fftshift(stack_tdomain)


ax.plot([],[],'k',linewidth=0.1,label='individual correlations sorted according to source azimuth')
ax.plot(tshift_axis,30*stack_tdomain-190,label='sum of all cross correlations')
ax.set_ylabel('Source azimuth')
ax.set_xlabel('time shift [s]')
ax.legend(loc='upper right')
plt.savefig("synthetic_example_stacking_correlations.png",bbox_inches='tight')


#%%
# now do a comparison with the direct Green's function between the two stations
# this should be the equal to the stacked ambient noise cross-correlations
    
r = np.sqrt((statx[0]-statx[1])**2+(staty[0]-staty[1])**2)
sourcefreq = source(t,sourcetype='spike')
if dispersion_curve:
    greens,signal=G2Dconvolve_freq(w,dispcurve,r,sourcefreq,farfield=False,noise=False) 
else:
    greens,signal=G2Dconvolve_freq(w,interstation_mean,r,sourcefreq,farfield=False,noise=False) 
# In the 'spike' case, the Green's function and the signal are identical

greensfunction_stat1_stat2 = np.fft.irfft(signal)
greensfunction_stat1_stat2 /= np.max(np.abs(greensfunction_stat1_stat2))

plt.figure()
plt.plot(t,greensfunction_stat1_stat2,label='Greens function between station 1 and 2')
plt.xlabel('time [s]')
plt.legend(loc='upper right')
plt.savefig("synthetic_example_greens_function.png",bbox_inches='tight')
plt.show()

# to get the Green's function we have to take the time derivative of the stacked cross correlation
stack_tdomain = np.fft.irfft(stack)
# gradient calculation has to be done before the FFTshift otherwise everything is shifted by one sample
stack_tdomain = np.gradient(stack_tdomain)
stack_tdomain = np.fft.fftshift(stack_tdomain)
stack_tdomain /= np.max(np.abs(stack_tdomain))

# The stacked cross correlation corresponds to G2D(-t) [time reversed Green's function]
# minus the original Green's function G2D(t) 
# (see for example Boschi and Weemstra 'Stationary-phase integrals in the cross
# correlation of ambient noise', 2015, Eq(70).)
greens_symmetric = greensfunction_stat1_stat2*-1 + np.roll(greensfunction_stat1_stat2,-1)[::-1]
#         original Green's function times minus 1    time reversed Green's function

# shift also the Green's function so it is analogue to the cross correlation plots
greens_symmetric = np.fft.fftshift(greens_symmetric)

plt.figure(figsize=(10,7))
plt.plot(tshift_axis,stack_tdomain,label='d(CCstacked)/dt')
plt.plot(tshift_axis,greens_symmetric,linestyle='dotted',label='G2D(-t) - G2D(t)')
plt.legend(loc='upper right')
plt.title("Comparison of time derivative of stacked cross correlation to Green's function")
plt.xlabel("time shift [s]")
plt.savefig("synthetic_example_stacked_cc_vs_greens_function.png",bbox_inches='tight')
plt.show()