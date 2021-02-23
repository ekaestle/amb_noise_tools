# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 14:07:02 2014

@author: emanuel

"""

################

crosscorrelations_folder = "./cross_correlation_spectra/ZZ/*"

output_folder = "picked_dispersioncurves"


# parameters
freqmin = 0.0 # in Hz, for the picking procedure
freqmax = 1.0 # in Hz, for the picking procedure

min_vel = 0.5 # used for plotting and for the velocity_filter/smoothing
max_vel = 4.5 # used for the velocity_filter

# smooth spectrum fits a smooth spline to the cross correlation spectrum (i.e. in the freq domain)
# the smoothness depends on min_vel.
smooth_spectrum=True

# velocity filter removes all signals at velocities slower than min_vel and
# faster than max_vel
velocity_filter=False

horizontal_polarization=False # only True for RR or TT correlations

# create figures
create_timedomain_figures = True
figure_folder = "crosscorrelation_figures"

#############################

import numpy as np
import pickle, os
import noise
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
# implement the default mpl key bindings
#from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from obspy.signal.invsim import cosine_taper
from obspy.signal.filter import lowpass
import sys, glob
if sys.version_info[0] < 3:
    import Tkinter as Tk
else:
    import tkinter as Tk

print("This program processes crossing files and lets the user pick a dispersion curve")

# Get a list of all files
files = glob.glob(crosscorrelations_folder)

if not os.path.exists(output_folder):
    os.makedirs(output_folder)


#%%
# create figures
    
if create_timedomain_figures: 

    if not os.path.exists(figure_folder):
        os.makedirs(figure_folder)
        
    for fname in files:
        basename = os.path.basename(fname)
        figurepath = os.path.join(figure_folder,basename+"_figure.png")
        
        #if os.path.isfile(figurepath):
        #    print(figurepath,"already exists, skipping.")
        #    continue
        
        with open(fname,"rb") as f:
            filedict = pickle.load(f)        
        spectrum = filedict['spectrum']
        freq = filedict['freq']
        dist = filedict['dist']
        dt = 1./freq[-1]/2.
                
        TCORR = np.fft.irfft(np.real(spectrum))   
        TCORR_prefilt = TCORR.copy()     
        
        idx_tmin = int((dist/max_vel)/dt*0.95) # 5percent extra for taper
        idx_tmax = int((dist/min_vel)/dt*1.05) # 5% extra for taper
        vel_filt_window = cosine_taper(idx_tmax-idx_tmin,p=0.1)
        win_samples=int((len(spectrum)-1)*2)
        vel_filt = np.zeros(win_samples)
        vel_filt[idx_tmin:idx_tmax] = vel_filt_window
        vel_filt[-idx_tmax:-idx_tmin] = vel_filt_window
        TCORR_velfilt = TCORR * vel_filt
        CORR_velfilt = np.fft.rfft(TCORR_velfilt)

        smoothspec,crossings = noise.get_zero_crossings(freq,spectrum, dist, freqmin=freqmin,
                   freqmax=999., min_vel=min_vel, max_vel=max_vel,
                   horizontal_polarization=horizontal_polarization,
                   smooth_spectrum=smooth_spectrum,return_smoothed_spectrum=True)   
        freq_smoothed = smoothspec[:,0]
        dt_smoothed = 1./freq_smoothed[-1]/2.
        CORR_smoothed = smoothspec[:,1]
        TCORR_smoothed = np.fft.irfft(CORR_smoothed)
             
        plt.ioff()
        fig = plt.figure(figsize=(10,9))
        ax1 = fig.add_subplot(211)
        ax1.plot(filedict['freq'],np.real(filedict['spectrum']),'k',linewidth=0.5,label='original cc spectrum (real part)')
        ax1.plot(freq_smoothed,CORR_smoothed,'g',linewidth=1.5,label='after spline smoothing')
        ax1.plot(freq,np.real(CORR_velfilt),'r--',linewidth=1.5,label='after velocity filtering')
        ax1.set_xlim(freqmin,freqmax)
        ax1.set_xlabel("frequency [Hz]")
        ax1.legend(loc='upper right')
        ax2 = fig.add_subplot(212)
        tcorr = np.fft.fftshift(TCORR_prefilt)
        ax2.plot((np.arange(len(TCORR_prefilt))-len(TCORR_prefilt)/2.)*dt,
                 tcorr/np.max(tcorr),'k',linewidth=0.5,label='original cross correlation') 
        tcorr = np.fft.fftshift(TCORR_velfilt)
        ax2.plot((np.arange(len(TCORR_velfilt))-len(TCORR_velfilt)/2.)*dt,
                  tcorr/np.max(tcorr),'r--',linewidth=0.8,label='after velocity filter (symmetrized)')
        tcorr = np.fft.fftshift(TCORR_smoothed)
        ax2.plot((np.arange(len(TCORR_smoothed))-len(TCORR_smoothed)/2.)*dt_smoothed,
                  tcorr/np.max(tcorr),'g',linewidth=0.8,label='after spline smooting (symmetrized)')
        #TCORR_lowpassed =  np.fft.fftshift(lowpass(TCORR_velfilt,3,freq[-1]*2,zerophase=True))
        #ax2.plot((np.arange(len(TCORR_prefilt))-len(TCORR_prefilt)/2.)*dt,
        #          TCORR_lowpassed/np.max(TCORR_lowpassed),'b',linewidth=1.5,label='after velocity filter, lowpassed 3Hz')
        ax2.set_xlim(-dist/min_vel*2.,dist/min_vel*2.)
        ax2.legend(loc='upper right')       
        ax2.set_xlabel("time [s]")
        plt.savefig(figurepath,bbox_inches='tight')
        #plt.show()
        plt.close(fig)    
    

    
#%%##########################
# INTERACTIVE PICKING
    
          
def onclick(event):
    if event.button == 1:
        return True
    valx = event.xdata
    valy = event.ydata
    ind = np.argmin(100*(x-valx)**2 + (y-valy)**2)
    pick = (x[ind],y[ind])
    if pick in chosen:
        a.plot(x[ind],y[ind],'bo')
        canvas.draw()
        chosen.remove(pick)
    else:
        a.plot(x[ind],y[ind],'ro')
        canvas.draw()
        chosen.append(pick)
    return True
    
    
def _next():
    root.quit()     # stops mainloop
    root.destroy()  # this is necessary on Windows to prevent
                    # Fatal Python Error: PyEval_RestoreThread: NULL tstate
def _done():
    root.quit()
    root.destroy()
    global done
    done = True
  

refcurve = np.loadtxt("Average_phase_velocity_rayleigh")
  
dictionary= {}
disp_curves = {}
done = False
for fname in files:
    basename = os.path.basename(fname)
    outfilename = os.path.join(output_folder,basename+"_dispersioncurve.txt")
    
    #if os.path.isfile(outfilename):
        #print(outfilename,"already exists, skipping.")
        #continue
    
    with open(fname,"rb") as f:
        filedict = pickle.load(f)        
    spectrum = filedict['spectrum']
    freq = filedict['freq']
    dist = filedict['dist']
    dt = 1./freq[-1]/2.
    
    idx_tmin = int((dist/max_vel)/dt*0.95) # 5percent extra for taper
    idx_tmax = int((dist/min_vel)/dt*1.05) # 5% extra for taper
    vel_filt_window = cosine_taper(idx_tmax-idx_tmin,p=0.1)
    win_samples=int((len(spectrum)-1)*2)
    vel_filt = np.zeros(win_samples)
    vel_filt[idx_tmin:idx_tmax] = vel_filt_window
    vel_filt[-idx_tmax:-idx_tmin] = vel_filt_window

    TCORR = np.fft.irfft(spectrum)   
    TCORR_prefilt = TCORR.copy()     
    overall_snr = np.mean(np.mean(np.abs(TCORR[idx_tmin:idx_tmax]))/np.mean(np.abs(np.append(TCORR[0:idx_tmin],TCORR[idx_tmax:int(len(TCORR)/2)]))))
    
    if velocity_filter:
        TCORR *= vel_filt
        CORR = np.fft.rfft(TCORR)
    else:
        CORR = spectrum
        
    if velocity_filter:
        smoothspec = False
    elif smooth_spectrum:
        smoothspec = True
    else:
        smoothspec = False
        
    crossings = noise.get_zero_crossings(freq,CORR, dist, freqmin=freqmin,
                       freqmax=freqmax, min_vel=min_vel, max_vel=max_vel,
                       horizontal_polarization=horizontal_polarization,
                       smooth_spectrum=smoothspec)    
    
    
    chosen=[]
    #x,y = np.random.random((2,20))
    #plt.plot(x,y,'o',picker=5)
    root = Tk.Tk()
    root.wm_title("Embedding in TK")

    f = Figure(figsize=(11,7))
    a = f.add_subplot(111)
    a.plot(refcurve[:,0],refcurve[:,1],'r-')
    x = crossings[:,0]
    y = crossings[:,1]
    a.plot(x,y,'bo',ms=8)
    a.set_title(fname.split("/")[-1]+"\n Picking with middle or right mouse "
    "button\n pick as many zero crossings as you like (or none)\n"
    "continue to the next station pair by clicking 'next' or end picking by "
    "clicking 'done'")
    canvas = FigureCanvasTkAgg(f, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
    
    toolbar = NavigationToolbar2Tk( f.canvas, root )
    toolbar.update()
    canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
    
    canvas.mpl_connect('button_press_event', onclick)
    button2 = Tk.Button(master=root, text='Done', command=_done)
    button2.pack(side=Tk.LEFT)      
    button = Tk.Button(master=root, text='Next', command=_next)
    button.pack(side=Tk.BOTTOM)
    
    Tk.mainloop()
    
    #outfiles.append(outname)

    if len(chosen) > 1:
        dispcurve = np.array(chosen)
        dispcurve[:,0] = 1./dispcurve[:,0]
        np.savetxt(outfilename,dispcurve,fmt="%.4f %.4f")

        disp_curves[outfilename] = dispcurve

    for pick in chosen:
        try:
            dictionary[str(pick[0])].append(pick[1])
        except:
            dictionary[str(pick[0])]=[]
            dictionary[str(pick[0])].append(pick[1])
            
    if done:
        break

#%%##############################################
# Calculate average dispersion curve from picks
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

if len(disp_curves)==0:
    print("no new correlations found to be processed!")

else:        
    f=np.array([])
    avg=np.array([])
    sterr=np.array([])
    for frequency in dictionary:
        f=np.append(f,float(frequency))
        avg=np.append(avg,np.mean(dictionary[frequency]))
        sterr=np.append(sterr,np.std(dictionary[frequency]))
    
    avg=avg[f.argsort()]
    sterr=sterr[f.argsort()]
    f=f[f.argsort()]
        
    minf=f[0]
    maxf=f[-1]
    func=interp1d(f,avg)
    f2=np.arange(int((maxf-minf)/0.002))*0.002+(int(minf*1000)+1)/1000.
    avg_ip=func(f2)
    
    if len(avg_ip) > 91:
        smooth_curve1 = savgol_filter(avg_ip, 91, 3)
    else:
        smooth_curve1 = avg_ip
    a,b,c,d,e = np.polyfit(f,avg,4)
    smooth_curve = a*f2**4 + b*f2**3 + c*f2**2 + d*f2 + e
    
    smooth_curve[f2<0.4] = smooth_curve1[f2<0.4]
    if len(smooth_curve) > 91:
        smooth_curve = savgol_filter(smooth_curve, 91, 3)
    
    plt.figure()
    for statpair in disp_curves:
        disp_curve = disp_curves[statpair]
        plt.plot(1./disp_curve[:,0],disp_curve[:,1],linewidth=0.5,color='grey')
    plt.errorbar(f,avg,sterr,marker='o',label='picks')
    plt.plot(f2,smooth_curve,marker='x',lw=2,label='smoothed average phase velocity curve')
    plt.title("Average phase velocity curve")
    plt.xlabel("Frequency")
    plt.ylabel("Velocity")
    plt.legend(numpoints=1)
    
    np.savetxt(os.path.join(output_folder,"Average_phase_velocity_picked.txt"),np.column_stack((f2,smooth_curve)),fmt=' %.6f    %.2f')
    plt.show()
