# -*- coding: utf-8 -*-
"""
Updated June 2021
- script makes sure now that the horizontal traces are sampled at the same points in time
  if there is a large subsample time shift. Before, this would result in an error.
- can read station xml files to get station information
- better error handling
- fixed memory leak in the horizontal component correlations



@author: emanuel
"""

# creates cross-correlation files named
#stat1_X_stat2_dist_distance_st_corrdays_ovlap_overlap

# The output files are python dictionaries that contain the cross correlations
# and some meta information (stat names, lat, lon, number of correlationdays, ...)

""" USER DEFINED PARAMETERS"""

# Path to the dataset folder (sac, mseed, etc. files). Files need not to be 
# in special format, but daily files are recommended (anything should 
# theoretically work). Searches through all subfolders
path='./preprocessed_data'

# only files with these file endings are being read
# (capitalization is not important)
formats = ['mseed','SAC','sync'] # for example: ['mseed','SAC'] or leave empty []

# station list (file is created if not existing yet)
statfilepath = './statlist.txt' # 3 columns: station_id latitude longitude

# OPTIONAL: folder where the station inventory files are stored (to get station location information)
# if there are no xml files, the lat/lon information has to be provided via the statfile
# or in the headers of the sac input files
inventory_directory = "./station_inventory" # xml inventory files

# path where the cross correlation spectra should be saved
# new data will be added to existing *.pkl files in that folder
spectra_path='cross_correlation_spectra'

# filename of sqlite database (created if not yet existing)
# this database lists all existing files, components, available timeranges, etc.
database_file = 'database_ambnoise.sqlite'

# check if there are new files in the path. The sqlite database is then updated
update_database = True # recommended to be True, can take long for many files

# check if there are stations missing in the the 'statfilepath' file.
# missing information is added from the station xml metadata if available
update_statlist = True # recommended to be True if working with xml inventory files

# traces are cut into windows. windowed data is then correlated
# ideal length depends on your typical station distances and if you're interested in the coda
window_length=3600. # in seconds

# overlap of subsequent windows (recommended between 0.3 and 0.6)
overlap = 0.5

# minimum allowed inter-station distance in km
min_distance = 5. 

# whiten spectra prior to cross correlation (see Bensen et al. 2007)
whiten = True

# use onebit normalization (see Bensen et al. 2007)
onebit = False

# if empty, all available years are processed. If you only one a specific year
# to be processed, set years = [2004,2020]
years = [] 

# list of components to be processed, e.g, ['ZZ','TT']
comp_correlations = ['ZZ','RR','TT']

save_monthly = False # additionally save monthly cross correlations

# can be a list with file IDs ['GU.CANO','CH.SIMPL','CH.ZUR'], otherwise put None
only_process_these_stations = None

# if necessary, see also other parameters for function noise.noisecorr below.
""" END OF USER DEFINED PARAMETERS"""
from mpi4py import MPI
import numpy as np
from obspy import read, Stream, UTCDateTime, read_inventory
from obspy.geodetics.base import gps2dist_azimuth
from itertools import combinations
import os, glob, datetime, pickle
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import noise
import sqlite3


#%%      
def process_noise(stream,pair,corrcomp,window_length,overlap,year,julday,flog):
    global spectra_path
    global statdict
    global save_monthly
    global pairdict

    #print(datetime.datetime.now(),">>>>>> Processing pair",stat1,stat2,file=flog) 
    
    stat1 = pair[0]
    stat2 = pair[1]
    net1,sta1 = stat1.split(".")
    net2,sta2 = stat2.split(".")
    
    if 'ZZ' in corrcomp:

        stream1 = stream.select(network=net1,station=sta1,component='Z')
        stream2 = stream.select(network=net2,station=sta2,component='Z')
        
        if len(stream1) == 0 or len(stream2) == 0:
            return None
            
        stream1,stream2 = noise.adapt_timespan(stream1, stream2,
                                               min_overlap=window_length)
        if len(stream1) != len(stream2):
            raise Exception("this should be impossible")
            
        if len(stream1)==0 or len(stream2)==0:
            return None        
    
        corr_list = []
        no_windows = []
        
        for idx in range(len(stream1)):
            
            tstart = stream1[idx].stats.starttime
            tend = stream1[idx].stats.endtime

            st1 = stream1.slice(starttime=tstart,endtime=tend)
            st2 = stream2.slice(starttime=tstart,endtime=tend)
            # for the unlikely case that we have several overlapping traces
            # for the same station
            if len(st1)!=1 or len(st2)!=1:
                st1._cleanup()
                st2._cleanup()
                if len(st1)>1:
                    # just take the longer trace
                    st1 = Stream(st1[np.array(list(map(len,st1))).argmax()])
                if len(st2)>1:
                    st2 = Stream(st2[np.array(list(map(len,st2))).argmax()])
                
            if (st1[0].stats.starttime != st2[0].stats.starttime or
                st1[0].stats.endtime != st2[0].stats.endtime or
                st1[0].stats.endtime - st2[0].stats.starttime < window_length):
                raise Exception("this should be impossible")
                
            if np.std(st1[0].data) == 0. or np.std(st2[0].data) == 0. :
                print("data all zero",file=flog)
                print(st1,file=flog)
                print(st2,file=flog)
                continue
                       
            if (np.isnan(st1[0].data).any() or np.isnan(st2[0].data).any() or
                np.isinf(st1[0].data).any() or np.isinf(st2[0].data).any() ):
                print("nan/inf in data",file=flog)
                print(st1,file=flog)
                print(st2,file=flog)
                continue
            
            #try:
            freq,spec,wincount = noise.noisecorr(st1[0],st2[0],window_length,overlap,
                                                  whiten=whiten,onebit=onebit)
            corr_list.append(spec)
            no_windows.append(wincount)
            #except:
            #    continue
        
        if len(corr_list) == 0:
            return None
        
        corr_spectrum = np.average(np.array(corr_list),axis=0,weights=no_windows)
        
        filepath = getfilepath(stat1,stat2,"ZZ",pairdict[pair]['dist'],overlap)
        
        if os.path.isfile(filepath):
            with open(filepath,"rb") as f:
                corr_dict = pickle.load(f)
        
            if (year,julday) in corr_dict['corrdays']:
                print("correlation day already in database!",filepath,year,julday, "skipping")          
            else:
                if len(corr_spectrum)!=len(corr_dict['spectrum']):
                    print("Warning: correlation trace has not the same length "+
                          "as the trace in the database. Maybe different "+
                          "sampling rates? Skipping.",net1,sta1,net2,sta2)
                    return None
                corr_dict['corrdays'].append((year,julday))
                corr_dict['spectrum'] = np.average([corr_spectrum,
                                                    corr_dict['spectrum']],
                                                   axis=0,
                                                   weights=[np.sum(no_windows),
                                                            corr_dict['no_wins']])
                corr_dict['no_wins'] += np.sum(no_windows)

        else:
            
            corr_dict = {}
            corr_dict['corrdays'] = [(year,julday)]
            corr_dict['spectrum'] = corr_spectrum
            corr_dict['freq'] = freq
            corr_dict['no_wins'] = np.sum(no_windows)
            corr_dict['dist'] = pairdict[pair]['dist']
            corr_dict['az'] = pairdict[pair]['az']
            corr_dict['baz'] = pairdict[pair]['baz']
            corr_dict['component'] = 'ZZ'
            corr_dict['station1'] = statdict[stat1]
            corr_dict['station2'] = statdict[stat2]
            corr_dict['station1']['id'] = stat1
            corr_dict['station2']['id'] = stat2
        
        if save_monthly:
            
            month = str(year)+"."+str(UTCDateTime(year=year,julday=day).month)
            
            if not 'spectrum.'+month in corr_dict.keys():
                
                corr_dict['spectrum.'+month] = corr_spectrum
                corr_dict['no_wins.'+month] = np.sum(no_windows)
                
            else:
            
                corr_dict['spectrum.'+month] = np.average([corr_spectrum,
                                            corr_dict['spectrum.'+month]],
                                            axis=0,
                                            weights=[np.sum(no_windows),
                                                     corr_dict['no_wins.'+month]])
                corr_dict['no_wins.'+month] += np.sum(no_windows)

        with open(filepath,"wb") as f:
            pickle.dump(corr_dict,f)
               
#        filename = stat1+"_X_"+stat2+"_ZZ_dist_%08d_wins_%02d_ovlap_%.2f"\
#                %(int(dist*1000.),np.sum(no_windows),overlap)
#        fpath = os.path.join(spectra_path,"ZZ",str(year)+"."+str(julday),filename)
#        np.save(fpath,np.column_stack((freq,\
#                        np.real(corr_spectrum),np.imag(corr_spectrum))))  
        
        #print("successfully correlated",stat1,stat2,"comp:",corrcomp,"day:",year,julday)
        
 

    if 'TT' in corrcomp or 'RR' in corrcomp:          
        
        stream1n = stream.select(network=net1,station=sta1,component='N')
        stream1e = stream.select(network=net1,station=sta1,component='E')        
        stream2n = stream.select(network=net2,station=sta2,component='N')
        stream2e = stream.select(network=net2,station=sta2,component='E')
        
        if (len(stream1n) == 0 or len(stream1e) == 0 or 
            len(stream2n) == 0 or len(stream2e) == 0):
            return None      
        
        stream1n,stream1e = noise.adapt_timespan(stream1n, stream1e,
                                                 min_overlap=window_length)
        stream2n,stream2e = noise.adapt_timespan(stream2n, stream2e,
                                                 min_overlap=window_length)
        
        if (len(stream1n) == 0 or len(stream1e) == 0 or 
            len(stream2n) == 0 or len(stream2e) == 0):
            return None 
        
        stream1n,stream2n = noise.adapt_timespan(stream1n,stream2n,
                                                 min_overlap=window_length)
        stream1e,stream2e = noise.adapt_timespan(stream1e,stream2e,
                                                 min_overlap=window_length)

        if (len(stream1n) == 0 or len(stream1e) == 0 or 
            len(stream2n) == 0 or len(stream2e) == 0):
            return None
        
        corr_list_rr = []
        corr_list_tt = []
        no_windows = []

        for idx in range(len(stream1n)):
            tstart = stream1n[idx].stats.starttime
            tend = stream1n[idx].stats.endtime
            st1 = (stream1n+stream1e).slice(starttime=tstart,endtime=tend)
            st2 = (stream2n+stream2e).slice(starttime=tstart,endtime=tend)
            if len(st1)!=2 or len(st2)!=2:
                st1._cleanup()
                st2._cleanup()
            if len(st1)!=2:
                for tr in st1:
                    if tr.stats.endtime-tr.stats.starttime < window_length:
                        st1.remove(tr)
            if len(st2)!=2:
                for tr in st2:
                    if tr.stats.endtime-tr.stats.starttime < window_length:
                        st2.remove(tr)
            if len(st1)!=2 or len(st2)!=2:
                print("error for streams:",file=flog)
                print(st1,file=flog)
                print(st2,file=flog)
                print("number of traces in stream is not okay",file=flog)
                raise Exception()
                continue
            
            # check that traces are not all zero
            if (np.std(st1[0].data) == 0. or np.std(st1[1].data) == 0. or
                np.std(st2[0].data) == 0. or np.std(st2[1].data) == 0. ):
                print("data all zero",file=flog)
                print(st1,file=flog)
                print(st2,file=flog)
                continue
            
            # check that the time span is really the same
            if (st1[0].stats.starttime != st1[1].stats.starttime or
                st1[0].stats.starttime != st2[0].stats.starttime or
                st1[0].stats.starttime != st2[1].stats.starttime or
                st1[0].stats.endtime != st1[1].stats.endtime or
                st1[0].stats.endtime != st2[0].stats.endtime or
                st1[0].stats.endtime != st2[1].stats.endtime or
                st1[0].stats.endtime - st1[0].stats.starttime < window_length):
                raise Exception("this should not be possible!")
              
            # check for nan/inf values
            if (np.isnan(st1[0].data).any() or np.isnan(st1[1].data).any() or
                np.isnan(st2[0].data).any() or np.isnan(st2[1].data).any() or
                np.isinf(st1[0].data).any() or np.isinf(st1[1].data).any() or
                np.isinf(st2[0].data).any() or np.isinf(st2[1].data).any() ):
                print("nan/inf in data",file=flog)
                print(st1,file=flog)
                print(st2,file=flog)
                continue
            # az = azimuth from station1 -> station2
            # baz = azimuth from station2 -> station1
            # for stream2 the back azimuth points in direction of station1
            # for stream1 the azimuth points in direction of station2
            # BUT 180. degree shift is needed so that the radial components point in the same direction!
            # otherwise they point towards each other => transverse comp would be also opposed
            try:
                st1.rotate('NE->RT',back_azimuth=(pairdict[pair]['az']+180.)%360.)
            except:
                print("Error rotating stream",file=flog)
                print(st1,file=flog)
                raise Exception("Error rotating stream")
                continue
            try:
               st2.rotate('NE->RT',back_azimuth=pairdict[pair]['baz'])
            except:
                print("Error rotating stream",file=flog)
                print(st2,file=flog)
                raise Exception("Error rotating stream")
                continue
                    
            #try:
            if 'RR' in corrcomp:
                freq,spec_rr,wincount = noise.noisecorr(st1.select(component='R')[0],
                                               st2.select(component='R')[0],
                                               window_length,overlap,
                                               whiten=whiten,onebit=onebit)
            if 'TT' in corrcomp:
                freq,spec_tt,wincount = noise.noisecorr(st1.select(component='T')[0],
                                               st2.select(component='T')[0],
                                               window_length,overlap,
                                               whiten=whiten,onebit=onebit)
            #except:
            #    continue


            if 'RR' in comp_correlations:
                corr_list_rr.append(spec_rr)
            if 'TT' in comp_correlations:
                corr_list_tt.append(spec_tt)
            no_windows.append(wincount)



        if len(corr_list_rr) == 0 and len(corr_list_tt) == 0:
            return None

        if 'RR' in corrcomp:

            corr_spectrum = np.average(np.array(corr_list_rr),axis=0,weights=no_windows)
                    
            filepath = getfilepath(stat1,stat2,"RR",pairdict[pair]['dist'],overlap)
            
            if os.path.isfile(filepath):
                with open(filepath,"rb") as f:
                    corr_dict = pickle.load(f)
            
                if (year,julday) in corr_dict['corrdays']:
                    print("correlation day already in database!",filepath,year,julday)
                
                else:                  
                    corr_dict['corrdays'].append((year,julday))
                    corr_dict['spectrum'] = np.average([corr_spectrum,
                                                    corr_dict['spectrum']],
                                                    axis=0,
                                                    weights=[np.sum(no_windows),
                                                             corr_dict['no_wins']])
                    corr_dict['no_wins'] += np.sum(no_windows)
    
            else:
                
                corr_dict = {}                
                corr_dict['corrdays'] = [(year,julday)]
                corr_dict['spectrum'] = corr_spectrum
                corr_dict['freq'] = freq
                corr_dict['no_wins'] = np.sum(no_windows)
                corr_dict['dist'] = pairdict[pair]['dist']
                corr_dict['az'] = pairdict[pair]['az']
                corr_dict['baz'] = pairdict[pair]['baz']
                corr_dict['component'] = 'RR'
                corr_dict['station1'] = statdict[stat1]
                corr_dict['station2'] = statdict[stat2]
                corr_dict['station1']['id'] = stat1
                corr_dict['station2']['id'] = stat2
            
            if save_monthly and not ((year,julday) in corr_dict['corrdays']):
                
                month = str(year)+"."+str(UTCDateTime(year=year,julday=day).month)
                
                if not 'spectrum.'+month in corr_dict.keys():
                    
                    corr_dict['spectrum.'+month] = corr_spectrum
                    corr_dict['no_wins.'+month] = np.sum(no_windows)
                    
                else:
                
                    corr_dict['spectrum.'+month] = np.average([corr_spectrum,
                                                corr_dict['spectrum.'+month]],
                                                axis=0,
                                                weights=[np.sum(no_windows),
                                                         corr_dict['no_wins.'+month]])
                    corr_dict['no_wins.'+month] += np.sum(no_windows)
    
            with open(filepath,"wb") as f:
                pickle.dump(corr_dict,f)


        if 'TT' in corrcomp:

            corr_spectrum = np.average(np.array(corr_list_tt),axis=0,weights=no_windows)
            
            filepath = getfilepath(stat1,stat2,"TT",pairdict[pair]['dist'],overlap)
                            
            if os.path.isfile(filepath):
                with open(filepath,"rb") as f:
                    corr_dict = pickle.load(f)
            
                if (year,julday) in corr_dict['corrdays']:
                    print("correlation day already in database!",filepath,year,julday)
                
                else:
                    corr_dict['corrdays'].append((year,julday))
                    corr_dict['spectrum'] = np.average([corr_spectrum,
                                                    corr_dict['spectrum']],
                                                    axis=0,
                                                    weights=[np.sum(no_windows),
                                                             corr_dict['no_wins']])
                    corr_dict['no_wins'] += np.sum(no_windows)
    
            else:
                
                corr_dict = {}                
                corr_dict['corrdays'] = [(year,julday)]
                corr_dict['spectrum'] = corr_spectrum
                corr_dict['freq'] = freq
                corr_dict['no_wins'] = np.sum(no_windows)
                corr_dict['dist'] = pairdict[pair]['dist']
                corr_dict['az'] = pairdict[pair]['az']
                corr_dict['baz'] = pairdict[pair]['baz']
                corr_dict['component'] = 'TT'
                corr_dict['station1'] = statdict[stat1]
                corr_dict['station2'] = statdict[stat2]
                corr_dict['station1']['id'] = stat1
                corr_dict['station2']['id'] = stat2
                
            
            if save_monthly and not ((year,julday) in corr_dict['corrdays']):
                
                month = str(year)+"."+str(UTCDateTime(year=year,julday=day).month)
                
                if not 'spectrum.'+month in corr_dict.keys():
                    
                    corr_dict['spectrum.'+month] = corr_spectrum
                    corr_dict['no_wins.'+month] = np.sum(no_windows)
                    
                else:
                
                    corr_dict['spectrum.'+month] = np.average([corr_spectrum,
                                                corr_dict['spectrum.'+month]],
                                                axis=0,
                                                weights=[np.sum(no_windows),
                                                         corr_dict['no_wins.'+month]])
                    corr_dict['no_wins.'+month] += np.sum(no_windows)
    
            with open(filepath,"wb") as f:
                pickle.dump(corr_dict,f)  
            
        #print("successfully correlated",stat1,stat2,"comp:",corrcomp,"day:",year,julday)
        
    
    return
##############################################################################
"""
##############################################################################
"""
#%%
    
def getfilepath(stat1,stat2,corr_comp,dist,overlap):
    global statdict
    
    if stat1[0] == ".":
        stat1 = stat1[1:]
    if stat2[0] == ".":
        stat2 = stat2[1:]
    
    filename = stat1+"_X_"+stat2+"_"+corr_comp+"_dist_%.2f_ovlap_%.2f.pkl"\
                %(dist,overlap)
    
    filepath = os.path.join(spectra_path,corr_comp,filename)
    
    return filepath
        

    
def get_julday_filelist(year,julday,comp,staids,window_length):
    
    filelist = []

    try:
        starttime = abs(UTCDateTime(year=year,julday=day))
    except:
        print("could not convert startdate")
        return filelist
    endtime = starttime+24*60*60
    
    for staid in data_dic:
        if not staid in staids:
            continue
        if len(data_dic[staid][comp]['windows']) == 0:
            continue
        timematch = np.where((data_dic[staid][comp]['windows'][:,0]+window_length/2. < endtime)*
                             (data_dic[staid][comp]['windows'][:,1]-window_length/2. > starttime))[0]
        for idx in timematch:
            filelist.append(data_dic[staid][comp]['paths'][idx])
            
    return filelist


def downsample_stream(st,sampling_frequency):
    st.detrend(type='linear')
    st.detrend(type='demean')
    st.filter("lowpass",freq = 0.4*sampling_frequency,zerophase=True)
    st.decimate(int(st[0].stats.sampling_rate/sampling_frequency),no_filter=True)
    return    

#%%
if __name__ == "__main__":
    # Initialize MPI
    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()

    if mpi_rank == 0:
        if not os.path.exists(spectra_path):
            os.makedirs(spectra_path)
            for ccorr in comp_correlations:
                os.mkdir(os.path.join(spectra_path,ccorr))
        else:
            for ccorr in comp_correlations:
                if not os.path.exists(os.path.join(spectra_path,ccorr)):
                    os.mkdir(os.path.join(spectra_path,ccorr))
    
    # all processes should wait until the folder is created
    mpi_comm.Barrier()   

    logfile = os.path.join(spectra_path,"log_rank%d.txt" %mpi_rank)
    if os.path.isfile(logfile):
        flog = open(logfile,'a')
    else:
        flog = open(logfile,'w')
    print("\n Starting processing\n ",datetime.datetime.now(),"\n------------\n",file=flog,flush=True)

    
    #%%#######################################################################
    # UPDATING THE DATABASE
    
    formats_all = []
    for fileformat in formats:
        if not fileformat in formats_all:
            formats_all.append(fileformat)
        if not fileformat.upper() in formats_all:
            formats_all.append(fileformat.upper())
        if not fileformat.lower() in formats_all:
            formats_all.append(fileformat.lower())
    formats = formats_all
        
    
    if update_database and mpi_rank==0:
        print("checking and updating database...",flush=True)
        conn = sqlite3.connect(database_file)
        c = conn.cursor()
        try:
            c.execute("""CREATE TABLE IF NOT EXISTS file_db (
                        staid,
                        component text,
                        starttime text,
                        endtime text,
                        path text,
                        PRIMARY KEY(staid,component,starttime));""")
            conn.commit()
        except:
            pass
        # update database
        #check if all paths still exist
        c.execute("SELECT path FROM file_db")
        paths = c.fetchall()
        pathlist = [i[0] for i in paths]
        for fpath in pathlist:
            if not os.path.isfile(fpath):
                fpath_alt = fpath.replace("emanuelk","emanuel")
                if os.path.isfile(fpath_alt):
                    c.execute("UPDATE file_db SET path=? WHERE path=?", (fpath_alt,fpath,))
                    print("updating",fpath_alt)
                else:
                    print("error",fpath)
                    print("deleting from database")
                    c.execute("DELETE FROM file_db WHERE path=?", (fpath,))
        conn.commit()
        # check for new paths
        count = 0
        pathlist_new = [os.path.join(dir_,f) for dir_,_,files in os.walk(path) for f in files]
        new_paths = set(pathlist_new) - set(pathlist) #elements that are uniquely in pathlist_new
        print(len(new_paths),"new paths found",flush=True)
        #new_paths = set(pathlist).symmetric_difference(set(pathlist_new))
        print("adding new entries (for each file the header is being read, this may take a while)...")
        #errorfile = open("read_errors_database.txt","w")
        #print("check for errors in",errorfile)
        for filepath in new_paths:
            #print "adding new entry:",filepath
            fname = os.path.basename(filepath)
            if count%10000 == 0 and count>0:
                print(count,"/",len(new_paths),"read")
            if len(formats)>0:
                if not filepath.split(".")[-1] in formats:
                    continue
            try:
                header = read(filepath,headonly=True)
            except:
                with open("logfile_create_ccs_unreadable.txt","a") as f:
                    f.write("could not read file: %s" %filepath)
                continue
               
            net = header[0].stats.network
            sta = header[0].stats.station
            for head in header:
                if head.stats.network != net or head.stats.station != sta:
                    print("Warning! Each file is supposed to contain data from one station. Other traces in the file are ignored.")
                    print(filepath)
                    print("network:",net,"station:",sta,"ignored station,network:",head.stats.network,head.stats.station)
                    break
            if sta=="":
                sta = fname[:3]
                #print("file header is missing the station name! script will not work!")
            loc = header[0].stats.location
            cha = header[0].stats.channel
            year0 = header.sort()[0].stats.starttime.year
            jday0 = header.sort()[0].stats.starttime.julday
            hr0 = header.sort()[0].stats.starttime.hour
            min0 = header.sort()[0].stats.starttime.minute
            sec0 = header.sort()[0].stats.starttime.second
            year1 = header.sort()[-1].stats.endtime.year
            jday1 = header.sort()[-1].stats.endtime.julday
            hr1 = header.sort()[-1].stats.endtime.hour
            min1 = header.sort()[-1].stats.endtime.minute
            sec1 = header.sort()[-1].stats.endtime.second
            fileformat = header[0].stats._format
            
            #net,sta,loc,cha,year0,jday0,hr0,min0,sec0,year1,jday1,hr1,min1,sec1,fileformat = fname.split(".")
            staid = net+'.'+sta

            comp = cha[-1]
            tstart = UTCDateTime(year=int(year0),julday=int(jday0),hour=int(hr0),minute=int(min0),second=int(sec0))
            tend = UTCDateTime(year=int(year1),julday=int(jday1),hour=int(hr1),minute=int(min1),second=int(sec1))
            try:
                c.execute("INSERT INTO file_db (staid,component,starttime,endtime,path) VALUES(?,?,?,?,?)",
                          (staid, comp, str(tstart), str(tend), filepath))
            except sqlite3.IntegrityError:
                print('Warning: row already exists in table!')
                print(net,sta,loc,cha,tstart,tend)
                #print(tr.stats.network,tr.stats.station,cha,comp,tstart,tend)
                c.execute("SELECT * FROM file_db WHERE staid=? and component =? and starttime=?",(staid, comp, str(tstart)))
                entry = c.fetchall()[0]
                staid_2,comp_2,tstart_2,tend_2,path_2 = entry
                print("existing entry:",path_2)
                print("new filepath:",filepath)
                if tend > UTCDateTime(tend_2):
                    c.execute("DELETE FROM file_db WHERE staid=? and component =? and starttime=?",(staid, comp, str(tstart)))
                    c.execute("INSERT INTO file_db (staid,component,starttime,endtime,path) VALUES(?,?,?,?,?)",
                          (staid, comp, str(tstart), str(tend), filepath))
                print('-----')
            count += 1
            if count%1000 == 0:
                conn.commit()
        conn.commit()
        conn.close()

        print("Database successfully updated.",flush=True)
    
    #%%#######################################################################
    # UPDATING THE STATION DICTIONARY
    
    statdict={}
    create_statlist=False
    if os.path.isfile(statfilepath):
        with open(statfilepath,'r') as f:
            for line in f:
                if line.startswith("#"):
                    continue
                line = line.split()
                statdict[line[0]] = {}
                statdict[line[0]]['latitude'] = float(line[1])
                statdict[line[0]]['longitude'] = float(line[2])
                try:
                    statdict[line[0]]['elevation'] = float(line[3])
                except:
                    pass
    else:
        print("no statlist found, creating statlist from files")
        create_statlist=True
        
    if (create_statlist or update_statlist) and mpi_rank==0:
        
        # find inventory files
        inventory_filelist = []
        if mpi_rank == 0:
            for dir_,_,files in os.walk(inventory_directory):
                for file in files:
                    if file.lower().endswith("xml"):
                        inventory_filelist.append(os.path.join(dir_,file))
        inventory_filelist = mpi_comm.bcast(inventory_filelist,root=0)
        
        conn = sqlite3.connect(database_file)
        c = conn.cursor()
        c.execute("SELECT staid FROM file_db")
        station_ids = np.unique(c.fetchall())
        
        for staid in station_ids:
            net = staid.split(".")[0]
            sta = staid.split(".")[1]
            if not staid in statdict.keys():
                # try to get the lat lon information from the station xml files
                statdict[staid] = {}
                inv_filepaths = []
                for inv_filepath in inventory_filelist:
                    if net in inv_filepath and sta in inv_filepath:
                        inv_filepaths.append(inv_filepath)
                for inv_filepath in inv_filepaths:
                    try:
                        inventory = read_inventory(inv_filepath)
                        if inventory[0].code == net and inventory[0][0].code==sta:
                            statdict[staid]['latitude'] = inventory[0][0].latitude
                            statdict[staid]['longitude'] = inventory[0][0].longitude
                            try:
                                statdict[staid]['elevation'] = inventory[0][0].elevation
                            except:
                                pass
                            break
                    except:
                        print("file not readable:",inv_filepath)
                else:
                    try: # try to get the station information from the sac headers
                         # if the input files are not sac, will not work
                        c.execute("SELECT path FROM file_db WHERE staid=?",(staid,))
                        filepath = c.fetchall()[0][0]
                        header = read(filepath,headonly=True)
                        statdict[staid]['latitude'] = header[0].stats.sac.stla
                        statdict[staid]['longitude'] = header[0].stats.sac.stlo
                    except:
                        print("could not get any lat/lon information for station",net,sta)
        
        conn.close()
        
        if create_statlist or update_statlist:
            with open(statfilepath,"w") as f:
                f.write("# staid     lat       lon      elevation\n")
                for staid in np.sort(list(statdict.keys())):
                    staid_str = staid + (9-len(staid))*" " 
                    if 'elevation' in statdict[staid].keys():
                        f.write("%s %9.6f %10.6f %8.3f\n" %(staid_str,statdict[staid]['latitude'],
                                                            statdict[staid]['longitude'],
                                                            statdict[staid]['elevation']))
                    elif ('latitude' in statdict[staid].keys() and 
                          'longitude' in statdict[staid].keys()):
                        f.write("%s %9.6f %10.6f\n" %(staid_str,statdict[staid]['latitude'],
                                                      statdict[staid]['longitude']))
                    else:
                        f.write("%s\n" %staid_str)
    
    #%%
        
    pairdict = {}
    available_corrdays = []
    
    if mpi_rank==0:
        # create a station list and list all possible pairs
        conn = sqlite3.connect(database_file)
        c = conn.cursor()
        c.execute("SELECT staid FROM file_db")
        database_list = c.fetchall()
        statlist = list(set(database_list))
        statlist.sort()
        if len(statlist) != len(statdict) and mpi_rank==0:
            print("*****")
            print("%d stations in statlist" %(len(statdict)))
            print("%d stations in database" %(len(statlist)))
            print("*****")
            for stat in statlist:
                try:
                    statdict[stat[0]]
                except:
                    pass
        pairs = list(combinations(np.array(statlist)[:,0],2))
        c.execute("SELECT * FROM file_db")
        database_list = c.fetchall()
        conn.close()
    
    #%%
    ############################                    
        
        data_dic = {}
        timezone = ""
        for line in database_list:
            staid,comp,tstart,tend,path = line
            tstamp = UTCDateTime(tstart)
            tstamp = (tstamp.year,tstamp.julday)
            if not tstamp in available_corrdays:
                available_corrdays.append(tstamp)
            if not comp in ['Z','N','E']:
                continue
            try:
                data_dic[staid]
            except:
                data_dic[staid] = {}
                data_dic[staid]['Z'] = {}
                data_dic[staid]['N'] = {}
                data_dic[staid]['E'] = {}
                data_dic[staid]['Z']['windows'] = []
                data_dic[staid]['Z']['paths'] = []
                data_dic[staid]['N']['windows'] = []
                data_dic[staid]['N']['paths'] = []
                data_dic[staid]['E']['windows'] = []
                data_dic[staid]['E']['paths'] = []
            #data_dic[staid][comp]['windows'].append([UTCDateTime(tstart).__abs__(),UTCDateTime(tend).__abs__()])
            data_dic[staid][comp]['windows'].append([tstart,tend])
            data_dic[staid][comp]['paths'].append(path)
            if timezone == "":
                timezone = pd.to_datetime(tstart).tzinfo

        available_corrdays.sort()

        # convert to float array with absolute seconds since 01/01/1970 for easier handling
        for staid in data_dic:
            for comp in ['Z','N','E']:
                data_dic[staid][comp]['windows'] = np.array(data_dic[staid][comp]['windows'])  
                if len(data_dic[staid][comp]['windows']) > 0:
                    data_dic[staid][comp]['windows'][:,0] = (pd.to_datetime(data_dic[staid][comp]['windows'][:,0])-pd.Timestamp(year=1970,month=1,day=1,tz=timezone)).total_seconds().values
                    data_dic[staid][comp]['windows'][:,1] = (pd.to_datetime(data_dic[staid][comp]['windows'][:,1])-pd.Timestamp(year=1970,month=1,day=1,tz=timezone)).total_seconds().values
                    data_dic[staid][comp]['windows'] = data_dic[staid][comp]['windows'].astype(float)
                data_dic[staid][comp]['paths'] = np.array(data_dic[staid][comp]['paths'])
        
       
        for pair in pairs:
            
            if only_process_these_stations != None and only_process_these_stations != []:
                if not (pair[0] in only_process_these_stations and pair[1] in only_process_these_stations):
                    continue
            
            try:
                dist,az,baz = gps2dist_azimuth(statdict[pair[0]]['latitude'],
                                               statdict[pair[0]]['longitude'],
                                               statdict[pair[1]]['latitude'],
                                               statdict[pair[1]]['longitude'])
            except:
                raise Exception("No lat/lon information found in statfile for %s or %s" %(pair[0],pair[1]))
                
            dist/=1000.
            
            if dist > min_distance:
                pairdict[pair] = {}
                pairdict[pair]['dist'] = dist
                pairdict[pair]['az'] = az
                pairdict[pair]['baz'] = baz

  
        existing_files = glob.glob(os.path.join(spectra_path,"**/*.pkl"),recursive=True)

        
        existing_corrdays = {}    
        for i,pair in enumerate(list(pairdict)): 
            existing_corrdays[pair] = {}
            for corrcomp in comp_correlations:  
                existing_corrdays[pair][corrcomp] = []
        
            
        # check that there is only one file for each pair
        print("reading existing files")
        for i,pair in enumerate(list(pairdict)):
            
            if i%10000==0:
                print(i)
                            
            for corrcomp in comp_correlations:
                                    
                filepath1 = getfilepath(pair[0],pair[1],corrcomp,pairdict[pair]['dist'],overlap)
                filepath2 = getfilepath(pair[1],pair[0],corrcomp,pairdict[pair]['dist'],overlap)

                if os.path.isfile(filepath1):
                    
                    if os.path.isfile(filepath2):
                        print(filepath1)
                        print(filepath2)
                        raise Exception("two files for the same pair!")
                        
                    with open(filepath1,"rb") as f:
                        corr_dict = pickle.load(f)
                        
                elif os.path.isfile(filepath2):

                    print("warning: renaming dictionary file!")                        
                    with open(filepath2,"rb") as f:
                        corr_dict = pickle.load(f)
                    with open(filepath1,"wb") as f:
                        pickle.dump(corr_dict,f)
                    os.remove(filepath2)
                    
                else:
                    continue
            
                existing_corrdays[pair][corrcomp] = corr_dict['corrdays']
      
        
                             
        # cleanup
        database_list = []
        statlist = []
        pairs = []
        existing_files = []

        
    # wait for the first process (mpi_rank=0, root=0) to get to this point      
    # share data among processes
    pairdict = mpi_comm.bcast(pairdict,root=0)
    available_corrdays = mpi_comm.bcast(available_corrdays,root=0)
    
    #%%######################
    """ PROCESS LOOP """
    time_start=datetime.datetime.now()
    no_processed_days = 0
    
    
    for corrday in available_corrdays:
        
        year = corrday[0]
        day = corrday[1]
        
        if year not in years and len(years)>0:
            continue
        #if not day==3:
        #    continue
                        
        if mpi_rank==0:
            print("working on correlation day:",year,day)
    
        stream_z = Stream()
        stream_n = Stream()    
        stream_e = Stream()            

        worklist = []

        if 'ZZ' in comp_correlations:
        
            print("Correlation ZZ\n",datetime.datetime.now(),"\nYear:",year,"Julday:",day,file=flog,flush=True)               

            if mpi_rank == 0:
                
                for pair in list(pairdict):                        
                    if (year,day) in existing_corrdays[pair]['ZZ']:
                        continue
                    else:
                        worklist.append(pair)

                station_ids = np.unique(worklist)
                
                filelist_z = get_julday_filelist(year,day,'Z',station_ids,window_length)
                
                if len(filelist_z)>1:                        
                    for fpath in filelist_z:
                        st = read(fpath)
                        #for tr in st:
                        #    if tr.stats.station == "":
                        #        tr.stats.station = os.path.basename(fpath)[:3]
                        stream_z += st

            # wait for the first process (mpi_rank=0, root=0) to get to this point      
            # share read in data among processes
            #mpi_comm.Barrier() 
            stream_z = mpi_comm.bcast(stream_z,root=0)
            worklist = mpi_comm.bcast(worklist,root=0)

            print("Working on",len(worklist[mpi_rank::mpi_size]),"station pairs",file=flog,flush=True)
            for i,pair in enumerate(worklist[mpi_rank::mpi_size]):
                                    
                if i%10000 == 0:
                    print("Processed",i,"/",len(worklist[mpi_rank::mpi_size]),"pairs",file=flog,flush=True)                
                process_noise(stream_z,pair,'ZZ',window_length,overlap,year,day,flog)
                      
            print("Finished processing ZZ correlation day.",file=flog,flush=True)                      
            stream_z = Stream()
                    
      
        if 'RR' in comp_correlations or 'TT' in comp_correlations:
            
            print("Correlation of horizontal components\n",datetime.datetime.now(),"\nYear:",year,"Julday:",day,file=flog,flush=True)                
              
            if mpi_rank == 0:
                
                if 'RR' in comp_correlations:
                    
                    worklist_rr = []
                    for pair in list(pairdict):                        
                        if (year,day) in existing_corrdays[pair]['RR']:
                            continue
                        else:
                            worklist_rr.append(pair)
                    
                if 'TT' in comp_correlations:
                    
                    worklist_tt = []
                    for pair in list(pairdict):                        
                        if (year,day) in existing_corrdays[pair]['TT']:
                            continue
                        else:
                            worklist_tt.append(pair)
                                            
                worklist = list(set(worklist_rr) | set(worklist_tt))
    
                station_ids = np.unique(worklist)
                
                filelist_n = get_julday_filelist(year,day,'N',station_ids,window_length)
                
                filelist_e = get_julday_filelist(year,day,'E',station_ids,window_length)

                if len(filelist_n)>1 and len(filelist_e)>1:               
                    for fpath in filelist_n:                 
                        st = read(fpath)                       
                        #for tr in st:
                        #    if tr.stats.station == "":
                        #        tr.stats.station = os.path.basename(fpath)[:3]
                        stream_n += st   
                    for fpath in filelist_e:
                        st = read(fpath)
                        #for tr in st:
                        #    if tr.stats.station == "":
                        #        tr.stats.station = os.path.basename(fpath)[:3]
                        stream_e += st               

            # wait for the first process (mpi_rank=0, root=0) to get to this point      
            # share read-in data among processes
            #mpi_comm.Barrier()          
            stream_n = mpi_comm.bcast(stream_n,root=0)
            stream_e = mpi_comm.bcast(stream_e,root=0)
            worklist = mpi_comm.bcast(worklist,root=0)

            if 'RR' in comp_correlations and 'TT' in comp_correlations:
                corrcomp = ['RR','TT']
            elif 'RR' in comp_correlations:
                corrcomp = 'RR'
            elif 'TT' in comp_correlations:
                corrcomp = 'TT'
            else:
                raise Exception("error")
            
            print("Working on",len(worklist[mpi_rank::mpi_size]),"station pairs",file=flog,flush=True)
            for i,pair in enumerate(worklist[mpi_rank::mpi_size]):
                
                if i%10000 == 0:
                    print("Processed",i,"/",len(worklist[mpi_rank::mpi_size]),"pairs",file=flog,flush=True)                
                process_noise((stream_n+stream_e),pair,corrcomp,window_length,overlap,year,day,flog)
               
            print("Finished processing horizontal components for one correlation day.",file=flog,flush=True)                      

            stream_n = Stream()    
            stream_e = Stream()    
                    
        
    flog.close()
