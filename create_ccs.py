# -*- coding: utf-8 -*-
"""
Updated April 2020

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
formats = ['mseed','SAC','sync']

# station list (file is created if not existing yet)
statfilepath = './statlist.txt' # 3 columns: station_id latitude longitude

# path where the cross correlation spectra should be saved
# if there are already existing files with the same name, they may be overwritten
# unless you set update=True
spectra_path='cross_correlation_spectra'

# update already existing crosscorrelation files if new data has been added
# to the database. If set to False, old files are overwritten!
update = True

# filename of sqlite database (created if not yet existing)
# this database lists all existing files, components, available timeranges, etc.
database_file = 'database_ambnoise.sqlite'
# check if there are new files in the path. The sqlite database is then updated
update_database = True

# traces are cut into windows. windowed data is then correlated
# ideal length depends on your typical station distances and if you're interested in the coda
window_length=3600. # in seconds

# overlap of subsequent windows (recommended between 0.3 and 0.6)
overlap = 0.6

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
from obspy import read, Stream, UTCDateTime
from obspy.geodetics.base import gps2dist_azimuth
from itertools import combinations
import os, glob, datetime, pickle
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import noise
import sqlite3


            
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
            #print(stat1,stat2,"no data in stream for",corrcomp,"correlation!")
            #print(stream)
            return None
            
        timeranges = []
        for tr1 in stream1:
            for tr2 in stream2:
                
                if tr1.stats.starttime > tr2.stats.starttime:
                    tstart = tr1.stats.starttime
                else:
                    tstart = tr2.stats.starttime
                if tr1.stats.endtime > tr2.stats.endtime:
                    tend = tr2.stats.endtime
                else:
                    tend = tr1.stats.endtime

                if tend >= tstart+window_length:                       
                    timeranges.append([tstart,tend])
                    

        if len(timeranges)==0:
            return None
        
    
        corr_list = []
        no_windows = []
        
        for tstart,tend in timeranges:

            st1 = stream1.slice(starttime=tstart,endtime=tend)
            st2 = stream2.slice(starttime=tstart,endtime=tend)
            if len(st1)!=1 or len(st2)!=1:
                st1._cleanup()
                st2._cleanup()
                if len(st1)>1:
                    # just take the longer trace
                    st1 = Stream(st1[np.array(list(map(len,st1))).argmax()])
                if len(st2)>1:
                    st2 = Stream(st2[np.array(list(map(len,st2))).argmax()])
                
            try:
                freq,spec = noise.noisecorr(st1[0],st2[0],window_length,overlap,
                                            whiten=whiten,onebit=onebit)
                corr_list.append(spec)
                no_windows.append(int((tend-tstart-window_length)/((1-overlap)*window_length))+1)
            except:
                continue
        
        if len(corr_list) == 0:
            return None
        
        corr_spectrum = np.average(np.array(corr_list),axis=0,weights=no_windows)
        
        filepath = getfilepath(stat1,stat2,"ZZ",pairdict[pair]['dist'],overlap)
        
        if os.path.isfile(filepath):
            with open(filepath,"rb") as f:
                corr_dict = pickle.load(f)
        
            if (year,julday) in corr_dict['corrdays']:
                raise Exception("correlation day already in database!",filepath,year,julday)            
            
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
        
        if len(stream1n) == 0 or len(stream1e) == 0 or len(stream2n) == 0 or len(stream2e) == 0:
            #print(stat1,stat2,"no data in stream for",corrcomp,"correlation!")
            #print(stream)
            return None      
        
        timeranges = []

        timeranges1 = []
        for tr1 in stream1n:
            for tr2 in stream1e:
                
                if tr1.stats.starttime > tr2.stats.starttime:
                    tstart = tr1.stats.starttime
                else:
                    tstart = tr2.stats.starttime
                if tr1.stats.endtime > tr2.stats.endtime:
                    tend = tr2.stats.endtime
                else:
                    tend = tr1.stats.endtime

                if tend >= tstart+window_length:                       
                    timeranges1.append([tstart,tend])
                    
        if len(timeranges1)==0:
            return None

        timeranges2 = []
        for tr1 in stream2n:
            for tr2 in stream2e:
                
                if tr1.stats.starttime > tr2.stats.starttime:
                    tstart = tr1.stats.starttime
                else:
                    tstart = tr2.stats.starttime
                    
                if tr1.stats.endtime > tr2.stats.endtime:
                    tend = tr2.stats.endtime
                else:
                    tend = tr1.stats.endtime

                if tend >= tstart+window_length:                       
                    timeranges2.append([tstart,tend])
                    
        if len(timeranges2)==0:
            return None
        
        
        for range1 in timeranges1:
            for range2 in timeranges2:
                
                if range1[0] > range2[0]:
                    tstart = range1[0]
                else:
                    tstart = range2[0]
                    
                if range1[1] > range2[1]:
                    tend = range2[1]
                else:
                    tend = range1[1]
                    
                if tend >= tstart+window_length:
                    timeranges.append([tstart,tend])
        
    
        corr_list_rr = []
        corr_list_tt = []
        no_windows = []

        for tstart,tend in timeranges:
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
                continue
                
            # az = azimuth from station1 -> station2
            # baz = azimuth from station2 -> station1
            # for stream2 the back azimuth points in direction of station1
            # for stream1 the azimuth points in direction of station2
            # BUT 180. degree shift is needed so that the radial components point in the same direction!
            # otherwise they point towards each other => transverse comp would be also opposed
            try:
                st1.rotate('NE->RT',back_azimuth=(pairdict[pair]['az']+180.)%360.)
                st2.rotate('NE->RT',back_azimuth=pairdict[pair]['baz'])                     
            except:
                print("Error rotating stream",file=flog)
                continue
                    
            try:
                if 'RR' in corrcomp:
                    freq,spec_rr = noise.noisecorr(st1.select(component='R')[0],
                                                st2.select(component='R')[0],
                                                window_length,overlap,
                                                whiten=whiten,onebit=onebit)
                if 'TT' in corrcomp:
                    freq,spec_tt = noise.noisecorr(st1.select(component='T')[0],
                                                st2.select(component='T')[0],
                                                window_length,overlap,
                                                whiten=whiten,onebit=onebit)
            except:
                continue


            if 'RR' in comp_correlations:
                corr_list_rr.append(spec_rr)
            if 'TT' in comp_correlations:
                corr_list_tt.append(spec_tt)
            no_windows.append(int((tend-tstart-window_length)/((1-overlap)*window_length))+1)



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
        

    
def get_julday_filelist(year,julday,comp,staids):
    
    filelist = []

    try:
        starttime = abs(UTCDateTime(year=year,julday=day))
    except:
        return filelist
    endtime = starttime+24*60*60
    
    for staid in data_dic:
        if not staid in staids:
            continue
        if len(data_dic[staid][comp]['windows']) == 0:
            continue
        timematch = np.where((data_dic[staid][comp]['windows'][:,0]<=endtime)*(data_dic[staid][comp]['windows'][:,1]>=starttime))[0]
        for idx in timematch:
            filelist.append(data_dic[staid][comp]['paths'][idx])
            
    return filelist
    

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

    statdict={}
    create_statlist=False
    if os.path.isfile(statfilepath):
        with open(statfilepath,'r') as f:
            for line in f:
                line = line.split()
                statdict[line[0]] = {}
                statdict[line[0]]['latitude'] = np.float(line[1])
                statdict[line[0]]['longitude'] = np.float(line[2])
                try:
                    statdict[line[0]]['elevation'] = np.float(line[3])
                except:
                    pass
    else:
        print("creating statlist from files")
        create_statlist=True
    
    
    #%%#############################################################################
    
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
        print("adding new entries...")
        #errorfile = open("read_errors_database.txt","w")
        #print("check for errors in",errorfile)
        if create_statlist:
            new_paths = set(pathlist_new).union(set(pathlist))
        for filepath in new_paths:
            #print "adding new entry:",filepath
            fname = os.path.basename(filepath)
            if count%10000 == 0:
                print(count,"/",len(new_paths),"read")
            if not filepath.split(".")[-1] in formats:
                continue
            header = read(filepath,headonly=True)
            if len(header)>1:
                print("Warning: more than 1 trace per file. Just taking first trace!")
            header = header[0]
            
            net = header.stats.network
            sta = header.stats.station
            if sta=="":
                sta = fname[:3]
                #print("file header is missing the station name! script will not work!")
            loc = header.stats.location
            cha = header.stats.channel
            year0 = header.stats.starttime.year
            jday0 = header.stats.starttime.julday
            hr0 = header.stats.starttime.hour
            min0 = header.stats.starttime.minute
            sec0 = header.stats.starttime.second
            year1 = header.stats.endtime.year
            jday1 = header.stats.endtime.julday
            hr1 = header.stats.endtime.hour
            min1 = header.stats.endtime.minute
            sec1 = header.stats.endtime.second
            fileformat = header.stats._format
            
            #net,sta,loc,cha,year0,jday0,hr0,min0,sec0,year1,jday1,hr1,min1,sec1,fileformat = fname.split(".")
            staid = net+'.'+sta
            if not staid in statdict.keys():
                statdict[staid] = {}
                statdict[staid]['latitude'] = header.stats.sac.stla
                statdict[staid]['longitude'] = header.stats.sac.stlo
                
            comp = cha[-1]
            tstart = UTCDateTime(year=int(year0),julday=int(jday0),hour=int(hr0),minute=int(min0),second=int(sec0))
            tend = UTCDateTime(year=int(year1),julday=int(jday1),hour=int(hr1),minute=int(min1),second=int(sec1))
            try:
                c.execute("INSERT INTO file_db (staid,component,starttime,endtime,path) VALUES(?,?,?,?,?)",
                          (staid, comp, str(tstart), str(tend), filepath))
            except sqlite3.IntegrityError:
                if not create_statlist:
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
    
        if create_statlist:
            with open(statfilepath,"w") as f:
                for staid in statdict:
                    f.write("%s %.5f %.5f\n" %(staid,statdict[staid]['latitude'],statdict[staid]['longitude']))
    
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

        #banana
        #with open("test.pkl","rb") as f:
        #    existing_corrdays = pickle.load(f)
        #    
        #for pair in existing_corrdays:
        #    for corrcomp in existing_corrdays[pair]:
        #        existing_corrdays[pair][corrcomp] = np.array(existing_corrdays[pair][corrcomp])
        #    
        #with open("test2.pkl","wb") as f:
        #    pickle.dump(existing_corrdays,f)
            
        if update:
            
            existing_files = glob.glob(os.path.join(spectra_path,"**/*.pkl"),recursive=True)

            existing_corrdays = {}  
            
            # check that there is only one file for each pair
            print("reading existing files")
            for i,pair in enumerate(list(pairdict)):
                
                if i%10000==0:
                    print(i)
                
                existing_corrdays[pair] = {}
                
                for corrcomp in comp_correlations:
                    
                    existing_corrdays[pair][corrcomp] = []
                    
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
                
                filelist_z = get_julday_filelist(year,day,'Z',station_ids)
                
                if len(filelist_z)>1:                        
                    for fpath in filelist_z:
                        st = read(fpath)
                        for tr in st:
                            if tr.stats.station == "":
                                tr.stats.station = os.path.basename(fpath)[:3]
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
                
                filelist_n = get_julday_filelist(year,day,'N',station_ids)
                
                filelist_e = get_julday_filelist(year,day,'E',station_ids)

                if len(filelist_n)>1 and len(filelist_e)>1:               
                    for fpath in filelist_n:                 
                        st = read(fpath)
                        for tr in st:
                            if tr.stats.station == "":
                                tr.stats.station = os.path.basename(fpath)[:3]
                        stream_n += st   
                    for fpath in filelist_e:
                        st = read(fpath)
                        for tr in st:
                            if tr.stats.station == "":
                                tr.stats.station = os.path.basename(fpath)[:3]
                        stream_e += st               

            # wait for the first process (mpi_rank=0, root=0) to get to this point      
            # share read in data among processes
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
