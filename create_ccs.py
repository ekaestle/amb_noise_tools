# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 17:32:43 2014

@author: emanuel

This script tries to read all .SAC and .mseed files in the specified folder and its subfolders.
It creates a database of all existing files with the respective components and time.
It is assumed that one file corresponds to a single day and a single component (Z,N,E).
If your database is in a different format, it may not work and the script below may have to be adapted.
Don't hesitate to contact me if you encounter any difficulties: Emanuel Kaestle @ FU Berlin
"""


import numpy as np
from obspy import read, Stream
from obspy.geodetics.base import gps2dist_azimuth
from itertools import combinations
import os, sys, glob,  datetime
from multiprocessing import Pool
import noise
import sqlite3

# creates cross-correlation files named
#stat1_X_stat2_dist_distance_st_corrdays_ovlap_overlap

""" USER DEFINED PARAMETERS"""
horizontal_polarization=False # If True, the script will read only N and E component files from the database.
# The traces will be rotated (station locations must be provided in the statlist file).
# zero crossings will be extracted with the additional J2 term. 
no_cores = 4 # might cause probems on windows machines, because of missing "if __name__ == '__main__':"
path='./preprocessed_data/' # path to dataset
# at the moment only files with .SAC and .mseed ending are read. you can change this in the code below (obspy can read many data formats)
statlistpath = './statlist.txt' # should contain 4 columns, statname, lat, lon, elevation (elevation is not necessary, can be a dummy)
spectra_path='./CROSS_SPECTRA/' # where to save CC spectra
if horizontal_polarization:
    spectra_path = './CROSS_SPECTRA_T/' 
    spectra_path_r='./CROSS_SPECTRA_R/'
database_file = './database.sqlite' # filename of sqlite database (created if not yet existing)
min_days = 3 # minimum number of days for CC, should probably be greater than half a year
overlap = 0.6 # overlap of successive cross-correlation windows
window_length=3600. # length of the CC window in seconds
min_distance = 20. # minimum allowed inter-station distance in km
#see also other parameters for function noise.noisecorr below.
""" END OF USER DEFINED PARAMETERS"""

statdict={}
with open(statlistpath,'r') as f:
    for line in f:
        line = line.split()
        statdict[line[0]] = {}
        statdict[line[0]]['latitude'] = np.float(line[1])
        statdict[line[0]]['longitude'] = np.float(line[2])
        try:
            statdict[line[0]]['elevation'] = np.float(line[3])
        except:
            pass
        
cachedict = {}

if not os.path.exists(spectra_path):
    os.makedirs(spectra_path)
if horizontal_polarization:
    if not os.path.exists(spectra_path_r):
        os.makedirs(spectra_path_r)
#for f in glob.glob(spectra_path+"*"):
#    os.remove(f)

#xcorr_path='./CROSS_CORRELATIONS/'
#if not os.path.exists(xcorr_path):
#    os.makedirs(xcorr_path)
#for f in glob.glob(xcorr_path+"*"):
#    os.remove(f)

existing_files=glob.glob(spectra_path+"/*")
print len(existing_files),"already calculated cross spectra have been found."

if horizontal_polarization:
    rayleigh_radial = True
else:
    rayleigh_radial = False

def multiprocess_fft(inlist):
    tr1=inlist[0]
    tr2=inlist[1]
    window_length=inlist[2]
    overlap = inlist[3]
    try:
        return noise.noisecorr(tr1,tr2,window_length,overlap)
    except:
        return None
        
# recommandation: use velocity filter only at the end, before continuing the processing. 
def process_noise(stat1,stat2,nextstat1,nextstat2,window_length,overlap,lovewaves=False):
    global spectra_path
    global spectra_path_r
    global cachedict
    global statdict
    global min_days

    try:
        dist,az,baz = gps2dist_azimuth(statdict[stat1]['latitude'],
                                                   statdict[stat1]['longitude'],
                                                   statdict[stat2]['latitude'],
                                                   statdict[stat2]['longitude'])
    except:
                        raise Exception("No lat/lon information found in statfile for %s or %s" %(stat1,stat2))       

    dist/=1000.
    if dist<min_distance:
        print "Aborting: Interstation distance too short (< %d km)!" %min_distance
        return None
        
    i=0
    joblist=[]
    if lovewaves:
        if rayleigh_radial:
            joblist_rad=[]
            print "Love- and Rayleighwave correlation..."
        else:
            print "Lovewave correlation..."
        filelist1=data_dic[stat1]['N']
        filelist2=data_dic[stat2]['N']
        
        for year in filelist1:
            for jday in filelist1[year]:
                stream1=Stream()
                stream2=Stream()
                try:
                    #stream1+=read(data_dic[stat1]['Z'][year][jday])
                    stream1+=read(data_dic[stat1]['N'][year][jday])
                    stream1+=read(data_dic[stat1]['E'][year][jday])
                    
                    #stream2+=read(data_dic[stat2]['Z'][year][jday])
                    stream2+=read(data_dic[stat2]['N'][year][jday])
                    stream2+=read(data_dic[stat2]['E'][year][jday])
                except:
                    continue
                
                try:
                    stream1,stream2 = noise.adapt_timespan(stream1,stream2)
                except:
                    continue
                
                if (stream1[0].stats.endtime-stream1[0].stats.starttime)/60./60. < 20.:
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
                
                joblist.append([tr1,tr2,window_length,overlap])
                i+=1

                if rayleigh_radial:
                    tr1_rad = stream1.select(component='R')[0]
                    tr2_rad = stream2.select(component='R')[0]
                    joblist_rad.append([tr1_rad,tr2_rad,window_length,overlap])
                
    else:  
        print "Rayleighwave correlation..."
        print "    reading"
        matches=[]
        filelist1=data_dic[stat1]['Z']
        filelist2=data_dic[stat2]['Z']
        for year in filelist1:
            for jday in filelist1[year]:
                try:
                    matches.append([filelist1[year][jday],filelist2[year][jday]])
                except:
                    continue

        #cleanup cachedict
        cachedict_copy = {} 
        for key in cachedict:
            if nextstat1 in key or nextstat2 in key:
                cachedict_copy[key] = cachedict[key]
        cachedict = cachedict_copy
        for filepath in matches:
            try:
                tr1 = cachedict[filepath[0]]
            except:
                tr1=read(filepath[0])[0]
                if stat1==nextstat1 or stat1==nextstat2:
                    cachedict[filepath[0]] = tr1                
            try:
                tr2 = cachedict[filepath[1]]
            except:
                tr2=read(filepath[1])[0]
                if stat2==nextstat1 or stat2==nextstat2:
                    cachedict[filepath[1]] = tr2                    
            
            joblist.append([tr1,tr2,window_length,overlap])
            i+=1
            
    if i<min_days:
        print "Less than %d days, aborting!" %min_days
        return None      
    
    print "    starting FFT with",no_cores,"cores"
    p = Pool(no_cores)
    corr_list = p.map(multiprocess_fft, joblist)
    p.close()
    # if multiprocessing fails we can do it in a loop:
    #corr_list = []
    #for j,element in enumerate(joblist):
    #    print "fft",j,"/",len(joblist)
    #    corr_list.append(multiprocess_fft(element))
    freq = corr_list[0][0]
    try:
        corr_spectrum = np.mean(np.array(corr_list)[:,1],axis=0)
    except:
        corr_spectrum = np.zeros(len(corr_list[0][0]),dtype=complex)
        for corrday in corr_list:
            if corrday == None:
                continue
            # eliminating the quadratic trend seems to be necessary as some data (esp. 2008) introduce a strange error
            #model = polyfit(freq, real(corrday[1]), 2)
            #corr_spectrum += (corrday[1] - polyval(model,freq))
            corr_spectrum += corrday[1]
        corr_spectrum/=float(len(corr_list))
    
    stat1 = tr1.stats.station
    stat2 = tr2.stats.station
    filename = stat1+"_X_"+stat2+"_dist_%08d_st_%05d_ovlap_%.2f"\
        %(int(dist*1000.),i,overlap)
    
    np.savetxt(spectra_path+filename,np.column_stack((freq,\
        np.real(corr_spectrum),np.imag(corr_spectrum))))
            
#        crossings_path='./CROSSINGS/'
#        if not os.path.exists(crossings_path):
#            os.makedirs(crossings_path)
#        savetxt(crossings_path+filename+"_cross",crossings)
    
            
    #==================================#
            #Rayleigh#
    if rayleigh_radial:
        p = Pool(no_cores)
        corr_list = p.map(multiprocess_fft, joblist_rad)
        p.close()
        freq = corr_list[0][0]
        corr_spectrum = np.zeros(len(corr_list[0][0]),dtype=complex)
        for corrday in corr_list:
            corr_spectrum+=corrday[1]
        corr_spectrum/=float(len(corr_list))
        
        stat1 = tr1.stats.station
        stat2 = tr2.stats.station
        filename = stat1+"_X_"+stat2+"_dist_%08d_st_%05d_ovlap_%.2f"\
            %(int(dist*1000.),i,overlap)
            
        np.savetxt(spectra_path_r+filename,np.column_stack((freq,\
            np.real(corr_spectrum),np.imag(corr_spectrum))))
    
    print "successfully processed",stat1,stat2
    return
##############################################################################
"""
##############################################################################
"""
#%%#############################################################################
print "checking and updating database..."
conn = sqlite3.connect(database_file)
c = conn.cursor()
try:
    c.execute("""CREATE TABLE IF NOT EXISTS file_db (
                network text,
                station text,
                component text,
                year integer,
                jday integer,
                path text,
                PRIMARY KEY(network,station,component,year,jday));""")
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
        print "error",fpath
        print "deleting from database"
        c.execute("DELETE FROM file_db WHERE path=?", (fpath,))
conn.commit()
# check for new paths
count = 0
pathlist_new = [os.path.join(dir_,f) for dir_,_,files in os.walk(path) for f in files if (f.endswith('.SAC') or f.endswith('.mseed'))]
new_paths = set(pathlist_new) - set(pathlist) #elements that are uniquely in pathlist_new
#new_paths = set(pathlist).symmetric_difference(set(pathlist_new))
for filepath in new_paths:
    print "adding new entry:",filepath
    st = read(filepath,headonly=True)
    for tr in st:
        year = (tr.stats.starttime+2*3600.).year
        jday = (tr.stats.starttime+2*3600.).julday
        try:
            c.execute("INSERT INTO file_db (network,station,component,year,jday,path) VALUES(?,?,?,?,?,?)",
                (tr.stats.network, tr.stats.station, tr.stats.channel[2], year, jday, filepath))
        except sqlite3.IntegrityError:
            print('ERROR: row already exists in table!')
            print tr.stats.network,tr.stats.station,tr.stats.channel[2],year,jday
conn.commit()
conn.close()

#%%
# create a station list and list all possible pairs
conn = sqlite3.connect(database_file)
c = conn.cursor()
c.execute("SELECT station FROM file_db")
database_list = c.fetchall()
statlist = list(set(database_list))
if len(statlist) != len(statdict):
    print "*****"
    print "station file not up to date! Length statlist:",len(statlist),"Length statdict:",len(statdict)
    print "*****"
    for stat in statlist:
        try:
            statdict[stat[0]]
        except:
            print "station",stat[0],"was not found in",statlistpath
pairs = list(combinations(statlist,2))
c.execute("SELECT * FROM file_db")
database_list = c.fetchall()
conn.close()

#%%
############################                    

existing_pairs={}
for pair in existing_files:
    existing_pairs[pair.split("/")[-1].split("_")[0],pair.split("/")[-1].split("_")[2]] = True
    existing_pairs[pair.split("/")[-1].split("_")[2],pair.split("/")[-1].split("_")[0]] = True

print len(existing_pairs)/2,"existing pairs found."
data_dic = {}
for line in database_list:
    try:
        data_dic[line[1]]
    except:
        data_dic[line[1]] = {}
    try:
        data_dic[line[1]][line[2]]
    except:
        data_dic[line[1]][line[2]] = {}
    try:
        data_dic[line[1]][line[2]][line[3]]
    except:
        data_dic[line[1]][line[2]][line[3]] = {}
    data_dic[line[1]][line[2]][line[3]][line[4]] = line[5]
    
#pair_dic = {}
print "create list of station pairs to be cross-correlated. checking pairs for a minimum of %d common days..." %min_days
worklist=[]  
for i,pair in enumerate(pairs):
    counter=0
    if i%10000 == 0:
        print i,"/",len(pairs),"pairs checked"
    stat1 = pair[0][0]
    stat2 = pair[1][0]
    new_entry=[stat1,stat2]
    try:
        existing_pairs[stat1,stat2]
        continue
    except:
        pass
    try:
        stat1list = data_dic[stat1]['Z']
    except:
        continue
    for year in stat1list:
        for jday in stat1list[year]:
            try:
                data_dic[stat2]['Z'][year][jday]
                counter+=1
            except:
                continue
    #stat2list = [tuple[3:5] for tuple in database_list if stat2 == tuple[1]]
    #common_days = set(stat1list) & set(stat2list)
    if counter < min_days:
        continue
    else:
        worklist.append(new_entry)
#    if len(common_days) < 150:
#        continue
#    else:
#        worklist.append([stat1,stat2])

print len(worklist),"pairs to be processed."
print "starting ..."
#%%###############################            
""" PROCESS LOOP """
time_start=datetime.datetime.now()
for count in xrange(len(worklist)):
    pair = worklist[count]
    try:
        nextpair = worklist[count+1]
        nextstat1=nextpair[0]
        nextstat2=nextpair[1]
    except:
        nextstat1='XXXXX'
        nextstat2='XXXXX'

    stat1 = pair[0]
    stat2 = pair[1]
         
    print ">>>>>> Processing pair",stat1,stat2
    sys.stdout.flush()
        
    try:
        phase_vel = process_noise(stat1,stat2,nextstat1,nextstat2,window_length=window_length,overlap=overlap,\
            lovewaves=horizontal_polarization)
    except Exception, e:
        print "not successfull"
        print e
        break
        #pass

    now=datetime.datetime.now()
    time_delta=(now-time_start).total_seconds()
    if count>0:
        time_left=time_delta*float(len(worklist)-count)/count
        print count,"of",len(worklist),"potential pairs processed. Estimated time left: %d:%.2d hrs" %(int(time_left/60./60.),int((time_left%(60*60))/60.))
    #tracker.print_diff()
