import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.cluster.vq import vq,kmeans,whiten,kmeans2
from sklearn import cluster as skcluster
import sys, os, fnmatch
from KNearestNeighbours import *
import matplotlib
import seaborn as sns
import pandas as pd
from astropy.coordinates import SkyCoord
from skimage.measure import label, regionprops
import threading, Queue
from pycuda import driver

def find_files(directory, pattern='*.dbase.drfi.clean.exp_time', sortby="auto"):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))

    files = np.sort(files)
    return files

def injectET(data1, freq_start, time_start):
        """
        The input is exp_time file read using python. The array should have five corresponding attributes
        freq,time,ra,dec,pow
        """
        # Find most common RA and DEC
        # There are two ways to do this. i
        # First is to find most frequently occuring pairs  

        #RA = data1['ra']
        #DEC = data1['dec']

        #d = zip(RA,DEC)
    
        # Find most common pairs of RA and DEC 
        #RAmax,DECmax = Counter(d).most_common(1)[0][0] 
        
        #(v1,c1) = np.unique(RA,return_counts=True)
        #(v2,c2) = np.unique(DEC,return_counts=True)
        
        #Extract the data for this block
        #ETdata = data1[((data1['dec']==DECmax) & (data1['ra']==RAmax))]
        # ^ not what we want eventually so drop

        # What we need is the pairs of RA and DEC which lasted longest. 
        # By looking at the data manully, I determind following ET location to inject birdie. 

        
        ETtime_start = time_start 
        ETtime_end = time_start + 50
        if False:
            ETdec = 16.9
            ETra =  19.1 
        else:
            #import IPython; IPython.embed()
            start_exind = np.random.choice(np.where(np.abs(data1[:,1]-ETtime_start)<1)[0])
            #end_exind = np.random.choice(np.where(np.abs(data1[:,0]-ETtime_end)<0.1)[0])
            ETdec = data1[start_exind][3]
            ETra = data1[start_exind][2]

        sortfreq = np.unique(np.sort(data1[0]))
        sorttime = np.unique(np.sort(data1[1]))
        if freq_start<min(sortfreq) or freq_start>max(sortfreq):
            freq_start = (min(sortfreq)+max(sortfreq))/2
        ETfreq_start = freq_start
        print "injecting signal at {},{}".format(ETtime_start, ETfreq_start)
        ETfreq_end = freq_start + 0.001
        ETpow = 15 # This is in log scale. 

        #From above values, calculate slope 
        ETslope = (ETfreq_end - ETfreq_start)/(ETtime_end - ETtime_start)
        
        #Fixed, this will overide ETfreq_end frequencies. The value must be in MHz/sec
        ETslope = 0.0001
        
        ETtime = ETtime_end - ETtime_start
        ETdata = np.zeros((ETtime,5))

        for i in range(ETtime):
            ETdata[i,:] = (ETfreq_start+(i)*ETslope,ETtime_start+i,ETra,ETdec,ETpow)    

        data2 = np.concatenate((data1,ETdata),axis=0)
        return data2

def get_flags(X, bin_val=30, freqbin_val=10, twoD_only=True):
    """
    Parameters:
    X: input array, of shape (num_data_points, 2)
    bin_val: adjustable parameter for cutoff
    freqbin_val: another adjustable parameter for cutoff

    Return:
    flags: dense region flags, Boolean array of length X.shape[0] 
    flags_: sparse region flags, Boolean array of length X.shape[0]
    cflags: complementary flags, Boolean array of length X.shape[0]
    """
    allindices, alldists, meandists, klist = KNN(X, 8, srcDims=2)
    binmax  = np.amax(meandists)/2
    counts, bins = np.histogram(meandists, range=(0, binmax), bins=100)
    cutoff = bins[bin_val]
    water_flags = meandists<cutoff

    if twoD_only: 
        return water_flags
    else:
        allindices, alldists, meandists, klist = KNN(X, 8, srcDims=1)
        binmax  = np.amax(meandists)/8
        counts, bins = np.histogram(meandists, range=(0, binmax), bins=100)
        cutoff = bins[freqbin_val]
        freq_flags = meandists<cutoff

        broadband_flags = water_flags ^ freq_flags
        
        flags = water_flags | freq_flags

        return flags

def make_plots(data1, flags, flags_, figname):
    data_candidate = data1.loc[(data1['remove'] == 0) & (data1['cluster']>0)]
    data_clean = data1.loc[data1['cluster']==0]
    #data_broad = data1.loc[data1['remove']==2]

    f, axes = plt.subplots(1,2, figsize=(16, 8))
    axes[0].scatter(data1['freq'][flags_],data1['time'][flags_],color='r',marker='.',s=2)
    axes[0].scatter(data1['freq'][flags],data1['time'][flags],color='b',marker='.',s=2)
    axes[0].set_title('knn selection')

    axes[1].scatter(data_clean['freq'],data_clean['time'],color='r',marker='.',s=2)
    axes[1].scatter(data_candidate['freq'],data_candidate['time'],color='g',marker='.',s=2)
    axes[1].set_title('Clean and Candidate')
    plt.savefig(figname)
    #plt.show()

    if False:
        print "generating pairplot"
        data_dense = data1.loc[flags]
        g = sns.pairplot(data_dense, hue='cluster', vars=['freq','time','ra','dec'],
            plot_kws={"s":10})
        for i, j in zip(*np.triu_indices_from(g.axes, 1)):
            g.axes[i, j].set_visible(False)



def process(queue, hist_bin_cut=30, nbins=200, maxsep=16, maxwidth=1):

    file_dir = os.path.dirname(infile)
    fname = infile.split('/')[-1].split('.')[0] #e.g. "20170604_172322"
    print "########## "+ fname+" ###########"

    data1 = np.loadtxt(infile)
    if False:
        data1 = injectET(data1, 1352, 500)
    data1 = pd.DataFrame(data=data1, columns=['freq','time','ra','dec','pow'])
    X = whiten(zip(data1['freq'], data1['time']))

    flags_bool = get_flags(X, hist_bin_cut, 5, twoD_only=True) 
    flags = np.where(flags_bool)[0]
    flags_ = np.where(~flags_bool)[0]


    data1['cluster'] = 0
    data1['remove'] = 0

    data_dense = data1.loc[flags]
    Xd = data_dense['freq']


    nbins = 200
    counts, bin_edges = np.histogram(Xd, bins=nbins)
    cluster_labels = label(counts>0)
    for i in xrange(nbins):
        if cluster_labels[i] > 0:
            cur_bin = (data1['freq'] >= bin_edges[i]) & (data1['freq'] < bin_edges[i+1])
            data1.loc[cur_bin & flags_bool, 'cluster'] = cluster_labels[i]
    ncluster = np.amax(cluster_labels)
    #print np.unique(data1['cluster'])
    

    for i in np.unique(data1['cluster']):
        if i == 0:
            continue
        cluster = data1.loc[data1['cluster'] == i]
        loc = SkyCoord(cluster['ra'],cluster['dec'],unit='deg',frame='icrs')
        sep = loc[0].separation(loc[:])
        clustersep = np.amax(sep.deg)
        print i, clustersep
        if clustersep > 16.:
            data1.loc[data1['cluster'] == i, 'remove'] = 1
        elif np.amax(cluster['freq'])-np.amin(cluster['freq']) > 1:
            data1.loc[data1['cluster'] == i, 'remove'] = 2
        else:
            print 'Candidate cluster {}'.format(i)


    data_candidate = data1.loc[(data1['remove'] == 0) & (data1['cluster']>0)]
    data_clean = data1.loc[data1['cluster']==0]
    data1.to_csv(data_dir+fname.split('.')[0]+".knn")
    figname = file_dir+'/'+fname+".knn.png"
    #import IPython; IPython.embed()
    make_plots(data1, flags, flags_, figname)


if __name__ == "__main__":

    data_dir = '/data1/SETI/SERENDIP/vishal/'
    #data_dir = '/home/yunfanz/Projects/SETI/serendip/Data/'
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    print "Process files in " + data_dir
    files = find_files(data_dir, pattern='*.dbase.drfi.clean.exp_time')


    for infile in files:
        process(infile)
    
        
        