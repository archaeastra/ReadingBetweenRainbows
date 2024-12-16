import numpy as np
import matplotlib.pyplot as plt
import HicSunt as hic
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import RobustScaler
import DBCV as valid
import cmasher as cmr

"""
WHAT THIS NEEDS TO DO:
- take a 2D array of xy points from the Tray
- run DBSCAN on those points
- plot over the OG plot with a cluster mask
- profit
"""

#UPLOAD TABLE
nbands=10
win=(5, 12)  #Make sure this is set correctly, and check save location.
##WINDOWS
#MIRI 5-12 NIRSPEC 1-5.3 | ECLIPS_NIR 1-2 _VIS 0.515-1.03  _NUV 0.2-0.5125
#0.515 is important, do NOT use 0.5
bands=hic.Bandwidth(win, nbands)
subres=280
tray, sims=hic.TrayTable("[YOUR LOCAL PATH HERE]/VPL Transits", bands, subres, win)

##Input
desiredX=2  #Usually it's x2y6 for MIRI stuff
desiredY=6
#target=10
#0: Temperature, 1: Flux, 2: Semimajor axis, 3: Radius, 4: Pressure
#5: O2, 6: H2O, 7: CO, 8: CO2, 9: O3, 10: N2, 11: CH4 
#12: O2/CO2. 13:H2O/CO2. Need to input math manually below, but this'll do the formatting.

for target in range(0, 12):
    vert=[]
    hori=[]
    zmark= []
    sim=[]
    T=[]
    S=[]
    R=[]
    for s in range(0,len(tray)):
        for t in range(0,tray[s].shape[2]):
            vert.append(tray[s][desiredY,0,t])
            hori.append(tray[s][desiredX,0,t])
            zmark.append(tray[s][0,1,t])
            sim.append(s)
            #If you need a ratio (range 12+), uncomment S and force T to be the correct value.
            T.append(tray[s][0,2,t][target])
            #S.append(tray[s][0,2,t][8])
    vals= np.column_stack((hori, vert))
    
    
    fig, ax = plt.subplot_mosaic("A;B") #remember to put A;B back in afterwards if you change it.
    plt.subplots_adjust(left=0.1, right=0.82, top=0.88, bottom=0.11)
    fig.set_size_inches(9,12);
    for j in range(0,vals.shape[0]):
        m=int(zmark[j])
        ax["A"].scatter(vals[j,0], vals[j,1], label=hic.types[m], s=150,
                        c=hic.colours[m], marker=hic.markers[m%(len(hic.markers))])
        ax["A"].annotate(str(sims[sim[j]]), xy=(vals[j,0], vals[j,1]), xytext=(1.5, 1.5), textcoords='offset points', fontsize=8)
        #Above prints names
    ax["A"].set_ylabel('Area {0:.1f}-{1:.1f} $um$'.format(bands[desiredY][0], bands[desiredY][1]), fontsize=12);
    ax["A"].set_xlabel('Area {0:.1f}-{1:.1f} $um$'.format(bands[desiredX][0], bands[desiredX][1]), fontsize=12);
    
    #Normalising the data for DBSCAN to digest better
    vals= RobustScaler().fit_transform(vals)
    gasratio=[]
    for em in range(len(T)):
        gasratio.append(T[em]/1)#S[em])  ##Uncomment as necessary of ratios
    #Running DBSCAN
    #Note: eps is the maximum distance between points to consider them in a group
    mpts=0.02
    mcls=0.15
    neareps=0.3
    db = HDBSCAN(cluster_selection_epsilon=neareps, min_samples=round(vals.shape[0]*mpts), min_cluster_size=round(vals.shape[0]*mcls), 
                 algorithm='brute', store_centers='both').fit(vals)
    labels = db.labels_
    medoids = db.medoids_
    centroids=db.centroids_
    error=db.probabilities_
    unique_labels = set(labels)
    #plot=ax["B"].scatter(vals[:,1], vals[:,0], c='k')
    ax["B"].set_ylabel('Scaled Area {0:.1f}-{1:.1f} $um$'.format(bands[desiredY][0], bands[desiredY][1]), fontsize=12);
    ax["B"].set_xlabel('Scaled Area {0:.1f}-{1:.1f} $um$'.format(bands[desiredX][0], bands[desiredX][1]), fontsize=12);
    fig.suptitle("Clustered Concentration Ratio on Area - VPLSE", fontsize=12); 
    ax["A"].annotate('a)',xy=(0, 1), xycoords='axes fraction', xytext=(+0.5, -0.5), 
                     textcoords='offset fontsize', fontsize='medium', verticalalignment='top')
    ax["B"].annotate('b)',xy=(0, 1), xycoords='axes fraction', xytext=(+0.5, -0.5), 
                     textcoords='offset fontsize', fontsize='medium', verticalalignment='top')
       
    color = [plt.cm.tab20b(each) for each in np.linspace(0, 1, len(unique_labels))]
    cmap = cmr.get_sub_cmap(*hic.gcl[target])
    #Creates a colourmap for the size of our dataset
    for k, col in zip(unique_labels, color):
        #Links a unique label to a colour
        #k is the stand in for label, col for colour
        class_member_mask = labels == k
        #isolate which cluster we're doing right now so we can seek that label in the data
        err = error[class_member_mask]
        m='o'
        label="_Cluster "+str(k)
        if k == -1:
            # Colour used for noise.
            col = 'Grey'#[0,0,0,1]
            err=1
            m='x'
            label="Noise"
            nrat=len(vals[class_member_mask])
            xy = vals[class_member_mask]
            ax["B"].scatter(xy[:,0],xy[:, 1],c=col, alpha=err, marker=m, s=50, label=label, zorder=10) #this makes the points for noise values
        else:
            xy = vals[class_member_mask]
            #print(xy.shape, labels[class_member_mask].shape)
            #ax["B"].scatter(xy[:,0],xy[:, 1],c=col, alpha=err, marker=m, s=50, label=label)
            hic.DrawContour(xy, col)
    
    msk = [(el==0) for el in T]
    ax["B"].scatter(vals[:,0][msk],vals[:, 1][msk], c="k", alpha=0.2, s=50) #this makes the grey dots for 0 values
    #ax["B"].scatter(vals[:,0],vals[:, 1],c='red', s=50, zorder=5, alpha=0.2) #backup main scatter with a single color for troubleshooting
    if target<5:
        barscl='linear'
        for j in range(0,vals.shape[0]):
            ax["B"].annotate('{0:.1f}'.format(gasratio[j]), xy=(vals[j,0], vals[j,1]), xytext=(2, 2), textcoords='offset points', fontsize=6)#"""

    else:
        barscl='log'
        for j in range(0,vals.shape[0]):
            ax["B"].annotate('{0:.0E}'.format(gasratio[j]), xy=(vals[j,0], vals[j,1]), xytext=(2, 2), textcoords='offset points', fontsize=6)#"""
    temp=ax["B"].scatter(vals[:,0],vals[:, 1],c=gasratio, s=50, cmap=cmap, zorder=5, norm=barscl) #this is the main scatterplot
    ax["B"].scatter(vals[-1,0],vals[-1, 1], c="Magenta", s=55, zorder=10)
         
    #ax["B"].scatter(medoids[:,0],medoids[:, 1],c='Cyan', marker='*', label="Medoids",s=20, zorder=1)
    #ax["B"].scatter(centroids[:,0],centroids[:, 1],c='Magenta', marker='*', label="Centroids",s=20, zorder=1)
    
    label = hic.traycols[target]
    if  target < 5:
        fl=label[:3]
    elif 5 <= target < 12:
        fl = hic.StripAlnum(label)        
        label=label+" Mixing Ratio"
    else:
        fl = hic.StripAlnum(label)
        label=label+" Ratio"

    print("Calculating {0} Validity...".format(fl))
    validity=valid.DBCV(vals, labels)
    try: 
        nrp = (nrat/vals.shape[0])*100 
    except: 
        nrp = 0
    fig.text(x=0.5, y=0.92, s= "DBCV: {0:.4f}".format(validity), fontsize=8, ha="center")    
    fig.text(x=0.5, y=0.9, s= "Noise Ratio: {0:.2f}%".format(nrp), fontsize=8, ha="center")    
    fig.text(x=0.88, y=0.9, s= "P{0:.0f}% C{1:.0f}% E{2:.1f}".format(mpts*100, mcls*100, neareps), fontsize=8, ha="right")    
    fig.text(x=0.12, y=0.9, s= "R: {0}".format(subres), fontsize=8, ha="left")    
    
    plt.colorbar(temp, ax=[ax["A"],ax["B"]], anchor=(1.51,0), shrink=0.45, label=str(label))
    plt.show()  #necessary to ensure sizing is correct, at least on my system
    fig.savefig("C:/Users/Lyan/StAtmos/HSD/Plots/Clustering/NIRSPEC/{0}x{0}_{1}{2}{3}_{4}.png".format(nbands, desiredX, desiredY, str(subres), fl))#,  
                    #bbox_inches="tight", dpi=600)    

