import numpy as np
import matplotlib.pyplot as plt
import HicSunt as hic
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import RobustScaler
import DBCV as valid
import string

"""
WHAT THIS NEEDS TO DO:
- run Integrator 2
- apply DBSCAN to each subplot
- plot
"""
#DEFINITIONS
nbands=10
win=(5,12)
bands=hic.Bandwidth(win, nbands)
subres=933

#UPLOAD TABLE
tray, sims=hic.TrayTable("[YOUR LOCAL PATH HERE]/VPL Transits", bands, subres, win)
#Pull in the Tray

fig, ax = plt.subplots(nbands,nbands, subplot_kw=dict(box_aspect=1));
fig.set_size_inches(9,12);
plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9, hspace=0.265, wspace=0.175)
fig.suptitle("VPL Transits - Clustered Integration Bands");    

validity=np.zeros((nbands, nbands,2), dtype='float')

for l in range(0,nbands):
    ax[l,0].set_ylabel("{0:.1f}-\n{1:.1f} $um$".format(bands[l][0], bands[l][1], fontsize=4));
    ax[l,nbands-1].yaxis.set_label_position("right")
    ax[l,nbands-1].set_ylabel(list(string.ascii_uppercase)[l], fontsize=15, rotation=0, labelpad=10)
    for m in range(0,nbands):
        ax[nbands-1,m].set_xlabel("{0:.1f}-\n{1:.1f} $um$".format(bands[m][0], bands[m][1], fontsize=4));    
        ax[0,m].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
        ax[0,m].set_title(m, fontsize=15)
        
        clr='k'
        ax[l, m].set_xticks([])
        ax[l, m].set_yticks([])

        if l<=m:
            ax[l,m].set_facecolor('k')
            ax[l,m].set_xticks([])
            ax[l,m].set_yticks([])

        else:
            #Sample the tray
            vert=[]
            hori=[]
            zmark= []
            for s in range(0,len(tray)):#for planet in tray
                for t in range(0,tray[s].shape[2]): #for atm in planet
                    vert.append(tray[s][l,0,t])
                    hori.append(tray[s][m,0,t])
                    zmark.append(tray[s][l,1,t]) #x can be anything here, they're all the same per collumn.
            
            vals= np.column_stack((hori, vert))      #store sample from tray for plotting
            vals= RobustScaler().fit_transform(vals) #<-- Algorithmic indigestion preventer.
            
            for w in range(len(zmark)):
                z=zmark[w]
                ax[l,m].scatter(vals[w,0],vals[w, 1], s=15,
                                c=hic.colours[z], marker=hic.markers[z%(len(hic.markers))], zorder=11-z)
            tempe=['Magenta', 'Lime']
            """
            #This section includes real data, if available, which it currently doesn't.
            for obs in range(0,np.shape(RL)[2]):
                for nt in range(0,np.shape(RL)[1]):
                    ax[l,m].scatter(RL[m, nt, obs], RL[l, nt, obs], s=obs+10, c=tempe[obs], zorder=11, alpha=0.25)#"""
                
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
            
            color = [plt.cm.tab20b(each) for each in np.linspace(0, 1, len(unique_labels))]
            #Creates a colourmap for the size of our dataset
            for k, col in zip(unique_labels, color):
                class_member_mask = labels == k
                err = error[class_member_mask]
                mk='o'
                label="_Cluster "+str(k)
                if k == -1:
                    # Colour used for noise.
                    col = 'Grey'
                    err=1
                    mk='x'
                    label="Noise"
                    nrat=len(vals[class_member_mask])
                    xy = vals[class_member_mask]
                    ax[l,m].scatter(xy[:,0],xy[:, 1],c=col, alpha=err, marker=mk, s=20, label=label, zorder=1)
                else:
                    xy = vals[class_member_mask]
                    ctr=hic.DrawContour(xy, col, False)
                    ax[l,m].fill(ctr[0], ctr[1], '--', c=col, alpha=0.4) 

            #ax[l,m].scatter(medoids[:,0],medoids[:, 1],c='Cyan', marker='*', label="Medoids")
            #ax[l,m].scatter(centroids[:,0],centroids[:, 1],c='Magenta', marker='*', label="Centroids")
            cv = valid.DBCV(vals, labels)
            if np.isfinite(cv):
                validity[l,m,0]=cv
            else:
                #print("NaN found. Setting to 0.")
                validity[l,m,0]=1
            validity[l,m,1]=(nrat/vals.shape[0])*100
        #Pull out valid cells for later analysis, human oversight necessary, just in case.
        if -0.4<validity[l,m,0]<0.15 and validity[l,m,1]<10 and l>m:
            print(l,m,str("({2}{3}) DBCV {0:.2f}, NR {1:.2f}".format(validity[l,m,0],validity[l,m,1], list(string.ascii_uppercase)[l], m)))
 
fig.text(x=0.88, y=0.92, s= "P{0:.0f}% C{1:.0f}% E{2:.1f}".format(mpts*100, mcls*100, neareps), fontsize=8, ha="right")    
fig.text(x=0.12, y=0.92, s= "Signal Reduction: {0}".format(subres), fontsize=8, ha="left")    
#hic.PareDown(ax[2,0], (3,3))  #This creates the legend.
plt.show()#"""


    
