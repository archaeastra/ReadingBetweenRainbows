import numpy as np
import matplotlib.pyplot as plt
import HicSunt as hic
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import RobustScaler
import DBCV as valid
import cmasher as cmr

"""
WHAT THIS NEDS TO DO:
- take a 2D array of xy points from the Tray
- run DBSCAN on those points
- plot over the original with a cluster mask
"""

#UPLOAD TABLE
nbands=10
bands=hic.Bandwidth((5,12), nbands) #1 to 5.3 for NIRSPEC
subres=300
tray, sims=hic.TrayTable("[YOUR LOCAL PATH HERE]/VPL Transits", bands, subres, win=(5,12)) #1 to 5.3 for NIRSPEC 

##Input
desiredX=2
desiredY=6

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
        zmark.append(tray[s][0,1,t]) #x can be anything here, they're all the same per collumn.
        sim.append(s)
        T.append(tray[s][0,2,t][5])
        S.append(tray[s][0,2,t][8])
        #0: Temperature, 1: Flux, 2: Semimajor axis, 3: Radius, 4: Pressure
        #5: O2, 6: H2O, 7: CO, 8: CO2, 9: O3, 10: N2, 11: CH4 
vals= np.column_stack((hori, vert))

fig, ax = plt.subplot_mosaic("A;B") #remeber to put A;B back in afterwards.
fig.set_size_inches(9,12);
plt.subplots_adjust(left=0.1, right=0.82, top=0.88, bottom=0.11)
for j in range(0,vals.shape[0]):
    m=int(zmark[j])
    ax["A"].scatter(vals[j,0], vals[j,1], label=hic.types[m], 
                    c=hic.colours[m], marker=hic.markers[m%(len(hic.markers))])
    ax["A"].annotate(str(sims[sim[j]]), xy=(vals[j,0], vals[j,1]), xytext=(1.5, 1.5), textcoords='offset points', fontsize=5)
ax["A"].set_ylabel('EqW {0:.1f}-{1:.1f} $um$'.format(bands[desiredY][0], bands[desiredY][1]), fontsize=12);
ax["A"].set_xlabel('EqW {0:.1f}-{1:.1f} $um$'.format(bands[desiredX][0], bands[desiredX][1]), fontsize=12);
#ax["A"].invert_yaxis()#"""

#Normalising the data for DBSCAN to digest better
vals= RobustScaler().fit_transform(vals)
gasratio=[]
for em in range(len(T)):
    gasratio.append(T[em]/1)#S[em])
    #Remove the 1)# to swap between single molecule or ratio calculations.

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
ax["B"].set_ylabel('Scaled EqW {0:.1f}-{1:.1f} $um$'.format(bands[desiredY][0], bands[desiredY][1]), fontsize=12);
ax["B"].set_xlabel('Scaled EqW {0:.1f}-{1:.1f} $um$'.format(bands[desiredX][0], bands[desiredX][1]), fontsize=12);
#for j in range(0,vals.shape[0]):
    #ax["B"].annotate('{0:.0E}'.format(T[j]), xy=(vals[j,0], vals[j,1]), xytext=(2, 2), textcoords='offset points', fontsize=6)
#The above notes down the mixing ratios beside each point, uncomment for this feature.
fig.suptitle("Clustered Concentration Ratio on Area - VPLSE", fontsize=12); 
ax["A"].annotate('a)',xy=(0, 1), xycoords='axes fraction', xytext=(+0.5, -0.5), 
                 textcoords='offset fontsize', fontsize='medium', verticalalignment='top')
ax["B"].annotate('b)',xy=(0, 1), xycoords='axes fraction', xytext=(+0.5, -0.5), 
                 textcoords='offset fontsize', fontsize='medium', verticalalignment='top')
   
color = [plt.cm.tab20b(each) for each in np.linspace(0, 1, len(unique_labels))]
cmap = cmr.get_sub_cmap('pink_r', 0.2, 1)
#Creates a colourmap for the size of the dataset
for k, col in zip(unique_labels, color):
    #Links a unique label to a colour
    #k is the stand in for label, col for colour
    class_member_mask = labels == k
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
        ax["B"].scatter(xy[:,0],xy[:, 1],c=col, alpha=err, marker=m, s=50, label=label, zorder=10)
    else:
        xy = vals[class_member_mask]
        hic.DrawContour(xy, col)

msk = [(el<10000) for el in T]
ax["B"].scatter(vals[:,0],vals[:, 1], c="k", alpha=0.2, s=50)
temp=ax["B"].scatter(vals[:,0][msk],vals[:, 1][msk],c=gasratio, s=50, cmap=cmap, zorder=5, norm='log')
ax["B"].scatter(vals[-1,0],vals[-1, 1], c="Magenta", s=55, zorder=10)

print("Calculating Validity...")
validity=valid.DBCV(vals, labels)
try: 
    nrp = (nrat/vals.shape[0])*100 
except: 
    nrp = 0
fig.text(x=0.5, y=0.92, s= "DBCV: {0:.4f}".format(validity), fontsize=8, ha="center")    
fig.text(x=0.5, y=0.9, s= "Noise Ratio: {0:.2f}%".format(nrp), fontsize=8, ha="center")    
fig.text(x=0.88, y=0.9, s= "P{0:.0f}% C{1:.0f}% E{2:.1f}".format(mpts*100, mcls*100, neareps), fontsize=8, ha="right")    
fig.text(x=0.12, y=0.9, s= "Signal Reduction: {0}".format(subres), fontsize=8, ha="left")    

hic.PareDown(ax["A"], (1.01,1))
plt.colorbar(temp, ax=[ax["A"],ax["B"]], anchor=(1.51,0), shrink=0.45, label="CO2 Concentration (rel. frac.)")
plt.show()


