import numpy as np
import matplotlib.pyplot as plt
import HicSunt as hic
from sklearn.cluster import HDBSCAN
import DBCV as valid
from sklearn.preprocessing import RobustScaler

"""
WHAT THIS DOES:
- take a plot, extract the x-y points
- run DBSCAN on those points
- plot over the OG plot with a cluster mask
"""
#DEFINITIONS
#THIS IS HR DIAGRAM DATA:
#UPLOAD TABLE
maxp=400

try:
    print("Pulling...")
    pull = np.genfromtxt("C:/Users/Lyan/StAtmos/HSD/HRD Star Data/hyglike_from_athyg_24.csv", 
                         dtype='float', delimiter=',', usecols=(14, 16), skip_header=1, max_rows=maxp, 
                         invalid_raise=False)
except:
    print("Pull Error, check spacing! Terminating Program...")
    pass

maskC = np.isfinite(pull[:,1])
pull=pull[maskC]
maskM = np.isfinite(pull[:,0])
pull=pull[maskM]

fig, ax = plt.subplot_mosaic("A;B")
fig.set_size_inches(9,12);
fig.suptitle("AT-HYG Subset v2.4", fontsize=12);      
plt.subplots_adjust(left=0.1, right=0.82, top=0.88, bottom=0.11)

ax["A"].scatter(pull[:,1], pull[:,0], c='k', label="Stars")
ax["A"].set_ylabel('Absolute Magnitude', fontsize=12);
ax["A"].set_xlabel('Colour Index', fontsize=12);
ax["A"].invert_yaxis()

#Normalising the data for DBSCAN to digest better
pull= RobustScaler().fit_transform(pull)

plot=ax["B"].scatter(pull[:,1], pull[:,0], c='k', alpha=0)
#need this to get the scaled values, or else dbscan can't digest it.
ax["B"].set_ylabel('Scaled Absolute Magnitude', fontsize=12);
ax["B"].set_xlabel('Scaled Colour Index', fontsize=12);
ax["B"].invert_yaxis()#"""

#Extracting X and Y (again), inefficient, but only useful in this dataset due to its structure.
vals = plot.get_offsets()

#Running DBSCAN
#Note: eps is the maximum distance between points to consider them in a group: cluster_selection_epsilon=0.3, 
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
for k, col in zip(unique_labels, color):
    class_member_mask = labels == k
    err = error[class_member_mask]
    m='o'
    label="Cluster "+str(k+1)
    
    if k == -1:
        # Colour used for noise.
        col = 'Grey'
        err=1
        m='x'
        label="Noise"
        nrat=len(vals[class_member_mask])

    xy = vals[class_member_mask] #& core_samples_mask]
    ax["B"].scatter(xy[:,0],xy[:, 1],color=col, alpha=err, marker=m, s=50, label=label)
    

ax["B"].scatter(medoids[:,0],medoids[:, 1],c='Cyan', marker='*', label="Medoids")
ax["B"].scatter(centroids[:,0],centroids[:, 1],c='Magenta', marker='*', label="Centroids")
    
print("Calculating Validity...")
validity=valid.DBCV(vals, labels)
fig.text(x=0.5, y=0.92, s= "DBCV: {0:.4f}".format(validity), fontsize=8, ha="center")    
fig.text(x=0.5, y=0.9, s= "Noise Ratio: {0:.2f}%".format((nrat/vals.shape[0])*100), fontsize=8, ha="center") 
fig.text(x=0.88, y=0.9, s= "P{0:.0f}% C{1:.0f}% E{2:.1f}".format(mpts*100, mcls*100, neareps), fontsize=8, ha="right")    
fig.text(x=0.12, y=0.9, s= "Number of Points: {0}".format(maxp), fontsize=8, ha="left")    
   
hic.PareDown(ax["A"], (1.01,1))
hic.PareDown(ax["B"], (1.01,1))

plt.show()#"""
