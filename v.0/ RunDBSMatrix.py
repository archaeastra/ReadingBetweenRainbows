import numpy as np
import matplotlib.pyplot as plt
import HicSunt as hic
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import RobustScaler
import DBCV as valid
import string
import sys

#This code not only creates the large clustering matrices used to assess bands in MASC, but also now assesses them autmatically.
#Whether you plot this or not, it will creat a .dmv file containing the valuation of each cell, and print the best ones to console for human verification.

#DEFINITIONS
nbands=10
win=(5, 12)  #Make sure this is set correctly.
##WINDOWS
#MIRI 5-12 NIRSPEC 1-5.3 | ECLIPS_NIR 1-2 _VIS 0.515-1.03  _NUV 0.2-0.5125 | LUMOS 0.1-1 >0.27 start to dodge nan sections in AD Leo

bands=hic.Bandwidth(win, nbands)
#bands=[(5,5.5),(5.5,6),(6,7),(7,8),(8,9),(9,10),(10,10.5),(10.5,11),(11,11.5),(11.5,12)]
#bands = [(5,6),(6,7),(7,9),(9,11),(11,12)] #MIRI manual
#bands = [(0.27,0.3),(0.3,0.4),(0.4,0.6),(0.6,0.7),(0.7,1)] #LUMOS manual 2 and 3
#bands = [(0.27,0.4),(0.4,0.6),(0.6,0.7),(0.7,0.9),(0.9,1)] #LUMOS manual 4

subres=None
ext= '.trn'
colu = 2

#UPLOAD TABLE
#Pull in the Tray
tray, sims = hic.LoadTray(ext, colu, nbands, subres, win)

fig, ax = plt.subplots(nbands,nbands, subplot_kw=dict(box_aspect=1));
fig.set_size_inches(8,8);
plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9, hspace=0.265, wspace=0.175)

validity=np.zeros((nbands, nbands,3), dtype='float')
print('Calculations in Progress. Please wait...')
for l in range(0,nbands):
    ax[l,0].set_ylabel("{0:.1f}-\n{1:.1f} $um$".format(bands[l][0], bands[l][1], fontsize=4));
    ax[l,nbands-1].yaxis.set_label_position("right")
    ax[l,nbands-1].set_ylabel(list(string.ascii_uppercase)[l], fontsize=15, rotation=0, labelpad=10)
    for m in range(0,nbands):
        #l is y, m is x, remember this!
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
                    vert.append(tray[s][l,0,t]) #[star][band, "area", atm] 
                    hori.append(tray[s][m,0,t])
                    zmark.append(tray[s][l,1,t]) #[star][band, "barcode", atm]
                                            
            vals= np.column_stack((hori, vert))      #store tray sample for plotting
            vals= RobustScaler().fit_transform(vals) #<-- Algorithmic indigestion preventer.
            
            for w in range(len(zmark)):
                z=zmark[w][0] #0 atm, 1 cld
                ax[l,m].scatter(vals[w,0],vals[w, 1], s=15, #This is atmosphere archetypes
                                c=hic.colours[z], marker=hic.markers[z%(len(hic.markers))], zorder=11-z)#"""
                """ax[l,m].scatter(vals[w,0],vals[w, 1], s=15, #This is cloud types
                                c=hic.cldcol[z], marker=hic.markers[z%(len(hic.markers))], zorder=11-z)#"""
            
            #"""
            #Running DBSCAN
            #Note: eps is the maximum distance between points to consider them in a group
            #eps_func=nbands/(bands[m][1] - bands[m][0])
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
                #Links a unique label to a colour
                #k is the stand in for label, col for colour
                class_member_mask = labels == k
                #isolate which cluster we're doing right now so we can seek that label in the data
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
                    nrat=0
                    #stdev=np.nanstd(xy)
            
            cv = valid.DBCV(vals, labels)
            if np.isfinite(cv):
                validity[l,m,0]=cv
            else:
                #print("NaN found. Setting to 0.")
                validity[l,m,0]=1
            validity[l,m,1]=(nrat/vals.shape[0])*100#"""
            #validity[l,m,2]=(stdev)
        
        ax[l,m].set_title(str("DBCV {0:.2f} \n NR {1:.2f}".format(validity[l,m,0],validity[l,m,1])), fontsize=5, color='r')
        dmvf='C:/Users/Lyan/StAtmos/HSD/Test/Trays/{0}-{1}x{2}_{3}({4}){5}.dmv'.format(win[0], win[1], nbands, ext[1:], colu, subres)
        with open(dmvf, 'a') as dmv:
            if l==0 and m==0: print("#X Y (Cell) DBCV    NR    SD", file=dmv)
            if -0.4<validity[l,m,0]<0.15 and validity[l,m,1]<10 and l>m:
                print(l,m,str("({3}{4}) {0:.2f} {1:.2f} {2:.2f}".format(validity[l,m,0],validity[l,m,1], validity[l,m,2],list(string.ascii_uppercase)[l], m)), file=dmv)
                

fig.text(x=0.88, y=0.96, s= "P{0:.0f}% C{1:.0f}% E{2:.1f}".format(mpts*100, mcls*100, neareps), fontsize=10, ha="right")    
fig.text(x=0.12, y=0.96, s= "R: {0}".format(subres), fontsize=10, ha="left")    
#hic.PareDown(ax[2,0], (3,3))  #Legend optional

print('Calculation Complete. Evaluating .dmv results...')
print("Selected:")
sel=hic.SelectCell(dmvf, 5)
for x in range(sel[1].shape[0]):
    print("{0:s} DB {1:.2f}  NR {2:.2f}".format(sel[0][x], sel[1][x,2], sel[1][x,3]))
    
#plt.show() #Do not plot anything larger than 12x12, it won't be visible.

    
