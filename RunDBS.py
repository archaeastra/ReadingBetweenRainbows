import numpy as np
import matplotlib.pyplot as plt
import HicSunt as hic
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import RobustScaler
import DBCV as valid
import cmasher as cmr

#UPLOAD TABLE
nbands=10
win=(5, 12)  #Make sure this is set correctly, and check save location.
##WINDOWS
#MIRI 5-12 NIRSPEC 1-5.3 | ECLIPS_NIR 1-2 _VIS 0.515-1.03  _NUV 0.2-0.5125 | LUMOS 0.1-1
#0.515 is important, do NOT use 0.5

bands=hic.Bandwidth(win, nbands)
#bands=[(5,5.5),(5.5,6),(6,7),(7,8),(8,9),(9,10),(10,10.5),(10.5,11),(11,11.5),(11.5,12)]
#bands = [(5,6),(6,7),(7,9),(9,11),(11,12)] #MIRI manual
#bands = [(0.27,0.3),(0.3,0.4),(0.4,0.6),(0.6,0.7),(0.7,1)] #LUMOS manual

subres=None
ext= '.trn'
colu = 2
tray, sims = hic.LoadTray(ext, colu, nbands, subres, win)
dmv='C:/Users/Lyan/StAtmos/HSD/Test/Trays/{0}-{1}x{2}_{3}({4}){5}.dmv'.format(win[0], win[1], nbands, ext[1:], colu, subres)
cell=hic.SelectCell(dmv, 1)
print("Best Cell: ", cell[0][0])
bst = 0
#best cell selector, keep 0 unless you want a specific cell from SelectCell

##Input
#cell[$1][row index of SelectCell, usually 0==best][$0 or 1]
desiredX=int(cell[1][bst][1]) 
desiredY=int(cell[1][bst][0])
#X: Type, 0: Temperature, 1: Flux, 2: Semimajor axis, 3: Radius, 4: Pressure
#5: O2, 6: H2O, 7: CO, 8: CO2, 9: O3, 10: N2, 11: CH4 
#12: O2/CO2. 13:H2O/CO2. Need to input math manually below, but this'll do the formatting.

for target in range(0,12):
    vert=[]
    hori=[]
    zmark= []
    sim=[]
    T=[]
    S=[]
    SD=[]
    for s in range(0,len(tray)):
        for t in range(0,tray[s].shape[2]):
            vert.append(tray[s][desiredY,0,t])
            hori.append(tray[s][desiredX,0,t])
            zmark.append(tray[s][0,1,t])
            sim.append(s)
            T.append(tray[s][0,2,t][target]) #remember to check if "target" for non-ratio runs.
            S.append(tray[s][0,2,t][8])
    vals= np.column_stack((hori, vert))
    
    
    fig, ax = plt.subplot_mosaic("A;B")
    plt.subplots_adjust(left=0.1, right=0.82, top=0.88, bottom=0.11)
    fig.set_size_inches(9,7);
    for j in range(0,vals.shape[0]):
        m=int(zmark[j][0]) #0atm 1cld
        ax["A"].scatter(vals[j,0], vals[j,1], label=hic.types[m], s=150, #atm archs
                        c=hic.colours[m], marker=hic.markers[m%(len(hic.markers))])#"""
        """ax["A"].scatter(vals[j,0], vals[j,1], label=hic.clouds[m], s=150, #cld archs
                        c=hic.cldcol[m], marker=hic.markers[m%(len(hic.markers))])#"""
        ax["A"].annotate(str(sims[sim[j]]), xy=(vals[j,0], vals[j,1]), xytext=(1.5, 1.5), textcoords='offset points', fontsize=8)
        #Above prints names
    ax["A"].set_ylabel('Area {0:.3f}-{1:.3f} $um$'.format(bands[desiredY][0], bands[desiredY][1]), fontsize=12);
    ax["A"].set_xlabel('Area {0:.3f}-{1:.3f} $um$'.format(bands[desiredX][0], bands[desiredX][1]), fontsize=12);
    
    #Normalising the data for DBSCAN to digest better
    vals= RobustScaler().fit_transform(vals)
    gasratio=[]
    for em in range(len(T)):
        gasratio.append(T[em]/S[em])
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

    ax["B"].set_ylabel('Scaled Area {0:.3f}-{1:.3f} $um$'.format(bands[desiredY][0], bands[desiredY][1]), fontsize=12);
    ax["B"].set_xlabel('Scaled Area {0:.3f}-{1:.3f} $um$'.format(bands[desiredX][0], bands[desiredX][1]), fontsize=12);
    fig.suptitle("Clustered Concentration Ratio on Area {0}x{0}{1:s}{2}- VPLSE".format(nbands,cell[0][bst], subres), fontsize=12); 
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
            col = 'Grey'
            err=1
            m='x'
            label="Noise"
            nrat=len(vals[class_member_mask])
            xy = vals[class_member_mask]
            ax["B"].scatter(xy[:,0],xy[:, 1],c=col, alpha=err, marker=m, s=50, label=label, zorder=10) #this makes the points for noise values
        else:
            xy = vals[class_member_mask]
            gassd=np.nanstd(np.copy(gasratio)[class_member_mask])
            mn=np.mean(np.copy(gasratio)[class_member_mask])
            #the percentage score here is plotted to provide an "at a glance" value of standard deviation versus values contained
            score=(gassd/mn)*100
            hic.DrawContour(xy, col)
            ax["B"].annotate("{0:.2f}%".format(score), 
                             xy=(centroids[k,0],centroids[k, 1]), 
                             xytext=(-k*3,+25), textcoords='offset points', fontsize=20, zorder=7)
            
    msk = [(el==0) for el in T]
    ax["B"].scatter(vals[:,0][msk],vals[:, 1][msk], c="k", alpha=0.2, s=50) #this makes the grey dots for 0 values
    if target<5:
        barscl='linear' #Annotates for non-gas values
        for j in range(0,vals.shape[0]):
            ax["B"].annotate('{0:.2f}'.format(gasratio[j]), xy=(vals[j,0], vals[j,1]), xytext=(2, 2), textcoords='offset points', fontsize=6)#"""
    else:
        barscl='log'
        for j in range(0,vals.shape[0]):  #Annotates for gas values
            ax["B"].annotate('{0:.0E}'.format(gasratio[j]), xy=(vals[j,0], vals[j,1]), xytext=(2, 2), textcoords='offset points', fontsize=6)#"""
    temp=ax["B"].scatter(vals[:,0],vals[:, 1],c=gasratio, s=50, cmap=cmap, zorder=5, norm=barscl) #this is the main scatterplot
    ax["B"].scatter(vals[-1,0],vals[-1, 1], c="Magenta", s=55, zorder=10)
             
    label = hic.traycols[target]
    if  target < 5:
        fl=label[:3]
    elif 5 <= target < 13:
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
    
    hic.PareDown(ax["A"], (1.01,1))
    plt.colorbar(temp, ax=[ax["A"],ax["B"]], anchor=(1.51,0), shrink=0.45, label=str(label))
    plt.show()
    fig.savefig("C:/Users/Lyan/StAtmos/HSD/Plots/Metric/LUMOS/albedo/{0}x{0}_{1}{2}{3}_{4}.png".format(nbands, desiredX, desiredY, str(subres), fl))#,
    
