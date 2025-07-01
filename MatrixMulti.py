import cmasher as cmr
import numpy as np
import matplotlib.pyplot as plt
import HicSunt as hic
import sys
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import RobustScaler
import DBCV as valid
from collections import Counter

"""
This function code create a multi-regime matrix with a specified overlay and ouputs a .dmv text file with header:
   X Y | DBCV NR  RDP/cluster

_How to use:
hic.RDBSM (A, B, ukn, overlist="arch", code=0, display=False)
A: str
- a telescope from hic.teltype
B: str
- a telescope from hic.teltype
ukn: negative integer
- the number of "unknowns" in the dataset, to highlight as magenta. These *must* be at the bottom of the file tree.
overlist: "arch", "spec", "clouds", "bar", or "atlas"
- the type of desired overlay, in order: atmospheric archetype (aka "bar" 0), spectral type (aka "bar" 1), clouds (aka "bar" 2), barcode (a catchall for the previous three), and a prior from the .atl file.
code: int, [0:20]
- The specific code of the desired prior within the list of overlays
display: True/False)
- outputs a graphic display, or not.
"""
                                     "\nValid lists are: arch, spec, clouds and atlas")
                #Store samples and prevent algorithmic indigestion
                
                vals= np.column_stack((vA, hB))
                vals= RobustScaler().fit_transform(vals)
                overlay=np.delete(overlay,0,0)
                
                #sys.exit("check overlay")
                if display==True:
                    if overlist in ["arch", "spec", "clouds", "bar"]:
                        for w in range(0,len(overlay)+ukn): #Main Run
                            ax[v,h].scatter(vals[w,0],vals[w, 1], s=15,
                                        c=overlay[w,2], marker=overlay[w,3], zorder=5)
                            #ax[v,h].annotate(overlay[w,0], xy=(vals[j,0], vals[j,1]), xytext=(1.5, 1.5), textcoords='offset points', fontsize=8)
                        for w in range(len(overlay)+ukn, len(overlay)): #Unknowns
                            ax[v,h].scatter(vals[w,0],vals[w, 1], s=15,
                                            c="Magenta", marker=hic.umrk[w%(len(hic.umrk))], zorder=5)
                    else:    
                        if code>=6: barscl='log' 
                        else: barscl='linear'
                        cmap = cmr.get_sub_cmap(*hic.gcl[code])
                        if ukn!=0: #Using 0 for values that are the same throughout, a waste of memory, I know.
                            ax[v,h].scatter(vals[:ukn,0],vals[:ukn, 1], s=15,
                                            c=overlay[:ukn,0], cmap=cmap, marker=overlay[0,3], zorder=5, label=overlay[:ukn,1], norm=barscl)
                            ax[v,h].scatter(vals[ukn+1:,0],vals[ukn+1:, 1], s=15, 
                                            c= "Magenta", marker= '+', zorder=10)
                            ax[v,h].scatter(vals[ukn,0],vals[ukn, 1], s=15,
                                            c="Cyan", marker='+', zorder=10)
                            #ax[v,h].annotate(overlay[:ukn,0], xy=(vals[:ukn,0], vals[:ukn,1]), xytext=(1.5, 1.5), textcoords='offset points', fontsize=8)
                        else:
                            ax[v,h].scatter(vals[:,0],vals[:, 1], s=15,
                                            c=overlay[:,0], cmap=cmap, marker=overlay[0,3], zorder=5)

                #Running DBSCAN
                mpts=0.02
                mcls=0.15
                neareps= 0.3-(0.25/SNRA) if SNRA!=None else 0.3
                db = HDBSCAN(cluster_selection_epsilon=neareps, min_samples=round(vals.shape[0]*mpts), min_cluster_size=round(vals.shape[0]*mcls), 
                             algorithm='brute', store_centers='both').fit(vals)
                labels = db.labels_
                unique_labels = set(labels)
                                
                score =[]
                color = [plt.cm.tab20b(each) for each in np.linspace(0, 1, len(unique_labels))]
                #Creates a colourmap for the size of our dataset
                for k, col in zip(unique_labels, color):
                    class_member_mask = labels == k
                    if k == -1:
                        nr=len(vals[class_member_mask])
                        if display==True:
                            xy = vals[class_member_mask]
                            ax[v,h].scatter(xy[:,0],xy[:, 1],c='Grey', marker='x', s=20, zorder=1)
                    else:
                        sd=np.nanstd(np.copy(overlay[:,0])[class_member_mask])
                        mn=np.mean(np.copy(overlay[:,0])[class_member_mask])
                        
                        if display==True:
                            xy = vals[class_member_mask]
                            ctr=hic.DrawContour(xy, col, False)
                            ax[v,h].fill(ctr[0], ctr[1], '--', c=col, alpha=0.4, zorder=1) 

                        try:
                            score.append((sd/mn)*100) 
                            #score.append("{0:.2f}".format(sdmn))
                        except:
                            score.append(0)
                cv = valid.DBCV(vals, labels)
                validity[v,h,0]= cv if np.isfinite(cv) else 1
                try: 
                    validity[v,h,1] = (nr/vals.shape[0])*10 
                except: 
                    validity[v,h,1] = 0
                #print(score)
                try:
                    validity[v,h,2]=np.min(score)
                    validity[v,h,3]=np.average(score)
                except:
                    pass
                #Saving to file                
                if outfile==True:
                    with open(dmvf, 'a') as dmv:
                        if v==1 and h==0: print("#X Y | DBCV NR  Ave.RDP  Min.RDP - ({0}{1})".format(overlist, code), file=dmv)
                        if -0.4<validity[v,h,0]<0.2 and validity[v,h,1]<1:
                            print(v,h,str("| {0:.2f} {1:.2f} {2:.2f} {3:.2f}".format(validity[v,h,0],validity[v,h,1],validity[v,h,3], 
                                                                                 validity[v,h,2])), file=dmv)
                                                         #', '.join(validity[v,h,2]))), file=dmv)
    if display==True: 
        fig.suptitle("MASCMulti {0}{1}v{2}{3}_S{4}-{5}{6}".format(A,noA,B,noB,SNRA,overlist,code));
        plt.show()
    return(dmvf)

def Redux(select, numcells=None):
    #select=hic.SelectCell(RDBSM("NIRSpec", "ECLIPS-VIS", 5, 'bar', code=1, display=False))
    #modal=(Counter(sel[:,0])+Counter(sel[:,1])).most_common(2)
    if isinstance(numcells, int)!=True: numcells=select.shape[0]
    modeA=Counter(select[:,0]).most_common(2)
    modeB=Counter(select[:,1]).most_common(2)
    #print("Redux Results: ")
    ideal=[]
    modal=[]
    for num in range(numcells):
        ideal.append("({0:.0f},{1:.0f})".format(select[num,0], select[num,1]))
    for i in range(2):
        modal.append("{0:.0f}, {1:.0f}".format(modeA[i][0], modeB[i][0]))
    return(', '.join(ideal), '; '.join(modal))    

def RDPPull(dmvf):
    #rdp = ak.Array(dmvf)
    rdp= np.genfromtxt(dmvf, dtype='float', comments='#', delimiter=" ", usecols= (0,1,5,6), missing_values="", filling_values="nan")
    mavindex = np.argmin(rdp[:,2]) 
    mindex = np.argmin(rdp[:,3])
    #print(mindex)
    #return("Min.RDP {0} @({1:.0f},{2:.0f})".format(rdp[mindex,2],rdp[mindex,0],rdp[mindex,1]))
    return(rdp[mindex,2],(int(rdp[mavindex,0]),int(rdp[mavindex,1])), rdp[mindex,3],(int(rdp[mindex,0]),int(rdp[mindex,1]))) #returns (mav, (x,y), min, (x,y))
