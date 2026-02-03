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

def RDBSM(A, B, ukn, SNR, flag=None, overlist="bar", code=0, display=False, verbose=True, outfile=True):
    #First, go fetch the trays and store their data.
    bandsA, TrayA, simsA, SNRA, rt  = hic.GetTRAY(A, flag, SNR=SNR)
    bandsB, TrayB = hic.GetTRAY(B, flag, SNR=SNR)[:2]

    
    ##HEALTH CHECK FUNCTIONS##
    if len(TrayA)!=len(TrayB): sys.exit("Trays are not the same size, please check your sample population.")
    if ukn>0: ukn = ukn*-1
    if overlist=="bar" and code>3: sys.exit("Code {0} not valid.".format(code) +
                                            "\nCorrect values are: Atmospheres (0), Clouds (1), Spectral Types (2).")
    if overlist not in ["bar", "spec", "clouds", "atlas", "arch", "archetype"]!=True:
        sys.exit("{0} is an invalid overlay.".format(overlist)+
                 "\nValid lists are: arch, spec, clouds, bar and atlas")

    ##If these fail, something is wrong and the program won't try to run. 
    #So fix these issues first, then the rest can run and waste your time in a different way
    
    noA=len(bandsA) #x, vertical
    noB=len(bandsB) #y, horizontal 
    dmvf='C:/Users/Lyan/StAtmos/HSD/Test/Trays/Multi/RBR{7}/{0}{1}v{2}{3}_S{4}-{5}{6}.dmv'.format(A,noA,B,noB,SNRA,overlist,code,flag)
    if outfile==True:
        try:
            open(dmvf, 'w').close() #make sure it's empty before writing to it
        except:
            os.makedirs("C:/Users/Lyan/StAtmos/HSD/Test/Trays/Multi/RBR{0}".format(flag), exist_ok = True)
    
    if display==True:
        fig, ax = plt.subplots(noA,noB, subplot_kw=dict(box_aspect=1));
        fig.set_size_inches(7,7);
        plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9, hspace=0.265, wspace=0.175)
    
    validity=np.zeros((noA,noB,4), dtype='object') #a place to store the validity calculations
    print('Calculations in Progress. Please wait...')

    for v in range(0,noA):
        if verbose==True: print("  Current Row: {0}/{1}".format(v, noA))
        if display==True:
            ax[v,0].set_ylabel("{0:.2f}-\n{1:.2f} $um$".format(bandsA[v][0], bandsA[v][1], fontsize=2));
            ax[v,noA-1].yaxis.set_label_position("right")
            ax[v,noA-1].set_ylabel(v, fontsize=15, rotation=0, labelpad=10)

        for h in range(0,noB):
            if display==True:
                ax[noB-1,h].set_xlabel("{0:.2f}-\n{1:.2f} $um$".format(bandsB[h][0], bandsB[h][1], fontsize=2));
                ax[0,h].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
                ax[0,h].set_title(h, fontsize=15)
        
                ax[v,h].set_xticks([])
                ax[v,h].set_yticks([])
            if A==B and v<=h:
                if display==True:
                    ax[v,h].set_facecolor('k')
                    ax[v,h].set_xticks([])
                    ax[v,h].set_yticks([])
                pass #skip diagonal and transposed cells to save time and energy
            else:               
                #Sample the trays
                vA=[]
                hB=[]
                overlay=np.zeros((1,4), dtype='object')

                for s in range(0,len(TrayA)):#for star in tray
                    #both trays Should be the same size
                    for p in range(0,TrayA[s].shape[2]): #for atm in planet
                        vA.append(TrayA[s][v,0,p]) #[star][band, "area", atm] 
                        hB.append(TrayB[s][h,0,p])
                        
                        if overlist in ["bar", "spec", "clouds"]:
                            overlay=np.row_stack((overlay,hic.GetOL(TrayA[s][0,1,p], "barcode", code)[:-1]))                    
                        elif overlist=="atlas":
                            overlay=np.row_stack((overlay,hic.GetOL(TrayA[s][0,2,p], "prior", code)[:-1]))
                        elif overlist=="arch" or "archetype":
                            overlay=np.row_stack((overlay,hic.GetOL(TrayA[s][0,1,p], "archetype", 0)[:-1]))                
                        else:
                            sys.exit("{0} is an invalid overlay.".format(overlist)+
                                     "\nValid lists are: arch, spec, clouds, bar and atlas")
                #Store samples and prevent algorithmic indigestion
                
                vals= np.column_stack((vA, hB))
                vals= RobustScaler().fit_transform(vals)
                overlay=np.delete(overlay,0,0)
                
                #sys.exit("check overlay")
                if display==True:
                    if A!=B and v==h:
                        ax[v,h].patch.set_edgecolor('black')
                        ax[v,h].patch.set_linewidth(2) 
                        
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
        if verbose == True:
            print(hic.SelectCell(dmvf, 10, verbose=True))
        fig.suptitle("MASCMulti {0}{1}v{2}{3}_S{4}-{5}{6}".format(A,noA,B,noB,SNRA,overlist,code));
        plt.show()
    return(dmvf)

def Redux(select, numcells=None, excludenan=True):
    if excludenan==True:
        select = select[~np.isnan(select).any(axis=1)]    
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
    for i in range(0,2):
        try: modal.append("{0:.0f}, {1:.0f}".format(modeA[i][0], modeB[i][0]))
        except: modal.append("X")
    return(', '.join(ideal), '; '.join(modal))    

def RDPPull(dmvf, excludenan=True):
    #rdp = ak.Array(dmvf)
    rdp= np.genfromtxt(dmvf, dtype='float', comments='#', delimiter=" ", usecols= (0,1,5,6), missing_values="", filling_values="nan")
    if excludenan==True:
        rdp = rdp[~np.isnan(rdp).any(axis=1)]    
    try:
        mavindex = np.argmin(rdp[:,2]) 
        mindex = np.argmin(rdp[:,3])
    except:
        print("no")
        pass
    #print(mindex)
    #return("Min.RDP {0} @({1:.0f},{2:.0f})".format(rdp[mindex,2],rdp[mindex,0],rdp[mindex,1]))
    return(rdp[mindex,2],(int(rdp[mavindex,0]),int(rdp[mavindex,1])), rdp[mindex,3],(int(rdp[mindex,0]),int(rdp[mindex,1]))) 
    #returns (mav, (x,y), min, (x,y))

