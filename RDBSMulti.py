import cmasher as cmr
import numpy as np
import matplotlib.pyplot as plt
import HicSunt as hic
import sys
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import RobustScaler
import DBCV as valid
from collections import Counter

""""
RDBS (A, B, ukn, overlist, code, bst)
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
bst: int or tuple
- when passing an integer, will pull the cell with that index from the SelectCell output, 0 being the Ideal Cell. When passing a tuple, will instead pull the cell at those coordinates.
""""

##CONSTANTS
ukn=-5 #No. of unknowns in the dataset, a negative number please
#They *need* to be at the end of the alphabetised Sims list.
#Just put a Z in front of their name or something.

def GetTEL(name, nbands=10, SNR=5):
    telstat=hic.telstat[hic.teltype.index(name)]
    #this is going to return [*win,RP,RT]
    #print(telstat)
    rt=telstat[2]
    sd=14
    subres=telstat[1]
    win=telstat[0]    
    bands=hic.Bandwidth(telstat[0], nbands)
    tray, sims = hic.LoadTray(rt, sd, nbands, subres, win, SNR) #Load the correct Tray.
    return(bands, tray, sims, SNR)

def GetOL(tray, source, code):
    #print("Fetching Overlay...")
    if source=="barcode":
        #print('Overlay type is "barcode".')
        ol=tray[code]
        cmap=None
        if code==0:
            label=hic.types[ol]
            c=hic.colours[ol]
            marker=hic.markers[ol%(len(hic.markers))]
        elif code==1:
            label=hic.clouds[ol]
            c=hic.cldcol[ol]
            marker='D'
        elif code==2:
            label=hic.spectype[ol]
            c=hic.spcol[ol]
            marker='*'
        else:
            sys.exit("Code {0} not valid. \n Correct values are: Atmospheres (0), Clouds (1), Spectral Types (2).".format(code))
    if source=="prior":
        #print('Overlay type is "prior".')
        if code>20: sys.exit("Code {0} out of range. Valid codes are:".format(code) +
                             "\n0: Temperature, 1: T_eff, 2: Flux, 3: SemJ, 4: R, 5: P" +
                             "\n6: O2, 7: H2O, 8: CO, 9: CO2, 10: O3, 11: N2, 12: CH4 | Bottom of Atmosphere" +
                             "\n13: O2.1, 14: H2O.1, 15: CO.1, 16: CO2.1, 17: O3.1, 18: N2.1, 19: CH4.1 |Top of Atmosphere"
                             "\n20: O2/CO2")
        elif code==20: #Will only do O2/CO2
            ol=(tray[6]/tray[9])
        else:
            ol=tray[code]
        label=hic.traycols[code]
        c=ol
        cmap=None
        marker='o'
    
    return(np.array([ol, label, c, marker, cmap]))

def RDBS (A, B, ukn, overlist, code, bst):
    #First, go fetch the trays and store their data.
    bandsA, TrayA, simsA, SNRA  = GetTEL(A)
    bandsB, TrayB, simsB, SNRB = GetTEL(B)
    noA=len(bandsA) #x, vertical
    noB=len(bandsB) #y, horizontal 
        
    ##HEALTH CHECK FUNCTIONS##
    if len(TrayA)!=len(TrayB): sys.exit("Trays are not the same size, please check your sample population.")
    if ukn>0: ukn = ukn*-1
    ##If these fail, something is wrong and the program won't try to run. 
    #So fix these issues first, then the rest can run and waste your time in a different way
    
    ##Input
    #Takes either a single integer for "Best Cell" or a tuple for a specific cell    
    if isinstance(bst, int)==True:
        #cell[$1][row index of SelectCell, usually 0:best][$0 or 1]
        print("Method: Auto SelectCell")
        dmv='C:/Users/Lyan/StAtmos/HSD/Test/Trays/Multi/{0}{1}v{2}{3}_S{4}-{5}{6}.dmv'.format(A,noA,B,noB,SNRA,overlist,code)
        cell=hic.SelectCell(dmv)
        v=int(cell[1][bst][0]) 
        h=int(cell[1][bst][1])
    else:
        print("Method: Manual Cell ID")
        v=bst[0]
        h=bst[1]

    print('Calculations in Progress. Please wait...')
    fig, ax = plt.subplot_mosaic("A;B") #remember to put A;B back in afterwards.
    plt.subplots_adjust(left=0.1, right=0.82, top=0.88, bottom=0.11)
    fig.set_size_inches(9,7);

    #Sample the trays
    vA=[]
    hB=[]
    sims=[]
    overlay=np.zeros((1,4), dtype='object')
    arch=np.zeros((1,4), dtype='object')

    for s in range(0,len(TrayA)):#for star in tray
        #both trays Should be the same size
        for p in range(0,TrayA[s].shape[2]): #for atm in planet
            vA.append(TrayA[s][v,0,p]) #[star][band, "area", atm] 
            hB.append(TrayB[s][h,0,p])
            sims.append(s)
            
            arch=np.row_stack((arch,GetOL(TrayA[s][0,1,p], "barcode", 0)[:-1])) 
            if overlist in ["spec", "clouds", "bar"] and code!=0:
                overlay=np.row_stack((overlay,GetOL(TrayA[s][0,1,p], "barcode", code)[:-1]))                    
            elif overlist=="atlas":
                overlay=np.row_stack((overlay,GetOL(TrayA[s][0,2,p], "prior", code)[:-1]))
            else:
                sys.exit("{0} is an invalid overlay.".format(overlist)+
                         "\nValid lists are: arch, spec, clouds and atlas")
    #Store samples and prevent algorithmic indigestion
    vals= np.column_stack((vA, hB))
    vals= RobustScaler().fit_transform(vals)
    overlay=np.delete(overlay,0,0)
    arch=np.delete(arch,0,0)
    
    for w in range(0,len(overlay)+ukn): #Main Run
        ax["A"].scatter(vals[w,0],vals[w, 1], s=80,
                    c=arch[w,2], marker=arch[w,3], zorder=5)
        ax["A"].annotate(str(simsA[sims[w]]), xy=(vals[w,0], vals[w,1]), xytext=(1.5, 1.5), textcoords='offset points', fontsize=8)
    for w in range(len(overlay)+(ukn+1), len(overlay)): #Unknowns
        ax["A"].scatter(vals[w,0],vals[w, 1], s=80,
                        c="Magenta", marker=hic.umrk[w%(len(hic.umrk))], zorder=5)
    if ukn!=0: ax["A"].scatter(vals[ukn,0],vals[ukn, 1], c="Cyan", s=80, zorder=10, marker=hic.umrk[1%(len(hic.umrk))])

    
    if ukn!=0: #Using 0 for values that are the same throughout, a waste of memory, I know.
        if code>=6: barscl='log' 
        else: barscl='linear'
        cmap = cmr.get_sub_cmap(*hic.gcl[code])
        temp=ax["B"].scatter(vals[:ukn,0],vals[:ukn, 1], s=15,
                        c=overlay[:ukn,2], cmap=cmap, marker=overlay[0,3], zorder=5, norm=barscl)
        ax["B"].scatter(vals[ukn+1:,0],vals[ukn+1:, 1], s=15, 
                        c= "Magenta", marker= '+', zorder=10)
        ax["B"].scatter(vals[ukn,0],vals[ukn, 1], s=15,
                        c="Cyan", marker='+', zorder=10)
    else:
        temp=ax["B"].scatter(vals[:,0],vals[:, 1], s=15,
                        c=overlay[:,2], cmap=cmap, marker=overlay[:,3], zorder=5)


    mpts=0.02
    mcls=0.15
    neareps= 0.3-(0.25/SNRA) if SNRA!=None else 0.3
    db = HDBSCAN(cluster_selection_epsilon=neareps, min_samples=round(vals.shape[0]*mpts), min_cluster_size=round(vals.shape[0]*mcls), 
                 algorithm='brute', store_centers='both').fit(vals)
    labels = db.labels_
    unique_labels = set(labels)
    centroids=db.centroids_

                    
    score =[]
    color = [plt.cm.tab20b(each) for each in np.linspace(0, 1, len(unique_labels))]
    #Creates a colourmap for the size of our dataset
    for k, col in zip(unique_labels, color):
        class_member_mask = labels == k
        if k == -1:
            nr=len(vals[class_member_mask])
            xy = vals[class_member_mask]
            ax["B"].scatter(xy[:,0],xy[:, 1],c='Grey', marker='x', s=20, zorder=1)
        else:
            sd=np.nanstd(np.copy(overlay[:,0])[class_member_mask])
            mn=np.mean(np.copy(overlay[:,0])[class_member_mask])
            try:
                eval=(sd/mn)*100 
                score="{0:.2f}%".format(eval)
            except:
                score="0%"            
            xy = vals[class_member_mask]
            ctr=hic.DrawContour(xy, col, False)
            ax["B"].fill(ctr[0], ctr[1], '--', c=col, alpha=0.4, zorder=1) 
            ax["B"].annotate(score, xy=(centroids[k,0],centroids[k, 1]), 
                             xytext=(-k*3,+25), textcoords='offset points', fontsize=20, zorder=7)
    
    ## PLOT LAYOUT
    label = hic.traycols[code]
    if  code < 6:
        fl = hic.StripAlnum(label)[:3]
    elif 6 <= code < 20:
        fl = hic.StripAlnum(label)        
        label=label+" Mixing Ratio"
    else:
        fl = hic.StripAlnum(label)
        label="Normalised "+label+" Ratio"

    print("Calculating {0} Validity...".format(fl))
    validity=valid.DBCV(vals, labels)
    try: 
        nrp = (nr/vals.shape[0])*100 
    except: 
        nrp = 0
    fig.text(x=0.5, y=0.92, s= "DBCV: {0:.4f}".format(validity), fontsize=8, ha="center")    
    fig.text(x=0.5, y=0.9, s= "Noise Ratio: {0:.2f}%".format(nrp), fontsize=8, ha="center")    
    fig.text(x=0.88, y=0.9, s= "P{0:.0f}% C{1:.0f}% E{2:.2f}".format(mpts*100, mcls*100, neareps), fontsize=8, ha="right")#""" 
    
    ax["B"].set_ylabel('Scaled Area {0:.3f}-{1:.3f} $um$'.format(bandsA[v][0], bandsA[v][1]), fontsize=12);
    ax["B"].set_xlabel('Scaled Area {0:.3f}-{1:.3f} $um$'.format(bandsB[h][0], bandsB[h][1]), fontsize=12);
    filename="MASCMulti {0}v{2}_S{4}-{5}{6}({1},{3})".format(A,v,B,h,SNRA,overlist,code)
    fig.suptitle(filename);
    ax["A"].annotate('a)',xy=(0, 1), xycoords='axes fraction', xytext=(+0.5, -0.5), 
                     textcoords='offset fontsize', fontsize='medium', verticalalignment='top')
    ax["B"].annotate('b)',xy=(0, 1), xycoords='axes fraction', xytext=(+0.5, -0.5), 
                     textcoords='offset fontsize', fontsize='medium', verticalalignment='top')    
    plt.colorbar(temp, ax=[ax["A"],ax["B"]], anchor=(1.51,0), shrink=0.45, label=str(label))
    plt.show()
    return(filename)
  

RDBS("MIRI", "MIRI", 5, "atlas", 20, bst=(2,6))
