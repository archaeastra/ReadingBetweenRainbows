import numpy as np
import matplotlib.pyplot as plt
import HicSunt as hic
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import RobustScaler
import DBCV as valid
import cmasher as cmr

"""
WHAT THIS DOES:
- take a 2 2D arrays of xy points from the Tray, ensuring to keep them ordered and labelled
- divide y/y and x/x
- show
"""
#UPLOAD TABLE
nbands=10
bands=hic.Bandwidth((5,12), nbands) #1 to 5.3 for NIRSPEC

subres=300
tray, sims=hic.TrayTable("C:/Users/Lyan/StAtmos/HSD/Test/VPL Transits", bands, subres, win=(5,12)) #1 to 5.3 for NIRSPEC

##Input
XA=6
YA=9
XB=4
YB=6

#Holding lists
vertA=[]
horiA=[]
vertB=[]
horiB=[]
zmark= []
sim=[]
T=[]
S=[]
for s in range(0,len(tray)):
    for t in range(0,tray[s].shape[2]):
        vertA.append(tray[s][YA,0,t])
        horiA.append(tray[s][XA,0,t]) 
        vertB.append(tray[s][YB,0,t]) 
        horiB.append(tray[s][XB,0,t]) 
        zmark.append(tray[s][0,1,t]) #x can be anything here, they're all the same per collumn.
        sim.append(s)
        T.append(tray[s][0,2,t][5])
        S.append(tray[s][0,2,t][8])
        #0: Temperature, 1: Flux, 2: Semimajor axis, 3: Radius, 4: Pressure
        #5: O2, 6: H2O, 7: CO, 8: CO2, 9: O3, 10: N2, 11: CH4 
        
valsA= np.column_stack((horiA, vertA))
valsB= np.column_stack((horiB, vertB))
vals=(valsA/valsB)

gasratio=[]
for em in range(len(T)):
    gasratio.append(T[em]/S[em])

fig, ax = plt.subplot_mosaic("AC;BB")
fig.set_size_inches(9,12);
plt.subplots_adjust(top=0.905,
                    bottom=0.11,
                    left=0.08,
                    right=0.825,
                    hspace=0.2,
                    wspace=0.23)
cmap = cmr.get_sub_cmap('cmr.iceburn_r', 0, 1)
for j in range(0,vals.shape[0]):
    m=int(zmark[j])
    ax["A"].scatter(valsA[j,0], valsA[j,1], label=hic.types[m], 
                    c=hic.colours[m], marker=hic.markers[m%(len(hic.markers))])
    ax["A"].annotate(str(sims[sim[j]]), xy=(vals[j,0], vals[j,1]), xytext=(1.5, 1.5), textcoords='offset points', fontsize=5)
    ax["C"].scatter(valsB[j,0], valsB[j,1], label=hic.types[m], 
                    c=hic.colours[m], marker=hic.markers[m%(len(hic.markers))])
    ax["C"].annotate(str(sims[sim[j]]), xy=(vals[j,0], vals[j,1]), xytext=(1.5, 1.5), textcoords='offset points', fontsize=5)
    ax["B"].scatter(vals[j,0], vals[j,1], label=hic.types[m], zorder=3, s=20,
                    c=hic.colours[m], marker=hic.markers[m%(len(hic.markers))])
ax["A"].set_ylabel('EqW {0:.1f}-{1:.1f} $um$'.format(bands[YA][0], bands[YA][1]), fontsize=12);
ax["A"].set_xlabel('EqW {0:.1f}-{1:.1f} $um$'.format(bands[XA][0], bands[XA][1]), fontsize=12);
ax["C"].set_ylabel('EqW {0:.1f}-{1:.1f} $um$'.format(bands[YB][0], bands[YB][1]), fontsize=12);
ax["C"].set_xlabel('EqW {0:.1f}-{1:.1f} $um$'.format(bands[XB][0], bands[XB][1]), fontsize=12);
ax["B"].set_ylabel('Scaled EqW {0:.1f}/{1:.1f} $um$'.format(bands[YA][0], bands[YB][0]), fontsize=12);
ax["B"].set_xlabel('Scaled EqW {0:.1f}/{1:.1f} $um$'.format(bands[XA][0], bands[XB][0]), fontsize=12);

ax["B"].scatter(vals[:,0],vals[:, 1], c="k", alpha=0.2, s=50, zorder=1)
temp=ax["B"].scatter(vals[:,0],vals[:, 1], c=gasratio, s=50, cmap=cmap, zorder=1, norm='log', alpha=0.8)
ax["B"].scatter(vals[-1,0],vals[-1, 1], c="Magenta", s=55, zorder=10)


fig.text(x=0.5, y=0.92, s= "Signal Reduction: {0}".format(subres), fontsize=8, ha="center")    
hic.PareDown(ax["C"], (1.01,1))

plt.colorbar(temp, ax=[ax["A"],ax["B"]], anchor=(1.51,0), shrink=0.45, label="O2/CO2 Concentration (rel. frac.)")
plt.show()
