import numpy as np
import os
import matplotlib.pyplot as plt
import HicSunt as hic
import cmasher as cmr

"""
WHAT THIS DOES:
- Find the TRN parser file, load the Total line.
- cut up the x axis into slices
- find the y at start and end of slice, take the midpoint, integrate under that line.
- add all the slices together.

NOTES:
This creates both the Degrade and BandPlot figures. 
Commented within are which sections need to be swapped out to swap between the two modes.
"""
## DEFINITIONS
def SpeedPlot(sub, label=None, clr='k', marker='*'):
    x = [xa, newx, newxA, newxB] #Used for Degrade
    y= [ya, F, FA, FB]
    #x = [xa, newx]  #Used for BandPlot
    #y= [ya, F]
    ax[str(sub[0])].plot(x[0], y[0], label=label, c=clr)
    ax[str(sub[1])].plot(x[1], y[1], label=label, c=clr);
    ax[str(sub[2])].plot(x[2], y[2], label=label, c=clr); #Used for Degrade
    ax[str(sub[3])].plot(x[3], y[3], label=label, c=clr); #Used for Degrade
#This definition exists to avoid repeating code.
    
#MAIN CODE
path = "[Your Local Path Here]/VPL Transits"

try:
    sims = sorted(next(os.walk(os.path.join(path,'.')))[1])
except StopIteration:
    pass

for f in sims:
    os.chdir(path + "/" + f)  
    mile=os.getcwd()
    tot, pans= hic.Extract3D(mile, 2, ".trn")
    print("Current Position: ", f)
    for i in range(len(pans)):
        fig, ax = plt.subplot_mosaic("A;B;C;D")
        fig.set_size_inches(8,8);
        fig.tight_layout(pad=4, w_pad=4);
        fig.suptitle("VPL Transits - {0:s}".format(f), fontsize=12);
        plt.subplots_adjust(top=0.898,
                            bottom=0.132,
                            left=0.127,
                            right=0.902,
                            hspace=0.452,
                            wspace=0.172)
        ax["A"].set_ylabel("Normalised Flux $(Jy)$", fontsize=10);
        ax["B"].set_xlabel("Wavelength $um$", fontsize=10);      
        ax["B"].set_ylabel("Fourier Smoothed NF \n$(Jy)$ Sub 300", fontsize=8);
        ax["C"].set_ylabel("Fourier Smoothed NF \n$(Jy)$ Sub 100", fontsize=8);
        ax["D"].set_ylabel("Fourier Smoothed NF \n$(Jy)$ Sub 50", fontsize=8);
        
        A=hic.Normalise(tot[i,0][0])
        x=tot[i,0][1]   
        small=min(x)
        win=(5, 12)  #Make sure this is set correctly, (5,12) is the MIRI-like window. (1,5.3) is the NIRSPEC-lik window.
        ya, xa = hic.Window(x, A, win)
        
        F,newx=hic.Unresolve(tot[i,0][0],tot[i,0][1], win, 300)  #Maximum without being Full. MIRIM MRS is 2k-3k
        FA,newxA=hic.Unresolve(tot[i,0][0],tot[i,0][1], win, 100) #Middle Ground
        FB,newxB=hic.Unresolve(tot[i,0][0],tot[i,0][1], win, 50) #MIRI LRS
        
        z = tot[i,1]
        z=int(z)
        SpeedPlot(["A","B","C","D"], label=hic.types[z], clr=hic.colours[z]) #Used for Degrade
        #SpeedPlot(["A","B"], label=hic.types[z], clr=hic.colours[z]) #Used for BandPlot.
        
        """      #Used for Bandplot.
        nbands=5
        bands=hic.Bandwidth((5,12), nbands)
        #bands = [(5,6),(6,7),(7,9),(9,11),(11,12)]    
        cmap = cmr.get_sub_cmap('cmr.gem', 0.1, 0.9)
        color = [cmap(each) for each in np.linspace(0, 1, nbands)]
        for b, col in zip(bands, color):
            plt.axvspan(b[0], b[1], color=col, alpha=1)#0.3)#"""
        
        hic.PareDown(ax["B"], (0.65,0.95))

        #fig.savefig("./{0:s}-{1:s}_IndivID.png".format(f, pans[i]), bbox_inches="tight", dpi=600)
 
        plt.show()
print("Program Conclusion, check figures!")
