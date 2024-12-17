import os
import matplotlib.pyplot as plt
import HicSunt as hic
from matplotlib.ticker import FuncFormatter
#Used for Bandplot
import cmasher as cmr
import numpy as np

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
def format_axis(x, pos):
    return(f'{x/1e-4:.2f}') #Doesn't put the 10^-4 at the top make that happen    

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
        fig.set_size_inches(6,6);
        fig.tight_layout(pad=4, w_pad=4);
        #fig.suptitle("VPL Transits - {0:s}".format(f), fontsize=12);
        plt.subplots_adjust(top=0.979,
                            bottom=0.082,
                            left=0.139,
                            right=0.98,
                            hspace=0.118,
                            wspace=0.172)
        
        ax["A"].set_ylabel("Transit Depth \n (Rp/Rs)$^{2}$ (10$^{-4}$)", fontsize=10);
        ax["A"].yaxis.set_major_formatter(FuncFormatter(format_axis))
        #ax["A"].set_xlabel("Wavelength $um$", fontsize=10);      
        ax["B"].set_ylabel("Fourier Smoothed \n Normalised TD \n R~1250", fontsize=8);
        ax["C"].set_ylabel("Fourier Smoothed NTD \n R~300", fontsize=8);
        ax["D"].set_ylabel("Fourier Smoothed NTD \n R~60", fontsize=8);
        #"""
        ax["A"].set_xticks([])
        ax["B"].set_xticks([])
        ax["C"].set_xticks([])        
        ax["D"].set_xlabel("Wavelength ($\mu$m)", fontsize=10);#"""      
        
        #A=hic.Normalise(tot[i,0][0]) #Remember, this permanently alters the array!! Comment out if desire raw
        x=tot[i,0][1]   
        win=(1, 5.3)  #Make sure this is set correctly.
        ##WINDOWS
        #MIRI 5-12 NIRSPEC 1-5.3 | ECLIPS_NIR 1-2 _VIS 0.515-1.03  _NUV 0.2-0.5125
        #ya, xa = hic.Window(x, A, win)
        rya, rxa=hic.Window(x, tot[i,0][0], win)
        
        F,newx=hic.Unresolve(tot[i,0][0],tot[i,0][1], win, 300)  #Maximum without being Full. MIRIM MRS is 2k-3k
        FA,newxA=hic.Unresolve(tot[i,0][0],tot[i,0][1], win, 100) #Middle Ground
        FB,newxB=hic.Unresolve(tot[i,0][0],tot[i,0][1], win, 50) #MIRI LRS
        
        z = tot[i,1]
        z=int(z)
        #SpeedPlot(["A","B","C","D"], label=hic.types[z], clr=hic.colours[z]) #Used for Degrade
        #SpeedPlot(["A","B"], label=hic.types[z], clr=hic.colours[z]) #Used for BandPlot.
        
        #"""
        mosaic = ["A","B","C","D"]
        x = [rxa, newx, newxA, newxB]
        y= [rya, F, FA, FB]
        h = [1, h1, h2, h3]
        for p in mosaic:
            ax[str(p)].plot(x[mosaic.index(p)], y[mosaic.index(p)], label=hic.types[z], c=hic.colours[z] if int(h[mosaic.index(p)])==1 else 'k')
            #ax[str(p)].set_xlim(*win)#"""
      
        """      #Used for Bandplot.
        nbands=5
        bands=hic.Bandwidth((5,12), nbands)
        #bands = [(5,6),(6,7),(7,9),(9,11),(11,12)]    
        cmap = cmr.get_sub_cmap('cmr.gem', 0.1, 0.9)
        color = [cmap(each) for each in np.linspace(0, 1, nbands)]
        for b, col in zip(bands, color):
            plt.axvspan(b[0], b[1], color=col, alpha=1)#0.3)#"""
        
        ax["A"].annotate('a)',xy=(0, 1), xycoords='axes fraction', xytext=(+0.5, -0.5), 
                 textcoords='offset fontsize', fontsize=16, verticalalignment='top')
        ax["B"].annotate('b)',xy=(0, 1), xycoords='axes fraction', xytext=(+0.5, -0.5), 
                 textcoords='offset fontsize', fontsize=16, verticalalignment='top')
        #"""
        ax["C"].annotate('c)',xy=(0, 1), xycoords='axes fraction', xytext=(+0.5, -0.5), 
                 textcoords='offset fontsize', fontsize=16, verticalalignment='top')
        ax["D"].annotate('d)',xy=(0, 1), xycoords='axes fraction', xytext=(+0.5, -0.5), 
                 textcoords='offset fontsize', fontsize=16, verticalalignment='top')#"""
        
        #hic.PareDown(ax["A"], (0.65,0.95))
        
        fig.savefig("C:/Users/Lyan/StAtmos/HSD/Plots/Indiv/FourierAuto/1000/{0:s}-{1:s}_catch.png".format(f, pans[i]), bbox_inches="tight", dpi=600)
    
        #plt.show()
        plt.close()
print("Program Conclusion, check figures!")
