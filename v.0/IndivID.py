import numpy as np
import os
import matplotlib.pyplot as plt
import HicSunt as hic
import cmasher as cmr
from matplotlib.ticker import FuncFormatter
from HicSunt import BBAdjust

"""
This code functions on v.O paradigms, but remains compatible in v.1 as it does not have dependencies outside itself.
Simply ensure the filetree is correct, the destination exists, and run.
"""

## DEFINITIONS
win=(0.515, 1.03)  #Make sure this is set correctly.
##WINDOWS    
#MIRI 5, 12 NIRSPEC 1, 5.3 | 
#ECLIPS_NIR 1, 2 _VIS 0.515, 1.03 _NUV 0.2, 0.5125 | 
subres=140
SNR=5
rt="RFL"

display=False

#MAIN CODE
path = "[YOUR PATH HERE]/VPLSE_Transits"

try:
    sims = sorted(next(os.walk(os.path.join(path,'.')))[1])
except StopIteration:
    pass


for f in sims:
    os.chdir(path + "/" + f)  
    mile=os.getcwd()
    print("Current Position: ", f)
    tot, pans= hic.Extract3D(mile, rt) 
    #trn: 0: wvl, 1: effective radius, 2: Rp/Rs
    #rfl: 0: wvl, 1: wvn, 2: stelflux@top of atm, 3: planet flux, 4: Albedo
    for i in range(len(pans)):
        fig, ax = plt.subplot_mosaic("A;B;C;D")
        fig.set_size_inches(6,6);
        fig.tight_layout(pad=4, w_pad=4);
        #fig.suptitle("{0:s}".format(f), fontsize=12);
        plt.subplots_adjust(top=0.979,
                            bottom=0.082,
                            left=0.155,
                            right=0.935,
                            hspace=0.143,
                            wspace=0.172)
        
        if rt=="TRN":
            ax["A"].set_ylabel("Transit Depth \n (Rp/Rs)$^{2}$ (10$^{-4}$)", fontsize=10);
        elif rt=="RFL":
            ax["A"].set_ylabel("Flux \n (W/m$^{2}$/$\mu$m) (10$^{0}$)", fontsize=10);
        ax["A"].yaxis.set_major_formatter(FuncFormatter(format_axis))
        
        ##GAUSSIAN AND FFT SETS
        ax["B"].set_ylabel("Normalised \n Fourier Smoothing \n R {0} SNR inf".format(subres), fontsize=8);
        ax["C"].set_ylabel("Normalised \n Gaussian Noise \n R inf SNR {0}".format(SNR), fontsize=8);
        ax["D"].set_ylabel("Normalised FFT+Noise \n R {0} SNR {1}".format(subres, SNR), fontsize=8);#"""
        
        ax["A"].set_xticks([])
        ax["B"].set_xticks([])
        ax["C"].set_xticks([])        
        ax["D"].set_xlabel("Wavelength ($\mu$m)", fontsize=10);#"""      
        
        rya, rxa=hic.Window(tot[i,0][1], tot[i,0][0], win)
        if rt=="RFL":
            bby, bb = hic.BBAdjust(tot[i,0][0], tot[i,0][1], tot[i,2][1])
        else:
            bby, bb = tot[i,0][0], 0
        
        ##STANDARD DEGRADED PLOT
        F,newx,h1=hic.Unresolve(bby,tot[i,0][1], win, subres) 
        h2=hic.Unresolve(tot[i,0][0],tot[i,0][1], win, subres)[2] 
        FB,newxB,h3=hic.Unresolve(bby,tot[i,0][1], win, subres)

        ##GAUSSIAN DEGRADED PLOT
        FA,newxA=hic.AddNoise(bby,tot[i,0][1], win, SNR)
        FB,newxB=hic.AddNoise(FB, newxB, win, SNR)
        
        z = tot[i,1][0]
        z=int(z)
        
        mosaic = ["A","B","C","D"]
        x = [rxa, newx, newxA, newxB]
        y= [rya, F, FA, FB]
        h = [1, h1, h2, h3]
        for p in mosaic:
            ax[str(p)].plot(x[mosaic.index(p)], y[mosaic.index(p)], label=hic.types[z], c=hic.colours[z] if int(h[mosaic.index(p)])==1 else 'k')
            ax[str(p)].set_xlim(*win)#"""
            
            if mosaic.index(p)!=0 and isinstance(bb,int)==False:
                ax2=ax[str(p)].twinx()
                ax2.plot(tot[i,0][1], bb, label="Stellar Blackbody", 
                             c=hic.colours[z] if int(h[mosaic.index(p)])==1 else 'k', alpha=0.3, linestyle='--')
                ax2.set_xlim(*win)#"""
                
            if display==True: 
                nbands=10
                bands=hic.Bandwidth(win, nbands)
                cmap = cmr.get_sub_cmap('cmr.gem', 0.1, 0.9)
                color = [cmap(each) for each in np.linspace(0, 1, nbands)]
                for b, col in zip(bands, color):
                    ax[str(p)].axvspan(b[0], b[1], color=col, alpha=0.3)#"""
        
        ax["A"].annotate('a)',xy=(0, 1), xycoords='axes fraction', xytext=(+0.5, -0.5), 
                 textcoords='offset fontsize', fontsize=16, verticalalignment='top')
        ax["B"].annotate('b)',xy=(0, 1), xycoords='axes fraction', xytext=(+0.5, -0.5), 
                 textcoords='offset fontsize', fontsize=16, verticalalignment='top')
        #"""
        ax["C"].annotate('c)',xy=(0, 1), xycoords='axes fraction', xytext=(+0.5, -0.5), 
                 textcoords='offset fontsize', fontsize=16, verticalalignment='top')
        ax["D"].annotate('d)',xy=(0, 1), xycoords='axes fraction', xytext=(+0.5, -0.5), 
                 textcoords='offset fontsize', fontsize=16, verticalalignment='top')#"""
        #ax["E"].plot(tot[i,0][1], bb, label=hic.types[z], c=hic.colours[z] if int(h[mosaic.index(p)])==1 else 'k')

        hic.PareDown(ax["A"], (0.65,0.95))
        fig.savefig("[YOUR PATH HERE]/Indiv14_VIS/{0:s}-{1:s}_sd14.png".format(f, pans[i]), bbox_inches="tight", dpi=600)
    
        #plt.show()
        plt.close()
print("Program Conclusion, check figures!")
