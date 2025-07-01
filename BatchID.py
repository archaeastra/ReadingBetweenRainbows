import sys
import string
import matplotlib.pyplot as plt
import numpy as np
import HicSunt as hic
import os
from matplotlib.ticker import FuncFormatter
import cmasher as cmr

""""
How To Use:
-  Nothing special, it ties it to the "batch" flag in RDBSMulti
"""

def Fetch(path, rt):
    try:
        os.chdir(path)
    except:
        pass

    for l in os.listdir('.'):
        if rt == "TRN":            
            if l[-4:] == ".trn":
                spectrum='./'+str(l)
                col=2
            elif l[-4:] == ".rad":
                spectrum='./'+str(l)
                col=4 #1 is Total, 4 is Exoplanet only, 5 is transit.
        elif rt == "RFL":
            if l[-4:] == ".rfl":
                spectrum='./'+str(l)
                col=3 # 2 Stellar, 3 Planetary, 4 Albedo
            elif l[-4:] == ".irr":
                spectrum='./'+str(l)
                col=4 #1 is Total, 4 is Exoplanet only.
        else:
            sys.exit('Run type not specified!\nPlease add "TRN" or "RFL" rt flag to TrayTable.')

        if l[-4:] == str('.atl'):
            stats = './'+str(l)

    try:
        #print("Pulling...")
        pull = np.genfromtxt(str(spectrum), dtype='float', comments='#', filling_values=np.nan, usecols=(0,col))
        Teff = np.genfromtxt(str(stats), dtype='float', comments='#', skip_header=3, usecols=1, max_rows=1, filling_values=np.nan)
        spectype = np.genfromtxt(str(stats), dtype='object', comments='#', max_rows=1, usecols=1, filling_values=np.nan)
    except:
        #print("oops")
        sys.exit("Pull Error in {0:s}! \n Probable causes: Missing .atl, improper headers. \nTerminating Program...".format(spectrum))

    if pull[1,0]<pull[0,0]:
        #print("VPL Source. Flipping Array.")
        pull=np.flip(pull, axis=0)
    else:
        #print("Standard Source. Array Unmodified.")
        pass
    
    for ty in hic.types:
        if str(path.split('\\')[-1]).find(ty)!=-1:
            if str(path.split('\\')[-1])=="ArcheanHaze": 
                pcode=3
            elif str(path.split('\\')[-1])[0]=="Y":
                pcode=1
            else:
                pcode=hic.types.index(str(ty))
            break
        else:
            pcode=1#"""

    return(pull, Teff, pcode)


def BatchID(input, tel, bst, nbands=10, SNR=5):
    print("Fetching spectral lines...")
    rt, seed, subres, win, bands=hic.GetTEL(tel, nbands)

    if input[-1]==0: input=np.delete(input, -1)
    abc=list(string.ascii_uppercase)[:len(input)]
    
    fig, ax = plt.subplot_mosaic(";".join(abc))
    if isinstance(bst,int)==False: fig.suptitle("MASCBatchID {0}_S{1}-({2},{3})".format(tel,SNR,*bst))    
    else: fig.suptitle("MASCBatchID {0}_S{1}-({2})".format(tel,SNR,bst))
    fig.set_size_inches(6,6);
    fig.tight_layout(pad=4, w_pad=4);
    plt.subplots_adjust(top=0.950,
                        bottom=0.082,
                        left=0.070,
                        right=0.935,
                        hspace=0.05,
                        wspace=0.172)
        
    if isinstance(bst,int)==False:
        bdv = [bands[bst[0]], bands[bst[1]]]
    else:
        bdv = [bands[bst]]

    #print(bdv)
    for tile in abc:
        #print("Tile: ", tile)
        ax[tile].set_yticks([])
        ax[tile].annotate(tile,xy=(0, 1), xycoords='axes fraction', xytext=(+0.5, -0.5), 
                         textcoords='offset fontsize', fontsize=16, verticalalignment='top')
        if tile!=abc[-1]: ax[tile].set_xticks([])
        
        #print(input[abc.index(tile)])
        for path in input[abc.index(tile)]:
            #print(path.split('\\')[-1])
            spc, Teff, bc = Fetch(path, rt) 

            y, x=hic.Window(spc[:,0], spc[:,1], win)
            if rt=="RFL":
                bby, bb = hic.BBAdjust(spc[:,1], spc[:,0], Teff)
            else:
                bby, bb = spc[:,1], 0
    
            y,x,h=hic.Unresolve(bby,spc[:,0], win, subres)
            y,x=hic.AddNoise(y, x, win, SNR)
            
            
            ax[tile].plot(x, y, label=hic.types[bc], c=hic.colours[bc], alpha=0.5)
            ax[tile].set_xlim(*win)
                        
            cmap = cmr.get_sub_cmap('cmr.gem', 0.1, 0.9)
            color = [cmap(each) for each in np.linspace(0, 1, len(bdv))]
        for b, col in zip(bdv, color):
            ax[tile].axvspan(b[0], b[1], color=col, alpha=0.2)#"""
                
        hic.PareDown(ax[tile], (0.75,0.95))
  
    ax[abc[-1]].set_xlabel("Wavelength ($\mu$m)", fontsize=10);
    fig.supylabel("Normalised Transit Depth (Rp/Rs)$^{2}$", fontsize=10);    
    plt.show()
    return("Plot Displayed")

