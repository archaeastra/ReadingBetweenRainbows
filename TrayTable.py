import numpy as np
import HicSunt as hic
#import os
#from Atlas import INARA

##SELECTORS
#Make sure these are set correctly.
"""
nbands=10
win=(5, 12)
##WINDOWS
#MIRI 5, 12 NIRSPEC 1, 5.3 | 
#ECLIPS_NIR 1, 2 _VIS 0.515, 1.03 _NUV 0.2, 0.5125 | 
#LUMOS 0.1, 1 >0.27 start to dodge nan sections in AD Leo
#POLLUX 0.1, 0.4 |
SNR=5
rt='TRN'
seed=14
subres=280

bands=hic.Bandwidth(win, nbands)
#bands = [(5,6),(6,7),(7,9),(9,11),(11,12)] #MIRI manual
#bands = [(0.27,0.3),(0.3,0.4),(0.4,0.6),(0.6,0.7),(0.7,1)] #LUMOS manual 2 and 3
#bands = [(0.27,0.4),(0.4,0.6),(0.6,0.7),(0.7,0.9),(0.9,1)] #LUMOS manual 4
"""
#ACTIVE VARIABLES
hr=False
nbands=10
SNR=5
flag="RL2"
CTRLList=["C:/Users/Lyan/StAtmos/HSD/Test/Other Transits"]
MCList=[]#"C:/Users/Lyan/StAtmos/HSD/Test/MC Spectra/GReducing", "C:/Users/Lyan/StAtmos/HSD/Test/MC Spectra/GOxic"]
INARAList=[]#"C:/Users/Lyan/StAtmos/HSD/Test/INARA Spectra/Earth", "C:/Users/Lyan/StAtmos/HSD/Test/INARA Spectra/Venus", "C:/Users/Lyan/StAtmos/HSD/Test/INARA Spectra/Mars"]
RLList=['C:/Users/Lyan/StAtmos/HSD/JWSTData/MAST_2024-07-10T0954/JWST/jw06456-o001_t001_nirspec_clear-prism-sub512/jw06456-o001_t001_nirspec_clear-prism-sub512_x1dints.fits', 'C:/Users/Lyan/StAtmos/HSD/JWSTData/MAST_2024-07-16T0720/JWST/jw02589-o006_t001_nirspec_clear-prism-sub512/jw02589-o006_t001_nirspec_clear-prism-sub512_x1dints.fits']
GenList=MCList+INARAList

#UPLOAD TABLE
#Pull in the Tray
# The tray is a list of the hold[x=band*s*, y=area|barcode|MRs, z=atmosphere*s*] per sim
rt, seed, subres, win, bands=hic.GetTEL("NIRISS", nbands)
tray, sims=hic.TrayTable("C:/Users/Lyan/StAtmos/HSD/Test/VPL Transits", rt, seed, bands, subres, win, SNR, source="VPLSE")
#tray, sims=hic.TrayTable("C:/Users/Lyan/StAtmos/HSD/Test/MC Spectra/GOxic", rt, seed, bands, subres, win, SNR, source="INARA")
for path in CTRLList:
    ctray, csims=hic.TrayTable(path, rt, seed, bands, subres, win, SNR, source="VPLSE")
    try:    
        print("Merging Control dataset {0}...".format(CTRLList.index(path)))
        tray=np.block([ctray, tray])
        sims=np.block([csims, sims]) 
        #We're putting it at the front to not mess with the unknowns counters
    except:
        print("Additional List Not Set.")
for path in GenList:
    intray, insims=hic.TrayTable(path, rt, seed, bands, subres, win, SNR, source="INARA")
    #Appending the INARA dataset, if it exists:
    try:    
        print("Merging Generated dataset {0}...".format(GenList.index(path)))
        tray=np.block([intray, tray])
        sims=np.block([insims, sims]) 
        #We're putting INARA at the front to not mess with the unknowns counters
    except:
        print("Generated List Not Set.")
for path in RLList:
    rltray, rlsims = hic.TrayTable(path, rt, seed, bands, subres, win, SNR, source="RL")
    #Appending RL dataset if it exists:    
    try:
        print("Merging Observed dataset {0}...".format(RLList.index(path)))
        tray=np.block([rltray, tray])
        sims=np.block([rlsims, sims]) 
    except:
        print("Observed List Not Set.")

print('Import Successful')

#Save the Tray in files
np.save('C:/Users/Lyan/StAtmos/HSD/Test/Trays/RBR/{0}-{1}x{2}_{3}({4}){5}S{6}{7}.npy'.format(win[0], win[1], nbands, rt, seed, subres, SNR, flag), tray)#, fmt='%s')
if hr==True:
    np.savetxt('C:/Users/Lyan/StAtmos/HSD/Test/Trays/RBR/{0}-{1}x{2}_{3}({4}){5}S{6}{7}.masc'.format(win[0], win[1], nbands, rt, seed, subres, SNR, flag), tray, fmt='%s')
else:
    pass
np.save('C:/Users/Lyan/StAtmos/HSD/Test/Trays/Imprint{0}{1}.npy'.format(rt, flag), sims)
np.savetxt('C:/Users/Lyan/StAtmos/HSD/Test/Trays/Imprint{0}{1}.sims'.format(rt, flag), sims, fmt='%s')
print('File Saved')
