import numpy as np
import HicSunt as hic
import os
#This code creates the files containing the value arrays for use in DBS code in this library.
#It outputs a readable .masc file to check your values, a binary .npy file for computer use and 
#optionally a .sims file displaying the list of folders, which only needs to be created once unless the files change.

hr=True #Flag to make a "human-readable" .masc file or not.

nbands=10
win=(5, 12)  #Make sure this is set correctly.
##WINDOWS
#MIRI 5-12 NIRSPEC 1-5.3 | ECLIPS_NIR 1-2 _VIS 0.515-1.03  _NUV 0.2-0.5125 | LUMOS 0.1-1 >0.27 start to dodge nan sections in AD Leo

bands=hic.Bandwidth(win, nbands)
#bands = [(5,6),(6,7),(7,9),(9,11),(11,12)] #MIRI manual
#bands = [(0.27,0.3),(0.3,0.4),(0.4,0.6),(0.6,0.7),(0.7,1)] #LUMOS manual 2 and 3
#bands = [(0.27,0.4),(0.4,0.6),(0.6,0.7),(0.7,0.9),(0.9,1)] #LUMOS manual 4

subres=None

#UPLOAD TABLE
ext='.trn'
col=2
tray, sims=hic.TrayTable("C:/Users/Lyan/StAtmos/HSD/Test/VPL Transits", ext, col, bands, subres, win)
#Pull in the Tray
# The tray is a list of the hold[x=band*s*, y=area|barcode|MRs, z=atmosphere*s*] per sim
print('Import Successful')

#Save the Tray in files
np.save('C:/Users/Lyan/StAtmos/HSD/Test/Trays/{0}-{1}x{2}_{3}({4}){5}.npy'.format(win[0], win[1], nbands, ext[1:], col, subres), tray)#, fmt='%s')
if hr==True:
    np.savetxt('C:/Users/Lyan/StAtmos/HSD/Test/Trays/{0}-{1}x{2}_{3}({4}){5}.masc'.format(win[0], win[1], nbands, ext[1:], col, subres), tray, fmt='%s')
else:
    pass
#np.savetxt('C:/Users/Lyan/StAtmos/HSD/Test/Trays/{0}-{1}x{2}_{3}({4}){5}.sims'.format(win[0], win[1], nbands, ext[1:], col, subres), sims, fmt='%s')
print('File Saved')


