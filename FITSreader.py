#NOTE: THIS CODE DOES NOT ACTUALLY FUNCTION. IT IS ARCHIVED HERE AS A WORK IN PROGRESS

from astropy.io import fits
import matplotlib.pyplot as plt
import cmasher as cmr
import HicSunt as hic
import numpy as np
from astropy import units as u
#import sys
#sys.stdout = open('./hdr0output.txt','wt')

##DEFINITIONS
def ConvertJF(J,w):
    c=(3E14)*(u.um/u.s)  #3E18 for Angstrom, 3E14 for um
    F=J*(10E-23)*(c/w)
    return (F)

ints=100

cmap = cmr.take_cmap_colors('cmr.amber', ints, cmap_range=(0, 0.8), return_fmt='hex')
path= '[LOCATTION OF FITS FILE]'
phoenixpath='C:/Users/Lyan/StAtmos/HSD/PHOENIXStellar/2500_log5_Met0.phoenix'

hdul = fits.open(path, memmap=True)
#hdr=hdul[0].header
#print(hdul.info())
#print(hdr)

phoenix = np.genfromtxt(phoenixpath, comments='#')

fig, ax = plt.subplot_mosaic("AB;CC")
fig.set_size_inches(6,6);
fig.tight_layout(pad=4, w_pad=4);
fig.suptitle("PHOENIX MODEL - 2500", fontsize=12);
#fig.text(x=0.5, y=0.92, s= "Integrations: {0}".format(ints), fontsize=8, ha="center")    
plt.subplots_adjust(top=0.913,
                    bottom=0.092,
                    left=0.112,
                    right=0.952,
                    hspace=0.302,
                    wspace=0.292)

#HANDLE JWST
for n in range(2,ints):
    data = fits.getdata(path, ext=n)
    y=data['FLUX'][~np.isnan(data['FLUX'])]
    y1=(y*u.Jy)#.to(u.ABflux)
    #print(min(y), max(y))
    #y=hic.Normalise(y)
    x=data['WAVELENGTH'][~np.isnan(data['FLUX'])]
    #No conversions, raw JWST data.
    x1=(x*u.um)
    #y1=ConvertJF(y, x)
    ax["A"].plot(x1, y1, c=cmap[n])
ax["A"].set_ylabel("Flux $(Jy)$", fontsize=10);
ax["A"].set_xlabel("Wavelength $um$", fontsize=10);     
  
#print(len(x))
#print(data[:5])
#print(data[-5:])
y1=ConvertJF(y, x) #Convert JWST' Jy to Flux

#HANDLE PHOENIX
y=phoenix[:,1]
x=phoenix[:,0]
#y2,x2=hic.Unresolve(y, x, (0,x[-1]), 407)

x2=(x*u.angstrom).to(u.um) #Use same x scale
y2=(y*10E-4) #Adjust y to use um instead of A, to keep par

ax["B"].plot(x2, y2)
ax["B"].set_xlim(0,6);      
ax["B"].set_ylabel("Flux $(erg cm^{-2} s^{-1} um^{-1})$", fontsize=8);
ax["B"].set_xlabel("Wavelength $um$", fontsize=10);      

#COMBINE
ax["C"].plot(x1, y1)
ax["C"].set_ylabel("Converted Flux $(erg cm^{-2} s^{-1} um^{-1})$", fontsize=8);
ax["C"].set_xlabel("Wavelength $um$", fontsize=10);#"""


plt.show()#"""
hdul.close()
