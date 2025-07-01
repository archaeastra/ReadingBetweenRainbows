import HicSunt as hic
import os
import sys
import RDBSMatrixMulti as multi
from time import sleep

"""
HOW TO USE
- Select number unknowns, bands, SNR and run type in CONSTANTS
- Ensure teltype is in the appropriate range (currently excluses UV)
- Run
> Adress errors if they occur.

Output comes in two forms:
- Stage One: a folder of .dmv files containing the list "viable" cells and their metrics
- Stage Two: a .rdx file that focuses on the RSD for each set of viable cells
"""
##DEFINITIONS

##CONSTANTS
ukn=-5
noA=noB=10
SNR=5
run="TRN"

if run == "PLN": codes = (0,6)
elif run == "TRN": codes = (6,13)
elif run == "RFL": codes = (13,20)
elif run == "RAT": codes = (20,23)
elif run== "BAR": codes = (0,3)
else: sys.exit("Please specify a valid range.\nRun options are: PLN, TRN, RFL, RAT.")


outf='C:/Users/Lyan/StAtmos/HSD/Test/Trays/Multi/MassRDBSM_S{0}U{1}_{2}b.rdx'.format(SNR,(ukn*-1), run)
open(outf, 'w').close() #make sure it's empty before writing to it

##MAIN LOOP
with open(outf, 'a') as outf:
    for A in hic.teltype[1:]:
        for B in hic.teltype[1:]:
            if hic.teltype.index(A)<hic.teltype.index(B): 
                pass
            else:
                print("#{0}v{1}".format(A,B), file=outf)
                for overlist in ["atlas"]:
                    #if overlist=="bar": codes=(0,3)
                    #elif overlist=="atlas": codes=run
                    for code in range(*codes):
                        print("{0}v{1}-{2}{3}...".format(A,B,overlist,code))
                        sleep(0.5)
                                            
                        dmvf='C:/Users/Lyan/StAtmos/HSD/Test/Trays/Multi/dmvmulti/{0}{1}v{2}{3}_S{4}-{5}{6}.dmv'.format(A,noA,B,noB,SNR,overlist,code)
                
                        try: #Don't repeat it if it already exists
                            sel=hic.SelectCell(dmvf)
                        except:
                            os.makedirs("C:/Users/Lyan/StAtmos/HSD/Test/Trays/Multi/dmvmulti", exist_ok = True)
                            dmvf=multi.RDBSM(A, B, 5, SNR, overlist, code, display=False, verbose=False, outfile=True)
                            sel=hic.SelectCell(dmvf)
                        print("{4}{5} \n Min.av.RDP {0} @{1} \n Min.RDP {2} @{3}".format(*multi.RDPPull(dmvf), overlist, code), file=outf)
                    print(" --", file=outf)
                #Flags ideal and modal cells
                print("   Ideal 3: {0}\n   Modal 2: {1}".format(*multi.Redux(sel, 3)), file=outf)
                print("   <<>>", file=outf)#A,B,overlist,code)
    
print("Program Conclusion, check verbose (.rdx) output file.")#"""
        
            
            
