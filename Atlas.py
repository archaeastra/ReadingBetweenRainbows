import os
import numpy as np
import HicSunt as hic
import pandas as pd
#This code makes the .atl files in each folder based on the MixingRatios.csv file. 
#It is only useful if you have a MixingRatios file without existing .atl files directly from model output, and is meant to save time in achieving compatibility.

##Load Mixing Ratios .csv
MR = pd.read_csv("./MixingRatios.csv")

#Cycle through
#MAIN CODE
path = "C:/Users/Lyan/StAtmos/HSD/Test/VPL Transits"
try:
    sims = sorted(next(os.walk(os.path.join(path,'.')))[1])
except StopIteration:
    pass
for f in sims[:-1]:
    os.chdir(path + "/" + f)  
    mile=os.getcwd()
    print("Current Position: ", f)
    star = MR.loc[MR['Star']==f]
    
    try:
        pans = sorted(next(os.walk(os.path.join(mile,'.')))[1])#[0:-2]
    except StopIteration:
        pass
    for p in pans: 
        try:
            os.chdir(mile + "/" + p)
        except:
            pass
        print("     Mapping: ", p)
        atm = star.loc[star['Atmosphere']==p]
       
        #Create each file and write values to it.
        with open("./{0}_{1}.atl".format(f, p), 'w') as atl:
            print("#Name    Value", file=atl)
            for label in ["Type", "T(K)", "Flux(S_earth)", "SemMaj(mAU)","R(R_earth)", "Pressure(bar)"]:
                print("{0}    {1}".format(label, atm[label].iloc[0]), file=atl)
            print("\n#Bottom of Atmosphere", file=atl)
            for label in ["O2","H2O","CO","CO2","O3","N2","CH4"]:
                #This section of logic checks the health of the N2 value and corrects it if necessary.
                if (atm[label].iloc[0]>atm['N2CALC'].iloc[0] or atm[label].iloc[0]==0) and label=="N2":
                    print("Replacing {0}_{1} {2} with Calculated Value: {3:.2E}".format(f,p,label,atm['N2CALC'].iloc[0]))
                    label='N2CALC'
                print("{0}    {1:.2E}".format(label, atm[label].iloc[0]), file=atl)
            print("\n#Top of Atmosphere", file=atl)
            for label in ["O2.1","H2O.1","CO.1","CO2.1","O3.1","N2.1","CH4.1"]:
                if (atm[label].iloc[0]>atm['N2CALC.1'].iloc[0] or atm[label].iloc[0]==0) and label=="N2.1":
                    print("Replacing {0}_{1} {2} with Calculated Value: {3:.2E}".format(f,p,label,atm['N2CALC.1'].iloc[0]))
                    label='N2CALC.1'
                print("{0}    {1:.2E}".format(label[:-2], atm[label].iloc[0]), file=atl)
            print("\n\n#Source: {0}".format(atm['Source'].iloc[0]), file=atl)

print("Program Conclusion. Check files")
