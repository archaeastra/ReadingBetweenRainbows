##IMPORTS
from astropy.modeling.physical_models import BlackBody 
from astropy.io import fits
from astropy import units as u
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import sys
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import RobustScaler
from scipy.spatial import ConvexHull
from scipy import interpolate
from scipy import fft as sft
import warnings

##CONSTANTS
types  = ['Error', 'Unknown', 'Archean', 'ArcheanHaze', 'E-type', 'ModernEarth', 'DryO2', 'WetO2', 'CO2', 'V-type', 'Proterozoic']
colours = ['r', 'Magenta', 'Firebrick', 'Maroon','Navy', 'b', 'SlateBlue', 'DodgerBlue', 'DarkKhaki', 'LimeGreen', 'CadetBlue']

clouds = ['Error', 'Unknown', 'clear', 'haze', 'cirrus', 'cloud', 'strato']
cldcol = ['r', 'Indigo', 'PowderBlue', 'SlateGray', 'SkyBlue', 'SteelBlue', 'DarkBlue']

spectype = ['Error', 'Unknown', 'O', 'B', 'A', 'F', 'G', 'K', 'M', 'L', 'T']
spcol = ['r', 'Lime', 'RoyalBLue', 'LightBlue', 'Aquamarine', 'PaleGoldenrod', 'Gold', 'Orange', 'Crimson', 'DarkRed', 'Maroon']

markers = ['.', '+', 'x', '*']
umrk = ['.', '+', 'x', '*', 'v', '1']

traycols = ["Temperature (K)", "T$_{eff}$(K)", "Flux (S$_{e}$)", "Semi-Major Axis (mAU)", "Radius (R$_{e}$)", "Pressure (bar)", 
            "O$_{2}$", "H$_{2}$O", "CO", "CO$_{2}$", "O$_{3}$", "N$_{2}$", "CH$_{4}$", #bottom of atm
            "O$_{2}$", "H$_{2}$O", "CO", "CO$_{2}$", "O$_{3}$", "N$_{2}$", "CH$_{4}$", #top of atm
            "O$_{2}$/CO$_{2}$", "P/CO$_{2}$", "P/H$_{2}O$"]
gcl = [("plasma", 0.1, 0.8), ("cmr.sunburst", 0.4, 0.8), ("cmr.sepia", 0.2, 1), ("RdBu", 0.2, 0.8), ("cmr.emergency", 0.2, 0.8), ("cmr.voltage", 0.1, 0.8),
                ("cmr.sapphire_r", 0.1, 0.8), ("cmr.ocean_r", 0.2, 0.8), ("cmr.fall_r", 0.1, 0.8), ("pink_r", 0.2, 1),
                ("GnBu_r", 0.1, 0.8), ("cmr.swamp_r", 0.1, 0.8), ("cmr.neutral_r", 0.1, 0.8), #bottom of atm
                ("cmr.sapphire_r", 0.1, 0.8), ("cmr.ocean_r", 0.2, 0.8), ("cmr.fall_r", 0.1, 0.8), ("pink_r", 0.2, 1),
                ("GnBu_r", 0.1, 0.8), ("cmr.swamp_r", 0.1, 0.8), ("cmr.neutral_r", 0.1, 0.8), #top of atm
                ("cmr.iceburn_r", 0, 1),("cmr.holly", 0.1, 0.8), ("cmr.waterlily", 0.2, 0.8)]
rcodes= [(6,9), (5,9), (5,7)]

teltype = ["ECLIPS-UV", "ECLIPS-VIS", "ECLIPS-NIR","NIRSpec", "MIRI"]
telstat = [((0.2,0.5125), 7, "RFL"), ((0.515,1.03), 140, "RFL"), ((1,2),70, "TRN"), ((1,5.3), 100, "TRN"), ((5,12), 280,"TRN")] 
#each is telstat[0]=win as a tuple, and telstat[1]=RP, telstat[2]=runtype


##DATA ROUTINES
#This one is only still here for backwards compatitibility with my older code
#It is not required for TrayTable or if Extract3D is present.
def ExtractTRN(path, col, ext):
    #This pulls the planets within a folder already.
    #Col marks the collumn to pull from. 1 is Total. anything else is a specific molecule. don't do 0, that's x.
    try:
        sims = sorted(next(os.walk(os.path.join(path,'.')))[1])#[0:-2]
    except StopIteration:
        pass
    #Z axis contains the value's barcode. See Barcode below for explanation.
    total = np.zeros([3, len(sims)])
    t=0
    for f in sims: 
        try:
            os.chdir(path + "/" + f)
        except:
            pass
        print(os.getcwd())        
        for l in os.listdir('.'):
            if l[-4:] == str(ext):
                spectrum='./'+str(l)
        #UPLOAD TABLE
        try:
            print("Pulling...")
            pull = np.genfromtxt(str(spectrum), dtype='float', comments='#')
        except:
            print("Pull Error, check spacing! Terminating Program...")
            pass
        total = np.resize(total, (pull.shape[0],len(sims)))  
        total[:,t] = pull[:,int(col)]
        #print(total[0:5,:])
        #print(total[-5:,:])
        #print(total.shape)
        x=pull[:,0]    
        t+=1
    if x[1]<x[0]:
        print("VPL Source. Flipping Array.")
        total=np.flip(total, axis=0)
        x=np.flip(x, axis=0)
    else:
        ("Standard Source. Array Unmodified.")
        pass
    return(total, sims, x)

def Extract3D(path, rt=None):
    #This pulls the planets within a folder already.
    #Col marks the collumn to pull from. 1 is Total. anything else is a specific molecule. don't do 0, that's x.
    #for the TRN type col will be automatically set to pull the correct one from either VLPSE or PSG
    #we are keeping the col variable to avoid breaking anything by accident
    try:
        sims = sorted(next(os.walk(os.path.join(path,'.')))[1])#[0:-2]
    except StopIteration:
        pass
    total = np.zeros([len(sims), 3], dtype="object")
    #Need to start as 3D, else it gets folded like pie dough.
    t=0    
    for f in sims: 
        try:
            os.chdir(path + "/" + f)
        except:
            pass
        #print(os.getcwd()
        for l in os.listdir('.'):
            if rt == "TRN":            
                if l[-4:] == ".trn":
                    spectrum='./'+str(l)
                    col=2 #VPLSE files
                elif l[-4:] == ".rad":
                    spectrum='./'+str(l)
                    col=4 #1 is Total, 4 is Exoplanet only, 5 is transit. PSG files
            elif rt == "RFL":
                if l[-4:] == ".rfl":
                    spectrum='./'+str(l)
                    col=3 # 2 Stellar, 3 Planetary, 4 Albedo. VPLSE files
                elif l[-4:] == ".irr":
                    spectrum='./'+str(l)
                    col=4 #1 is Total, 4 is Exoplanet only. PSG or INARA files.
            else:
                sys.exit('Run type not specified!\nPlease add "TRN" or "RFL" rt flag to TrayTable.')
                    
            if l[-4:] == str('.atl'):
                stats = './'+str(l)
        #UPLOAD TABLE
        try:
            #print("Pulling...")
            pull = np.genfromtxt(str(spectrum), dtype='float', comments='#', filling_values=np.nan)
            hold = pull[:,int(col)]
            physpln = np.genfromtxt(str(stats), dtype='float', comments='#', skip_header=2, usecols=1, filling_values=np.nan)
            spectype = np.genfromtxt(str(stats), dtype='object', comments='#', max_rows=1, usecols=1, filling_values=np.nan)
        except:
            #print("oops")
            sys.exit("Pull Error in {0:s} {1:s}! \n Probable causes: Missing .atl, improper headers. \nTerminating Program...".format(f, spectrum))
        x=pull[:,0]   
        #print(x.shape)
        if x[1]<x[0]:
            #print("VPL Source. Flipping Array.")
            hold=np.flip(hold, axis=0)
            x=np.flip(x, axis=0)
            #print(hold.shape)
        else:
            #print("Standard Source. Array Unmodified.")
            pass
        total[t,0] = [hold,x]
        #print(total.shape)
        total[t,1] = Barcode(f, spectrum, spectype)#0: atmtype, 1: cloudtype, 2:spectype
        #print(Barcode(f, spectrum, spectype))
        total[t,2]=physpln
        t+=1
    return(total, sims)

def Barcode(pan, cloud, atl):
    #1 is used for "spare" values that aren't part of the pre-determined dataset.
    #0 is not used to avoid issues with floats. A 0 value is an error and should be treated as such.
    for ty in types:
        if str(pan).find(ty)!=-1:
            if str(pan)=="ArcheanHaze": 
                pcode=3
            else:
                pcode=types.index(str(ty))
            break
        else:
            pcode=1#"""
    """if str(pan) in types: 
        pcode=types.index(str(pan))
    else:
        pcode=1#"""
    for cld in clouds:
        if str(cloud).find(cld)!=-1:
            ctype=clouds.index(str(cld))
            #print('match found: ', cld)
            break
        else:
            ctype=1#"""
    for spec in spectype:
        if str(atl).find(spec)!=-1:
            stype=spectype.index(str(spec))
            break
        else:
            stype=1
    return([pcode, ctype, stype])

def TrayTable(path, rt, sd, bands, subres=None, win=(5,12), SNR=None):
    # 0 = Temperature, 1 = Flux, 2 = Semi-major Axis, 3 = Radius, 4+: individual gas concentrations (rough)
    #This is a temporary kluge to get around an error.
    #hr is "human readable", will output a .masc txt file as well as a .npy file
    try:
        print("Extracting VPLSE Data...")
        sims = sorted(next(os.walk(os.path.join(path,'.')))[1])
    except StopIteration:
        pass
    
    tray = np.zeros([len(sims)], dtype="object")
    imprt = np.zeros([len(sims)], dtype="object")
    for f in sims:
        print("    Current Position: ", f)
        hold = np.zeros([len(bands),3,1], dtype="object")
        os.chdir(path + "/" + f)  
        mile=os.getcwd()
        simin=sims.index(f)
        tot, pans= Extract3D(mile, rt) #extract spectrum 0 (2 item list), barcode 1 (2 item list), Mixing ratios 2 (array)
        #print("shape ", np.shape(tot))
        tree = np.zeros([len(pans)], dtype="object")
        for i in range(0,len(pans)):
            tree[i]="{0}\\{1}".format(mile,pans[i])
            print("     Calculating: ", pans[i])
            if rt == "RFL":
                bby = BBAdjust(tot[i,0][0], tot[i,0][1], tot[i,2][1])[0] #perform the blackbody adjustment
            else:
                bby = tot[i,0][0] #skip BBAdjust for TRN which is already a dimensionless ratio
            A0=Normalise(bby) #normalise y values of spectrum.
            try: 
                A, x, h = Unresolve(A0, tot[i,0][1], win, subres) #math both
                A, x = AddNoise(A, x, win, SNR, sd)
            except: 
                print("Skipping ", f, i)
                pass
            for tup in bands:
                try:
                    #print(tot[i,1])
                    hold[bands.index(tup),1,i]=tot[i,1] #put barcode in storage in allx-1sty-ithz
                except IndexError:
                    hold=np.pad(hold,((0,0),(0,0),(0,1)),'constant') #add a line       
                    hold[bands.index(tup),1,i]=tot[i,1] #keep going
                    
                y, xt = Window(x, A, tup) #isolate band from spectrum
                #ct = np.column_stack((y, xt))
                #print('y', y[0], y[-1], 'xt', xt[0], xt[-1])
                #np.savetxt('C:/Users/Lyan/StAtmos/yxttst.out', ct, fmt='%s') #ct was used to find albedo nan discontinuity
                ar = np.trapz(y, xt) #calculate area
                #print(tup, ' ar', ar)
                hold[bands.index(tup),0,i]=ar #store area in bandx-0thy-ithz
                hold[bands.index(tup),2,i]=tot[i,2] #store MRs on the 2ndy to keep them and move on.
            #print('hold', hold[:,:,i])  # hold is x=band*s*, y=area|barcode|MRs, z=atmosphere*s*
        #sys.exit("Yoink")
        #print(hold.shape)
        tray[simin]=hold
        imprt[simin]=tree
    print("VPLSE Data Loaded")  # The tray is a list of the hold[x=band*s*, y=area|barcode|MRs, z=atmosphere*s*] per sim
    return(tray, imprt)

def LoadTray(rt, sd, bands, subres, win, SNR): 
    # Here for back-compatibility
    warnings.warn("\nDeprecated for v.1, use GetTRAY.")
    try:
        tray = np.load('C:/Users/Lyan/StAtmos/HSD/Test/Trays/BBProv/{0}-{1}x{2}_{3}({4}){5}S{6}E.npy'.format(win[0], win[1], bands, rt, sd, subres, SNR), 
               allow_pickle=True)
        sims = np.genfromtxt('C:/Users/Lyan/StAtmos/HSD/Test/Trays/VPLSE.sims', dtype='str', delimiter=',', comments='#') 
    except FileNotFoundError:
        sys.exit(".npy file for not found. Please run TrayTable first.")
    print('Tray Loaded')
    return(tray, sims)

def GetTEL(name, nbands=10):
    telstats=telstat[teltype.index(name)]
    #this is going to return [*win,RP,RT]
    #print(telstat)
    rt=telstats[2]
    sd=14
    subres=telstats[1]
    win=telstats[0]    
    bands=Bandwidth(telstats[0], nbands)
    return(rt, sd, subres, win, bands)

def GetTRAY(name, nbands=10, SNR=5):
    rt, sd, subres, win, bands=GetTEL(name, nbands)
    try:
        tray = np.load('C:/Users/Lyan/StAtmos/HSD/Test/Trays/BBProv/{0}-{1}x{2}_{3}({4}){5}S{6}E.npy'.format(win[0], win[1], nbands, rt, sd, subres, SNR), 
               allow_pickle=True)
        sims = np.genfromtxt('C:/Users/Lyan/StAtmos/HSD/Test/Trays/VPLSE.sims', dtype='str', delimiter=',', comments='#') 
    except FileNotFoundError:
        sys.exit(".npy file for {0} not found. Please run TrayTable first.".format(name))
    print('{0} Tray Loaded'.format(name))        
    return(bands, tray, sims, SNR)

def GetOL(tray, source, code):
    #print("Fetching Overlay...")
    if source=="barcode":
        #print('Overlay type is "barcode".')
        ol=tray[code]
        cmap=None
        if code==0:
            label=types[ol]
            c=colours[ol]
            marker=markers[ol%(len(markers))]
        elif code==1:
            label=clouds[ol]
            c=cldcol[ol]
            marker='D'
        elif code==2:
            label=spectype[ol]
            c=spcol[ol]
            marker='*'
        else:
            sys.exit("Code {0} not valid. \n Correct values are: Atmospheres (0), Clouds (1), Spectral Types (2).".format(code))
    if source=="prior":
        #print('Overlay type is "prior".')
        #print("tray", tray)
        if code>22: sys.exit("Code {0} out of range. Valid codes are:".format(code) +
                             "\n0: Temperature, 1: T_eff, 2: Flux, 3: SemJ, 4: R, 5: P" +
                             "\n6: O2, 7: H2O, 8: CO, 9: CO2, 10: O3, 11: N2, 12: CH4 | Bottom of Atmosphere" +
                             "\n13: O2.1, 14: H2O.1, 15: CO.1, 16: CO2.1, 17: O3.1, 18: N2.1, 19: CH4.1 |Top of Atmosphere"
                             "\n20: O2/CO2, 21: P/CO2, 22: P/H2O")
        elif code>=20:
            rcode=rcodes[code-20]
            ol=tray[rcode[0]]/tray[rcode[1]]
        else:
            ol=tray[code]
        label=str(ol)#hic.traycols[code]
        c=ol
        cmap=None
        marker='o'    
    return(np.array([ol, label, c, marker, cmap])) #cmap is empty. It is also loadbearing. I don't understand either


def RLDataSingle(path, bands, ints=1):
    print("Extracting Observatory Data...")
    hdul = fits.open(path, memmap=True)
    hold = np.zeros([len(bands),ints], dtype='object')
    for n in range(0,ints):
        data = fits.getdata(path, ext=n+2) #+2 to avoid the info cards
        y=data['FLUX'][~np.isnan(data['FLUX'])]
        y=Normalise(y)
        x=data['WAVELENGTH'][~np.isnan(data['FLUX'])]
        
        for tup in bands:
            yt, xt = Window(x, y, tup)
            ar = np.trapz(yt, xt)
            hold[bands.index(tup),n]=ar

    hdul.close()
    print("Observatory Data Loaded")
    return(hold)

def RLDataAuto(infile, bands, ints=1):
    with open(infile) as f:
        l=sum(1 for _ in f)
        hold = np.zeros([len(bands),ints, l], dtype='object')
        f.seek(0)
        ln=0
        for line in f:
            #print(line)
            lclean=str(line.rstrip("\n"))
            hdul = fits.open(lclean, memmap=True, ignore_missing_simple=True)
            for n in range(0,ints):
                data = fits.getdata(lclean, ext=n+2) #+2 to avoid the info cards
                y=data['FLUX'][~np.isnan(data['FLUX'])]
                y=Normalise(y)
                x=data['WAVELENGTH'][~np.isnan(data['FLUX'])]
                
                for tup in bands:
                    yt, xt = Window(x, y, tup)
                    ar = np.trapz(yt, xt)
                    hold[bands.index(tup),n, ln]=ar   
            ln+=1 
            hdul.close()
    print("Observatory Data Loaded")
    return(hold)

def SelectCell(dmv, numcells=100):
    #Default is 100 because there are never going to be that many valid cells under the current paradigm
    #therefore all of them are printed.
    hold = np.loadtxt(dmv, dtype='float', comments='#', delimiter=' ', skiprows=0, usecols=(0,1,3,4)) #pull the values from dmv        
    DBmin=np.absolute(hold[:,2]).argsort()
    NRmin=hold[:,3].argsort()
    #print("DB", hold[:,2][DBmin])
    #print("NR", hold[:,3][NRmin])
    #print()
    for x in range(0, hold.shape[0]):
        if hold[x,3]>1:
            hold[x,3]=hold[x,3]/10 #Fixing the NR value in case it was improperly stored
        else:
            pass
    hold= np.pad(hold,((0,0),(0,1)),'constant') #add a collumn to store the score       
    hold[:,4]=np.absolute(hold[:,2]-hold[:,3]) #abs(DB-NR)
    score=hold[:,4].argsort()
    #print("DB", DBmin)
    #print("NR", NRmin)
    #print("SR", hold[:,4])
    return(hold[score][:numcells])

def ValidSlice(validity):
    #Only extracts the "best" one but thankfully, that's usually all we need. 
    #If you need more, output a .dmv and use SelectCell instead.
    score=np.absolute(validity[:,:,0]-validity[:,:,1])
    amin=score.argmin()
    return((int(amin/(score.shape[1])),amin%(score.shape[1]))) #returns a tuple of the coordinates of the lowest value   
    
## PROCESSING ROUTINES
def Normalise(y):
    ny=y#np.ma.masked_invalid(y)
    mn=min(ny)
    mx=max(ny)
    #print(mn, mx)
    for l in range(0,len(y)):    
        y[l] = (y[l]-mn)/(mx-mn)
    return(y)

def Recut(x,step=300):
    #print("OG: ", len(x))
    gap=len(x)/step
    #print(gap)
    newx=[]
    for i in range(0, step):
        newx.append(x[round(gap*i)])
    #print("New: ",len(newx))
    return(newx) 
 

def Smooth(wave, a=0, b=-1, rev=False):
    out = sft.fft(wave)
    #print("Full", len(out))
    if rev==True:
        out = sft.ifft(out[a:b])
        #print("Cut", len(out))
        return(out)
    else:
        return(out)

def Window(x, y, bounds=(0,1)):
    #bounds takes a tuple, for ease of looping
    window=[]
    sill = []
    for i in range(0,len(x)):
        #if int(bounds[0])<=x[i]<=int(bounds[1]):
        if bounds[0]<=x[i]<=bounds[1]:
            #print(x[i])
            window.append(y[i]) 
            sill.append(x[i])
    #print("win", window[:5], "\nsill", sill[:5])          
    return(window, sill)

def Bandwidth(spec, nbands):
    #spec is a tuple shped (min, max)
    #nbands is the number of bands desired
    synthspec=np.linspace(spec[0],spec[1], nbands+1)
    bands=[]
    for b in range(0,len(synthspec)-1):
        bands.append((synthspec[b], synthspec[b+1]))
    return(bands)    

def Unresolve(iny, inx, win, subres=None):
        if win == None: sys.exit("Please specify a window.")
        
        A=Normalise(iny)
        ya, xa = Window(inx, A, (win[0], win[1]))
        if subres!=None and len(xa)-1>subres:
            F = Normalise(Smooth(ya, 0, subres, rev=True).real)
            newx=Recut(xa, subres)
            #print("{0}<{1}".format(subres, len(xa)-1))
            health=1
        elif subres==None:
            #print("subres=None".format(subres, len(xa)-1))
            F=ya
            newx=xa            
            health=1
        else:
            print("{0}>={1}. Defaulting to full.".format(subres, len(xa)-1))
            F=Normalise(ya)
            newx=xa
            health=0
        return(F, newx, health)

def AddNoise(y, x, win, SNR=None, sd=14):
    if SNR==None:
        ya, xa = y, x
        noise=np.zeros((len(ya)))
    else:
        A=Normalise(y)
        ya, xa = Window(x, A, (win[0], win[1]))        
        mn=np.mean(ya)
        #stv=np.nanstd(ya)
        mn_noise=mn/SNR
        #print("Mean: ", mn, ", Noise: ", mn_noise)
        np.random.seed(sd)
        noise = np.random.normal(0, mn_noise, len(ya))
    return(Normalise(ya+noise), xa)

def BBAdjust(y, x, Teff):
    TGU = u.W / (u.m ** 2 * u.um * u.sr)
    bb= BlackBody(temperature= Teff*u.K, scale=1.0*TGU)
    flux=bb(x*u.um)
    Px=bb.lambda_max.to(u.um, equivalencies=u.spectral()).value
    #print("Px", Px)
    step=np.absolute(x[90]-x[89])
    #Sx = x[np.where(np.isclose(x, Px, rtol=20*step))]
    #print("Sx", Sx, "step", step)
    S = y[np.where(np.isclose(x, Px, rtol=20*step))][0]
    P = max(flux.value)
    #print("S", S, "P", P)
    C=S/P

    fxadj=C*flux
    div = Normalise(y/fxadj.value)
    return(div, flux)

## PLOTTING ROUTINES
def PareDown(ax, tup):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique), bbox_to_anchor=tup, loc='upper left', borderaxespad=0)
    
def ContourCluster(xy):
    #Run HDBSCAN
    vals= RobustScaler().fit_transform(xy)
    mpts=0.02
    mcls=0.15
    neareps=0.3
    db = HDBSCAN(cluster_selection_epsilon=neareps, min_samples=round(vals.shape[0]*mpts), min_cluster_size=round(vals.shape[0]*mcls), 
                 algorithm='brute', store_centers='both').fit(vals)
    labels = db.labels_
    error=db.probabilities_
    unique_labels = set(labels)
    color = [plt.cm.tab20b(each) for each in np.linspace(0, 1, len(unique_labels))]
    #Prepare Cluster Plotting
    for k, col in zip(unique_labels, color):
        class_member_mask = labels == k
        err = error[class_member_mask]
        m='o'
        label="_Cluster "+str(k)
        if k == -1:
            #Isolate Noise
            col = 'Grey'
            err=1
            m='x'
            label="Noise"
            nrat=len(vals[class_member_mask])
            xy = vals[class_member_mask]
            plt.scatter(xy[:,0],xy[:, 1],c=col, alpha=err, marker=m, s=50, label=label, zorder=10)
        else:
            xy = vals[class_member_mask]
            #Draw Contours
            hull = ConvexHull(xy)
            x_hull = np.append(xy[hull.vertices,0],
                               xy[hull.vertices,0][0])
            y_hull = np.append(xy[hull.vertices,1],
                               xy[hull.vertices,1][0])
            dist = np.sqrt((x_hull[:-1] - x_hull[1:])**2 + (y_hull[:-1] - y_hull[1:])**2)
            dist_along = np.concatenate(([0], dist.cumsum()))
            spline, u = interpolate.splprep([x_hull, y_hull], 
                                            u=dist_along, s=0, per=1)
            interp_d = np.linspace(dist_along[0], dist_along[-1], 50)
            interp_x, interp_y = interpolate.splev(interp_d, spline)
            plt.fill(interp_x, interp_y, '--', c=col, alpha=0.2)  

def DrawContour(xy, col, plot=True):
    #Adapted from Carvalho, T. (2021)  Visualizing Clusters with Python’s Matplotlib. Towards Data Science.
    #https://towardsdatascience.com/visualizing-clusters-with-pythons-matplolib-35ae03d87489
    
    # draw enclosure
        # get convex hull
        hull = ConvexHull(xy)
        # get x and y coordinates
        # repeat last point to close the polygon
        x_hull = np.append(xy[hull.vertices,0],
                           xy[hull.vertices,0][0])
        y_hull = np.append(xy[hull.vertices,1],
                           xy[hull.vertices,1][0])
        #interpolate
        dist = np.sqrt((x_hull[:-1] - x_hull[1:])**2 + (y_hull[:-1] - y_hull[1:])**2)
        dist_along = np.concatenate(([0], dist.cumsum()))
        spline, u = interpolate.splprep([x_hull, y_hull], 
                                        u=dist_along, s=0, per=1)
        interp_d = np.linspace(dist_along[0], dist_along[-1], 50)
        interp_x, interp_y = interpolate.splev(interp_d, spline)
        # plot shape
        if plot==True:
            plt.fill(interp_x, interp_y, '--', c=col, alpha=0.2) 
        else:
            return(interp_x, interp_y)

def StripAlnum(str):
    fl=re.sub(r'[^a-zA-Z0-9]', "", str)
    return(fl)
