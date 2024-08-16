##IMPORTS
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import RobustScaler
from scipy.spatial import ConvexHull
from scipy import interpolate
from scipy import fft as sft


##CONSTANTS
types  = ['Error', 'Unknown', 'Archean', 'ArcheanHaze', 'E-type', 'ModernEarth', 'DryO2', 'WetO2', 'CO2', 'V-type']
#Note, unless Earth is Planet, Earth label is "earth-like"
colours = ['r', 'k', 'Firebrick', 'Maroon','Navy', 'b', 'SlateBlue', 'DodgerBlue', 'DarkKhaki', 'LimeGreen']
markers = ['.', '+', 'x', '*']

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

def Extract3D(path, col, ext):
    #This pulls the planets within a folder already.
    #Col marks the collumn to pull from. 1 is Total. anything else is a specific molecule. don't do 0, that's x.
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
        #print(os.getcwd())        
        for l in os.listdir('.'):
            if l[-4:] == str(ext):
                spectrum='./'+str(l)
            if l[-4:] == str('.atl'):
                stats = './'+str(l)
        #UPLOAD TABLE
        try:
            #print("Pulling...")
            pull = np.genfromtxt(str(spectrum), dtype='float', comments='#')
            hold = pull[:,int(col)]
            physpln = np.genfromtxt(str(stats), dtype='float', comments='#', usecols=1, filling_values=np.nan)
            #print(physpln)
        except:
            print("Pull Error in {0:s}, check headers! \nTerminating Program...".format(f))
            pass
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
        total[t,1] = Barcode(f)#[0]
        total[t,2]=physpln
        t+=1
    return(total, sims)

def Barcode(pan):
    #z=0, values
    #z=1, planet type
    #z=2, haze indicator.
    #Note, unless Earth is Planet, Earth label is "earth-like"
    if str(pan) in types: 
        code= types.index(str(pan))
    else:
        #1 is used for "spare" values that aren't part of the pre-determined dataset.
        #0 is not used to avoid issues with floats. A 0 value is an error and should be treated as such.
        code=1
    """
    if str(pan).find("Haze")!=-1:
        #A simple binary value for the presence of haze or not.
        haze=1
    else:
        haze=0#"""
    return(code)#, haze)

def TrayTable(path, bands, subres=None, win=(5,12)):
    # 0 = Temperature, 1 = Flux, 2 = Semi-major Axis, 3 = Radius, 4+: individual gas concentrations (rough)
    #This is a temporary kluge to get around an error.
    try:
        print("Extracting VPLSE Data...")
        sims = sorted(next(os.walk(os.path.join(path,'.')))[1])
    except StopIteration:
        pass
    
    tray = np.zeros([len(sims)], dtype="object")
    for f in sims:
        print("    Current Position: ", f)
        hold = np.zeros([len(bands),3,1], dtype="object")
        os.chdir(path + "/" + f)  
        mile=os.getcwd()
        simin=sims.index(f)
        tot, pans= Extract3D(mile, 2, ".trn")
        #print("shape ", np.shape(tot))
        for i in range(0,len(pans)):
            A0=Normalise(tot[i,0][0])
            x0=tot[i,0][1]
            A, x = Unresolve(A0, x0, win, subres)
            try:
                hold[:,1,i]=tot[i,1]
            except IndexError:
                hold=np.pad(hold,((0,0),(0,0),(0,1)),'constant')          
                hold[:,1,i]=tot[i,1]
            for tup in bands:
                y, xt = Window(x, A, tup)
                ar = np.trapz(y, xt)
                #print('ar', ar)
                hold[bands.index(tup),0,i]=ar
                hold[bands.index(tup),2,i]=tot[i,2]
            #print('hold', hold[:,:,i])
        #print(hold.shape)
        tray[simin]=hold
    print("VPLSE Data Loaded")
    return(tray, sims)

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
    
## PROCESSING ROUTINES
def Normalise(y):
    mn=min(y)
    mx=max(y)
    #print(mn, mx)
    for l in range(0,len(y)):    
        y[l] = (y[l]-mn)/(mx-mn)
    return(y)

def Recut(x,step=300):
    #print(len(x))
    gap=len(x)/step
    #print(gap)
    newx=[]
    for i in range(0, step):
        newx.append(x[round(gap*i)])
    #print(len(newx))
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
    #print(window, sill)          
    return(window, sill)

def Bandwidth(spec, nbands):
    #spec is a tuple shped (min, max)
    #nbands is the number of bands desired
    synthspec=np.linspace(spec[0],spec[1], nbands+1)
    bands=[]
    for b in range(0,len(synthspec)-1):
        bands.append((synthspec[b], synthspec[b+1]))
    return(bands)    

def Unresolve(iny, inx, win=(5,12), subres=None):
        #win is the window, defaults to 5-12 as weve been doing before with MIRI
        A=Normalise(iny)
        x=inx
        ya, xa = Window(x, A, (win[0], win[1]))
        if subres!=None:
            F = Normalise(Smooth(ya, 0, subres, rev=True).real)
            newx=Recut(xa, subres)
        else:
            F=ya
            newx=xa
        return(F, newx)

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
