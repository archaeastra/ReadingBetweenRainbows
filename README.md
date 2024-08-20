## Reading Between Rainbows

Reading Between Rainbows (RBR) gathers all the code necessary to reproduce the figures from the eponymous study.

All code uses the HicSunt custom function library, which has the following dependencies:
- astropy.io.fits
- matplotlib.pyplot
- numpy
- sklearn.cluster.HDBSCAN
- sklearn.preprocessing.RobustScaler
- scipy.spatial.ConvexHull
- scipy.interpolate
- scipy.fft

> [!NOTE]
> Reproducing RBR figures requires the presence of `.atl` files in the appropriate file structure shown below:
> [!WARNING]
> Tree Not Ready Yet

An example `.atl` file is provided in the example `Modern Sol` directory, and all data used is lised in csv format in the `MixingRatios.csv` file.

**IndivID.py**

IndivID.py produces Figure 1 in RBR, as well as similar graphs for all atmopheres within the given file structure. Ouput looks like the following:
> [!WARNING]
> Image Not Ready Yet

**RunDBS.py**

RunDBS.py produces the individual panels in RBR Figure 5. Manual edits to the code are require to change molecule, please mind the comments for instructions.
Ouput looks like the following:
> [!WARNING]
> Image Not Ready Yet

**RatioDBS.py**

RatioDBS.py produces Figure 7 in RBR. No edits are necessary.
Ouput looks like the following:
> [!WARNING]
> Image Not Ready Yet
