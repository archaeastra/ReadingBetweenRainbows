## Reading Between Rainbows

Reading Between Rainbows (RBR) gathers all the code necessary to reproduce the figures from the eponymous study.

> [!WARNING]
> This README is under renovation.

All code uses the HicSunt custom function library, which has the following dependencies:
- astropy.io.fits
- [CMasher](https://cmasher.readthedocs.io/index.html)
- [DBCV](https://github.com/christopherjenness/DBCV)
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

An example `.atl` file is provided in the example `Modern Sol` directory, and all data used is lised in csv format in the `MixingRatios.csv` file. Samples without associated `.atl` files will be marked as "unknown" and labelled "Observed" during plotting, as well as reported to the console. Generated samples are expected in formats similar to the INARA database, with a single `.psg` file containing all priors. Real observed data is expected as a `.fits` file and does not take priors. A separate category in `TrayTable.py` (CTRLList) was created to handle observed data in other formats.

**IndivID.py**

IndivID.py produces figures like Figure 2.1 in RBR, as well as similar graphs for all atmopheres within the given file structure. 
Expected Output:
![Degrade](https://github.com/user-attachments/assets/fa8a06ab-f3be-47d4-9cd3-4be7a17cec84)

**DBStest.py**

DBStest.py produces figures like Figure 2.4 in RBR.
Expected Output:
![test](https://github.com/user-attachments/assets/a530c1c3-e236-4973-94e0-8c756d6e454b)

**MatrixMulti.py**

RunDBSMatrix.py produces figures like RBR Figure 2.5. No edits are necessary, but beware of validity "Divide by 0" errors. They are, for the most part, inconsequential but may be indicative of unhealthy data distributions.
Expected Output:
![DBMatrix](https://github.com/user-attachments/assets/6bf3c9a9-d8a7-4398-884e-5e88c517722a)

**RDBSMulti.py**

RunDBS.py produces the individual panels found throughout RBR, such as Figure 2.6. Manual edits to the code are require to change molecule, please mind the comments for instructions.
Expected Output:
![DBS](https://github.com/user-attachments/assets/22bbf5c5-4274-465f-ab22-4f4b3bf49ac0)


**RatioDBS.py**

RatioDBS.py produces figures like Figure 2.8 in RBR. No edits are necessary.
Expected Output:
![ratio](https://github.com/user-attachments/assets/3f1a83f1-a9cb-4e05-b89f-a3ba9ecea041)

**BatchID.py**

BatchID.py produces figures like Figure 4.2 in RBR. It is associated with the "batch" argument in RDBSMulti.py being set to "True".
