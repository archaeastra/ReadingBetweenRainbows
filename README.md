## Reading Between Rainbows

Reading Between Rainbows (RBR) gathers all the code necessary to reproduce the figures from the eponymous study.

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

An example `.atl` file is provided in the example `Modern Sol` directory, and all data used is lised in csv format in the `MixingRatios.csv` file.

**IndivID.py**

IndivID.py produces Figure 1 in RBR, as well as similar graphs for all atmopheres within the given file structure. Ouput looks like the following:
![Degrade](https://github.com/user-attachments/assets/fa8a06ab-f3be-47d4-9cd3-4be7a17cec84)

**DBStest.py**

DBStest.py produces Figure 3 in RBR. No edits are necessary.
Ouput looks like the following:
![test](https://github.com/user-attachments/assets/a530c1c3-e236-4973-94e0-8c756d6e454b)

**RunDBSMatrix.py**

RunDBSMatrix.py produces RBR Figure 4. No edits are necessary, but beware of validity "Divide by 0" errors. They are, for the most part, inconsequential but may be indicative of unhealthy data distributions.
Ouput looks like the following:
![DBMatrix](https://github.com/user-attachments/assets/6bf3c9a9-d8a7-4398-884e-5e88c517722a)

**RunDBS.py**

RunDBS.py produces the individual panels in RBR Figure 5. Manual edits to the code are require to change molecule, please mind the comments for instructions.
Ouput looks like the following:
![DBS](https://github.com/user-attachments/assets/22bbf5c5-4274-465f-ab22-4f4b3bf49ac0)


**RatioDBS.py**

RatioDBS.py produces Figure 7 in RBR. No edits are necessary.
Ouput looks like the following:
![ratio](https://github.com/user-attachments/assets/3f1a83f1-a9cb-4e05-b89f-a3ba9ecea041)

