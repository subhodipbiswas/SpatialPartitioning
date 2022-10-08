# Memetic Algorithms for Spatial Partitioning problems

This is the GitHub repository corresponding to the algorithm **S**warm-based s**pa**atial meme**ti**c **al**gorithm (**SPATIAL**) applied to the problem of school boundary formation, also called school (re)districting.

## Installation

The code is written in Python3.8 and the experiments were run on a machine using Ubuntu 18.04 LTS. You can follow the commands below for setting up your project.

### Setting up virtual environment
Assuming you have Python3, set up a virtual environment
```
pip install virtualenv
virtualenv -p python3 venv
```

### Activate the environment
Always make sure to activate it before running any python script of this project
```
source venv/bin/activate
```

## Package installation
Install the required packages contained in the file requirements.txt. This is a one-time thing, make sure the virtual environment is activated before performing this step.
```
pip install -r requirements.txt
```

Note that some geospatial packages in Python require dependencies like [GDAL](https://gdal.org/) to be already installed.  
You might also want to uninstall some unnecessary packages by doing the following:

```
pip uninstall pygeos
```

## Folder Structure
  ```
  ./
  │
  ├── README.md - Overview of the code repository
  │
  ├── fcps/
  │   ├── data/ - dataset corresponding to FCPS for school year 2020-21
  │   ├── solutions/ - the initial solutions that are input to the algorithms
  │   │        ├── existing/ - the solutions corresponding to the existing plan
  │   │        └── seeded/ - the solutions corresponding to randomly generated plans
  │   └── results - results of the simulation run on FCPS dataset used in the paper
  │
  ├── lcps/
  │   ├── data/ - dataset corresponding to LCPS for school year 2020-21
  │   ├── solutions/ - the initial solutions that are input to the algorithms
  │   │        ├── existing/ - the solutions corresponding to the existing plan
  │   │        └── seeded/ - the solutions corresponding to randomly generated plans
  │   └── results - results of the simulation run on LCPS dataset used in the paper
  │
  ├── get_inputs.py - processes the geospatial files in data/ stores them in datastructures
  │
  ├── functions.py - contains objective functions corresponding to school (re)districting problem
  │
  ├── make_sol.py - generates initial solutions for the algorithm to work on
  │
  ├── utils.py - contains utility functions for the algorithms to use
  │
  ├── REGAL.py - contains the code for the local search techniques, i.e., SA, SHC and TS
  │  
  ├── run_algo.py - script that calls the local search techniques in REGAL.py
  │
  ├── run_spatial.py - script that executes the SPATIAL algorithm
  │  
  └── requirements.txt - contains the libraries that need to be imported 
  ```

## Run the code

Create executables:
```
make runable
```

You can simulate all the experiments using SPATIAL algorithm as:
```
make SPATIAL
```

Similarly, for local search techniques you run one of the following:
```
make SA
make TS
make SHC
```
**Note:** These simulations are reported for randomly generated solutions. Should you want to run the simulations for school redistricting, you need to modify the code in the `Makefile` by replacing ` -i 1` with ` -i 3` and rerunning the above commands.

We already have provided the results of our simulations in `lcps/results` and `fcps/results`. Feel free to rerun them in your own machine.

### Deactivate the environment
Deactivate it before exiting the project
```
deactivate
```

## Citation
If you use this data/code for your work, please consider citing the following article(s):
```
@article{biswas2022memetic,
author = {Biswas, Subhodip and Chen, Fanglan and Chen, Zhiqian and Lu, Chang-Tien and Ramakrishnan, Naren},
title = {Memetic Algorithms for Spatial Partitioning Problems},
year = {2022},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
issn = {2374-0353},
url = {https://doi.org/10.1145/3544779},
doi = {10.1145/3544779},
journal = {ACM Trans. Spatial Algorithms Syst.},
month = {May}
}


@phdthesis{biswas2022phd,
  title={Spatial Optimization Techniques for School Redistricting},
  author={Biswas, Subhodip},
  year={2022},
  school={Virginia Tech},
  url={http://hdl.handle.net/10919/110433}
}
```
## Help
Should you have queries, please reach out to me at subhodipbiswas@vt.edu
