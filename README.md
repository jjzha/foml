# Fundamentals of Machine Learning: Theory and Practice

Code for FoML students.

__author__ = Mike Zhang  
__version__ = 1.0  
__license__ = GPL v3  
__maintainer__ = Mike Zhang  
__status__ = early alpha  

## Instructions on running Python scripts
### 1. Install Miniconda

    https://docs.conda.io/en/latest/miniconda.html    
Pick the Python 3.7 version, also check whether your system is 32-bit or 64-bit.  

### 2. Install packages

After installing miniconda:  

(1) Go to your terminal (cmd for Windows)  
(2) Type (or copy-paste)  

    conda activate  

(3) Type (or copy-paste)  

    conda install matplotlib numpy pandas scikit-learn  

#### If some packages are missing:  

    conda install <package_name>  

replace <package_name> with preferred package.

### 3. Download code from github

#### Option 1 (static (.zip file)):  
    https://github.com/jjzha/foml/archive/master.zip  

#### Option 2 (dynamic, if you know how to use github):  
(1) Do only once:  

    git clone https://github.com/jjzha/foml.git  

(2) To get most recent version:  

    git pull

### 4. Running an experiment

(1) Go to your foml map (by using 'cd' command). For example:  

    cd ~/Downloads/foml

(2) To run an experiment, try:  

    python3 run_experiment.py --csv data/trainset-sentiment-extra.csv --algorithms nb --nwords 1

### Questions?

If you have any questions, please ask during the lab or contact me via e-mail:
    j.j.zhang@rug.nl
