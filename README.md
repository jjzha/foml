# Fundamentals of Machine Learning: Theory and Practice

Code for FoML students.

__author__ = Mike Zhang  
__version__ = 1.0  
__license__ = GPL v3  
__maintainer__ = Mike Zhang  
__status__ = early alpha  

## Instructions on running Python scripts
### 1. Install Miniconda

First, we are going to install a package manager. This makes sure that all your packages are up-to-date:

    https://docs.conda.io/en/latest/miniconda.html    
    
Please note: pick the **Python 3.7 version**, also check whether your system is **32-bit** or **64-bit**.  

### 2. Install packages

After installing miniconda, we are going to install some packages:  

(1) go to your terminal (cmd for Windows)  
(2) type (or copy-paste)  

    conda activate  

(3) type (or copy-paste)  

    conda install matplotlib numpy pandas scikit-learn  

#### If some packages are missing:  

    conda install <package_name>  

Please, replace **<package_name>** with your preferred package!  

Having a conda *environment* on your pc is very convenient. You are able to run your scripts in this environment without having python locally.

### 3. Download code from github

#### Option 1 (static, .zip file):  
    https://github.com/jjzha/foml/archive/master.zip  

#### Option 2 (dynamic, if you know how to use github):  
(1) do only once:  

    git clone https://github.com/jjzha/foml.git  

(2) to get most recent version:  

    git pull

### 4. Running an experiment

(1) go to your foml map (by using 'cd' command). For example:  

    cd ~/Downloads/foml
   
It might be that your **foml** directory is not in the **Downloads** folder. To check where you are currently, type in the terminal/command prompt: **pwd** (for Linux/MacOS users) or **echo %cd%** (for Windows users). Then try to navigate to your folder where the **foml** directory is with **cd** (for Linux/MacOS/Windows).

(2) to run an experiment, try:  

    python3 run_experiment.py --csv data/trainset-sentiment-extra.csv --algorithms nb --nwords 1

### Questions?

If you have any questions, please ask during the lab or contact me via e-mail:
    j.j.zhang@rug.nl
