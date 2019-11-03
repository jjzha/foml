__author__ = Mike Zhang
__version__ = 1.0
__license__ = "GPL v3"
__maintainer__ = "Mike Zhang"
__email__ = "j.j.zhang@rug.nl"
__status__ = "early alpha"

# Fundamentals of Machine Learning: Theory and Practice
Code for FoML students.

## Instructions on running Python scripts
### 1. Install Miniconda
https://docs.conda.io/en/latest/miniconda.html
Pick the Python 3.7 version, also check whether your system is 32-bit or 64-bit.

### 2. Install packages
    conda install matplotlib numpy pandas scikit-learn
If some packages are missing:
    conda install <package_name>




To run an experiment, try:

python run_experiment.py --csv data/trainset-sentiment-extra.csv --algorithms nb --nwords 1
