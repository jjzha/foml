# Fundamentals of Machine Learning: Theory and Practice

Code for FoML students.

__author__ = Malvina Nissim & Mike Zhang  
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

(1) go to your terminal (*Anaconda prompt (miniconda3)* for Windows. Windows users go to (3))  
(2) type (or copy-paste)  

    conda activate  

(3) type (or copy-paste)  

    conda install matplotlib numpy pandas scikit-learn  
    
If you see the **Proceed ([y]/n)?** prompt: press 'y' to continue

#### If some packages are missing:  

    conda install <package_name>  

Please, replace **<package_name>** with your preferred package!  

#### Update conda regularly

    conda update conda

Having a conda *environment* on your pc is very convenient. You are able to run your scripts in this environment without having python locally.

### 3. Download code from github

#### Option 1 (static, .zip file):  
    https://github.com/jjzha/foml/archive/master.zip  

#### Option 2 (dynamic, if you know how to use github):  
(1) do only once:  

    git clone https://github.com/jjzha/foml.git  

(2) to get most recent version:  

    git pull

The difference between static and dynamic is that every time we update some code, you have to download the new code manually. By using **git pull** you are able to *pull* the code from the link.

### 4. Running an experiment

(1) go to your foml map (by using **cd** command). For example:  

    cd ~/Downloads/foml
   
It might be that your **foml-master** directory is not in the **Downloads** folder. To check where you are currently, type in the terminal/command prompt: **pwd** (for Linux/MacOS users) or **echo %cd%** (for Windows users). Then try to navigate to your folder where the **foml** directory is with **cd** (for Linux/MacOS/Windows).

(2) to run an experiment, try:  

    python run_experiment.py --csv data/trainset-sentiment-extra.csv --algorithms nb --nwords 1


# Additional References

With a focus on NLP:  

- Christopher D. Manning and Hinrich Schütze, Foundations of Statistical Natural Language Processing, MIT Press. Cambridge, MA. 1999. http://nlp.stanford.edu/fsnlp/
- Christopher D. Manning, Prabhakar Raghavan and Hinrich Schütze, Introduction to Information Retrieval, Cambridge University Press. 2008. http://nlp.stanford.edu/IR-book/
- James Pustejovsky and Amber Stubbs, Natural Language Annotation for Machine Learning, O’Reilly. 2012.
- Steven Bird, Ewan Klein, and Edward Loper, Natural Language Processing with Python, O’Reilly. 2009. http://www.nltk.org
- Hal Daumé III. A course in Machine Learning. http://ciml.info (incomplete manuscript available online – some parts available for free.)

More generally on machine learning:  

- Tom Mitchell, Machine Learning, McGraw Hill. 1997.
- Ian H. Witten, Eibe Frank, Mark A. Hall, Data Mining: Practical Machine Learning Tools and Techniques, The Morgan Kaufmann Series in Data Management Systems. 2011.
- Yaser S. Abu-Mostafa, Malik Magdon-Ismail, Hsuan-Tien Lin, Learning from Data, AMLBook. 2012.
- Peter Flach, Machine Learning: The Art and Science of Algorithms that Make Sense of Data, Cambridge University Press. 2012.

More specific to Scikit learn (and ML with Python):  

- Luis Pedro Coehlo and Willi Richert, Building Machine Learning Systems with Python, PACKT Publishing. 2013.
- Raúl Garreta and Guillermo Moncecchi, Learning scikit-learn: Machine Learning in Python, PACKT Publishing. 2013.


# Questions?

If you have any questions, please ask during the lab or contact me via e-mail:
    j.j.zhang@rug.nl
