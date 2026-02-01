# dls_sls_technical_internship

This directory contains python classes and notebooks for reading, analyzing and visualizing the data from the ALV goniometer system.

## read_ASC.py
This is the main python file containing four classes. 

### read_asc
This class reads the content of an .ASC file. It extracts the parameters as wel as the data from the experiment in an object oriented way. A few functions have been added like simple visualizations and a fit for dls. 

### dls_sls_analysis
This class takes the path to a directory and makes read_asc objects of each file. The structure of the directory is as follows.

**directory_name**
|
|--> standard
|
|--> solvent
|
 --> solution

The directory_name/standard should contain calibration data using only toluene in the cuvette at the angles to be experimented on.

The directory_name/solvent should contain calibration data using the same solvent and angles that are to be used in the experiment.

*the directories mentiond above are only used for the sls analysis*

The directory_name/solution should contain experimental data with the sample.

### contin_fit
Is a dls_sls_anasis inheritance class. 

### SLS_functions
Is a Python file that contains every function needed to perform the SLS analysis.
For this, the above named file structure is sightly changed:

**directory_name**
|
|--> concentration 1
|
|--> concentration 2
|
 --> concentration 3

 Each concentration folder follows the same file structure as the structure named in the dls_sls_analysis section.
