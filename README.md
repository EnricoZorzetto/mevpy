# mevpy 

Version 1.02

Metastatistical Extreme Value Analysis in Python (mevpy) is a set of functions that implement 
the Metastatistical Extreme Value (MEV) Distribution and related
methods. The package was developed for application to rainfall daily data,
but the method is general and may be applied to different fields. 

Instructions:
To install the package, run the setup.py file as follows: 

$ git clone https://github.com/EnricoZorzetto/mevpy 
$ cd path/to/mevpy
$ python setup.py install

Alternatively, place the folder mevpy with its content on your project working directory. 
An example of application can be found in tests/test_mevpy.ipynb notebook.

Dependencies: 
The package works with any version of Python 3 and requires the following packages: numpy, scipy, pandas, and matplotlib. Recommended convention to import this package is 'import mevpy as mev'.

Enrico Zorzetto acknowledges support from the Division of Earth and Ocean
Sciences, Duke University and the NASA Earth and Space Science Fellowship 
(NESSF 17-EARTH17F-0270).

For further information please contact me at enrico.zorzetto@duke.edu

