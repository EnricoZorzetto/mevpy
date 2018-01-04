
from setuptools import setup

setup(
   name='mevpy',
   version='1.00',
   description='mevpy package',
   author='Enrico Zorzetto',
   author_email='enrico.zorzetto@duke.edu',
   packages=['mevpy'],  #same as name
   install_requires=['matplotlib', 'pandas', 'numpy', 'scipy'], #external packages as dependencies
)
