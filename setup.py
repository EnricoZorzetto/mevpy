

from distutils.core import setup

setup(
   name='mevpy',
   version='v1.0.5',
   description='mevpy package',
   author='Enrico Zorzetto',
   author_email='enrico.zorzetto@duke.edu',
   url = 'https://github.com/EnricoZorzetto/mevpy',
   download_url = 'https://github.com/EnricoZorzetto/mevpy/archive/v1.0.5.tar.gz',
   packages=['mevpy'],  
   install_requires=['matplotlib', 'pandas', 'numpy', 'scipy'], 
)


