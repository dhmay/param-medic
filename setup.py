#!/usr/bin/env python

from setuptools import setup, Extension
from Cython.Distutils import build_ext
import numpy as np
import os
from parammedic.version import __version__

long_description = 'Param-Medic breathes new life into MS/MS database searches by optimizing search parameter settings for your data.'
if os.path.exists('README.rst'):
    long_description = open('README.rst').read()

ext_names = ['mixturemodel', 'binning']
EXTENSIONS = []
for ext_name in ext_names:
    EXTENSIONS.append(Extension("parammedic." + ext_name,
                                ["parammedic/" + ext_name + ".pyx"],
                                libraries=[],
                                include_dirs=[np.get_include()]))

setup(name='param-medic',
      version=__version__,
      description='Param-Medic optimizes MS/MS search parameter settings.',
      author='Damon May',
      author_email='damonmay@uw.edu',
      packages=['parammedic'],
      license='Apache',
      install_requires=['numpy', 'cython', 'pyteomics'],
      scripts=['bin/param-medic'],
      long_description=long_description,
      classifiers=[
          'Development Status :: 4 - Beta',
          'Topic :: Scientific/Engineering :: Bio-Informatics',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: Apache Software License',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.7',
          ],
      keywords='proteomics LC-MS/MS MS/MS spectrometry',
      cmdclass={"build_ext": build_ext},
      ext_modules=EXTENSIONS
     )
