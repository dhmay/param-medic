#!/usr/bin/env python

from setuptools import setup, Extension
from Cython.Distutils import build_ext
import numpy as np
import os
from parammedic.version import __version__

#VERSIONFILE=os.path.join("param-medic","version.py")
#exec(open(VERSIONFILE).read())

long_description = 'Param-Medic breathes new life into MS/MS database searches by optimizing search parameter settings for your data.'
if os.path.exists('README.rst'):
    long_description = open('README.rst').read()

#version = open(VERSIONFILE, "rt").read().strip()

#ext_names = ['base', 'mixturemodel']
ext_names = ['mixturemodel']
EXTENSIONS = []
for ext_name in ext_names:
    EXTENSIONS.append(Extension("parammedic." + ext_name,
                                ["parammedic/" + ext_name + ".pyx"],
                                libraries=[],
                                include_dirs=[np.get_include()]))
#ext_mixturemodel = Extension("parammedic.mixturemodel",
#                             ["parammedic/mixturemodel.pyx"],
#                             libraries=[],
#                             include_dirs=[np.get_include()])
#
#ext_mixturemodel = Extension("parammedic.mixturemodel",
#                  ["parammedic/mixturemodel.pyx"],
#                  libraries=[],
#                  include_dirs=[np.get_include()])

#EXTENSIONS = [ext_base, ext_mixturemodel]

setup(name='param-medic',
      version=__version__,
      description='Param-Medic optimizes MS/MS search parameter settings.',
      author='Damon May',
      author_email='damonmay@uw.edu',
      packages=['param-medic'],
      license='Apache',
      install_requires=['numpy','cython'],
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
