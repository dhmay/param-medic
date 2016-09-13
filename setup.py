#!/usr/bin/env python

from setuptools import setup
import os

VERSIONFILE=os.path.join("param-medic","version.py")
exec(open(VERSIONFILE).read())

long_description = 'Param-Medic breathes new life into MS/MS database searches by optimizing search parameter settings for your data.'
if os.path.exists('README.rst'):
    long_description = open('README.rst').read()

version = open(VERSIONFILE, "rt").read().strip()

setup(name='param-medic',
      version=__version__,
      description='Param-Medic breathes new life into MS/MS database searches by optimizing search parameter settings for your data.',
      author='Damon May',
      author_email='damonmay@uw.edu',
      packages=['param-medic'],
      license='Apache',
      install_requires=['numpy'],
      scripts=['bin/param-medic']
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
     )
