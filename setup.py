#!/usr/bin/env python

from distutils.core import setup
from setuptools import find_packages
import os
import platform
import sys
import re
import ast
from glob import glob


# version parsing from __init__ pulled from Flask's setup.py
# https://github.com/mitsuhiko/flask/blob/master/setup.py

# python version control from Flask's setup.py
# https://github.com/mitsuhiko/flask/blob/master/setup.py
CURRENT_PYTHON = sys.version_info[:2]
REQUIRED_PYTHON = (3, 6)

# This check and everything above must remain compatible with Python 2.7.
if CURRENT_PYTHON > REQUIRED_PYTHON or CURRENT_PYTHON < REQUIRED_PYTHON:
    sys.stderr.write("""
==========================
Unsupported Python version
==========================
This version of TADA requires Python {}.{}, but you're trying to
install it on Python {}.{}.
""".format(*(REQUIRED_PYTHON + CURRENT_PYTHON)))
    sys.exit(1)


_version_re = re.compile(r'__version__\s+=\s+(.*)')

with open('src/utils/tada/__init__.py', 'rb') as f:
    hit = _version_re.search(f.read().decode('utf-8')).group(1)
    __version__ = str(ast.literal_eval(hit))

long_description = ("TADA: phylogenetic augmentation of microbiome samples enhances phenotype classification")


setup(name = 'TADA',
      version = __version__,
      long_description = long_description,
      description = 'TADA',
      python_requires='>={}.{}'.format(*REQUIRED_PYTHON),
      author = 'Erfan Sayyari, Ban Kawas, Siavash Mirarab',
      author_email = 'smirarabbaygi@eng.ucsd.edu',
      url = 'https://github.com/tada-alg/TADA/',
      packages=find_packages(),
      install_requires=["dendropy>=4.0.0", "numpy>=1.14.0", "biom-format>=2.1.5","imbalanced-learn>=0.4.3",
                        "scikit-learn>=0.19.1",
                        "scipy>=1.0.0","pandas>=0.22.0"],
      scripts=glob("src/utils/scripts/*"),
      package_data={'': 'data'},
      classifiers = ["Environment :: Console",
                     "Intended Audience :: Developers",
                     "Intended Audience :: Science/Research",
                     ("License :: OSI Approved :: GNU General Public "
                     "License (GPL)"),
                     "Natural Language :: English",
                     "Operating System :: POSIX :: Linux",
                     "Operating System :: MacOS :: MacOS X",
                     "Programming Language :: Python",
                     "Programming Language :: Python :: 3.6",
                     "Topic :: Scientific/Engineering :: Bio-Informatics"])
