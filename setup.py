#!/usr/bin/env python

from distutils.core import setup

setup(name='trajectory_evaluation',
      version='1.0',
      description='Python Distribution Utilities',
      author='Roland Jung',
      author_email='roland.jung@aau.at',
      url='https://gitlab.aau.at/aau-cns/py3_pkgs/trajectory_evaluation/',
      packages=['distutils', 'distutils.command', 'numpy', 'tqdm', 'pandas', 'argparse', 'PyYAML', 'matplotlib', 'spatialmath-python', 'numpy_utils', 'trajectory', 'csv2dataframe', 'ros_csv_formats'],
     )
