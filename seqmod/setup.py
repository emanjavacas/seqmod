#!/usr/bin/env python

from setuptools import setup


setup(
    name='seqmod',
    version='0.1',
    package_dir={'seqmod.modules': 'modules',
                 'seqmod.misc': 'misc',
                 'seqmod': './'},
    packages=['seqmod', 'seqmod.modules', 'seqmod.misc'],
    description='Pytorch implementations of sequence modellers for language',
    author='Enrique Manjavacas',
    author_email='enrique.manjavacas@gmail.com',
    url='https://www.github.com/emanjavacas/seqmod/',
    download_url='https://api.github.com/repos/emanjavacas/seqmod/tarball',
    install_requires=['numpy>=1.10.4',
                      'torch>=0.1.10+ac9245a',
                      'graphviz>=0.5.2',
                      'visdom>=0.1.2'],
    license='MIT')
