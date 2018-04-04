
#!/usr/bin/env python
from setuptools import setup


setup(
    name='seqmod',
    version='0.3',
    package_dir={'seqmod.modules': 'modules',
                 'seqmod.misc': 'misc',
                 'seqmod': './'},
    packages=['seqmod', 'seqmod.modules', 'seqmod.misc'],
    description='Pytorch implementations of sequence modellers for language',
    author='Enrique Manjavacas',
    author_email='enrique.manjavacas@gmail.com',
    url='https://www.github.com/emanjavacas/seqmod/',
    download_url='https://api.github.com/repos/emanjavacas/seqmod/tarball',
    license='GPL',
    install_requires=[
        'numpy==1.13.3',
        'tqdm==4.17.1',
        'lorem==0.1.1',
        'gensim==2.3.0',
        'fasttext==0.8.3',
        'scikit_learn==0.19.1',
        'torch==0.3.1',
        'PyYAML==3.12'])
