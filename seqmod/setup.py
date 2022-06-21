
#!/usr/bin/env python
from setuptools import setup


setup(
    name='seqmod',
    version='0.4',
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
        'numpy==1.22.0',
        'tqdm==4.23.3',
        'lorem==0.1.1',
        'gensim==3.4.0',
        'scikit_learn==0.19.1',
        'torch==0.4.0',
        'matplotlib==2.0.2',
        'PyYAML==3.12'
    ])
