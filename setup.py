"""Installation script."""
from os import path
from io import open
from setuptools import setup

HERE = path.abspath(path.dirname(__file__))

with open(path.join(HERE, 'README.md'), encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read().strip()

setup(
    name='protopt',
    description='',
    long_description=LONG_DESCRIPTION,
    url='https://github.com/bouthilx/protopt.git',
    author='Xavier Bouthillier',
    license='MIT',
    packages=['protopt'],
    install_requires=['pymongo', 'sacred',  # (my branch)
                      'gitpython', 'scikit-optimize'],
    scripts=['bin/opt-launch',
             'bin/opt-run'],
    #         'bin/opt-stats']
    # Optional: pymongo, sacred (my branch), gitpython
)
