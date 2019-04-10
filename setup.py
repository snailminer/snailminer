#!/usr/bin/env python
import os
from setuptools import find_packages, Extension, setup
import numpy

packages = find_packages()
print('find package:%s' % packages)

sources = [
    'snailminer/libminerva/minerva.c',
    'snailminer/libminerva/fruithash.c',
    'snailminer/libminerva/sha3.c',
]

depends = [
    'snailminer/libminerva/hash.h',
    'snailminer/libminerva/sha3.h',
]

ext = [
    Extension('truehash',
              sources, depends=depends,
              extra_compile_args=["-Isnailminer/libminerva/", "-std=gnu99", "-Wall"],
              include_dirs=[numpy.get_include()]),
]

setup(
    name='snailminer',
    license='GPL',
    version='0.0.1',
    url='https://github.com/snailminer/snailminer',
    description=('Miner for the fair fruit chain'),
#   install_requires=['numpy'],
    packages=packages,
    ext_modules=ext,
)
