import os
import re
from setuptools import find_packages
from setuptools import setup
import io, shutil
from distutils.extension import Extension

cmdclass = {}
ext_modules = []

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, 'dandelion', '__init__.py'), 'r') as f:
    init_py = f.read()
version = re.search('__version__ = "(.*)"', init_py).groups()[0]

with io.open(os.path.join(here, 'README.md'), 'r', encoding='utf-8') as f:
    README = f.read()

# install_requires = [
#     'numpy',
#     'Theano',
#     ]
#
# tests_require = [
#     'mock',
#     'pytest',
#     'pytest-cov',
#     ]
setup(
    name="Dandelion",
    version=version,
    description="A light weight deep learning framework",
    long_description=README,
    long_description_content_type="text/markdown",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
		"Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
    keywords="DL framework, Theano",
    author="David Leon (Dawei Leng)",
    author_email="daweileng@outlook.com",
    license="Mozilla Public License v2.0",
    url="https://github.com/david-leon/Dandelion",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    # install_requires=install_requires,
    # extras_require={
    #     'testing': tests_require,
    #     },
    cmdclass=cmdclass,
    ext_modules=ext_modules,
    )
