import os
from setuptools import setup, find_packages

def read_file(path):
    with open(os.path.join(os.path.dirname(__file__), path), "r") as fobj:
        return fobj.read()

setup(
    name="starman",
    version="1.0.0",
    packages=find_packages(),

    install_requires=["future", "numpy"],

    author="Rich Wareham",
    author_email="rich.starman@richwareham.com",
    url="https://github.com/rjw57/starman",
    description="A library which implements algorithms of use when "
                "trying to track the true state of one or more systems over "
                "time in the presence of noisy observations.",
    long_description=read_file("README.rst"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
    ],
    license="MIT",
)
