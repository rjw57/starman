from setuptools import setup, find_packages

setup(
    name="starman",
    version="0.0.1",
    author="Rich Wareham",
    author_email="rich.starman@richwareham.com",

    packages=find_packages(),

    install_requires=[
        "future",
        "numpy",
        "scipy",
    ],
)
