# -*- coding: utf-8 -*-
"""Setup file for Video Stream Processor framework.
"""

from setuptools import setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="vsp",
    version="0.0.1",
    description="Video Stream Processor",
    license="GPLv3",
    long_description=long_description,
    author="John Lloyd",
    author_email="jlloyd237@gmail.com",
    url="https://github.com/jlloyd237/vsp/",
    packages=["vsp", "vsp.pygrabber"],
    install_requires=["numpy", "scipy", "opencv-python>=3.4.5.20", "comtypes", "scikit-image"]
)
