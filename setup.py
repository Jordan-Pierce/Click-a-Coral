import os
from setuptools import setup, find_packages

# Check if requirements.txt exists
assert os.path.exists(
    "requirements.txt"), "ERROR: Cannot find requirements.txt"

# Read requirements from requirements.txt
with open('requirements.txt') as f:
    required_packages = f.read().splitlines()

# Filter out any empty lines or comments
required_packages = [
    line for line in required_packages if line and not line.startswith('#')]
required_packages = [
    line for line in required_packages if not line.startswith("https://")]

# Setup
setup(
    name='cac',
    version='0.0.1',
    url='https://github.com/Jordan-Pierce/Click-a-Coral',
    author='Jordan Pierce, Kira Kapplan',
    author_email='jordan.pierce@noaa.gov, kira.kapplan@noaa.gov',
    packages=find_packages(),
    install_requires=required_packages,
    python_requires='>=3.8',
    extras_require={'dev': ['jupyter', 'ipython']},
)
