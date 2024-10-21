import sys
import shutil
import platform
import subprocess

# ----------------------------------------------
# OS
# ----------------------------------------------
osused = platform.system()

if osused not in ['Windows', 'Linux']:
    raise Exception("This install script is only for Windows or Linux")

# ----------------------------------------------
# Conda
# ----------------------------------------------
# Need conda to install NVCC if it isn't already
console_output = subprocess.getstatusoutput('conda --version')

# Returned 1; conda not installed
if console_output[0]:
    raise Exception("This install script is only for machines with Conda already installed")

conda_exe = shutil.which('conda')

# ----------------------------------------------
# Python version
# ----------------------------------------------
python_v = f"{sys.version_info[0]}{sys.version_info[1]}"
python_sub_v = int(sys.version_info[1])

# check python version
if python_sub_v != 8:
    raise Exception(f"Only Python 3.{python_sub_v} is supported.")

# ---------------------------------------------
# MSVC for Windows
# ---------------------------------------------
if osused == 'Windows':

    try:
        print(f"NOTE: Installing msvc-runtime")
        subprocess.check_call([sys.executable, "-m", "pip", "install", 'msvc-runtime'])

    except Exception as e:
        print(f"There was an issue installing msvc-runtime\n{e}")
        sys.exit(1)

# ----------------------------------------------
# Other dependencies
# ----------------------------------------------
install_requires = [
    'wheel',
    'tqdm',
    'numpy',
    'pandas',
    'scipy',
    'scikit_learn',
    'matplotlib',
    'scikit_image',

    'tator',
    'panoptes-client',
]

# Installing all the other packages
for package in install_requires:

    try:
        print(f"NOTE: Installing {package}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

    except Exception as e:
        print(f"There was an issue installing {package}\n{e}\n")
        print(f"If you're not already, please try using a conda environment with python 3.8")
        sys.exit(1)

print("Done.")