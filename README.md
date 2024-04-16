# Zooniverse

(for uploading frames and downloading annotations for [`Click-a-Coral`](https://www.zooniverse.org/lab/21853))

### Install

```python
# In Anaconda terminal

# First, create a venv for this repo
conda create --name zooniverse python=3.8 -y 

# Second, clone this repo somewhere (e.g. /Documents/GitHub/)
git clone https://github.com/Jordan-Pierce/Zooniverse.git

# Third, install the dependencies within venv
conda activate zooniverse
python Zooniverse/install.py

```

### from_Zooniverse

Follow instructions [here](https://aggregation-caesar.zooniverse.org/Scripts.html); export the following:

- "Request new classification export" --> click-a-coral-classifications.csv  
- "Request new workflow export" - CSV --> click-a-coral-workflows.csv

```python
# In Anaconda terminal

# This outputs 4 files 
panoptes_aggregation config click-a-coral-workflows.csv 25828 -v 355.143 

# This outputs 2 files
panoptes_aggregation extract click-a-coral-classifications Extractor_config_workflow_25828_V355.143.yaml -o example


```