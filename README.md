# Click-a-Coral

(for uploading data and downloading annotations for [`Click-a-Coral`](https://www.zooniverse.org/lab/21853))

### Install

```python
# First, create a venv for this repo
conda create --name cac python=3.8 -y 

# Second, clone this repo somewhere (e.g. /Documents/GitHub/)
git clone https://github.com/Jordan-Pierce/Click-a-Coral.git

# Third, install the dependencies within venv
conda activate cac
python install.py

```

```python
# Set up the environment variables
setx ZOONIVERSE_USERNAME "your_username"
setx ZOONIVERSE_PASSWORD "your_password"
```

### Usage

```python   
# cmd

python src/to_Zooniverse.py --media_ids 123456789 --upload
```