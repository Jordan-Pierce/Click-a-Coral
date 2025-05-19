# Click-a-Coral

for uploading data and downloading annotations for [`Click-a-Coral`](https://www.zooniverse.org/lab/21853)

### Install

```python
git clone https://github.com/Jordan-Pierce/Click-a-Coral.git

conda create --name cac python=3.8 -y 
conda activate cac

pip install uv
uv pip install -e .
```

#### CUDA

```bash
# cmd

# Example for CUDA 11.8
conda install nvidia/label/cuda-11.8.0::cuda-nvcc -y
conda install nvidia/label/cuda-11.8.0::cuda-toolkit -y

# Example for torch w/ CUDA 11.8
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --upgrade
```

### Credentials

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