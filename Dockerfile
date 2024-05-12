# Use a base image with Conda installed
FROM continuumio/miniconda3

# Install essential build tools including g++
RUN apt-get update && apt-get install -y build-essential cmake libgl1-mesa-glx


# Create a Conda environment with Python 3.7
RUN conda create -n botsort python=3.7

# Activate the Conda environment
SHELL ["conda", "run", "-n", "botsort", "/bin/bash", "-c"]


# Change to the home directory
WORKDIR /root

# Install PyTorch and other required packages within the Conda environment
RUN conda install pytorch torchvision cudatoolkit=9.0 -c pytorch

# Copy requirements.txt to docker
COPY requirements.txt .

# Install additional Python packages within the Conda environment
RUN pip install -r requirements.txt
RUN pip install cython
RUN pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
RUN pip install cython_bbox
RUN pip install faiss-cpu
RUN pip install faiss-gpu

#ENTRYPOINT ["conda", "run", "-n", "botsort", "/bin/bash"]
