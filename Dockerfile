FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
# https://github.com/pytorch/pytorch/blob/master/docker/pytorch/Dockerfile
ARG PYTHON_VERSION=3.6.10
ARG CUDA_TOOLKIT_VERSION=10.1
ARG PYTORCH_VERSION=1.6.0

# Install some basic utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    sudo \
    git \
    vim \
    bzip2 \
    libx11-6 \
 && rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /app
WORKDIR /app

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

# Install Miniconda and Python 3.x
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=/home/user/miniconda/bin:$PATH
RUN curl -sLo ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh \
 && conda install -y python==$PYTHON_VERSION numpy scipy pandas matplotlib tqdm \
 && conda clean -ya

# Install PyTorch1.x
# https://pytorch.org/get-started/previous-versions/
# https://github.com/anibali/docker-pytorch/blob/master/dockerfiles/
 RUN conda install -y -c pytorch \
    cudatoolkit=$CUDA_TOOLKIT_VERSION \
    "pytorch=1.6.0=py3.6_cuda10.1.243_cudnn7.6.3_0" \
 && conda clean -ya