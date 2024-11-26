FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

RUN apt update && \
    apt install -y \
    python3 \
    python3-pip \
    build-essential \
    ninja-build \
    cuda-nvcc-12-1 \
    libcusparse-dev-12-1 && \
    pip3 install wheel packaging ninja torch pytest einops torchvision

COPY . /fast-hadamard-transform
WORKDIR /fast-hadamard-transform
RUN python3 setup.py install
