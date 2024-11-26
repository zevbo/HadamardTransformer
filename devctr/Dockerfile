FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

# Install needed packages.
RUN apt update && \
    apt -y install python3 cuda-nvcc-12-4
