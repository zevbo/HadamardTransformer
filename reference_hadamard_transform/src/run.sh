#!/bin/bash

echo "Hello World"
# apt-get install cuda-nvcc-12-4 -y
echo $(which nvcc)
echo $(ls /usr/local/)


pip install pytest einops torchvision

# apt-get install locate -y
echo $(locate cusparse.h)

echo "Running..."

cd fast-hadamard-transform
export CUDA_HOME=/usr/local/cuda
# python3 setup.py install

echo "Finished install"

cd tests
# python3 test_fast_hadamard_transform.py
