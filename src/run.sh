#!/bin/bash

echo "Starting..."

pip install ninja

echo $(ls)
echo $(ls /usr/local/cuda) 

# python3 setup.py build_ext --inplace
# python3 test.py
