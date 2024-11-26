#!/bin/bash

# Output build directory path.
CTR_BUILD_DIR=/build

echo "Building the project..."
echo $(ls) 

cp -r fast-hadamard-transform ${CTR_BUILD_DIR}/

# cd fast-hadamard-transform
# python3 setup.py build
