#!/bin/bash
# A bash script to install PyNIROM with GPU support

git clone --depth 1 --branch v0.0.1-pre0 https://github.com/titu1994/tfdiffeq.git
cd tfdiffeq
pip install .[tf-gpu]
cd ..
rm -rf tfdiffeq
