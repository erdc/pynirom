#!/bin/bash
# A bash script to install PyNIROM with GPU support

git clone --depth 1 --branch 6113f10 git@github.com:titu1994/tfdiffeq.git
cd tfdiffeq
pip install .[tf-gpu]
cd ..
rm -rf tfdiffeq
