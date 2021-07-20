#!/bin/bash
# A bash script to install PyNIROM with GPU support

conda install numpy==1.20.3 scipy==1.6.2 matplotlib==3.3.4 notebook==6.4.0
conda install -c conda-forge scikit-learn==0.24.2
conda install tensorflow-gpu==2.2
conda install tensorflow-probability==0.7

wget -O tfdiffeq.zip https://github.com/titu1994/tfdiffeq/archive/ef646f85cbd0821749a03e7ab51e03e16798fab1.zip && unzip -q tfdiffeq.zip
cd tfdiffeq-ef646f85cbd0821749a03e7ab51e03e16798fab1
pip install .[tf-gpu]
cd ..
rm -rf tfdiffeq-ef646f85cbd0821749a03e7ab51e03e16798fab1 && rm -rf tfdiffeq.zip

python3 setup.py install
