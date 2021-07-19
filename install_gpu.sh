#!/bin/bash
# A bash script to install PyNIROM with GPU support

wget -O tfdiffeq.zip https://github.com/titu1994/tfdiffeq/archive/ef646f85cbd0821749a03e7ab51e03e16798fab1.zip && unzip -q tfdiffeq.zip
cd tfdiffeq-ef646f85cbd0821749a03e7ab51e03e16798fab1
pip install .[tf-gpu]
cd ..
rm -rf tfdiffeq-ef646f85cbd0821749a03e7ab51e03e16798fab1 && rm -rf tfdiffeq.zip
