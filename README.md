# Propagation-Filter-CNN
- Based on zilongzhong/SSRN (https://github.com/zilongzhong/SSRN.git) open source framework.
- Adding pf-cnn, cnn, pca-cnn code ( all the cnns have the same structure ). Some codes need to obtain MATLAB filtered data in advance( Useing the pf.m to get the relevant datasets ). 

## Parameter:
The data sets and  number of training samples can be selected within the code, such as:
- DataSetName = 'Indianpines'      # Indianpines  Salinas  PaviaU
- train_num = 400.0

## Prerequisites:
Use environment.yml to build environment
- TensorFlow-gpu 1.13.1 + Keras 2.2.4 on Python 3.6.
- scipy
- collections
- sklearn
- numpy

## Usage:
- python pf_cnn.py     
- python cnn.py
- python pca_cnn.py
