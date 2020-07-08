# Propagation-Filter-CNN
- Based on zilongzhong/SSRN (https://github.com/zilongzhong/SSRN.git) open source framework
- Adding pf-cnn， cnn， pca-cnn， gabor-cnn， emp-cnn code ( all the cnns have the same structure ). Some codes need to obtain MATLAB filtered data in advance( Useing the pf.m or gabor.m to get the relevant datasets ). 

## Parameter:
The data sets and  number of training samples can be selected within the code, such as:
- DataSetName = 'Indianpines'      # Indianpines  Salinas  PaviaU
- train_num = 400.0

## Prerequisites:
- TensorFlow 1.3.0 + Keras 2.0.6 on Python 3.6.
- scipy
- collections
- sklearn
- numpy

## Usage:
- python pf_cnn.py     
- python gabor_cnn.py
