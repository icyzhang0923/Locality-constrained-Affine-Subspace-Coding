# Locality-constrained-Affine-Subspace-Coding
This is the implementation of Locality-constrained affine subspace coding for image classification and retrieval  
_Now in experimental release, suggestions welcome._  
## Paper    
B. Zhang, Q. Wang, X. Lu, F. Wang and P. Li. Locality-constrained affine subspace coding for image classification and retrieval, Pattern Recognition, https://doi.org/10.1016/j.patcog.2019.107167    
## Introduction    
Feature coding is a key component of the bag of visual words (BoVW) model, which is designed to improve image classification and retrieval performance. In the feature coding process, each feature of an image is nonlinearly mapped via a dictionary of visual words to form a high-dimensional sparse vector. Inspired by the well-known locality-constrained linear coding (LLC), we present a locality-constrained affine subspace coding (LASC) method to address the limitation whereby LLC fails to consider the local geometric structure around visual words. LASC is distinguished from all the other coding methods since it constructs a dictionary consisting of an ensemble of affine subspaces. As such, the local geometric structure of a manifold is explicitly modeled by such a dictionary. In the process of coding, each feature is linearly decomposed and weighted to form the first-order LASC vector with respect to its top-k neighboring subspaces. To further boost performance, we propose the second-order LASC vector based on information geometry.       
In this repository, we release the implemetntations of several classical feature coding methods (LASC, SC, LLC, VLAD, and FV). We also give the bilinear pooling method for deep features as the comparison method. It supports both Linux OS and Windows OS, and it provides basic pipelines for most of image classification and image retrieval datasets.    
## Framework  
## Environment     
* Linux OS and Windows OS    
* MATLAB 2017b      
* CUDA 8.0+    
* CUDNN 5.0+    
## Installation and Usage  
* Always use `git clone --recursive git@github.com:icyzhang0923/Locality-constrained-Affine-Subspace-Coding.git` to clone this project.  
* Download the image classification datasets ([VOC2007](https://pjreddie.com/projects/pascal-voc-dataset-mirror/), [Caltech256](http://www.vision.caltech.edu/Image_Datasets/Caltech256/), [Scene67](http://web.mit.edu/torralba/www/indoor.html), and [SUN397](https://vision.princeton.edu/projects/2010/SUN/)) or the image retrieval datasets ([INRIA Holidays](http://lear.inrialpes.fr/~jegou/data.php), [UKBench](http://www.vis.uky.edu/~stewe/ukbench/), and [Oxford5K](http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/)).    
* Download the pretrained [VGG-VD-16](http://www.vlfeat.org/matconvnet/pretrained/) model on ImageNet dataset.
* Complie the MatConvNet and VLFeat toolbox refer to http://www.vlfeat.org/matconvnet.
* Run the main function `traintest.m` to start the experiment.  
## Optimal Parameters Settings for LASC  
* Affine subspace dictionary size: 128  
* Number of nearest subspaces: 5  
* Subspace dimensions: 256  
* Regularizations in objective function: ridge regression  
* Proximity measures: Euclidean distance between feature to cluster center  
* Fetrue normalization method: matrix  





