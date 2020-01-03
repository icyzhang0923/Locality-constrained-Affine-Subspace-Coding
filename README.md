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
## Image Classification Results 

Methods | VOC 2007 | Caltech 256 | MIT67 | SUN397  
----    | ----     |----         |----   | -----  
SC  | 85.6 | 84.4 | 81.2 | 61.7
LLC  | 85.4 | 83.9 | 81.0 | 60.7
VLAD  | 84.7 | 84.9 | 77.5 | 62.2
FV  | 84.9 | 86.2 | 81.0 | 64.3
LASC | 87.6 | 88.1 | 81.5 | 64.3



## image Retrieval Results
Methods | Dim | Holidays | UKB | Oxford5K  
----    | ----     |----         |----   | -----  
SC  | 512 | 88.9 |3.65  |59.1 
LLC  | 512 | 88.1 | 3.61 | 57.0
VLAD  | 512 | 88.8 | 3.71 | 57.5
BCNN  | 512 | 88.8 | 3.78 | 66.8
LASC | 512 | 90.9 | 3.85 | 67.1






