# Locality-constrained-Affine-Subspace-Coding
This is the implementation of Locality-constrained affine subspace coding for image classification and retrieval  
_Now in experimental release, suggestions welcome._  
## Paper    
B. Zhang, Q. Wang, X. Lu, F. Wang and P. Li. Locality-constrained affine subspace coding for image classification and retrieval, Pattern Recognition, https://doi.org/10.1016/j.patcog.2019.107167    
## Introduction    
Feature coding is a key component of the bag of visual words (BoVW) model, which is designed to improve image classification and retrieval performance. In the feature coding process, each feature of an image is nonlinearly mapped via a dictionary of visual words to form a high-dimensional sparse vector. Inspired by the well-known locality-constrained linear coding (LLC), we present a locality constrained affine subspace coding (LASC) method to address the limitation whereby LLC fails to consider the local geometric structure around visual words. LASC is distinguished from all the other coding methods since it constructs a dictionary consisting of an ensemble of affine subspaces. As such, the local geometric structure of a manifold is explicitly modeled by such a dictionary. In the process of coding, each feature is linearly decomposed and weighted to form the first-order LASC vector with respect to its top-k neighboring subspaces. To further boost performance, we propose the second-order LASC vector based on information geometry.       
In this repository, we release the implementations of two feature coding methods (LASC and FV). It supports both Linux OS and Windows OS, and it provides basic pipelines image classification.    
## Framework  
![](https://github.com/icyzhang0923/Locality-constrained-Affine-Subspace-Coding/blob/master/LASC%20_framework.jpg)  
* Figure 1(a) shows our idea of LASC and a comparison with LLC.  
* Figure 1(b) demonstrates the feature coding process and the dictionary of LLC is visually compared with that of LASC.  
## Environment     
* Linux OS and Windows OS    
* MATLAB 2017b      
* CUDA 8.0+    
* CUDNN 5.0+    
## Installation and Usage  
* Always use `git clone --recursive git@github.com:icyzhang0923/Locality-constrained-Affine-Subspace-Coding.git` to clone this project.  
* Download the image classification datasets ([VOC2007](https://pjreddie.com/projects/pascal-voc-dataset-mirror/), [Caltech256](http://www.vision.caltech.edu/Image_Datasets/Caltech256/), [Scene67](http://web.mit.edu/torralba/www/indoor.html), and [SUN397](https://vision.princeton.edu/projects/2010/SUN/)).      
* Download the pretrained [VGG-VD-16](http://www.vlfeat.org/matconvnet/pretrained/) model on ImageNet dataset.
* Download and complie the [VLFeat](http://www.vlfeat.org/) and [MatConvNet](http://www.vlfeat.org/matconvnet/) tool box. Please refer to the official website for more details.  
* Run the main function `traintest.m` to start the experiments.  
## Optimal Parameters Settings for LASC  
* Affine subspace dictionary size: 128  
* Number of nearest subspaces: 5  
* Subspace dimensions: 256  
* Regularizations in objective function: ridge regression  
* Proximity measures: Euclidean distance between feature to cluster center  
* Fetrue normalization method: matrix  
## Image Classification Results   
  
Methods | VOC 2007 | Caltech 256 | MIT67 | SUN397  
:----:    | :----:     |:----:         |:----:   | :-----:  
FV  | 84.9 | 86.2 | 81.0 | 64.3
LASC | 87.6 | 88.2 | 81.5 | 64.5  

## Acknowledgments    
* We thank the [VLFeat](http://www.vlfeat.org/) and [MatConvNet](http://www.vlfeat.org/matconvnet/) team to develop these two useful computer vision toolbox.    
* We thank the the developers of the optimization toolbox [SPAMS](http://spams-devel.gforge.inria.fr/).     
## Contact Information      
Should you have any question or suggestion regarding our released code, please feel free to contact us: icyzhang@mail.dlut.edu.cn.  







