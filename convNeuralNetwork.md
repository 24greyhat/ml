# Convolutional Neural Networks


- Unlike Fully Connected Layers a Convolutional Layer can process multidimensional input data and does not require flattening of input tensors.



## Convolutional Neural Network Kernel

- The kernel is an (N x N matrix) that acts as a filter used to extract the features from the input data:

    - the kernel matrix moves over the input data perfroms the dot product with the sub-region of input data.
    
    -  gets the output as the matrix of dot products.
 
    -  comon kernel sizes are (3x3) sometimes even a (5x5) anything higher is considered excessive and uncommon.




## (1) Feature mapping

  - When passed an image the kernel will capture the first 3x3 square of the image assuming the kernel size is 3x3.
 
  - Then the kernel will then slide to the next 3x3 square in the image and will keep doing so until it has a new condensed version of the input image which is called a feature map.

 

## (2) Kernel Pooling


- Kernel Pooling is the process of encoding and aggregating feature maps into a global feature vector.
  
- The architecture of Convolutional Neural Networks (CNNs) can be regarded as fully convolutional layers followed by the subsequent pooling layers and a linear classifier.



# (3) Linear Layer


- The linear layer following the pooling layer  acts as a classifier.



# (4) Loss

- Loss calculation and optimization.