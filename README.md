# AMD-DAS-Brain-CT-Segmentation

This is the implementation of a paper titled: An Unsupervised Domain Adaptation Brain CT Segmentation Method Across Image Modalities and Diseases, currently under review by Journal Expert Systems with Applications. More details can be seen in our upcoming paper.

## Built With

* [PyTorch](https://pytorch.org/)

## Getting Started

### Prerequisites

Python 3

CPU or NVIDIA GPU + CUDA CuDNN

### Stage 1: Training of Image Synthesis Network
In the first stage, we train a pseudo-CT image synthesis network to minimize the difference between the two modalities.

### Stage 2: Domain adaptation Segmentation
In the second stage, we use labelled pseudo-CT images (obtained from the first stage network) and unlabeled CT images to train domain adaptation segmentation network.

### Matching Mechanism
In the training process of the image synthesis network, inputting the image pairs selected by this method can improve the performance of downstream tasks.


## Contact

Daqiang Dong: dongdaqiang@emails.bjut.edu.cn

Guanghui Fu: guanghui.fu@icm-institute.org; aslanfu123@gmail.com


