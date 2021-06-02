# DCGAN Implementation with Pytorch
- Unofficial implementation of the paper *Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks*


## 0. Develop Environment
```
Docker Image
- pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel
```


## 1. Implementation Details
- model.py : DCGAN model (generator, discriminator)
- train.py : train DCGAN (both generator, discriminator)
- DCGAN - CelebA.ipynb : install library, download dataset, preprocessing, train and result, inference (image generation)
- Details
  * Follow the train details of paper
    * batch size 128, learning rate 0.0002, momentum beta1 0.5 with Adam
    * augmentation : normalize between [-1, 1]


## 2. Reference
- Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks [[paper]](https://arxiv.org/pdf/1511.06434.pdf)
