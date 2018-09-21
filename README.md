# building-autoencoders-in-Pytorch
This is a reimplementation of the blog post "Building Autoencoders in Keras". Instead of using MNIST, this project uses CIFAR10.

## Current Results (Trained on Tesla K80 using Google Colab)
First attempt: (BCEloss=~0.57)  
![decode](/weights/colab_predictions.png)

Best Predictions so far: (BCEloss=~0.555)  
![decode](/weights/colab_predictions2.png)

Targets:  
![target](/weights/colab_tar.png)

## Previous Results (Trained on GTX1070)
First attempt:  
![decode](/weights/decoded_img.png)

Second attempt:  
![decode](/weights/decoded_img2.png)

Targets:  
![decode](/weights/target.png)
