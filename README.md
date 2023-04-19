# ViT1
initial attempt at a vision transformer to carry out binary classification of whether a thermal image of a PV module contains an anomaly or not

I tried to piece together my own from scratch, because the images in my dataset came in the form of a temperature distribution, and I was curious to see whether my own custom features would perform better

The ViT1.pth file contains the weights after 10 epochs of training on a dataset of 3000 images

The ViT_loss.png file illustrates the loss through all 600 steps (chose a batch size of 50). As can be seen, it does not optimise well

The ViT_class.py file contains the ViT object 

Mainly went off the processes described in:
https://towardsdatascience.com/implementing-visualttransformer-in-pytorch-184f9f16f632
https://medium.com/mlearning-ai/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c


