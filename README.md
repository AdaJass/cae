Convolutional Auto-encoders
===

This repo is aiming to provide a set of Convolutional Auto-encoder implementations (CAEs) for Deep Learning.

Convolutional Auto-encoder did not draw so much attention since it's proposed. However, I think that this model offered a very nice unsupervised feature learning model from neural network persepective.

Following dependencies are required:
+ Python 2.7.8
+ `numpy`
+ `scipy`
+ `theano`

You can also use _Anaconda_ directly, this python distribution will offer you all dependencies.

##Updates

+ ConvNet layer [20141021]
+ Original ConvNet Auto-encoder [20141021]
+ Tested for AWS GPU instance [20141025]
+ Example for classification [20141025]

##To-do

+ Multiple activation function support
+ Support functions for ConvNet Layer and ConvNet AE
+ Sparse ConvNet Auto-encoder
+ Stacked ConvNet Auto-encoder

##Notes

+ All experiment in this repo are conducted on GPU, in order to run it faster, you are suggested to have a GPU on your machine.

+ If you forked this repo, use and modifiy `update.sh` to avoid updating unnecessary files and data.

+ Tested on AWS GPU instance, the performance looks like Tesla K20C.

##Contacts

Hu Yuhuang  
Advanced Robotic Lab  
Department of Artificial Intelligence  
Faculty of Computer Science & IT  
University of Malaya  
Email: duguyue100@gmail.com
