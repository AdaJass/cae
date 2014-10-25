"""Convolutional Autoencoder

Author : Hu Yuhuang
Date   : 2014-10-21

Examples
--------

Notes
-----

"""

import numpy;

import theano;
import theano.tensor as T;
from theano.tensor.signal import downsample;
from theano.tensor.nnet import conv;

from ConvNet import ConvNetLayer;

class ConvAE(object):
    """Convolutional Auto-encoder
    
    This class presents implementation of Convolutional Auto-encoder.
    It consists of one input layer and two ConvNet layers.
    """
    
    def __init__(self,
                 rng,
                 data_in,
                 image_shape,
                 filter_shape,
                 ):
        """Initializing a convolutional auto-encoder
        
        Parameters
        ----------
        rng : numpy.random.RandomState
            a random number generator for initializing weights.
            
        data_in : theano.tensor.dtensor4
            symbolic image tensor, of shape image_shape
            
        filter_shape : tuple or list of length 4
            (number of filters, number of input feature maps,
             filter height, filter width)
        
        image_shape : tuple or list of length 4
            (batch size, number of input feature maps,
             image height, image width)
        """
        
        self.input=data_in;
        
        hidden_layer=ConvNetLayer(rng,
                                  data_in=data_in,
                                  image_shape=image_shape,
                                  filter_shape=filter_shape,
                                  border_mode='full'
                                  );
        
        image_shape_temp=numpy.asarray(image_shape);
        filter_shape_temp=numpy.asarray(filter_shape);
        image_shape_hidden=(image_shape_temp[0],
                            filter_shape_temp[0],
                            image_shape_temp[2]+filter_shape_temp[2]-1,
                            image_shape_temp[3]+filter_shape_temp[3]-1,);
                            
        filter_shape_hidden=(image_shape_temp[1],
                             filter_shape_temp[0],
                             filter_shape_temp[2],
                             filter_shape_temp[3]);
                             
        recon_layer=ConvNetLayer(rng,
                                 data_in=hidden_layer.output,
                                 image_shape=image_shape_hidden,
                                 filter_shape=filter_shape_hidden,
                                 border_mode='valid');
                                 
        self.hidden_layer=hidden_layer;
        self.recon_layer=recon_layer;
        
        self.params=hidden_layer.params+recon_layer.params;
        
    def get_hidden_values(self):
        """Get hidden layer's value
        """
        
        return self.hidden_layer.output;
        
    def get_reconstruction(self):
        """Get reconstruction of auto-encoder
        """
        
        return self.recon_layer.output;
    
    def get_cost_update(self, learning_rate=0.1):
        """Get cost updates
        
        Parameters
        ----------
        learning_rate : float
            learning rate of sgd
        """
        L=T.sum(T.pow(T.sub(self.get_reconstruction(), self.input),2), axis=1);
        
        cost = 0.5*T.mean(L);

        grads=T.grad(cost, self.params);
        
        updates = [
                   (param_i, param_i-learning_rate*grad_i)
                   for param_i, grad_i in zip(self.params, grads)
                   ];
           
        return (cost, updates);