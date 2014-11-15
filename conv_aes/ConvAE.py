"""Convolutional Autoencoder

Author : Hu Yuhuang
Date   : 2014-10-21

Examples
--------

Notes
-----

"""

# public library

import numpy;

import theano;
import theano.tensor as T;
from theano.tensor.signal import downsample;
from theano.tensor.nnet import conv;

# private library

from ConvNet import ConvNetLayer;
from ConvNet import ConvNetRandomWeightsLayer;
import nntool as nnt;


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

class ConvRWAE(object):
    """Convolutional Random Weights Auto-encoder

    This class is to construct Convolutional Random Weights Auto-encoder.
    """

    def __init__(self,
                 rng,
                 data_in,
                 image_shape,
                 filter_shape,
                 ):
        """Initialize a Convolutional Random Weights Auto-encoder.

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
        
        encode_layer=ConvNetRandomWeightsLayer(rng,
                                               data_in=data_in,
                                               image_shape=image_shape,
                                               filter_shape=filter_shape,
                                               border_mode='full');
        
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
                             
        decode_layer=ConvNetRandomWeightsLayer(rng,
                                               data_in=encode_layer.output,
                                               image_shape=image_shape_hidden,
                                               filter_shape=filter_shape_hidden,
                                               border_mode='valid');
                                 
        self.encode_layer=encode_layer;
        self.decode_layer=decode_layer;
        
        self.WEIGHTS=[self.encode_layer.W, self.decode_layer.W];
        self.BIAS=[self.encode_layer.b, self.decode_layer.b];
        self.RW=[self.encode_layer.B, self.decode_layer.B];

        self.params=self.WEIGHTS+self.BIAS;

    def get_encode_values(self):
        """Get hidden layer's value
        """
        
        return self.encode_layer.output;
        
    def get_decode(self):
        """Get reconstruction of auto-encoder
        """
        
        return self.decode_layer.output;
    
    def get_cost_update(self, learning_rate=0.1):
        """Get cost updates
        
        Parameters
        ----------
        learning_rate : float
            learning rate of sgd
        """
        L=T.sum(T.pow(T.sub(self.get_decode(), self.input),2), axis=1);
        cost = 0.5*T.mean(L);

        d_b=T.grad(cost, self.BIAS);
        d_net_out=T.grad(cost, self.decode_layer.pooled_out);

        d_b_decode=T.grad(cost, self.decode_layer.b);
        d_W_decode=T.grad(cost, self.decode_layer.W);
        print d_b_decode.type();
        print d_W_decode.type();
        #d_b_encode=T.sum(d_net_out, axis=[0,1,2]);
        #print d_b_encode.type();
        #d_W_decode=self.decode_layer.getCP(data_in=self.encode_layer.output,
        #                                   filters=d_net_out);
        #print d_W_decode.shape;

        d_net_in=self.decode_layer.getConvPoolB(data_in=d_net_out,
                                                filters=self.decode_layer.B);
        #T.dot(self.decode_layer.B, d_net_out);
        #print d_net_in.type();
        d_net_in_delta=d_net_in*self.encode_layer.d_activation(self.encode_layer.pooled_out);
        print d_net_in_delta.type();
        d_b_encode=T.sum(d_net_in_delta, axis=[0,1,2]);
        d_W_encode=T.dot(d_net_in_delta, self.input.T);
        print d_W_encode.type();

        d_W=[d_W_encode, d_W_decode];

        updates_weights=[(param_i, param_i-learning_rate*d_W_i)
                         for param_i, d_W_i in zip (self.WEIGHTS, d_W)];

        updates_bias=[(param_i, param_i-learning_rate*d_b_i)
                      for param_i, d_b_i in zip(self.BIAS, d_b)];

        updates=updates_weights+updates_bias;

        #L_B=T.sum(T.pow(T.sub(self.recon_layer.output_B, self.input),2), axis=1);
        #cost_B = 0.5*T.mean(L_B);

        #grad_weights=T.grad(cost, self.WEIGHTS);

        #updates_weights=[(param_i, param_i-learning_rate*(grad_i+learning_rate*rw_i))
        #                for param_i, grad_i, rw_i in zip(self.WEIGHTS, grad_weights, self.RW)];
        
        #grad_bias=T.grad(cost, self.BIAS);

        #updates_bias=[(param_i, param_i-learning_rate*grad_i)
        #              for param_i, grad_i in zip(self.BIAS, grad_bias)];

        #updates=updates_weights+updates_bias;
           
        return (cost, updates);