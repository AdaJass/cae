"""Class of ConvNet Layer

Author : Hu Yuhuang
Date   : 2014-10-21

This code contains a class for ConvNet layer.
It serves as the fundamental building blocks of
ConvNet AEs.

Example
-------

Notes
-----


"""

# public library

import numpy;

import theano;
import theano.tensor as T;
from theano.tensor.signal import downsample;
from theano.tensor.nnet import conv;

class ConvNetLayer(object):
    """ConvNet Layer
    
    This class contains a basic implementation of ConvNet Layer.
    The idea is inspired by LISA lab's deep learning tutorial:
    http://deeplearning.net/tutorial/lenet.html 
    
    """

    def __init__(self,
                 rng,
                 data_in,
                 filter_shape,
                 image_shape,
                 if_pool=False,
                 poolsize=(2, 2),
                 border_mode='valid'):
        """Initialize a ConvNet Layer
        
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
             
        if_pool : bool
            Indicates if there is a max-pooling process in the layer.
            
        pool_size : tuple or list of length 2
            the pooling factor (#rows, #cols)
            
        border_mode : string
            convolution mode,
            "valid" for valid convolution;
            "full" for full convolution;
        """

        assert image_shape[1] == filter_shape[1];
        self.input = data_in;

        fan_in = numpy.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) / numpy.prod(poolsize))
        
        
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype='float32'
            ),
            borrow=True
        )

        b_values = numpy.zeros((filter_shape[0],), dtype='float32')
        self.b = theano.shared(value=b_values, borrow=True)

        conv_out = conv.conv2d(
            input=data_in,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape,
            border_mode=border_mode
        )

        if (if_pool==True): 
            pooled_out = downsample.max_pool_2d(
                input=conv_out,
                ds=poolsize,
                ignore_border=True
            )
        else :
            pooled_out=conv_out;

        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        self.params = [self.W, self.b]
