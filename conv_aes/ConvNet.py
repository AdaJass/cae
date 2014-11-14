"""Class of ConvNet Layer

Author : Hu Yuhuang
Date   : 2014-10-21

This code contains a class for ConvNet layer.
It serves as the fundamental building blocks of
ConvNet AEs.

Example
-------

1. Construct a ConvNet Layer without Pooling using full convolution and ReLU activation

layer = ConvNetLayer(rng,
                     data_in=data_in,
                     image_shape=image_shape,
                     filter_shape=filter_shape,
                     border_mode='full',
                     activate_mode='relu');
                     
2. Constuct a ConvNet Layer with Max-pooling using valid convolution and tanh activation

layer = ConvNetLayer(rng,
                     data_in=data_in,
                     image_shape=image_shape,
                     filter_shape=filter_shape,
                     if_pool=True);

3. Construct a ConvNetRandomWeightsLayer

layer = ConvNetRandomWeightsLayer(rng,
                                  data_in=data_in,
                                  image_shape=image_shape,
                                  filter_shape=filter_shape,
                                  if_pool=False)

Notes
-----

1. Now the ConvNetLayer supports two activation functions: tanh and ReLU

2. A trained network can be use to get activation from outside untrained input.

3. I predefined 4 parameters in __init__ function. They can be used to construct
   different network configuration.
"""

# public library

import numpy;

import theano;
import theano.tensor as T;
from theano.tensor.signal import downsample;
from theano.tensor.nnet import conv;

# private library

import nntool as nnt;

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
                 pool_size=(2, 2),
                 border_mode='valid',
                 activate_mode='tanh'):
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
            
        activate_mode : string
            activation mode,
            "tanh" for tanh function;
            "relu" for ReLU function;
            "sigmoid" for Sigmoid function;
            "softplus" for Softplus function
        """

        assert image_shape[1] == filter_shape[1];
        
        ## Document all input parameters
        self.input = data_in;
        self.filter_shape = filter_shape;
        self.image_shape = image_shape;
        self.if_pool = if_pool;
        self.pool_size = pool_size;
        self.border_mode = border_mode;
        self.activate_mode = activate_mode;

        if (self.activate_mode=="tanh"):
            self.activation=nnt.tanh;
            self.d_activation=nnt.d_tanh;
        elif (self.activate_mode=="relu"):
            self.activation=nnt.relu;
        elif (self.activate_mode=="sigmoid"):
            self.activation=nnt.sigmoid;
            self.d_activation=nnt.d_sigmoid;
        elif (self.activate_mode=="softplus"):
            self.activation=nnt.softplus;
            self.d_activation=nnt.d_softplus;
        else:
            raise ValueError("Value %s is not a valid choice of activation function"
                             % self.activate_mode);

        ## Generate weights for each filters
        fan_in = numpy.prod(filter_shape[1:]);
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) / numpy.prod(pool_size));
        
        W_bound = numpy.sqrt(6. / (fan_in + fan_out));
        self.W = theano.shared(numpy.asarray(
                                             rng.uniform(low=-W_bound,
                                                         high=W_bound,
                                                         size=filter_shape),
                                             dtype='float32'
                                             ),
                               borrow=True);
        self.bound=W_bound;

        ## Generate bias values, initialize as 0
        b_values = numpy.zeros((filter_shape[0],), dtype='float32');
        self.b = theano.shared(value=b_values, borrow=True);

        ## Get network pool

        self.pooled, self.pooled_out=self.getConvPool(data_in=self.input,
                                                      filters=self.W,
                                                      filter_shape=self.filter_shape,
                                                      image_shape=self.image_shape,
                                                      bias=self.b,
                                                      if_pool=if_pool,
                                                      pool_size=pool_size,
                                                      border_mode=border_mode);
        
        ## Get network activation
        self.output=self.getInputActivation(pooled_out=self.pooled_out);

        self.params = [self.W, self.b];
        
    def getConvPool(self,
                    data_in,
                    filters,
                    filter_shape,
                    image_shape,
                    bias,
                    if_pool=False,
                    pool_size=(2,2),
                    border_mode='valid'):
        """get convolution and max pooling result
        
        Parameters
        ----------
        data_in : theano.tensor.dtensor4
            symbolic image tensor, of shape image_shape
            
        filters : theano.tensor.dtensor4
            weights for all feature maps
            
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
        
        ## Perform 2d convolution on selected image
        conv_out=conv.conv2d(input=data_in,
                             filters=filters,
                             filter_shape=filter_shape,
                             image_shape=image_shape,
                             border_mode=border_mode);
        
        ## Perform max-pooling
        if (if_pool==True):
            pooled_out = downsample.max_pool_2d(input=conv_out,
                                                ds=pool_size,
                                                ignore_border=True);
        else:
            pooled_out = conv_out;
            
        return pooled_out, pooled_out+bias.dimshuffle('x', 0, 'x', 'x');

    def getCP(self,
              data_in,
              filters,
              if_pool=False):
        return self.getConvPool(data_in=data_in,
                                filters=filters,
                                filter_shape=self.filter_shape,
                                image_shape=self.image_shape,
                                bias=self.b,
                                if_pool=if_pool,
                                pool_size=self.pool_size,
                                border_mode=self.border_mode);
        
    def getActivation(self,
                      pooled_out):
        """Get network activation based on activate mode
        """
                
        ## Calculate network activation
        act=self.activation(pooled_out);
                    
        return act;
    
    def getNetActivation(self,
                         pooled_out):
        """Get a trained ConvNet Layer's activation
        """
        
        return self.getActivation(pooled_out=pooled_out);
        
    def getInputActivation(self,
                           pooled_out):
        """Get network activation from self.input
        """
        
        return self.getActivation(pooled_out=pooled_out);

class ConvNetRandomWeightsLayer(ConvNetLayer):
    """ConvNet Layer with Random Feedback Weights

    This class implements the idea from the paper:
    Random feedback weights support learning in deep neural networks

    The major difference is that the SGD is updated without using previous step's weights.
    """

    def __init__(self,
                 rng,
                 data_in,
                 filter_shape,
                 image_shape,
                 if_pool=False,
                 pool_size=(2, 2),
                 border_mode='valid',
                 activate_mode='tanh'):
        """Initialize a ConvNet Layer with Random Feedback Weights

        The only difference here is that there is another random weights matrix added.
        """

        super(ConvNetRandomWeightsLayer, self).__init__(rng=rng,
                                                        data_in=data_in,
                                                        filter_shape=filter_shape,
                                                        image_shape=image_shape,
                                                        if_pool=if_pool,
                                                        pool_size=pool_size,
                                                        border_mode=border_mode,
                                                        activate_mode=activate_mode);

        image_shape_temp=numpy.asarray(image_shape);
        filter_shape_temp=numpy.asarray(filter_shape);
        image_shape_B=(image_shape_temp[0],
                       filter_shape_temp[0],
                       image_shape_temp[2]+filter_shape_temp[2]-1,
                       image_shape_temp[3]+filter_shape_temp[3]-1,);
        filter_shape_B=(filter_shape_temp[1],
                        filter_shape_temp[0],
                        filter_shape_temp[2],
                        filter_shape_temp[3]);
        
        self.image_shape_B=image_shape_B;
        self.filter_shape_B=filter_shape_B;
        self.B = theano.shared(numpy.asarray(rng.uniform(low=-self.bound,
                                                         high=self.bound,
                                                         size=filter_shape_B),
                                             dtype='float32'),
                               borrow=True);

        self.params=[self.W, self.b, self.B];

    def getConvPoolB(self,
                     data_in,
                     filters,
                     if_pool=False):
        return self.getConvPool(data_in=data_in,
                                filters=filters,
                                filter_shape=self.filter_shape_B,
                                image_shape=self.image_shape_B,
                                bias=self.b,
                                if_pool=if_pool,
                                pool_size=self.pool_size,
                                border_mode=self.border_mode);