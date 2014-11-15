"""Neural Network Tools

Author : Hu Yuhuang
Date   : 2014-10-21

This code consists of a set of tools for assisting neural
networks operations
"""

# puvlic library

import numpy;

import theano;
import theano.tensor as T;

def tanh(x):
    """Tanh activation function

    Parameter
    ---------
    x : theano tensor
        input of the function
    """
    return T.tanh(x);

def d_tanh(x):
    """Derivative of Tanh activation function

    Parameter
    ---------
    x : theano tensor
        input of function
    """
    return 1.0-T.tanh(x)**2;

def sigmoid(x):
    """Sigmoid activation function

    Parameter
    ---------
    x : theano tensor
        input of the function
    """
    return T.nnet.sigmoid(x);

def d_sigmoid(x):
    """Derivative of Sigmoid function

    Parameters
    ----------
    x : theano tensor
        input of the function
    """
    return T.nnet.sigmoid(x)*(1.0-T.nnet.sigmoid(x));

def relu(x):
    """ReLU activation function

    Parameter
    ---------
    x : theano tensor
        input of the function
    """
    return T.max(0.0, x);

def softplus(x):
    """Softplus activation function

    A smooth approximation of ReLU function

    Parameters
    ----------
    x : theano tensor
        input of the function
    """
    return T.log(1.0+T.exp(x));

def d_softplus(x):
    """Derivative of Softplus function

    The derivative is the sigmoid function

    Parameters
    ---------
    x : theano tensor
        input of the function
    """
    return sigmoid(x);

def softmax(x):
    """Softmax function

    A softmax function

    Parameters
    ----------
    x : theano tensor
        input of the function
    """
    return T.nnet.softmax(x);