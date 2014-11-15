"""
Author : Hu Yuhuang
Date   : 2014-11-15

This code is to demonstrate and tests:

- MLP network
- RWMLP network
"""

# system library modules

import os, sys, time;

try:
    import PIL.image as Image;
except ImportError:
    import Image;

# public library modules

import numpy;
import theano;
import theano.tensor as T;

# private library modules

from cae_tools import load_data;
from cae_tools import MLP;
from cae_tools import RWMLP;
#from cae_tools import RWMLP;
from dlt_utils import tile_raster_images;

## global wise parameter

print "... Loading data and parameters";

batch_size=20;                      # number of images in each batch
n_epochs=1000;                      # number of experiment epochs
learning_rate=0.1;                  # learning rate of SGD
L1_reg=0.00;
L2_reg=0.0001;
dataset='data/mnist.pkl.gz';        # address of data
rng = numpy.random.RandomState(23455); # random generator

n_in=28*28;
n_hidden=500;
n_out=10;

datasets=load_data(dataset);

### Loading and preparing dataset
train_set_x, train_set_y = datasets[0];
valid_set_x, valid_set_y = datasets[1];
test_set_x, test_set_y = datasets[2];

n_train_batches=train_set_x.get_value(borrow=True).shape[0];
n_valid_batches=valid_set_x.get_value(borrow=True).shape[0];
n_test_batches=test_set_x.get_value(borrow=True).shape[0];

n_train_batches /= batch_size; # number of train data batches
n_valid_batches /= batch_size; # number of valid data batches
n_test_batches /= batch_size;  # number of test data batches


print "... Build the model"

index=T.lscalar(); # batch index

x=T.matrix('x');  # input data source
y=T.ivector('y'); # input data label

#mlp=MLP(rng,
#        data_in=x,
#        n_in=n_in,
#        n_hidden=n_hidden,
#        n_out=n_out);

mlp=RWMLP(rng,
          data_in=x,
          n_in=n_in,
          n_hidden=n_hidden,
          n_out=n_out);

cost, updates = mlp.get_cost_update(y);

test_model = theano.function(inputs=[index],
                             outputs=mlp.errors(y),
                             givens={x: test_set_x[index * batch_size:(index + 1) * batch_size],
                                     y: test_set_y[index * batch_size:(index + 1) * batch_size]});

validate_model = theano.function(inputs=[index],
                                 outputs=mlp.errors(y),
                                 givens={x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                                         y: valid_set_y[index * batch_size:(index + 1) * batch_size]});

train_model = theano.function(inputs=[index],
                              outputs=cost,
                              updates=updates,
                              givens={x: train_set_x[index * batch_size: (index + 1) * batch_size],
                                      y: train_set_y[index * batch_size: (index + 1) * batch_size]});

print "... training"

patience = 10000;
patience_increase = 2;
improvement_threshold = 0.995;
validation_frequency = min(n_train_batches, patience / 2);
best_validation_loss = numpy.inf;
best_iter = 0;
test_score = 0.;
start_time = time.clock();

epoch = 0;
done_looping = False;

while (epoch < n_epochs) and (not done_looping):
    epoch = epoch + 1;
    for minibatch_index in xrange(n_train_batches):
        
        minibatch_avg_cost = train_model(minibatch_index);
        iter = (epoch - 1) * n_train_batches + minibatch_index;
        
        if (iter + 1) % validation_frequency == 0:
            # compute zero-one loss on validation set
            validation_losses = [validate_model(i) for i
                                 in xrange(n_valid_batches)];
            this_validation_loss = numpy.mean(validation_losses);
            
            print('epoch %i, minibatch %i/%i, validation error %f %%' %
                  (epoch,
                   minibatch_index + 1,
                   n_train_batches,
                   this_validation_loss * 100.));
            
            # if we got the best validation score until now
            if this_validation_loss < best_validation_loss:
                #improve patience if loss improvement is good enough
                if (this_validation_loss < best_validation_loss * improvement_threshold):
                    patience = max(patience, iter * patience_increase);
                    
                    best_validation_loss = this_validation_loss;
                    best_iter = iter;
                    
                    # test it on the test set
                    test_losses = [test_model(i) for i
                                   in xrange(n_test_batches)];
                    test_score = numpy.mean(test_losses);
                    
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.));
                    
            if patience <= iter:
                done_looping = True;
                break;
                        
end_time = time.clock();

print(('Optimization complete. Best validation score of %f %% '
       'obtained at iteration %i, with test performance %f %%') %
      (best_validation_loss * 100., best_iter + 1, test_score * 100.));
print >> sys.stderr, ('The code for file ' +
                      os.path.split(__file__)[1] +
                      ' ran for %.2fm' % ((end_time - start_time) / 60.));