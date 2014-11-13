"""
Author : Hu Yuhuang
Date   : 2014-10-21

This code demonstrates and tests:

- convolutional autoencoder
- sparse convolutional autoencoder

"""

# system library modules

import os, sys, time;

try:
    import PIL.image as Image;
except ImportError:
    import Image;

# public library modules

import numpy as np;
import theano;
import theano.tensor as T;

# private library modules

from cae_tools import load_data;
from conv_aes.ConvAE import ConvAE;
from conv_aes.ConvAE import ConvRWAE;
from dlt_utils import tile_raster_images;


### Global wise parameter

print '... Loading data and parameters'

batch_size=500;                     # number of images in each batch
n_epochs=20;                        # number of experiment epochs
learning_rate=0.1;                  # learning rate of SGD
nkerns=50;                          # number of feature maps in ConvAE
dataset='data/mnist.pkl.gz';        # address of data
rng = np.random.RandomState(23455); # random generator

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

index=T.lscalar(); # batch index

x=T.matrix('x');  # input data source
y=T.ivector('y'); # input data label

ishape=(28,28); # image shape

print '... Input data and parameters are loaded'

### Building model

print '... Building CAE model'

data_in=x.reshape((batch_size, 1, 28, 28));

CAE=ConvRWAE(rng,
             data_in=data_in,
             image_shape=(batch_size, 1, 28, 28),
             filter_shape=(nkerns, 1, 14, 14));

cost, updates=CAE.get_cost_update(learning_rate=learning_rate);

## Test model

test_model=theano.function(
                           [index],
                           cost,
                           givens={
                                   x: test_set_x[index*batch_size:(index+1)*batch_size]
                                   }
                           );

## Validate model

validate_model=theano.function(
                               [index],
                               cost,
                               givens={
                                       x: valid_set_x[index*batch_size:(index+1)*batch_size]
                                       }
                               );
## Train model

train_model=theano.function(
                            inputs=[index],
                            outputs=cost,
                            updates=updates,
                            givens={
                                    x:train_set_x[index*batch_size:(index+1)*batch_size]
                                    }
                            );

print '... CAE model is built';

print '... Training CAE'

start_time = time.clock();

for epoch in xrange(n_epochs):
    c=[];

    for batch_index in xrange(n_train_batches):
        c.append(train_model(batch_index));

    print 'Training epoch %d, cost' % epoch, np.mean(c);

end_time=time.clock();

print 'Training is complete in %.2fm' % ((end_time-start_time)/60.);


#I=np.zeros((nkerns, 196));
#A=CAE.hidden_layer.W.get_value(borrow=True);
#for i in xrange(nkerns):
#    I[i,:]=A[i,:,:].flatten(1);

#image = Image.fromarray(
#                        tile_raster_images(X=I,
#                        img_shape=(14, 14), tile_shape=(5, 10),
#                        tile_spacing=(2, 2))
#                        )
#image.save('data/filter_image.png')



patience = 10000;
patience_increase = 2;

improvement_threshold=0.995;

validation_frequency=min(n_train_batches, patience/2);

best_params=None;

best_validation_loss = np.inf;
best_iter = 0;
test_score = 0.;
start_time = time.clock();

epoch = 0;
done_looping = False;

while (epoch < n_epochs) and (not done_looping):
    epoch=epoch+1;

    for minibatch_index in xrange(n_train_batches):
        iteration=(epoch-1)*n_train_batches+minibatch_index;

        if iteration%100==0:
            print 'training @ iter =', iteration;

        cost_ij=train_model(minibatch_index);

        if (iteration + 1) % validation_frequency == 0:

            # compute zero-one loss on validation set
            validation_losses = [validate_model(i) for i
                                 in xrange(n_valid_batches)]
            this_validation_loss = np.mean(validation_losses)
            print('epoch %i, minibatch %i/%i, validation error %f %%' %
                  (epoch, minibatch_index + 1, n_train_batches,
                   this_validation_loss * 100.))

                # if we got the best validation score until now
            if this_validation_loss < best_validation_loss:

                    # improve patience if loss improvement is good enough
                if this_validation_loss < best_validation_loss * \
                    improvement_threshold:
                    patience = max(patience, iteration * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iteration

                    # test it on the test set
                    test_losses = [
                                   test_model(i)
                                   for i in xrange(n_test_batches)
                                   ]
                    test_score = np.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                           (epoch, minibatch_index + 1, n_train_batches,
                            test_score * 100.))

        if patience <= iteration:
            done_looping = True
            break


end_time = time.clock();
print('Optimization complete.');
print('Best validation score of %f %% obtained at iteration %i,'
      'with test performance %f %%' %
      (best_validation_loss * 100., best_iter + 1, test_score * 100.));
print >> sys.stderr, ('The code for file ' +
                      os.path.split(__file__)[1] +
                      ' ran for %.2fm' % ((end_time - start_time) / 60.));

print '... CAE training is completed'
