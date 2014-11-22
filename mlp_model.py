"""
Author : Hu Yuhuang
Date   : 2014-11-21

The code describes MLP models
mainly to perform comparison study between
MLP and RWMLP
"""

# system library modules

import os, sys, time;

# public library modules

import numpy;
import theano;
import theano.tensor as T;

# private library modules

from cae_tools import load_data;
from cae_tools import MLP;
from cae_tools import RWMLP;

def mlp_model(n_in,
              n_hidden,
              n_out,
              dataset='data/mnist.pkl.gz',
              training_portion=1,
              batch_size=20,
              n_epochs=1000,
              learning_rate=0.1,
              L1_reg=0.00,
              L2_reg=0.0001,
              hidden_limits=0.1,
              out_limits=0.1):

    datasets=load_data(dataset);
    rng=numpy.random.RandomState(23455);

    ### Loading and preparing dataset
    train_set_x, train_set_y = datasets[0];
    valid_set_x, valid_set_y = datasets[1];
    test_set_x, test_set_y = datasets[2];
    
    n_train_batches=int(train_set_x.get_value(borrow=True).shape[0]*training_portion);
    n_valid_batches=valid_set_x.get_value(borrow=True).shape[0];
    n_test_batches=test_set_x.get_value(borrow=True).shape[0];
    
    n_train_batches /= batch_size; # number of train data batches
    n_valid_batches /= batch_size; # number of valid data batches
    n_test_batches /= batch_size;  # number of test data batches

    print "... Build the model"

    index=T.lscalar(); # batch index
    
    x=T.matrix('x');  # input data source
    y=T.ivector('y'); # input data label

    mlp=MLP(rng,
            data_in=x,
            n_in=n_in,
            n_hidden=n_hidden,
            n_out=n_out);
    
    rwmlp=RWMLP(rng,
                data_in=x,
                n_in=n_in,
                n_hidden=n_hidden,
                n_out=n_out,
                B_hidden_limits=hidden_limits,
                B_out_limits=out_limits);


    ## MLP model
    mlp_cost, mlp_updates = mlp.get_cost_update(y);

    mlp_test_model = theano.function(inputs=[index],
                                     outputs=mlp.errors(y),
                                     givens={x: test_set_x[index * batch_size:(index + 1) * batch_size],
                                             y: test_set_y[index * batch_size:(index + 1) * batch_size]});
    
    mlp_validate_model = theano.function(inputs=[index],
                                         outputs=mlp.errors(y),
                                         givens={x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                                                 y: valid_set_y[index * batch_size:(index + 1) * batch_size]});
    
    mlp_train_model = theano.function(inputs=[index],
                                      outputs=mlp_cost,
                                      updates=mlp_updates,
                                      givens={x: train_set_x[index * batch_size: (index + 1) * batch_size],
                                              y: train_set_y[index * batch_size: (index + 1) * batch_size]});

    ## RWMLP model

    rwmlp_cost, rwmlp_updates = rwmlp.get_cost_update(y);

    rwmlp_test_model = theano.function(inputs=[index],
                                       outputs=rwmlp.errors(y),
                                       givens={x: test_set_x[index * batch_size:(index + 1) * batch_size],
                                               y: test_set_y[index * batch_size:(index + 1) * batch_size]});
    
    rwmlp_validate_model = theano.function(inputs=[index],
                                           outputs=rwmlp.errors(y),
                                           givens={x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                                                   y: valid_set_y[index * batch_size:(index + 1) * batch_size]});
    
    rwmlp_train_model = theano.function(inputs=[index],
                                        outputs=rwmlp_cost,
                                        updates=rwmlp_updates,
                                        givens={x: train_set_x[index * batch_size: (index + 1) * batch_size],
                                                y: train_set_y[index * batch_size: (index + 1) * batch_size]});

    print "... training"

    validation_frequency = n_train_batches;
    start_time = time.clock();

    mlp_validation_record=numpy.zeros((n_epochs, 1));
    mlp_test_record=numpy.zeros((n_epochs, 1));

    rwmlp_validation_record=numpy.zeros((n_epochs, 1));
    rwmlp_test_record=numpy.zeros((n_epochs, 1));
    
    epoch = 0;
    while (epoch < n_epochs):
        epoch = epoch + 1;
        for minibatch_index in xrange(n_train_batches):
        
            mlp_minibatch_avg_cost = mlp_train_model(minibatch_index);
            iter = (epoch - 1) * n_train_batches + minibatch_index;
        
            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set

                # MLP model loss
                mlp_validation_losses = [mlp_validate_model(i) for i
                                         in xrange(n_valid_batches)];
                mlp_validation_record[epoch-1] = numpy.mean(mlp_validation_losses);

                print 'MLP MODEL';
            
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches, mlp_validation_record[epoch-1] * 100.));

                mlp_test_losses = [mlp_test_model(i) for i
                                   in xrange(n_test_batches)];
                mlp_test_record[epoch-1] = numpy.mean(mlp_test_losses);
                
                print(('     epoch %i, minibatch %i/%i, test error %f %%') %
                      (epoch, minibatch_index + 1, n_train_batches, mlp_test_record[epoch-1] * 100.));

    epoch = 0;
    while (epoch < n_epochs):
        epoch = epoch + 1;
        for minibatch_index in xrange(n_train_batches):
        
            rwmlp_minibatch_avg_cost = rwmlp_train_model(minibatch_index);
            iter = (epoch - 1) * n_train_batches + minibatch_index;
        
            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set

                # RWMLP model loss
                rwmlp_validation_losses = [rwmlp_validate_model(i) for i
                                           in xrange(n_valid_batches)];
                rwmlp_validation_record[epoch-1] = numpy.mean(rwmlp_validation_losses);

                print 'RWMLP MODEL';
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches, rwmlp_validation_record[epoch-1] * 100.));

                rwmlp_test_losses = [rwmlp_test_model(i) for i
                                     in xrange(n_test_batches)];
                rwmlp_test_record[epoch-1] = numpy.mean(rwmlp_test_losses);
                
                print(('     epoch %i, minibatch %i/%i, test error %f %%') %
                      (epoch, minibatch_index + 1, n_train_batches, rwmlp_test_record[epoch-1] * 100.));
                        
    end_time = time.clock();

    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.));

    return mlp, rwmlp, mlp_validation_record, mlp_test_record, rwmlp_validation_record, rwmlp_test_record;
