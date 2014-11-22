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

## if have only CLI, enable following two lines
#import matplotlib
#matplotlib.use('Agg');

import matplotlib.pyplot as plt;
import theano;
import theano.tensor as T;

# private library modules

from cae_tools import load_data;
from cae_tools import MLP;
from cae_tools import RWMLP;
from dlt_utils import tile_raster_images;
from mlp_model import mlp_model;

## compare learned feature from different model

n_epochs=300;
(mlp,
 rwmlp,
 mlp_cv_record,
 mlp_test_record,
 rwmlp_cv_record,
 rwmlp_test_record)=mlp_model(n_in=28*28,
                              n_hidden=1000,
                              n_out=10,
                              dataset='data/mnist.pkl.gz',
                              batch_size=20,
                              n_epochs=n_epochs,
                              learning_rate=0.1,
                              L1_reg=0.00,
                              L2_reg=0.0001,
                              hidden_limits=0.25,
                              out_limits=0.25);

image = Image.fromarray(tile_raster_images(X=mlp.hidden_layer.W.get_value(borrow=True).T,
                                           img_shape=(28, 28), tile_shape=(10, 10),
                                           tile_spacing=(1, 1)));

image.save('mlp_learned_feature.png');

image = Image.fromarray(tile_raster_images(X=rwmlp.hidden_layer.W.get_value(borrow=True).T,
                                           img_shape=(28, 28), tile_shape=(10, 10),
                                           tile_spacing=(1, 1)));

image.save('rwmlp_learned_feature.png');


## compare result between different noise

noise_level=numpy.asarray([0.0001, 0.001, 0.01, 0.05, 0.1, 0.125, 0.15, 0.2, 0.25]);
n_epochs=200;
n_steps=9;
best_noisy_mlp_cv=numpy.zeros((n_steps,1));
best_noisy_mlp_test=numpy.zeros((n_steps, 1));
best_noisy_rwmlp_cv=numpy.zeros((n_steps, 1));
best_noisy_rwmlp_test=numpy.zeros((n_steps, 1));

noise_training_iter=0
for noise in noise_level:
    print (('Interation %d of %d') % (noise_training_iter+1,n_steps));

    (mlp,
     rwmlp,
     mlp_cv_record,
     mlp_test_record,
     rwmlp_cv_record,
     rwmlp_test_record)=mlp_model(n_in=28*28,
                                  n_hidden=1000,
                                  n_out=10,
                                  dataset='data/mnist.pkl.gz',
                                  batch_size=20,
                                  n_epochs=n_epochs,
                                  learning_rate=0.1,
                                  L1_reg=0.00,
                                  L2_reg=0.0001,
                                  hidden_limits=noise,
                                  out_limits=noise);

    # get best result

    mlp_index=numpy.argmin(mlp_test_record);
    best_noisy_mlp_cv[noise_training_iter]=mlp_cv_record[mlp_index];
    best_noisy_mlp_test[noise_training_iter]=mlp_test_record[mlp_index];

    rwmlp_index=numpy.argmin(rwmlp_test_record);
    best_noisy_rwmlp_cv[noise_training_iter]=rwmlp_cv_record[rwmlp_index];
    best_noisy_rwmlp_test[noise_training_iter]=rwmlp_test_record[rwmlp_index];

    noise_training_iter=noise_training_iter+1;

numpy.savetxt('mlp_rwmlp_noise_diff.txt', (best_noisy_mlp_cv,
                                           best_noisy_mlp_test,
                                           best_noisy_rwmlp_cv,
                                           best_noisy_rwmlp_test));


plt.figure(1);
idx=numpy.arange(n_steps);
width=0.4;

bnrc_g=plt.barh(idx, best_noisy_mlp_test*100, width, color='y');
bnrt_g=plt.barh(idx+width, best_noisy_rwmlp_test*100, width, color='g');
bnrcl_g=plt.plot(best_noisy_mlp_test*100, idx+width/2, color='y', linewidth=2);
bnrcl_g=plt.plot(best_noisy_rwmlp_test*100, idx+width+width/2, color='g', linewidth=2);

plt.xlabel('Error (%)');
plt.ylabel('Random weight');
plt.yticks(idx+width, noise_level);
plt.xticks(numpy.arange(7));

plt.legend((bnrc_g[0], bnrt_g[0]), ('MLP testing error', 'RWMLP testing error'));

plt.savefig('mlp_rwmlp_comp_error_diff.png');
plt.savefig('mlp_rwmlp_comp_error_diff.eps');

plt.show();


## compare result between different training samples

## probably take you forever to run if you have too many steps.
## you can divide the work and depoly them to different cloud instance
## it will make it much more faster

n_epochs=200;
step=0.004;
n_steps=int(1/step);

best_mlp_cv=numpy.zeros((n_steps,1));
best_mlp_test=numpy.zeros((n_steps, 1));
best_rwmlp_cv=numpy.zeros((n_steps, 1));
best_rwmlp_test=numpy.zeros((n_steps, 1));

for training_portion in xrange(n_steps):
    tp=(training_portion+1)*step;
    print (('Interation %d of %d') % (training_portion+1,n_steps));

    # training model
    (mlp,
     rwmlp,
     mlp_cv_record,
     mlp_test_record,
     rwmlp_cv_record,
     rwmlp_test_record)=mlp_model(n_in=28*28,
                                  n_hidden=1000,
                                  n_out=10,
                                  dataset='data/mnist.pkl.gz',
                                  training_portion=tp,
                                  batch_size=20,
                                  n_epochs=n_epochs,
                                  learning_rate=0.1,
                                  L1_reg=0.00,
                                  L2_reg=0.0001,
                                  hidden_limits=0.1,
                                  out_limits=0.1);

    # get best result

    mlp_index=numpy.argmin(mlp_test_record);
    best_mlp_cv[training_portion]=mlp_cv_record[mlp_index];
    best_mlp_test[training_portion]=mlp_test_record[mlp_index];

    rwmlp_index=numpy.argmin(rwmlp_test_record);
    best_rwmlp_cv[training_portion]=rwmlp_cv_record[rwmlp_index];
    best_rwmlp_test[training_portion]=rwmlp_test_record[rwmlp_index];

# display result

x=numpy.linspace(0, n_steps-1, n_steps);

plt.figure(2);

mlp_cv_g,=plt.plot(x, best_mlp_cv, 'b', label='MLP validation error')
mlp_test_g,=plt.plot(x, best_mlp_test, 'r', label='MLP testing error')
rwmlp_cv_g,=plt.plot(x, best_rwmlp_cv, 'g', label='RWMLP validation error')
rwmlp_test_g,=plt.plot(x, best_rwmlp_test,'black', label='RWMLP testing error')

plt.legend(handles=[mlp_cv_g, mlp_test_g, rwmlp_cv_g, rwmlp_test_g]);

plt.xlabel(("Training samples (*%d samples)") % int(50000/n_steps));
plt.ylabel("Error (%)");
plt.axis([0,n_steps-1, 0, 0.5]);

plt.savefig('mlp_rwmlp_comp_train_samples.png');
plt.savefig('mlp_rwmlp_comp_train_samples.eps');
    
## compare result between different epochs
n_epochs=300;
(mlp,
 rwmlp,
 mlp_cv_record,
 mlp_test_record,
 rwmlp_cv_record,
 rwmlp_test_record)=mlp_model(n_in=28*28,
                              n_hidden=1000,
                              n_out=10,
                              dataset='data/mnist.pkl.gz',
                              batch_size=20,
                              n_epochs=n_epochs,
                              learning_rate=0.1,
                              L1_reg=0.00,
                              L2_reg=0.0001,
                              hidden_limits=0.1,
                              out_limits=0.1);

x=numpy.linspace(0, n_epochs-1, n_epochs);

# plot the result
plt.figure(3);

mlp_cv_g,=plt.plot(x, mlp_cv_record, 'b', label='MLP validation error')
mlp_test_g,=plt.plot(x, mlp_test_record, 'r', label='MLP testing error')
rwmlp_cv_g,=plt.plot(x, rwmlp_cv_record, 'g', label='RWMLP validation error')
rwmlp_test_g,=plt.plot(x, rwmlp_test_record,'black', label='RWMLP testing error')

plt.legend(handles=[mlp_cv_g, mlp_test_g, rwmlp_cv_g, rwmlp_test_g]);

plt.xlabel("Training epochs");
plt.ylabel("Error (%)");
plt.axis([0,n_epochs-1, 0, 0.15]);

plt.savefig('mlp_rwmlp_comp.png');
plt.savefig('mlp_rwmlp_comp.eps');