#! /usr/bin/env python

import numpy as np
import scipy
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time
import os

import tensorflow as tf
if tf.__version__ == '1.15.0':
    tf.compat.v1.enable_eager_execution()
elif tf.__version__.split('.')[0] == 2: # in ['2.2.0','2.3.0']:
    tf.keras.backend.set_floatx('float64')

from tfdiffeq import odeint,odeint_adjoint
from tfdiffeq.adjoint import odeint as adjoint_odeint

import pynirom
from pynirom.node import main as main

"""
Module for pre- and post-processing data and implementing
Neural ODE to model temporal dynamics in the latent space
"""


def scale_time(time_array):
    """
    Simple utility method to scale a
    given time array [0,T] to [0,1]

    Input::
    time_array: numpy time array

    Output::
    scaled numpy time array
    """
    tscale = np.amax(time_array)

    return time_array/tscale


def scale_states(state_array, method='centered'):
    """
    Scale input states using different
    methods
    Input::
    state_array: 2d numpy array (Time X State)
    method: Specifies what the scaled data is mapped to
            centered : --> [-1,1] for each element
            abs : --> [0,1] for each element

    Output::
    Scaled state array
    """
    if method == 'centered':  ## Scale each element between [-1,1]
        max_g = np.amax(state_array,axis=0);
        min_g = np.amin(state_array,axis=0)
        scaler = lambda x: (2*(x - min_g)/(max_g - min_g) - 1)
        state_array = scaler(state_array)
    elif method == 'abs':  ## Scale each element between [0,1]
        scale_mm = MinMaxScaler()
        scale_mm.fit(state_array)
        state_array = scale_mm.transform(state_array)

    return state_array


def augment_state(state_array, time_array, aug_dim=0):
    """
    Compute augmented state arrays for
    Augmented NODE implementation
    Input::
    state_array: Original state array
    time_array: Original array of time points
    aug_dim: How many rows to augment state vector
    Output::
    state_tensor: Augmented state array casted to a TF tensor
    time_tensor: Time array casted to a TF tensor
    init_tensor: Initial state vector casted to a TF tensor
    """
    if aug_dim == 0:
        state_tensor = tf.convert_to_tensor(state_array)
    else:
        augment_zeros = np.zeros((state_array.shape[0],aug_dim))
        state_tensor = tf.convert_to_tensor(np.hstack((state_array, augment_zeros)))
    time_tensor = tf.convert_to_tensor(time_array)
    init_tensor = state_tensor[0,:]

    return state_tensor, time_tensor, init_tensor


def set_learning_rate(decay=False, init_lr = 0.001, **kwargs):
    """
    Set the learning rate scheduler or a fixed learning rate

    Input::
    decay: Boolean. If True, use a scheduler to decay learning rate
    init_lr: float32/64. Initial learning rate. If decay==False, this is
            the value of the fixed learning rate
    decay_steps: [Optional] int32/64. If decay==True, set the number of
                epochs to wait before reducing learning rate
    decay_rate: [Optional] float32/64. If decay==True, set the decay rate
    staircase: Boolean. If True decay the learning rate at
                discrete intervals
    Output::
    Returns the learning rate
    """

    if decay:
        try:
            learn_rate = tf.keras.optimizers.schedules.ExponentialDecay(init_lr,
                                        kwargs['decay_steps'],
                                        kwargs['decay_rate'],
                              staircase=kwargs['staircase'])
        except:
            print("Learning rate decay function parameters not provided")
    else:
        learn_rate = tf.constant(init_lr)

    return learn_rate


def set_optimizer(opt='RMSProp', learn_rate=0.001):
    """
    Set the optimizer for computing network hyperparameters
    Input::
    opt: string. Denotes the optimization algorithm to be used.
            Currently supports 'Adam' and 'RMSProp'.
    learn_rate: Either a fixed float32/64 scalar or a learning rate
            schedule that defines how the learning rate of the
            optimizer changes over time

    Output::
    Returns the optimizer
    """

    if opt == 'Adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate = learn_rate)
    elif opt == 'RMSprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate = learn_rate, momentum = 0.9)

    return optimizer


def run_node(true_state_tensor, times_tensor, init_state, epochs, savedir, optimizer, learn_rate, device,
                solver='rk4', purpose='train', adjoint=False, minibatch=False,):
    """
    """

    print('\n------------Begin training---------')
    train_loss_results = []
    train_lr = []
    saved_ep = []
    start_time = time.time()

    state_len = tf.shape(true_state_tensor)[1]
    try:
        learning_rate_decay = learn_rate.dtype == tf.float32
    except:
        learning_rate_decay = False

    if adjoint == True:
        int_ode = adjoint_odeint
    elif adjoint == False:
        int_ode = odeint

    if purpose == 'train':
        if not os.path.exists(savedir+'/model_weights/'):
            os.makedirs(savedir+'/model_weights/')

        if minibatch == True:
            dataset = tf.data.Dataset.from_tensor_slices((true_state_tensor, times_tensor))
            dataset = dataset.batch(128)
            with tf.device(device):
                model = main.DNN(state_len)
                for epoch in range(epochs):
                    datagen = iter(dataset)
                    avg_loss = tf.keras.metrics.Mean()
                    for batch, (true_state_trainer, times_trainer) in enumerate(datagen):
                        with tf.GradientTape() as tape:
                            preds = int_ode(model, tf.expand_dims(init_state, axis=0), times_trainer, method=solver)
                            loss = tf.math.reduce_mean(tf.math.square(true_state_trainer - tf.squeeze(preds)))
                        grads = tape.gradient(loss, model.trainable_variables)
                        optimizer.apply_gradients(zip(grads, model.trainable_variables))
                        avg_loss(loss)

                    train_loss_results.append(avg_loss.result().numpy())
                    print("Epoch %d: Loss = %0.6f" % (epoch + 1, avg_loss.result().numpy()))
                    print()

        elif minibatch == False:
            with tf.device(device):
                model = main.DNN(state_len)
                print()
                for epoch in range(epochs):

                    with tf.GradientTape() as tape:
                        preds = int_ode(model, tf.expand_dims(init_state, axis=0), times_tensor, method=solver)
                        loss = tf.math.reduce_mean(tf.math.square(true_state_tensor - tf.squeeze(preds)))
                    grads = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(grads, model.trainable_variables))

                    train_loss_results.append(loss.numpy())
                    print("Epoch {0}: Loss = {1:0.6f}, LR = {2:0.6f}".format(epoch+1, loss.numpy(), learn_rate(optimizer.iterations).numpy()))
                    print()
                    if (epoch+1)%(epochs//4) == 0:
                        print("******Saving model state. Epoch {0}******\n".format(epoch + 1))
                        model.save_weights(savedir+'/model_weights/ckpt', save_format='tf')
                        if learning_rate_decay:
                            train_lr.append(learn_rate(optimizer.iterations).numpy())
                        else:
                            train_lr.append(learn_rate)
                        saved_ep.append(epoch+1)
                        np.savez_compressed(savedir+'/model_weights/train_lr', lr=train_lr, ep=saved_ep)


        model.save_weights(savedir+'/model_weights/ckpt', save_format='tf')
        if learning_rate_decay:
            train_lr.append(learn_rate(optimizer.iterations).numpy())
        else:
            train_lr.append(learn_rate)
        saved_ep.append(epoch+1)
        np.savez_compressed(savedir+'/model_weights/train_lr', lr=train_lr, ep=saved_ep)
        end_time = time.time()
        print("****Total training time = {0}****\n".format(end_time - start_time))

    elif purpose == 'retrain':

        saved_lr = np.load(pre_trained_dir+'train_lr.npz')
        initial_learning_rate = saved_lr['lr'][-1]
        ep = saved_lr['ep'][-1]
        print("Initial lr = {0}".format(initial_learning_rate))
        if not os.path.exists(savedir+'/current/model_weights_cyl/'):
            os.makedirs(savedir+'/current/model_weights_cyl/')

        if learning_rate_decay == True:
            learn_rate = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps,
                                                        decay_rate, staircase=staircase_opt)
        elif learning_rate_decay == False:
            learn_rate = initial_learning_rate

        if optimizer == 'Adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate = learn_rate)
        elif optimizer == 'RMSprop':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate = learn_rate, momentum = 0.9)


        if minibatch == True:
            dataset = tf.data.Dataset.from_tensor_slices((true_state_tensor, times_tensor))
            dataset = dataset.batch(128)

            with tf.device(device):
                model = main.DNN(state_len)
                print()
                model.load_weights(pre_trained_dir+'ckpt')

                for epoch in range(epochs):
                    datagen = iter(dataset)
                    avg_loss = tf.keras.metrics.Mean()
                    for batch, (true_state_trainer, times_trainer) in enumerate(datagen):
                        with tf.GradientTape() as tape:
                            preds = int_ode(model, tf.expand_dims(init_state, axis=0), times_trainer, method=solver)
                            loss = tf.math.reduce_mean(tf.math.square(true_state_trainer - tf.squeeze(preds)))

                        grads = tape.gradient(loss, model.trainable_variables)
                        optimizer.apply_gradients(zip(grads, model.trainable_variables))
                        avg_loss(loss)

                    train_loss_results.append(avg_loss.result().numpy())
                    print("Epoch %d: Loss = %0.6f, LR = %0.6f" %(epoch + 1, avg_loss.result().numpy(), learn_rate(optimizer.iterations).numpy()))
                    print()

        elif minibatch == False:

            with tf.device(device):
                model = main.DNN(state_len)
                model.load_weights(pre_trained_dir+'ckpt')
                for epoch in range(epochs):

                    with tf.GradientTape() as tape:
                        preds = int_ode(model, tf.expand_dims(init_state, axis=0), times_tensor, method=solver)
                        loss = tf.math.reduce_mean(tf.math.square(true_state_tensor - tf.squeeze(preds)))

                    grads = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(grads, model.trainable_variables))

                    train_loss_results.append(loss.numpy())
                    print("Epoch %d: Loss = %0.6f, LR = %0.6f" %(ep+epoch+1, loss.numpy(), learn_rate(optimizer.iterations).numpy()))
                    print()
                    if (epoch+1)%(epochs//4) == 0:
                        print("Saving model state. Epoch {0}\n".format(epoch + ep + 1))
                        model.save_weights(savedir+'/model_weights/ckpt', save_format='tf')
                        if learning_rate_decay:
                            train_lr.append(learn_rate(optimizer.iterations).numpy())
                        else:
                            train_lr.append(learn_rate)
                        saved_ep.append(epoch+ep+1)
                        np.savez_compressed(savedir+'/model_weights/train_lr', lr=train_lr, ep=saved_ep)


        end_time = time.time()
        print("****Total training time = {0}****\n".format(end_time - start_time))

        model.save_weights(savedir+'/model_weights/ckpt', save_format='tf')
        if learning_rate_decay:
            train_lr.append(learn_rate(optimizer.iterations).numpy())
        else:
            train_lr.append(learn_rate)
        saved_ep.append(epoch+ep+1)
        np.savez_compressed(savedir+'/model_weights/train_lr', lr=train_lr, ep=saved_ep)


    elif purpose == 'eval':
        model = main.DNN(state_len)
        model.load_weights(pre_trained_dir+'ckpt')


    return train_loss_results, train_lr, saved_ep
