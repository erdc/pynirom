#! usr/bin/env python

"""
Base module for PODNODE NIROM
"""

import numpy as np
import pickle

import pynirom
from pynirom.node import node as node

import tensorflow as tf
if tf.__version__ == '1.15.0':
    tf.compat.v1.enable_eager_execution()
elif tf.__version__.split('.')[0] == 2: # in ['2.2.0','2.3.0']:
    tf.keras.backend.set_floatx('float64')

from tfdiffeq import odeint,odeint_adjoint
from tfdiffeq.adjoint import odeint as adjoint_odeint


class NODEBase(object):
    """
    Base class for Non-intrusive Reduced Order Modeling (NIROM)
    with Proper Orthogonal Decomposition (POD) for the selection
    of an optimal linear reduced basis space and Neural ODE (NODE)
    for the approximation of the dynamics in the space spanned
    by the POD modes.

    """
    @staticmethod
    def save_to_disk(filename, ROM, **options):
        """
        Save the instance in ROM to disk using options
        Start with absolutely simplest approach using pickle
        """
        outfile = open(filename, 'wb')
        protocol = options.get('protocol', pickle.HIGHEST_PROTOCOL)
        try:
            pickle.dump(ROM, outfile, protocol=protocol)
        finally:
            outfile.close()
        return ROM

    @staticmethod
    def read_from_disk(filename, **options):
        """
        Read the instance in ROM to disk using options
        Start with absolutely simplest approach using pickle
        """
        infile = open(filename, 'rb')
        encoding = options.get('encoding', 'latin1')
        ROM = None
        try:
            ROM = pickle.load(infile, encoding=encoding)
        except TypeError:
            ROM = pickle.load(infile)
        finally:
            infile.close()
        return ROM

    def __init__(self, device, **options):
        self._device = device
        # for thing in ['_U_r','_Sigma_r','_V_r','_X']:
        #     setattr(self,thing,None)

    @property
    def n_latent(self):
        """
        User specified size of the latent
        space vector used to model dynamics
        with NODE. For multicomponent systems,
        this is equal to the combined size of
        the latent space representations of all
        system components.
        """
        return self._n_latent

    @property
    def n_snap(self):
        """
        Return number of snapshots used in training
        """
        return self._n_snap_train

    @property
    def n_reduced(self):
        """
        Size of latent space representation of
        each system component.
        Output as a dictionary with
        system components as keys
        """
        return self._n_reduced

    @property
    def train_state(self):
        """
        Latent space snapshots used for training
        array dim. = Latent space size X training time points
        """
        return self._train_state

    @property
    def train_times(self):
        """
        Time points array used for training
        """
        return self._train_time

    @property
    def pred_state(self):
        """
        Latent space predictions using NODE
        array dim. = Latent space size X prediction time points
        """
        return self._predicted_state

    def prepare_input_data(self, Z_train, nw, times_train, stack_order, times_predict=None, Z_pred_true=None):
        """
        Set up latent space data in a format suitable
        for modeling with NODE

        Input::
        Z_train: Dict of latent space training snapshots for each component
        nw: Dict of latent space dimensions for each component
        times_train: Numpy array of training time points
        stack_order: String denoting the order in which the components
                    are vertically stacked in the combined latent space
                    snapshot vector
        times_predict: [Optional] Numpy array of prediction time points
        Z_pred_true: [Optional] Dict of true latent space snapshots at the
                    prediction time points for each component. Used for
                    computing NODE prediction error

        Output::
        train_state_array: Numpy array of vertically stacked latent space
                    training snapshots with time on 0th axis
        init_state: Stacked latent space vector at initial training time
        state_len: Total dimension of the latent space combining all components
        dt_train: Time step of training snapshots
        true_pred_state_array: Numpy array of vertically stacked true latent
                    space snapshots at prediction time points with time on
                    0th axis [Optional]
        dt_predict: Time step of the time series to be used for computing
                    the NODE prediction [Optional]
        """
        npod_total = np.sum(list(nw.values()))
        train_state_array = np.zeros((times_train.size, npod_total));
        if times_predict is not None:
            assert Z_pred_true is not None, "Prediction time points provided, but true snapshots not provided"
            true_pred_state_array = np.zeros((times_predict.size, npod_total));
        else:
            true_pred_state_array = None
        ctr=0
        stack = stack_order.split(',')
        for key in stack:
            train_state_array[:,ctr:ctr+nw[key]] = Z_train[key].T
            if times_predict is not None:
                true_pred_state_array[:,ctr:ctr+nw[key]] = Z_pred_true[key].T
            ctr+=nw[key]

        init_state = train_state_array[0,:]
        state_len = np.shape(train_state_array)[1]
        dt_train = (times_train[-1]-times_train[0])/(times_train.size-1)
        if times_predict is not None:
            dt_predict = (times_predict[-1]-times_predict[0])/(times_predict.size)

        self._train_state = train_state_array
        self._train_time = times_train
        self._n_snap_train = times_train.size
        self._n_latent = npod_total
        self._n_reduced = nw

        if times_predict is not None:
            return train_state_array, true_pred_state_array, init_state, state_len, dt_train, dt_predict
        else:
            return train_state_array, init_state, state_len, dt_train


    def preprocess_data(self,scale_states=False, augmented=False, lr_decay=True, init_lr = 0.001, opt='RMSProp', **options):
        """

        Input::
        scale_states: Boolean. If True, scale training state array
        scaling_method: [Optional] string. If scale_states++True, specify
                        the scaling function to be used.
                        'centered' - maps each element to [-1,1]
                        'abs' - maps each element to [0,1]
        augmented: Boolean. If True, augment states arrays by a fixed size
        aug_dim: [Optional] int32/64. If augmented==True, specify size of
                            augmentation
        lr_decay: Boolean. If True, define a learning rate scheduler
        init_lr: float32/64. Initial learning rate used either as a fixed
                        learning rate for training or as the starting
                        learning rate used by the scheduler
        decay_steps: [Optional] int32/64. If lr_decay==True, set the number of
                    epochs to wait before reducing learning rate
        decay_rate: [Optional] float32/64. If lr_decay==True, set the decay rate
        staircase: Boolean. [Optional] If True, decays the learning rate at
                    discrete intervals
        opt: string. Specify the Optimization algorithm for hyperparameter tuning
                Currently accepts 'Adam' or 'RMSProp'

        Output::
        true_state_tensor: Augmented state array casted to a TF tensor
        time_tensor: Time array casted to a TF tensor
        init_tensor: Initial state vector casted to a TF tensor
        learn_rate: Fixed learning rate or Scheduler
        opimizer: Returns the chosen optimizer
        """
        if scale_states:
            train_state_array = node.scale_states(self._train_state, method=options['scaling_method'])
        else:
            train_state_array = self._train_state

        try:
            aug_dim = options['aug_dim']
        except:
            aug_dim = 0
        true_state_tensor, time_tensor, init_tensor = node.augment_state(train_state_array,
                                                    self._train_time, aug_dim)

        learn_rate = node.set_learning_rate(decay=lr_decay, init_lr = init_lr,
                            decay_steps = options['decay_steps'], decay_rate = options['decay_rate'],
                            staircase = options['staircase'])

        optimizer = node.set_optimizer(opt=opt, learn_rate=learn_rate)

        self._learn_rate = learn_rate
        self._optimizer = optimizer

        return true_state_tensor, time_tensor, init_tensor, learn_rate, optimizer


    def train_model(self, true_state_tensor, times_tensor, init_state, epochs, savedir,
                    solver='rk4', purpose='train', adjoint=False, minibatch=False):
        """
        """
        train_loss_results, train_lr, saved_ep = node.run_node(true_state_tensor, times_tensor, init_state,
                        epochs, savedir, optimizer=self._optimizer, learn_rate= self._learn_rate,
                        device=self._device, solver=solver, purpose=purpose, adjoint=adjoint, minibatch=minibatch)

        return train_loss_results, train_lr, saved_ep


    def predict_time(self, times_online, use_greedy=False, **options):
        """
        Evaluate reduced order POD-RBF model
        at queried online time points

        Input:
        times_online -- array of time points to be queried
                    for evaluating ROM
        eng_cap -- Total energy fraction captured in mode list L (Optional)
        """
        self._times_online = times_online
        if 't0_ind' in options:
            t0_ind = options['t0_ind']
        else:
            t0_ind = 0


        self._rbf_centers = self._rbf_centers_train
        self._rbf_coeff = self._rbf_coeff_train
        self._scale_user = self._scale
        # self._ind_greedy = None

        print("\n---- Computing RBF NIROM solution ----\n")
        S_pred, Z_pred = rom.rbf_online(self._rbf_centers, self._rbf_coeff, self._S,
                                        self._times_online, self._scale_user, self._Phi, self._S_mean,
                                        self._kernel, time_disc=self._time_disc_method, beta=2.5,
                                        t0_ind=t0_ind)
        self._S_pred = S_pred
        self._Z_pred = Z_pred

        if use_greedy:
            return S_pred, Z_pred, self._ind_greedy
        else:
            return S_pred, Z_pred

    def compute_error(self, true, pred, soln_names, metric='rms'):
        """
        Utility function to compute different
        error metrics based on the provided true solution
        and the nirom solution projected on to the full
        dimensional space

        Input::
        true: Dictionary with 2d arrays of true solutions
        pred: Dictionary with 2d arrays of PODRBF predictions
        soln_names: Dictionary with names of solution
                    components
        metric: type of error metric
                'rms' = Root mean square error
                'rel' = relative error
        Output::
        err: Dictionary of error values.
             Each key is a 1d array with error
             values for a solution component
        """

        err = {}
        Nn = self.n_fine
        Nc = len(soln_names)
        if metric == 'rms':
            for ivar, key in enumerate(soln_names):
                err[key] = np.linalg.norm(true[key][:,:] - pred[key][:,:], axis = 0)/ \
                                np.sqrt(Nn)
        elif metric == 'rel':
            for ivar, key in enumerate(soln_names):
                err[key] = np.linalg.norm(true[key][:,:] - pred[key][:,:], axis = 0)/ \
                                np.linalg.norm(true[key][:,:], axis = 0)

        return err




### ------- Define NN approximation of time derivative-----------------
class DNN(tf.keras.Model):

    def __init__(self, state_len, N_layers=1, N_neurons=256, act_f='tanh', **kwargs):
        super().__init__(**kwargs)

        try:
            aug_dims = kwargs['aug_dim']
        except:
            aug_dims = 0
        if N_layers == 1:

            self.eqn = tf.keras.Sequential([tf.keras.layers.Dense(N_neurons, activation=act_f,
                                           kernel_initializer = tf.keras.initializers.glorot_uniform(),
                                           bias_initializer='zeros',
                                           input_shape=(state_len+aug_dims,)),
                         tf.keras.layers.Dense(state_len+aug_dims)])

        elif N_layers == 2:

            self.eqn = tf.keras.Sequential([tf.keras.layers.Dense(N_neurons, activation=act_f,
                                           kernel_initializer = tf.keras.initializers.glorot_uniform(),
                                           bias_initializer='zeros',
                                           input_shape=(state_len+aug_dims,)),
                                           tf.keras.layers.Dense(N_neurons, activation='linear',
                                           kernel_initializer = tf.keras.initializers.glorot_uniform(),
                                           bias_initializer='zeros'),
                         tf.keras.layers.Dense(state_len+aug_dims)])

        elif N_layers == 3:

            self.eqn = tf.keras.Sequential([tf.keras.layers.Dense(N_neurons, activation=act_f,
                                           kernel_initializer = tf.keras.initializers.glorot_uniform(),
                                           bias_initializer='zeros',
                                           input_shape=(state_len+aug_dims,)),
                                           tf.keras.layers.Dense(N_neurons, activation=act_f,
                                           kernel_initializer = tf.keras.initializers.glorot_uniform(),
                                           bias_initializer='zeros'),
                                           tf.keras.layers.Dense(N_neurons, activation='linear',
                                           kernel_initializer = tf.keras.initializers.glorot_uniform(),
                                           bias_initializer='zeros'),
                         tf.keras.layers.Dense(state_len+aug_dims)])

        elif N_layers == 4:
            self.eqn =  tf.keras.Sequential([tf.keras.layers.Dense(N_neurons, activation=act_f,
                                           kernel_initializer = tf.keras.initializers.glorot_uniform(),
                                           bias_initializer='zeros',
                                           input_shape=(state_len+aug_dims,)),
                                           tf.keras.layers.Dense(N_neurons, activation=act_f,
                                           kernel_initializer = tf.keras.initializers.glorot_uniform(),
                                           bias_initializer='zeros'),
                                           tf.keras.layers.Dense(N_neurons, activation=act_f,
                                           kernel_initializer = tf.keras.initializers.glorot_uniform(),
                                           bias_initializer='zeros'),
                                           tf.keras.layers.Dense(N_neurons, activation='linear',
                                           kernel_initializer = tf.keras.initializers.glorot_uniform(),
                                           bias_initializer='zeros'),
                         tf.keras.layers.Dense(state_len+aug_dims)])

    @tf.function
    def call(self, t, y):
        i0 = self.eqn(y)

        return i0
