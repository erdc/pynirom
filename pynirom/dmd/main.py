#! /usr/bin/env python

import numpy as np
import pickle


class DMDBase(object):
    """
    Base class for Non-intrusive Reduced Order Modeling (NIROM)
    using Dynamic Mode Decomposition (DMD)

    """
    @staticmethod
    def save_to_disk(filename,ROM,**options):
        """
        Save the instance in ROM to disk using options
        Start with absolutely simplest approach using pickle
        """
        outfile = open(filename,'wb')
        protocol= options.get('protocol',pickle.HIGHEST_PROTOCOL)
        try:
            pickle.dump(ROM,outfile,protocol=protocol)
        finally:
            outfile.close()
        return ROM
    @staticmethod
    def read_from_disk(filename,**options):
        """
        Read the instance in ROM to disk using options
        Start with absolutely simplest approach using pickle
        """
        infile = open(filename,'rb')
        encoding= options.get('encoding','latin1')
        ROM = None
        try:
            ROM = pickle.load(infile,encoding=encoding)
        except TypeError:
            ROM = pickle.load(infile)
        finally:
            infile.close()
        return ROM

    def __init__(self,rank,**options):
        self._rank=rank
        for thing in ['_U_r','_Sigma_r','_V_r','_X']:
            setattr(self,thing,None)
    @property
    def rank(self):
        """
        Truncation level
        """
        return self._rank
    @property
    def n_snap(self):
        """
        return number of snapshots used in training
        """
        return self._n_snap_fit
    @property
    def n_fine(self):
        """
        return fine grid dimension
        """
        return self._Phi.shape[0]
    @property
    def t0_fit(self):
        """
        initial time value for DMD calculation
        """
        return self._t0_fit
    @property
    def dt_fit(self):
        """
        time step assumed for fit
        """
        return self._dt_fit
    @property
    def approximate_operator(self):
        """
        Best fit operator truncated to mode dim
        """
        return self._Atilde
    @property
    def basis(self):
        """
        DMD basis (truncated)
        """
        return self._Phi
    @property
    def omega(self):
        return self._omega
    @property
    def amplitudes(self):
        return self._b
    @property
    def pod_basis(self):
        return self._U_r
    @property
    def pod_singular_values(self):
        return self._Sigma_r

    def fit_basis(self,S,dt_fit,t0_fit=0.):
        """
        DMD algorithm

        Input::
        S: snapshot matrix (N X M == D.O.F. X Time points)
        dt_fit: Normalized time step (should be = 1.0)
        t0_fit: Initial time of simulation (Default = 0.0)

        Output::
        Phi: DMD modes
        D:
        """
        X, Xp = S[:,0:-1], S[:,1:]
        #time information
        self._t0_fit, self._dt_fit = t0_fit, dt_fit
        t0, dt = 0., 1.
        ## compute svd
        U, Sigma, Vt = np.linalg.svd(X, full_matrices = False)
        V = (Vt.conj()).transpose()
        r = min(self._rank, U.shape[1])
        U_r, Sigma_r, V_r=U[:,:r], Sigma[:r], V[:,:r]
        #invert Sigma (diagonal)
        assert np.abs(Sigma_r).min() > 0.
        SigmaInv_r = np.reciprocal(Sigma_r)
        SigmaInv_r = np.diag(SigmaInv_r)

        ## compute approximate temporal dynamics through Atilde
        tmp1   = ((U_r.conj().transpose()).dot(Xp)).dot(V_r)
        Atilde = tmp1.dot(SigmaInv_r)
        # discrete time eigenvalues and eigenvectors
        D, W   = np.linalg.eig(Atilde)
        ## DMD modes (using exact DMD algorithm)
        tmp2   = (Xp.dot(V_r)).dot(SigmaInv_r)
        Phi    = tmp2.dot(W)
        ## continuous time eigenvalues
        omega  = np.log(D)/dt
        assert np.isnan(omega).any() == False

        ## amplitudes from projection of initial condition
        x0 = X[:,0]
        PhiInv = np.linalg.pinv(Phi)
        b = PhiInv.dot(x0)

        toffline = t0 + np.arange(0, X.shape[1])*dt
        time_dynamics = np.zeros((r, X.shape[1]), dtype=np.complex_)

        for kk in range(X.shape[1]):
            time_dynamics[:,kk] = b*(np.exp(omega*toffline[kk]))
        #
        Xdmd = Phi.dot(time_dynamics)
        Xdmd = Xdmd.real

        ##Save DMD components
        self._n_snap_fit = S.shape[1]
        self._U_r, self._Sigma_r, self._SigmaInv_r, self._V_r = U_r, Sigma_r, SigmaInv_r, V_r
        #approximate linear system operator
        self._Atilde = Atilde
        #DMD modes,frequencies, eignenvectors, and projection of the initial condition
        self._Phi, self._omega, self._D, self._b = Phi, omega, D, b
        self._PhiInv = PhiInv

        return Phi,D,Xdmd,time_dynamics,Sigma

    def predict(self,t):
        """
        Utility function to predict solutions at
        provided time points
        Input::
        t: array of time points at which solution is sought

        Output::
        uh: DMD solution (full order space)
        """
        tnorm = (t - self.t0_fit)/(self.dt_fit)
        time_dynamics = self.amplitudes*(np.exp(self.omega*tnorm))
        uh = self.basis.dot(time_dynamics).real

        return uh

    def compute_error(self, true, pred, soln_names, metric='rms'):
        """
        Utility function to compute different
        error metrics based on provided DMD and
        true solutions

        Input::
        true: 2d array of true solutions
        pred: 2d array of DMD predictions
                Both true and pred arrays should have
                the same stacking conventions for
                multicomponent systems, with time on
                the 2nd axis
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
                err[key] = np.linalg.norm(true[ivar::Nc,:] - pred[ivar::Nc,:], axis = 0)/np.sqrt(Nn)
        elif metric == 'rel':
            for ivar, key in enumerate(soln_names):
                err[key] = np.linalg.norm(true[ivar::Nc,:] - pred[ivar::Nc,:], axis = 0)/ \
                                np.linalg.norm(true[ivar::Nc,:], axis = 0)

        return err
