import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.cluster import KMeans


class GaussianMixtureModel:
    def __init__(self, K):
        self.K = K
        self.Pi = None
        self.Mu = None
        self.Sigma = None
                
    def _init_params(self, X, random_state=None):
        '''
        Method for initializing model parameterse based on the size and variance of the input data array. 
        
        Parameters
        ----------
        X : 2D numpy array
            2-D numpy array representing input data, where X[n, i] represents the i-th element of n-th point in X.
        '''
        n_samples, n_features = np.shape(X)
        rnd = np.random.RandomState(seed=random_state)
        
        self.Pi = np.ones(self.K)/self.K
        self.Mu = X[rnd.choice(n_samples, size=self.K, replace=False)]
        self.Sigma = np.tile(np.diag(np.var(X, axis=0)), (self.K, 1, 1))

        
    def _calc_nmat(self, X):
        '''
        Method for calculating array corresponding $\mathcal{N}(x_n | \mu_k)$
        
        Parameters
        ----------
        X : 2D numpy array
            2-D numpy array representing input data, where X[n, i] represents the i-th element of n-th point in X.
            
        Returns
        ----------
        Nmat : 2D numpy array
            2-D numpy array representing probability density for each sample and each component, 
            where Nmat[n, k] = $\mathcal{N}(x_n | \mu_k)$.
        
        '''
        n_samples, n_features = np.shape(X)
         
        Diff = np.reshape(X, (n_samples, 1, n_features) ) - np.reshape(self.Mu, (1, self.K, n_features) )
        L = np.linalg.inv(self.Sigma)
        exponent = np.einsum("nkj,nkj->nk", np.einsum("nki,kij->nkj", Diff, L), Diff)
        Nmat = np.exp(-0.5*exponent)/np.sqrt(np.linalg.det(self.Sigma))   / (2*np.pi)**(n_features/2)
        return Nmat
        
    def _Estep(self, X):
        '''
        Method for calculating the array corresponding to responsibility.
        
        Parameters
        ----------
        X : 2D numpy array
            2-D numpy array representing input data, where X[n, i] represents the i-th element of n-th point in X.
            
        Returns
        ----------
        Gam : 2D numpy array
            2-D numpy array representing responsibility of each component for each sample in X, 
            where Gamt[n, k] = $\gamma_{n, k}$.
        
        '''
        n_samples, n_features = np.shape(X)
        Nmat = self._calc_nmat(X)
        tmp = Nmat * self.Pi
        Gam = tmp/np.reshape(np.sum(tmp, axis=1), (n_samples, 1) )
        return Gam
        
    def _Mstep(self, X, Gam):
        '''
        Method for calculating the model parameters based on the responsibility gamma.
        
        Parameters
        ----------
        X : 2D numpy array
            2-D numpy array representing input data, where X[n, i] represents the i-th element of n-th point in X.
        Gam : 2D numpy array
            2-D numpy array representing responsibility of each component for each sample in X, 
            where Gamt[n, k] = $\gamma_{n, k}$.
        '''
        n_samples, n_features = np.shape(X)
        Diff = np.reshape(X, (n_samples, 1, n_features) ) - np.reshape(self.Mu, (1, self.K, n_features) )
        Nk = np.sum(Gam, axis=0)
        self.Pi = Nk/n_samples
        self.Mu = Gam.T @ X / np.reshape(Nk, (self.K, 1))
        self.Sigma = np.einsum("nki,nkj->kij", np.einsum("nk,nki->nki", Gam, Diff), Diff) / np.reshape(Nk, (self.K, 1, 1))
        
    def calc_prob_density(self, X):
        '''
        Method for calculating the probablity density $\sum_k \pi_k \mathcal{N}(x_n | \mu_k)$
        
        Parameters
        ----------
        X : 2D numpy array
            2-D numpy array representing input data, where X[n, i] represents the i-th element of n-th point in X.
            
        Returns
        ----------
        prob_density : 2D numpy array

        '''
        prob_density = self._calc_nmat(X) @ self.Pi
        return prob_density
        
        
    def calc_log_likelihood(self, X):
        '''
        Method for calculating the log-likelihood for the input X and current model parameters.
        
        Parameters
        ----------
        X : 2D numpy array
            2-D numpy array representing input data, where X[n, i] represents the i-th element of n-th point in X.
        Returns
        ----------
        loglikelihood : float
            The log-likelihood of the input data X with respect to current parameter set.
        
        '''
        log_likelihood = np.sum(np.log(self.calc_prob_density(X)))
        return log_likelihood
        
        
    def fit(self, X, max_iter, tol, disp_message, random_state=None):
        '''
        Method for performing learning. 
        
        Parameters
        ----------
        X : 2D numpy array
            2-D numpy array representing input data, where X[n, i] represents the i-th element of n-th point in X.
        max_iter : int
            Maximum number of iteration
        tol : float, positive
            Precision. If the change of parameter is below this value, the iteration is stopped
        disp_message : Boolean
            Whether or not to show the message about the number of iteration
        '''
        self._init_params(X, random_state=random_state)
        log_likelihood = - np.float("inf")
        
        for i in range(max_iter):
            Gam = self._Estep(X)
            self._Mstep(X, Gam)
            log_likelihood_old = log_likelihood
            log_likelihood = self.calc_log_likelihood(X)
            
            if  log_likelihood - log_likelihood_old < tol:
                break
        if disp_message:
            print(f"n_iter : {i}")
            print(f"log_likelihood : {log_likelihood}")
            
    def predict_proba(self, X):
        '''
        Method for calculating the array corresponding to responsibility. Just a different name for _Estep
        
        Parameters
        ----------
        X : 2D numpy array
            2-D numpy array representing input data, where X[n, i] represents the i-th element of n-th point in X.
            
        Returns
        ----------
        Gam : 2D numpy array
            2-D numpy array representing responsibility of each component for each sample in X, 
            where Gamt[n, k] = $\gamma_{n, k}$.
        
        '''
        Gam = self._Estep(X)
        return Gam
    
    def predict(self, X):
        '''
        Method for make prediction about which cluster input points are assigned to.
        
        Parameters
        ----------
        X : 2D numpy array
            2-D numpy array representing input data, where X[n, i] represents the i-th element of n-th point in X.
            
        Returns
        ----------
        pred : 1D numpy array
            1D numpy array, with dtype=int, representing which class input points are assigned to.
        '''
        pred = np.argmax(self.predict_proba(X), axis=1)
        return pred
	

#Taking input from csv file and taking x and y out
data=pd.read_csv("dataset/coast/train.csv")
#data=data.dropna()
data=data.to_numpy()


data=data[:,1:]

gmm = GaussianMixtureModel(K=5)

