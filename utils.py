'''
This script contains utility functions
'''

import torch
import pickle
import os
import numpy as np
from scipy.optimize import linear_sum_assignment

def save_data(data, path):
    '''
    Saves data as a .p file

    @Params:
        data...     Serializable object
        path...     path, where to store (should end with .p)
    '''

    pickle.dump(data, open( path, 'wb' ))

def load_data(path):
    '''
    Loads pickeled data

    @Params:
        path...     path, where object is stored (ends with .p)

    @Returns:
        unpickeled object
    '''

    return pickle.load(open( path, 'rb') )

'''
Sampling from model
'''
def generate_data(mus, sigmas, alphas, N, labels=False):
    '''
    Generates dataset from toy model

    @Params:
        mus...      DxK matrix, mean vectors as columns
        sigmas...   vector of size K
        alphas...   vector of size K
        N...        number of samples

    @Returns:
        N x D data matrix
    '''
    D,K = mus.shape

    # Draw samples
    x = torch.zeros((N,D))

    # Sample distribution of topics
    omegas = torch.distributions.Dirichlet(alphas).sample((N,))

    # Sample all continous variables at once for speed up
    for k in range(K):
        tmp = torch.distributions.Normal(loc=mus[:,k], scale=sigmas[k]).sample((N,))
        x += (omegas[:, k] * tmp.T).T
    if labels:
        return x.double(), omegas.double()
    else:
        return x.double()

'''
Sampling model parameters
'''
def sample_uniform_alphas(t, alpha0, vmin=0, vmax=1):
    tmp = np.random.uniform(vmin, vmax, size=t)
    alphas = tmp / tmp.sum() * alpha0
    return alphas


    if dist == "equal":
        alphas = create_equal_alphas(t, alpha0)
    elif dist == "uniform":
        alphas = sample_uniform_alphas(t, alpha0, 0.05, 0.95)
    elif dist == "skewed_high":
        alphas = sample_skewed_alphas(t, alpha0, mass="high")
    elif dist == "skewed_low":
        alphas = sample_skewed_alphas(t, alpha0, mass="low")
    else:
        raise Exception(f"Invalid Argument {dist}")
    
    assert np.abs(alphas.sum() - alpha0) < 1e-8
    return alphas

def cossim(a, b):
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    cos = dot / (norm_a * norm_b)
    return cos

def create_pairwise_independent_mus(t, d, vmin=-1, vmax=1, maxiter=1000, maxsim=0.95):
    mus = [np.random.uniform(low=vmin, high=vmax, size=(d,))]
    for _ in range(maxiter):
        if len(mus) == t:
            break
        # create new vector
        mu_new = np.random.uniform(low=vmin, high=vmax, size=(d,))

        # check for approx independence
        for mu in mus:
            cos = cossim(mu, mu_new)
            if np.abs(cos) > maxsim:
                break
        else:
            mus.append(mu_new)
    else:
        print("Couldnt finish job!")
    mus = np.array(mus)
    return mus

'''
Matching of estimated parameters and original parameters
'''
def create_cost_matrix(orig_params, estimated_params, cost_func):
    '''
    Creates a cost matrix for a set of estimated and original parameters.
    Value at i,j would be the cost of assigning the estimated index i to the original index j. 
    @Params:
        orig_params...      list of original parameters; shape[0] must be number of components
        estimated_params... list of estimated parameters
        cost_func...        cost function for a permutation of indices
        
    @Returns:
        quadratic cost matrix
    '''
    nmb_indices = orig_params[0].shape[0]
    C = torch.zeros((nmb_indices, nmb_indices))
    for i1 in range(nmb_indices):
        for i2 in range(nmb_indices):
            o_params = [p[i1] for p in orig_params]
            e_params = [p[i2] for p in estimated_params]
            C[i1,i2] = cost_func(o_params, e_params)
    return C

def cost_func(params1, params2):
    '''
    Example for a cost function, summed MSE for each type of parameter.
    '''
    return sum([torch.sqrt(torch.sum((p1.flatten()-p2.flatten())**2)) for p1,p2 in zip(params1,params2)])

def get_matching(orig_params, estimated_params, cost_func):
    '''
    Matches sets of parameters.

    @Params:
        orig_params...      list of original parameters; shape[0] must be number of components
        estimated_params... list of estimated parameters
        cost_func...        cost function for a permutation of indices
        
    @Returns:
        List of indices for estimated parameters that has lowest cost
        If 'l' is this list, estimated_params[l] will sort these parameters
    '''
    C = create_cost_matrix(orig_params, estimated_params, cost_func)
    row_ind, col_ind = linear_sum_assignment(C)
    return col_ind
