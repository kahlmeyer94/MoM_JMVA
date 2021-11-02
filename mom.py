'''
This script contains functions to apply Method of Moments to a gaussian mixed membership model
'''
import numpy as np
import torch
import math
import opt_einsum as oe
import time
from scipy.optimize import nnls
import query_solver
import inspect
from tqdm import tqdm
from timeit import default_timer as timer


class MoM():
    def __init__(self, alpha_0, t, d, timing=False, **kwargs):
        '''
        @Params:
            alpha_0...      sum of alpha_i from dirichlet prior
            t...            number of topics
            d...            dimensionality of gaussians
        '''
        self.alpha_0 = alpha_0
        self.t = t
        self.d = d

        self.est_thetas = None
        self.est_alphas = None
        self.est_vars = None

        self.timing = timing
        self.times = []
    
    def classify(self, data, print_progress=False):
        ''' 
        Given an output from our model, will calculate the most probably mixture (h)

        @Params:
            data...             iterable object with torch batches as elements (e.g. dataloader) or torch.tensor with data as rows
            print_progress...   [optional] prints tqdm progress bar
        @Returns:
            h, vector (size nmb topics)
        '''
        assert self.est_thetas is not None
        is_batched = type(data) is torch.utils.data.dataloader.DataLoader
        if type(data) is torch.Tensor:
            is_batched = is_batched or len(data.shape) == 3

        if not is_batched:
            assert len(data.shape) == 2
            data = [data]

        ret = []
        # params for solver
        Mu = self.est_thetas.numpy()
        sigma = self.est_vars.numpy()
        alpha = self.est_alphas.numpy()
        n = 0
        fails = []
        for batch in data:
            h_matrix = torch.zeros(batch.shape[0], self.t).double()
            if print_progress:
                pbar = tqdm(range(batch.shape[0]))
            else:
                pbar = range(batch.shape[0])
            for i in pbar:
                y = batch[i]
                n+=1
                solution = query_solver.solve(Mu, y.numpy(), sigma, alpha, Mu.shape[1])
                if not solution['success']:
                    fails.append(n)
                h_matrix[i,:] = torch.as_tensor(solution['h']).double()
            ret.append(h_matrix)
        if len(fails)>0:
            print(f'Warning: Solver failed at {len(fails)} out of {n} datapoints')
        return torch.cat(ret, dim=0)

    def fit(self, data):
        '''
        Uses Method of Moments on dataset.
        Estimated parameters are stored in self.est_thetas, self.est_alphas (and self.est_vars)
        @Params:
        data...             iterable object with torch batches as elements (e.g. dataloader) or torch.tensor with data as rows
        algorithm...        CP decomposition algorithm (see tutorial for examples)
        '''

        is_batched = type(data) is torch.utils.data.dataloader.DataLoader
        if type(data) is torch.Tensor:
            is_batched = is_batched or len(data.shape) == 3

        if not is_batched:
            data = [data]

        
        n = sum([batch.shape[0] for batch in data])
        alg_params = {}
        alg_params['dataloader'] = data
        alg_params['alpha_0'] = self.alpha_0
        alg_params['rank'] = self.t
        alg_params['N'] = n
        alg_params['D'] = self.d
        alg_params['timing'] = self.timing
        if self.timing:
            [est_mus, est_alphas, est_vars], times = power_iteration_whiten(**alg_params)
            self.times = times
        else:
            est_mus, est_alphas, est_vars = power_iteration_whiten(**alg_params)

        self.est_thetas = est_mus
        self.est_alphas = est_alphas
        self.est_vars = est_vars

    def cluster_probs(self, data):
        '''
        Calculates the topic probabilities for a set of data 
        @Params:
            data...     iterable object with torch batches as elements (e.g. dataloader) or torch.tensor with data as rows

        @Returns:
           n x t matrix with log-probabilities of topic t generating datapoint n
        '''
        assert self.est_thetas is not None
        is_batched = type(data) is torch.utils.data.dataloader.DataLoader
        if type(data) is torch.Tensor:
            is_batched = is_batched or len(data.shape) == 3

        if not is_batched:
            data = [data]
            
        if self.est_alphas is not None:
            probs = []
            dists = [torch.distributions.multivariate_normal.MultivariateNormal(loc = self.est_thetas[:,i], covariance_matrix= self.est_vars[i]*torch.eye(self.est_thetas.shape[0]).double()) for i in range(self.t)]
            for batch in data:
                prob_part = torch.zeros(batch.shape[0], self.t)
                for i in range(self.t):
                    prob_part[:,i] = dists[i].log_prob(batch)
                probs.append(prob_part)
            return torch.cat(probs, dim=0)
        else:
            print('Unable to find estimated parameters!\nFit to dataset first!')


'''
Functions used by the Algorithm
'''

def power_iteration_whiten(dataloader, alpha_0, rank, N, D, dev='cpu', generator_args={}, timing=False, **kwargs):
    '''
    Applies Method of Moments powered by Power iteration to a dataset to retrieve original parameters.
    Note: This version uses optimized whitening

    @Params:
        dataloader...   iterable object with batches as elements or generator function, each batch is of size batchsize x D
        alpha_0...      sum of alpha_i from dirichlet prior
        rank...         number of mixtures    
        N...            number of data
        D...            dimensionality of data
        prob...         with probability prob we will get good estimations
        epsilon...      parameter for tensor_power_method
        generator_args...       dict of arguments for the dataloader if dataloader is a generator function

    @Returns:
        List of estimated parameters: 
        est_mus...      D x K matrix with mean vectors as columns
        est_alphas...   vector of size K with alpha_i
        est_sigmas...   vector of size K with sigma_i
    '''
    times = []

    c = 2/(alpha_0+2)

    # 1. Create empirical moments + whiten T
    s_time = timer()
    M, T, eta, B, tmp_times = create_moments(dataloader, N, D, alpha_0, dev=dev, whiten=True, rank=rank, timing=True, generator_args=generator_args)
    e_time = timer()
    times += tmp_times

    # 2. Calculate CP decomposition
    s_time = timer()
    e_vectors, e_values = tensor_power_method(T, rank, dev=dev, epsilon=1e-10)
    e_time = timer()
    times.append(e_time-s_time)

    # 3. Unwhiten estimated parameters
    s_time = timer()
    lambdas, est_mus = unwhiten_params(B, e_vectors, e_values, c)
    e_time = timer()
    times.append(e_time-s_time)

    # 4. Transfer eigenvalue to alpha_i
    est_alphas = lambdas*((alpha_0+1)*alpha_0)
    est_alphas *= alpha_0/torch.sum(est_alphas)

    # 5. Reconstruct sigma_i^2
    s_time = timer()
    est_vars = estimate_variances(est_mus, est_alphas, eta, alpha_0=alpha_0, dev=dev)
    e_time = timer()
    times.append(e_time-s_time)

    if timing:
        return [est_mus, est_alphas, est_vars], times
    else:
        return [est_mus, est_alphas, est_vars]

def get_whitening_matrices(M, rank):
    '''
    Calculates the whitening and unwhitening matrices.

    @Params:
        M...    Matrix of which to calculate the whitening matrices from

    @Returns:
        Whitening matrix W, Unwhitening matrix B
    '''

    U,D,V = torch.svd(M)
    U = U[:,:rank]
    D = 1/D[:rank]
    W = U@torch.sqrt(torch.diag(D))
    B = U@torch.sqrt(torch.diag(1/D))  
    return W.double(),B.double()

def contract_tensor(T,A,v):
    '''
    Tensor contraction used in Power Method.
    @Params:
        T...    Tensor of shape KxKxK
        A...    Matrix of shape KxK
        v...    Vector of shape K
    
    @Returns:
        Tensor T contracted with A, v and v in its three dimensions
    '''
    ret = np.einsum('ijk,k->ij',T,v)
    ret = np.einsum('ij,j->i',ret,v)
    return np.einsum('i, ij -> j', ret, A)

def mult_lin_transf(T,A,B,C):
    '''
    Calculates the multilinear transform

    @Params:
        T...    3D tensor
        A...    2D tensor
        B...    2D tensor
        C...    2D tensor

    @Returns:
        T(A,B,C) as a 3D tensor
    '''
    h,w,c = T.shape
    assert A.shape[0] == h
    assert B.shape[0] == w
    assert C.shape[0] == c
    if len(A.shape)==1:
        A = A.unsqueeze(1)
    if len(B.shape)==1:
        B = B.unsqueeze(1)
    if len(C.shape)==1:
        C = C.unsqueeze(1)
    return oe.contract('abc, ai, bj, ck -> ijk', T.double(),A.double(),B.double(),C.double())

def outer_prod(x,y):
    '''
    Calculates the outer product of x and y. 
    Note, at least x or y should be 1D!

    @Params:
        x...    torch tensor, 1D or 2D
        y...    torch tensor, 1D or 2D
    
    @Returns:
        outer product of x and y
    '''


    dim_x = len(x.shape)
    dim_y = len(y.shape)
    
    if dim_x==1 and dim_y==1:
        return oe.contract('i,j->ij', x,y)
    elif dim_x==2 and dim_y==1:
        return oe.contract('ij,k->ijk', x,y)
    elif dim_x==1 and dim_y==2:
        return oe.contract('i,jk->ijk', x,y)
    else:
        return None

def triple_prod(a,b,c):
    '''
    Calculates the outer product of a,b and c. 

    @Params:
        a...    torch tensor, 1D
        b...    torch tensor, 1D
        c...    torch tensor, 1D
    
    @Returns:
        outer product of a,b and c
    '''
    return oe.contract('i,j,k->ijk', a,b,c)

def create_moments(dataloader, N, D, alpha_0, dev='cpu', whiten=False, rank=None, timing=False, generator_args={}):
    '''
    Creates 2nd Moment Matrix M and 3rd Moment Tensor T
    
    @Params:
        dataloader...   iterable object with batches as elements or generator, each batch is of size batchsize x D
        N...            number of data
        D...            dimensionality of data
        alpha_0...      sum of alpha_i from Dirichlet Prior
        dev...          one of 'cpu' or 'cuda:n' with n being the device number
        whiten...       if set, will perform whitening while constructing the T 
        rank...         has to be set if whiten is True
        generator_args...       dict of arguments for the dataloader if dataloader is a generator function
    @Returns:
        if whiten is set:
            M, T, eta (needed for retrieval of sigma_i^2)
    '''
    times = []

    device = torch.device(dev) 

    # calculate some statistics from batches
    is_generator = inspect.isgeneratorfunction(dataloader)

    # First and second moment
    s_time = timer()
    exp_y = torch.zeros(D).to(device)
    exp_y_y = torch.zeros((D,D)).to(device)
    if is_generator:
        loader = dataloader(**generator_args)
    else:
        loader = dataloader
    for batch in loader:
        batch = batch.to(device)
        exp_y += torch.sum(batch, dim=0)
        exp_y_y += batch.T @ batch
    exp_y /= N
    exp_y_y /= N
    exp_y = exp_y.double()
    exp_y_y = exp_y_y.double()
    e_time = timer()
    times.append(e_time-s_time)
    
    # Covariance matrix
    s_time = timer()
    cov_y = torch.zeros((D,D)).to(device)
    if is_generator:
        loader = dataloader(**generator_args)
    for batch in loader:
        batch = batch.to(device)
        tmp = batch-exp_y
        cov_y += tmp.T @ tmp
    cov_y /= (N-1)
    cov_y = cov_y.double()
    # estimation of sigma^2
    values, vectors = torch.symeig(cov_y, eigenvectors=True)
    v_min = vectors[:, 0]
    lambda_min = values[0].item()
    outer_exp_y = outer_prod(exp_y, exp_y)
    I = torch.eye(D).double().to(device)
    e_time = timer()
    times.append(e_time-s_time)

    # 1. Create M
    s_time = timer()
    # scale
    s_0 = 1
    # Error term
    E_0 = -lambda_min*I-(alpha_0/(alpha_0+1)*outer_exp_y)
    # create M
    M = s_0*exp_y_y+E_0

    if whiten:
        assert rank is not None
        W,B = get_whitening_matrices(M, rank)

    e_time = timer()
    times.append(e_time-s_time)
    
    
    # 2. Create T
    s_time = timer()
    if whiten:
        exp_y_y_y = torch.zeros((rank,rank,rank)).to(device)
        if is_generator:
            loader = dataloader(**generator_args)
        for batch in loader:
            batch = batch.to(device).double()
            exp_y_y_y += oe.contract('ni,nj,nk,ia,jb,kc->abc', batch, batch, batch, W, W, W)
        exp_y_y_y /= N
    else:
        exp_y_y_y = torch.zeros((D,D,D)).to(device)
        if is_generator:
            loader = dataloader(**generator_args)
        for batch in loader:
            batch = batch.to(device)
            exp_y_y_y += oe.contract('ni,nj,nk->ijk', batch, batch, batch)
        exp_y_y_y /= N
    exp_y_y_y = exp_y_y_y.double()

    # scale
    s_1 = 1

    # Error Term
    H_mu_mu = exp_y_y-lambda_min*I

    eta = torch.zeros(D).to(device)
    if is_generator:
        loader = dataloader(**generator_args)
    for batch in loader:
        batch = batch.to(device)
        tmp = torch.sum(batch.T* (v_min.double() @ ((batch-exp_y)).T.double())**2, dim=1)
        eta += tmp
    eta /= N
    eta = eta.double()

    if whiten:
        tmp = oe.contract('ij,k,ia,jb,kc->abc', H_mu_mu, exp_y, W, W, W)
        term_1 = -alpha_0/(alpha_0+2)*(tmp.permute(2,1,0)+ tmp.permute(0,2,1) + tmp)
        

        tmp = oe.contract('i,j,k,ia,jb,kc->abc', exp_y, exp_y, exp_y, W, W, W)
        term_2 = 2*alpha_0**2/((alpha_0+2)*(alpha_0+1))*tmp

        tmp = eta.repeat((D,1)).double().to(device)
        term_3 = torch.zeros((rank,rank,rank)).to(device)
        term_3 += oe.contract('ij,ia,ib,jc->abc', tmp, W, W, W)
        term_3 += oe.contract('ij,ia,jb,ic->abc', tmp, W, W, W)
        term_3 += oe.contract('ij,ja,ib,ic->abc', tmp, W, W, W)
        
    else:
        tmp = outer_prod(H_mu_mu, exp_y)
        term_1 = -alpha_0/(alpha_0+2)*(tmp.permute(2,1,0)+ tmp.permute(0,2,1) + tmp)

        term_2 = 2*alpha_0**2/((alpha_0+2)*(alpha_0+1))*outer_prod(exp_y, outer_prod(exp_y,exp_y))
        
        term_3 = torch.zeros((D,D,D)).to(device)
        tmp = eta.repeat((D,1)).double().to(device)
        term_3 += oe.contract('ij,ia,ib,jc->abc', tmp, I, I, I)
        term_3 += oe.contract('ij,ia,jb,ic->abc', tmp, I, I, I)
        term_3 += oe.contract('ij,ja,ib,ic->abc', tmp, I, I, I)

    E_1 = term_1+term_2-term_3
    # Create T
    T = s_1*exp_y_y_y+E_1
    e_time = timer()
    times.append(e_time-s_time)

    if whiten and timing:
        return M.double(), T.double(), eta.double(), B.double(), times
    return M.double(), T.double(), eta.double()

def unwhiten_params(B, theta, lambdas, c):
    '''
    Performs unwhitening of estimated parameters

    @Params:
        B...        Unwhitening Matrix (Pseudoinverse of Whitening Matrix)
        theta...    Matrix with estimated eigenvectors as columns
        lambdas...  Array of estimated eigenvectors
        c...        Scaling factor, from theoretical CP decomposition of 3rd order tensor

    @Returns:
        Unwhitened lambdas, thetas
    '''

    lambdas_est = (c**2)/(lambdas**2)
    theta_est = (1/c)*lambdas*(B@theta)
    return lambdas_est, theta_est

def tensor_power_method(T, rank, max_iterations=1000, max_searches=150, dev='cpu', epsilon=1e-10, warn=False):
    '''
    Calculates Eigenvalue, Eigenvector decomposition of a 3D Tensor

    @Params:
        T...            3D Tensor
        rank...         rank of T (number of eigenvalues)
        n_iterations... maximum number of power updates for each eigenvalue, if set to negative->iterate until convergence
        n_searches...   number of times it is searched for the greatest eigenvalue
        epsilon...      criterium to abort power iterations, if found eigenvector does not change more than epsilon

    @Returns:
        e_vectors, matrix with eigenvectors as columns
        e_values, array with eigenvalues
    '''
    device = torch.device(dev) 

    d = T.shape[0]
    e_values = torch.zeros(rank).to(device)
    e_vectors = torch.zeros((d,rank)).to(device)
    
    I = torch.eye(d)
    T_tmp = T.clone()
    for k in range(rank):
        conv = False
        count_search = 0
        while not conv and count_search<max_searches:
            count_search += 1
            e_vector = (torch.rand(d)*2-1).to(device)
            e_vector /= torch.norm(e_vector)
            count_it = 0
            while not conv and count_it<max_iterations:
                count_it += 1
                prev_vector = e_vector
                new_vec = torch.as_tensor(contract_tensor(T_tmp.numpy(), I.numpy(), e_vector.numpy()))
                e_vector = (new_vec/torch.norm(new_vec))
                conv = torch.norm(prev_vector-e_vector).item()<epsilon
        if count_search==max_searches and warn:
            print(f'Warning: Tensor power method did not converge for {k}-th component')

        vec = e_vector
        value = (mult_lin_transf(T_tmp,torch.eye(d).to(device),e_vector,e_vector).flatten()/e_vector.flatten())[0] 
        # store eigenvalue, eigenvector
        e_values[k] = value
        e_vectors[:,k] = vec
        # deflate tensor
        T_tmp -= value*outer_prod(vec,outer_prod(vec,vec))
        
    order = torch.argsort(e_values)
    e_values = e_values[order]
    e_vectors = e_vectors[:,order]

    return e_vectors.double(),e_values.double()

def estimate_variances(mus, alphas, eta, alpha_0=None, dev='cpu', force_pos=False):
    '''
    Estimates sigma_i from definition of eta

    @Params:
        mus...          matrix of size DxK; estimated mean vectors
        alphas...       vector of size K; estimated alpha_i
        eta...          vector of size D; from create moments function
        alpha_0...      True alpha0, if unknown, set to sum of estimated alpha_i
        dev...          'cpu' or 'cuda:x' with x being the device
        force_pos...    If True, will estimate variances with non negative least squares

    @Returns:
        vector of size K, contains variances (sigma^2)
    '''
    device = torch.device(dev)

    D,K = mus.shape
    if alpha_0 is None:
        alpha_0 = torch.sum(alphas)
    

    V = torch.zeros((D,K)).double().to(device)
    factors = (alphas+1)*alphas/((alpha_0+2)*(alpha_0+1)*alpha_0)
    for i in range(K):
        V[:, i] = factors[i]* (mus @ alphas + 2* mus[:,i])

    if force_pos:
        return torch.as_tensor(nnls(V, eta)[0])
    else:
        return torch.pinverse(V) @ eta

def to_tensor(weights, A, B, C):
    '''
    Creates Tensor from CP decomposition (asymmetric)

    @Params:
        weights...  scalar factors for each rank
        A...        Matrix with vectors as columns
        B...        Matrix with vectors as columns
        C...        Matrix with vectors as columns

    @Returns:
        3D Tensor
    '''


    k = A.shape[1]
    sum_tensor = torch.zeros((A.shape[0],B.shape[0],C.shape[0]))
    for i in range(k):
        sum_tensor+= weights[i]*triple_prod(A[:,i],B[:,i],C[:,i])
    return sum_tensor

if __name__ == "__main__":
    pass