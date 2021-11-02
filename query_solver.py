"""
Sample code automatically generated on 2020-12-14 09:26:58

by geno from www.geno-project.org

from input

parameters
  matrix Mu
  vector y
  vector sigma
  vector alpha
  scalar d
variables
  vector h
min
  d/2*log((sigma.^2)'*h.^2)+norm2(Mu*h-y).^2/(2*(sigma.^2)'*h.^2)-(alpha-vector(1))'*log(h)
st
  h >= 0
  sum(h) == 1


The generated code is provided "as is" without warranty of any kind.
"""

from __future__ import division, print_function, absolute_import

from math import inf
from timeit import default_timer as timer
import numpy as np
import scipy


try:
    from genosolver import minimize, check_version
    USE_GENO_SOLVER = True
except ImportError:
    from scipy.optimize import minimize
    USE_GENO_SOLVER = False
    WRN = 'WARNING: GENO solver not installed. Using SciPy solver instead.\n' + \
          'Run:     pip install genosolver'
    print('*' * 63)
    print(WRN)
    print('*' * 63)

class GenoNLP:
    def __init__(self, Mu, y, sigma, alpha, d):
        self.Mu = Mu
        self.y = y
        self.sigma = sigma
        self.alpha = alpha
        self.d = d
        assert isinstance(Mu, np.ndarray)
        dim = Mu.shape
        assert len(dim) == 2
        self.Mu_rows = dim[0]
        self.Mu_cols = dim[1]
        assert isinstance(y, np.ndarray)
        dim = y.shape
        assert len(dim) == 1
        self.y_rows = dim[0]
        self.y_cols = 1
        assert isinstance(sigma, np.ndarray)
        dim = sigma.shape
        assert len(dim) == 1
        self.sigma_rows = dim[0]
        self.sigma_cols = 1
        assert isinstance(alpha, np.ndarray)
        dim = alpha.shape
        assert len(dim) == 1
        self.alpha_rows = dim[0]
        self.alpha_cols = 1
        if isinstance(d, np.ndarray):
            dim = d.shape
            assert dim == (1, )
        self.d_rows = 1
        self.d_cols = 1
        self.h_rows = self.Mu_cols
        self.h_cols = 1
        self.h_size = self.h_rows * self.h_cols
        # the following dim assertions need to hold for this problem
        assert self.Mu_rows == self.y_rows
        assert self.Mu_cols == self.alpha_rows == self.sigma_rows == self.h_rows

    def getBounds(self):
        bounds = []
        bounds += [(1E-8, inf)] * self.h_size
        return bounds

    def getStartingPoint(self):
        self.hInit = np.full(self.h_rows * self.h_cols, 1.0/(self.h_rows))
        return self.hInit.reshape(-1)

    def variables(self, _x):
        h = _x
        return h

    def fAndG(self, _x):
        h = self.variables(_x)
        t_0 = (self.sigma ** 2)
        t_1 = (h ** 2)
        t_2 = (t_0).dot(t_1)
        t_3 = (t_1).dot(t_0)
        t_4 = ((self.Mu).dot(h) - self.y)
        t_5 = (np.linalg.norm(t_4) ** 2)
        t_6 = (t_0 * h)
        t_7 = (self.alpha - np.ones(self.h_rows))
        f_ = ((((self.d * np.log(t_2)) / 2) + (t_5 / (2 * t_2))) - (t_7).dot(np.log(h)))
        g_0 = (((((self.d / t_3) * t_6) + ((1 / t_3) * (self.Mu.T).dot(t_4))) - (((t_5 * 4) / (4 * (t_3 ** 2))) * t_6)) - (t_7 / h))
        g_ = g_0
        return f_, g_

    def functionValueEqConstraint000(self, _x):
        h = self.variables(_x)
        f = (np.sum(h) - 1)
        return f

    def gradientEqConstraint000(self, _x):
        h = self.variables(_x)
        g_ = (np.ones(self.h_rows))
        return g_

    def jacProdEqConstraint000(self, _x, _v):
        h = self.variables(_x)
        gv_ = ((_v * np.ones(self.h_rows)))
        return gv_

def toArray(v):
    return np.ascontiguousarray(v, dtype=np.float64).reshape(-1)

def solve(Mu, y, sigma, alpha, d):
    start = timer()
    NLP = GenoNLP(Mu, y, sigma, alpha, d)
    x0 = NLP.getStartingPoint()
    x0 = scipy.optimize.nnls(Mu, y)[0]
    bnds = NLP.getBounds()
    tol = 1E-6
    # These are the standard GENO solver options, they can be omitted.
    options = {'tol' : tol,
               'constraintsTol' : 1E-1,
               'maxiter' : 1000,
               'verbosity' : 0  # Set it to 0 to fully mute it.
              }

    if USE_GENO_SOLVER:
        # Check if installed GENO solver version is sufficient.
        check_version('0.0.3')
        constraints = ({'type' : 'eq',
                        'fun' : NLP.functionValueEqConstraint000,
                        'jacprod' : NLP.jacProdEqConstraint000})
        result = minimize(NLP.fAndG, x0,
                          bounds=bnds, options=options,
                          constraints=constraints)
    else:
        constraints = ({'type' : 'eq',
                        'fun' : NLP.functionValueEqConstraint000,
                        'jac' : NLP.gradientEqConstraint000})
        result = minimize(NLP.fAndG, x0, jac=True, method='SLSQP',
                          bounds=bnds,
                          constraints=constraints)

    # assemble solution and map back to original problem
    x = result.x
    eqConstraint000 = NLP.functionValueEqConstraint000(x)
    h = NLP.variables(x)
#    print(h.sum())
#    print(h)
    h = h / np.sum(h)
    solution = {}
    solution['success'] = result.success
    solution['message'] = result.message
    solution['fun'] = result.fun
    solution['grad'] = result.jac
    if USE_GENO_SOLVER:
        solution['slack'] = result.slack
    solution['h'] = h
    solution['eqConstraint000'] = toArray(eqConstraint000)
    solution['elapsed'] = timer() - start
    return solution

def generateRandomData():
#    np.random.seed(0)
    d, t = 100, 10
    Mu = np.random.randn(d, t)
    y = np.random.randn(d)
    sigma = np. full(t, 1.0) # np.random.randn(3)
    alpha = np.full(t, 2.0) # np.random.randn(3)
#    d = np.random.randn(1)
    return Mu, y, sigma, alpha, d

if __name__ == '__main__':
    print('\ngenerating random instance')
    Mu, y, sigma, alpha, d = generateRandomData()
    print('solving ...')
    solution = solve(Mu, y, sigma, alpha, d)
    print('*'*5, 'solution', '*'*5)
    print(solution['message'])
    if solution['success']:
        print('optimal function value   = ', solution['fun'])
        print('norm of the gradient     = ',
              np.linalg.norm(solution['grad'], np.inf))
        if USE_GENO_SOLVER:
            print('maximal compl. slackness = ', solution['slack'])
        print('optimal variable h = ', solution['h'])
        print('norm of the 1st   equality constraint violation ',
              np.linalg.norm(solution['eqConstraint000'], np.inf))
        print('solving took %.3f sec' % solution['elapsed'])
