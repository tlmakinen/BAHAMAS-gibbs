import os, sys
import numpy as np
import pandas as pd

import pymultinest
import bahamas
import priors
import bahamas.get_stats as get_stats

# create output directory
if not os.path.exists('chains'): 
    os.mkdir('chains')

# load data 
sim_number = sys.argv[1] if len(sys.argv) == 2 else 1 # default to dataset 1
names = ['z', 'c', 'x1', 'mb']

datafname = 'sel_lcparams.txt'
datastats = 'sel_simstatsys.txt'
                    
data = pd.read_csv(datafname, header=0, usecols=names, sep='\s+')
zHD = data['z'].values
# add second z column so that we have a zHD value
data.insert(loc=1, column='zHD', value=zHD)
data = np.array(data[['z', 'zHD', 'c', 'x1', 'mb']])


# get covariance values for matrix
sigmaC = np.array(pd.read_csv(datastats, sep='\s+', header=None))


log_sigmaCinv = (-1 * np.linalg.slogdet(sigmaC)[1])
sigmaCinv = np.linalg.inv(sigmaC)

ndat = len(data)

J = []
for i in range(ndat):
    J.append([1., 0., 0.])
    J.append([0., 1., 0.])
    J.append([0., 0., 1.])
J  = np.matrix(J)

# define parameters to learn
parameters = ['alpha', 'beta', 'rx', 'rc', 'sigma_res',
              'cstar', 'xstar', 'mstar',
              'gc', 'gx', 'gm', 'eps',
              'omegam', 'omegade', 'h']

ndim = len(parameters)

eps,gmB,gc,gx1 = (29.96879778, -1.34334963,  0.45895811,  0.06703621)


def prior(cube, ndim=1, nparams=1):
    cube[0] = cube[0] * 1                                 # alpha
    cube[1] = cube[1] * 4                                 # beta
    cube[2] = priors.log_uniform(cube[2], 10**-5, 10**2)  # Rx
    cube[3] = priors.log_uniform(cube[3], 10**-5, 10**2)  # Rc
    cube[4] = cube[4] * 1                                 # sigma_res
    cube[5] = priors.gaussian(cube[5], 0.0, .1)           # cstar
    cube[6] = priors.gaussian(cube[6], 0.0, .1)           # xstar
    cube[7] = priors.gaussian(cube[7], -19.3, .1)         # mstar (M0)
    cube[8] = priors.gaussian(cube[8], 0.4589, .4)        # gc
    cube[9] = priors.gaussian(cube[9], 0.0670, .02)       # gx
    cube[10] = priors.gaussian(cube[10], -1.344, .4)      # gm
    cube[11] = priors.gaussian(cube[11], 29.968, 12)      # eps
    cube[12] = cube[12] * 2                               # omegam
    cube[13] = cube[13] * 2                               # omegade
    cube[14] = .3 + cube[14] * .7                         # h
    return cube

# shape log likelihood to fit with multinest
def log_likelihood_wrapper(cube, ndim, nparams):
    return bahamas.vincent_log_likelihood(J, sigmaCinv, log_sigmaCinv, cube, data, ndat)

# original snls selection pars: -1.7380184749999987, 0.07955165160000005, -2.60483735, 62.15588654
import time
param = [0.13, 2.56, 
            1.0, 0.1, 0.1, 0., 0., -19.3, 0.45895811, 0.06703621, 
                                -1.34334963, 29.96879778, 0.3, 0.7, 0.72]

t0 = time.time()
print('Log likelihood near true theta: ',
        log_likelihood_wrapper(param, 0, 0))
t1 = time.time()
print('LL evaluation time: ', t1 - t0)

# run multinest
n_live_points = 400
max_iter = 0 # 0 means infinity
print('Running multinest with {} live points, {} max_iter on simulation {}'.format(n_live_points, max_iter, sim_number))

t2 = time.time()
pymultinest.run(LogLikelihood=log_likelihood_wrapper, Prior=prior, n_dims=ndim, init_MPI=True, 
        outputfiles_basename='./chains/', verbose=True, n_live_points=n_live_points, max_iter=max_iter)
t3 = time.time()
print('Multinest evaluation time: ', t3 - t2)
