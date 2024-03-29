'''
Run script for executing Shariff et al's (2016) Gibbs Sampler on Imperial's HPC.
Sampler built in gibbs_sampler.py, with attributes comupted in gibbs_library.py
'''
import os, sys
import numpy as np
import pandas as pd
import time
import bahamas
import priors

from bahamas import gibbs_library, gibbs_sampler, get_stats

# create output directory
if not os.path.exists('gibbs_chains'): 
    os.mkdir('gibbs_chains')

# load data 
sim_number = sys.argv[1] if len(sys.argv) == 2 else 1 # default to dataset 1
names = ['CID', 'zCMB', 'zCMBERR', 'zHD', 'zHDERR', 
            'x1', 'x1ERR', 'c', 'cERR', 'mB', 'mBERR', 'x0', 
                    'x0ERR', 'COV_x1_c', 'COV_x1_x0', 'COV_c_x0']

datafname = '/c/Users/lucas/Datasets/mlacceleration/SNe_samples/test_{}.txt'.format(sim_number)      
data = pd.read_csv(datafname, header=0, sep='\s+')

# use helper function to build covariance cube
sigmaC,data = get_stats.get_cov_cube(data)
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
              'omegam', 'omegade', 'h']

ndim = len(parameters)


# prior cube for each parameter. Takes as input uniform prior cube
def makePrior(cube, ndim=1, nparams=1):
    cube[0] = np.random.uniform(0, 1)                    # alpha
    cube[1] = np.random.uniform(0, 4)                    # beta
    cube[2] = priors.log_uniform(cube[2], 10**-5, 10**2) # Rx
    cube[3] = priors.log_uniform(cube[3], 10**-5, 10**2) # Rc (CHANGE FOR NON-SYMMETRIC)
    cube[4] = cube[4] * 1                                # sigma_res (in likelihood code)
    #cube[4] = np.exp(priors.log_invgamma(cube[4], 0.003, 0.003))
    cube[5] = priors.gaussian(cube[5], 0, 1**2)          # cstar 
    cube[6] = priors.gaussian(cube[6], .0796, .02)       # xstar
    cube[7] = priors.gaussian(cube[7], -19.3, 2.)        # mstar
    cube[8] = cube[8] * 1                                # omegam
    cube[9] = cube[9] * 1                                # omegade
    #cube[9] = cube[9] * -4                               # w   EDIT
    cube[10] = 0.3 + cube[10] * 0.7                      # h   EDIT
    return cube

# true theta for likelihood
#param = [.14, 3.2, np.exp(.560333), np.exp(-2.3171), .1, -0.06, 0.0, -19.1, .3, 0.7, .7]
param = [0.14, 3.2, 
            1.6790592304550125, 0.09821477355778524, 0.1, 0.002154842112638425, -0.042003094724252336, -19.362573244911182, 0.3, 0.7, 0.7]

# Create Gibbs Posterior Model Object
t1 = time.time()
posterior_object_for_sample = gibbs_library.posteriorModel(J, sigmaCinv, log_sigmaCinv, data, ndat)
log_post = posterior_object_for_sample.log_likelihood(param)
t2 = time.time()
print('log-likelihood near true theta = ', log_post)
print('log-posterior evaluation time = ', t2-t1)

# set up empty D column-stacked latent variable array
D = []
for i in range(ndat):
    D.append(np.zeros((1,3)))

# make prior cube
cube = gibbs_library.makePriorCube(ndim)
# start off params with priors
prior = makePrior(cube)


niters = 7000  # iterations for gibbs
niters_burn = 1000

niters = int(sys.argv[1]) if len(sys.argv) == 3 else niters
niters_burn = int(sys.argv[2]) if len(sys.argv) == 3 else niters_burn


t1 = time.time()
# run the sampler to estimate posterior distributions
#gibbs_sampler.runGibbsFreeze(prior=prior, param=param, posterior_object_for_sample=posterior_object_for_sample, ndim=ndim, 
  
gibbs_sampler.runGibbs(prior=prior, posterior_object_for_sample=posterior_object_for_sample, ndim=ndim, 
                                                niters=niters, niters_burn=niters_burn, outdir='./gibbs_chains/', plot=True, datafname=datafname)                                             # niters=niters, niters_burn=niters_burn, outdir='./gibbs_chains/', plot=True)
t2 = time.time()

print('Gibbs Sampler Evaluation time = ', t2-t1)






