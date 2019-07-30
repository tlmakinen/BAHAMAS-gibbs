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


from bahamas import gibbs_library, gibbs_sampler_selection, get_stats

# create output directory
outdir = './gibbs_chains/'
if not os.path.exists(outdir): 
    os.mkdir(outdir)

datafname = 'sel_lcparams.txt'

# load data 
sim_number = sys.argv[1] #if len(sys.argv) == 3 else 1 # default to dataset 1
names = ['z', 'c', 'x1', 'mb']
data = pd.read_csv(datafname, header=0, sep='\s+', usecols=names)
#print(data)
zHD = data['z'].values
# add second z column so that we have a zHD value
data.insert(loc=1, column='zHD', value=zHD)

data = np.array(data[['z', 'zHD', 'c', 'x1', 'mb']])

sigmaC = np.array(pd.read_csv('sel_simstatssys.txt', sep='\s+', header=None))

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
    cube[6] = priors.gaussian(cube[6], 0, 10**2)       # xstar
    cube[7] = priors.gaussian(cube[7], -19.3, 2**2)        # mstar
    cube[8] = cube[8] * 1                                # omegam
    cube[9] = cube[9] * 1                                # omegade   EDIT
    cube[10] = 0.3 + cube[10] * 0.7                      # h   EDIT
    return cube

# true theta for likelihood
#param = [.14, 3.2, np.exp(.560333), np.exp(-2.3171), .1, -0.06, 0.0, -19.1, .3, -1, .7]
param = [0.13, 2.56, 
            1.0, 0.1, 0.1, 0., 0., -19.3, 0.3, 0.7, 0.72]

# Create Gibbs Posterior Model Object
t1 = time.time()
posterior_object_for_sample = gibbs_library.posteriorModel(J, sigmaCinv, log_sigmaCinv, data, ndat)
log_post = posterior_object_for_sample.log_like_selection(param)
t2 = time.time()
print('log-likelihood near true theta = ', log_post)
print('log-posterior evaluation time = ', t2-t1)


# make prior cube
cube = gibbs_library.makePriorCube(ndim)
# start off params with priors
prior = makePrior(cube)

# just to test, freeze cosmo
#prior[:2] = [0.13, 2.56]
#prior[8:10] = [0.3, 0.7]

print('prior = ', prior)

#niters = 2  # iterations for gibbs
#niters_burn = 3
niters_burn = int(sys.argv[2]) #if len(sys.argv) == 3 else niters_burn
niters = int(sys.argv[3]) #if len(sys.argv) == 3 else niters


t1 = time.time()
# run the sampler to estimate posterior distributions
n_cosmo, n_B = gibbs_sampler_selection.runGibbs(prior=prior, posterior_object_for_sample=posterior_object_for_sample, ndim=ndim, 
                                                niters=niters, niters_burn=niters_burn, outdir=outdir, diagnose=True, snapshot=False, datafname=datafname)
t2 = time.time()

print('Gibbs Sampler Evaluation time = ', t2-t1)

# save run stats for later
import csv
fname = outdir + 'sampler_stats.csv'
with open(fname, 'w') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        #for i in [[n_accept_cosmo/niters , n_accept_B/niters]]:
    wr.writerow([n_cosmo, n_B, t2-t1])

# make latent diagnostic posterior plot
from latent_plots import plot_post_means
start = int(niters * 0.4)
D = pd.read_csv('./gibbs_chains/D_latent.csv', sep=',', header=None)[start::10] # thin chain
datafname = 'sel_lcparams.txt'

plot_post_means(D, datafname)






