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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


from bahamas import gibbs_library, gibbs_sampler_selection, get_stats

# create output directory
outdir = './gibbs_chains/'
if not os.path.exists(outdir): 
    os.mkdir(outdir)

datafname = 'lc_params.txt'

# load data 
sim_number = sys.argv[1] #if len(sys.argv) == 3 else 1 # default to dataset 1
names = ['z', 'c', 'x1', 'mb']
data = pd.read_csv(datafname, header=0, sep='\s+', usecols=names)
#print(data)
zHD = data['z'].values
# add second z column so that we have a zHD value
data.insert(loc=1, column='zHD', value=zHD)

data = np.array(data[['z', 'zHD', 'c', 'x1', 'mb']])

sigmaC = np.array(pd.read_csv('sim_statssys.txt', sep='\s+', header=None))

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
    cube[5] = priors.gaussian(cube[5], 0, 1)          # cstar 
    cube[6] = priors.gaussian(cube[6], 0, 10)       # xstar
    cube[7] = priors.gaussian(cube[7], -19.3, 2)        # mstar
    cube[8] = cube[8] * 1                                # omegam
    cube[9] = cube[9] * 1                                # omegade   EDIT
    cube[10] = 0.3 + cube[10] * 0.7                      # h   EDIT
    return cube

# true theta for likelihood
#param = [.14, 3.2, np.exp(.560333), np.exp(-2.3171), .1, -0.06, 0.0, -19.1, .3, -1, .7]
true_param = [0.13, 2.56, 
            1.0, 0.1, 0.1, 0., 0., -19.3, 0.3, 0.7, 0.72]

# Create Gibbs Posterior Model Object
t1 = time.time()
posterior_object_for_sample = gibbs_library.posteriorModel(J, sigmaCinv, log_sigmaCinv, data, ndat)
log_post = posterior_object_for_sample.log_likelihood(true_param)
t2 = time.time()
print('log-likelihood near true theta = ', log_post)
print('log-posterior evaluation time = ', t2-t1)

# LIKELIHOOD SCANS
parameters = [ 'alpha', 'beta', 'rx', 
                'rc', 'sigma_res',
              'cstar', 'xstar', 'mstar',
           #   'gc', 'gx', 'gm', 'eps',
              'omegam', 'omegade']

param_prior_range = {
    'alpha': (0.05, .5), 'beta': (1., 5.6),
    'rx': (.1, 1.5), 'rc':(0.01, 0.20), 'sigma_res':(0.001,0.30),
    'cstar': (-.005, 0.02), 'xstar':(-1., 0.8), 'mstar':(-19.6, -19.1),
    'omegam': (0., 2), 'omegade': (0, 2), 'h': (.675, .725), 
    'loglike': (-900, -200)
}


# perform likelihood scans over prior ranges:
param = true_param
for i in range(len(parameters)):
    scan = []
    start = param_prior_range[parameters[i]][0]
    stop = param_prior_range[parameters[i]][1]
    param_range = np.linspace(start, stop, 40)

    print(param, i)

    for dx in param_range:
        
        param[i] = dx  # increment desired parameter over prior range
        print('parameter', param)
        loglike = posterior_object_for_sample.log_likelihood(param)
        print('current param and loglike', parameters[i], param[i], loglike)
        scan.append(loglike)

        param = [0.13, 2.56, 
            1.0, 0.1, 0.1, 0., 0., -19.3, 0.3, 0.7, 0.72]

    # make plot
    mask = (np.array(scan) > -1e90)
    plt.scatter(param_range[mask], np.array(scan)[mask], marker='+', label=parameters[i])
    plt.legend(loc='best')
    plt.ylabel('corrected loglike')
    plt.xlabel(parameters[i])
    fname = './vanilla_on_vanilla_likelihood_scans/' + parameters[i] + '.png'
    plt.savefig(fname, dpi='figure')
    plt.close()

print('dataset length: ', len(data))








