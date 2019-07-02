'''
script for importance sampling in posterior obtained through
collapsed gibbs sampling to remove selection effects bias

Step 1: Run vanilla inference on data subject to selection effects

Step 2: Compute posterior distribution from 400 points at the end of
        the collapsed gibbs sampler chain

Step 3: For each of the s 400 posterior chain points, compute the modified
        selection effects log-posterior and compute the set of sample weights:
        for s = 1, ... 400:
            w_i = \mathscr{L^{corr}} / \mathscr{L^{vanilla}}
        
        Then re-compute the posterior distributions as:
        \mathscr{L^{cor}} = \int \mathscr{L^vanilla} \vec{w} \approx \sum_{s=1}^{S} \mathscr{L^{vanilla}}_s w_s

'''

import numpy as np
import pandas as pd

# read in dataset that we ran the inference on
import os, sys
import csv
import numpy as np
import pandas as pd
import time
import bahamas
import priors


job_id = sys.argv[1]
sim_number = sys.argv[2]


datafname = '~/jobs/gibbs_selection/{}[{}].pbs/sel_lcparams.txt'.format(job_id, sim_number)

# load data 
#sim_number = sys.argv[1] #if len(sys.argv) == 3 else 1 # default to dataset 1
names = ['z', 'c', 'x1', 'mb']
data = pd.read_csv(datafname, header=0, sep='\s+', usecols=names)
zHD = data['z'].values
# add second z column so that we have a zHD value
data.insert(loc=1, column='zHD', value=zHD)

data = np.array(data[['z', 'zHD', 'c', 'x1', 'mb']])

sigmaC = np.array(pd.read_csv('~/jobs/gibbs_selection/{}[{}].pbs/sel_simstatssys.txt'.format(job_id, sim_number), sep='\s+', header=None))

log_sigmaCinv = (-1 * np.linalg.slogdet(sigmaC)[1])
sigmaCinv = np.linalg.inv(sigmaC)

ndat = len(data)

J = []
for i in range(ndat):
    J.append([1., 0., 0.])
    J.append([0., 1., 0.])
    J.append([0., 0., 1.])
J  = np.matrix(J)


# define parameters learned
parameters = ['alpha', 'beta', 'rx', 'rc', 'sigma_res',
              'cstar', 'xstar', 'mstar',
              'omegam', 'omegade', 'h', 'loglike']

posterior_params = ['alpha', 'beta', 'rx', 'rc', 'sigma_res',
                    'cstar', 'xstar', 'mstar',
                    'omegam', 'omegade', 'h']

ndim = len(parameters)

# compute offset constant for weights
true_param = [0.13, 2.56, 
            1.0, 0.1, 0.1, 0., 0., -19.3, 0.3, 0.7, 0.72]
#print('vanilla LL at true params: ', bahamas.vanilla_log_likelihood(J, sigmaCinv, log_sigmaCinv, true_param, data, ndat))
#print('corrected LL at true param: ', bahamas.vincent_log_likelihood(J, sigmaCinv, log_sigmaCinv, true_param, data, ndat))
const = bahamas.vincent_log_likelihood(J, sigmaCinv, log_sigmaCinv, true_param, data, ndat) - bahamas.vanilla_log_likelihood(J, sigmaCinv, log_sigmaCinv, true_param, data, ndat)

def compute_weight(J, sigmaCinv, log_sigmaCinv, param, data, ndat, vanilla_log_like):
    corrected_log_like = bahamas.vincent_log_likelihood(J, sigmaCinv, log_sigmaCinv, param, data, ndat)
    print('corrected loglike: ', corrected_log_like)
    print('vanilla loglike: ', vanilla_log_like)
    return np.exp(corrected_log_like - vanilla_log_like - const)



weights = []
weight_param = []

path = '~/jobs/gibbs_selection/{}[{}].pbs/gibbs_chains/'.format(job_id, sim_number)
vanilla_post = pd.read_csv('~/jobs/gibbs_selection/{}[{}].pbs/gibbs_chains/post_chains.csv'.format(job_id, sim_number), sep=',', header=None)
vanilla_post.columns = parameters
vanilla_loglikes = vanilla_post['loglike'][6000::10]
post_params = vanilla_post[posterior_params][6000::10]

for s in range(len(post_params)):
    param_i = list(post_params.iloc[s])
    vanilla_ll = (vanilla_loglikes.values)[s]
    w_i = compute_weight(J, sigmaCinv, log_sigmaCinv, param_i, data, ndat, vanilla_ll)
    weights.append(w_i)
    # now re-weight parameter vectors
    weight_param.append(w_i * np.array(param_i))

#outdir = 
# save weights
fname = 'weights_{}.csv'.format(sim_number)
with open(fname, 'w') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(weights)

# save new posterior chains
fname =  'weighted_post_{}.csv'.format(sim_number)
with open(fname, 'w') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    for p in weight_param:
        wr.writerow(p)