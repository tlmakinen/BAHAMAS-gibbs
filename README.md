# BAHAMAS
BAyesian HierArchical Modeling for the Analysis of Supernova Cosmology

## Background

BAHAMAS is a software package used to analyze supernova cosmology. At
a high level, this means determination of cosmological parameters given
data as well as investigating our current model for supernovae.

The bones of BAHAMAS are laid out in [March et al. (2011)](https://spiral.imperial.ac.uk:8443/bitstream/10044/1/28655/7/MNRAS-2011-March-2308-29.pdf) and 
formalized/extended upon in [Shariff et al. (2016)](https://arxiv.org/abs/1510.05954). For further
background, read [Kelly (2007)](http://iopscience.iop.org/article/10.1086/519947/pdf).

## BAHAMAS Gibbs Sampling

This is a re-implementation of the posterior sampling scheme laid out in [Shariff et al. (2016)]. While much slower than mpi-enabled sampling schemes (see BAHAMAS with MultiNest), Gibbs Sampling gives us full access to our posterior distribution at all sampling steps, enabling nonsymmetric tweaking of our hierarchical model. 

The sampler is to be augmented with selection effects bias correction (in progress), and can be run in the same environment as MutiNest-enabled BAHAMAS. In its current implementation, the model can probe both Lambda-CDM and wCDM cosmological models, varying either Dark Energy density or equation of state parameters, respectively.

## Dependencies

Standard Python modules (e.g. Numpy, Scipy, Pandas) are required, as well as the specialty packages scikit-learn

## Contents

1) run_vanilla_gibbs.sh
    Shell script for executing job on HPC cluster. Working on OpenMP threading to speed up computation

2) `run_gibbs*.py` 
    "Main" script for Python execution. Requires data filename and number of iterations as arguments. 

3) `bahamas/`
    Folder with all the main ingredients for posterior computation:
    i) gibbs_library.py
        Script for matrix manipulation and log-posterior computation for sampler

    ii) `gibbs_sampler.py`
        Code for iterative steps of the Gibbs Sampler. Calls on aspects defined in library script.

    iii) `cosmology.py`
        Code for computing distance modulus as a function of cosmological parameters and observed variables. Can be changed for Lambda-CDM or wCDM cosmological models

    iv) `selection_effects.py`
        Code for implementing selection effects addition to log-posterior. Outlined in Chen et al (in prog)

    v) `get_stats.py`
        Helper function for creating observed covariance matrix

4) `data/`
    i) `sn1a_generator.py`
        Code for generating JLA-like simulations according to BAHAMAS framework to test inference 

## Posterior Sampling

Modify and run `python run_gibbs.py` to take in the dataset desired. Computation time for 11,000 iterations (niters_burn = 1000, niters = 10,000) single-node CPU is between 8 and 12 hours in current implementation

If using the Imperial HPC, run `qsub run_vanilla_gibbs.sh` to run `run_gibbs.py` on the HPC. To learn
more about the HPC, take some time to explore [this](http://www.imperial.ac.uk/admin-services/ict/self-service/research-support/rcs/support/getting-started/).


### Posterior Analysis
Output from the sampler, unless otherwise directed, will be located in a directory labeled `gibbs_chains/post_chains.csv`. This file contains the trace, or posterior "chains" of the sampler, with burn-in iterations discarded. Thinning the chain by every 10th element should yield a plottable distribution.

Use one of the Jupyter Notebook files located in `notebooks` for `get_dist`-enhanced corner plot analysis of the posterior.

## Authors
I believe until this point, the list of authors includes Hik Shariff, Wahidur Rahman,
Dylan Jow, Evan Tey, and Lucas Makinen.