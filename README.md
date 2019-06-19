# BAHAMAS
BAyesian HierArchical Modeling for the Analysis of Supernova Cosmology

## Background

BAHAMAS is a software package used to analyze supernova cosmology. At
a high level, this means determination of cosmological parameters given
data as well as investigating our current model for supernovae.

The bones of BAHAMAS are laid out in [March et al. (2011)](https://spiral.imperial.ac.uk:8443/bitstream/10044/1/28655/7/MNRAS-2011-March-2308-29.pdf) and 
formalized/extended upon in [Shariff et al. (2016)](https://arxiv.org/abs/1510.05954). For further
background, read [Kelly (2007)](http://iopscience.iop.org/article/10.1086/519947/pdf).

## Dependencies

The main library we use is [MultiNest](https://ccpforge.cse.rl.ac.uk/gf/project/multinest/)
for posterior sampling. Follow [these instructions](http://johannesbuchner.github.io/PyMultiNest/install.html)
to install MultiNest and PyMultiNest, a Python wrapper for MultiNest.

If MPI is being used to parallelize sampling, `mpi4py` needs to be installed. 

You'll also need to have a few other common libraries (e.g. `pandas`, `numpy`, etc.).

## Usage

### Posterior Sampling
Run `python run.py` to start generating posterior samples. Depending on the complexity
of your setup (e.g. number of parameters, sampling efficiency, etc.) sampling can take on
the order of minutes to hours (or even days) to converge.

If using the Imperial HPC, run `qsub run.sh` to run `run.py` on the HPC. To learn
more about the HPC, take some time to explore [this](http://www.imperial.ac.uk/admin-services/ict/self-service/research-support/rcs/support/getting-started/).

### Posterior Analysis
Once the sampling converges, you'll be able to find the posterior samples in 
`chains/post_equal_weights.dat`. 

Posterior analysis can be conducted using the notebooks found in `notebooks/`.

## Troubleshooting
If your sampling has problems when you start to use MPI and you're working on the 
Imperial HPC, make sure you have the conda version of `mpi4py` because the pip version
won't work (as of Summer 2018).

## Authors
I believe until this point, the list of authors includes Hik Shariff, Wahidur Rahman,
Dylan Jow, Evan Tey, and Lucas Makinen.