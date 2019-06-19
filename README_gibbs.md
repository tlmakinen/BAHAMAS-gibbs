# TEST TEST TEST

# README file for Gibbs Sampling work

The Gibbs sampling method follows the work in the Appendix of [Shariff et al (2016)]:
    https://arxiv.org/pdf/1510.05954.pdf

This method serves as a diagnostic for the larger BAHAMAS model in supernova cosmology. We want
to take a look at the posterior population-level distributions that our probabilistic model puts 
out to compare the effect of skewed color distributions on our inference. BAHAMAS currently 
propagates a symmetric gaussian in c*, so that we can marginalize latents out for computational
efficiency.

This sampling routine gives us a way to view these posterior distributions. Module created by Lucas 
Makinen in December 2018.

## SNANA Inference Data

Data for inference takes the form of 10 files of ~500 simulated supernova observations (created with
SNANA simulations), some subject to selection effects, and some "ideal" simulations. Each selection effects
file is labeled `selected_i.txt`, while ideal non-selection effect SNe are labeled `ideal_i.txt`.
We uniformly draw each set of 500 supernovae randomly from a bulk simulation generation file. On a high performance 
computing cluster, an inference is performed on each file parallel. The posterior distributions of each of 
these runs can be compared to assess average performance of our inference machinery.

Test inference data is located in `SNe_samples/`.

## Gibbs Sampling within the BAHAMAS environment

The BAHAMAS Gibbs Sampler is designed to work within the same framework as the pymultinest
posterior evaluation method.

## Attributes of the Posterior Distribution

To run the Gibbs Sampler, the Python script `run_gibbs.py`, located at the top level of the repository, is modified 
according to the datasets to be used in the inference. Data is passed into and stored in a posterior distribution object whose attributes can be 
accessed to streamline computation of whacky matrices.

To streamline the package, desired attributes of the posterior distribution are computed
in `gibbs_library.py`, located in the bahamas directory.

The algorithm itself is coded in `gibbs_sampler.py`, in which both latent and cosmological parameters are
sampled in sequential steps. This file is located in the `bahamas` sub-directory of the repository.

## Running a Gibbs Sampler Job

As reported by Shariff et al (2016), the sampler takes roughly 15,000 iterations to converge. Before running the full sampler,
however, a burn-in chain (run "niters_burn" times) is needed to estimate proposal distributions for Metropolis-Hastings sampling

For performance reasons, it is recommended that users run the package on a high throughput computing bank, like Imperial College's 
HPC system. In this example, the Gibbs Sampler job shell script is `run_vanilla_gibbs.sh`. 

Posterior chains are spit into the file `chains_gibbs/post_chains.csv`, located in the repository main level.

## Setting Up & Testing the sampler

To make the Gibbs runnable in a new environment, the shell script and `run_gibbs.py` must be edited to point
the setup procedure to the datasets for inference. To test that the environment works, point the `data` variable in
the run script to the dataset `snana_des/SNe_samples`. Set `niters` and `niters_burn` to a low number and see that the sampler works in
your environment. 