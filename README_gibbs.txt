# README file for Gibbs Sampling work. Module created by Lucas Makinen in December 2018.

The Gibbs sampling method follows the work in the Appendix of Shariff et al (2016):
    https://arxiv.org/pdf/1510.05954.pdf

This method serves as a diagnostic for the larger BAHAMAS model in supernova cosmology. We want
to take a look at the posterior population-level distributions that our probabilistic model puts 
out to compare the effect of skewed color distributions on our inference. BAHAMAS currently 
propagates a symmetric gaussian in c*, so that we can marginalize latents out for computational
efficiency.

This sampling routine should give us a way to view these posterior distributions.