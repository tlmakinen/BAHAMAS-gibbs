import numpy as np
import bahamas.vanilla_log_likelihood as vanilla
import bahamas.selection_effects as selection


def vanilla_log_likelihood(J, sigmaCinv, log_sigmaCinv, param, data, ndat):
    cosmo_param = param[8:11]
    param = param[:8]
    
    Zcmb, Zhel = data.T[0:2]
    
    mu = cosmology.muz(cosmo_param, Zcmb, Zhel)
    if np.any(np.isnan(mu)): # quit if hubble integral is not integrable
        return -np.inf
    
    return vanilla.log_likelihood(J, sigmaCinv, log_sigmaCinv, param, cosmo_param, data, mu, ndat)

def rubin_log_likelihood(J, sigmaCinv, log_sigmaCinv, param, data, ndat):
    selection_param = param[8:12]
    cosmo_param = param[12:15]
    param = param[:8]
    
    Zcmb, Zhel = data.T[0:2]
    phi = data[:, 2:5]
    
    mu = cosmology.muz(cosmo_param, Zcmb, Zhel)
    if np.any(np.isnan(mu)): # quit if hubble integral is not integrable
        return -np.inf
    
    return (vanilla.log_likelihood(J, sigmaCinv, log_sigmaCinv, param, cosmo_param, data, mu, ndat) 
            + selection.rubin_log_correction(param, selection_param, phi, mu))

def vincent_log_likelihood(J, sigmaCinv, log_sigmaCinv, param, data, ndat):
    selection_param = param[8:12]
    cosmo_param = param[12:15]
    param = param[:8]
    
    #Zcmb, Zhel = data.T[0:2]

    Zcmb, Zhel = data[0:2].T

    phi = data.iloc[:, 2:5]
    
    mu = cosmology.muz(cosmo_param, Zcmb, Zhel)
    if np.any(np.isnan(mu)): # quit if hubble integral is not integrable
        return -np.inf
    
    return (vanilla.log_likelihood(J, sigmaCinv, log_sigmaCinv, param, cosmo_param, data, mu, ndat) 
            + selection.vincent_log_correction(param, selection_param, cosmo_param, phi, mu, ndat))
