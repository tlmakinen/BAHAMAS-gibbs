# plot script for visualizing latent distributions in gibbs sampling routine

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from bahamas import cosmology

# Make a nifty plot to show latent vs observed SNIa attributes in a grid

def plot_attributes(D, param, iter):
    

    m_latent = D[2::3]
    c_latent = D[0::3]
    x_latent = D[1::3]

    
    latent_plot_labels = [r'$M$', r'$c$', r'$x_1$']
    latent_labels = [
                        [r'$m_*$', r'$\sigma_{res}$'],
                        [r'$c_*$', r'$r_c$'],
                        [r'$x_{1*}$', r'$r_x$']
    ]


    latent_pops = [m_latent, c_latent, x_latent]

    latent_means = [np.mean(m_latent), np.mean(c_latent), np.mean(x_latent)]
    latent_spreads = [np.std(m_latent), np.std(c_latent), np.std(x_latent)]

    # from current chain position
    latent_mean_chain = [param[7], param[5], param[6]]
    latent_spread_chain = [param[4], param[3], param[2]]

    fig,axs = plt.subplots(1, 3, figsize = (28,6))

    for i in range(len(axs)):
        # plot data attributes
        #axs[0][i].hist()

    # plot selected SNe
        axs[i].hist(latent_pops[i], bins=50, color='#9467bd', alpha=0.5)

        axs[i].set_xlabel(latent_plot_labels[i], fontsize=22)

    # for text labels
        histo = np.histogram(latent_pops[i], bins=50)
        x_pos = np.median(histo[1]) + 0.6*np.std(histo[1])
        y_pos = 0.41*np.max(histo[0])

        s = 'params computed \n from population \n'
        s += latent_labels[i][0] + ': {0:8.3f} '.format(latent_means[i]) + '\n'
        s += latent_labels[i][1] + ': {0:8.3f} '.format(latent_spreads[i])

        s += '\n' + 'chain values \n'
        s += latent_labels[i][0] + ': {0:8.3f} '.format(latent_mean_chain[i]) + '\n'
        s += latent_labels[i][1] + ': {0:8.3f} '.format(latent_spread_chain[i])

        axs[i].text(x_pos, y_pos, s, fontsize=15)
    # restrict xlims to observed SNe    
    #axs[i].set_xlim(xlims[i])    
    # add labels
    #axs[i].set_xlabel(latent_names[i], fontsize=25)
    #axs[i].set_ylabel('$p(S_i=1|$' + latent_names[i] + '$)$', fontsize=22)
    fig.suptitle('latent populations for iter {}'.format(iter), fontsize=22)
    plt.subplots_adjust(wspace=0.3, left=0.125)

#fig.suptitle('Selection Function Classifier on Test SNIa set', fontsize=17)
#axs[2].legend(loc='best', fontsize=11)
    plt.savefig(fname='latent_distr_comp.png', dpi='figure')


# make diagnostic plot for posterior mean for 50 SN1a
def plot_post_means(D_chain, datafname):

    
    m_means = [] # vector of 50 type 1a SNe mean values for M in chain
    c_means = []
    x_means = []
    
    m = []
    c = []
    x = []
    
    columns = D_chain.columns
    for i in range(len(columns)):
        if i % 3 == 0:
            m_means.append(np.mean(D_chain[columns[i+2]]))
            c_means.append(np.mean(D_chain[columns[i]]))
            x_means.append(np.mean(D_chain[columns[i+1]]))

    # compare to JLA-like sims -- take only every 10th snia
    data = pd.read_csv(datafname, sep='\s+', header=0)

    ndat = len(data)
    Zhel = data['z'].values
    zHD = data['z'].values
    # add second z column so that we have a zHD value
    data.insert(loc=1, column='zHD', value=zHD)
    Zcmb = data['z'].values

    # unpack data values
    mb_true = data['true_mb'].values
    c_true = data['true_c'].values
    x1_true = data['true_x1'].values

    # compute true residual
    a_true = 0.13
    b_true = 2.56
    cosmo_param = [0.3, 0.7, 0.72]              # true simulated params
    mu = cosmology.muz(cosmo_param, Zcmb, Zhel) # distance modulus
    M_true = []


    for i in range(ndat):
        M_true.append(mb_true[i] - mu[i] + a_true*x1_true[i] - b_true*c_true[i])
    

    # make 1 x 3 diagnostic scatter plot for M, c, x1
    fig,(ax0,ax1,ax2) = plt.subplots(1, 3, figsize = (28,6))

    # plot observed modulus vs M_true to show data scatter

    plt.suptitle('Diagnostic Plot for Posterior Means', fontsize=32)
    # plot M vs M_true
    ax0.scatter(np.array(m_means), np.array(M_true), 
                            color='r', marker = 'x', s=10)
    x = np.linspace(start=min(m_means)-0.2, stop = max(m_means)+0.2, num=50)
    y = np.linspace(start=min(m_means)-0.2, stop = max(m_means)+0.2, num=50)
    ax0.plot(x,y, linestyle='--', color='k', label='truth')
    ax0.set_xlabel('latent $M$', fontsize=28)
    ax0.set_ylabel('$M_{sim}$', fontsize=28)
    #ax0.legend(fontsize = 22, loc='upper left')
    # observed c vs simulated c
    ax1.scatter(data['c'].values, c_true, 
                    color = 'b', marker = 'x', s=10, label='observed $\hat{c}$')
    # latent c vs simulated c
    ax1.scatter(c_means, c_true, color = 'r', 
                    marker = 'x', s=10, label='latent $c$')
    x = np.linspace(start=min(c_means), stop=max(c_means), num=50)
    y = np.linspace(start=min(c_means), stop=max(c_means), num=50)
    ax1.plot(x,y, linestyle='--', color='k', label='truth')  # plot truth line
    ax1.set_xlabel('$c$', fontsize=28)
    ax1.set_ylabel('$c_{sim}$', fontsize=28)
    ax1.legend(fontsize = 22, loc='lower right')
    # x1
    ax2.scatter(data['x1'].values, x1_true, 
                    color = 'b', marker = 'x', s=10, label='observed $\hat{x}_1$')
    ax2.scatter(x_means, x1_true, 
                    color = 'r', marker = 'x', s=10, label='latent $x_1$')

    x = np.linspace(start=min(x_means)-0.5, stop=max(x_means)+0.5, num=50)
    y = np.linspace(start=min(x_means)-0.5, stop=max(x_means)+0.5, num=50)
    ax2.plot(x,y, linestyle='--', color='k', label='truth')  # plot truth line
    ax2.set_xlabel('$x_1$', fontsize=28)
    ax2.set_ylabel('$x_{1,sim}$', fontsize=28)
    ax2.legend(fontsize = 22, loc='lower right')
    
    #plt.show()

    plt.savefig(fname='posterior_diagnostic.png', dpi='figure')


