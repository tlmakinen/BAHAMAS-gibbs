# plot script for visualizing latent distributions in gibbs sampling routine

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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

    # data for comparison
    #data = pd.read_csv(datafname, sep='\s+', header=0)

    #data_pops = [data['mB'].values, data['c'].values, data['x1'].values]
    #data_spreads = [np.std(data['mB'].values)]

    #data_labels = [' $\hat{m_B}$', '$\hat{c}$', '$\hat{x_1}$']
#xlims = [(15, 27), (-0.8, 0.8), (-3.5, 3.5)]


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
