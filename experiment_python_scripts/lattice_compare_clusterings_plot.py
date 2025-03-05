# Used to create Figure 3

import pickle
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['toolbar'] = 'None'

def draw_plots(data, save_name, b, beta):
    df = pd.DataFrame(data)
    df = df[df['beta'] == beta]

    df['mse'] = df['bias']**2 + df['var']

    colors = ["tab:blue","tab:orange","tab:green"]

    f,ax = plt.subplots(1,3, sharex=True, sharey=True)
    f.set_figheight(4)
    f.set_figwidth(12)

    plt.xlim(min(df['q']), 1)
    plt.setp(ax, ylim=(0,1))

    ax[1].yaxis.set_tick_params(which='both', labelleft=True)
    ax[2].yaxis.set_tick_params(which='both', labelleft=True)

    ax[0].set_title("Coarse (10 by 10)", fontsize=16)
    ax[1].set_title("Fine (2 by 2)", fontsize=16)
    ax[2].set_title("No Clustering", fontsize=16)
    clusterings = ["coarse","fine","none"]

    for i,cl in enumerate(clusterings):
        axis = ax[i]
        cell_df = df[df['cl'] == cl] 

        axis.plot(cell_df['q'], cell_df['bias']**2 + cell_df['var'], color='k', linewidth=2, label="MSE")
        
        axis.fill_between(cell_df['q'], cell_df['bias']**2, cell_df['bias']**2 + cell_df['var'], color=colors[2], hatch='\\\\', alpha=0.25,label="Variance")

        axis.plot(cell_df['q'], cell_df['bias']**2, color=colors[0],alpha=0.5)
        axis.fill_between(cell_df['q'], 0, cell_df['bias']**2, color=colors[0], hatch='++', alpha=0.25, label="Bias$^2$")

    for axis in ax:
        axis.set_xlabel("q", fontsize=14)

    f.suptitle("Lattice Network, $\\beta=${}, b={}".format(beta, b), fontsize=20)
    f.subplots_adjust(bottom=0.25)
    ax[0].legend(prop={'size': 12})
    plt.tight_layout()
    plt.show()
    f.savefig(save_name,bbox_inches='tight')

if __name__ == '__main__':
    data_file = open("Lattice/compare_clusterings.pkl", 'rb')
    save_name = "compare_clusterings_Lattice.png"
    data = pickle.load(data_file)
    data_file.close()

    b = 0.5
    beta = 3
    
    draw_plots(data, save_name, b, beta)