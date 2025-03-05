# Used to create Figures 4, 11, 12, and 14 (comparing 2-stage performance under different clusterings for real-world networks)
import pickle
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['toolbar'] = 'None'

def draw_plots(network_name, save_name, data, beta):
    df = pd.DataFrame(data)
    df = df[df['beta'] == beta]
    if network_name == "Amazon":
        df = df[df['nc'] == 250] 
    if network_name == 'BlogCatalog':
        df = df[df['nc'] == 50] 

    df['mse'] = df['bias']**2 + df['var']

    colors = ["tab:blue","tab:orange","tab:green"]

    f,ax = plt.subplots(1,3, sharex=True, sharey=True)
    f.set_figheight(4)
    f.set_figwidth(12)

    plt.xlim(min(df['q']), 1)
    plt.setp(ax, ylim=(0,1))

    ax[1].yaxis.set_tick_params(which='both', labelleft=True)
    ax[2].yaxis.set_tick_params(which='both', labelleft=True)

    ax[0].set_title("Full Graph Knowledge", fontsize=16)
    ax[1].set_title("Covariate Knowledge", fontsize=16)
    ax[2].set_title("No Clustering", fontsize=16)
    clusterings = ["graph","feature","none"]

    for i,cl in enumerate(clusterings):
        axis = ax[i]
        cell_df = df[df['clustering'] == cl]

        axis.plot(cell_df['q'], cell_df['bias']**2 + cell_df['var'], color='k', linewidth=2, label="MSE")
        
        axis.fill_between(cell_df['q'], cell_df['bias']**2, cell_df['bias']**2 + cell_df['var'], color=colors[2], hatch='\\\\', alpha=0.25,label="Variance")

        axis.plot(cell_df['q'], cell_df['bias']**2, color=colors[0],alpha=0.5)
        axis.fill_between(cell_df['q'], 0, cell_df['bias']**2, color=colors[0], hatch='++', alpha=0.25, label="Bias$^2$")

    for axis in ax:
        axis.set_xlabel("q", fontsize=14)

    f.suptitle("{} Network, $\\beta=${}".format(network_name, beta), fontsize=20)
    f.subplots_adjust(bottom=0.25)
    ax[0].legend(prop={'size': 12})
    plt.tight_layout()
    plt.show()
    f.savefig(save_name,bbox_inches='tight')

if __name__ == '__main__':
    network_name = "Email"
    
    data_file = open(network_name + "/Experiments/compare_clusterings.pkl", 'rb')
    save_name = "compare_clusterings_" + network_name + ".png"
    data = pickle.load(data_file)
    data_file.close()
    
    beta = 2
    
    draw_plots(network_name, save_name, data, beta)