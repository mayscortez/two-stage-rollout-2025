# Creates a file named lattice_cluster_metrics.txt containing different clustering metrics for the Lattice network

import numpy as np
import pandas as pd
import pickle
from experiment_python_scripts.experiment_functions import clustering_metrics, homophily_effects, pom_ugander_yin, lattice2Dsq,bf_clusters

betas = [2,3]
network = "Lattice"
experiment = 'compare_clusterings' # if using the files with some homophily (b=0.5) use 'compare_clusterings_b05'

def create_table(betas, network, experiment, save_file=False):
    if save_file:
        f = open('cluster_metrics_lattice.txt', 'w')

    data_file = open(network + '/' + experiment + '.pkl', 'rb')
    df = pd.DataFrame(pickle.load(data_file))
    df["mse"] = df["bias"]**2 + df["var"]

    n = 10000
    lat_n = int(np.sqrt(n))
    A = lattice2Dsq(lat_n,lat_n)#.toarray()
    G = [A[[i],:].nonzero()[1] for i in range(n)] #adjacency list
    h = homophily_effects(A)

    # Fine Clustering
    fine = 2
    c_num = n//(fine**2)
    _, cluster_flat = bf_clusters(c_num, n)
    Cl_fine = [list(np.where(cluster_flat==i)[0]) for i in range(c_num)]

    # compute number of cut edges for fine clustering
    fine_cuts = 0
    for i,edge_list in enumerate(G):
        for j in edge_list:
            if cluster_flat[i] != cluster_flat[j]:
                fine_cuts = fine_cuts + 1

    # Coarse Clustering
    coarse = 10
    c_num = n//(coarse**2)
    _, cluster_flat = bf_clusters(c_num, n)
    Cl_coarse = [list(np.where(cluster_flat==i)[0]) for i in range(c_num)]

    # compute number of cut edges for coarse clustering
    coarse_cuts = 0
    for i,edge_list in enumerate(G):
        for j in edge_list:
            if cluster_flat[i] != cluster_flat[j]:
                coarse_cuts = coarse_cuts + 1
    print() 
    
    for beta in betas:
        fY = pom_ugander_yin(A,h,beta)
        
        var_cluster_avgs, effect_of_cut_edges = clustering_metrics(n, Cl_fine, fY)
        
        print("{} Network, beta = {}, 'fine' clustering\nVar: {}, cut effect: {}, cuts: {}".format(network,beta,var_cluster_avgs, effect_of_cut_edges, fine_cuts))
        if save_file:
            print("{} Network, beta = {}, 'fine' clustering\nVar: {}, cut effect: {}, cuts: {}".format(network,beta,var_cluster_avgs, effect_of_cut_edges, fine_cuts),file=f)

        new_df = df.loc[(df["beta"]==beta) & (df["cl"]=='fine')]
        min_mse_row = new_df.loc[new_df['mse'].idxmin()]  # Get the row with the smallest MSE
        q_value = min_mse_row['q']
        mse_value = min_mse_row['mse']

        print("q_min = {}, MSE(q_min) = {}".format(q_value, mse_value))
        if save_file:
            print("q_min = {}, MSE(q_min) = {}".format(q_value, mse_value), file=f)

        var_cluster_avgs, effect_of_cut_edges = clustering_metrics(n, Cl_coarse, fY)

        print()
        print("{} Network, beta={}, 'coarse' clustering\nVar: {}, cut effect: {}, cuts: {}".format(network,beta,var_cluster_avgs, effect_of_cut_edges, coarse_cuts))
        if save_file:
            print("{} Network, beta={}, 'coarse' clustering\nVar: {}, cut effect: {}, cuts: {}".format(network,beta,var_cluster_avgs, effect_of_cut_edges, coarse_cuts),file=f)

        new_df = df.loc[(df["beta"]==beta) & (df["cl"]=='coarse')]
        min_mse_row = new_df.loc[new_df['mse'].idxmin()]  # Get the row with the smallest MSE
        q_value = min_mse_row['q']
        mse_value = min_mse_row['mse']

        print("q_min = {}, MSE(q_min) = {}\n".format(q_value, mse_value))
        if save_file:
            print("q_min = {}, MSE(q_min) = {}\n".format(q_value, mse_value), file=f)

    if save_file:
        print("##########################################################################",file=f)


if __name__ == '__main__':
    betas = [2,3]
    network = "Lattice"
    experiment = 'compare_clusterings' # if using the files with some homophily (b=0.5) use 'compare_clusterings_b05'
    create_table(betas, network, experiment, save_file=True)