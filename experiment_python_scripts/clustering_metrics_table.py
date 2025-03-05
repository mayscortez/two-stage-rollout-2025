# Creates a file named cluster_metrics.txt containing different clustering metrics for the real-world networks

import numpy as np
import pandas as pd
import pickle
import pymetis
from experiment_python_scripts.experiment_functions import clustering_metrics, homophily_effects, pom_ugander_yin

def create_tables(networks, ncs, betas, experiment, save_file=False):
    if save_file:
        f = open('cluster_metrics_real-world.txt', 'w')
    
    for ind,network in enumerate(networks):
        data_file = open(network + '/Experiments/' + experiment + '.pkl', 'rb')
        df = pd.DataFrame(pickle.load(data_file))
        df["mse"] = df["bias"]**2 + df["var"]
        if network=="Email":
            df["nc"] = 42
        
        nf = open(network + '/Network/data.pkl','rb') # 'Neurips_Experiments/' +
        G,Cls = pickle.load(nf)
        nf.close()
        n = G.shape[0]
        h = homophily_effects(G)

        A = [[] for _ in range(n)]
        for i,j in zip(*G.nonzero()):
            A[i].append(j)
            A[j].append(i)

        # Graph Clustering
        cuts,membership = pymetis.part_graph(nparts=ncs[ind],adjacency=A)
        membership = np.array(membership)
        Cl_graph = []
        for i in range(ncs[ind]):
            Cl_graph.append(np.where(membership == i)[0])

        # Covariate Clustering
        memberships = np.zeros(n)
        for i,cluster in enumerate(Cls[ncs[ind]]):
            memberships[cluster] = i
        
        count = 0
        for i,edge_list in enumerate(A):
            for j in edge_list:
                if memberships[i] != memberships[j]:
                    count = count + 1
        
        for beta in betas:
            fY = pom_ugander_yin(G,h,beta)
            
            var_cluster_avgs, effect_of_cut_edges = clustering_metrics(n, Cl_graph, fY)
            print("{} Network, beta = {}, 'graph' clustering with {} clusters\nVar: {}, cut effect: {}, cuts: {}".format(network,beta,ncs[ind],var_cluster_avgs, effect_of_cut_edges, cuts))
            if save_file:
                print("{} Network, beta = {}, 'graph' clustering\nVar: {}, cut effect: {}, cuts: {}".format(network,beta,var_cluster_avgs, effect_of_cut_edges, cuts),file=f)

            new_df = df.loc[(df["beta"]==beta) & (df["clustering"]=='graph') & (df["nc"]==ncs[ind])]
            min_mse_row = new_df.loc[new_df['mse'].idxmin()]  # Get the row with the smallest MSE
            q_value = min_mse_row['q']
            mse_value = min_mse_row['mse']
            print("q_min = {}, MSE(q_min) = {}".format(q_value, mse_value))
            if save_file:
                print("q_min = {}, MSE(q_min) = {}".format(q_value, mse_value), file=f)

            var_cluster_avgs, effect_of_cut_edges = clustering_metrics(n, Cls[ncs[ind]], fY)
            print()
            print("{} Network, beta={}, 'covariate' clustering with {} clusters\nVar: {}, cut effect: {}, cuts: {}".format(network,beta,ncs[ind],var_cluster_avgs, effect_of_cut_edges, count//2))
            if save_file:
                print("{} Network, beta={}, 'covariate' clustering\nVar: {}, cut effect: {}, cuts: {}".format(network,beta,var_cluster_avgs, effect_of_cut_edges, count//2),file=f)

            new_df = df.loc[(df["beta"]==beta) & (df["clustering"]=='feature') & (df["nc"]==ncs[ind])]
            min_mse_row = new_df.loc[new_df['mse'].idxmin()]  # Get the row with the smallest MSE
            q_value = min_mse_row['q']
            mse_value = min_mse_row['mse']
            print("q_min = {}, MSE(q_min) = {}\n".format(q_value, mse_value))
            if save_file:
                print("q_min = {}, MSE(q_min) = {}\n".format(q_value, mse_value), file=f)
        
        if save_file:
            print("##########################################################################",file=f)


if __name__ == '__main__':
    networks = ['Amazon', 'BlogCatalog', 'Email']
    ncs = [250, 50, 42]
    betas = [2,3]
    experiment = 'compare_clusterings' # if using the files with some homophily (b=0.5) use 'compare_clusterings_b05'
    create_tables(networks, ncs, betas, experiment,save_file=True)