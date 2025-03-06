# Used to create data for Figures 4, 11, 12, and 14 (comparing 2-stage performance under different clusterings for real-world networks)
import json
import pickle
import pymetis

from experiment_python_scripts.experiment_functions import *
from itertools import product
from joblib import Parallel, delayed 

def pi_estimate_tte_two_stage_complete(Y,nu,k,beta):
    '''
    Estimates the treatment effect (TTE) using polynomial interpolation.
    
    Parameters:
        Y: ndarray (beta+1 x r x n)
            Potential outcomes under different treatment levels.
        nu: int
            Expected size of U (subset of units).
        k: int
            Budget (in units).
        beta: int
            Degree of polynomial interpolation.

    Returns:
        Estimated TTE value.
    '''
    H = complete_coeffs(beta,nu,k) # Compute interpolation coefficients
    return 1/nu * H @ np.sum(Y,axis=-1) # Compute estimate using polynomial interpolation

def estimate_two_stage(fY,Cl,n,nu,r,beta,K):
    '''
    Runs a two-stage experimental design to estimate treatment effects.
    
    Parameters:
    fY    : function, generates potential outcomes given treatment assignments
    Cl    : graph clustering (list of lists)
    n     : int, total number of units (nodes)
    nu    : int, expected size of treated group
    r     : int, number of replications
    beta  : int, polynomial interpolation degree
    K     : array, treatment budget allocation
    
    Returns:
    tuple containing (nu, TTE estimates, expected TTE given U)
    '''

    # Generate treatment assignments using a staggered rollout design
    Z,U = complete_staggered_rollout_two_stage(n, Cl, nu, K, r) 
    Y = fY(Z)

    tte_hat = []
    # Compute treatment effect estimate using polynomial interpolation
    tte_hat = np.append(tte_hat,pi_estimate_tte_two_stage_complete(Y,nu,K[-1],beta))
    
    e_tte_hat_given_u = []
    # Compute conditional expectations
    e_tte_hat_given_u = np.append(e_tte_hat_given_u, (1/nu)*np.sum(fY(U) - fY(np.zeros(n)),axis=1))

    return (nu, tte_hat, e_tte_hat_given_u)

def run_experiment(G,Cls,fixed,varied,r):
    '''
    Runs a full experiment over varying parameters and collects results.
    
    Parameters:
    G      : graph adjacency matrix (scipy sparse matrix)
    Cls    :  graph clustering (list of lists)
    fixed  : dictionary of fixed parameters
    varied : dictionary of varied parameters
    r      : int, number of replications
    
    Returns:
    Dictionary containing experiment results
    '''

    n = G.shape[0]
    print("n: {}".format(n))
    h = homophily_effects(G)

    # Convert adjacency matrix to adjacency list
    A = [[] for _ in range(n)]
    for i,j in zip(*G.nonzero()):
        A[i].append(j)
        A[j].append(i)

    # Extract parameters to iterate over
    betas = [fixed["beta"]] if "beta" in fixed else varied["beta"] 
    ncs = [fixed["nc"]] if "nc" in fixed else varied["nc"]
    ps = [fixed["p"]] if "p" in fixed else varied["p"]

    # Initialize result storage
    data = { "q":[], "nu":[], "clustering":[], "bias":[], "var":[], "var_s":[], "var_Lc":[], "cut_effect":[]}
    if "beta" in varied: data["beta"] = []
    if "nc" in varied: data["nc"] = []
    if "p" in varied: data["p"] = []

    cluster_dict = {nc:{} for nc in ncs}

    for nc in ncs:
        print("Preparing Clusterings with {} Clusters".format(nc))

        # Use given clustering
        cluster_dict[nc]["feature"] = Cls[nc]

        # Generate clustering based on graph structure using Metis
        _,membership = pymetis.part_graph(nparts=nc,adjacency=A)
        membership = np.array(membership)
        Cl_graph = []
        for i in range(nc):
            Cl_graph.append(np.where(membership == i)[0])

        cluster_dict[nc]["graph"] = Cl_graph

        cluster_dict[nc]["none"] = [] # Placeholder for unclustered case

    for beta in betas:
        print("\nbeta = {}".format(beta))

        fY = pom_ugander_yin(G,h,beta)
        TTE = np.sum(fY(np.ones(n))-fY(np.zeros(n)))/n

        for nc,p in product(ncs,ps):
            K = np.linspace(0,int(n*p),beta+1,dtype=int)
            for label,Cl in cluster_dict[nc].items():
                print("Clustering: {}".format(label))
                # Run two-stage estimation for multiple values of nu (expected treated size)
                for (nu,TTE_hat,E_given_U) in Parallel(n_jobs=-2, verbose=10)(delayed(lambda nu : estimate_two_stage(fY,Cl,n,nu,r,beta,K))(nu) for nu in np.linspace(int(n*p), n, 10, dtype=int)):
                    data["nu"].append(nu)
                    data["q"].append((n*p)/nu)
                    data["clustering"].append(label)
                    if "beta" in varied: data["beta"].append(beta)
                    if "nc" in varied: data["nc"].append(nc)
                    if "p" in varied: data["p"].append(p)

                    # Compute bias and variance of estimates
                    data["bias"].append(np.mean(TTE_hat) - TTE)
                    data["var"].append(np.var(TTE_hat))
                    data["var_s"].append(np.var(E_given_U))

                    # Compute clustering metrics
                    var_cluster_avgs, effect_of_cut_edges = clustering_metrics(n, Cl, fY)
                    data["var_Lc"].append(var_cluster_avgs)
                    data["cut_effect"].append(effect_of_cut_edges)

    return data

if __name__ == '__main__':
    my_path = "Amazon/Experiments/compare_clusterings.json" # path to .json file for the experiment
    jf = open(my_path,'rb')
    j = json.load(jf)
    jf.close()

    exp_name = j["name"]
    network_folder = j["network"]
    in_file = j["input"]

    print("Loading Graph")

    nf = open(network_folder + "/" + in_file,'rb')
    G,Cls = pickle.load(nf)
    nf.close()

    fixed = j["fix"]
    varied = j["vary"]
    r = j["replications"]

    data = run_experiment(G,Cls,fixed,varied,r)

    out_file = network_folder + "/Experiments/" + exp_name + ".pkl"
    print(f"Writing output to {out_file}")
    of = open(out_file,'wb')
    pickle.dump(data,of)
    of.close()