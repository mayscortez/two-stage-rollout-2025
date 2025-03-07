# Runs the compare clusterings experiments for the Lattice network

import sys
import os
#sys.path.insert(0, "../")
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from experiment_python_scripts.experiment_functions import *
from joblib import Parallel, delayed 
import pickle
import pymetis
import time
import itertools
import networkx

startTime = time.time()
n = 10000  # Number of units, should be a perfect square for grid-based clustering

# Parameters
betas = [2, 3]  # Polynomial degrees for outcome model
p = 0.125  # Treatment budget proportion

# Define values of nu (subset sizes) and q (treatment probabilities)
nus = np.linspace(int(n * p), n, 15, dtype=int)
qs = np.divide((n * p), nus)

r = 1000  # Number of replications

# Stage 1 assignment designs and clustering strategies
stage1_designs = ['complete']  # Available assignment designs for stage 1; other options are ['bernoulli','complete'] or ['bernoulli]
cluster_strategies = ["none", "fine", "coarse"]  # Different clustering strategies for treatment assignment; all possible strategies are ["none","graph","fine","coarse","random"] 

if (("graph" in cluster_strategies) or ("random" in cluster_strategies)):
    k = 50      # number of communities, should divide n evenly
    c = n//k    # community size 

if "coarse" in cluster_strategies:
    # Defines cluster size for coarse clustering (e.g., 10x10 grids)
    coarse = 10  #square should divide n evenly, e.g. if n=1600, coarse**2 should divide 1600

if "fine" in cluster_strategies:
    # Defines cluster size for fine clustering (e.g., 2x2 grids)
    fine = 2

##############################################

# Initialize dictionary to store results
data = {"beta":[], "q":[], "cl":[], "s1_design":[], "bias":[], "var":[], "var_s":[]}

def pi_estimate_tte_two_stage_complete(Y, nu, k):
    '''
    Computes TTE estimate from polynomial interpolation.
    :param Y: Potential outcomes, shape (beta+1, r, n)
    :param nu: Expected size of subset U
    :param k: Treatment budget (in units)
    :return: Estimated treatment effect
    '''
    n = Y.shape[-1]
    H = complete_coeffs(beta, nu, k)  # Compute polynomial coefficients
    return 1 / nu * H @ np.sum(Y, axis=-1)

# Function to estimate TTE in a two-stage experiment
def estimate_two_stage(fY,Cl,r,nu,K,cl_design):
    tte_hat = []
    e_tte_hat_given_u = []

    # Generate treatment and control assignments, compute potential outcomes
    Z,U = complete_staggered_rollout_two_stage(n, Cl, nu, K, r) 
    Y = fY(Z)

    # Compute TTE estimates
    tte_hat = np.append(tte_hat,pi_estimate_tte_two_stage_complete(Y,nu,int(n*p)))

    # Compute conditional expectations
    e_tte_hat_given_u = np.append(e_tte_hat_given_u, (1/nu)*np.sum(fY(U) - fY(np.zeros(n)),axis=1))

    return (nu, tte_hat, e_tte_hat_given_u)

# Initialize nested dictionaries for storing results
# e.g. Bias_dict[b][c][d][q] constains a list of the biases of the two-stage estimator under a model with degree b, a clustering c, and treatment probability q where clusters are chosen in the first stage with design d
TTE_hat_dict = {b: {c: {d: {q:[] for q in qs} for d in stage1_designs} for c in cluster_strategies} for b in betas}
Bias_dict = {b: {c: {d: {q:[] for q in qs} for d in stage1_designs} for c in cluster_strategies} for b in betas}
E_given_U_dict = {b: {c: {d: {q:[] for q in qs} for d in stage1_designs} for c in cluster_strategies} for b in betas}

# Construct the network
lat_n = int(np.sqrt(n))
A = lattice2Dsq(lat_n,lat_n)#.toarray() # adjacency matrix
G = [A[[i],:].nonzero()[1] for i in range(n)] # adjacency list
h = homophily_effects(A)
clusterings = {"none":[]} # Initialize clustering dictionary

if "graph" in cluster_strategies:
    # create a clustering based on the graph
    _,membership = pymetis.part_graph(nparts=k,adjacency=G)
    membership = np.array(membership)
    Cl_graph = []
    for i in range(k):
        Cl_graph.append(np.where(membership == i)[0])

    clusterings["graph"] = Cl_graph
    #print(Cl_graph)

if "random" in cluster_strategies:
    # randomly chosen balanced clustering
    membership = np.array(list(range(k))*(c+1))[:n]
    np.random.shuffle(membership)

    Cl_random = []
    for i in range(k):
        Cl_random.append(np.where(membership == i)[0])

    clusterings["random"] = Cl_random

if "fine" in cluster_strategies:
    # fine grid clustering
    c_num = n//(fine**2)
    _, cluster_flat = bf_clusters(c_num, n)
    Cl_fine = [list(np.where(cluster_flat==i)[0]) for i in range(c_num)]
    clusterings["fine"] = Cl_fine

if "coarse" in cluster_strategies:
    # coarse grid clustering
    c_num = n//(coarse**2)
    _, cluster_flat = bf_clusters(c_num, n)
    Cl_coarse = [list(np.where(cluster_flat==i)[0]) for i in range(c_num)]
    clusterings["coarse"] = Cl_coarse

# Run experiments for different beta values
for beta in betas:
    print("\nbeta={}".format(beta))

    K = np.linspace(0,int(n*p),beta+1,dtype=int)
    fY = pom_ugander_yin(A,h,beta)
    TTE_true = np.sum(fY(np.ones(n))-fY(np.zeros(n)))/n

    # run the experiment (RCT + estimation), iterate over clustering strategies and designs
    for cl, d in itertools.product(cluster_strategies,stage1_designs):
        print("\nclustering: {}, design={}\n".format(cl,d))

        for (nu,TTE_hat,E_given_U) in Parallel(n_jobs=-2, verbose=5)(delayed(lambda nu : estimate_two_stage(fY,clusterings[cl],r,nu,K,d))(nu) for nu in nus):
            q = (n*p)/nu
            Bias_dict[beta][cl][d][q].append(TTE_hat - TTE_true)
            TTE_hat_dict[beta][cl][d][q].append(TTE_hat)
            E_given_U_dict[beta][cl][d][q].append(E_given_U)
            var_cluster_avgs, effect_of_cut_edges = clustering_metrics(n, clusterings[cl], fY)

# save the data
for beta in betas:
    for cl, d in itertools.product(cluster_strategies,stage1_designs):
        for nu in nus:
            q = (n*p)/nu
            data["q"].append(q)
            data["cl"].append(cl)
            data["s1_design"].append(d)
            data["beta"].append(beta)
            data["bias"].append(np.average(Bias_dict[beta][cl][d][q]))
            data["var"].append(np.var(TTE_hat_dict[beta][cl][d][q]))
            data["var_s"].append(np.var(E_given_U_dict[beta][cl][d][q]))

file = open("compare_clusterings.pkl", "wb")
pickle.dump((data), file)
file.close()

executionTime = (time.time() - startTime)
print('Total runtime in minutes: {}'.format(executionTime/60)) 
    
        