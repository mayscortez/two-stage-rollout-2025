import sys
import os
#sys.path.insert(0, "../")
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from experiment_functions import *
from joblib import Parallel, delayed 
import pickle
import pymetis
import time
import itertools
import networkx

startTime = time.time()
n = 10000    # perfect square, e.g. 100*100=10000

# parameters
betas = [2,3]             # model degrees
p = 0.125                   # treatment budget

nus = np.linspace(int(n*p), n, 15, dtype=int)
qs = np.divide((n*p),nus)

r = 1000                  # number of replications

stage1_designs = ['complete'] # ['bernoulli','complete'] 
cluster_strategies = ["none","fine","coarse"] #['none', 'fine' ,'coarse'] ["none","graph","fine","coarse","random"] 

if (("graph" in cluster_strategies) or ("random" in cluster_strategies)):
    k = 50      # number of communities, should divide n evenly
    c = n//k    # community size 

if "coarse" in cluster_strategies:
    # defines the coarseness of the overlaying grid; e.g. coarse = 10 means grids of 10 by 10 (which means 100 people per cluster)
    coarse = 10  #10; square should divide n evenly, e.g. if n=1600, coarse**2 should divide 1600

if "fine" in cluster_strategies:
    # defines the coarseness of the overlaying grid; e.g. fine = 2 means grids of 2 by 2 (which means 4 people per cluster)
    fine = 2 #2, 5

##############################################

data = {"beta":[], "q":[], "cl":[], "s1_design":[], "bias":[], "var":[], "var_s":[]}

def pi_estimate_tte_two_stage_complete(Y,nu,k):
    '''
    Returns TTE estimate from polynomial interpolation
        Y = potential outcomes: (beta+1) x r x n
        poq = the (expected size) of U
        Q = treatment probabilities of selected units for each time step: beta+1
    '''
    n = Y.shape[-1]
    H = complete_coeffs(beta,nu,k)
    return 1/nu * H @ np.sum(Y,axis=-1)

def estimate_two_stage(fY,Cl,r,nu,K,cl_design):
    tte_hat = []
    e_tte_hat_given_u = []

    Z,U = complete_staggered_rollout_two_stage(n, Cl, nu, K, r) 
    Y = fY(Z)
    tte_hat = np.append(tte_hat,pi_estimate_tte_two_stage_complete(Y,nu,int(n*p)))
    #print(Z.shape, Y.shape)
    e_tte_hat_given_u = np.append(e_tte_hat_given_u, (1/nu)*np.sum(fY(U) - fY(np.zeros(n)),axis=1))

    return (nu, tte_hat, e_tte_hat_given_u)

# each of these is a dict of dicts of dicts of lists... 
# the outermost dictionary has keys corresponding to the model degrees (betas)
# the value corresponding to each beta is itself a dictionary with keys corresponding to clustering type ["none","graph","community","random","anti"]
# the value corresponding to each clustering type is a dictionary with keys corresponding to the cluster selection design ["bernoulli", "complete"]
# the value corresponding to each design is a dictionary with keys corresponding to the q values in qs
# the value corresponding to each q value is an empty list (to be filled later)
# e.g. Bias_dict[b][c][d][q] constains a list of the biases of the two-stage estimator under a model with degree b, a clustering c, and treatment probability q where clusters are chosen in the first stage with design d
TTE_hat_dict = {b: {c: {d: {q:[] for q in qs} for d in stage1_designs} for c in cluster_strategies} for b in betas}
Bias_dict = {b: {c: {d: {q:[] for q in qs} for d in stage1_designs} for c in cluster_strategies} for b in betas}
E_given_U_dict = {b: {c: {d: {q:[] for q in qs} for d in stage1_designs} for c in cluster_strategies} for b in betas}


# Construct the network
lat_n = int(np.sqrt(n))
A = lattice2Dsq(lat_n,lat_n)#.toarray()
G = [A[[i],:].nonzero()[1] for i in range(n)] #adjacency list
h = homophily_effects(A)
clusterings = {"none":[]} 

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
    c_num = n//(fine**2)
    _, cluster_flat = bf_clusters(c_num, n)
    Cl_fine = [list(np.where(cluster_flat==i)[0]) for i in range(c_num)]
    clusterings["fine"] = Cl_fine

if "coarse" in cluster_strategies:
    c_num = n//(coarse**2)
    _, cluster_flat = bf_clusters(c_num, n)
    Cl_coarse = [list(np.where(cluster_flat==i)[0]) for i in range(c_num)]
    clusterings["coarse"] = Cl_coarse

for beta in betas:
    print("\nbeta={}".format(beta))

    K = np.linspace(0,int(n*p),beta+1,dtype=int)
    fY = pom_ugander_yin(A,h,beta)
    TTE_true = np.sum(fY(np.ones(n))-fY(np.zeros(n)))/n

    # run the experiment (RCT + estimation)
    for cl, d in itertools.product(cluster_strategies,stage1_designs):
        print("\nclustering: {}, design={}\n".format(cl,d))
        for (nu,TTE_hat,E_given_U) in Parallel(n_jobs=-2, verbose=5)(delayed(lambda nu : estimate_two_stage(fY,clusterings[cl],r,nu,K,d))(nu) for nu in nus):
            #print(TTE_true, TTE_hat,'\n')
            q = (n*p)/nu
            Bias_dict[beta][cl][d][q].append(TTE_hat - TTE_true)
            TTE_hat_dict[beta][cl][d][q].append(TTE_hat)
            E_given_U_dict[beta][cl][d][q].append(E_given_U)
            var_cluster_avgs, effect_of_cut_edges = clustering_metrics(n, clusterings[cl], fY)

# save the data (?)
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

file = open("compare_clusterings.pkl", "wb") #sbm_clustering
pickle.dump((data), file)
file.close()

executionTime = (time.time() - startTime)
print('Total runtime in minutes: {}'.format(executionTime/60)) 
    
        