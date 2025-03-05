import numpy as np
from numpy.random import RandomState
from scipy.special import binom
import scipy
import itertools
import networkx as nx

rng = RandomState(19025)
rng_new = np.random.default_rng(19025)

######## Constructed Networks ########

def lattice2Dsq(x=2,y=2):
    '''
    Returns adjacency matrix of an x by y lattice graph on x*y nodes as a sparse matrix
    
    x (int): number of nodes in the x direction
    y (int): number of nodes in the y direction
    '''
    G = nx.grid_graph(dim=(x,y))
    G = nx.DiGraph(G)
    A = nx.to_scipy_sparse_array(G)
    A.setdiag(np.ones(x*y))
    return A 

######## Lattice Clustering ##############
def lat_toCluster(I,J,k,q1=0,q2=0,divides = False):
    '''
    Thinking of a clustering as a coarser grid on top of the lattice, returns the cluster assignment (s,t) of unit(s) (i,j) for i in I and j in J
    population size: n*n
    number of clusters: nc*nc

    Parameters
    -----------
    i : int or np.array
        row position of unit on n by n lattice (or array of row positions)
    j : int or np.array
        column position of unit on n by n lattice (or array of col positions)
    k : int
        typical cluster side length (each cluster is itself a k by k grid graph with k << n)
    q1 : int
        "origin" row position marking the END (inclusive) of first cluster
    q2 : int
        "origin" col position marking the END (inclusive) of first cluster
    divides : bool
        if k divides n, set to True

    Returns
    -----------
    s : int
        row position of the cluster on nc by nc lattice (or array of row positions)
    t : int
        column position of the cluster on nc by nc lattice (or array of col positions)
    '''
    if divides:
        s = np.floor(I/k)
        t = np.floor(J/k)
    else:
        s = np.ceil((I-q1)/k)
        t = np.ceil((J-q2)/k)

    return s.astype(int),t.astype(int)

def bf_clusters(num, pop):
    '''
    Returns cluster assignments for a lattice where num is the number of clusters and pop is the size of the population

    Parameters
    ----------
    num (int)
        number of clusters (should be a perfect square nc*nc)
    pop (int)
        size of the population (should be a perfect square n*n)

    Returns
    -------
    clusters
        cluster assignments for each person
    
    clusters.flatten() (numpy array of size 1 by pop)
        cluster assignments for each person
        clusters[i]=j means that population unit i in [pop] is assigned to cluster j in [num]
    '''
    
    nc = int(np.sqrt(num)) #sqrt of the total number of clusters
    n = int(np.sqrt(pop))  #sqrt of the population size
    k = int(np.ceil(n/nc)) #"typical" cluster contains k*k units
    print("nc={},n={},k={}".format(nc,n,k))
    divides = n%k==0
    clusters = np.zeros((n,n), dtype=int)
    
    for i in range(n):
        for j in range(n):
            s,t = lat_toCluster(i,j,k,divides=divides) 
            clusters[i,j] = nc*s + t
    return clusters, clusters.flatten()

def clustering_metrics(n, clustering, fY):
    """
    Inputs:
        n : size of the graph
        clustering: (list of lists) disjoint partition of the graph
        fY: (function) potential outcomes model that takes in a treatment vector and spits out potential outcomes

    Outputs:
        var_cluster_avgs: (float) variance (across clusters) of average influence within a cluster
        effect_of_cut_edges: (float) the total influence coming from edges that are not contained with a cluster (hmm not sure if I buy this)
     """
    if len(clustering) > 0:
        TTE = np.sum(fY(np.ones(n))-fY(np.zeros(n)))/n

        cluster_influences = np.zeros(len(clustering))
        cluster_sizes = np.zeros(len(clustering))
        
        for c,cluster in enumerate(clustering):
            # treatment vector with 1s if unit is in current cluster, 0 o/w
            z_cluster = np.zeros(n)
            z_cluster[cluster] = 1

            cluster_influences[c] = np.sum(fY(z_cluster) - fY(np.zeros(n))) # compute and save L_cluster
            cluster_sizes[c] = len(cluster) # store size of cluster

        var_cluster_avgs = np.var(np.divide(cluster_influences, cluster_sizes)) # Variance of cluster averages
        effect_of_cut_edges = TTE - np.sum(cluster_influences)/n
    else: 
        var_cluster_avgs = np.inf
        effect_of_cut_edges = np.inf

    return var_cluster_avgs, effect_of_cut_edges

######## Potential Outcomes Model ########

def homophily_effects(G):
    '''
    Returns vectors of (normalized) homophily effects as described in the Ugander/Yin paper
        G = adjacency list representation of graph
    '''
    n = G.shape[0]
    degrees = G.sum(axis=0)
    normalized_laplacian = (-G/degrees).tocsr()
    normalized_laplacian[range(n),range(n)] = 1

    _,eigvecs_s = scipy.sparse.linalg.eigs(normalized_laplacian,k=2,which='SM')
    h = eigvecs_s[:,1].real

    h = 2*(h-min(h))/(max(h)-min(h)) - 1   # adjust so smallest value is -1 and largest value is 1

    return h

def _outcomes(Z,G,C,d,beta,delta):
    '''
    Returns a matrix of outcomes for the given tensor of treatment assignments
        Z = treatment assignments: (beta+1) x r x n
        G = adjacency list representation of graph
        C = Ugander Yin coefficients: (beta+1) x n
        d = vector of vertex degrees: n 
        beta = model degree
        delta = magnitide of direct
    '''

    if Z.ndim == 3:
        t = np.empty_like(Z)
        for b in range(Z.shape[0]):
            t[b,:,:] = Z[b,:,:] @ G
    else:
        t = Z @ G           # number of treated neighbors 

    Y = delta * Z
    for k in range(beta+1):
        Y += (binom(t,k) / np.where(d>k,binom(d,k),1)) * C[k,:]

    return Y

def pom_ugander_yin(G,h,beta):
    '''
    Returns vectors of coefficients c_i,S for each individual i and neighborhood subset S 
    Coefficients are given by a modification of Ugander/Yin's model to incorporate varied treatment effects across individuals and higher-order neighbor effects
        G = adjacency list representation of graph
        d = vector of vertex degrees
        h = vector of homophily effects
        beta = model degree
    '''

    # parameters 
    a = 1                                         # baseline effect
    b = 0                                       # magnitude of homophily effects on baselines
    sigma = 0.1                                   # magnitude of random perturbation on baselines
    delta = 0.5                                   # magnitude of direct effect
    '''
    if beta==2:
        gamma = [0, 0, 2]   # magnitude of subset treatment effects
    else:
        gamma = [0.5**(k-1) for k in range(beta+1)]   # magnitude of subset treatment effects
    '''
    gamma = [0.5**(k-1) for k in range(beta+1)] #0.75
    tau = 0                                       # magnitude of random perturbation on treatment effects

    n = G.shape[0]
    d = np.ones(n) @ G         # vertex degrees
    dbar = np.sum(d)/n

    baseline = ( a + b * h + sigma * rng.normal(size=n) ) * d/dbar

    C = np.empty((beta+1,n)) # C[k,i] = uniform effect coefficient c_i,S for |S| = k, excluding individual boost delta
    C[0,:] = baseline

    for k in range(1,beta+1):
        C[k,:] = baseline * (gamma[k] + tau * rng.normal(size=n))

    return lambda Z : _outcomes(Z,G,C,d,beta,delta)

def TTE(Y):
    '''
    Convenience function to compute the total treatment effect of a potential outcomes model
    Y : potential outcomes function Y : {0,1}^n -> R^n
    '''
    return np.mean(Y(1) - Y(0))

######## Treatment Assignments ########

def complete_staggered_rollout_two_stage_unit(n,K,r=1):
    '''
        Returns treatment samples from Completely randomized staggered rollout: _ x r x n
        n = number of individuals
        K = numpy array of length t with number of individuals to treat at time step t
        r = number of replications
    '''

    ### Initialize ###
    Z = np.zeros(shape=(K.size,r,n))
    V = rng.rand(r,n)   # random values that determine when individual i starts being treated

    # Form U
    k = K[-1]
    U_ind = np.argpartition(V, -k, axis=1)[:, -k:] # for each repetition, get the indices of the k (budget) individuals with the highest values in V
    U = np.zeros(shape=(r,n))
    np.put_along_axis(U, U_ind, 1, axis=1) # in repetion (row) r, we have an array of n elements: the i-th entry of that array is 1 if unit i is in U, 0 o/w

    # Staggered Rollout
    for t in np.arange(1,len(K)):
        Z_t_ind = np.argpartition(V, -K[t], axis=1)[:, -K[t]:] # get indices of the K[t] indivudals with the highest V_i
        np.put_along_axis(Z[t,:,:], Z_t_ind, 1, axis=1) # set the corresponding index in Z to 1

    return (Z,U)

""" #Test Case for complete_staggered_rollout_two_stage_unit
import numpy as np
from numpy.random import RandomState
rng = RandomState(19025)

n = 6
K = np.array([0,1,2])
r = 2

Z = np.zeros(shape=(K.size,r,n))
V = rng.rand(r,n)   # random values that determine when individual i starts being treated
print("V:\n{}\n".format(V))
V_sorted = np.sort(V)

# Form U
budget = K[-1]
U_ind = np.argpartition(V, -budget, axis=1)[:, -budget:] # for each repetition, get the indices of the k (bugdet) individuals with the highest values in V
print("U_ind:\n{}\n".format(U_ind))
rows = np.arange(V.shape[0])[:, None]
print("Checking that we have the {} individuals with the highest values in V: \n{}\n".format(budget,V[rows,U_ind]==V_sorted[:,budget:]))

U = np.zeros(shape=(r,n))
np.put_along_axis(U, U_ind, 1, axis=1) # in repetion (row) r, we have an array of n elements: the i-th entry of that array is 1 if unit i is in U, 0 o/w
print("U:\n{}\n".format(U))

print("Checking that ones are in the right places:\n{}\n".format(np.where(U)[1].reshape(U_ind.shape) == np.sort(U_ind)))

# Staggered Rollout
print("Z[0]:\n{}\n".format(Z[0]))
for t in np.arange(1,len(K)):
    Z_t_ind = np.argpartition(V, -K[t], axis=1)[:, -K[t]:] # get indices of the K[t] indivudals with the highest V_i
    np.put_along_axis(Z[t,:,:], Z_t_ind, 1, axis=1) # set the corresponding index in Z to 1
    print("Z[{}]:\n{}\nThe {} individuals with the highest V values should be treated in each row (repetition)\n".format(t,Z[t],K[t]))
"""


def complete_staggered_rollout_two_stage(n,Cl,nu,K,r=1):
    '''
        Returns treatment samples from Completely randomized staggered rollout: (beta+1) x r x n
        n = number of individuals
        Cl = clusters, list of lists that partition [n]
        nu = number of units selected 
        K = treatment number for each time step
        r = number of replications
    '''
    if len(Cl) == 0:
        Z,U = complete_staggered_rollout_two_stage_unit(n,K,r)
        return (Z,U)
    else:
        shuffled_clusters_flat = np.zeros((r,n),dtype=int)
        for rep in range(r):
            rng_new.shuffle(Cl) # shuffle the clusters 
            shuffled_clusters_flat[rep] = np.array(list(itertools.chain.from_iterable(Cl))) # flatten the list of shuffled clusteres

        # Select the first nu units, shape (r,nu)
        first_stage_selected = shuffled_clusters_flat[:,:nu]

        # Create selection_mask, of size (r,n), indicating for each repetition which nu units were chosen
        selection_mask = np.zeros((r,n),dtype=int)
        # Set the elements at the specified indices to 1
        selection_mask[np.arange(r)[:, None], first_stage_selected] = 1

        # Run a completely randomized staggered rollout on units in selected clusters
        Z = np.zeros((len(K),r,n))
        second_stage_indices = rng_new.permuted(first_stage_selected, axis=1) 

        for t in range(1,len(K)):
            to_treat = second_stage_indices[:,:K[t]]
            rows = np.arange(r)[:, None]
            Z[t,rows,to_treat] = 1

        return (Z, selection_mask)

"""
#Test Case for complete_staggered_rollout_two_stage
import numpy as np
Cl = [[0,1,2,3],[4,5,6,7],[8,9,10,11]]
n = 12
nc = len(Cl) # number of clusters
r = 3
nu = 4
rng_new = np.random.default_rng()
K = [0,2,4]

shuffled_clusters_flat = np.zeros((r,n),dtype=int)
for rep in range(r):
    rng_new.shuffle(Cl) # shuffle the clusters 
    shuffled_clusters_flat[rep] = np.array(list(itertools.chain.from_iterable(Cl))) # flatten the list of shuffled clusteres

print("Shuffled units in first stage for each repetition: \n{}".format(shuffled_clusters_flat))

# Select the first nu units, shape (r,nu)
first_stage_selected = shuffled_clusters_flat[:,:nu]
print(first_stage_selected,first_stage_selected.shape)

# Create selection_mask, of size (r,n), indicating for each repetition which nu units were chosen
selection_mask = np.zeros((r,n),dtype=int)
# Set the elements at the specified indices to 1
selection_mask[np.arange(r)[:, None], first_stage_selected] = 1
print(selection_mask,selection_mask.shape)

Z = np.zeros((len(K),r,n))
#print(np.nonzero(selection_mask))
second_stage_indices = rng_new.permuted(first_stage_selected, axis=1) 
print("Indices to be treated (rows correspond to repetitions) :\n{}\n".format(second_stage_indices))

print("Treatments during each phase of the rollout:\n")
print(0)
print(Z[0,:,:])
for t in range(1,len(K)):
    to_treat = second_stage_indices[:,:K[t]]
    print(t, to_treat)
    rows = np.arange(r)[:, None]
    Z[t,rows,to_treat] = 1
    print(Z[t,:,:])
"""

######## Estimator ########

def _interp_coefficients(P):
    '''
    Returns coefficients h_t = l_t,P(1) - l_t,P(0) for pi estimator
        P = treatment probabilities for each time step: beta+1
    '''
    T = len(P)

    H = np.zeros(T)

    for t in range(T):
        denom = np.prod([(P[t] - P[s]) for s in range(T) if s != t])
        H[t] = np.prod([(1 - P[s]) for s in range(T) if s != t]) 
        H[t] -= np.prod([(-P[s]) for s in range(T) if s != t])
        H[t] /= denom

    return H

def complete_coeffs(beta, nu, k):
    '''
    Returns coefficients l_t from Complete Staggered Rollout

    n (int): size of population
    K (numpy array): total number of individuals treated by each timestep
    TODO: change from nu to n?
    '''

    ### Initialize ###
    L = np.zeros(beta+1)             # for the coefficients L_t

    for t in range(beta+1):
        left = np.divide(np.multiply(np.ones(beta), beta*nu) - np.multiply(np.concatenate((np.arange(t), np.arange(t + 1, beta+1))), k), 
                         t*k - np.multiply(np.concatenate((np.arange(t), np.arange(t + 1, beta+1))), k))
        right = np.divide(np.multiply(np.concatenate((np.arange(t), np.arange(t + 1, beta+1))), -1),
                          t - np.concatenate((np.arange(t), np.arange(t + 1, beta+1))))
        L[t] = np.prod(left) - np.prod(right)        
    return L

def pi_estimate_tte_two_stage(Y,p,Q):
    '''
    Returns TTE estimate from polynomial interpolation
        Y = potential outcomes: (beta+1) x r x n
        poq = cluster selection probability
        Q = treatment probabilities of selected units for each time step: beta+1
    '''
    n = Y.shape[-1]
    H = _interp_coefficients(Q)
    
    return 1/(n*p/Q[-1]) * H @ np.sum(Y,axis=-1)

def two_stage_restricted_estimator(Y,U,p,Q):
    '''
    Returns TTE estimate from polynomial interpolation
        Y = potential outcomes: (beta+1) x r x n
        U = selected individuals: r x n
        p = treatment budget
        Q = treatment probabilities of selected units for each time step: beta+1
    '''
    n = Y.shape[-1]
    H = _interp_coefficients(Q)
    
    return 1/(n*p/Q[-1]) * H @ np.sum(Y*U,axis=-1)

def one_stage_pi(Y, probs):
    '''
    Staggered-rollout polynomial interpolation estimator 
    Y: Tensor of potential outcomes - T x r x n
    probs: Array of treatment probabilities at each time step - T
    Returns: Array of TTE estimates in each replication - r
    '''
    H = _interp_coefficients(probs)

    return H @ np.mean(Y,axis=2)


######## Other Estimators ########

def dm_estimate_tte(Z,Y):
    '''
    Returns TTE estimate from difference in means
        Z = treatment assignments: r x n
        Y = potential outcomes: r x n
    '''
    T,_,n = Z.shape

    num_treated = Z.sum(axis=2)

    DM_data = np.sum(Y*Z,axis=2)/np.maximum(num_treated,1)          # (beta+1) x r
    DM_data -= np.sum(Y*(1-Z),axis=2)/np.maximum(n-num_treated,1)  
    return np.sum(DM_data,axis=0)/T


def dm_threshold_estimate_tte(Z,Y,G,gamma):
    '''
    Returns TTE estimate from a thresholded difference in means
    i is "sufficiently treated" if they are treated, along with a (1-gamma) fraction of Ni
    i is "sufficiently control" if they are control, along with a (1-gamma) fraction of Ni
        Z = treatment assignments: (beta+1) x r x n
        Y = potential outcomes function: {0,1}^n -> R^n
        G = causal network (this estimator is not graph agnostic)
        gamma = tolerance parameter (as described above)
    '''

    T,_,n = Z.shape

    d = np.ones(n) @ G                             # vertex degrees
    num_Ni_treated = np.empty_like(Z)              # number of treated neighbors
    for t in range(T):
        num_Ni_treated[t,:,:] = Z[t,:,:] @ G
    frac_Ni_treated = num_Ni_treated / d

    sufficiently_treated = (frac_Ni_treated >= (1-gamma)) * Z
    num_sufficiently_treated = sufficiently_treated.sum(axis=2)
    sufficiently_control = (frac_Ni_treated <= gamma) * (1-Z)
    num_sufficiently_control = sufficiently_control.sum(axis=2)

    DM_data = np.sum(Y*sufficiently_treated,axis=2)/np.maximum(num_sufficiently_treated,1)          # (beta+1) x r
    DM_data -= np.sum(Y*sufficiently_control,axis=2)/np.maximum(num_sufficiently_control,1)  
    return np.sum(DM_data,axis=0)/T

def _neighborhood_cluster_sizes(N,Cl):
    '''
    Returns a list which, for each i, has an array of its number of neighbors from each cluster
        N = neighborhoods, list of adjacency lists
        Cl = clusters, list of lists that partition [n]
    '''
    n = len(N)

    membership = np.zeros(n,dtype=np.uint32)
    for i,C in enumerate(Cl):
        membership[C] = i

    neighborhood_cluster_sizes = np.zeros((n,len(Cl)))
    for i in range(n):
        for j in N[i]:
            neighborhood_cluster_sizes[i,membership[j]] += 1
    
    return neighborhood_cluster_sizes

def ht_hajek_estimate_tte(Z,Y,G,Cl,p,q):
    '''
    Returns TTE Horvitz-Thompson/Hajek estimates
        Z = treatment assignments: r x n
        Y = potential outcomes function: {0,1}^n -> R^n
        G = causal network (this estimator is not graph agnostic)
        Cl = clusters, list of lists that partition [n]
        p = treatment budget
        q = treatment probabilities in selected clusters
    '''
    _,n = Z.shape

    N = []
    for i in range(n):
        N.append(G[:,[i]].nonzero()[0])

    ncs = _neighborhood_cluster_sizes(N,Cl)
    d = ncs.sum(axis=1)               # degree
    cd = np.count_nonzero(ncs,axis=1) # cluster degree

    Ni_fully_treated = np.empty_like(Z)
    for i in range(n):
        Ni_fully_treated[:,i] = np.prod(Z[:,N[i]],axis=1)

    Ni_fully_control = np.empty_like(Z)
    for i in range(n):
        Ni_fully_control[:,i] = np.prod(1-Z[:,N[i]],axis=1)

    prob_fully_treated = np.power(p/q,cd) * np.power(q,d)
    prob_fully_control = np.prod(1 - p/q*(1-np.power(1-q,ncs)),axis=1)

    HT_data = (np.sum(Y * Ni_fully_treated/prob_fully_treated, axis=1) - np.sum(Y * Ni_fully_control/prob_fully_control, axis=1))/n

    nhat1 = np.sum(Ni_fully_treated/prob_fully_treated, axis=1)
    nhat2 = np.sum(Ni_fully_control/prob_fully_control, axis=1)

    Hajek_data = np.sum(Y * Ni_fully_treated/prob_fully_treated, axis=1)/nhat1
    Hajek_data -= np.sum(Y * Ni_fully_control/prob_fully_control, axis=1)/nhat2  
    return (HT_data,Hajek_data)

def dyadic_HT(Z_last, Y_last, U, select_prob, treat_prob):
    '''
    Returns estimates using 2-stage Horvitz-Thompson estimator from Deng, et al (2024), for each repetition

    Z_last (arr): treatment assignment from final stage, shape=(r,n)
    Y_last (arr): outcomes from final stage, shape=(r,n)
    U (arr): units selected from the first stage, shape=(r,n)
    select_prob (float): probability of unit selection in first stage
    treat_prob (float): marginaly treatment probability for all units
    '''
    n = U.shape[-1]
    TTE_hat_Y = (1/(n*select_prob*treat_prob)) * np.sum(Z_last * Y_last * U, axis=1)
    TTE_hat_D = (1/(n*select_prob*(1-treat_prob))) * np.sum((1-Z_last) * Y_last * U, axis=1)
    return TTE_hat_Y + TTE_hat_D

######## Utility Function for Computing Effect Sizes ########

def e(n,S):
    v = np.zeros(n)
    v[S] = 1
    return v

def LPis(fY,Cl,n):
    L = {}

    for i,C1 in enumerate(Cl):
        L[frozenset([i])] = np.sum(fY(e(n,C1)) - fY(np.zeros(n)))

        for ip,C2 in enumerate(Cl):
            if ip >= i: continue
            L[frozenset([i,ip])] = np.sum(fY(e(n,list(C1)+list(C2))) - fY(e(n,C1)) - fY(e(n,C2)) + fY(np.zeros(n)))

    return L