import re
import pickle
import pymetis
import numpy as np
import scipy.sparse
from joblib import Parallel, delayed 

print("Importing Graph Edges")

n = 10312
G_dict = { i:[i] for i in range(n)}  # dictionary mapping each vertex to list of neighbors

for line in open("edges.csv", "r"):
    m = re.match("^([0-9]*),([0-9]*)$",line)
    u,v = m.group(1,2)
    u,v = int(u)-1,int(v)-1

    G_dict[u].append(v) 
    G_dict[v].append(u)

G = scipy.sparse.lil_array((n,n))
    
for i in range(n):
    G[i,G_dict[i]] = 1

G = G.tocsr()

print("Importing Features")

feature_dict = { i:[] for i in range(n)} # dictionary mapping each vertex to list of communities

all_features = set()
for line in open("communities.csv", "r"):
    m = re.match("^([0-9]*),([0-9]*)$",line)
    v,c = m.group(1,2)
    v,c = int(v)-1,int(c)-1

    feature_dict[v].append(c)
    all_features.add(c)

nf = len(all_features)

features = np.zeros((n,nf))
for i,fi in feature_dict.items():
    features[i,fi] = 1

print("Constructing Feature Graph")

A = features @ features.T

xadj = [0]
adjncy = []
eweights = []
for i in range(n):
    for j in np.nonzero(A[i,:])[0]:
        adjncy.append(j)
        eweights.append(int(A[i,j]))
    xadj.append(len(adjncy))

print("Computing Clusterings")
Cls = {}
for (nc,(_,membership)) in Parallel(n_jobs=-1, verbose=20)(delayed(lambda nc : (nc,pymetis.part_graph(nparts=nc,xadj=xadj,adjncy=adjncy,eweights=eweights)))(nc) for nc in range(10,51,5)):
    membership = np.array(membership)
    Cl = []
    for i in range(nc):
        Cl.append(np.where(membership == i)[0])
    Cls[nc] = Cl

file = open("data.pkl", "wb")
pickle.dump((G,Cls), file)
file.close()