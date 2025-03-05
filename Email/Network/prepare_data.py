import pickle
import re
import numpy as np
import scipy.sparse

n = 1005
G_dict = { i:[i] for i in range(n)}  # dictionary mapping each vertex to list of neighbors

for line in open("edges.txt", "r"):
    m = re.match("^([0-9]*) ([0-9]*)$",line)
    u,v = m.group(1,2)
    u,v = int(u),int(v)

    G_dict[v].append(u) 

G = scipy.sparse.lil_array((n,n))
    
for i in range(n):
    G[i,G_dict[i]] = 1

G = G.tocsr()

membership = np.empty(n)

for line in open("communities.txt", "r"):
    m = re.match("^([0-9]*) ([0-9]*)$",line)
    v,c = m.group(1,2)
    v,c = int(v),int(c)

    membership[v] = c

Cls = {42:[]}
for i in range(int(np.max(membership))+1):
    Cls[42].append(list(np.where(membership == i)[0]))

file = open("data.pkl", "wb")
pickle.dump((G,Cls), file)
file.close()
