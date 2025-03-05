import scipy
import pickle
import pymetis
import numpy as np
from joblib import Parallel, delayed 

print("Loading Products")

file = open("products.pkl", "rb")
products = pickle.load(file)

n = len(products)
print("Found {} DVDs".format(n))

print("Assigning sequential ids to products")

id_dict = {}
for i,product in enumerate(products):
    id_dict[product["id"]] = i

for product in products:
    product["id"] = id_dict[product["id"]]
    product["neighbors"] = [id_dict[s] for s in product["neighbors"] if s in id_dict]

# # symmetrize
# for i in range(n):
#     for j in products[i]["neighbors"]:
#         products[j]["neighbors"].append(i)

# print("Restrict to largest connected component")

# G1 = scipy.sparse.lil_array((n,n))

# for p in products:
#     i = p["id"]
#     for j in p["neighbors"]:
#         G1[i,j] = 1
#         G1[j,i] = 1

# G1 = nx.to_networkx_graph(G1)
# largest_cc = max(nx.connected_components(G1), key=len)

# products = [p for p in products if p["id"] in largest_cc]
# n = len(products)
# print("Reduced to {} movies".format(n))

# new_ids = {}
# for (i,p) in enumerate(products):
#     new_ids[p["id"]] = i

# for product in products:
#     product["id"] = new_ids[product["id"]]
#     product["neighbors"] = [new_ids[s] for s in product["neighbors"]]

print("Forming Product Graph")

G = scipy.sparse.lil_array((n,n))

for p in products:
    i = p["id"]
    G[i,i] = 1
    for j in p["neighbors"]:
        G[i,j] = 1

G = G.tocsr()

print("Constructing Feature Graph")

xadj = [0]
adjncy = []
eweights = []
for i in range(n):
    if i%1000 == 0: print(i)
    for j in range(n):
        l = len(set(products[i]["categories"]).intersection(set(products[j]["categories"])))
        if l > 0:
            adjncy.append(j)
            eweights.append(l)
    xadj.append(len(adjncy))

print("Computing Clusterings")
Cls = {}
for (nc,(_,membership)) in Parallel(n_jobs=-1, verbose=20)(delayed(lambda nc : (nc,pymetis.part_graph(nparts=nc,xadj=xadj,adjncy=adjncy,eweights=eweights)))(nc) for nc in range(50,1001,50)):
    membership = np.array(membership)
    Cl = []
    for i in range(nc):
        Cl.append(np.where(membership == i)[0])
    Cls[nc] = Cl

file = open("data.pkl", "wb")
pickle.dump((G,Cls), file)
file.close()
