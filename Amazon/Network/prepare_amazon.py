import re
import scipy
import pickle

products = []

current = {}
accept = False

for line in open("amazon-meta.txt", "r"):
    if re.match("^Id:.*$",line):
        if accept:
            products.append(current)

        current = {}
        accept = True
    elif re.match("^ASIN:.*$",line):
        asin = re.match("^ASIN:\\s*([0-9A-Z]*)$",line).group(1)
        current["id"] = asin
    elif re.match("^\\s*discontinued.*$",line):
        accept = False
    elif re.match("^\\s*group:.*$",line):
        group = re.match("^\\s*group: (.*)$",line).group(1)
        if group != "DVD": #and group != "Video":
            accept = False
    elif re.match("^\\s*similar:.*$",line):
       similar_list = re.match("\\s*similar:\\s*[0-9]*\\s*(.*)$",line).group(1)
       similar_asins = re.split("\\s\\s*",similar_list)
       current["neighbors"] = similar_asins


n = len(products)
print("Found {} DVDs".format(n))

print("Assigning sequential ids to products")

id_dict = {}
for i,product in enumerate(products):
    id_dict[product["id"]] = i

for product in products:
    product["id"] = id_dict[product["id"]]
    product["neighbors"] = [id_dict[s] for s in product["neighbors"] if s in id_dict]

print("Forming Product Graph")

G = scipy.sparse.lil_array((n,n))

for p in products:
    i = p["id"]
    G[i,i] = 1
    for j in p["neighbors"]:
        G[i,j] = 1

G = G.tocsr()

file = open("graph.pkl", "wb")
pickle.dump(G, file)
file.close()
