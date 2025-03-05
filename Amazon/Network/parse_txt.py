# you will need to download amazon-meta.txt from https://snap.stanford.edu/data/amazon-meta.html and put it inside this folder

import re
import pickle

products = []

current = {}
accept = False

for line in open("amazon-meta.txt", "r"):
    if re.match("^Id:.*$",line):
        if accept:
            products.append(current)

        current = {"categories":[]}
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
    elif re.match("\\s*salesrank:\\s*[0-9].*$",line):
       rank = int(re.match("\\s*salesrank:\\s*([0-9]*)$",line).group(1))
       current["rank"] = rank
    elif re.match("^\\s*similar:.*$",line):
       similar_list = re.match("\\s*similar:\\s*[0-9]*\\s*(.*)$",line).group(1)
       similar_asins = re.split("\\s\\s*",similar_list)
       current["neighbors"] = similar_asins
    elif re.match("^\\s*\\|.*$",line):
       category = int(re.match("^.*\\[([0-9]*)\\]$",line).group(1))
       current["categories"].append(category)

file = open("products.pkl", "wb")
pickle.dump(products, file)
file.close()