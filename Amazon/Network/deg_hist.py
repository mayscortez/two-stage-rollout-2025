import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

import os
print(os.getcwd())

file = open("data.pkl", "rb")
G,Cls = pickle.load(file)

d = np.sum(G,axis=0)

plt.rc('text',usetex=True)

plt.figure(figsize=(10,3))
plt.hist(d,bins=range(0,25,1))
plt.title("Amazon Network In-degree Histogram")
plt.xlabel("Degree")
plt.ylabel("Frequency")
plt.subplots_adjust(bottom=0.25)
plt.savefig("amazon_degrees.png", dpi=300)
plt.show()
