# Creates Figure 1: Visualization of extrapolated polynomials
import matplotlib.pyplot as plt
from experiment_functions import *

plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"

n = 400
beta = 3
r = 200

lat_n = int(np.sqrt(n))
A = lattice2Dsq(lat_n,lat_n)

h = homophily_effects(A)
fY = pom_ugander_yin(A,h,beta)

p = 0.15
P = [0, p/3, 2*p/3, p] 

nu = 160
K = np.linspace(0,int(n*p),beta+1,dtype=int)
Q = K/nu

l0 = lambda P: lambda x: (x-P[1])*(x-P[2])*(x-P[3])/((P[0]-P[1])*(P[0]-P[2])*(P[0]-P[3]))
l1 = lambda P: lambda x: (x-P[0])*(x-P[2])*(x-P[3])/((P[1]-P[0])*(P[1]-P[2])*(P[1]-P[3]))
l2 = lambda P: lambda x: (x-P[0])*(x-P[1])*(x-P[3])/((P[2]-P[0])*(P[2]-P[1])*(P[2]-P[3]))
l3 = lambda P: lambda x: (x-P[0])*(x-P[1])*(x-P[2])/((P[3]-P[0])*(P[3]-P[1])*(P[3]-P[2]))

f = lambda Y: lambda P: lambda x : Y[0]*l0(P)(x) + Y[1]*l1(P)(x) + Y[2]*l2(P)(x) + Y[3]*l3(P)(x)

fig,ax = plt.subplots(1,2,sharey=True,sharex=True, figsize=(10,5))

fig.set_figheight(4)
fig.set_figwidth(12)
ax[0].set_title("One-Stage Rollout",fontsize=16)
ax[1].set_title("Two-Stage Rollout",fontsize=16)
ax[1].yaxis.set_tick_params(which='both', labelleft=True)

plt.setp(ax,xlim=(0,1))
plt.setp(ax,ylim=(0,10))
plt.setp(ax,xlabel="$x$")

x = np.linspace(0,1,10000)

Z,_ = complete_staggered_rollout_two_stage_unit(n, K, r=r)
Y = 1/n*np.sum(fY(Z),axis=2)

for i in range(r):
    f_hat = f(Y[:,i])(P)
    ax[0].plot(x,f_hat(x),color="tab:blue",alpha=0.1)


for i in range(r):
    f_hat = f(Y[:,i])(Q)
    ax[1].plot(x,Q[-1]/p*(f_hat(x)-f_hat(0))+f_hat(0),color="tab:blue",alpha=0.1)

true_y = [1/n*np.sum(fY(np.ones(n)*P[0])),1/n*np.sum(fY(np.ones(n)*P[1])),1/n*np.sum(fY(np.ones(n)*P[2])), 1/n*np.sum(fY(np.ones(n)*P[3]))]

for axis in ax:
    axis.set_ylabel("$\\widehat{F}(x)$", labelpad=20, rotation='horizontal', fontsize=16)
    axis.xaxis.label.set(fontsize=16)
    axis.plot(x,f(true_y)(P)(x),"k",linewidth=2)

    axis.axvline(x=0.05,color="k",linewidth=0.5,linestyle='--')
    axis.axvline(x=0.1,color="k",linewidth=0.5,linestyle='--')
    axis.axvline(x=0.15,color="k",linewidth=0.5,linestyle='--')

fig.tight_layout()
plt.show()
