# Used to create data for Figures 2,8,9,10, and 13
import json
import pickle
from experiment_functions import *
from itertools import product
from joblib import Parallel, delayed 

def pi_estimate_tte_two_stage_complete(Y,nu,k,beta):
    '''
    Returns TTE estimate from polynomial interpolation
        Y = potential outcomes: (beta+1) x r x n
        nu = the (expected size) of U
        k = budget (in units)
    '''
    H = complete_coeffs(beta,nu,k)
    return 1/nu * H @ np.sum(Y,axis=-1)

def estimate_two_stage(fY,G,Cl,n,p,r,beta,gamma,q):
    nu = int(p*n/q)
    K = np.linspace(0,int(n*p),beta+1,dtype=int)
    q = (n*p)/nu

    tte_hat = {"pi_cluster":[],
               "dm_cluster":[],
               "dmt_cluster":[],
               "ht_cluster":[],
               "hajek_cluster":[],
               "pi_linear": [],
               "pi_bernoulli":[]
               }
   
    for _ in range(r//100):
        Zc,_ = complete_staggered_rollout_two_stage(n, Cl, nu, K, r) 
        Yc = fY(Zc)

        tte_hat["pi_cluster"] = np.append(tte_hat["pi_cluster"],pi_estimate_tte_two_stage_complete(Yc,nu,K[-1],beta))
        tte_hat["dm_cluster"] = np.append(tte_hat["dm_cluster"],dm_estimate_tte(Zc[:-1,:,:],Yc[:-1,:,:]))
        tte_hat["dmt_cluster"] = np.append(tte_hat["dmt_cluster"],dm_threshold_estimate_tte(Zc[:-1,:,:],Yc[:-1,:,:],G,gamma))
        (ht_estimate,hajek_estimate) = ht_hajek_estimate_tte(Zc[-1,:,:],Yc[-1,:,:],G,Cl,p,q)
        tte_hat["ht_cluster"] = np.append(tte_hat["ht_cluster"],ht_estimate)
        tte_hat["hajek_cluster"] = np.append(tte_hat["hajek_cluster"],hajek_estimate)

        # q=1 estimator
        tte_hat["pi_linear"] = np.append(tte_hat["pi_linear"],pi_estimate_tte_two_stage_complete(Yc[[0,-1],:,:],nu,K[-1],1))

        # vanilla PI estimator (I think)
        Zb,_ = complete_staggered_rollout_two_stage_unit(n,K,r)
        Yb = fY(Zb)

        tte_hat["pi_bernoulli"] = np.append(tte_hat["pi_bernoulli"],pi_estimate_tte_two_stage_complete(Yb,n,K[-1],beta))


    return (p, tte_hat)

def run_experiment(G,Cls,fixed,varied,r,gamma):
    n = G.shape[0]
    h = homophily_effects(G)

    betas = [fixed["beta"]] if "beta" in fixed else varied["beta"] 
    ncs = [fixed["nc"]] if "nc" in fixed else varied["nc"]
    qs = [fixed["q"]] if "q" in fixed else varied["q"]

    data = { "p":[], "est":[], "treatment":[], "bias":[], "var":[] }
    if "beta" in varied: data["beta"] = []
    if "nc" in varied: data["nc"] = []
    if "q" in varied: data["q"] = []

    for beta in betas:
        print("beta = {}".format(beta))
        fY = pom_ugander_yin(G,h,beta)
        TTE = np.sum(fY(np.ones(n))-fY(np.zeros(n)))/n

        for nc,q in product(ncs,qs):
            print("nc = {}, q = {}".format(nc, q))
            for (p,results) in Parallel(n_jobs=-1, verbose=10)(delayed(lambda p : estimate_two_stage(fY,G,Cls[nc],n,p,r,beta,gamma,q))(p) for p in np.linspace(0.1,0.5,16)): #24
                nu = int(p*n/q)
                data["p"] += [p]*len(results)
                if "beta" in varied: data["beta"] += [beta]*len(results)
                if "nc" in varied: data["nc"] += [nc]*len(results)
                if "q" in varied: data["q"] += [(n*p)/nu]*len(results)

                for label,TTE_hat in results.items():
                    est,treatment = label.split("_")
                    data["treatment"].append(treatment)
                    data["est"].append(est)

                    mean = np.nanmean(TTE_hat)
                    variance = np.nanmean((TTE_hat - mean)**2)

                    data["bias"].append(mean - TTE)
                    data["var"].append(variance)

    return data

if __name__ == '__main__':
    my_path = "Amazon/Experiments/compare_estimators.json" # path to .json file for the experiment
    jf = open(my_path,'rb')
    j = json.load(jf)
    jf.close()

    exp_name = j["name"]
    network_folder = j["network"]
    in_file = j["input"]

    print("Loading Graph")

    nf = open(network_folder + "/" + in_file,'rb')
    G,Cls = pickle.load(nf)
    nf.close()

    fixed = j["fix"]
    varied = j["vary"]
    r = j["replications"]
    gamma = j["gamma"]

    data = run_experiment(G,Cls,fixed,varied,r,gamma)

    out_file = network_folder + "/Experiments/" + exp_name + ".pkl"
    print(f"Writing output to {out_file}")
    of = open(out_file,'wb')
    pickle.dump(data,of)
    of.close()
