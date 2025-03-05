# Used to create Figures 2,8,9,10, and 13
import pickle
import pandas as pd
import matplotlib.pyplot as plt

def plot(ax,df,mse,**kwargs):
    if mse:
        ax.plot(df['p'], df['mse'],**kwargs)
    else:
        ax.plot(df['p'], df['bias'],**kwargs)

        kwargs["label"]=None
        if "marker" in kwargs: kwargs.pop("marker")
        if "markersize" in kwargs: kwargs.pop("markersize")
        if "linestyle" in kwargs: kwargs.pop("linestyle")
        ax.fill_between(df['p'], df['bias']-df['sd'], df['bias']+df['sd'],alpha=0.2, **kwargs)

def draw_plots(data,outfile,mse,network_name,b=0,**kwargs):
    df = pd.DataFrame(data)
    df['sd'] = (df['var'])**(1/2)
    df['mse'] = df['mse'] = df['bias']**2 + df['var']

    df = df[(df['treatment']=='cluster') | (df['est'] == 'pi')]
    df = df[(df['est']!='ht')] # don't show results for Horvitz-Thompson estimator
    if network_name!="Amazon":
        df = df[(df['est']!='hajek')]
    df['est'] = df.apply( lambda row: row['est'] if row['treatment']=='cluster' else 'pi1', axis=1)
    df['est'] = df.apply( lambda row: 'piL' if row['treatment']=='linear' else row['est'], axis=1)

    colors = ["tab:blue","tab:purple","tab:orange","tab:red","tab:green","tab:pink"]

    est_kws = {
        'pi' : {
            'label': '2-Stage',
            'color': colors[0],
            'marker': 'o',
            'markersize': 4
        },
        'pi1' : {
            'label': 'PI',
            'color': colors[1],
            'linestyle': '--'
        },
        'dm' : {
            'label': 'DM',
            'color': colors[2],
            'marker': 'x',
            'markersize': 6
        },
        'dmt' : {
            'label': 'DM(0.75)',
            'color': colors[3],
            'linestyle': '-.'
        },
        'hajek' : {
            'label': 'Hájek',
            'color': colors[4],
            'linestyle': 'dotted'
        },
        'piL' : {
            'label': 'q=1',
            'color': colors[5],
            'linestyle': ':'
        }
    }

    betas = [1,2,3]

    f,ax = plt.subplots(1,len(betas), figsize=(len(betas)*5,5))

    plt.setp(ax,xlim=(min(df['p']),max(df['p'])))

    # Amazon
    if network_name == "Amazon":
        if "dimensions" in kwargs:
            ax[0].set_ylim(kwargs["dimensions"][0][0], kwargs["dimensions"][0][1])
            ax[1].set_ylim(kwargs["dimensions"][1][0], kwargs["dimensions"][1][1]) 
            ax[2].set_ylim(kwargs["dimensions"][2][0], kwargs["dimensions"][2][1])
        elif mse:
            #plt.setp(ax,ylim=(-0.1,6))
            ax[0].set_ylim(-0.1,13) #-0.025,1 #-0.1,13
            ax[1].set_ylim(-0.1,13) #-0.1,6   #-0.1,13
            ax[2].set_ylim(-0.1,13) #-0.1,6   #-0.1,13
        else:
            #plt.setp(ax,ylim=(-2.5,2)) #-0.8,0.6
            ax[0].set_ylim(-0.4,0.2) #-0.5,0.4 / appendix: -0.4,0.2
            ax[1].set_ylim(-4,2) #-2.5,2 / appendix: -4,2
            ax[2].set_ylim(-4,2) #-2.5,2 / appendix: -4,2
            # ax.set_ylim(-0.05,0.05)

    # BlogCatalog
    if network_name == "BlogCatalog":
        if "dimensions" in kwargs:
            ax[0].set_ylim(kwargs["dimensions"][0][0], kwargs["dimensions"][0][1])
            ax[1].set_ylim(kwargs["dimensions"][1][0], kwargs["dimensions"][1][1]) 
            ax[2].set_ylim(kwargs["dimensions"][2][0], kwargs["dimensions"][2][1])
        elif mse:
            #plt.setp(ax,ylim=(-0.1,6))
            ax[0].set_ylim(-0.025,0.9)
            ax[1].set_ylim(-0.1,9)
            ax[2].set_ylim(-0.1,9)
        else:
            #plt.setp(ax,ylim=(-2.5,2)) #-0.8,0.6
            ax[0].set_ylim(-3,3)
            ax[1].set_ylim(-3,3)
            ax[2].set_ylim(-3,3)
            # ax.set_ylim(-0.05,0.05)

    # Email
    if network_name == "Email":
        if "dimensions" in kwargs:
            ax[0].set_ylim(kwargs["dimensions"][0][0], kwargs["dimensions"][0][1])
            ax[1].set_ylim(kwargs["dimensions"][1][0], kwargs["dimensions"][1][1]) 
            ax[2].set_ylim(kwargs["dimensions"][2][0], kwargs["dimensions"][2][1])
        elif mse:
            #plt.setp(ax,ylim=(-0.1,6))
            ax[0].set_ylim(-0.025,0.9)
            ax[1].set_ylim(-0.1,9)
            ax[2].set_ylim(-0.1,9)
        else:
            #plt.setp(ax,ylim=(-2.5,2)) #-0.8,0.6
            ax[0].set_ylim(-3,3)
            ax[1].set_ylim(-3,3)
            ax[2].set_ylim(-3,3)
            # ax.set_ylim(-0.05,0.05)
    
    ests = df["est"].unique()
    if len(betas) == 1:
        ax.set_xlabel('p',fontsize=14)
        ax.set_ylabel('MSE' if mse else 'Bias', fontsize=14)
        for est in ests:
            #if est == 'hajek': continue
            for j,beta in enumerate(betas):
                ax.set_title(f"$\\beta={beta}$", fontsize=16)
                plot(ax,df[(df["est"] == est) & (df["beta"] == beta)],mse,**est_kws[est])

        ax.legend(ncol=2,prop={'size': 12})
    else:
        for a in ax:
            a.set_xlabel('p',fontsize=14)
            a.set_ylabel('MSE' if mse else 'Bias', fontsize=14)
        for est in ests:
            #if est == 'hajek': continue
            for j,beta in enumerate(betas):
                ax[j].set_title(f"$\\beta={beta}$", fontsize=16)
                plot(ax[j],df[(df["est"] == est) & (df["beta"] == beta)],mse,**est_kws[est])

        ax[0].legend(ncol=2,prop={'size': 12})
    

    #f.subplots_adjust(bottom=0.25)
    f.suptitle("{} Network, b={}".format(network_name, b), fontsize=20)
    f.tight_layout()
    plt.show()
    f.savefig(outfile)


if __name__ == '__main__':
    mse = True # plot MSE ?  if false, plots bias and variance plots instead
    network_name = "Email"    
    data_file = open(network_name + "/Experiments/compare_estimators.pkl", 'rb')
    data = pickle.load(data_file)
    data_file.close()

    save_name = "compare_estimators_" + network_name
    if mse:
        save_name = save_name + "_MSE.png"
    else:
        save_name = save_name + ".png"
    draw_plots(data,save_name,mse,network_name,b=0)