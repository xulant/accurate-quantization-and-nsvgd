import numpy as np
import torch
import math
import matplotlib.pyplot as plt
import scipy
from matplotlib.pyplot import figure
from scipy.integrate import quad
from scipy.stats import norm
from botorch.utils.sampling import sample_hypersphere

def score(x):   #score of normalized Gaussian distribution
    return -x*p 

def mmd(x, sig_mmd = 1):
    n,p = x.shape
    var = 1/p
    dists_xx = ((x[:, None, :] - x[None, :, :]) ** 2).sum(axis=-1)
    norm_x = (x**2).sum(axis=-1)
    return (sig_mmd/(2*var+sig_mmd))**(p/2) \
            - (sig_mmd/(var+sig_mmd))**(p/2)*(np.exp(- norm_x / (var + sig_mmd) / 2)).sum()/n*2 \
            + (np.exp(- dists_xx / sig_mmd / 2)).sum()/ (n**2)
    
def gaussian_kernel(x, y, sigma):
    d = (x[:, None, :] - y[None, :, :])
    dists = (d ** 2).sum(axis=-1)
    return torch.exp(- dists / sigma / 2)

def gaussian_stein_kernel_single(x, score_x, sigma, return_kernel=False):
    _, p = x.shape
    # Gaussian kernel:
    norms = (x ** 2).sum(-1)
    dists = -2 * x @ x.t() + norms[:, None] + norms[None, :]
    k = (-dists / 2 / sigma).exp()

    # Dot products:
    diffs = (x * score_x).sum(-1, keepdim=True) - (x @ score_x.t())
    diffs = diffs + diffs.t()
    scalars = score_x.mm(score_x.t())
    der2 = p - dists / sigma
    stein_kernel = k * (scalars + diffs / sigma + der2 / sigma)
    if return_kernel:
        return stein_kernel, k
    return stein_kernel

def ksd(x, bw):
    if type(x) is np.ndarray:
        x = torch.from_numpy(x)
    K = gaussian_stein_kernel_single(x, score(x), bw)
    return K.mean().item() / bw

n_dir = 50 # number of directions
bw = 1  # kernel bandwidth
p = 2  # dimension
n_dict = [4,8,16,32,64, 128, 256, 512, 1024, 1600]
label_dict = ['mmd', 'ksd', 'svgd', 'nsvgd']# add ['herd', 'stein'] if needed
l = len(n_dict)
num_output = len(label_dict)
repeat = 3

def cdf_dis(theta, X, u):
    proj = X@theta
    return (proj <= u).sum().item()/len(X)
    
def cdf_Gauss(u):
    return norm.cdf(u*math.sqrt(p))

def sw_theta(theta, X):
    res = quad(lambda x: abs(cdf_dis(theta, X, x)-cdf_Gauss(x)), -np.inf, np.inf)
    return res[0]

def Gauss_sw(X):
    sw = 0
    for j in range(n_dir):
        theta = Theta[j]
        sw += sw_theta(theta, X)
        sw /= n_dir
    return sw

Theta = sample_hypersphere(p, n_dir).double()

def evaluation(n,i,bw):
    '''
    import final states here
    '''
    mmd_mmd, mmd_ksd, mmd_svgd, mmd_nsvgd = mmd(x_mmd, bw), mmd(x_ksd, bw), mmd(x_svgd, bw), mmd(x_nsvgd, bw)   
    mmd_dict = np.sqrt([mmd_mmd, mmd_ksd, mmd_svgd, mmd_nsvgd])# mmd_herd, mmd_stein

    ksd_mmd, ksd_ksd, ksd_svgd, ksd_nsvgd = ksd(x_mmd, bw), ksd(x_ksd, bw), ksd(x_svgd, bw),  ksd(x_nsvgd, bw) 
    ksd_dict = np.sqrt([ksd_mmd, ksd_ksd, ksd_svgd, ksd_nsvgd])  # ksd_herd, ksd_stein

    return mmd_dict, ksd_dict#, inte_dict

result_mmd, result_ksd = np.zeros((l,repeat+4,num_output)), np.zeros((l,repeat+4,num_output))    
for j in range(l):
    for i in range(repeat):
        result_mmd[j,i,:], result_ksd[j,i,:] = evaluation(n_dict[j],i,bw)
    result_mmd[j,repeat,:] = [result_mmd[j,0:repeat,k].mean() for k in range(num_output)]
    result_ksd[j,repeat,:] = [result_ksd[j,0:repeat,k].mean() for k in range(num_output)]
    
    result_mmd[j,repeat+1,:] = [result_mmd[j,0:repeat,k].min() for k in range(num_output)]
    result_ksd[j,repeat+1,:] = [result_ksd[j,0:repeat,k].min() for k in range(num_output)]    
    
    result_mmd[j,repeat+2,:] = [result_mmd[j,0:repeat,k].max() for k in range(num_output)]
    result_ksd[j,repeat+2,:] = [result_ksd[j,0:repeat,k].max() for k in range(num_output)]

    result_mmd[j,repeat+3,:] = [np.std(result_mmd[j,0:repeat,k]) for k in range(num_output)]
    result_ksd[j,repeat+3,:] = [np.std(result_ksd[j,0:repeat,k]) for k in range(num_output)]  
 
repeat_iid = 100
iid_mmd, iid_ksd = np.zeros((l,repeat_iid+2)), np.zeros((l,repeat_iid+2))
for c,n in enumerate(n_dict):
    for i in range(repeat_iid):
        x_iid_test = torch.randn(n,p)/math.sqrt(p)
        iid_ksd[c,i] = np.sqrt(ksd(x_iid_test, bw))
        iid_mmd[c,i] = np.sqrt(mmd(x_iid_test, bw))#abs(val_inte - integral(x_iid_test)/n)

for j in range(l):
    iid_mmd[j,repeat_iid] = iid_mmd[j,0:repeat_iid].mean()
    iid_mmd[j,repeat_iid+1] = np.std(iid_mmd[j,0:repeat_iid])
    iid_ksd[j,repeat_iid] = iid_ksd[j,0:repeat_iid].mean()
    iid_ksd[j,repeat_iid+1] = np.std(iid_ksd[j,0:repeat_iid])


methods = ["mmd-lbfgs","ksd-lbfgs","SVGD","NSVGD","mmd herding","stein point"]
colors = ["darkblue","lightblue","red","darkred","grey","cadetblue"]

########## KSD plot
figure(figsize=(5, 5), dpi=100)
for i in range(num_output):
    m,b = np.polyfit(np.log(n_dict), np.log(result_ksd[:,repeat,i]), 1)
    plt.plot(np.log(n_dict), m*np.log(n_dict)+ b, linestyle=(0,(5,1)), color=colors[i])
    plt.plot(np.log(n_dict), np.log(result_ksd[:,repeat,i]), '.', color=colors[i], label=methods[i])
    
plt.axline((np.log(n_dict)[0], np.log(iid_ksd[0,repeat_iid])), slope=-0.5, color = 'black')
plt.plot(np.log(n_dict), np.log(iid_ksd[:,repeat_iid]), '.', color="black", label="i.i.d.")
ax = plt.gca()
ax.set_aspect(1)
plt.xlabel("log(n)", fontsize=20)
plt.ylabel("log(ksd)", fontsize=18)
plt.grid()
plt.legend(loc = 'upper right', prop={'size': 8})
plt.title("2d", fontsize=20)
plt.savefig('2d_ksd_plot.pdf',bbox_inches='tight')

figure(figsize=(5, 5), dpi=100)
for i in range(num_output):
    m,b = np.polyfit(np.log(n_dict), np.log(result_mmd[:,repeat,i]), 1)
    y = np.log(result_mmd[:,repeat,i])
    plt.plot(np.log(n_dict), m*np.log(n_dict)+ b, linestyle=(0,(5,1)), color=colors[i])
    plt.plot(np.log(n_dict), np.log(result_mmd[:,repeat,i]), '.', color=colors[i], label=methods[i])


########## MMD plot
plt.axline((np.log(n_dict)[0], np.log(iid_mmd[0,repeat_iid])), slope=-0.5, color = 'black')
plt.plot(np.log(n_dict), np.log(iid_mmd[:,repeat_iid]), '.', color="black", label="i.i.d.")
ax = plt.gca()
ax.set_aspect(1)
plt.xlabel("log(n)", fontsize=20)
plt.ylabel("log(mmd)", fontsize=18)
plt.grid()
plt.legend(loc = 'upper right', prop={'size': 8})
plt.title("2d", fontsize=20)
plt.savefig('2d_mmd_plot.pdf',bbox_inches='tight')
for i in range(num_output):
    slope_ksd, _, r, _, _ = scipy.stats.linregress(np.log(n_dict), np.log(result_ksd[:,0,i])/2)
    print("ksd of "+methods[i], slope_ksd, r**2)
    slope_mmd, _, r, _, _ = scipy.stats.linregress(np.log(n_dict), np.log(result_mmd[:,0,i])/2)
    print("mmd of "+methods[i], slope_mmd, r**2)
    
############# change MMD bandwidth
bw_list = [1,0.49,0.09]
def plot_slope(b, ax):
    bw = b
    print(bw)
    n_dict = [4,8,16,32,64, 128, 256, 512, 1024, 1600]
    l = len(n_dict)
    repeat = 3
    num_output = 6
    c = 0
    result_mmd = np.zeros((l,repeat+3,num_output))
    result_ksd = np.zeros((l,repeat+3,num_output))    
    for j in range(l):
        for i in range(repeat):
            result_mmd[j,i,:], result_ksd[j,i,:] = evaluation(n_dict[j],i,bw)
        result_mmd[j,repeat,:] = [result_mmd[j,0:repeat,k].mean() for k in range(num_output)]
        result_ksd[j,repeat,:] = [result_ksd[j,0:repeat,k].mean() for k in range(num_output)]
        
        result_mmd[j,repeat+1,:] = [result_mmd[j,0:repeat,k].min() for k in range(num_output)]
        result_ksd[j,repeat+1,:] = [result_ksd[j,0:repeat,k].min() for k in range(num_output)]    
        
        result_mmd[j,repeat+2,:] = [result_mmd[j,0:repeat,k].max() for k in range(num_output)]
        result_ksd[j,repeat+2,:] = [result_ksd[j,0:repeat,k].max() for k in range(num_output)]
            
    iid_mmd = np.zeros(l)
    iid_ksd = np.zeros(l)
    c = 0
    repeat_iid = 100
    for n in n_dict:
        for i in range(repeat_iid):
            x_iid_test = torch.randn(n,p)/math.sqrt(p)
            iid_ksd[c] += ksd(x_iid_test, bw)
            iid_mmd[c] += mmd(x_iid_test, bw)
        iid_ksd[c] /= repeat_iid
        iid_mmd[c] /= repeat_iid
        c += 1
    
    methods = ["mmd-lbfgs","ksd-lbfgs","SVGD","NSVGD","mmd herding","stein point"]
    colors = ["darkblue","lightblue","red","darkred","grey","cadetblue"]
    
    figure(figsize=(5, 5), dpi=100)
    for i in range(len(methods)):
        m,b = np.polyfit(np.log(n_dict), np.log(result_mmd[:,repeat,i]), 1)
        ax.plot(np.log(n_dict), m*np.log(n_dict)+ b, linestyle=(0,(5,1)), color=colors[i])
        ax.plot(np.log(n_dict), np.log(result_mmd[:,repeat,i]), '.', color=colors[i], label=methods[i])
    ax.axline((np.log(n_dict)[0], np.log(iid_mmd)[0]/2), slope=-0.5, color = 'black')
    ax.plot(np.log(n_dict), np.log(iid_mmd)/2, '.', color="black", label="i.i.d.")
    ax.set_aspect(1)
    ax.grid()

fig, axes = plt.subplots(ncols=3, nrows=1, sharex=True, sharey=True, figsize=(7,4), dpi = 200)
for i, ax in enumerate(axes.flatten()):
    plot_slope(bw_list[i], ax)

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower right', bbox_to_anchor=(0.87, 0.14), prop={'size': 8})
        
fig.text(0.5, 0.02, 'log(n)', ha='center',fontsize=14)
fig.text(0.04, 0.5, 'log(mmd)', va='center', rotation='vertical',fontsize=14)
plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.show()
