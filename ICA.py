import seaborn as sns
import torch
from algorithms import nsvgd, svgd
import matplotlib.pyplot as plt
import numpy as np
import time
import math
from matplotlib.pyplot import figure

def amari_distance(W, A):
    """
    Computes the Amari distance between two matrices W and A.
    It cancels when WA is a permutation and scale matrix.
    Parameters
    ----------
    W : ndarray, shape (n_features, n_features)
        Input matrix
    A : ndarray, shape (n_features, n_features)
        Input matrix
    Returns
    -------
    d : float
        The Amari distance
    """
    P = np.dot(W, A)

    def s(r):
        return np.sum(np.sum(r ** 2, axis=1) / np.max(r ** 2, axis=1) - 1)

    return (s(np.abs(P)) + s(np.abs(P.T))) / (2 * P.shape[0])


def one_expe(n, p, sigma, bw, bw_exp, n_samples, i):
    torch.manual_seed((i+1)*100)
    W = sigma * np.random.randn(p, p)
    A = np.linalg.pinv(W)
    S = np.random.laplace(size=(p, n))
    X = np.dot(A, S)
    X = torch.tensor(X, dtype=torch.float)

    def score(w):
        N, _ = w.shape
        w_list = w.reshape(N, p, p)
        z = w_list.matmul(X)
        psi = torch.tanh(z).matmul(X.t()) / n - torch.inverse(
            w_list
        ).transpose(-1, -2)
        sc = -psi - w_list / sigma
        return sc.reshape(N, p ** 2)
    
    x = torch.randn(n_samples, p ** 2)
    x_svgd = svgd(x.clone(), score, 0.05, bw=0.5, max_iter=3000, kernel = 'Gaussian')#when p=8 we take bw = 1
    x_nsvgd = nsvgd(x.clone(), score, 0.05, bw=0.5, max_iter=1000) 
    score_svgd = torch.norm(score(x_svgd)).item()
    score_nsvgd = torch.norm(score(x_nsvgd)).item()
    score_random = torch.norm(score(x)).item()
    w_svgd = (x_svgd.reshape(n_samples, p, p)).detach().numpy()
    w_nsvgd = (x_nsvgd.reshape(n_samples, p, p)).detach().numpy()

    amari_svgd = np.sort([amari_distance(w, A) for w in w_svgd])
    amari_nsvgd = np.sort([amari_distance(w, A) for w in w_nsvgd])
    amari_random = np.sort([amari_distance(np.random.randn(p, p), A) for w in w_svgd])
    return (
        amari_svgd,
        amari_nsvgd,
        amari_random,
        score_svgd,
        score_nsvgd,
        score_random,
    )


p_list = [4] #[8]
n = 1000
sigma = 1
bw = 0.1
bw_exp = 1
n_samples = 50
n_tries = 10

f, axes = plt.subplots(1, len(p_list), figsize = (3*len(p_list),3))#, sharey=True)
d_save = {}
for j,p in enumerate(p_list):
    print(p)
    d_save[p] = {}
    amari_ksds = []
    amari_svgds = []
    amari_nsvgds = []
    amari_randoms = []
    score_ksds = []
    score_svgds = []
    score_nsvgds = []
    score_randoms = []
    for i in range(n_tries):
        (
            amari_svgd,
            amari_nsvgd,
            amari_random,
            score_svgd,
            score_nsvgd,
            score_random,
        ) = one_expe(n, p, sigma, bw, bw_exp, n_samples, i)
        
        amari_svgds.append(amari_svgd)
        amari_nsvgds.append(amari_nsvgd)
        amari_randoms.append(amari_random)
        
        score_svgds.append(score_svgd)
        score_nsvgds.append(score_nsvgd)
        score_randoms.append(score_random)
        
    plt.plot(np.sort(np.ravel(amari_svgds)), label="svgd", color = 'orange')
    plt.plot(np.sort(np.ravel(amari_nsvgds)), label="nsvgd", color = 'green')
    plt.plot(np.sort(np.ravel(amari_randoms)), label="random", color = 'red')
    plt.yscale("log")
    plt.title('p = %i' %p)
    plt.legend()
    # plt.show()

    figure(figsize=(4, 2.2), dpi=100)
    sns.distplot(np.ravel(amari_svgds), color = 'orange')
    sns.kdeplot(np.ravel(amari_svgds), color = 'orange', label='SVGD',linewidth=2)
    sns.distplot(np.ravel(amari_nsvgds), color = 'lightblue')
    sns.kdeplot(np.ravel(amari_nsvgds), color = 'blue', label='NSVGD',linewidth=2)
    sns.distplot(np.ravel(amari_randoms), color = 'grey')
    sns.kdeplot(np.ravel(amari_randoms),color = 'black', label='i.i.d.',linewidth=2)
    plt.xlabel("Amari distance, p=8",fontsize=14)
    plt.ylabel("density",fontsize=14)
    plt.legend(prop={'size': 13})
    plt.savefig("Amari_p=8.pdf",bbox_inches='tight')
