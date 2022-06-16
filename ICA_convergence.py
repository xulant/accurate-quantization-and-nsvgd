import torch
from algorithms import nsvgd, svgd
import matplotlib.pyplot as plt
import numpy as np
from numpy import random, linalg
from time import time

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


def random_ball(num_points, dimension, radius=1):
    random_directions = random.normal(size=(dimension,num_points))
    random_directions /= linalg.norm(random_directions, axis=0)
    random_radii = random.random(num_points) ** (1/dimension)
    return radius * (random_directions * random_radii).T


def one_expe(n, p, sigma, bw, bw_exp, step_svgd, step_nsvgd, n_samples, i):
    print(i)
    torch.manual_seed((i+1)*100)
    W = sigma * np.random.randn(p, p)
    A = np.linalg.pinv(W)
    S = np.random.laplace(size=(p, n))
    X = torch.tensor(np.dot(A, S), dtype=torch.float)

    def score(w):
        N, _ = w.shape
        w_list = w.reshape(N, p, p)
        z = w_list.matmul(X)
        psi = torch.tanh(z).matmul(X.t()) / n - torch.inverse(
            w_list
        ).transpose(-1, -2)
        sc = -psi - w_list / sigma
        return sc.reshape(N, p ** 2)
    
    def mean_Amari(x):
        w_x = (x.reshape(n_samples, p, p)).detach().numpy()
        return np.mean([amari_distance(w, A) for w in w_x])
    
    x = torch.tensor(random_ball(n_samples, p**2, 10), dtype=torch.float32).detach()#.double()#, dtype=torch.double)#
    #x = torch.randn(n_samples, p ** 2)
    max_iter = 1000

    t_svgds = []
    traj_svgds = []
    for step in step_svgd:
        # print(step)
        t0 = time()
        svgd(x.clone(), score, step, max_iter=max_iter, bw=bw, kernel = 'Gaussian', verbose=True)
        t_svgd = time() - t0
        t_svgds.append(t_svgd)
        x_svgd, traj_svgd, _ = svgd(x.clone(), score, step, max_iter=max_iter, bw=bw, kernel = 'Gaussian', store=True)
        traj_svgds.append(traj_svgd)

    t0 = time()
    nsvgd(x.clone(), score, step_nsvgd, max_iter=max_iter, bw=bw_exp, kernel = 'Gaussian', verbose=True)
    t_nsvgd = time() - t0
    x_nsvgd, traj_nsvgd, _ = nsvgd(
        x.clone(), score, step_nsvgd, max_iter=max_iter, bw=bw_exp, kernel = 'Gaussian', store=True
    )
    
    amari_svgd = np.array(
        [
            [mean_Amari(x) for x in traj_svgd]
            for traj_svgd in traj_svgds
        ]
    )
    amari_nsvgd = [mean_Amari(x) for x in traj_nsvgd]

    return amari_svgd, amari_nsvgd, t_svgds, t_nsvgd

#p_list = [2]#[2,4,8]
n = 1000
sigma = 1
p = 4
n_samples = 50

bw = 1 #for Gaussian kernel it equals to bandwidth^2
bw_exp = 1
n_expes = 10
time_list = []
steps_svgd = [0.1, 1, 3]#[0.1,1,10]#[0.1, 1, 3]
step_nsvgd = 1

outputs = [
    one_expe(n, p, sigma, bw, bw_exp, steps_svgd, step_nsvgd, n_samples, i) for i in range(n_expes)
]

amari_dict = {}
amari_dict["svgd"] = {}
for i, step in enumerate(steps_svgd):
    amari_dict["svgd"][step] = [op[0][i] for op in outputs]
amari_dict['nsvgd'] = [op[1] for op in outputs]

times_svgds_list = np.array([op[2] for op in outputs]).T
times_nsvgds = np.array([op[3] for op in outputs])

timing = True


def average_curves(times, values):
    times = np.array(times)
    t_max = np.max(times)
    time_grid = np.linspace(0, t_max, 200)
    interp_values = [
        np.interp(time_grid, np.linspace(0, time, len(value)), value)
        for time, value in zip(times, values)
    ]
    return time_grid, np.nanmedian(interp_values, axis=0)#median(interp_values, axis=0)


lw = 2
for step, times_svgds, label, color in zip(
    steps_svgd,
    times_svgds_list,
    ["SVGD, small step", "SVGD, good step", "SVGD, big step"],
    ["gold", "orange", "red"],
):
    t_avg_svgd, plot_svgd = average_curves(
        times_svgds, amari_dict["svgd"][step]
    )
    plt.plot(t_avg_svgd, plot_svgd, color=color, label=label, linewidth=lw)
    
t_avg_nsvgd, plot_nsvgd = average_curves(times_nsvgds, amari_dict["nsvgd"])
lw = 2
plt.plot(t_avg_nsvgd, plot_nsvgd, color="black", label="NSVGD", linewidth=lw)    
plt.axis(xmin=0,xmax=1.0)
plt.yscale("log")
plt.xlabel("Time (s.)")
plt.ylabel("Amari distance(log)")
plt.grid()
#plt.suptitle('2d Gaussian convergence rate, i.i.d. initialization')
#plt.suptitle('ICA Amari distance')
plt.legend(loc = 'upper right')
plt.savefig('ICA_cvg_4.pdf',bbox_inches='tight')
plt.show()
