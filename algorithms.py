import torch
import math
import numpy as np
from time import time
from scipy.optimize import minimize 

def svgd(x0, score, step, max_iter=1000, bw=-1, tol=1e-5, verbose=False,
         store=False, backend='auto', kernel = 'Laplace', ada = False):    
    x_type = type(x0)
    width = bw
    if backend == 'auto':
        if x_type is np.ndarray:
            backend = 'numpy'
        elif x_type is torch.Tensor:
            backend = 'torch'
    if x_type not in [torch.Tensor, np.ndarray]:
        raise TypeError('x0 must be either numpy.ndarray or torch.Tensor '
                        'got {}'.format(x_type))
    if backend not in ['torch', 'numpy', 'auto']:
        raise ValueError('backend must be either numpy or torch, '
                         'got {}'.format(backend))
    if backend == 'torch' and x_type is np.ndarray:
        raise TypeError('Wrong backend')
    if backend == 'numpy' and x_type is torch.Tensor:
        raise TypeError('Wrong backend')
    if backend == 'torch':
        x = x0.detach().clone()
    else:
        x = torch.from_numpy(x0)
    n_samples, n_features = x.shape
    if store:
        storage = []
        t0 = time()
        timer = []
    alpha = 0.9
    fudge_factor = 1e-6
    historical_grad = 0
    for i in range(max_iter):
        if store:
            storage.append(x.clone())
            timer.append(time() - t0)
        d = (x[:, None, :] - x[None, :, :])
        dists = (d ** 2).sum(axis=-1) #dists: ||x_i-x_j||^2 matrix
        if width == -1:
            h = math.sqrt(dists.mean())*n_samples**(-1/(n_features+4))
            if kernel == 'Laplace':
                bw = h
            else:
                bw == h**2

        if kernel == 'Laplace':
            nx = torch.sqrt(dists)
            k = torch.exp(- nx / bw) # Laplace kernel matrix     
            k_der = (d / (nx+ 1e15*torch.eye(n_samples))[:,:,None]) * k[:, :, None] / bw # -Laplace kernel gradient, or directly set diagonal entries to be 0
        else:
            k = torch.exp(- dists / bw / 2) #k: \eta(x_i-x_j) = e^{-||x_i-x_j||^2/(2*bw)}
            k_der = d * k[:, :, None] / bw # -+\nabla \eta(x_i-x_j)       
            
        scores_x = score(x) #-x
        ks = k.mm(scores_x) #ks: [ -\sum \eta(x_i-x_j) \nabla U(x_j) ]
        kd = k_der.sum(axis=0)
        direction = (ks - kd) / n_samples
        
        if ada:
            if i == 0:
                historical_grad = historical_grad + direction ** 2
            else:
                historical_grad = alpha * historical_grad + (1 - alpha) * (direction ** 2)
            adj_grad = np.divide(direction, fudge_factor+np.sqrt(historical_grad))
            x += step * adj_grad
        else:
            x += step * direction
    if store:
        return x, storage, timer
    return x

def nsvgd(x0, score, step, max_iter=1000, bw=-1, tol=1e-5, alpha = 0.5, h = 1, verbose=False,
          store=False, backend='auto', kernel = 'Laplace', ada = False):
    x_type = type(x0)
    width = bw
    if backend == 'auto':
        if x_type is np.ndarray:
            backend = 'numpy'
        elif x_type is torch.Tensor:
            backend = 'torch'
    if x_type not in [torch.Tensor, np.ndarray]:
        raise TypeError('x0 must be either numpy.ndarray or torch.Tensor '
                        'got {}'.format(x_type))
    if backend not in ['torch', 'numpy', 'auto']:
        raise ValueError('backend must be either numpy or torch, '
                          'got {}'.format(backend))
    if backend == 'torch' and x_type is np.ndarray:
        raise TypeError('Wrong backend')
    if backend == 'numpy' and x_type is torch.Tensor:
        raise TypeError('Wrong backend')
    if backend == 'torch':
        x = x0.detach().clone()
    else:
        x = torch.from_numpy(x0)
        
    n_samples, n_features = x.shape    
    if store:
        storage = []
        timer = []
        t0 = time()
    alpha = 0.9
    fudge_factor = 1e-6
    historical_grad = 0
    for i in range(max_iter):
        if store:
            if backend == 'torch':
                storage.append(x.clone())
            else:
                storage.append(x.copy())
                timer.append(time() - t0)
        d = x[:, None, :] - x[None, :, :] # x_i - x_j
        dists = (d ** 2).sum(axis=-1) #dists: ||x_i-x_j||^2 matrix
        h = 1.059*math.sqrt(dists.mean())*n_samples**(-1/(n_features+4))
        
        if width == -1:
            if kernel == 'Laplace':
                bw = h
            else:
                bw = h**2

        if kernel == 'Laplace':
            nx = torch.sqrt(dists)
            k = torch.exp(- nx / bw) # Laplace kernel matrix   
            kh = torch.exp(- nx / h)  # Laplace KDE kernel matrix
            k_der = (d / (nx+ 1e15*torch.eye(n_samples))[:,:,None]) * k[:, :, None] / bw # -Laplace kernel gradient, or set diagonal entries to be 0
            kh_der = (d / (nx+ 1e15*torch.eye(n_samples))[:,:,None]) * kh[:, :, None] / h # -Laplace KDE kernel gradient
        else:
            k = torch.exp(- dists / bw / 2) #k: \eta(x_i-x_j) = e^{-||x_i-x_j||^2/(2*bw)}
            kh = torch.exp(- dists /(h**2)/ 2)  # Gaussian KDE kernel matrix   
            k_der = d * k[:, :, None] / bw # -+\nabla \eta(x_i-x_j)        
            kh_der = d * kh[:, :, None] / (h**2) 

        scores_x = score(x)
        rho_eta = kh.sum(axis=0)/ n_samples #/ (h**n_features) 
        rho_mtx = (rho_eta[:,None].mm(rho_eta[None,:])).pow(alpha).reciprocal()
        term1 = (rho_mtx[:,:,None] * k_der).sum(axis = 0)
        term2 = (rho_mtx * k).mm(alpha * (kh_der.sum(axis = 0)/ n_samples) / rho_eta[:,None])
        term3 = -(rho_mtx * k).mm(scores_x)
        direction = -(term1 + term2 + term3) / n_samples 
        
        if ada:
            if i == 0:
                historical_grad = historical_grad + direction ** 2
            else:
                historical_grad = alpha * historical_grad + (1 - alpha) * (direction ** 2)
            adj_grad = np.divide(direction, fudge_factor+np.sqrt(historical_grad))
            x += step * adj_grad
        else:
            x += step * direction
    if store:
        return x, storage, timer
    return x


########################   Herding
p=2 #dimension

def expKer(x,samples,gamma):    # here we consider Gaussian target distribution instead of empirical samples
    return np.exp(-np.linalg.norm(x)**2 / (2*(gamma+1/p)))*(gamma/(gamma+1/p))**(p/2)
def sumKer(x,xss,numSSsoFar,gamma):

    total=0
    k=np.zeros(numSSsoFar)
    #calculate sof of kernels
    for i in range(numSSsoFar):
        k[i] = np.exp(-np.linalg.norm(x-xss[i,:])**2 /gamma**2/2)
    total = np.sum(k)
    s = total/(numSSsoFar+1)
    return s

def herd(samples,totalSS,gamma):
    numDim = samples.shape[1]
    numSamples = samples.shape[0]
    
    #init vars
    gradientFail = 0; #count when optimization fails, debugging
    xss = np.zeros((totalSS,numDim)) #open space in mem for array of super samples
    i=1
    minBound = np.min(samples)
    maxBound = np.max(samples)
    bestSeed = np.zeros(numDim)
    
    while i< totalSS:
        f = lambda x: -expKer(x,samples,gamma)+sumKer(x,xss,i,gamma)
        results = minimize(f,
                           bestSeed,
                           method='nelder-mead',
                           options={'xtol': 1e-8, 'disp': False})
        
        if np.min(results.x) < minBound or np.max(results.x) > maxBound:
            bestSeed=samples[np.random.choice(numSamples)]
            gradientFail=gradientFail+1
            continue
        
        seed=np.array([])
        for j in range(i):
            seed = np.append(seed,-expKer(xss[j,:],samples,gamma)+sumKer(xss[j,:],xss,i,gamma))
        bestSeedIdx = np.argmin(seed)
        bestSeed=xss[bestSeedIdx,:]

        xss[i,:]=results.x
        
        i += 1
    return xss

########################   Stein points
def gaussian_stein_kernel(x, y, scores_x, scores_y, sigma, return_kernel=False):
    _, p = x.shape
    d = x[:, None, :] - y[None, :, :]
    dists = (d ** 2).sum(axis=-1)
    k = torch.exp(-dists / sigma / 2)
    scalars = scores_x.mm(scores_y.T)
    scores_diffs = scores_x[:, None, :] - scores_y[None, :, :]
    diffs = (d * scores_diffs).sum(axis=-1)
    der2 = p - dists / sigma
    stein_kernel = k * (scalars + diffs / sigma + der2 / sigma)
    if return_kernel:
        return stein_kernel, k
    return stein_kernel

def gaussian_stein_kernel_diag(x, scores_x, sigma, return_kernel=False):
    _, p = x.shape
    scalars = (scores_x ** 2).sum(axis=-1)
    stein_kernel = scalars + p / sigma
    return stein_kernel

def stein_points(n_samples, score, space_limits, grid_size=200, bw=1):
    def loss(x_grid, x_part):
        score_grid = score(x_grid)
        if x_part is not None:
            score_part = score(x_part)
            K_part = gaussian_stein_kernel(x_grid, x_part, score_grid, score_part, sigma=bw).sum(axis=1)
        else:
            K_part = 0
        K_self = gaussian_stein_kernel_diag(x_grid, score_grid, sigma=bw, return_kernel=False)
        return K_self + K_part
    d = len(space_limits)
    x_part = None
    grid = torch.meshgrid([torch.linspace(m, M, grid_size) for m, M in space_limits])
    x_grid = torch.stack([x.ravel() for x in grid], -1)
    with torch.no_grad():
        for n_iter in range(n_samples):
            losses = loss(x_grid, x_part)
            best_loc = torch.argmin(losses)
            new_part = x_grid[best_loc]
            x_new = torch.zeros(n_iter + 1, d)
            x_new[0] = new_part
            if x_part is not None:
                x_new[1:] = x_part
            x_part = x_new
    return x_part
