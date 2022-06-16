import torch
import numpy as np
from ksddescent import ksdd_lbfgs
# from ksddescent.contenders import svgd
from algorithms import svgd, nsvgd
from scipy.io import loadmat
import argparse
datasets = ['banana', 'breast_cancer', 'diabetis',
            'flare_solar', 'german', 'heart', 'image',
            'ringnorm', 'splice', 'thyroid', 'titanic',
            'twonorm', 'waveform']


def get_dataset(splitIdx=1, dataName='banana'):
    data = loadmat('benchmarks.mat')
    dataset = data[dataName]
    X = dataset['x'][0, 0]
    y = dataset['t'][0, 0][:, 0]
    y = (y + 1) // 2
    train_idx = dataset['train'][0, 0][splitIdx] - 1
    test_idx = dataset['test'][0, 0][splitIdx] - 1
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]
    X_train = np.hstack([X_train, np.ones([X_train.shape[0], 1])])
    X_test = np.hstack([X_test, np.ones([X_test.shape[0], 1])])
    return X_train, y_train, X_test, y_test


def score_and_eval(X_train, y_train, X_test, y_test):
    b = 0.01
    X = torch.tensor(X_train).float()
    y = torch.tensor(y_train).long()
    X_test = torch.tensor(X_test).float()
    y_test = torch.tensor(y_test).long()
    n, p = X.shape

    def score(theta, beta=1.):
        alpha = torch.exp(theta[:, 0])
        w = theta[:, 1:]
        r = w.mm(X.t()).exp()
        invr = 1/(1+r)
        score = torch.zeros_like(theta)
        score_w = - w * alpha[:, None] + (invr + y-1).mm(X)
        score_alpha = -(w ** 2).sum(axis=-1) / 2 - b + 1 / alpha
        score[:, 0] = score_alpha
        score[:, 1:] = score_w
        return beta * score

    def evaluation(theta):
        w = theta[:, 1:]
        inprods = w.mm(X_test.t())
        prob = (y_test[None, :] * inprods).exp() / (1 + torch.exp(inprods))
        prob = torch.mean(prob, axis=0)
        acc = torch.sum(prob > 0.5) / len(prob)
        llh = torch.mean(torch.log(prob))
        return [acc.item(), llh.item()]

    return score, evaluation


def run(split_idx, dataname, n_particles=10, n_tries=3):

    X_train, y_train, X_test, y_test = get_dataset(split_idx, dataname)
    print('\n dataset: ',dataname)
    #print(dataname, split_idx, n_particles, n_tries, X_train.shape)

    score, evaluation = score_and_eval(X_train, y_train, X_test, y_test)
    bws = np.logspace(-2, 2, 6)
    # print(bws)
    accs_svgd = []
    accs_Nsvgd = []
    accs_ksd = []
    
    for bw in bws:
        # print(bw)
        accs_svgd_bw = []
        accs_Nsvgd_bw = []
        accs_ksd_bw = []
        for _ in range(n_tries):
            
            _, p = X_train.shape
            w0 = .1 * torch.randn(n_particles, p + 1)
            print(p)
            w_svgd = svgd(w0.clone(), score, step=.0001, max_iter=1000, bw=bw, verbose=False)
            w_Nsvgd = nsvgd(w0.clone(), score, step=.0001, max_iter=1000, bw=bw, verbose=False)
            w_star = ksdd_lbfgs(w0.clone(), score, bw=bw, kernel='gaussian')
            
            acc_svgd, _ = evaluation(w_svgd)
            accs_svgd_bw.append(acc_svgd)
            acc_Nsvgd, _ = evaluation(w_Nsvgd)
            accs_Nsvgd_bw.append(acc_Nsvgd)
            acc_ksd, _ = evaluation(w_star)
            accs_ksd_bw.append(acc_ksd)
            
        accs_svgd.append(np.mean(accs_svgd_bw))
        accs_Nsvgd.append(np.mean(accs_Nsvgd_bw))
        accs_ksd.append(np.mean(accs_ksd_bw))
    
    print(1)
    print('SVGD : %.2f' % np.max(accs_svgd))
    print('Normalized SVGD : %.2f' % np.max(accs_Nsvgd))
    print('KSDD : %.2f' % np.max(accs_ksd))
    np.save('results/logreg/%s_%d_%d_%d.npy' % (dataname, split_idx, n_particles, n_tries),
            np.array([accs_svgd, accs_Nsvgd, accs_ksd]))


parser = argparse.ArgumentParser()
parser.add_argument('--batch-idx', type=int, default=0)
parser.add_argument('--particles', type=int, default=10)
parser.add_argument('--tries', type=int, default=5)
args = parser.parse_args()
for dataname in datasets:
    try:
        run(args.batch_idx, dataname, args.particles, args.tries)
    except:
        pass
