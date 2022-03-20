import torch
from math import erf
import sklearn.datasets as data
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import ToTensor


class ComplexityMeasurer():
    def __init__(self,x):
        self.x = x
        self.get_smallest_increment(x)
        pass

    def interpret(self):
        total_num_clusters = 0
        while True:
            num_clusters_at_this_level, dl = self.mdl_cluster()
            print('num clusters at this level is', num_clusters_at_this_level)
            if num_clusters_at_this_level == 1:
                return total_num_clusters
            total_num_clusters += num_clusters_at_this_level
            self.x = apply_random_conv_layer(self.x)
            print(f'applying cl to make im size {self.x.shape}')
        return total_num_clusters

    def get_smallest_increment(self,x):
        sx = sorted(x.flatten())
        increments = [sx2-sx1 for sx1,sx2 in zip(sx[:-1],sx[1:])]
        #self.prec = min([item for item in increments if item != 0])
        self.prec = 0.1

    def log_model_prob(self,dpoint,cluster_label):
        pc = self.model.precisions_cholesky_[cluster_label]
        m = self.model.means_[cluster_label]
        mahalanobis_distance = np.matmul(np.transpose(pc),(dpoint-m)).norm()**2
        ml_in_nats = -0.5 * (3*np.log(2*np.pi) + np.matmul(np.transpose(pc),(dpoint-m)).norm()**2) + np.log(np.linalg.det(pc))
        #ml_in_nats = erf((mahalanobis_distance + self.prec)/2**.5) - erf((mahalanobis_distance - self.prec)/2**.5)
        return np.log2(np.e) * ml_in_nats

    def log_model_probs_for_dset(self):
        dpoints_and_labels = zip(self.x.flatten(1).transpose(0,1),self.cluster_labels)
        probs = [self.log_model_prob(x_item,cl).item() for x_item,cl in dpoints_and_labels]
        return torch.tensor(probs)

    def mdl_cluster(self):
        x = self.x.flatten(1).transpose(0,1)
        assert x.ndim == 2
        N,nz = x.shape
        len_of_each_cluster = nz + (nz*(nz+1)/2) * np.log2((x.max() - x.min())/self.prec)
        len_of_outlier = nz * np.log2((x.max() - x.min())/self.prec)
        best_dl = np.inf
        best_nc = -1
        for nc in range(1,21):
            self.model = GMM(nc)
            self.cluster_labels = self.model.fit_predict(x)
            model_len = nc*(len_of_each_cluster)
            indices_len = N * np.log2(nc)
            #error = -model.score(x) * N
            log_probs = self.log_model_probs_for_dset()
            residual_errors = -log_probs.sum()
            outliers = -log_probs>len_of_outlier
            #error_or_outlier_lens = torch_min(residual_errors,len_of_outlier).sum()
            #residual_errors = residual_errors_[~outliers].sum()
            #residual_errors = self.model.score(x) * N
            len_outliers = len_of_outlier * outliers.sum()
            total_description_len = model_len + indices_len + residual_errors + len_outliers
            print(f'model len: {model_len:.3f}\tindices len: {indices_len:.3f}\terror: {residual_errors:.3f}\toutliers: {len_outliers:.3f}\ttotal len: {total_description_len:.3f}')
            if total_description_len < best_dl:
                best_dl = total_description_len
                best_nc = nc
        print(f'best dl is {best_dl:.3f} with {best_nc} clusters')
        return best_nc, best_dl

def torch_min(t,val):
    return torch.minimum(t,val*torch.ones_like(t))

def apply_random_conv_layer(x):
    nin = x.shape[0]
    cnvl = nn.Conv2d(nin, 2*nin, 3, device=x.device)
    x = cnvl(x)
    x = F.max_pool2d(x,2)
    x = F.relu(x)
    return x

blobs, _ = data.make_blobs(n_samples=500, centers=[(-0.75,2.25), (-.75,-.75),(2,-.75),(1.0, 2.0)], cluster_std=0.1)
dset = torchvision.datasets.CIFAR10(root='/home/louis/datasets',download=True,train=True,transform=ToTensor())
dloader = torch.utils.data.DataLoader(dset, batch_size=1,shuffle=True, num_workers=2)
im = next(iter(dloader))[0][0]
blobs = torch.tensor(blobs).transpose(0,1)
#comp_meas = ComplexityMeasurer(im)
comp_meas = ComplexityMeasurer(im)
comp_meas.interpret()
