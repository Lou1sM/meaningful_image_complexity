import torch
from dl_utils.tensor_funcs import numpyify
import sklearn.datasets as data
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import ToTensor


class ComplexityMeasurer():
    def __init__(self,x,ks,verbose):
        self.x = x
        self.ks = ks
        self.verbose = verbose
        self.get_smallest_increment(x)
        pass

    def interpret(self):
        total_num_clusters = 0
        print(f'\nim shape: {self.x.shape}')
        while True:
            num_clusters_at_this_level, dl = self.mdl_cluster()
            if self.verbose:
                print('num clusters at this level is', num_clusters_at_this_level)
            if num_clusters_at_this_level == 1:
                return total_num_clusters
            total_num_clusters += num_clusters_at_this_level
            self.x = apply_random_conv_layer(self.x)
            if self.verbose:
                print(f'applying cl to make im size {self.x.shape}')
            if self.x.shape[0] < 4:
                return total_num_clusters
        return total_num_clusters

    def get_smallest_increment(self,x):
        sx = sorted(x.flatten())
        increments = [sx2-sx1 for sx1,sx2 in zip(sx[:-1],sx[1:])]
        self.prec = min([item for item in increments if item != 0])

    def log_model_prob(self,dpoint,cluster_label):
        pc = self.model.precisions_cholesky_[cluster_label]
        m = self.model.means_[cluster_label]
        mahalanobis_distance = sqeuclidean(np.matmul(np.transpose(pc),(dpoint-m)))
        ml_in_nats = -0.5 * (3*np.log(2*np.pi) + mahalanobis_distance) + np.log(np.linalg.det(pc))
        return np.log2(np.e) * ml_in_nats

    def log_model_probs_for_dset(self,x):
        dpoints_and_labels = zip(x,self.cluster_labels)
        probs = [self.log_model_prob(x_item,cl).item() for x_item,cl in dpoints_and_labels]
        return torch.tensor(probs)

    def mdl_cluster(self):
        x = self.x.reshape(-1,self.x.shape[-1])
        assert x.ndim == 2
        N,nz = x.shape
        len_of_each_cluster = nz + (nz*(nz+1)/2) * np.log2((x.max() - x.min())/self.prec)
        len_of_outlier = nz * np.log2(x.max() - x.min())
        best_dl = np.inf
        best_nc = -1
        for nc in range(1,11):
            self.model = GMM(nc)
            self.cluster_labels = self.model.fit_predict(x)
            model_len = nc*(len_of_each_cluster)
            indices_len = N * np.log2(nc)
            neg_log_probs = -self.log_model_probs_for_dset(x)
            outliers = neg_log_probs>len_of_outlier
            residual_errors = neg_log_probs[~outliers].sum()
            len_outliers = len_of_outlier * outliers.sum()
            total_description_len = model_len + indices_len + residual_errors + len_outliers
            if self.verbose:
                print(f'{nc}\ttotal: {total_description_len:.2f}\tmodel: {model_len:.2f}\terror: {residual_errors:.2f}\tindices: {indices_len:.2f}\toutliers: {outliers.sum()} {len_outliers:.2f}')
            if total_description_len < best_dl:
                best_dl = total_description_len
                best_nc = nc
        if self.verbose:
            print(f'best dl is {best_dl:.3f} with {best_nc} clusters')
        return best_nc, best_dl

def torch_min(t,val):
    return torch.minimum(t,val*torch.ones_like(t))

def apply_random_conv_layer(x):
    torch_x = torch.tensor(x).transpose(0,2).float()
    if torch_x.ndim == 3:
        torch_x = torch_x.unsqueeze(0)
    nin = torch_x.shape[1]
    cnvl = nn.Conv2d(nin, 2*nin, 3, device=torch_x.device)
    torch_x = cnvl(torch_x)
    torch_x = F.max_pool2d(torch_x,2)
    torch_x = F.relu(torch_x)
    return numpyify(torch_x.squeeze(0).transpose(0,2))

sqeuclidean = lambda x: np.inner(x,x)
blobs, _ = data.make_blobs(n_samples=500, centers=[(-0.75,2.25), (-.75,-.75),(2,-.75),(1.0, 2.0)], cluster_std=0.1)
dset = torchvision.datasets.CIFAR10(root='/home/louis/datasets',download=True,train=True,transform=ToTensor())
for i in range(10):
    im = dset.data[i]/255
    comp_meas = ComplexityMeasurer(im,ks=4,verbose=False)
    assembly_num = comp_meas.interpret()
    print(f'Assemby num is {assembly_num}')
nonim = np.random.rand(*im.shape)*im.max()
comp_meas = ComplexityMeasurer(nonim,ks=4,verbose=False)
assembly_num = comp_meas.interpret()
print(assembly_num)
