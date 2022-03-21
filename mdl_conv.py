import torch
import argparse
from PIL import Image
from dl_utils.tensor_funcs import numpyify
import sklearn.datasets as data
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import ToTensor
from load_imagenette import load_rand_imagenette_val


class ComplexityMeasurer():
    def __init__(self,x,ks,verbose,ncs_to_check):
        self.x = x
        self.ks = ks
        self.verbose = verbose
        self.ncs_to_check = ncs_to_check
        self.get_smallest_increment(x)
        pass

    def interpret(self):
        total_num_clusters = 0
        highest_meaningful_level = 0
        while True:
            num_clusters_at_this_level, dl = self.mdl_cluster()
            if self.verbose:
                print('num clusters at this level is', num_clusters_at_this_level)
            if num_clusters_at_this_level == 1:
                return total_num_clusters, highest_meaningful_level
            total_num_clusters += num_clusters_at_this_level
            self.x = apply_random_conv_layer(self.x)
            if self.verbose:
                print(f'applying cl to make im size {self.x.shape}')
            if (self.x.shape[0]-1)*(self.x.shape[1]-1) < 20:
                return total_num_clusters, highest_meaningful_level
            highest_meaningful_level += 1

    def get_smallest_increment(self,x):
        sx = sorted(x.flatten())
        increments = [sx2-sx1 for sx1,sx2 in zip(sx[:-1],sx[1:])]
        self.prec = min([item for item in increments if item != 0])

    def mdl_cluster(self):
        patched = patch_averages(self.x)
        x = patched.reshape(-1,patched.shape[-1])
        assert x.ndim == 2
        N,nz = x.shape
        data_range = x.max() - x.min()
        if self.verbose:
            print(f'drange: {data_range:.3f} prec: {self.prec:.3f},nz:{nz}')
        len_of_each_cluster = nz + (nz*(nz+1)/2) * np.log2((x.max() - x.min())/self.prec)
        len_of_outlier = nz * np.log2(x.max() - x.min())
        best_dl = np.inf
        best_nc = -1
        for nc in range(1,self.ncs_to_check):
            self.model = GMM(nc)
            self.cluster_labels = self.model.fit_predict(x)
            model_len = nc*(len_of_each_cluster)
            indices_len = N * np.log2(nc)
            built_in_scores = -self.model._estimate_log_prob(x)[np.arange(len(x)),self.cluster_labels]
            neg_log_probs = built_in_scores * np.log2(np.e)
            outliers = neg_log_probs>len_of_outlier
            residual_errors = neg_log_probs[~outliers].sum()
            len_outliers = len_of_outlier * outliers.sum()
            total_description_len = model_len + indices_len + residual_errors + len_outliers
            if self.verbose:
                print(f'{nc}: {total_description_len:.2f}\tmodel: {model_len:.2f}\terror: {residual_errors:.2f}\tidxs: {indices_len:.2f}\toutliers: {outliers.sum()} {len_outliers:.2f}')
            if total_description_len < best_dl:
                best_dl = total_description_len
                best_nc = nc
        if self.verbose:
            print(f'best dl is {best_dl:.3f} with {best_nc} clusters')
        return best_nc, best_dl

def patch_averages(a):
    padded = np.pad(a,1)[:,:,1:-1]
    summed = padded[:-1,:-1] + padded[:-1,1:] + padded[1:,:-1] + padded[1:,1:]
    return (summed/4)[1:-1,1:-1]

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


parser = argparse.ArgumentParser()
parser.add_argument('--resize',action='store_true')
parser.add_argument('--verbose',action='store_true')
parser.add_argument('--display_images',action='store_true')
parser.add_argument('--dset',type=str,choices=['im','cifar','mnist','rand'],required=True)
parser.add_argument('--num_ims',type=int,default=10)
parser.add_argument('--ncs_to_check',type=int,default=10)
ARGS = parser.parse_args()

if ARGS.dset == 'cifar':
    dset = torchvision.datasets.CIFAR10(root='/home/louis/datasets',download=True,train=True)
elif ARGS.dset == 'mnist':
    dset = torchvision.datasets.MNIST(root='/home/louis/datasets',train=False,download=True)
elif ARGS.dset == 'rand':
    dset = np.random.rand(ARGS.num_ims,224,224,3)
all_assembly_idxs = []
all_levels = []
for i in range(ARGS.num_ims):
    if ARGS.dset == 'im':
        im, label = load_rand_imagenette_val(ARGS.resize)
    elif ARGS.dset == 'rand':
        im = dset[i]
        label = 'none'
    else:
        if ARGS.dset == 'cifar':
            im = dset.data[i]/255
        elif ARGS.dset == 'mnist':
            im = numpyify(dset.data[i].unsqueeze(2))/255
        im = np.array(dset.data[i])/255
        im = np.resize(im,(224,224,3))
        label = dset.targets[i]
    if ARGS.display_images:
        plt.imshow(im);plt.show()
    comp_meas = ComplexityMeasurer(im,ks=4,verbose=ARGS.verbose,ncs_to_check=ARGS.ncs_to_check)
    assembly_idx,level = comp_meas.interpret()
    print(f'Class: {label}\tAssemby num: {assembly_idx}\tLevel: {level}')
    all_assembly_idxs.append(assembly_idx)
    all_levels.append(level)
mean_assembly_idx = np.array(all_assembly_idxs).mean()
mean_level = np.array(all_levels).mean()
print(f'Mean assembly idx: {mean_assembly_idx}')
print(f'Mean level idx: {mean_level}')
