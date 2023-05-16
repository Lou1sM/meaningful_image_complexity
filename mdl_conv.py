from scipy.special import softmax,logsumexp
import sys
from copy import copy
from time import time
from scipy.stats import entropy
from matplotlib.colors import BASE_COLORS
import matplotlib.cm as cm
from dl_utils.misc import scatter_clusters
from dl_utils.tensor_funcs import numpyify, recursive_np_or
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from numpy import log,log2,exp
import torch
import torch.nn as nn
import skfuzzy as fuzz


PALETTE = list(BASE_COLORS.values()) + [(0,0.5,1),(1,0.5,0)]
class ComplexityMeasurer():
    def __init__(self,verbose,ncs_to_check,n_cluster_inits,print_times,
                    display_cluster_label_imgs,compare_to_true_entropy,
                    nz,alg_nz,display_scattered_clusters,cluster_model,
                    subsample,patch_comb_method,num_layers,ass,
                    no_cluster_idxify,info_subsample,**kwargs):

        self.verbose = verbose
        self.num_layers = num_layers
        self.print_times = print_times
        self.cluster_idxify = not no_cluster_idxify
        self.compare_to_true_entropy = compare_to_true_entropy
        self.patch_comb_method = patch_comb_method
        self.subsample = subsample
        self.n_cluster_inits = n_cluster_inits
        self.display_cluster_label_imgs = display_cluster_label_imgs
        self.display_scattered_clusters = display_scattered_clusters
        self.ncs_to_check = ncs_to_check
        self.nz = nz
        self.alg_nz = alg_nz
        self.info_subsample = info_subsample
        self.cluster_model = cluster_model
        self.is_mdl_abl = False
        self.complexity_func = labels_ass_idx if ass else labels_entropy

    def interpret(self,given_x):
        img_start_time = time()
        x = np.copy(given_x)
        total_num_clusters = 0
        all_single_labels_ass_idxs = []
        all_patch_entropys = []
        all_nums_clusters = []
        self.set_smallest_increment(x)
        for layer_being_processed in range(self.num_layers):
            self.layer_being_processed = layer_being_processed
            cluster_start_time = time()
            num_clusters_at_this_level, dl = self.mdl_cluster(x)
            all_nums_clusters.append(num_clusters_at_this_level)
            if self.display_cluster_label_imgs:
                self.viz_cluster_labels()
            if num_clusters_at_this_level <= 1:
                pad_len = self.num_layers - len(all_patch_entropys)
                all_single_labels_ass_idxs += [0]*pad_len
                all_patch_entropys += [0]*pad_len
                break
            single_labels_ass_idx = self.complexity_func(self.best_cluster_labels.flatten())
            if self.print_times:
                print(f'mdl_cluster time: {time()-cluster_start_time:.2f}')
            total_num_clusters += num_clusters_at_this_level
            all_single_labels_ass_idxs.append(single_labels_ass_idx)
            bool_ims_by_c = [(self.best_cluster_labels==c)
                            for c in np.unique(self.best_cluster_labels)]
            one_hot_im = np.stack(bool_ims_by_c,axis=2)
            patch_size = 4*(2**layer_being_processed)
            if self.verbose:
                print(f'patch_size: {patch_size}')
            c_idx_patches = combine_patches(one_hot_im,patch_size,self.patch_comb_method)
            if self.cluster_idxify:
                x = c_idx_patches
            else:
                x = combine_patches(x,patch_size,self.patch_comb_method)
            info_start_time = time()
            patch_entropy = self.info_in_patches(c_idx_patches,self.info_subsample)
            if self.info_subsample != 1 and self.compare_to_true_entropy:
                true_entropy = self.info_in_patches(c_idx_patches,1)
                print(f'true entropy: {true_entropy}, '
                        f'approx at {self.info_subsample}: {patch_entropy}')
            if self.print_times:
                print(f'time to compute entropy: {time()-info_start_time:.2f}')
            all_patch_entropys.append(patch_entropy)
            print(f'{layer_being_processed}: single ent: {single_labels_ass_idx}, patch_ent: {patch_entropy}')
            if sum([x==8 for x in all_nums_clusters]) == 2:
                break
        if self.print_times:
            print(f'image time: {time()-img_start_time:.2f}')
        return all_patch_entropys, all_nums_clusters, all_single_labels_ass_idxs

    def set_smallest_increment(self,x):
        sx = sorted(x.flatten())
        increments = [sx2-sx1 for sx1,sx2 in zip(sx[:-1],sx[1:])]
        self.prec = min([item for item in increments if item != 0])
        #print(f'Setting set_smallest_increment to {self.prec}')

    def project_clusters(self):
        prev_clabs_as_img = np.expand_dims(self.best_cluster_labels,2)
        if self.layer_being_processed==2:
            proj_clusters_as_img = prev_clabs_as_img
        elif self.layer_being_processed==1:
            proj_clusters_as_img = self.apply_conv_layer(prev_clabs_as_img,custom_cnvl=self.dummy_initial_layer)
        else:
            proj_clusters_as_img = self.apply_conv_layer(prev_clabs_as_img,custom_cnvl=self.dummy_downsample_rlayer)
        return proj_clusters_as_img.flatten().round().astype(int)

    def mdl_cluster(self,x_as_img,fixed_nc=-1):
        full_x = x_as_img
        full_x = full_x.reshape(-1,full_x.shape[-1])
        if self.subsample != 1:
            num_to_subsample = int(len(full_x) * self.subsample)
            subsample_idxs = np.random.choice(len(full_x),size=num_to_subsample,replace=False)
            x = full_x[subsample_idxs]
        else:
            x = full_x
        assert x.ndim == 2
        N,nz = x.shape
        if self.verbose:
            print('number pixel-likes to cluster:', N)
            print('number different pixel-likes:', nz)
        if nz > 50:
            x = PCA(50).fit_transform(x)
        if nz > 3:
            dim_red_start_time = time()
            x = PCA(self.nz).fit_transform(x)
            if self.print_times:
                print(f'dim red time: {time()-dim_red_start_time:.2f}')
        N,nz = x.shape
        data_range = x.max() - x.min()
        self.len_of_each_cluster = 2 * nz * (log2(data_range) + 32) # Float precision
        self.len_of_outlier = nz * log2(data_range) #No float prec. cuz will have that either way
        best_dl = N*self.len_of_outlier # Could call everything an outlier and have no clusters
        best_nc = 0
        nc_start_times = []
        ncs_to_check = [5] if self.is_mdl_abl else range(1,self.ncs_to_check+1)
        for nc in ncs_to_check:
            nc_start_times.append(time())
            found_nc = self.cluster(x,nc)
            if found_nc == nc-1:
                print(f"only found {nc-1} clusters when looking for {nc}, terminating here"); break
            if self.verbose:
                print(( f'{nc} {self.dl/N:.3f}\tMod: {self.model_len/N:.3f}\t'
                        f'Err: {self.residuals/N:.3f}\t'
                        f'Idxs: {self.idxs_len/N:.3f}\t'
                        f'O: {self.outliers.sum()} {self.len_outliers/N:.3f}\t'))
            if self.dl < best_dl or self.is_mdl_abl:
                best_dl = self.dl
                best_nc = nc
                self.best_cluster_labels = self.cluster_labels.reshape(*x_as_img.shape[:-1])
        if self.print_times:
            nc_times = [nc_start_times[i+1] - ncs for i,ncs in enumerate(nc_start_times[:-1])]
            tot_c_time = f' tot: {nc_start_times[-1] - nc_start_times[0]:.2f}'
            print(' '.join([f'{i}: {s:.2f}' for i,s in enumerate(nc_times)]) + tot_c_time)
        if self.subsample != 1:
            self.cluster(full_x,best_nc)
            self.best_cluster_labels = self.cluster_labels.reshape(*x_as_img.shape[:-1])
        if self.verbose: print(f'found {best_nc} clusters')
        return best_nc, best_dl

    def cluster(self,x,nc):
        N = len(x)
        if self.cluster_model == 'cmeans':
            fuzzy_cluster_results = fuzz.cluster.cmeans(np.transpose(x),nc,m=2,error=.005,maxiter=1000)
            neg_log_probs = -log2(fuzzy_cluster_results[1].max(axis=0))
            self.cluster_labels = fuzzy_cluster_results[1].argmax(axis=0)
            self.model_len = 0.5 * nc * self.len_of_each_cluster
            found_nc = nc
        elif self.cluster_model == 'kmeans':
            self.model_len = 0.5 * nc * self.len_of_each_cluster
            found_nc = nc
            self.cluster_labels = KMeans(nc).fit_predict(x)
            sizes_of_each_cluster = [x[self.cluster_labels==i].max(axis=0)-x[self.cluster_labels==i].min(axis=0) for i in range(nc)]
            neg_log_prob_per_cluster = np.array([log2(dr).sum() for dr in sizes_of_each_cluster])
            neg_log_probs = neg_log_prob_per_cluster[self.cluster_labels]
            print(np.bincount(self.cluster_labels))
        else:
            self.model = GMM(nc,n_init=self.n_cluster_inits,covariance_type='diag')
            while True:
                try:
                    self.cluster_labels = self.model.fit_predict(x)
                    break
                except ValueError:
                    print(f'failed to cluster with {nc} components, and reg_covar {self.model.reg_covar}')
                    self.model.reg_covar *= 10
                    print(f'trying again with reg_covar {self.model.reg_covar}')
            found_nc = len(np.unique(self.cluster_labels))
            if nc > 1 and self.display_scattered_clusters:
                scatter_clusters(x,self.best_cluster_labels.flatten(),show=True)
            self.model_len = nc*(self.len_of_each_cluster)
            new_model_scores = -self.model._estimate_log_prob(x)[np.arange(len(x)),self.cluster_labels]
            neg_log_probs = new_model_scores * log2(np.e)
        self.outliers = neg_log_probs > self.len_of_outlier
        self.len_outliers = (self.len_of_outlier * self.outliers).sum()
        self.idxs_len_per_cluster = log2(N) - log2(np.bincount(self.cluster_labels))
        self.idxs_len = self.idxs_len_per_cluster[self.cluster_labels][~self.outliers].sum()
        self.residuals = (neg_log_probs * ~self.outliers).sum()
        self.dl = self.residuals + self.len_outliers + self.idxs_len + self.model_len
        return found_nc

    def viz_cluster_labels(self):
        nc = len(np.unique(self.best_cluster_labels))
        pallete = PALETTE[:nc]
        coloured_clabs = np.array(pallete)[self.best_cluster_labels]
        coloured_clabs = np.resize(coloured_clabs,(*self.best_cluster_labels.shape,3))
        plt.imshow(coloured_clabs)
        plt.savefig('segmentation_by_clusters.png',bbox_inches=0)
        plt.show()

    def apply_conv_layer(self,x,layer_num='none',custom_cnvl='none'):
        assert (layer_num == 'none') ^ (custom_cnvl == 'none')
        try:
            torch_x = torch.tensor(x).transpose(0,2).float().cuda()
            if torch_x.ndim == 3:
                torch_x = torch_x.unsqueeze(0)
            layer_to_apply = self.layers[layer_num].cuda() if custom_cnvl == 'none' else custom_cnvl
            torch_x = layer_to_apply(torch_x)
            return numpyify(torch_x.squeeze(0).transpose(0,2))
        except Exception as e:
            print(e)
            breakpoint()

    def info_in_patches(self,patched_im,subsample):
        assert patched_im.ndim == 3
        nc = patched_im.shape[2]
        flattened = patched_im.reshape(-1,nc)
        if subsample == 1:
            to_use = flattened
        else:
            num_to_subsample = int(len(flattened) * subsample)
            subsample_idxs = np.random.choice(len(flattened),size=num_to_subsample,replace=False)
            to_use = flattened[subsample_idxs]
        y = [tuple(z) for z in np.unique(to_use,axis=0)]
        if self.verbose:
            print(f'{len(flattened)} labels of {len(y)} distinct values')
        tuples_as_idxs = np.array([y.index(tuple(z)) for z in to_use])
        return self.complexity_func(tuples_as_idxs)

def labels_ass_idx(labels: np.array):
    assert labels.ndim == 1
    bin_counts = np.bincount(labels.flatten())
    return log(bin_counts).sum() / bin_counts.sum()

def labels_entropy(labels: np.array):
    assert labels.ndim == 1
    bin_counts = np.bincount(labels.flatten())
    return entropy(bin_counts,base=2)

def info_in_label_counts(labels):
    assert labels.ndim == 1
    N = len(labels)
    counts = np.bincount(labels)
    log_counts = log2(counts)
    return N*log2(N) - np.dot(counts,log_counts)

def make_dummy_layer(ks,stride,padding):
    cnvl = nn.Conv2d(1,1,ks,stride,padding=padding,padding_mode='replicate')
    cnvl.weight.data=torch.ones_like(cnvl.weight).requires_grad_(False)/(ks**2)
    return cnvl

def viz_proc_im(x):
    plt.imshow(x.sum(axis=2)); plt.show()

def combine_patches(a,ps,comb_method):
    if comb_method == 'concat':
        comb_func = lambda x: np.concatenate(x.astype(float),axis=2)
    elif comb_method == 'sum':
        comb_func = lambda x: sum([z.astype(float) for z in x])
    elif comb_method == 'or':
        comb_func = lambda x: recursive_np_or(x).astype(float)
    row_shifts = [a[i:i-ps] for i in range(ps)]
    row_combined = comb_func(row_shifts)
    column_row_shifts = [row_combined[:,i:i-ps] for i in range(ps)]
    return comb_func(column_row_shifts)

def patch_averages(a):
    try:
        padded = np.pad(a,1)[:,:,1:-1]
    except Exception as e:
        print(e)
        breakpoint()
    summed = padded[:-1,:-1] + padded[:-1,1:] + padded[1:,:-1] + padded[1:,1:]
    return (summed/4)[1:-1,1:-1]

def torch_min(t,val):
    """return the minimum of val (float) and t (tensor), with val broadcast"""
    return torch.minimum(t,val*torch.ones_like(t))

def sum_logs(labels):
    counts = np.bincount(labels)
    log_counts = log2(counts)
    return log_counts.sum()
