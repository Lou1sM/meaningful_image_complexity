from PIL import Image
from scipy.special import softmax
from matplotlib.colors import BASE_COLORS
from dl_utils.misc import scatter_clusters
from dl_utils.tensor_funcs import numpyify
from dl_utils.label_funcs import compress_labels
from load_non_torch_dsets import load_rand
from make_simple_imgs import get_simple_img
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture as GMM
from torchvision import models
from torchvision.transforms import ToTensor
import argparse
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets as data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from umap import UMAP


PALETTE = list(BASE_COLORS.values()) + [(0,0.5,1),(1,0.5,0)]
class ComplexityMeasurer():
    def __init__(self,verbose,ncs_to_check,resnet,n_cluster_inits,display_cluster_imgs):
        self.verbose = verbose
        self.display_cluster_imgs = display_cluster_imgs
        self.n_cluster_inits = n_cluster_inits
        self.ncs_to_check = ncs_to_check
        self.layer1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu,resnet.maxpool)
        self.layer2 = resnet.layer1
        self.layer3 = resnet.layer2
        self.layer4 = resnet.layer3
        self.layer5 = resnet.layer4
        self.layers = (self.layer1,self.layer2,self.layer3,self.layer4,self.layer5)
        for layer in self.layers:
            for m in layer.modules():
                if hasattr(m,'padding_mode'):
                    m.padding_mode = 'replicate'
        self.dummy_initial_layer = nn.Sequential(make_dummy_layer(7,2,3),nn.MaxPool2d(3,2,1,dilation=1,ceil_mode=False))
        self.dummy_downsample_rlayer = make_dummy_layer(3,2,1)

    def interpret(self,given_x):
        x = np.copy(given_x)
        total_num_clusters = 0
        total_weighted = 0
        self.get_smallest_increment(x)
        for layer_being_processed in range(6):
            if (x.shape[0]-1)*(x.shape[1]-1) < 50:
                break
            self.layer_being_processed = layer_being_processed
            if self.verbose:
                print(f'applying cl to make im size {x.shape}')
            num_clusters_at_this_level, dl, weighted = self.mdl_cluster(x)
            total_num_clusters += num_clusters_at_this_level
            total_weighted += weighted
            if ARGS.centroidify:
                full_im_size_means = [x[self.best_cluster_labels==c].mean(axis=0)
                    for c in np.unique(self.best_cluster_labels)]
                x = np.array(full_im_size_means)[self.best_cluster_labels]
            x = self.apply_conv_layer(x,layer_being_processed)
        return total_num_clusters, layer_being_processed, total_weighted

    def get_smallest_increment(self,x):
        sx = sorted(x.flatten())
        increments = [sx2-sx1 for sx1,sx2 in zip(sx[:-1],sx[1:])]
        self.prec = min([item for item in increments if item != 0])

    def project_clusters(self):
        prev_clabs_as_img = np.expand_dims(self.best_cluster_labels,2)
        if self.layer_being_processed==2:
            proj_clusters_as_img = prev_clabs_as_img
        elif self.layer_being_processed==1:
            proj_clusters_as_img = self.apply_conv_layer(prev_clabs_as_img,custom_cnvl=self.dummy_initial_layer)
        else:
            proj_clusters_as_img = self.apply_conv_layer(prev_clabs_as_img,custom_cnvl=self.dummy_downsample_rlayer)
        return proj_clusters_as_img.flatten().round().astype(int)

    def mdl_cluster(self,x_as_img):
        x = patch_averages(x_as_img) if ARGS.patch else x_as_img
        x = x.reshape(-1,x.shape[-1])
        assert x.ndim == 2
        N,nz = x.shape
        if nz > 50:
            x = PCA(50).fit_transform(x)
        if nz > 3:
            x = UMAP(min_dist=0,n_neighbors=50).fit_transform(x).squeeze()
        if ARGS.display_cluster_imgs:
            scatter_clusters(x,labels=None,show=True)
        N,nz = x.shape
        data_range_by_axis = x.max(axis=0) - x.min(axis=0)
        len_of_each_cluster = (nz+1)/2 * (np.log2(data_range_by_axis).sum() + 32) # Float precision
        len_of_outlier = np.log2(data_range_by_axis).sum() # Omit the 32 here because implicitly omitted in the model log_prob computation
        best_dl = np.inf
        best_nc = -1
        hangover_neg_log_probs = np.inf*np.ones(len(x))
        neg_dls_by_dpoint = []
        idxs_lens = []
        all_rs = []
        all_ts = []
        for nc in range(1,self.ncs_to_check+1):
            self.model = GMM(nc,n_init=self.n_cluster_inits)
            self.cluster_labels = self.model.fit_predict(x)
            if len(np.unique(self.cluster_labels)) == nc-1:
                print(f"only found {nc-1} clusters when looking for {nc}, terminating here"); break
            if nc > 1 and self.display_cluster_imgs:
                self.viz_cluster_labels(x_as_img.shape[:2])
            model_len = nc*(len_of_each_cluster)
            idxs_len_per_cluster = np.log2(N) - np.log2(np.bincount(self.cluster_labels))
            idxs_len_per_dpoint = idxs_len_per_cluster[self.cluster_labels]
            new_model_scores = -self.model._estimate_log_prob(x)[np.arange(len(x)),self.cluster_labels]
            neg_log_probs = new_model_scores * np.log2(np.e)
            outliers = neg_log_probs > len_of_outlier
            residual_errors = neg_log_probs * ~outliers
            len_outliers = len_of_outlier * outliers
            dl_by_dpoint = residual_errors + len_outliers + idxs_len_per_dpoint + model_len/N
            if self.verbose:
                print(f'{nc} {dl_by_dpoint.sum()}\tMod: {model_len}\tErr: {residual_errors.sum()}\tO: {outliers.sum()} {len_outliers.sum()}')
            all_rs.append(residual_errors)
            all_ts.append(dl_by_dpoint)
            idxs_lens.append(idxs_len_per_dpoint)
            neg_dls_by_dpoint.append(-dl_by_dpoint)
        if ARGS.display_cluster_imgs:
            scatter_clusters(x,self.best_cluster_labels.flatten(),show=True)
        idxs_lens_array = np.stack(idxs_lens,axis=1)
        log_likelihood_per_dpoint = np.stack(neg_dls_by_dpoint,axis=1)
        posterior_per_dpoint = softmax(log_likelihood_per_dpoint,axis=1)
        votes_for_nc = posterior_per_dpoint.sum(axis=0)
        weighted = (idxs_lens_array * posterior_per_dpoint).sum()
        #print(f"votes for nc: {votes_for_nc}")
        print(f"weighted for layer {self.layer_being_processed}: {weighted:.2f}")
        return best_nc, best_dl, weighted

    def viz_cluster_labels(self,size):
        nc = len(np.unique(self.cluster_labels))
        pallete = PALETTE[:nc]
        coloured_clabs = np.array(pallete)[self.cluster_labels]
        coloured_clabs = np.resize(coloured_clabs,(*size,3))
        plt.imshow(coloured_clabs); plt.show()

    def apply_conv_layer(self,x,layer_num='none',custom_cnvl='none'):
        assert (layer_num == 'none') ^ (custom_cnvl == 'none')
        try:
            torch_x = torch.tensor(x).transpose(0,2).float()
            if torch_x.ndim == 3:
                torch_x = torch_x.unsqueeze(0)
            layer_to_apply = self.layers[layer_num] if custom_cnvl == 'none' else custom_cnvl
            torch_x = layer_to_apply(torch_x)
            return numpyify(torch_x.squeeze(0).transpose(0,2))
        except Exception as e:
            print(e)
            breakpoint()

def info_in_label_counts(labels):
    assert labels.ndim == 1
    N = len(labels)
    counts = np.bincount(labels)
    log_counts = np.log2(counts)
    return N*np.log2(N) - np.dot(counts,log_counts)

def make_dummy_layer(ks,stride,padding):
    cnvl = nn.Conv2d(1,1,ks,stride,padding=padding,padding_mode='replicate')
    cnvl.weight.data=torch.ones_like(cnvl.weight).requires_grad_(False)/(ks**2)
    return cnvl

def viz_proc_im(x):
    plt.imshow(x.sum(axis=2)); plt.show()

def patch_averages(a):
    try:
        padded = np.pad(a,1)[:,:,1:-1]
    except Exception as e:
        breakpoint()
    summed = padded[:-1,:-1] + padded[:-1,1:] + padded[1:,:-1] + padded[1:,1:]
    return (summed/4)[1:-1,1:-1]

def torch_min(t,val):
    return torch.minimum(t,val*torch.ones_like(t))


parser = argparse.ArgumentParser()
parser.add_argument('--no_resize',action='store_true')
parser.add_argument('--display_cluster_imgs',action='store_true')
parser.add_argument('--patch',action='store_true')
parser.add_argument('--no_pretrained',action='store_true')
parser.add_argument('--verbose','-v',action='store_true')
parser.add_argument('--display_images',action='store_true')
parser.add_argument('--centroidify',action='store_true')
parser.add_argument('--conv_abl',action='store_true')
parser.add_argument('--dset',type=str,choices=['im','cifar','mnist','rand','dtd','simp'],default='simp')
parser.add_argument('--num_ims',type=int,default=1)
parser.add_argument('--ncs_to_check',type=int,default=10)
parser.add_argument('--n_cluster_inits',type=int,default=1)
ARGS = parser.parse_args()

if ARGS.dset == 'cifar':
    dset = torchvision.datasets.CIFAR10(root='/home/louis/datasets',download=True,train=True)
elif ARGS.dset == 'mnist':
    dset = torchvision.datasets.MNIST(root='/home/louis/datasets',train=False,download=True)
elif ARGS.dset == 'rand':
    dset = np.random.rand(ARGS.num_ims,224,224,3)
all_assembly_idxs = []
all_levels = []
all_weighteds = []
net = models.resnet18(pretrained=~ARGS.no_pretrained)
comp_meas = ComplexityMeasurer(verbose=ARGS.verbose,ncs_to_check=ARGS.ncs_to_check,resnet=net,n_cluster_inits=ARGS.n_cluster_inits,display_cluster_imgs=ARGS.display_cluster_imgs)
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
for i in range(ARGS.num_ims):
    if ARGS.dset == 'im':
        im, label = load_rand('imagenette',~ARGS.no_resize)
        im = im/255
        if im.ndim == 2:
            im = np.resize(im,(*(im.shape),1))
    elif ARGS.dset == 'dtd':
        im, label = load_rand('dtd',~ARGS.no_resize)
        im /= 255
    elif ARGS.dset == 'simp':
        #label = np.random.choice(('stripes','halves'))
        label = 'halves'
        slope = np.random.rand()+.5
        im = get_simple_img(label,slope,line_thickness=5)
    elif ARGS.dset == 'rand':
        im = dset[i]
        label = 'none'
    else:
        if ARGS.dset == 'cifar':
            im = dset.data[i]/255
            im = np.array(Image.fromarray(im).resize((224,224)))
        elif ARGS.dset == 'mnist':
            im = numpyify(dset.data[i])
            im = np.array(Image.fromarray(im).resize((224,224)))
            im = np.expand_dims(im,2)
        label = dset.targets[i]
    if ARGS.display_cluster_imgs:
        plt.imshow(im);plt.show()
    im = (im-mean)/std
    if ARGS.conv_abl:
        comp_meas.get_smallest_increment(im)
        nc_in_image_itself,_,weighted = comp_meas.mdl_cluster(im)
        print(f'Class: {label}\tNC in image itself: {nc_in_image_itself}')
        all_assembly_idxs.append(nc_in_image_itself)
    else:
        assembly_idx,level,weighted  = comp_meas.interpret(im)
        print(f'Class: {label}\tAssemby num: {assembly_idx}\tLevel: {level}\tWeighted: {weighted:.3f}')
        all_assembly_idxs.append(assembly_idx)
        all_levels.append(level)
        all_weighteds.append(weighted)
mean_assembly_idx = np.array(all_assembly_idxs).mean()
mean_level = np.array(all_levels).mean()
mean_weighted = np.array(all_weighteds).mean()
print(f'Mean weighted: {mean_weighted:.3f}')
