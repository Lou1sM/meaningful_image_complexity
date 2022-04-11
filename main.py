import argparse
from PIL import Image
import torchvision
import numpy as np
from torchvision import models
from mdl_conv import ComplexityMeasurer
from load_non_torch_dsets import load_rand
from make_simple_imgs import get_simple_img
import matplotlib.pyplot as plt
from dl_utils.tensor_funcs import numpyify
from baselines import rgb2gray, fractal_dimension, glcm_entropy


parser = argparse.ArgumentParser()
parser.add_argument('--no_resize',action='store_true')
parser.add_argument('--display_cluster_imgs',action='store_true')
parser.add_argument('--patch',action='store_true')
parser.add_argument('--no_pretrained',action='store_true')
parser.add_argument('--verbose','-v',action='store_true')
#parser.add_argument('--display_images',action='store_true')
parser.add_argument('--centroidify',action='store_true')
parser.add_argument('--conv_abl',action='store_true')
parser.add_argument('--is_choose_model_per_dpoint',action='store_true')
parser.add_argument('--dset',type=str,choices=['im','cifar','mnist','rand','dtd','simp'],default='simp')
parser.add_argument('--num_ims',type=int,default=1)
parser.add_argument('--ncs_to_check',type=int,default=10)
parser.add_argument('--n_cluster_inits',type=int,default=1)
parser.add_argument('--nz',type=int,default=2)
parser.add_argument('--alg_nz',type=str,choices=['pca','umap','tsne'],default='pca')
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
net = models.resnet18(pretrained=not ARGS.no_pretrained)
comp_meas_kwargs = ARGS.__dict__
comp_meas = ComplexityMeasurer(resnet=net,**comp_meas_kwargs)
#comp_meas = ComplexityMeasurer(verbose=ARGS.verbose,ncs_to_check=ARGS.ncs_to_check,resnet=net,n_cluster_inits=ARGS.n_cluster_inits,display_cluster_imgs=ARGS.display_cluster_imgs)
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
weighted_by_class = {}

def append_or_add_key(d,key,val):
    try:
        d[key].append(val)
    except KeyError:
        d[key] = [val]

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
        label = np.random.choice(('stripes','halves'))
        slope = np.random.rand()+.5
        im = get_simple_img(label,slope,line_thickness=5)
    elif ARGS.dset == 'rand':
        im = dset[i]
        label = 'none'
    else:
        if ARGS.dset == 'cifar':
            im = dset.data[i]
            im = np.array(Image.fromarray(im).resize((224,224)))/255
        elif ARGS.dset == 'mnist':
            im = numpyify(dset.data[i])
            im = np.array(Image.fromarray(im).resize((224,224)))
            im = np.expand_dims(im,2)
        label = int(dset.targets[i])
    if ARGS.display_cluster_imgs:
        plt.imshow(im);plt.show()
    im_normed = (im-mean)/std
    greyscale_im = rgb2gray(im)
    print("Minkowski–Bouligand dimension (computed): ", fractal_dimension(greyscale_im))
    print("GLCM entropy: ", glcm_entropy((greyscale_im*255).astype(np.uint8)))
    if ARGS.conv_abl:
        comp_meas.get_smallest_increment(im_normed)
        nc_in_image_itself,_,weighted = comp_meas.mdl_cluster(im_normed)
        all_assembly_idxs.append(nc_in_image_itself)
    else:
        assembly_idx,level,weighted = comp_meas.interpret(im_normed)
        all_assembly_idxs.append(assembly_idx)
        all_levels.append(level)
        all_weighteds.append(weighted)
        append_or_add_key(weighted_by_class,label,weighted)
    print(f'Class: {label}\tWeighted: {weighted:.3f}')
mean_assembly_idx = np.array(all_assembly_idxs).mean()
mean_level = np.array(all_levels).mean()
mean_weighted = np.array(all_weighteds).mean()
print()
for k,v in weighted_by_class.items():
    print(f'{k}: {np.array(v).mean():.2f}')
print(f'\nMean across all classes: {mean_weighted:.3f}')
