import argparse
import json
import pandas as pd
from PIL import Image
import torchvision
import numpy as np
from torchvision import models
from mdl_conv import ComplexityMeasurer
from load_non_torch_dsets import load_rand
from create_simple_imgs import create_simple_img
import matplotlib.pyplot as plt
from dl_utils.tensor_funcs import numpyify
from baselines import rgb2gray, compute_fractal_dimension, glcm_entropy
from skimage.measure import shannon_entropy


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
parser.add_argument('--dset',type=str,choices=['im','cifar','mnist','rand','dtd','stripes','halves'],default='stripes')
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
methods = ['mdl','gclm','fract','ent','ass']
results_dict = {m:{} for m in methods}

def append_or_add_key(d,key,val):
    try:
        d[key].append(val)
    except KeyError:
        d[key] = [val]

line_thicknesses = np.random.permutation(np.arange(3,10))
for i in range(ARGS.num_ims):
    if ARGS.dset == 'im':
        im, label = load_rand('imagenette',~ARGS.no_resize)
        im = im/255
        if im.ndim == 2:
            im = np.resize(im,(*(im.shape),1))
    elif ARGS.dset == 'dtd':
        im, label = load_rand('dtd',~ARGS.no_resize)
        im /= 255
    elif ARGS.dset == 'stripes':
        slope = np.random.rand()+.5
        line_thickness = line_thicknesses[i%len(line_thicknesses)]
        im = create_simple_img('stripes',slope,line_thickness)
        label = f'stripes-{line_thickness}'
    elif ARGS.dset == 'halves':
        slope = np.random.rand()+.5
        im = create_simple_img('halves',slope,-1)
        label = 'halves'
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
    fractal_dim = compute_fractal_dimension(greyscale_im)
    print("Minkowski–Bouligand dimension (computed): ", fractal_dim)
    im_unint8 = (greyscale_im*255).astype(np.uint8)
    gclm_ent = glcm_entropy(im_unint8)
    print("GLCM entropy: ", gclm_ent)
    ent = shannon_entropy(im_unint8)
    print("entropy: ", ent)
    #if ARGS.conv_abl:
        #comp_meas.get_smallest_increment(im_normed)
        #nc_in_image_itself,_,bayes_mdl = comp_meas.mdl_cluster(im_normed)
        #all_assembly_idxs.append(nc_in_image_itself)
    #else:
    assembly_idx,level,bayes_mdl = comp_meas.interpret(im_normed,ARGS.conv_abl)
    all_assembly_idxs.append(assembly_idx)
    all_levels.append(level)
    append_or_add_key(results_dict['mdl'],label,bayes_mdl)
    append_or_add_key(results_dict['ass'],label,assembly_idx)
    append_or_add_key(results_dict['gclm'],label,gclm_ent)
    append_or_add_key(results_dict['fract'],label,fractal_dim)
    append_or_add_key(results_dict['ent'],label,ent)

def build_innerxy_df(class_results_dict):
    dict_of_dicts = {}
    for k,v in class_results_dict.items():
        ar = np.array(v)
        mean = ar.mean()
        var,std = (ar.var(), ar.std()) if len(v) > 1 else (0,0)
        dict_of_dicts[k] = {'mean':mean,'var':var,'std':std,'raw':ar}
    return dict_of_dicts

mean_var_results = {method_k:{class_k:{'mean':np.array(v).mean(),'var':np.array(v).mean()}
                    for class_k,v in method_v.items()}
                    for method_k,method_v in results_dict.items()}
results_by_method = [pd.DataFrame(build_innerxy_df(d)).T for d in results_dict.values()]
mi_df_for_this_dset = pd.concat(results_by_method,axis=0,keys=results_dict.keys())
mi_df_for_this_dset.to_csv(f'experiments/{ARGS.dset}_results.csv')
mi=pd.MultiIndex.from_product([methods,list(results_dict['mdl'])])
pd.DataFrame(mean_var_results,columns=mi,index=['mean','var'])
mean_weighted = np.array(all_weighteds).mean()
print()
for k,v in weighted_by_class.items():
    print(f'{k}: {np.array(v).mean():.2f}')
print(f'\nMean across all classes: {mean_weighted:.3f}')
