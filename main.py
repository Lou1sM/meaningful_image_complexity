import argparse
from time import time
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
from dl_utils.misc import check_dir
from baselines import rgb2gray, compute_fractal_dimension, glcm_entropy, machado2015, jpg_compression_ratio, khan2021, redies2012
from skimage.measure import shannon_entropy


parser = argparse.ArgumentParser()
parser.add_argument('--no_resize',action='store_true')
parser.add_argument('--display_cluster_imgs',action='store_true')
parser.add_argument('--rand_dpoint',action='store_true')
parser.add_argument('--patch_comb_method',type=str,choices=['sum','concat','or'],default='sum')
parser.add_argument('--skip_layers',type=int,nargs='+',default=[])
parser.add_argument('--subsample',type=float,default=1)
parser.add_argument('--info_subsample',type=float,default=1)
parser.add_argument('--verbose','-v',action='store_true')
parser.add_argument('--print_times',action='store_true')
parser.add_argument('--centroidify',action='store_true')
parser.add_argument('--cluster_idxify',action='store_true')
parser.add_argument('--show_df',action='store_true')
parser.add_argument('--is_choose_model_per_dpoint',action='store_true')
parser.add_argument('--dset',type=str,choices=['im','cifar','mnist','rand','dtd','stripes','halves'],default='stripes')
parser.add_argument('--num_ims',type=int,default=1)
parser.add_argument('--ncs_to_check',type=int,default=10)
parser.add_argument('--n_cluster_inits',type=int,default=1)
parser.add_argument('--nz',type=int,default=2)
parser.add_argument('--alg_nz',type=str,choices=['pca','umap','tsne'],default='pca')
ARGS = parser.parse_args()

if ARGS.dset == 'cifar':
    dset = torchvision.datasets.CIFAR10(root='~/datasets',download=True,train=True)
elif ARGS.dset == 'mnist':
    dset = torchvision.datasets.MNIST(root='~/datasets',train=False,download=True)
elif ARGS.dset == 'rand':
    dset = np.random.rand(ARGS.num_ims,224,224,3)
all_c_idx_infos = []
all_patch_entropys = []
comp_meas_kwargs = ARGS.__dict__
comp_meas = ComplexityMeasurer(**comp_meas_kwargs)
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
weighted_by_class = {}
methods = ['mdl','gclm','fract','ent','ncs','jpg','mach','khan','redies','patch_ent']
results_dict = {m:{} for m in methods}

def append_or_add_key(d,key,val):
    try:
        d[key].append(val)
    except KeyError:
        d[key] = [val]

line_thicknesses = np.random.permutation(np.arange(3,10))
img_start_times = []
for i in range(ARGS.num_ims):
    img_start_times.append(time())
    if ARGS.dset == 'im':
        im, label = load_rand('imagenette',~ARGS.no_resize)
        im = im/255
        if im.ndim == 2:
            im = np.resize(im,(*(im.shape),1))
    elif ARGS.dset == 'dtd':
        im, label = load_rand('dtd',~ARGS.no_resize)
        im = im/255
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
        idx = np.random.randint(len(dset)) if ARGS.rand_dpoint else i
        print(idx)
        if ARGS.dset == 'cifar':
            im = dset.data[idx]
            im = np.array(Image.fromarray(im).resize((224,224)))/255
        elif ARGS.dset == 'mnist':
            im = numpyify(dset.data[idx])
            im = np.array(Image.fromarray(im).resize((224,224)))
            im = np.tile(np.expand_dims(im,2),(1,1,3))
        label = int(dset.targets[idx])
    if ARGS.display_cluster_imgs:
        plt.imshow(im);plt.show()
    #im_normed = (im-mean)/std
    im_normed = im
    greyscale_im = rgb2gray(im)
    fractal_dim = compute_fractal_dimension(greyscale_im)
    im_unint8 = (greyscale_im*255).astype(np.uint8)
    gclm_ent = glcm_entropy(im_unint8)
    ent = shannon_entropy(im_unint8)
    new_patch_entropys, ncs, new_c_idx_infos = comp_meas.interpret(im_normed)
    all_c_idx_infos.append(new_c_idx_infos)
    all_patch_entropys.append(new_patch_entropys)
    jpg = jpg_compression_ratio(im)
    mach = machado2015(im)
    khan = khan2021(im)
    redies = redies2012(im)
    c_idx_info = sum(new_c_idx_infos)
    patch_entropy = sum(new_patch_entropys)
    append_or_add_key(results_dict['patch_ent'],label,patch_entropy)
    append_or_add_key(results_dict['mdl'],label,c_idx_info)
    append_or_add_key(results_dict['ncs'],label,ncs)
    append_or_add_key(results_dict['gclm'],label,gclm_ent)
    append_or_add_key(results_dict['fract'],label,fractal_dim)
    append_or_add_key(results_dict['ent'],label,ent)
    append_or_add_key(results_dict['jpg'],label,jpg)
    append_or_add_key(results_dict['mach'],label,mach)
    append_or_add_key(results_dict['khan'],label,khan)
    append_or_add_key(results_dict['redies'],label,redies)

img_times = [img_start_times[i+1] - imgs for i,imgs in enumerate(img_start_times[:-1])]
tot_c_time = (img_start_times[-1] - img_start_times[0])/ARGS.num_ims
#print(' '.join([f'{i}: {s:.1f}' for i,s in enumerate(img_times)]) + tot_c_time)
print(f'Avg time per image: {tot_c_time}')
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
check_dir('experiments')
mi_df_for_this_dset.to_csv(f'experiments/{ARGS.dset}_results.csv')
if ARGS.show_df:
    print(mi_df_for_this_dset)
mean_weighted = np.array(all_c_idx_infos).mean(axis=0)
mean_patch_entropys = np.array(all_patch_entropys).mean(axis=0)
print(*[f'{m:.3f}' for m in mean_weighted])
print(*[f'{pe:.3f}' for pe in mean_patch_entropys])
