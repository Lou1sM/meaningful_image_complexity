import argparse
from time import time
import json
import pandas as pd
from PIL import Image
import torchvision
import numpy as np
from torchvision import models
from mdl_conv import ComplexityMeasurer
#from load_non_torch_dsets import load_rand
from get_dsets import ImageStreamer
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
parser.add_argument('--compare_to_true_entropy',action='store_true')
parser.add_argument('--centroidify',action='store_true')
parser.add_argument('--cluster_idxify',action='store_true')
parser.add_argument('--show_df',action='store_true')
parser.add_argument('--is_choose_model_per_dpoint',action='store_true')
parser.add_argument('--dset',type=str,choices=['im','cifar','mnist','rand','dtd','stripes','halves'],default='stripes')
parser.add_argument('--num_ims',type=int,default=1)
parser.add_argument('--ncs_to_check',type=int,default=2)
parser.add_argument('--run_sequential',action='store_true')
parser.add_argument('--n_cluster_inits',type=int,default=1)
parser.add_argument('--nz',type=int,default=2)
parser.add_argument('--alg_nz',type=str,choices=['pca','umap','tsne'],default='pca')
ARGS = parser.parse_args()

all_single_labels_entropys = []
all_patch_entropys = []
comp_meas_kwargs = ARGS.__dict__
comp_meas = ComplexityMeasurer(**comp_meas_kwargs)
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
single_labels_entropy_by_class = {}
methods = ['mdl','gclm','fract','ent','ncs','jpg','mach','khan','redies','patch_ent']
results_dict = {m:{} for m in methods}

def append_or_add_key(d,key,val):
    try:
        d[key].append(val)
    except KeyError:
        d[key] = [val]

img_start_times = []
img_times_real = []
labels = []
img_streamer = ImageStreamer(ARGS.dset,~ARGS.no_resize)
for im,label in img_streamer.stream_images(ARGS.num_ims):
    print(label)
    img_start_times.append(time())
    if ARGS.display_cluster_imgs:
        plt.imshow(im);plt.show()
    #im_normed = (im-mean)/std
    im_normed = im
    new_patch_entropys, ncs, new_single_labels_entropys = comp_meas.interpret(im_normed)

    results_dict_for_this_im = {}
    #patch_entropy = sum(new_patch_entropys)
    results_dict_for_this_im['mdl'] = sum(new_single_labels_entropys)
    results_dict_for_this_im['patch_ent'] = sum(new_patch_entropys)
    results_dict_for_this_im['ncs'] = ncs
    greyscale_im = rgb2gray(im)
    results_dict_for_this_im['fract'] = compute_fractal_dimension(greyscale_im)
    #fractal_dim = compute_fractal_dimension(greyscale_im)
    im_unint8 = (greyscale_im*255).astype(np.uint8)
    results_dict_for_this_im['gclm'] = glcm_entropy(im_unint8)
    #gclm_ent = glcm_entropy(im_unint8)
    results_dict_for_this_im['ent'] = shannon_entropy(im_unint8)
    #ent = shannon_entropy(im_unint8)
    img_start_time_real = time()
    img_times_real.append(time()-img_start_time_real)
    all_single_labels_entropys.append(new_single_labels_entropys)
    all_patch_entropys.append(new_patch_entropys)
    results_dict_for_this_im['jpg'] = jpg_compression_ratio(im)
    #jpg = jpg_compression_ratio(im)
    results_dict_for_this_im['mach'] = machado2015(im)
    #mach = machado2015(im)
    results_dict_for_this_im['khan'] = khan2021(im)
    #khan = khan2021(im)
    results_dict_for_this_im['redies'] = redies2012(im)
    #redies = redies2012(im)
    label_for_df = label.split('_')[0] if ARGS.dset in ['im','dtd'] else label
    for m,r in results_dict_for_this_im.items():
        append_or_add_key(results_dict[m],label_for_df,r)
        append_or_add_key(results_dict[m],'all',r)
    #append_or_add_key(results_dict['patch_ent'],label_for_df,patch_entropy)
    #append_or_add_key(results_dict['mdl'],label_for_df,single_labels_entropy)
    #append_or_add_key(results_dict['ncs'],label_for_df,ncs)
    #append_or_add_key(results_dict['gclm'],label_for_df,gclm_ent)
    #append_or_add_key(results_dict['fract'],label_for_df,fractal_dim)
    #append_or_add_key(results_dict['ent'],label_for_df,ent)
    #append_or_add_key(results_dict['jpg'],label_for_df,jpg)
    #append_or_add_key(results_dict['mach'],label_for_df,mach)
    #append_or_add_key(results_dict['khan'],label_for_df,khan)
    #append_or_add_key(results_dict['redies'],label_for_df,redies)
    labels.append(label)

img_times = [img_start_times[i+1] - imgs for i,imgs in enumerate(img_start_times[:-1])]
avg_img_time = (img_start_times[-1] - img_start_times[0])/(ARGS.num_ims-1)
#print(' '.join([f'{i}: {s:.1f}' for i,s in enumerate(img_times)]) + tot_c_time)
avg_img_time_real = np.array(img_times_real).mean()
print(f'Avg time per image: {avg_img_time}')
print(f'Avg time per image real: {avg_img_time_real}')


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
breakpoint()
mi_df_for_this_dset = pd.concat(results_by_method,axis=0,keys=results_dict.keys())
all_df_for_this_dset = pd.DataFrame([x['all'] for x in results_dict.values()],index=results_dict.keys(),columns=['mean','var','std'])
check_dir('experiments')
mi_df_for_this_dset.to_csv(f'experiments/{ARGS.dset}_results_by_class.csv')
all_df_for_this_dset.to_csv(f'experiments/{ARGS.dset}_results.csv')
if ARGS.show_df:
    print(mi_df_for_this_dset)
    print(all_df_for_this_dset)
mean_single_labels_entropy = np.array(all_single_labels_entropys).mean(axis=0)
mean_patch_entropys = np.array(all_patch_entropys).mean(axis=0)
print(*[f'{m:.3f}' for m in mean_single_labels_entropy] + f'total:{mean_single_labels_entropy.sum()}')
print(*[f'{pe:.3f}' for pe in mean_patch_entropys] + f'total:{mean_patch_entropys.sum()}')
