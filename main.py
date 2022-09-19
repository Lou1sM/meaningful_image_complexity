import argparse
from utils import append_or_add_key, results_dict_to_df, make_alls_df
import pandas as pd
from time import time
import numpy as np
from mdl_conv import ComplexityMeasurer
from get_dsets import ImageStreamer
import matplotlib.pyplot as plt
from os.path import join,isfile
import sys
from dl_utils.misc import check_dir, get_user_yesno_answer
from baselines import rgb2gray, compute_fractal_dimension, glcm_entropy, machado2015, jpg_compression_ratio, khan2021, redies2012
from skimage.measure import shannon_entropy


parser = argparse.ArgumentParser()
parser.add_argument('--abls_only',action='store_true')
parser.add_argument('--alg_nz',type=str,choices=['pca','umap','tsne'],default='pca')
parser.add_argument('--ass',action='store_true')
parser.add_argument('--no_cluster_idxify',action='store_true')
parser.add_argument('--compare_to_true_entropy',action='store_true')
parser.add_argument('--display_cluster_label_imgs',action='store_true')
parser.add_argument('--display_input_imgs',action='store_true')
parser.add_argument('--display_scattered_clusters',action='store_true')
parser.add_argument('-d','--dset',type=str,choices=['im','cifar','mnist','rand','dtd','stripes','halves','fractal_imgs'],default='stripes')
parser.add_argument('--downsample',type=int,default=-1)
parser.add_argument('--exp_name',type=str,default='jim')
parser.add_argument('--given_fname',type=str,default='none')
parser.add_argument('--given_class_dir',type=str,default='none')
parser.add_argument('--include_mdl_abl',action='store_true')
parser.add_argument('--info_subsample',type=float,default=1)
parser.add_argument('--gaussian_noisify',type=float,default=0.)
parser.add_argument('--is_choose_model_per_dpoint',action='store_true')
parser.add_argument('--n_cluster_inits',type=int,default=1)
parser.add_argument('--ncs_to_check',type=int,default=2)
parser.add_argument('--no_resize',action='store_true')
parser.add_argument('--num_ims',type=int,default=1)
parser.add_argument('--num_layers',type=int,default=4)
parser.add_argument('--nz',type=int,default=2)
parser.add_argument('--overwrite',action='store_true')
parser.add_argument('--patch_comb_method',type=str,choices=['sum','concat','or'],default='sum')
parser.add_argument('--print_times',action='store_true')
parser.add_argument('--rand_dpoint',action='store_true')
parser.add_argument('--run_other_methods',action='store_true')
parser.add_argument('--save_last',action='store_true')
parser.add_argument('--select_randomly',action='store_true')
parser.add_argument('--show_df',action='store_true')
parser.add_argument('--subsample',type=float,default=1)
parser.add_argument('--cluster_model',type=str,choices=['kmeans','cmeans','GMM'],default='GMM')
parser.add_argument('--verbose','-v',action='store_true')
ARGS = parser.parse_args()

if ARGS.abls_only:
    ARGS.run_other_methods = True

exp_dir = f'experiments/{ARGS.exp_name}/{ARGS.dset}'
if (isfile(join(exp_dir,f'{ARGS.dset}_results.csv')) and
    not ARGS.exp_name.endswith('jim') and not ARGS.overwrite):
    is_overwrite = get_user_yesno_answer(f'experiment {ARGS.exp_name}/{ARGS.dset} already exists, overwrite?')
    if not is_overwrite:
        print('aborting')
        sys.exit()
check_dir(exp_dir)
comp_meas_kwargs = ARGS.__dict__
comp_meas = ComplexityMeasurer(**comp_meas_kwargs)
single_labels_entropy_by_class = {}
methods = ['img_label','proc_time','total'] + [f'level {i+1}' for i in range(ARGS.num_layers)]
if ARGS.run_other_methods:
    methods += ['glcm','no_mdl','fract','ent','jpg','mach','khan','redies','no_patch']
results_df = pd.DataFrame(columns=methods,index=list(range(ARGS.num_ims))+['stds','means'])

img_start_times = []
img_times_real = []
labels = []
img_streamer = ImageStreamer(ARGS.dset,~ARGS.no_resize)
for idx,(im,label) in enumerate(img_streamer.stream_images(ARGS.num_ims,ARGS.downsample,ARGS.given_fname,ARGS.given_class_dir,ARGS.select_randomly)):
    img_label = label.split('_')[0] if ARGS.dset in ['im','dtd'] else label
    print(idx, img_label)
    plt.axis('off')
    if ARGS.save_last:
        plt.imshow(im); plt.savefig('image_just_used.png')
    if ARGS.gaussian_noisify > 0:
        noise = np.random.randn(*im.shape)
        im += ARGS.gaussian_noisify*noise
        im = np.clip(im,0,1)

    img_start_times.append(time())
    if ARGS.display_input_imgs:
        plt.imshow(im);plt.show()
    im_normed = im
    if ARGS.include_mdl_abl:
        comp_meas.is_mdl_abl = True
        no_mdls, _, _ = comp_meas.interpret(im)
        comp_meas.is_mdl_abl = False
    else:
        no_mdls = [0]
    img_start_time = time()
    comp_meas.is_mdl_abl = False
    if ARGS.abls_only:
        scores_at_each_level, ncs, new_single_labels_entropys = [-np.ones(ARGS.num_layers)]*3
    else:
        scores_at_each_level, ncs, new_single_labels_entropys = comp_meas.interpret(im)
    results_df.loc[idx,'img_label'] = img_label
    results_df.loc[idx,'proc_time'] = time()-img_start_time
    results_df.loc[idx,'no_patch'] = sum(new_single_labels_entropys)
    results_df.loc[idx,'total'] = sum(scores_at_each_level)
    for i,pe in enumerate(scores_at_each_level):
        results_df.loc[idx,f'level {i+1}'] = pe
    if ARGS.run_other_methods:
        comp_meas.is_mdl_abl = True
        no_mdls, _, _ = comp_meas.interpret(im)
        results_df.loc[idx,'no_mdl'] = sum(no_mdls)
        greyscale_im = rgb2gray(im)
        results_df.loc[idx,'fract'] = compute_fractal_dimension(greyscale_im)
        im_unint8 = (greyscale_im*255).astype(np.uint8)
        results_df.loc[idx,'glcm'] = glcm_entropy(im_unint8)
        results_df.loc[idx,'ent'] = shannon_entropy(im_unint8)
        results_df.loc[idx,'jpg'] = jpg_compression_ratio(im)
        results_df.loc[idx,'mach'] = machado2015(im)
        results_df.loc[idx,'khan'] = khan2021(im)
        results_df.loc[idx,'redies'] = redies2012(im)
    labels.append(label)

stds = results_df.std(axis=0)
means = results_df.mean(axis=0)
results_df.loc['stds'] = stds
results_df.loc['means'] = means
results_df.to_csv(join(exp_dir,f'{ARGS.dset}_results.csv'))
with open(join(exp_dir,f'{ARGS.dset}_ARGS.txt'),'w') as f:
    for a in dir(ARGS):
        if not a.startswith('_'):
           f.write(f'{a}: {getattr(ARGS,a)}'+ '\n')
if ARGS.show_df:
    print(results_df)
print(results_df.loc['means'].drop('img_label'))
