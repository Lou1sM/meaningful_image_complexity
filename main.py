import argparse
from utils import append_or_add_key, results_dict_to_df, make_alls_df
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
parser.add_argument('--alg_nz',type=str,choices=['pca','umap','tsne'],default='pca')
parser.add_argument('--cluster_idxify',action='store_true')
parser.add_argument('--compare_to_true_entropy',action='store_true')
parser.add_argument('--display_cluster_imgs',action='store_true')
parser.add_argument('--dset',type=str,choices=['im','cifar','mnist','rand','dtd','stripes','halves'],default='stripes')
parser.add_argument('--exp_name',type=str,default='jim')
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
parser.add_argument('--show_df',action='store_true')
parser.add_argument('--subsample',type=float,default=1)
parser.add_argument('--verbose','-v',action='store_true')
ARGS = parser.parse_args()

exp_dir = f'experiments/{ARGS.exp_name}/{ARGS.dset}'
if (isfile(join(exp_dir,f'{ARGS.dset}_results.csv')) and
    not ARGS.exp_name.endswith('jim') and not ARGS.overwrite):
    is_overwrite = get_user_yesno_answer(f'experiment {ARGS.exp_name}/{ARGS.dset} already exists, overwrite?')
    if not is_overwrite:
        print('aborting')
        sys.exit()
check_dir(exp_dir)
all_single_labels_entropys = []
all_patch_entropys = []
comp_meas_kwargs = ARGS.__dict__
comp_meas = ComplexityMeasurer(**comp_meas_kwargs)
single_labels_entropy_by_class = {}
methods = ['no_patch','no_mdl','gclm','fract','ent','ncs','jpg','mach','khan','redies','patch_ent']
methods += [f'patch_ent{i}' for i in range(ARGS.num_layers)]
results_dict = {m:{} for m in methods}

img_start_times = []
img_times_real = []
labels = []
img_streamer = ImageStreamer(ARGS.dset,~ARGS.no_resize)
for idx,(im,label) in enumerate(img_streamer.stream_images(ARGS.num_ims)):
    print(idx)
    if ARGS.gaussian_noisify > 0:
        noise = np.random.randn(*im.shape)
        im += ARGS.gaussian_noisify*noise
        im = np.clip(im,0,1)

    img_start_times.append(time())
    if ARGS.display_cluster_imgs:
        plt.imshow(im);plt.show()
    im_normed = im
    if ARGS.include_mdl_abl:
        comp_meas.is_mdl_abl = True
        no_mdls, _, _ = comp_meas.interpret(im)
        comp_meas.is_mdl_abl = False
    else:
        no_mdls = [-1,-1,-1,-1]
    img_start_time_real = time()
    new_patch_entropys, ncs, new_single_labels_entropys = comp_meas.interpret(im)
    img_times_real.append(time()-img_start_time_real)

    results_dict_for_this_im = {}
    for i,pe in enumerate(new_patch_entropys):
        results_dict_for_this_im[f'patch_ent{i}'] = pe
    results_dict_for_this_im['no_patch'] = sum(new_single_labels_entropys)
    results_dict_for_this_im['no_mdl'] = sum(no_mdls)
    results_dict_for_this_im['patch_ent'] = sum(new_patch_entropys)
    results_dict_for_this_im['ncs'] = ncs
    greyscale_im = rgb2gray(im)
    results_dict_for_this_im['fract'] = compute_fractal_dimension(greyscale_im)
    im_unint8 = (greyscale_im*255).astype(np.uint8)
    results_dict_for_this_im['gclm'] = glcm_entropy(im_unint8)
    results_dict_for_this_im['ent'] = shannon_entropy(im_unint8)
    all_single_labels_entropys.append(new_single_labels_entropys)
    all_patch_entropys.append(new_patch_entropys)
    results_dict_for_this_im['jpg'] = jpg_compression_ratio(im)
    results_dict_for_this_im['mach'] = machado2015(im)
    results_dict_for_this_im['khan'] = khan2021(im)
    results_dict_for_this_im['redies'] = redies2012(im)
    label_for_df = label.split('_')[0] if ARGS.dset in ['im','dtd'] else label
    for m,r in results_dict_for_this_im.items():
        append_or_add_key(results_dict[m],label_for_df,r)
        append_or_add_key(results_dict[m],'all',r)
    labels.append(label)

img_start_times.append(time())
img_times = [img_start_times[i+1] - imgs for i,imgs in enumerate(img_start_times[:-1])]
avg_img_time = (img_start_times[-1] - img_start_times[0])/ARGS.num_ims
avg_img_time_real = np.array(img_times_real).mean()
print(f'Avg time per image: {avg_img_time}')
print(f'Avg time per image real: {avg_img_time_real}')


mean_var_results = {method_k:{class_k:{'mean':np.array(v).mean(),'var':np.array(v).mean()}
                    for class_k,v in method_v.items()}
                    for method_k,method_v in results_dict.items()}
df_for_this_dset = results_dict_to_df(results_dict)
alls_df = make_alls_df(df_for_this_dset)
df_for_this_dset.to_csv(join(exp_dir,f'{ARGS.dset}_results_by_class.csv'),index=True)
alls_df.to_csv(join(exp_dir,f'{ARGS.dset}_results.csv'),index=True)
with open(join(exp_dir,f'{ARGS.dset}_ims_used.txt'),'w') as f:
    for lab in labels:
        f.write(lab + '\n')
with open(join(exp_dir,f'{ARGS.dset}_ARGS.txt'),'w') as f:
    for a in dir(ARGS):
        if not a.startswith('_'):
           f.write(f'{a}: {getattr(ARGS,a)}'+ '\n')
if ARGS.show_df:
    print(df_for_this_dset)
    print(alls_df)
mean_single_labels_entropy = np.array(all_single_labels_entropys).mean(axis=0)
mean_patch_entropys = np.array(all_patch_entropys).mean(axis=0)
print(*[f'{m:.3f}' for m in mean_single_labels_entropy], f'total:{mean_single_labels_entropy.sum():.3f}')
print(*[f'{pe:.3f}' for pe in mean_patch_entropys], f'total:{mean_patch_entropys.sum():.3f}')
