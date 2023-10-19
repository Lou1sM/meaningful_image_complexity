import argparse
import pandas as pd
from time import time
import numpy as np
from measure_complexity import ComplexityMeasurer
from get_dsets import ImageStreamer
import matplotlib.pyplot as plt
from os.path import join,isfile
import sys
from dl_utils.misc import check_dir, get_user_yesno_answer
from baselines import rgb2gray, compute_fractal_dimension, glcm_entropy, machado2015, jpg_compression_ratio, khan2021, redies2012
from skimage.measure import shannon_entropy


parser = argparse.ArgumentParser()
parser.add_argument('--abls_only',action='store_true')
parser.add_argument('--ass',action='store_true',help='use assembly index instead of entropy of labels')
parser.add_argument('--no_cluster_idxify',action='store_true', help="don't replace patches with cluster idxs")
parser.add_argument('--compare_to_true_entropy',action='store_true',help='compare subsampled entropy to full entropy, just use to see how small a subsample is reasonable')
parser.add_argument('--display_cluster_label_imgs',action='store_true')
parser.add_argument('--display_input_imgs',action='store_true')
parser.add_argument('--display_scattered_clusters',action='store_true')
parser.add_argument('-d','--dset',type=str,choices=['im','cifar','mnist','rand', 'bitrand','dtd','stripes','halves','fractal_imgs'],default='stripes')
parser.add_argument('--downsample',type=int,default=-1,help='lower resolution of input images, -1 means no downsampling')
parser.add_argument('--exp_name',type=str,default='tmp')
parser.add_argument('--given_fname',type=str,default='none')
parser.add_argument('--given_class_dir',type=str,default='none')
parser.add_argument('--include_mdl_abl',action='store_true',help='include ablation setting where mdl is not used and nc is set to 5 instead')
parser.add_argument('--info_subsample',type=float,default=1,help='fraction of labels to use for computing entropy, using <1 speeds up computation and generally causes little error when > 0.3')
parser.add_argument('--gaussian_noisify',type=float,default=0.,help='add some fraction of gaussian noise to the input image')
parser.add_argument('--n_cluster_inits',type=int,default=1,help='passed to the clustering model training')
parser.add_argument('--ncs_to_check',type=int,default=2,help='range of values of K to select from using mdl')
parser.add_argument('--no_resize',action='store_true',help="don't resize the input images")
parser.add_argument('--num_ims','-n',type=int,default=1)
parser.add_argument('--num_levels',type=int,default=4,help="how many scales to evaluate complexity at")
parser.add_argument('--nz',type=int,default=2, help='dimension to reduce to before clustering')
parser.add_argument('--overwrite',action='store_true', help='overwrite experiment of the same name if it exists')
parser.add_argument('--print_times',action='store_true')
parser.add_argument('--run_other_methods',action='store_true', help='also compute complexity scores from existing methods')
parser.add_argument('--select_randomly',action='store_true', help='shuffle the dataset before looping through')
parser.add_argument('--show_df',action='store_true', help='print the results as a pandas dataframe to stdout')
parser.add_argument('--cluster_model',type=str,choices=['kmeans','cmeans','GMM'],default='GMM')
parser.add_argument('--verbose','-v',action='store_true')
parser.add_argument('--suppress_all_prints',action='store_true')
ARGS = parser.parse_args()

if ARGS.abls_only:
    ARGS.run_other_methods = True

exp_dir = f'experiments/{ARGS.exp_name}/{ARGS.dset}'
if (isfile(join(exp_dir,f'{ARGS.dset}_results.csv')) and
    not ARGS.exp_name.endswith('tmp') and not ARGS.overwrite):
    is_overwrite = get_user_yesno_answer(f'experiment {ARGS.exp_name}/{ARGS.dset} already exists, overwrite?')
    if not is_overwrite:
        print('aborting')
        sys.exit()

check_dir(exp_dir)
comp_meas = ComplexityMeasurer(ncs_to_check=ARGS.ncs_to_check,
                               n_cluster_inits=ARGS.n_cluster_inits,
                               nz=ARGS.nz,
                               num_levels=ARGS.num_levels,
                               cluster_model=ARGS.cluster_model,
                               no_cluster_idxify=ARGS.no_cluster_idxify,
                               compare_to_true_entropy=ARGS.compare_to_true_entropy,
                               info_subsample=ARGS.info_subsample,
                               print_times=ARGS.print_times,
                               display_cluster_label_imgs=ARGS.display_cluster_label_imgs,
                               display_scattered_clusters=ARGS.display_scattered_clusters,
                               suppress_all_prints=ARGS.suppress_all_prints,
                               verbose=ARGS.verbose)

methods = ['img_label','proc_time','total'] + [f'level {i+1}' for i in range(ARGS.num_levels)]
if ARGS.run_other_methods:
    methods += ['glcm','no_mdl','fract','ent','jpg','mach','khan','redies','no_patch']
results_df = pd.DataFrame(columns=methods)

img_start_times = []
img_times_real = []
labels = []
img_streamer = ImageStreamer(ARGS.dset,~ARGS.no_resize)
image_generator = img_streamer.stream_images(ARGS.num_ims,ARGS.downsample,ARGS.given_fname,
                                            ARGS.given_class_dir,ARGS.select_randomly)

for idx,(im,label) in enumerate(image_generator):
    img_label = label.split('_')[0] if ARGS.dset in ['im','dtd'] else label
    print(idx, img_label)
    plt.axis('off')
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
    if ARGS.abls_only:
        scores_at_each_level = [-np.ones(ARGS.num_levels)]*3
    else:
        scores_at_each_level = comp_meas.interpret(im)
    results_df.loc[idx,'img_label'] = img_label
    results_df.loc[idx,'proc_time'] = time()-img_start_time
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

results_df.index = results_df['img_label']
results_df = results_df.drop('img_label',axis=1)
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
print(results_df.loc['means'])
