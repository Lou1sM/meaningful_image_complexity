import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
import math
import pandas as pd
from measure_complexity import ComplexityMeasurer


comp_meas = ComplexityMeasurer(ncs_to_check=8,
                               n_cluster_inits=10,
                               nz=2,
                               num_levels=3,
                               cluster_model='GMM',
                               no_cluster_idxify=False,
                               compare_to_true_entropy=False,
                               info_subsample=1.,
                               print_times=False,
                               display_cluster_label_imgs=False,
                               display_scattered_clusters=False,
                               suppress_all_prints=True,
                               verbose=False)


with open('arecibo.txt') as f:
    arecibo_signal = np.array(list(f.read().replace('\n',''))).astype(float)*255

orig = arecibo_signal.reshape(73,23)
N = len(arecibo_signal)

results = []
rescale_each_axis_by = int(224 / (N**0.5)) # to give same n_pixels as a 224*224

for w in range(1,int(N**.5)):
    h = int(math.ceil(N/w))
    n_extra_bits_needed = w*h - N
    #scaled_h, scaled_w = h*rescale_each_axis_by, w*rescale_each_axis_by
    extra_junk = (np.random.rand(n_extra_bits_needed) > 0.5)
    extended_signal = np.concatenate([arecibo_signal, extra_junk])
    this_aspect_ratio = extended_signal.reshape([h,w])
    #resized = np.array(Image.fromarray(this_aspect_ratio).resize((scaled_w,scaled_h)))
    #resized = np.array(Image.fromarray(this_aspect_ratio).resize((224,224)))
    resized = this_aspect_ratio
    resized = np.tile(np.expand_dims(resized,2),(1,1,3))
    #plt.imshow(resized)
    #plt.savefig(f'aspect_ratios_of_arecibo/arecibo_{h}{w}.png')
    #os.system(f'/usr/bin/xdg-open aspect_ratios_of_arecibo/arecibo_{h}{w}.png')
    #print(f'\nOrig shape: {h},{w}, resized to{resized.shape}')
    #plt.show()
    #continue
    scores_at_each_level = comp_meas.interpret(resized)
    bitrand_like = (np.random.rand(*resized.shape) > 0.5).astype(float)
    br_scores_at_each_level = comp_meas.interpret(bitrand_like)
    this_results = {f'level {i+1}':v for i,v in enumerate(scores_at_each_level)}
    total = sum(scores_at_each_level)
    #print(f'arecibo: {total}\tbitrand: {sum(br_scores_at_each_level)}')
    #print(total)
    this_results['total'] = total
    this_results['height'] = h
    this_results['width'] = w
    results.append(this_results)


results_df = pd.DataFrame(results)
print(results_df)

