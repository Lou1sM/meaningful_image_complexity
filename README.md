# Code for a method that measures the meaningful complexity of images
Unlike various methods, like entropy of pixel values, which measure white noise as very complex, this method uses the minimum description length to measure only the amount of meaningful complexity. White noise images get a very low score.
The method is described in this paper: https://www.sciencedirect.com/science/article/pii/S0031320323005873.

## To use
Clone the repo and create a conda enviroment (tested with python 3.11).

Run

`. install_requirements_conda.sh` 

to install the required libraries.

Then use the method by running

`python main.py -d <dataset-name> --info_subsample 0.3 -n <number-of-images-to-test>`

The dataset name can be 'im' (imagenette2), 'cifar' (cifar10), 'mnist', 'stripes' (simple dataset we created for testing), 'halves' (simple dataset we created for test), or 'rand' (random noise).

The class implementing the complexity metric can also be imported:

```
from measure complexity import ComplexityMeasurer
import numpy as np

comp_meas = ComplexityMeasurer(ncs_to_check=8,                                           
                               n_cluster_inits=1,                                     
                               nz=2,
                               num_levels=4,
                               cluster_model=GMM,                                         
                               info_subsample=0.3,
                               print_times=False,
                               no_cluster_idxify=False,                                 
                               compare_to_true_entropy=False,            
                               display_cluster_label_imgs=False,               
                               display_scattered_clusters=False,          
                               verbose=False)

img = np.load(<path-to-img-file>)

complexity_of_img_at_each_level = comp_meas.interpret(img)
```

For questions or problems with the code, contact oneillml@tcd.ie.


## Citation
If you use or refer to this work, please cite 
```
@article{mahon2023minimum,
  title={Minimum Description Length Clustering to Measure Meaningful Image Complexity},
  author={Mahon, Louis and Lukasiewicz, Thomas},
  journal={Pattern Recognition},
  volume={144},
  year={2023}
}
```
