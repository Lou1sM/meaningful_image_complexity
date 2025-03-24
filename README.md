# Code for a method that measures the meaningful complexity of images
Unlike various methods, such as entropy of pixel values, which measure white noise as very complex, this method uses the minimum description length to measure only the amount of meaningful complexity. White noise images get a very low score.
The method is described in this paper: https://www.sciencedirect.com/science/article/pii/S0031320323005873.

## To use
Clone the repo and create a conda enviroment (tested with python 3.11).

Run

`. install_requirements_conda.sh` 

to install the required libraries.

Then, to apply the metric to your own images, you can import the ComplexityMeasurer class as follows:

```
from measure_complexity import ComplexityMeasurer
import numpy as np

comp_meas = ComplexityMeasurer(ncs_to_check=8,
                               n_cluster_inits=1,
                               nz=2,
                               num_levels=4,
                               cluster_model='GMM',
                               info_subsample=0.3,
                               )

img = np.load(<path-to-img-file>)

complexity_of_img_at_each_level = comp_meas.interpret(img)
```
The result, `complexity_of_img_at_each_level`, will be a list of floats giving the complexity scores at each level of the input image. It will be of length 4 in this example, corresponding to the argument in initialising the class. The first element of the list is the score at the most local level, which picks out fine detail, the last element is the score at the most coarse-grained level, which picks out global structure. To get an overall score for the image, as in the paper, use `sum(complexity_of_img_at_each_level)`.

If you want to reproduce the results in the paper, run

`python main.py -d <dataset-name> --info_subsample 0.3 -n <number-of-images-to-test>`

The dataset name can be 'im' (imagenette2), 'cifar' (cifar10), 'mnist', 'stripes' (simple dataset we created for testing), 'halves' (simple dataset we created for test), or 'rand' (random noise).


For questions or problems with the code, contact lmahonology@gmail.com.


## Citation
If you use or refer to this work, please cite 
```
@article{mahon2024minimum,
title = {Minimum description length clustering to measure meaningful image complexity},
journal = {Pattern Recognition},
volume = {145},
pages = {109889},
year = {2024},
issn = {0031-3203},
doi = {https://doi.org/10.1016/j.patcog.2023.109889},
url = {https://www.sciencedirect.com/science/article/pii/S0031320323005873},
author = {Louis Mahon and Thomas Lukasiewicz},
}
```
