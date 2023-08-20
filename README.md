# Code for a method that measures the meaningful complexity of images
Unlike various methods, like entropy of pixel values, which measure white noise as very complex, this method uses the minimum description length to measure only the amount of meaningful complexity. White nose images get a very low score.
The method is described in this paper: https://www.sciencedirect.com/science/article/pii/S0031320323005873

# To use
Clone the repo and create a conda enviroment (tested with python 3.11).
Run . install_conda_requirements.sh to install the required libraries. 
python main.py -d <dataset-name> --info_subsample 0.3 -n <number-of-images-to-test>
The dataset name can be 'im' (imagenette2), 'cifar' (cifar10), 'mnist', 'stripes' (simple dataset we created for testing), 'halves' (simple dataset we created for test), or 'rand' (random noise).

# Citation
If you use or refer to this work, please cite 
@article{mahon2023minimum,
  title={Minimum Description Length Clustering to Measure Meaningful Image Complexity},
  author={Mahon, Louis and Lukasiewicz, Thomas},
  journal={Pattern Recognition},
  volume={144},
  year={2023}
}
