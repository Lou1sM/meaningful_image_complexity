import numpy as np
from PIL import Image
from os import listdir
from os.path import join


synset_dict = {
    'n01440764': 'tench',
    'n02102040': 'spaniel',
    'n02979186': 'cassette player',
    'n03000684': 'chain saw',
    'n03028079': 'church',
    'n03394916': 'French horn',
    'n03417042': 'garbage truck',
    'n03425413': 'gas pump',
    'n03445777': 'golf ball',
    'n03888257': 'parachute'
    }

def load_rand_imagenette_val(is_resize=False):
    class_dir = np.random.choice(listdir('imagenette2/val'))
    class_dir_path = join('imagenette2/val',class_dir)
    fname = np.random.choice(listdir(class_dir_path))
    fpath = join(class_dir_path,fname)
    im = Image.open(fpath)
    if is_resize:
        breakpoint()
        h,w = im.size[:2]
        aspect_ratio = h/w
        new_h = (224*224*aspect_ratio)**0.5
        new_w = 224*224/new_h
        new_h_int = round(new_h)
        new_w_int = round(new_w)
        assert (new_h_int*new_w_int - 224*224) < min(new_h,new_w)
        im.resize((new_h_int,new_w_int))
    class_name = synset_dict[class_dir]
    return np.array(im), class_name
