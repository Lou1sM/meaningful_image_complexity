import numpy as np
from PIL import Image
from os import listdir
from os.path import join


imagenette_synset_dict = {
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

def load_rand(dset,is_resize=False):
    dset_dir = 'imagenette2/val' if dset=='imagenette' else 'dtd/images'
    class_dir = np.random.choice(listdir(dset_dir))
    class_dir_path = join(dset_dir,class_dir)
    fname = np.random.choice(listdir(class_dir_path))
    fpath = join(class_dir_path,fname)
    im = Image.open(fpath)
    if is_resize:
        h,w = im.size[:2]
        aspect_ratio = h/w
        new_h = (224*224*aspect_ratio)**0.5
        new_w = 224*224/new_h
        new_h_int = round(new_h)
        new_w_int = round(new_w)
        max_possible_error = (new_h_int + new_w_int) / 2
        if not (new_h_int*new_w_int - 224*224) < max_possible_error:
            breakpoint()
        im = im.resize((new_h_int,new_w_int))
    class_name = imagenette_synset_dict[class_dir] if dset=='imagenette' else class_dir
    return np.array(im), class_name
