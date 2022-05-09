import numpy as np
from dl_utils.tensor_funcs import numpyify
import torchvision
from create_simple_imgs import create_simple_img
import sys
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
    if dset=='imagenette':
        dset_dir = 'imagenette2/val'
        class_dir = np.random.choice(listdir(dset_dir))
        class_dir_path = join(dset_dir,class_dir)
    elif dset=='dtd':
        class_dir_path = 'dtd/suitable'
    fname = np.random.choice(listdir(class_dir_path))
    fpath = join(class_dir_path,fname)
    print(fname)
    return load_fpath(fpath,is_resize)

def load_fpath(fpath,is_resize):
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
    return np.array(im)

def generate_non_torch_im(dset,is_resize,subsample):
    if dset=='imagenette':
        dset_dir = 'imagenette2/val'
    elif dset=='dtd':
        dset_dir = 'dtd/suitable'
    for i in range(subsample):
        if dset=='imagenette':
            num_classes = len(listdir(dset_dir))
            class_dir = listdir(dset_dir)[i%num_classes]
            idx_within_class = i//num_classes
            fname = listdir(join(dset_dir,class_dir))[idx_within_class]
            fpath = join(dset_dir,class_dir,fname)
        elif dset=='dtd':
            try:
                fname = listdir(dset_dir)[i]
            except IndexError:
                print(f"have run out of images, at image number {i}")
                sys.exit()
            fpath = join(dset_dir,fname)
        yield load_fpath(fpath,is_resize), fpath

class ImageStreamer():
    def __init__(self,dset,is_resize):
        self.dset = dset
        self.is_resize = is_resize
        if dset=='im':
            self.dset_dir = 'imagenette2/val'
        elif dset=='dtd':
            self.dset_dir = 'dtd/suitable'
        elif dset == 'cifar':
            self.prepared_dset = torchvision.datasets.CIFAR10(root='~/datasets',download=True,train=True)
        elif dset == 'mnist':
            self.prepared_dset = torchvision.datasets.MNIST(root='~/datasets',train=False,download=True)
        elif dset == 'rand':
            self.prepared_dset = np.random.rand(1000,224,224,3)
        elif dset == 'stripes':
            self.line_thicknesses = np.random.permutation(np.arange(3,10))


    def stream_images(self,num_ims):
        if self.dset in ['cifar','mnist','usps']:
            indices = np.random.choice(len(self.prepared_dset),size=num_ims,replace=False)
        elif self.dset == 'dtd':
            n = min(len(listdir('dtd/suitable')),num_ims)
            indices = range(n)
        else: indices = range(num_ims)

        for i in indices:
            if self.dset in ['im','dtd']:
                if self.dset=='im':
                    num_classes = len(listdir(self.dset_dir))
                    class_dir = listdir(self.dset_dir)[i%num_classes]
                    idx_within_class = i//num_classes
                    fname = listdir(join(self.dset_dir,class_dir))[idx_within_class]
                    fpath = join(self.dset_dir,class_dir,fname)
                elif self.dset=='dtd':
                    try:
                        fname = listdir(self.dset_dir)[i]
                    except IndexError:
                        print(f"have run out of images, at image number {i}")
                        sys.exit()
                    fpath = join(self.dset_dir,fname)
                im = load_fpath(fpath,self.is_resize)
                im = im/255
                if im.ndim == 2:
                    im = np.resize(im,(*(im.shape),1))
                label = fname
            elif self.dset == 'stripes':
                slope = np.random.rand()+.5
                line_thickness = self.line_thicknesses[i%len(self.line_thicknesses)]
                im = create_simple_img('stripes',slope,line_thickness)
                label = f'stripes-{line_thickness}'
            elif self.dset == 'halves':
                slope = np.random.rand()+.5
                im = create_simple_img('halves',slope,-1)
                label = 'halves'
            elif self.dset == 'rand':
                im = self.prepared_dset[i]
                label = 'rand'
            else:
                if self.dset == 'cifar':
                    im = self.prepared_dset.data[i]
                    im = np.array(Image.fromarray(im).resize((224,224)))/255
                    label = str(self.prepared_dset.targets[i])
                elif self.dset == 'mnist':
                    im = numpyify(self.prepared_dset.data[i])
                    im = np.array(Image.fromarray(im).resize((224,224)))
                    im = np.tile(np.expand_dims(im,2),(1,1,3))
                    label = str(self.prepared_dset.targets[i].item())
            yield im, label

