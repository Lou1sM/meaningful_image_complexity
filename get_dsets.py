import numpy as np
import pickle
from dl_utils.tensor_funcs import numpyify
from create_simple_imgs import create_simple_img
import sys
from PIL import Image
from os import listdir
from os.path import join
import struct


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

def load_fpath(fpath,is_resize,downsample):
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
        if downsample != -1:
            im = im.resize((downsample,downsample))
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
            with open('cifar-10-batches-py/data_batch_1', 'rb') as f:
                d = pickle.load(f, encoding='bytes')
                imgs = d[b'data']
                imgs = np.transpose(imgs.reshape((-1,3,32,32)),(0,2,3,1))
                labels = d[b'labels']
                self.prepared_dset = list(zip(imgs,labels))
        elif dset == 'mnist':
            with open('mnist_data/t10k-images-idx3-ubyte','rb') as f:
                magic, size = struct.unpack(">II", f.read(8))
                nrows, ncols = struct.unpack(">II", f.read(8))
                imgs = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
                imgs = imgs.reshape((size, nrows, ncols))
            with open('mnist_data/t10k-labels-idx1-ubyte','rb') as f:
                magic, size = struct.unpack(">II", f.read(8))
                labels = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
            self.prepared_dset = list(zip(imgs,labels))
        elif dset == 'rand':
            self.prepared_dset = np.random.rand(1000,224,224,3)
        elif dset == 'bitrand':
            self.prepared_dset = (np.random.rand(1000,224,224,3) > 0.5).astype(float)
        elif dset == 'stripes':
            self.line_thicknesses = np.random.permutation(np.arange(3,10))

    def stream_images(self,num_ims,downsample,given_fname='none',given_class_dir='none',select_randomly=False):
        if self.dset in ['cifar','mnist','usps']:
            indices = np.random.choice(len(self.prepared_dset),size=num_ims,replace=False)
        elif self.dset == 'dtd':
            n = min(len(listdir('dtd/suitable')),num_ims)
            indices = range(n)
        elif self.dset == 'fractal_imgs':
            n = min(len(listdir('fractal_imgs')),num_ims)
            indices = range(n)
        else: indices = range(num_ims)

        for i in indices:
            if self.dset in ['im','dtd']:
                if self.dset=='im':
                    num_classes = len(listdir(self.dset_dir))
                    if select_randomly:
                        class_dir = np.random.choice(listdir(self.dset_dir))
                        fname = np.random.choice(listdir(join(self.dset_dir,class_dir)))
                    else:
                        class_dir = given_class_dir if given_class_dir != 'none' else listdir(self.dset_dir)[i%num_classes]
                        idx_within_class = i//num_classes
                        fname = given_fname if given_fname != 'none' else listdir(join(self.dset_dir,class_dir))[idx_within_class]
                    fpath = join(self.dset_dir,class_dir,fname)
                elif self.dset=='dtd':
                    try:
                        fname = given_fname if given_fname != 'none' else listdir(self.dset_dir)[i]
                    except IndexError:
                        print(f"have run out of images, at image number {i}")
                        sys.exit()
                    fpath = join(self.dset_dir,fname)
                im = load_fpath(fpath,self.is_resize,downsample)
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
            elif self.dset in['rand', 'bitrand']:
                im = self.prepared_dset[i]
                label = self.dset
            elif self.dset == 'fractal_imgs':
                fname = listdir('fractal_imgs')[i]
                fpath = join('fractal_imgs',fname)
                im = load_fpath(fpath,self.is_resize,downsample)
                label = 'fract_dim' + fname.split('.')[0][-1]
            elif self.dset in ['cifar']:
                im,label = self.prepared_dset[i]
                im = np.array(Image.fromarray(im).resize((224,224)))/255
            elif self.dset == 'mnist':
                im,label = self.prepared_dset[i]
                #im = np.array(Image.fromarray(im).resize((224,224)))
                im = np.tile(np.expand_dims(im,2),(1,1,3))
            else:
                print("INVALID DSET NAME:", self.dset)
            yield im, label
