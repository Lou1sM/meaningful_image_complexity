import torchvision
import sys
from create_simple_imgs import create_simple_img
from PIL import Image
import pandas as pd
import numpy as np
from mdl_conv import ComplexityMeasurer
from os.path import join
from os import listdir
import math
from get_dsets import load_fpath
from dl_utils.misc import check_dir
from dl_utils.tensor_funcs import numpyify


def save_from_results_list(results_list,dset):
    results_arr = np.array([r for r in results_list if not math.isnan(r)])
    df = pd.DataFrame({'mean':results_arr.mean(), 'var':results_arr.var(), 'std':results_arr.std()},index=[dset])
    mdl_abl_dir = 'experiments/main_run/mdl_abls'
    check_dir(mdl_abl_dir)
    df.to_csv(join(mdl_abl_dir,f'{dset}_no_mdl_abl.csv'))
    np.save(join(mdl_abl_dir,f'{dset}_no_mdl_abl_raw.npy'),results_arr)
    print(df)

def get_used_ims(fpath):
    with open(fpath) as f:
        return [x.strip('\n') for x in f.readlines()]

if __name__ == '__main__':
    # im
    if sys.argv[1] == 'im':
        im_ims_used = get_used_ims('experiments/main_run_old/im_ims_used.txt')
        with open(f'experiments/main_run_old/im_ARGS.txt') as f:
            im_ARGS = f.readlines()
            im_ARGS = dict([x.strip('\n').split(': ') for x in im_ARGS])
            for k,v in im_ARGS.items():
                if v == 'False':
                    im_ARGS[k] = False
                elif v == 'True':
                    im_ARGS[k] = True
                else:
                    try:
                        im_ARGS[k] = int(v)
                    except ValueError:
                        try:
                            im_ARGS[k] = float(v)
                        except ValueError:
                            pass

        comp_meas = ComplexityMeasurer(**im_ARGS)
        comp_meas.is_mdl_abl = True
        results_list = []
        for fname in im_ims_used:
            print(fname)
            class_dir = fname.split('_')[0]
            if fname.startswith('ILSVRC'):
                for class_dir in listdir('imagenette2/val'):
                    try:
                        fpath = join('imagenette2/val',class_dir,fname)
                        print('trying', fpath)
                        im = load_fpath(fpath,is_resize=True)
                        break
                    except: pass
            else:
                class_dir = fname.split('_')[0]
                fpath = join('imagenette2/val',class_dir,fname)
                im = load_fpath(fpath,is_resize=True)
            if im.ndim == 2:
                im = np.stack([im]*3,axis=2)
            no_mdls, _, _ = comp_meas.interpret(im)
            results_list.append(sum(no_mdls))

        save_from_results_list(results_list,'im')

    # cifar
    elif sys.argv[1] == 'cifar':
        cifar_ims_used = get_used_ims('experiments/main_run/cifar/cifar_ims_used.txt')
        with open(f'experiments/main_run/cifar/cifar_ARGS.txt') as f:
            cifar_ARGS = f.readlines()
            cifar_ARGS = dict([x.strip('\n').split(': ') for x in cifar_ARGS])
            for k,v in cifar_ARGS.items():
                if v == 'False':
                    cifar_ARGS[k] = False
                elif v == 'True':
                    cifar_ARGS[k] = True
                else:
                    try:
                        cifar_ARGS[k] = int(v)
                    except ValueError:
                        try:
                            cifar_ARGS[k] = float(v)
                        except ValueError:
                            pass

        comp_meas = ComplexityMeasurer(**cifar_ARGS)
        comp_meas.is_mdl_abl = True
        results_list = []

        torch_dset = torchvision.datasets.CIFAR10(root='~/datasets',download=True,train=True)
        for i,im_used_idx in enumerate(cifar_ims_used):
            print(i)
            im = torch_dset.data[int(im_used_idx)]
            im = np.array(Image.fromarray(im).resize((224,224)))/255
            no_mdls, _, _ = comp_meas.interpret(im)
            results_list.append(sum(no_mdls))

        save_from_results_list(results_list,'cifar')

    elif sys.argv[1] == 'dtd':
        # dtd
        with open(f'experiments/main_run/dtd/dtd_ARGS.txt') as f:
            dtd_ARGS = f.readlines()
            dtd_ARGS = dict([x.strip('\n').split(': ') for x in dtd_ARGS])
            for k,v in dtd_ARGS.items():
                if v == 'False':
                    dtd_ARGS[k] = False
                elif v == 'True':
                    dtd_ARGS[k] = True
                else:
                    try:
                        dtd_ARGS[k] = int(v)
                    except ValueError:
                        try:
                            dtd_ARGS[k] = float(v)
                        except ValueError:
                            pass

        comp_meas = ComplexityMeasurer(**dtd_ARGS)
        comp_meas.is_mdl_abl = True
        results_list = []
        for i,fname in enumerate(listdir('dtd/suitable')):
            print(i)
            fpath = join('dtd/suitable',fname)
            im = load_fpath(fpath,is_resize=True)
            if im.ndim == 2:
                im = np.stack([im]*3,axis=2)
            no_mdls, _, _ = comp_meas.interpret(im)
            results_list.append(sum(no_mdls))

        save_from_results_list(results_list,'dtd')

    elif sys.argv[1] == 'mnist':
        # mnist
        mnist_ims_used = get_used_ims('experiments/main_run_old/mnist_ims_used.txt')
        with open(f'experiments/main_run_old/mnist_ARGS.txt') as f:
            mnist_ARGS = f.readlines()
            mnist_ARGS = dict([x.strip('\n').split(': ') for x in mnist_ARGS])
            for k,v in mnist_ARGS.items():
                if v == 'False':
                    mnist_ARGS[k] = False
                elif v == 'True':
                    mnist_ARGS[k] = True
                else:
                    try:
                        mnist_ARGS[k] = int(v)
                    except ValueError:
                        try:
                            mnist_ARGS[k] = float(v)
                        except ValueError:
                            pass

        comp_meas = ComplexityMeasurer(**mnist_ARGS)
        comp_meas.is_mdl_abl = True
        results_list = []
        torch_dset = torchvision.datasets.MNIST(root='~/datasets',download=True,train=True)
        for i,im_used_idx in enumerate(mnist_ims_used):
            print(i)
            im = numpyify(torch_dset.data[int(im_used_idx)])
            im = np.array(Image.fromarray(im).resize((224,224)))/255
            im = np.tile(np.expand_dims(im,2),(1,1,3))
            no_mdls, _, _ = comp_meas.interpret(im)
            results_list.append(sum(no_mdls))

        save_from_results_list(results_list,'mnist')

    elif sys.argv[1] == 'stripes':
        # stripes
        with open(f'experiments/main_run/stripes/stripes_ARGS.txt') as f:
            stripes_ARGS = f.readlines()
            stripes_ARGS = dict([x.strip('\n').split(': ') for x in stripes_ARGS])
            for k,v in stripes_ARGS.items():
                if v == 'False':
                    stripes_ARGS[k] = False
                elif v == 'True':
                    stripes_ARGS[k] = True
                else:
                    try:
                        stripes_ARGS[k] = int(v)
                    except ValueError:
                        try:
                            stripes_ARGS[k] = float(v)
                        except ValueError:
                            pass

        comp_meas = ComplexityMeasurer(**stripes_ARGS)
        comp_meas.is_mdl_abl = True
        results_list = []
        line_thicknesses = np.random.permutation(np.arange(3,10))
        for i in range(500):
            print(i)
            slope = np.random.rand()+.5
            line_thickness = line_thicknesses[i%len(line_thicknesses)]
            im = create_simple_img('stripes',slope,line_thickness)
            no_mdls, _, _ = comp_meas.interpret(im)
            results_list.append(sum(no_mdls))

        save_from_results_list(results_list,'stripes')

    elif sys.argv[1] == 'halves':
        # halves
        with open(f'experiments/main_run_old/halves_ARGS.txt') as f:
            halves_ARGS = f.readlines()
            halves_ARGS = dict([x.strip('\n').split(': ') for x in halves_ARGS])
            for k,v in halves_ARGS.items():
                if v == 'False':
                    halves_ARGS[k] = False
                elif v == 'True':
                    halves_ARGS[k] = True
                else:
                    try:
                        halves_ARGS[k] = int(v)
                    except ValueError:
                        try:
                            halves_ARGS[k] = float(v)
                        except ValueError:
                            pass

        comp_meas = ComplexityMeasurer(**halves_ARGS)
        comp_meas.is_mdl_abl = True
        results_list = []
        line_thicknesses = np.random.permutation(np.arange(3,10))
        for i in range(500):
            print(i)
            slope = np.random.rand()+.5
            im = create_simple_img('halves',slope,-1)
            no_mdls, _, _ = comp_meas.interpret(im)
            results_list.append(sum(no_mdls))

        save_from_results_list(results_list,'halves')

    elif sys.argv[1] == 'rand':
        # rand
        with open(f'experiments/main_run/rand/rand_ims_used.txt') as f:
            rand_ims_used = [x.strip('\n') for x in f.readlines()]
        with open(f'experiments/main_run/rand/rand_ARGS.txt') as f:
            rand_ARGS = f.readlines()
            rand_ARGS = dict([x.strip('\n').split(': ') for x in rand_ARGS])
            for k,v in rand_ARGS.items():
                if v == 'False':
                    rand_ARGS[k] = False
                elif v == 'True':
                    rand_ARGS[k] = True
                else:
                    try:
                        rand_ARGS[k] = int(v)
                    except ValueError:
                        try:
                            rand_ARGS[k] = float(v)
                        except ValueError:
                            pass

        comp_meas = ComplexityMeasurer(**rand_ARGS)
        comp_meas.is_mdl_abl = True
        results_list = []
        for i in range(500):
            print(i)
            im = np.random.rand(224,224,3)
            no_mdls, _, _ = comp_meas.interpret(im)
            results_list.append(sum(no_mdls))

        save_from_results_list(results_list,'rand')
