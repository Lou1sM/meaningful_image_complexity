from skimage.feature import graycomatrix
import numpy as np
from PIL import Image
from io import BytesIO
from scipy import ndimage


def rgb2gray(rgb):
    if rgb.shape[2] == 1:
        return rgb[:,:,0]
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def boxcount(im_to_reduce, box_size):
    y_reduce_idxs = np.arange(0, im_to_reduce.shape[0], box_size)
    x_reduce_idxs = np.arange(0, im_to_reduce.shape[1], box_size)
    y_axis_reduced = np.add.reduceat(im_to_reduce, y_reduce_idxs, axis=0)
    reduced = np.add.reduceat(y_axis_reduced, x_reduce_idxs, axis=1)

    # We count non-empty (0) and non-full boxes (k*k)
    return len(np.where((reduced > 0) & (reduced < box_size**2))[0])

def compute_fractal_dimension(Z_, threshold=None):
    threshold = .5
    assert Z_.ndim == 2
    Z = (Z_ < threshold)
    p = min(Z.shape)

    # Greatest power of 2 less than or equal to p
    n = 2**np.floor(np.log(p)/np.log(2))

    # Extract the exponent
    n = int(np.log(n)/np.log(2))

    # Build successive box sizes (from 2**n down to 2**1)
    sizes = 2**np.arange(n, 1, -1)

    # Actual box counting with decreasing size
    counts = []
    for size in sizes:
        counts.append(boxcount(Z, size))

    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]

def glcm_entropy(Z):
    glcm = np.squeeze(graycomatrix(Z, distances=[1],
                      angles=[0], symmetric=True,
                      normed=True))
    return -np.sum(glcm*np.log2(glcm + (glcm==0)))

def jpg_compressed_size(im):
    x = im - im.min()
    x = im/im.max()
    x = (x*255).astype(np.uint8)
    if x.shape[2] == 1:
        x = np.squeeze(x,2)
    pim = Image.fromarray(x)
    compressed = BytesIO()
    pim.save(compressed,format='jpeg')
    return compressed.getbuffer().nbytes

def jpg_compression_ratio(im):
    full_size = 8 * im.size
    compressed_size = jpg_compressed_size(im)
    return compressed_size/full_size

def machado2015(im):
    sob = ndimage.sobel(im)
    return jpg_compression_ratio(sob)

def khan2021(im):
    return np.median(abs(np.fft.fft2((255*im).astype(np.uint8))))
