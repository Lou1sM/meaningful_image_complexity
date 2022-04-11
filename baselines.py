from skimage.feature import graycomatrix
import numpy as np

def rgb2gray(rgb):
    if rgb.shape[2] == 1:
        return rgb[:,:,0]
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def fractal_dimension(Z_, threshold=None):
    threshold = .5
    assert Z_.ndim == 2

    # From https://github.com/rougier/numpy-100 (#87)
    def boxcount(Z, k):
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                               np.arange(0, Z.shape[1], k), axis=1)

        # We count non-empty (0) and non-full boxes (k*k)
        return len(np.where((S > 0) & (S < k*k))[0])

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
    print(counts)

    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]

def glcm_entropy(Z):
    glcm = np.squeeze(graycomatrix(Z, distances=[1],
                      angles=[0], symmetric=True,
                      normed=True))
    return -np.sum(glcm*np.log2(glcm + (glcm==0)))
