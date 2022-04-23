import matplotlib.pyplot as plt
import numpy as np


def add_colour_dimension(a):
    return np.stack([a]*3,axis=2)

def row_for_stripes(line_thickness,offset):
    offset = offset % (2*line_thickness)
    b = np.array([0]*line_thickness + [255]*line_thickness)
    n_repeats = int((224-offset)/(2*line_thickness))
    a = np.tile(b,n_repeats)
    if offset > 0:
        a = np.concatenate((b[-offset:],a))
    remainder = 224 - len(a)
    assert remainder >= 0
    a = np.concatenate((a,b[:remainder]))
    assert a.ndim == 1 and a.shape[0] == 224
    return a

def diagonal_stripes(slope,line_thickness):
    rows = [row_for_stripes(line_thickness,int(i*slope)) for i in range(224)]
    a = np.stack(rows)
    return add_colour_dimension(a)

def half_and_half(slope):
    rows = [np.arange(224)>(112*(1-slope)+(i*slope)) for i in range(224)]
    a = np.array(rows)*255
    return add_colour_dimension(a)

def create_simple_img(img_type,slope,line_thickness,as_float=True):
    if img_type == 'stripes':
        return diagonal_stripes(slope,line_thickness)/255
    elif img_type == 'halves':
        return half_and_half(slope)/255
    else:
        print(f"Unrecognised simple image type: {img_type}")

if __name__ == "__main__":
    plt.imshow(diagonal_stripes(5,1/2)); plt.show()
