# coding:utf-8
'''
  Function set for image processing and computer vision
  Created   :  10, 19, 2015
  Revised   :   4,  4, 2018   total rewrite of `imread()` to add support of EXIF rotation handling
  All rights reserved
'''
__author__ = 'dawei.leng'

import numpy as np
from PIL import Image, ImageFile, ExifTags
ImageFile.LOAD_TRUNCATED_IMAGES = True

def imread(f, flatten=False, dtype='float32'):
    """
    Read image from file, backend=scipy
    f       : str or file object -> The file name or file object to be read.
    flatten : bool, optional  -> If True, flattens the color channels into a single gray-scale channel.
    Returns ndarray
    """
    # return sp.misc.imread(f, flatten)

    image=Image.open(f)
    
    try:
        # --- handle EXIF rotation ---#
        for orientation in ExifTags.TAGS.keys() :
            if ExifTags.TAGS[orientation]=='Orientation' : break
        exif=dict(image._getexif().items())
        if   exif[orientation] == 3 :
            image=image.rotate(180, expand=True)
        elif exif[orientation] == 6 :
            image=image.rotate(270, expand=True)
        elif exif[orientation] == 8 :
            image=image.rotate(90, expand=True)
    finally:
        #--- return proper numpy array ---#
        if dtype == 'uint8':
            if flatten:
                return np.array(image.convert('L'))
            else:
                return np.array(image.convert('RGB'))
        else: # dtype == 'float32'
            if flatten:
                return np.array(image.convert('F'))
            else:
                return np.array(image.convert('RGB'), dtype='float32')

def imsave(f, I, **params):
    """
    Save image into file, backend=pillow
    For jpeg, image should be in uint8 type
    :param f: str or ﬁle object
    :param I:
    :return:
    """
    image = Image.fromarray(I)
    return image.save(fp=f, format=None, **params)

def imresize(I, size, interp='bilinear', mode=None):
    import scipy.misc
    return scipy.misc.imresize(I, size, interp, mode)

def imrotate(I, angle, padvalue=0.0, interpolation=1, target_size=None):
    """
    Return a rotated image with border padded with padvalue
    :param I:  2D np array, dtype not limited
    :param angle:   in degree, positive for counter-clockwise
    :param padvalue:
    :param interpolation: default = cv2.INTER_LINEAR (1)
    :return: rotated image, dtype same with `I`
    """
    import cv2
    if abs(angle) < 0.01:
        return I
    rows, cols = I.shape[:2]
    rmatrix = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1.0)                       # 2 * 3
    vertex_matrix = np.array([[0, 0, 1], [cols, 0, 1], [0, rows, 1], [cols, rows, 1]]).T  # 3 * 4
    vertex_matrix_target = np.dot(rmatrix, vertex_matrix)                                 # 2 * 4
    rowmin, colmin = vertex_matrix_target.min(axis=1)
    rowmax, colmax = vertex_matrix_target.max(axis=1)
    newshape = (rowmax-rowmin, colmax-colmin)
    rmatrix[0, 2] += newshape[0]/2 - cols/2
    rmatrix[1, 2] += newshape[1]/2 - rows/2
    if target_size is not None:
        newshape = target_size
    else:
        newshape = (np.round(newshape[0]).astype('int'), np.round(newshape[1]).astype('int'))
    I2 = cv2.warpAffine(I, rmatrix, newshape,
                        flags=interpolation,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=padvalue)
    return I2


