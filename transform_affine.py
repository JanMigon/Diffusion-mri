import numpy as np

from skimage.transform import AffineTransform, warp


def transform_image(image, scale=1, rotation=0, translation=0, shear=0):
    transform = AffineTransform(scale=scale, rotation=np.deg2rad(rotation), translation=translation,
                                shear=np.deg2rad(shear))
    transformed = warp(image, transform, order=1, preserve_range=True, mode='symmetric')
    return transformed

if __name__ == '__main__':


