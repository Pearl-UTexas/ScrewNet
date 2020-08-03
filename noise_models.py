########################################################################################################
# Code forked from :
# Abbatematteo, Ben (2019), GeneralizingKinematics, https://github.com/babbatem/GeneralizingKinematics
# Code for the paper: B. Abbatematteo, S. Tellex, and G. Konidaris.
# Learning to Generalize Kinematic Models to Novel Objects.
# In Proceedings of the Third Conference on Robot Learning, 2019
########################################################################################################

import skimage
import numpy as np
import torch


class Distractor(object):
    ''' Add one distractor object to the background of the first '''

    def __init__(self, trans_weight=0.0, rotate=True):
        self.trans_weight = trans_weight
        self.rotate = rotate

    def __call__(self, x, x2):
        x = x.float()
        x2 = x2.float()
        x_mask = (x > 0)
        x2_mask = (x2 > 0)
        randy = torch.rand(1)
        trans = self.trans_weight * randy + (x.max() + x2.min())
        # trans = (x.max() + x2.min())
        translated = (x2 + trans) * x2_mask.float()
        if self.rotate:
            angle = torch.rand(1) * 360
            rotated = skimage.transform.rotate(translated, angle)
            rotated = torch.tensor(rotated).float()
        else:
            rotated = translated
        composed = (1 - x_mask.float()) * rotated + x
        return composed


class Noise(object):
    '''Add noise to depth images!'''

    def __init__(self, loc, scale):
        self.dist = torch.distributions.normal.Normal(loc, scale)

    def __call__(self, depth):
        return depth + self.dist.sample(sample_shape=depth.size())


class DropPixelsMasked(object):
    """Set pixels (on the object) to zero with probability p"""

    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, x):
        imgx = x.shape[1]
        imgy = x.shape[2]
        foreground = (x < 0.99)
        background = (x > 0.99)
        noise = np.random.choice([1.0, 0.9], size=108 * 192, p=[1 - self.p, self.p]).astype(int).reshape(imgx, imgy)
        noise_torch = torch.tensor(noise)
        masked_noise = foreground.float() * noise_torch.float()
        noised_train = x * (masked_noise + background.float())
        return noised_train


class DropPixels(object):
    '''Drop pixels all over the image.'''

    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, x):
        imgx = x.shape[1]
        imgy = x.shape[2]
        noise = np.random.choice([1.0, 0.9], size=108 * 192, p=[1 - self.p, self.p]).astype(int).reshape(imgx, imgy)
        noise_torch = torch.tensor(noise).float()
        return x * noise_torch


# %% sim data
def test():
    import matplotlib.pyplot as plt
    noiser = DropPixels()
    for i in range(1000):
        x = torch.load('data/micro/microwave/depth' + str(i).zfill(6) + '.pt')
        x = noiser(x)
        plt.imshow(x.numpy())
        plt.show()


if __name__ == '__main__':
    test()
