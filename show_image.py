
from argparse import ArgumentError
import sys
import subprocess
import os
from matplotlib import pyplot as plt
import torch
from PIL import Image
import math
import matplotlib.colors as mcolors
import random
from pathlib import Path


#
# for segmentation
#

import cv2
import torch
import numpy as np

def fill_small_closed(mask:np.ndarray):
    from skimage.morphology import disk, binary_opening, binary_closing
    closed_mask = mask.copy()
    level = 3
    closed_mask = binary_opening(closed_mask, disk(level))
    closed_mask = binary_closing(closed_mask, disk(level))
    return closed_mask

def show_seg(image:np.ndarray, mask:np.ndarray = None, name:str = ""):
    if isinstance(image, torch.Tensor): image = image.numpy()
    if isinstance(mask, torch.Tensor): mask = mask.numpy()

    plt.title(name)
    if image.shape[0] <= 4: # CHW?
        image = image.transpose(1, 2, 0)
    if mask is None:
        plt.imshow(image) # for visualization we have to transpose back to HWC
        return

    mask = mask.squeeze()  # BHW?

    mask = fill_small_closed(mask)
    # print(info(image))
    # print(info(mask))

    # green contours
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image = cv2.drawContours(image if image.flags['C_CONTIGUOUS'] else image.copy(), contours, -1, (0, 255, 0), 1)
    plt.imshow(image) # for visualization we have to transpose back to HWC

    mask = mask[...,None] * np.array([1,0,1,0.3])
    plt.imshow(mask)
    plt.title(name)

#
#
#
def get_plots(num_plots, ncols=3, figsize_per_plot=(3,3)):
    '''
    figsize_per_plot=(width_inch, height_inch)
    '''
    nrows = math.ceil(num_plots/ncols)
    fig, plots = plt.subplots(nrows=nrows, ncols=ncols, subplot_kw = dict( facecolor = "white"))
    if figsize_per_plot is not None:
        fig.set_figwidth(ncols*figsize_per_plot[0])
        fig.set_figheight(nrows*figsize_per_plot[1])

    reverse_plots = plots.flatten()[::-1]
    for i in range(nrows*ncols - num_plots):
        reverse_plots[i].axis('off')
    return fig, plots

def show_image(image, title=""):

    if isinstance(image, (str, Path)):
        image = Path(image)
        assert image.exists(), "No file:"+str(image)
        image = cv2.cvtColor(cv2.imread(str(image)), cv2.COLOR_BGR2RGB)
        # image = Image.open(image)

    elif isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0) # c,h,w -> h,w,c

    if title: plt.title(title, fontsize=7)
    plt.axis('off')
    plt.imshow(image)
    plt.show()

def show_images(images, titles=None, facecolor='white', img_inch = 2., ncols=3, save_path=None, show=True):
    if images is None:
        raise ValueError('images is None in show_images')

    if type(images) not in [set, list, dict , tuple]: # for single image
        images = [images]

    if type(titles) not in [set, list, dict , tuple]: # for single image
        titles = [ titles for _ in range(len(images))]

    image_count = len(images)
    row = math.ceil(image_count / ncols)

    if facecolor.lower() == 'random':
         facecolor = random.sample(list(mcolors.CSS4_COLORS), 1)[0]

    fig = plt.figure(figsize=(int(img_inch*ncols), int(img_inch*row)), dpi=150, facecolor=facecolor)

    for idx, (image,title) in enumerate(zip(images, titles)):
        if isinstance(image, str):
            image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
            # image = Image.open(image)

        elif isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0) # c,h,w -> h,w,c

        plt.subplot(row, ncols, idx+1)
        if title: plt.title(title, fontsize=7)
        plt.axis('off')
        # plt.tight_layout()
        plt.imshow(image)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        if show: plt.show()
    else:
        plt.show()
    plt.close(fig)



def openImage(paths):
    """ open image with external viewer """
    imageViewerFromCommandLine = {'linux':'xdg-open',
                                  'win32':'explorer',
                                  'darwin':'open'}[sys.platform]
    if type(paths) not in [list, dict , tuple]: # for single image
        paths = [paths]
    for p in paths:
        if os.path.exists(p):
            subprocess.run([imageViewerFromCommandLine, p])
        else:
            print("No file", p)

