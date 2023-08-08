
from argparse import ArgumentError
import sys
import subprocess
import os
import torch
from PIL import Image
import random

import matplotlib.colors as mcolors
from matplotlib import pyplot as plt
import matplotlib

matplotlib.rcParams['font.family'] ='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] =False

def get_plots(num_plots, ncols=6, figsize_per_plot=(5,5)):
    '''
    figsize_per_plot=(width_inch, height_inch)
    '''
    ncols = min(num_plots, ncols)
    nrows = int(num_plots/ncols)
    if nrows*ncols < num_plots: nrows += 1

    fig, plots = plt.subplots(nrows=nrows, ncols=ncols, facecolor='white')
    if figsize_per_plot is not None:
        fig.set_figwidth(ncols*figsize_per_plot[0])
        fig.set_figheight(nrows*figsize_per_plot[1])

    reverse_plots = plots.flatten()[::-1]
    for i in range(nrows*ncols - num_plots):
        reverse_plots[i].axis('off')
    return fig, plots


def show_images(images, labels=None, suptitle='', facecolor='white', img_inch = 4., ncols=3, save_path=None, show=True):
    '''
    facecolor: random, white, ...
    img_inch: float or [w,h]
    '''
    if images is None:
        raise ValueError('images is None in show_images')

    if type(images) not in [list, dict, tuple]: # for single image
        images = [images]

    if type(labels) not in [list, dict, tuple]: # for single image
        labels = [labels for _ in range(len(images))]

    if type(img_inch) not in [list, tuple]:
        img_inch = [img_inch, img_inch]

    image_count = len(images)
    nrows = int(image_count / ncols)
    if nrows*ncols < image_count: nrows += 1

    if facecolor.lower() == 'random':
         facecolor = random.sample(list(mcolors.CSS4_COLORS), 1)[0]

    fig = plt.figure(figsize=(int(img_inch[0]*ncols), int(img_inch[1]*nrows)), facecolor=facecolor)

    for idx, (image,title) in enumerate(zip(images, labels)):
        if isinstance(image, str):
            try:
                image = Image.open(image).convert("RGB")
            except Exception as ex:
                print("Can't open", image)
                image = None

        elif isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0) # c,h,w -> h,w,c

        plt.subplot(nrows, ncols, idx+1)
        if title: plt.title(title, fontsize=12)
        plt.axis('off')
        if image is not None: plt.imshow(image)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.suptitle(suptitle, fontsize=20)
    fig.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        if show: plt.show()
    else:
        plt.show()
    plt.close(fig)



def openImage(paths):
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

