
import numpy as np
import torch
import cv2
from pathlib import Path

def npy_to_png(npypath, pngpath=''):
    mask = np.load(str(npypath))
    if not pngpath:
        pngpath = Path(npypath).with_suffix('.png')

    cv2.imwrite(str(pngpath), mask.astype(np.bool_)*255)

# npy_to_png('/mnt/ntfs/MLDataset/people-seg/web_scrapping/merged_jpegfiles/seggpt/Hash_005790bda193a154.npy')


#
#
#
class Mask:
    @staticmethod
    def shrink(mask, kernel_size=3, iterations=1):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        shrunk_mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=iterations)
        return shrunk_mask

    @staticmethod
    def inflate(mask, kernel_size=7, iterations=1):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        bgmask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=iterations)

    @staticmethod
    def close_hole(mask, kernel_size=7):
        # from skimage.morphology import disk, binary_opening, binary_closing
        # mask = binary_opening(mask, disk(kernel_size))
        # return binary_closing(mask, disk(kernel_size))
        from scipy import ndimage
        return ndimage.binary_fill_holes(mask.astype(np.uint8))


    @staticmethod
    def contour_points(mask, count=10) -> np.ndarray:
        '''
        return (N,2)
        '''
        # get outter points with contours
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # (nparray...)
        list_pts = []
        for c in contours:
            # print(f'\033[33mL50: {c.shape=}, {c.reshape((-1,2)).shape} \033[0m')
            c = c.reshape((-1, 2))
            indices = np.linspace(0, len(c)-1, num=count).astype(int) # using index
            list_pts.append(c[indices])

        if len(list_pts) == 0:
            return []
        return np.concatenate(list_pts, axis=0) # (N,2)

    @staticmethod
    def mask_to_rgba(mask:np.ndarray, clr=np.array([30/255, 144/255, 255/255, 0.6])) -> np.ndarray:
        mask = mask.astype(np.bool_)
        if clr is None:
            clr = np.array([*np.random.random(3), 0.6])
        return mask[...,None] * clr

    @staticmethod
    def iou(mask1, mask2):
        mask1 = mask1.astype(np.bool_)
        mask2 = mask2.astype(np.bool_)
        intersection = mask1 & mask2
        union = mask1 | mask2

        iou = np.sum(intersection) / np.sum(union)
        return iou

#
#
#
import re
def get_jpgfiles(str1):
    files = []
    str1 = re.sub(r'#.*\n', '', str1, flags=re.MULTILINE) # remove comments
    str1 = str1.replace('"', '').replace("'", '')

    for line in str1.splitlines():
        idx = line.find('.jpg') + 4
        if idx > 5:
            files.append(line[:idx].strip())
    return files

from pathlib import Path
from glob import glob

def get_best_matched_files(partial_names, workdir:Path):
    files = glob( str(workdir / "*.jpg"))

    hits = {}
    for name in partial_names:
        for f in files:
            if name in f:
                try:
                    hits.setdefault(name, f)
                except Exception as ex:
                    print(f'\033[33mL12: {ex} for {name} \033[0m')

    return hits