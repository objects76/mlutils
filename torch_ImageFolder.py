
import torchvision.datasets as datasets
from PIL import Image


import os
import sys
import tqdm

def get_files(rootdir, folders, filter=None, max_files_per_class=sys.maxsize):
    images = []
    rootdir = os.path.expanduser(rootdir)
    rootdir_len = len(rootdir)+1
    loop = tqdm.tqdm( folders )
    for target in loop:
        loop.set_postfix(dir=target)
        d = os.path.join(rootdir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d, followlinks=True)):
            cnt = 0
            for fname in sorted(fnames):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')):
                    path = os.path.join(root, fname)
                    if filter is None or filter(path):
                        images.append(path[rootdir_len:])
                        #images.append(path[rootdir_len:].replace("\\","/"))
                        cnt += 1
                        if cnt == max_files_per_class:
                            break
    return images



class ImageFolder2(datasets.VisionDataset):
    def __init__(self, root, loader=None,
        transform=None, target_transform=None,
        max_classes = sys.maxsize, max_files_per_class=sys.maxsize
    ):
        super().__init__(root, transform=transform, target_transform=target_transform)
        classes, class_to_idx = ImageFolder2._find_classes(self.root, max_classes)

        samples = ImageFolder2._make_dataset(self.root, class_to_idx, max_files_per_class)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root))

        self.loader = loader if loader is not None else ImageFolder2._pil_loader
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]


    def __getitem__(self, index):
        """
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _pil_loader(path):
        return Image.open(path)
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')

    @staticmethod
    def _find_classes(dir, max_classes):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()][:max_classes]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    @staticmethod
    def _make_dataset(dir, class_to_idx, max_files_per_class=sys.maxsize):
        images = []
        dir = os.path.expanduser(dir)
        for target in sorted(class_to_idx.keys()):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d, followlinks=True)):
                cnt = 0
                for fname in sorted(fnames):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')):
                        path = os.path.join(root, fname)
                        item = (path, class_to_idx[target])
                        images.append(item)
                        cnt += 1
                        if cnt == max_files_per_class:
                            break
        return images



