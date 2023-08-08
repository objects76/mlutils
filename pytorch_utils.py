import sys
sys.path.append('.')

import os
import torch
from debug import callerline
import numpy as np



def filter_dataset(dset, class_list):
    idx = torch.tensor(dset.targets) == class_list[0]
    for i in range(1,len(class_list)):
        idx += torch.tensor(dset.targets) == class_list[i]
    print('filter_dataset: idx=', idx.shape)
    return torch.utils.data.dataset.Subset(dset, np.where(idx==1)[0])


def randoms_seed(SEED = 42):
    '''
    print("*** remove random for debugging ***")
    '''
    import torch
    import numpy as np
    import random
    print("*** remove random for debugging ***")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # cuda: it makes cuad slow
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

# torch.save({
#             'epoch': epoch,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'loss': loss,
#             ...
#             }, PATH)
# Load:
# model = TheModelClass(*args, **kwargs)
# optimizer = TheOptimizerClass(*args, **kwargs)

# checkpoint = torch.load(PATH)
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']
# loss = checkpoint['loss']
# is_primitive = isinstance(myvar, (int, float, bool))
class Checkpoint:
    '''
    '''
    def __init__(self, objs, checkpoint_file, do_load = True) -> None:
        self.file = checkpoint_file
        self.objs = objs
        if do_load:
            self.load()
        pass

    def save(self):
        print("=> Saving checkpoint to", self.file)
        states = {}
        for obj in self.objs:
            if hasattr(obj, 'state_dict'):
                key = Checkpoint._key(obj)
                assert key not in states # assert no dupl key.
                states[key] = obj.state_dict()
            else:
                print('No state_dict in', type(obj))
        torch.save(states, self.file)


    def load(self):
        if os.path.exists(self.file):
            print("=> Loading checkpoint from", self.file)
            cp = torch.load(self.file)
            for obj in self.objs:
                if hasattr(obj, 'load_state_dict'):
                    key = Checkpoint._key(obj)
                    obj.load_state_dict(cp[key])
                else:
                    print('No load_state_dict in', type(obj))

    @staticmethod
    def _key(obj):
        key = str(type(obj)).split('.')[-1]
        return key

#
# Tensor helper
#
#       torch.Tensor.assert_shape = assert_shape
#       torch.Tensor.info = tensor_info
#
def tensor_info(self:torch.Tensor, msg=None):
    print(f'[{callerline()}]',
        msg+": " if msg else '',
        str(self.shape).replace('torch.Size', 'shape '),
        self.device )

def assert_shape(self, shape, *args):
    '''
    v in args: v <= 0 or None: don't care.
    '''
    if type(shape) == list or type(shape) == tuple:
        expected_shape = torch.Size([value if value and value>0 else self.size(i) for i, value in enumerate(shape)])
    elif type(shape) == torch.Tensor:
        expected_shape = shape.shape
        shape = shape.shape
    else:
        raise ValueError('invalid shape:'+shape)

    assert self.shape == expected_shape, f'[{callerline()}] expect {shape}, but {self.shape}'

def isinstance_all(layers, type_):
    '''
    assert isinstance_all(self.enc_layers, nn.Module)
    '''
    if not isinstance(layers, type_):
        return False
    if hasattr(layers, '__iter__'):
        if not all(isinstance(layer, type_) for layer in layers):
            return False
    return True

# # Filesystem
#
import glob
import os

def get_latest_file(folder_path, file_type=''):
    '''
    get latest .ext file
    '''
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(folder_path)

    files = glob.glob(os.path.join(folder_path, '*' + file_type))
    if len(files) == 0:
        raise FileNotFoundError(os.path.join(folder_path, '*' + file_type))

    return max(files, key=os.path.getctime)

import sys
import random
import pathlib
def get_random_files(folder, ext='*', count=sys.maxsize, seed = None):
    path_gen = pathlib.Path(folder).rglob(ext)
    files = [f for f in path_gen]
    if count < len(files):
        if seed is not None:
            random.seed(seed)
        files = random.sample(files, count)
    return files



#
# torchvision
#
# show source images
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.utils
import matplotlib.pyplot as plt
import glob, random

def show_images(files, transform=None):
    '''
    convert files to [torchvision tensor]
    '''
    assert isinstance(files, (list,tuple)) and isinstance(files[0], str)
    if transform is None:
        transform = A.Compose([
            # A.LongestMaxSize(max_size=IMAGE_SIZE),
            # A.PadIfNeeded(min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT, value=[100,0,0]),
            ToTensorV2()])
    imgs = [ np.array(Image.open(f).convert("RGB")) for f in files]
    imgs = [transform(image=img)['image'] for img in imgs ]

    fig,ax = plt.subplots(figsize = (16*4,12*4))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(torchvision.utils.make_grid(imgs,nrow=10).permute(1,2,0))


''' split validataion_data
picked_img, picked_mask = get_random_files('./data/train', '*.jpg', 48, seed=0), get_random_files('./data/train_masks', '*.gif', 48, seed=0)
os.makedirs('./data/val/', exist_ok=True)
os.makedirs('./data/val_masks/', exist_ok=True)
for img,mask in zip(picked_img, picked_mask):
    assert img.stem in mask.stem, f'not matched: {img}, {mask}'

for img,mask in zip(picked_img, picked_mask):
    assert img.stem in mask.stem, f'not matched: {img}, {mask}'
    shutil.move(str(img), './data/val/'+img.name)
    shutil.move(str(mask), './data/val_masks/', mask.name)
print(f'{len(picked_img)} files are moved')

'''

if __name__ == '__main__':
    def function1():
        pass

    obj = torch.nn.Module()
    cp = Checkpoint([], 'aa')
    key = str(type(function1)).split('.')[-1]
    print(type(obj), type(cp), type(function1), key)
    print(function1.__name__)

