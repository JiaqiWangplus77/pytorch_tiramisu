import os
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
from torchvision.datasets.folder import is_image_file, default_loader

classes = ['0_intact_road','5_crack']
#classes = ['0_intact_road', '1_applied_patch', '2_pothole', 
#           '3_inlaid_patch', '4_open_joint', '5_crack']
#classes = ['Sky', 'Building', 'Column-Pole', 'Road',
#           'Sidewalk', 'Tree', 'Sign-Symbol', 'Fence', 'Car', 'Pedestrain',
#           'Bicyclist', 'Void']

# https://github.com/yandex/segnet-torch/blob/master/datasets/camvid-gen.lua
#class_weight = torch.FloatTensor([
#    0.58872014284134, 0.51052379608154, 2.6966278553009,
#    0.45021694898605, 1.1785038709641, 0.77028578519821, 2.4782588481903,
#    2.5273461341858, 1.0122526884079, 3.2375309467316, 4.1312313079834, 0])

# to test the principle  of class_weight
class_weight = torch.FloatTensor([6.25130619e-02, 8.05004041e-01, 5.62802143e+00, 1.33807906e+00,
       1.31966127e+00, 8.27907404e-01])
# original weight for crack: 1.27907404e-01
#  original weight for crack 2.25130619e-03
#mean = [0.3980712041733329]
#std = [0.15851423320841515]

#CFD
mean = [0.49623959446225074]
std = [0.060382144781743356]

#mean = [0.41189489566336, 0.4251328133025, 0.4326707089857]
#std = [0.27413549931506, 0.28506257482912, 0.28284674400252]

class_color = [
    (128, 128, 128), # gray  0_intact_road
#    (128, 0, 0),    # brown 1_applied_patch
#    (192, 192, 128), #   2_pothole  
#    (128, 64, 128),  #  3_inlaid_patch
#    (0, 0, 192),       #  4_open_joint blue+
    (128, 128, 0),    # 5_crack
    (192, 128, 128),
    (64, 64, 128),
    (64, 0, 128),
    (64, 64, 0),
    (0, 128, 192),
    (0, 0, 0),
]


def _make_dataset(dir):
    images = []
    for root, _, fnames in sorted(os.walk(dir)):
        # generate the file names in directory tree
        # root = 'SegNet-Tutorial/CamVid/train'
        # fnames: all the filenames in this folder
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                # 'SegNet-Tutorial/CamVid/train/0006R0_f02430.png'
                item = path
                images.append(item)
    return images


class LabelToLongTensor(object):
    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            label = torch.from_numpy(pic).long()
        else:
            label = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            label = label.view(pic.size[1], pic.size[0], 1)
            label = label.transpose(0, 1).transpose(0, 2).squeeze().contiguous().long()
        return label


class LabelTensorToPILImage(object):
    def __call__(self, label):
        label = label.unsqueeze(0)
        colored_label = torch.zeros(3, label.size(1), label.size(2)).byte()
        for i, color in enumerate(class_color):
            mask = label.eq(i)
            for j in range(3):
                colored_label[j].masked_fill_(mask, color[j])
        npimg = colored_label.numpy()
        npimg = np.transpose(npimg, (1, 2, 0))
        mode = None
        if npimg.shape[2] == 1:
            npimg = npimg[:, :, 0]
            mode = "L"

        return Image.fromarray(npimg, mode=mode)
    
#class RandomCrop(object):
#    def __call__(self, img, target,size):

#
#        return Image.fromarray(npimg, mode=mode)


class CamVid(data.Dataset):

    def __init__(self, root, split='train', joint_transform=None,
                 transform=None, target_transform=LabelToLongTensor(),
                 download=False,
                 loader=default_loader):
        self.root = root
        assert split in ('train', 'val', 'test')  # True if splis is in ()
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform
        self.loader = loader
#        self.class_weight = class_weight
        self.classes = classes  # name of each class
#        self.mean = mean
#        self.std = std

        if download:
            self.download()

        self.imgs = _make_dataset(os.path.join(self.root, self.split))
        # return empty if there is no files in the folder

    def __getitem__(self, index):            
        path = self.imgs[index]
        
#        if self.split == 'test':
#            print('filename:\n',path, '\n')
        #img = self.loader(path)
        img = Image.open(path).convert('L') # for single channel
        target_name = path.replace(self.split+'/', self.split + 'annot'+'/').replace('jpg','png')
        target = Image.open(target_name)

        if self.joint_transform is not None:
            img, target = self.joint_transform([img, target])
            

        if self.transform is not None:
            img = self.transform(img)

        target = self.target_transform(target)
        return img, target,path

    def __len__(self):
        return len(self.imgs)

    def download(self):
        # TODO: please download the dataset from
        # https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid
        raise NotImplementedError

class CamVid2(data.Dataset):

    def __init__(self, root, path='', joint_transform=None,
                 transform=None, target_transform=LabelToLongTensor(),
                 download=False,
                 loader=default_loader):
        self.root = root
        self.path = path
#        assert split in ('trainannot', 'valannot', 'testannot')  # True if splis is in ()
#        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform
        self.loader = loader
#        self.class_weight = class_weight
        self.classes = classes  # name of each class
#        self.mean = mean
#        self.std = std

        if download:
            self.download()

        with open(path, "r") as file:
            self.imgs = file.read().splitlines()
        # return empty if there is no files in the folder

    def __getitem__(self, index):            
        path = self.imgs[index]
        img = Image.open(path)
        target_path = path.replace("images", "labels").replace(".jpg", ".png")        
        target = Image.open(target_path)
        
        
        #target = Image.open(target_name)

        if self.joint_transform is not None:
            img, target = self.joint_transform([img, target])
            

        if self.transform is not None:
            img = self.transform(img)

        target = self.target_transform(target)
        return img, target,path

    def __len__(self):
        return len(self.imgs)

    def download(self):
        # TODO: please download the dataset from
        # https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid
        raise NotImplementedError
