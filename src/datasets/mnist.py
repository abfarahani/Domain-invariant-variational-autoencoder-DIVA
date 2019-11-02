import numpy as np
from torch.utils.data import Subset
from PIL import Image
from torchvision.datasets import MNIST
from base.torchvision_dataset import TorchvisionDataset
from .preprocessing import get_target_label_idx, get_multi_domain, global_contrast_normalization, rnd_obj_move


import torchvision.transforms as transforms


class MNIST_Dataset(TorchvisionDataset):

    def __init__(self, root: str, normal_class=0, train_size=float('inf')):#, classification=False, org_data=False):
        super().__init__(root)
        # self.classification = classification       
        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = tuple([normal_class]) #if 5 is normal then normal_classes is (5,)
        self.outlier_classes = list(range(0, 10))
        self.outlier_classes.remove(normal_class)
        self.normal_class = normal_class
        # Pre-computed min and max values (after applying GCN) from train data per class
        min_max = [(-0.8826567065619495, 9.001545489292527),
                   (-0.6661464580883915, 20.108062262467364),
                   (-0.7820454743183202, 11.665100841080346),
                   (-0.7645772083211267, 12.895051191467457),
                   (-0.7253923114302238, 12.683235701611533),
                   (-0.7698501867861425, 13.103278415430502),
                   (-0.778418217980696, 10.457837397569108),
                   (-0.7129780970522351, 12.057777597673047),
                   (-0.8280402650205075, 10.581538445782988),
                   (-0.7369959242164307, 10.697039838804978)]

        # MNIST preprocessing: GCN (with L1 norm) and min-max feature scaling to [0,1]
        #transforms.Grayscale(3), 
        #for 32x32: transforms.Pad(padding=2)     transforms.Resize(32)
        #for 64x64: transforms.Pad(padding=18)    transforms.Resize(64)
        #for 128x128: transforms.Pad(padding=50)  transforms.Resize(128)
        # transform = transforms.Compose([transforms.ToTensor(),
        #                                 transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l1')),
        #                                 transforms.Normalize([min_max[normal_class][0]]*3,
        #                                                      [min_max[normal_class][1] - min_max[normal_class][0]]*3)])
        # target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))

        # train_set = MyMNIST(root=self.root, train=True, download=True, transform=transform, target_transform=target_transform)
        
        # if classification:

        transform = transforms.ToTensor()
        train_set = MyMNIST(root=self.root, train=True, download=True, transform=transform, target_transform=None)
        # lst = train_set.train_labels      
        train_idx = get_target_label_idx(train_set.train_labels.clone().data.cpu().numpy(), self.normal_classes, train_size)
            
        self.train_set = Subset(train_set, train_idx)

        # rotating the images to create different domains
        self.train_set = get_multi_domain(self.train_set, min_max, train=True)

        # self.train_size = len(train_idx)
        self.test_set = MyMNIST(root=self.root, train=False, download=True,
                                transform=transform, target_transform=None)
        self.test_set = get_multi_domain(self.test_set, min_max, train=False)


class MyMNIST(MNIST):
    """Torchvision MNIST class with patch of __getitem__ method to also return the index of a data sample."""

    def __init__(self, *args, **kwargs):
        super(MyMNIST, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        """Override the original method of the MNIST class.
        Args:
            index (int): Index
        Returns:
            triple: (image, target, index) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image

        img = Image.fromarray(img.numpy(), mode='L')
        #print('$$$$$$$$$$$$$$$$$', len(img))
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index  # only line changed
