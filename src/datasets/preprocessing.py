import torch
import numpy as np
import random
import sys
from PIL import Image
import torchvision.transforms as transforms

def get_target_label_idx(labels, targets, train_size):
    """
    Get the indices of labels that are included in targets.
    :param labels: array of labels
    :param targets: list/tuple of target labels
    :return: list with indices of target labels
    """
    # train_idx_normal = np.argwhere(np.isin(labels, targets)).flatten()    
    all_labels = tuple([1,2,3,4,5,6,7,8,9,0])
    data_labels = list(all_labels)
    data_labels = tuple(data_labels)

    ##This part is to sample from whole dataset
    train_idx = []
    for l in data_labels:
        label_idx = np.argwhere(np.isin(labels , tuple([l]))).flatten()
        train_idx += reduce_train_idx(label_idx, train_size)

    return np.array(train_idx)
    
def reduce_train_idx(train_idx_normal, train_size):
    """
    random sample the training samples based on train_size number.
    """
    train_idx_normal = np.random.choice(train_idx_normal, size = train_size, replace = False).tolist()
    return train_idx_normal

def get_multi_domain(data_set, min_max, train: bool=True):
    """
    Generating multi domains with rotating each image by
    different degrees. 
    """
    new_set = []
    if train:
        angle_range = [0, 15, 30, 45, 60]
    else:
        angle_range = [75] # it should be just 75 since the test is just on 75
    for i, angle in enumerate(angle_range):
        for item in data_set:

            img, label, idx = item
            transform = transforms.Compose([transforms.RandomRotation(angle), transforms.ToTensor(),
                                        transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l1')),
                                        transforms.Normalize([min_max[label][0]],
                                                             [min_max[label][1] - min_max[label][0]])])
            img = torch.squeeze(img,0)
            # img = Image.fromarray(img.numpy(), mode='L') # if the number of channels are 3
            img = Image.fromarray(img.numpy())
            img = transform(img)
            if train:
                new_set += [(img, label, idx, i)]
            else:
                new_set += [(img, label, idx, 5)]

    return new_set





def global_contrast_normalization(x: torch.tensor, scale='l2'):
    """
    Apply global contrast normalization to tensor, i.e. subtract mean across features (pixels) and normalize by scale,
    which is either the standard deviation, L1- or L2-norm across features (pixels).
    Note this is a *per sample* normalization globally across features (and not across the dataset).
    """

    assert scale in ('l1', 'l2')

    n_features = int(np.prod(x.shape))
    mean = torch.mean(x)  # mean over all features (pixels) per sample
    x -= mean
    
    if scale == 'l1':
        x_scale = torch.mean(torch.abs(x))
    if scale == 'l2':
        x_scale = torch.sqrt(torch.sum(x ** 2)) / n_features
        
    x /= x_scale   
    return x

def rnd_obj_move(img):
    """
    Creating bounding box around the digits in mnist and moving them randomly 
    in different position of image.
    """
    np.set_printoptions(threshold=sys.maxsize)
    x = np.array(img)
    w, l = x.shape

    lst=[]
    lst = [[j,i] for i in range(l) for j in range(w) if x[i,j]>50]
    x1 = min(np.array(lst)[:,0])
    y1 = min(np.array(lst)[:,1])
    x2, y2 = max(np.array(lst)[:,0]), max(np.array(lst)[:,1])

    img_width = (x2 - x1)
    img_height = (y2 - y1)

    x_start = random.randrange(0, w - img_width)
    y_start = random.randrange(0, l - img_height)
    
    img = np.zeros_like(x)
    for y, i in zip(range(y_start, y_start+img_height), range(y1, y2)):
        for z, j in zip(range(x_start,x_start+img_width), range(x1, x2)):
            img[y,z]=x[i,j]

    return Image.fromarray(img)


