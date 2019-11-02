from .mnist import MNIST_Dataset
from .cifar10 import CIFAR10_Dataset
from .svhn import SVHN_Dataset
from .stl10 import STL10_Dataset


def load_dataset(dataset_name, data_path, normal_class  , train_size): #, classification, org_data):
    """Loads the dataset."""

    implemented_datasets = ('mnist', 'cifar10', 'svhn', 'stl10')
    assert dataset_name in implemented_datasets

    dataset = None

    if dataset_name == 'mnist':
        dataset = MNIST_Dataset(root=data_path, normal_class=normal_class, train_size=train_size)#, classification=classification, org_data=org_data)

    # if dataset_name == 'cifar10':
    #     dataset = CIFAR10_Dataset(root=data_path, normal_class=normal_class, train_size=train_size)#, classification=classification, org_data=org_data)

    # if dataset_name == 'svhn':
    #     dataset = SVHN_Dataset(root=data_path, normal_class=normal_class, train_size=train_size)#, classification=classification, org_data=org_data)
    
    # if dataset_name == 'stl10':
    #     dataset = STL10_Dataset(root=data_path, normal_class=normal_class, train_size=train_size)#, classification=classification, org_data=org_data)

    return dataset
