import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset


class Cifar:
    def __init__(self, batch_size, threads, img_size=32, root=None, dataset="cifar10", cutout=False, pre_trained=False, bz_test=None, custom={},
                 train_transform=None, test_transform=None):
        """

        :param batch_size:
        :param threads:
        :param root:
        :param dataset:
        :param cutout:
        :param pre_trained:
        :param bz_test:
        :param custom: {'type': "fft", "data": [ ]}
        """
        if bz_test is None:
            bz_test = batch_size
        if not root:
            root = './data'
        # if pre_trained or dataset == "tinyimagenet":
        if train_transform is None:
            if dataset == "cifar10":
                mean = (0.4914, 0.4822, 0.4465)
                std = (0.2471, 0.2435, 0.2616)
            elif dataset == "cifar100":
                mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
                std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
            else:
                mean, std = self._get_statistics(root, dataset=dataset)
            train_transformer_lst = [
                transforms.RandomCrop(size=(img_size, img_size), padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]
            train_transform = transforms.Compose(train_transformer_lst)

            test_transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

        if dataset == "cifar10":
            train_set = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=train_transform)
            test_set = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=test_transform)
            self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        elif dataset == "cifar100":
            train_set = torchvision.datasets.CIFAR100(root=root,
                                         train=True,
                                         download=True,
                                         transform=train_transform)
            test_set = torchvision.datasets.CIFAR100(root=root,
                                        train=False,
                                        download=True,
                                        transform=test_transform)
            self.classes = train_set.classes
        # elif dataset == "tinyimagenet":
        #     root = os.path.join(root, "tiny-imagenet-200")
        #
        #     val_path = os.path.join(root, "val")
        #     restore_tiny_imagenet(val_path)
        #
        #     train_path = os.path.join(root, "train")
        #     train_set = torchvision.datasets.ImageFolder(train_path, transform=train_transform)
        #     test_set = torchvision.datasets.ImageFolder(os.path.join(val_path, "images"),
        #                                    transform=test_transform)
        #
        #     self.classes = [str(i) for i in range(200)]

        self.train = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=threads)
        self.test = torch.utils.data.DataLoader(test_set, batch_size=bz_test, shuffle=False, num_workers=threads)

    def _get_statistics(self, root, dataset="cifar10"):
        if dataset == "cifar10":
            train_set = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transforms.ToTensor())
        elif dataset == "cifar100":
            train_set = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=transforms.ToTensor())
        else:
            return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        data = torch.cat([d[0] for d in DataLoader(train_set)])
        return data.mean(dim=[0, 2, 3]), data.std(dim=[0, 2, 3])


def restore_tiny_imagenet(VALID_DIR):
    # Create separate validation subfolders for the validation images based on
    # their labels indicated in the val_annotations txt file
    val_img_dir = os.path.join(VALID_DIR, 'images')

    # Open and read val annotations text file
    fp = open(os.path.join(VALID_DIR, 'val_annotations.txt'), 'r')
    data = fp.readlines()

    # Create dictionary to store img filename (word 0) and corresponding
    # label (word 1) for every line in the txt file (as key value pair)
    val_img_dict = {}
    for line in data:
        words = line.split('\t')
        val_img_dict[words[0]] = words[1]
    fp.close()

    # Create subfolders (if not present) for validation images based on label,
    # and move images into the respective folders
    count = 0
    for img, folder in val_img_dict.items():
        newpath = (os.path.join(val_img_dir, folder))
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        if os.path.exists(os.path.join(val_img_dir, img)):
            os.rename(os.path.join(val_img_dir, img), os.path.join(newpath, img))
        else:
            count += 1
        if count > val_img_dict.__len__()//2:
            print("Val set is already separated")
            return
    print("Complete separate val set")