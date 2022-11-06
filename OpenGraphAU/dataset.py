import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset
import os


def make_dataset(image_list, label_list, au_relation=None):
    len_ = len(image_list)
    if au_relation is not None:
        images = [(image_list[i].strip(),  label_list[i, :],au_relation[i,:]) for i in range(len_)]
    else:
        images = [(image_list[i].strip(),  label_list[i, :]) for i in range(len_)]
    return images


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def default_loader(path):
    return pil_loader(path)


class HybridDataset(Dataset):
    def __init__(self, root_path, phase='train', transform=None, stage=1, loader=default_loader):

        assert stage>0 and stage <=2, 'The stage num must be restricted from 1 to 2'
        assert phase in ['train', 'val', 'test'], 'phase must be train, val or test'

        self._root_path = root_path
        self._phase = phase
        self._stage = stage
        self._transform = transform
        self.loader = loader
        self.img_folder_path = os.path.join(root_path,'img')
        if self._phase == 'train':
            # img
            train_image_list_path = os.path.join(root_path, 'list', 'hybrid_train_img_path.txt')
            train_image_list = open(train_image_list_path).readlines()
            # img labels
            train_label_list_path = os.path.join(root_path, 'list', 'hybrid_train_label.txt')
            train_label_list = np.loadtxt(train_label_list_path)

            # AU relation
            if self._stage == 2:
                au_relation_list_path = os.path.join(root_path, 'list', 'hybrid_train_AU_relation.txt')
                au_relation_list = np.loadtxt(au_relation_list_path)
                self.data_list = make_dataset(train_image_list, train_label_list, au_relation_list)
            else:
                self.data_list = make_dataset(train_image_list, train_label_list)
        elif self._phase == 'val':
            # img
            eval_image_list_path = os.path.join(root_path, 'list', 'hybrid_val_img_path.txt')
            eval_image_list = open(eval_image_list_path).readlines()

            # img labels
            eval_label_list_path = os.path.join(root_path, 'list', 'hybrid_val_label.txt')
            eval_label_list = np.loadtxt(eval_label_list_path)
            self.data_list = make_dataset(eval_image_list, eval_label_list)

        else:
            # img
            eval_image_list_path = os.path.join(root_path, 'list', 'hybrid_test_img_path.txt')
            eval_image_list = open(eval_image_list_path).readlines()

            # img labels
            eval_label_list_path = os.path.join(root_path, 'list', 'hybrid_test_label.txt')
            eval_label_list = np.loadtxt(eval_label_list_path)
            self.data_list = make_dataset(eval_image_list, eval_label_list)

    def __getitem__(self, index):
        if self._stage == 2 and self._phase == 'train':
            img, label, au_relation = self.data_list[index]
            img = self.loader(os.path.join(self.img_folder_path, img))
            if self._transform is not None:
                img = self._transform(img)
            return img, label, au_relation
        else:
            img, label = self.data_list[index]
            img = self.loader(os.path.join(self.img_folder_path, img))

            if self._transform is not None:
                img = self._transform(img)
            return img, label

    def __len__(self):
        return len(self.data_list)
