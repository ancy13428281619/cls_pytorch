import os
import cv2
import numpy as np
import torch
import torch.utils.data
import torchvision
from torchvision import transforms
from utils.utils import loadYaml
from base_datalayer import BaseDataLayer
import albumentations as albu


class Datalayer(BaseDataLayer):

    def __init__(self, config, train=True, transform=None):
        super(Datalayer, self).__init__()
        print("Loading data")
        self.config = config
        train_dir = self.config['Dataset']['TrainPath']

        bg_imgs_dir = os.path.join(train_dir, 'bg')
        self.bg_imgs_path = [os.path.join(bg_imgs_dir, bg_img_name) for bg_img_name in os.listdir(bg_imgs_dir) if
                             bg_img_name.endswith('.png')]

        ng_imgs_dir = os.path.join(train_dir, 'ng')
        self.ng_imgs_path = [os.path.join(ng_imgs_dir, ng_img_name) for ng_img_name in os.listdir(ng_imgs_dir) if
                             ng_img_name.endswith('.png')]

        self.transform = transform

    def get_training_augmentation(self):
        # 增强库 https://github.com/albu/albumentations
        train_transform = [
            albu.HorizontalFlip(p=0.5),
            albu.VerticalFlip(p=0.5)
        ]
        return albu.Compose(train_transform)

    def __len__(self):
        return len(self.bg_imgs_path) + len(self.ng_imgs_path)

    def __getitem__(self, item):
        # bg
        if np.random.random() > 0.5:
            random_id_bg = np.random.randint(0, len(self.bg_imgs_path))
            img_path, label = self.bg_imgs_path[random_id_bg], 0.0
        # ng
        else:
            random_id_ng = np.random.randint(0, len(self.ng_imgs_path))
            img_path, label = self.ng_imgs_path[random_id_ng], 1.0
        img = cv2.imread(img_path)
        img = self.get_training_augmentation()(image=img)
        img = img['image']
        img = np.array(img, np.float32)
        img = img / 255.0

        if self.transform is not None:
            img = self.transform(img)
        return img, label


if __name__ == '__main__':
    config = loadYaml('../config/config.yaml')
    dl = Datalayer(config=config)
    for i, (img, label) in enumerate(dl):
        img_npy = img

        label_info = 'bg' if label == 0 else 'ng'

        cv2.putText(img_npy, 'label: {}'.format(label_info), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
        cv2.imshow('img', img_npy)

        cv2.waitKey(0)
