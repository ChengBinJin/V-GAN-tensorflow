# ---------------------------------------------------------
# Tensorflow Vessel-GAN (V-GAN) Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin, based on code from Jaemin Son
# ---------------------------------------------------------
import os
import random
import numpy as np
from datetime import datetime

import utils as utils


class Dataset(object):
    def __init__(self, dataset, flags):
        self.dataset = dataset
        self.flags = flags

        self.image_size = (640, 640) if self.dataset == 'DRIVE' else (720, 720)
        self.ori_shape = (584, 565) if self.dataset == 'DRIVE' else (605, 700)
        self.val_ratio = 0.1  # 10% of the training data are used as validation data
        self.train_dir = "../data/{}/training/".format(self.dataset)
        self.test_dir = "../data/{}/test/".format(self.dataset)

        self.num_train, self.num_val, self.num_test = 0, 0, 0

        self._read_data()  # read training, validation, and test data)
        print('num of training images: {}'.format(self.num_train))
        print('num of validation images: {}'.format(self.num_val))
        print('num of test images: {}'.format(self.num_test))

    def _read_data(self):
        if self.flags.is_test:
            # real test images and vessels in the memory
            self.test_imgs, self.test_vessels, self.test_masks, self.test_mean_std = utils.get_test_imgs(
                target_dir=self.test_dir, img_size=self.image_size, dataset=self.dataset)
            self.test_img_files = utils.all_files_under(os.path.join(self.test_dir, 'images'))

            self.num_test = self.test_imgs.shape[0]

        elif not self.flags.is_test:
            random.seed(datetime.now())  # set random seed
            self.train_img_files, self.train_vessel_files, mask_files = utils.get_img_path(
                self.train_dir, self.dataset)

            self.num_train = int(len(self.train_img_files))
            self.num_val = int(np.floor(self.val_ratio * int(len(self.train_img_files))))
            self.num_train -= self.num_val

            self.val_img_files = self.train_img_files[-self.num_val:]
            self.val_vessel_files = self.train_vessel_files[-self.num_val:]
            val_mask_files = mask_files[-self.num_val:]
            self.train_img_files = self.train_img_files[:-self.num_val]
            self.train_vessel_files = self.train_vessel_files[:-self.num_val]

            # read val images and vessels in the memory
            self.val_imgs, self.val_vessels, self.val_masks, self.val_mean_std = utils.get_val_imgs(
                self.val_img_files, self.val_vessel_files, val_mask_files, img_size=self.image_size)

            self.num_val = self.val_imgs.shape[0]

    def train_next_batch(self, batch_size):
        train_indices = np.random.choice(self.num_train, batch_size, replace=True)
        train_imgs, train_vessels = utils.get_train_batch(
            self.train_img_files, self.train_vessel_files, train_indices.astype(np.int32),
            img_size=self.image_size)
        train_vessels = np.expand_dims(train_vessels, axis=3)

        return train_imgs, train_vessels
