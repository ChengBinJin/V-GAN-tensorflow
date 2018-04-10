# ---------------------------------------------------------
# Tensorflow Vessel-GAN (V-GAN) Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# ---------------------------------------------------------
import os
import time
import collections
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
from PIL import Image

from dataset import Dataset
# noinspection PyPep8Naming
import TensorFlow_utils as tf_utils
import utils as utils
from model import CGAN


class Solver(object):
    def __init__(self, flags):
        run_config = tf.ConfigProto()
        run_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=run_config)

        self.flags = flags
        self.dataset = Dataset(self.flags.dataset, self.flags)
        self.model = CGAN(self.sess, self.flags, self.dataset.image_size)

        self.best_auc_sum = 0.
        self._make_folders()

        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

        tf_utils.show_all_variables()

    def _make_folders(self):
        self.model_out_dir = "{}/model_{}_{}_{}".format(self.flags.dataset, self.flags.discriminator,
                                                        self.flags.train_interval, self.flags.batch_size)
        if not os.path.isdir(self.model_out_dir):
            os.makedirs(self.model_out_dir)

        if self.flags.is_test:
            self.img_out_dir = "{}/seg_result_{}_{}_{}".format(self.flags.dataset,
                                                               self.flags.discriminator,
                                                               self.flags.train_interval,
                                                               self.flags.batch_size)
            self.auc_out_dir = "{}/auc_{}_{}_{}".format(self.flags.dataset, self.flags.discriminator,
                                                        self.flags.train_interval, self.flags.batch_size)

            if not os.path.isdir(self.img_out_dir):
                os.makedirs(self.img_out_dir)
            if not os.path.isdir(self.auc_out_dir):
                os.makedirs(self.auc_out_dir)

        elif not self.flags.is_test:
            self.sample_out_dir = "{}/sample_{}_{}_{}".format(self.flags.dataset, self.flags.discriminator,
                                                              self.flags.train_interval, self.flags.batch_size)
            if not os.path.isdir(self.sample_out_dir):
                os.makedirs(self.sample_out_dir)

    def train(self):
        for iter_time in range(0, self.flags.iters+1, self.flags.train_interval):
            self.sample(iter_time)  # sampling images and save them

            # train discrminator
            for iter_ in range(1, self.flags.train_interval+1):
                x_imgs, y_imgs = self.dataset.train_next_batch(batch_size=self.flags.batch_size)
                d_loss = self.model.train_dis(x_imgs, y_imgs)
                self.print_info(iter_time + iter_, 'd_loss', d_loss)

            # train generator
            for iter_ in range(1, self.flags.train_interval+1):
                x_imgs, y_imgs = self.dataset.train_next_batch(batch_size=self.flags.batch_size)
                g_loss = self.model.train_gen(x_imgs, y_imgs)
                self.print_info(iter_time + iter_, 'g_loss', g_loss)

            auc_sum = self.eval(iter_time, phase='train')

            if self.best_auc_sum < auc_sum:
                self.best_auc_sum = auc_sum
                self.save_model(iter_time)

    def test(self):
        if self.load_model():
            print(' [*] Load Success!\n')
            self.eval(phase='test')
        else:
            print(' [!] Load Failed!\n')

    def sample(self, iter_time):
        if np.mod(iter_time, self.flags.sample_freq) == 0:
            idx = np.random.choice(self.dataset.num_val, 2, replace=False)
            x_imgs, y_imgs = self.dataset.val_imgs[idx], self.dataset.val_vessels[idx]
            samples = self.model.sample_imgs(x_imgs)

            # masking
            seg_samples = utils.remain_in_mask(samples, self.dataset.val_masks[idx])

            # crop to original image shape
            x_imgs_ = utils.crop_to_original(x_imgs, self.dataset.ori_shape)
            seg_samples_ = utils.crop_to_original(seg_samples, self.dataset.ori_shape)
            y_imgs_ = utils.crop_to_original(y_imgs, self.dataset.ori_shape)

            # sampling
            self.plot(x_imgs_, seg_samples_, y_imgs_, iter_time, idx=idx, save_file=self.sample_out_dir,
                      phase='train')

    def plot(self, x_imgs, samples, y_imgs, iter_time, idx=None, save_file=None, phase='train'):
        # initialize grid size
        cell_size_h, cell_size_w = self.dataset.ori_shape[0] / 100, self.dataset.ori_shape[1] / 100
        num_columns, margin = 3, 0.05
        width = cell_size_w * num_columns
        height = cell_size_h * x_imgs.shape[0]
        fig = plt.figure(figsize=(width, height))  # (column, row)
        gs = gridspec.GridSpec(x_imgs.shape[0], num_columns)  # (row, column)
        gs.update(wspace=margin, hspace=margin)

        # convert from normalized to original image
        x_imgs_norm = np.zeros_like(x_imgs)
        std, mean = 0., 0.
        for _ in range(x_imgs.shape[0]):
            if phase == 'train':
                std = self.dataset.val_mean_std[idx[_]]['std']
                mean = self.dataset.val_mean_std[idx[_]]['mean']
            elif phase == 'test':
                std = self.dataset.test_mean_std[idx[_]]['std']
                mean = self.dataset.test_mean_std[idx[_]]['mean']
            x_imgs_norm[_] = np.expand_dims(x_imgs[_], axis=0) * std + mean
        x_imgs_norm = x_imgs_norm.astype(np.uint8)

        # 1 channel to 3 channels
        samples_3 = np.stack((samples, samples, samples), axis=3)
        y_imgs_3 = np.stack((y_imgs, y_imgs, y_imgs), axis=3)

        imgs = [x_imgs_norm, samples_3, y_imgs_3]
        for col_index in range(len(imgs)):
            for row_index in range(x_imgs.shape[0]):
                ax = plt.subplot(gs[row_index * num_columns + col_index])
                plt.axis('off')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_aspect('equal')
                plt.imshow(imgs[col_index][row_index].reshape(
                    self.dataset.ori_shape[0], self.dataset.ori_shape[1], 3), cmap='Greys_r')

        if phase == 'train':
            plt.savefig(save_file + '/{}_{}.png'.format(str(iter_time), idx[0]), bbox_inches='tight')
            plt.close(fig)
        else:
            # save compared image
            plt.savefig(os.path.join(save_file, 'compared_{}.png'.format(os.path.basename(
                self.dataset.test_img_files[idx[0]])[:-4])), bbox_inches='tight')
            plt.close(fig)

            # save vessel alone, vessel should be uint8 type
            Image.fromarray(np.squeeze(samples*255).astype(np.uint8)).save(os.path.join(
                save_file, '{}.png'.format(os.path.basename(self.dataset.test_img_files[idx[0]][:-4]))))

    def print_info(self, iter_time, name, loss):
        if np.mod(iter_time, self.flags.print_freq) == 0:
            ord_output = collections.OrderedDict([(name, loss), ('dataset', self.flags.dataset),
                                                  ('discriminator', self.flags.discriminator),
                                                  ('train_interval', np.float32(self.flags.train_interval)),
                                                  ('gpu_index', self.flags.gpu_index)])
            utils.print_metrics(iter_time, ord_output)

    def eval(self, iter_time=0, phase='train'):
        total_time, auc_sum = 0., 0.
        if np.mod(iter_time, self.flags.eval_freq) == 0:
            num_data, imgs, vessels, masks = None, None, None, None
            if phase == 'train':
                num_data = self.dataset.num_val
                imgs = self.dataset.val_imgs
                vessels = self.dataset.val_vessels
                masks = self.dataset.val_masks
            elif phase == 'test':
                num_data = self.dataset.num_test
                imgs = self.dataset.test_imgs
                vessels = self.dataset.test_vessels
                masks = self.dataset.test_masks

            generated = []
            for iter_ in range(num_data):
                x_img = imgs[iter_]
                x_img = np.expand_dims(x_img, axis=0)  # (H, W, C) to (1, H, W, C)

                # measure inference time
                start_time = time.time()
                generated_vessel = self.model.sample_imgs(x_img)
                total_time += (time.time() - start_time)

                generated.append(np.squeeze(generated_vessel, axis=(0, 3)))  # (1, H, W, 1) to (H, W)

            generated = np.asarray(generated)
            # calculate measurements
            auc_sum = self.measure(generated, vessels, masks, num_data, iter_time, phase, total_time)

            if phase == 'test':
                # save test images
                segmented_vessel = utils.remain_in_mask(generated, masks)

                # crop to original image shape
                imgs_ = utils.crop_to_original(imgs, self.dataset.ori_shape)
                cropped_vessel = utils.crop_to_original(segmented_vessel, self.dataset.ori_shape)
                vessels_ = utils.crop_to_original(vessels, self.dataset.ori_shape)

                for idx in range(num_data):
                    self.plot(np.expand_dims(imgs_[idx], axis=0),
                              np.expand_dims(cropped_vessel[idx], axis=0),
                              np.expand_dims(vessels_[idx], axis=0),
                              'test', idx=[idx], save_file=self.img_out_dir, phase='test')

        return auc_sum

    def measure(self, generated, vessels, masks, num_data, iter_time, phase, total_time):
        # masking
        vessels_in_mask, generated_in_mask = utils.pixel_values_in_mask(
            vessels, generated, masks)

        # averaging processing time
        avg_pt = (total_time / num_data) * 1000  # average processing tiem

        # evaluate Area Under the Curve of ROC and Precision-Recall
        auc_roc = utils.AUC_ROC(vessels_in_mask, generated_in_mask)
        auc_pr = utils.AUC_PR(vessels_in_mask, generated_in_mask)

        # binarize to calculate Dice Coeffient
        binarys_in_mask = utils.threshold_by_otsu(generated, masks)
        dice_coeff = utils.dice_coefficient_in_train(vessels_in_mask, binarys_in_mask)
        acc, sensitivity, specificity = utils.misc_measures(vessels_in_mask, binarys_in_mask)
        score = auc_pr + auc_roc + dice_coeff + acc + sensitivity + specificity

        # auc_sum for saving best model in training
        auc_sum = auc_roc + auc_pr

        # print information
        ord_output = collections.OrderedDict([('auc_pr', auc_pr), ('auc_roc', auc_roc),
                                              ('dice_coeff', dice_coeff), ('acc', acc),
                                              ('sensitivity', sensitivity), ('specificity', specificity),
                                              ('score', score), ('auc_sum', auc_sum),
                                              ('best_auc_sum', self.best_auc_sum), ('avg_pt', avg_pt)])
        utils.print_metrics(iter_time, ord_output)

        # write in tensorboard when in train mode only
        if phase == 'train':
            self.model.measure_assign(
                auc_pr, auc_roc, dice_coeff, acc, sensitivity, specificity, score, iter_time)
        elif phase == 'test':
            # write in npy format for evaluation
            utils.save_obj(vessels_in_mask, generated_in_mask,
                           os.path.join(self.auc_out_dir, "auc_roc.npy"),
                           os.path.join(self.auc_out_dir, "auc_pr.npy"))

        return auc_sum

    def save_model(self, iter_time):
        self.model.best_auc_sum_assign(self.best_auc_sum)

        model_name = "iter_{}_auc_sum_{:.3}".format(iter_time, self.best_auc_sum)
        self.saver.save(self.sess, os.path.join(self.model_out_dir, model_name))

        print('===================================================')
        print('                     Model saved!                  ')
        print(' Best auc_sum: {:.3}'.format(self.best_auc_sum))
        print('===================================================\n')

    def load_model(self):
        print(' [*] Reading checkpoint...')

        ckpt = tf.train.get_checkpoint_state(self.model_out_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(self.model_out_dir, ckpt_name))

            self.best_auc_sum = self.sess.run(self.model.best_auc_sum)
            print('====================================================')
            print('                     Model saved!                   ')
            print(' Best auc_sum: {:.3}'.format(self.best_auc_sum))
            print('====================================================')

            return True
        else:
            return False
