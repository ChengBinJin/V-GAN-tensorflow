# ---------------------------------------------------------
# Tensorflow Vessel-GAN (V-GAN) Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin, based on code from Jaemin Son
# ---------------------------------------------------------
import os
import sys

import pickle
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageEnhance
from skimage import filters
from scipy.ndimage import rotate
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc, confusion_matrix


def get_img_path(target_dir, dataset):
    img_files, vessel_files, mask_files = None, None, None
    if dataset == 'DRIVE':
        img_files, vessel_files, mask_files = DRIVE_files(target_dir)
    elif dataset == 'STARE':
        img_files, vessel_files, mask_files = STARE_files(target_dir)

    return img_files, vessel_files, mask_files


# noinspection PyPep8Naming
def STARE_files(data_path):
    img_dir = os.path.join(data_path, "images")
    vessel_dir = os.path.join(data_path, "1st_manual")
    mask_dir = os.path.join(data_path, "mask")

    img_files = all_files_under(img_dir, extension=".ppm")
    vessel_files = all_files_under(vessel_dir, extension=".ppm")
    mask_files = all_files_under(mask_dir, extension=".ppm")

    return img_files, vessel_files, mask_files


# noinspection PyPep8Naming
def DRIVE_files(data_path):
    img_dir = os.path.join(data_path, "images")
    vessel_dir = os.path.join(data_path, "1st_manual")
    mask_dir = os.path.join(data_path, "mask")

    img_files = all_files_under(img_dir, extension=".tif")
    vessel_files = all_files_under(vessel_dir, extension=".gif")
    mask_files = all_files_under(mask_dir, extension=".gif")

    return img_files, vessel_files, mask_files


def load_images_under_dir(path_dir):
    files = all_files_under(path_dir)
    return imagefiles2arrs(files)


def all_files_under(path, extension=None, append_path=True, sort=True):
    if append_path:
        if extension is None:
            filenames = [os.path.join(path, fname) for fname in os.listdir(path)]
        else:
            filenames = [os.path.join(path, fname)
                         for fname in os.listdir(path) if fname.endswith(extension)]
    else:
        if extension is None:
            filenames = [os.path.basename(fname) for fname in os.listdir(path)]
        else:
            filenames = [os.path.basename(fname)
                         for fname in os.listdir(path) if fname.endswith(extension)]

    if sort:
        filenames = sorted(filenames)

    return filenames


def imagefiles2arrs(filenames):
    img_shape = image_shape(filenames[0])
    images_arr = None

    if len(img_shape) == 3:
        images_arr = np.zeros((len(filenames), img_shape[0], img_shape[1], img_shape[2]), dtype=np.float32)
    elif len(img_shape) == 2:
        images_arr = np.zeros((len(filenames), img_shape[0], img_shape[1]), dtype=np.float32)

    for file_index in range(len(filenames)):
        img = Image.open(filenames[file_index])
        images_arr[file_index] = np.asarray(img).astype(np.float32)

    return images_arr


def get_train_batch(train_img_files, train_vessel_files, train_indices, img_size):
    batch_size = len(train_indices)
    batch_img_files, batch_vessel_files = [], []
    for _, idx in enumerate(train_indices):
        batch_img_files.append(train_img_files[idx])
        batch_vessel_files.append(train_vessel_files[idx])

    # load images
    fundus_imgs = imagefiles2arrs(batch_img_files)
    vessel_imgs = imagefiles2arrs(batch_vessel_files) / 255
    fundus_imgs = pad_imgs(fundus_imgs, img_size)
    vessel_imgs = pad_imgs(vessel_imgs, img_size)
    assert (np.min(vessel_imgs) == 0 and np.max(vessel_imgs) == 1)

    # random mirror flipping
    for idx in range(batch_size):
        if np.random.random() > 0.5:
            fundus_imgs[idx] = fundus_imgs[idx, :, ::-1, :]  # flipped imgs
            vessel_imgs[idx] = vessel_imgs[idx, :, ::-1]  # flipped vessel

    # flip_index = np.random.choice(batch_size, int(np.ceil(0.5 * batch_size)), replace=False)
    # fundus_imgs[flip_index] = fundus_imgs[flip_index, :, ::-1, :]  # flipped imgs
    # vessel_imgs[flip_index] = vessel_imgs[flip_index, :, ::-1]  # flipped vessel

    # random rotation
    for idx in range(batch_size):
        angle = np.random.randint(360)
        fundus_imgs[idx] = random_perturbation(rotate(input=fundus_imgs[idx], angle=angle, axes=(0, 1),
                                                      reshape=False, order=1))
        vessel_imgs[idx] = rotate(input=vessel_imgs[idx], angle=angle, axes=(0, 1), reshape=False, order=1)

    # z score with mean, std of each image
    for idx in range(batch_size):
        mean = np.mean(fundus_imgs[idx, ...][fundus_imgs[idx, ..., 0] > 40.0], axis=0)
        std = np.std(fundus_imgs[idx, ...][fundus_imgs[idx, ..., 0] > 40.0], axis=0)

        assert len(mean) == 3 and len(std) == 3
        fundus_imgs[idx, ...] = (fundus_imgs[idx, ...] - mean) / std

    return fundus_imgs, np.round(vessel_imgs)


def get_val_imgs(img_files, vessel_files, mask_files, img_size):
    # load images
    fundus_imgs = imagefiles2arrs(img_files)
    vessel_imgs = imagefiles2arrs(vessel_files) / 255
    mask_imgs = imagefiles2arrs(mask_files) / 255

    # padding
    fundus_imgs = pad_imgs(fundus_imgs, img_size)
    vessel_imgs = pad_imgs(vessel_imgs, img_size)
    mask_imgs = pad_imgs(mask_imgs, img_size)

    assert (np.min(vessel_imgs) == 0 and np.max(vessel_imgs) == 1)
    assert (np.min(mask_imgs) == 0 and np.max(mask_imgs) == 1)

    # augmentation
    # augment the original image (flip, rotate)
    all_fundus_imgs = [fundus_imgs]
    all_vessel_imgs = [vessel_imgs]
    all_mask_imgs = [mask_imgs]

    flipped_imgs = fundus_imgs[:, :, ::-1, :]  # flipped imgs
    flipped_vessels = vessel_imgs[:, :, ::-1]
    flipped_masks = mask_imgs[:, :, ::-1]

    all_fundus_imgs.append(flipped_imgs)
    all_vessel_imgs.append(flipped_vessels)
    all_mask_imgs.append(flipped_masks)

    for angle in range(3, 360, 3):  # rotated imgs (3, 360, 3)
        print("Val data augmentation {} degree...".format(angle))
        all_fundus_imgs.append(random_perturbation(rotate(fundus_imgs, angle, axes=(1, 2), reshape=False,
                                                          order=1)))
        all_fundus_imgs.append(random_perturbation(rotate(flipped_imgs, angle, axes=(1, 2), reshape=False,
                                                          order=1)))

        all_vessel_imgs.append(rotate(vessel_imgs, angle, axes=(1, 2), reshape=False, order=1))
        all_vessel_imgs.append(rotate(flipped_vessels, angle, axes=(1, 2), reshape=False, order=1))

        all_mask_imgs.append(rotate(mask_imgs, angle, axes=(1, 2), reshape=False, order=1))
        all_mask_imgs.append(rotate(flipped_masks, angle, axes=(1, 2), reshape=False, order=1))

    fundus_imgs = np.concatenate(all_fundus_imgs, axis=0)
    vessel_imgs = np.concatenate(all_vessel_imgs, axis=0)
    mask_imgs = np.concatenate(all_mask_imgs, axis=0)

    # z score with mean, std of each image
    mean_std = []
    n_all_imgs = fundus_imgs.shape[0]
    for index in range(n_all_imgs):
        mean = np.mean(fundus_imgs[index, ...][fundus_imgs[index, ..., 0] > 40.0], axis=0)
        std = np.std(fundus_imgs[index, ...][fundus_imgs[index, ..., 0] > 40.0], axis=0)

        assert len(mean) == 3 and len(std) == 3
        fundus_imgs[index, ...] = (fundus_imgs[index, ...] - mean) / std

        mean_std.append({'mean': mean, 'std': std})

    return fundus_imgs, np.round(vessel_imgs), np.round(mask_imgs), mean_std


def get_test_imgs(target_dir, img_size, dataset):
    img_files, vessel_files, mask_files, mask_imgs = None, None, None, None
    if dataset == 'DRIVE':
        img_files, vessel_files, mask_files = DRIVE_files(target_dir)
    elif dataset == 'STARE':
        img_files, vessel_files, mask_files = STARE_files(target_dir)

    # load images
    fundus_imgs = imagefiles2arrs(img_files)
    vessel_imgs = imagefiles2arrs(vessel_files) / 255
    fundus_imgs = pad_imgs(fundus_imgs, img_size)
    vessel_imgs = pad_imgs(vessel_imgs, img_size)
    assert (np.min(vessel_imgs) == 0 and np.max(vessel_imgs) == 1)

    mask_imgs = imagefiles2arrs(mask_files) / 255
    mask_imgs = pad_imgs(mask_imgs, img_size)
    assert (np.min(mask_imgs) == 0 and np.max(mask_imgs) == 1)

    # z score with mean, std of each image
    mean_std = []
    n_all_imgs = fundus_imgs.shape[0]
    for index in range(n_all_imgs):
        mean = np.mean(fundus_imgs[index, ...][fundus_imgs[index, ..., 0] > 40.0], axis=0)
        std = np.std(fundus_imgs[index, ...][fundus_imgs[index, ..., 0] > 40.0], axis=0)

        assert len(mean) == 3 and len(std) == 3
        fundus_imgs[index, ...] = (fundus_imgs[index, ...] - mean) / std

        mean_std.append({'mean': mean, 'std': std})

    return fundus_imgs, np.round(vessel_imgs), mask_imgs, mean_std


def image_shape(filename):
    img = Image.open(filename)
    img_arr = np.asarray(img)
    img_shape = img_arr.shape
    return img_shape


def pad_imgs(imgs, img_size):
    padded = None
    img_h, img_w = imgs.shape[1], imgs.shape[2]
    target_h, target_w = img_size[0], img_size[1]
    if len(imgs.shape) == 4:
        d = imgs.shape[3]
        padded = np.zeros((imgs.shape[0], target_h, target_w, d))
    elif len(imgs.shape) == 3:
        padded = np.zeros((imgs.shape[0], img_size[0], img_size[1]))

    start_h, start_w = (target_h - img_h) // 2, (target_w - img_w) // 2
    end_h, end_w = start_h + img_h, start_w + img_w
    padded[:, start_h:end_h, start_w:end_w, ...] = imgs

    return padded


def crop_to_original(imgs, ori_shape):
    # imgs: (N, 640, 640, 3 or None)
    # ori_shape: (584, 565)
    pred_shape = imgs.shape
    assert len(pred_shape) > 2

    if ori_shape == pred_shape:
        return imgs
    else:
        if len(imgs.shape) > 3:  # images (N, 640, 640, 3)
            ori_h, ori_w = ori_shape[0], ori_shape[1]
            pred_h, pred_w = pred_shape[1], pred_shape[2]

            start_h, start_w = (pred_h - ori_h) // 2, (pred_w - ori_w) // 2
            end_h, end_w = start_h + ori_h, start_w + ori_w

            return imgs[:, start_h:end_h, start_w:end_w, :]
        else:  # vesels
            ori_h, ori_w = ori_shape[0], ori_shape[1]
            pred_h, pred_w = pred_shape[1], pred_shape[2]

            start_h, start_w = (pred_h - ori_h) // 2, (pred_w - ori_w) // 2
            end_h, end_w = start_h + ori_h, start_w + ori_w

            return imgs[:, start_h:end_h, start_w:end_w]


def random_perturbation(imgs):
    for i in range(imgs.shape[0]):
        im = Image.fromarray(imgs[i, ...].astype(np.uint8))
        en = ImageEnhance.Color(im)
        im = en.enhance(np.random.uniform(0.8, 1.2))
        imgs[i, ...] = np.asarray(im).astype(np.float32)

    return imgs


def pixel_values_in_mask(true_vessels, pred_vessels, masks, split_by_img=False):
    assert np.max(pred_vessels) <= 1.0 and np.min(pred_vessels) >= 0.0
    assert np.max(true_vessels) == 1.0 and np.min(true_vessels) == 0.0
    assert np.max(masks) == 1.0 and np.min(masks) == 0.0
    assert pred_vessels.shape[0] == true_vessels.shape[0] and masks.shape[0] == true_vessels.shape[0]
    assert pred_vessels.shape[1] == true_vessels.shape[1] and masks.shape[1] == true_vessels.shape[1]
    assert pred_vessels.shape[2] == true_vessels.shape[2] and masks.shape[2] == true_vessels.shape[2]

    if split_by_img:
        n = pred_vessels.shape[0]
        return (np.array([true_vessels[i, ...][masks[i, ...] == 1].flatten() for i in range(n)]),
                np.array([pred_vessels[i, ...][masks[i, ...] == 1].flatten() for i in range(n)]))
    else:
        return true_vessels[masks == 1].flatten(), pred_vessels[masks == 1].flatten()


def remain_in_mask(imgs, masks):
    imgs[masks == 0] = 0
    return imgs


# noinspection PyPep8Naming
def AUC_ROC(true_vessel_arr, pred_vessel_arr):
    """
    Area under the ROC curve with x axis flipped
    ROC: Receiver operating characteristic
    """
    # roc_auc_score: sklearn function
    AUC_ROC_ = roc_auc_score(true_vessel_arr.flatten(), pred_vessel_arr.flatten())
    return AUC_ROC_


# noinspection PyPep8Naming
def AUC_PR(true_vessel_arr, pred_vessel_arr):
    """
    Precision-recall curve: sklearn function
    auc: Area Under Curve, sklearn function
    """
    precision, recall, _ = precision_recall_curve(true_vessel_arr.flatten(),
                                                  pred_vessel_arr.flatten(), pos_label=1)
    AUC_prec_rec = auc(recall, precision)
    return AUC_prec_rec


def threshold_by_f1(true_vessels, generated, masks, flatten=True, f1_score=False):
    vessels_in_mask, generated_in_mask = pixel_values_in_mask(true_vessels, generated, masks)
    precision, recall, thresholds = precision_recall_curve(
        vessels_in_mask.flatten(), generated_in_mask.flatten(), pos_label=1)
    best_f1, best_threshold = best_f1_threshold(precision, recall, thresholds)

    pred_vessels_bin = np.zeros(generated.shape)
    pred_vessels_bin[generated >= best_threshold] = 1

    if flatten:
        if f1_score:
            return pred_vessels_bin[masks == 1].flatten(), best_f1
        else:
            return pred_vessels_bin[masks == 1].flatten()
    else:
        if f1_score:
            return pred_vessels_bin, best_f1
        else:
            return pred_vessels_bin


def best_f1_threshold(precision, recall, thresholds):
    best_f1, best_threshold = -1., None
    for index in range(len(precision)):
        curr_f1 = 2. * precision[index] * recall[index] / (precision[index] + recall[index])
        if best_f1 < curr_f1:
            best_f1 = curr_f1
            best_threshold = thresholds[index]

    return best_f1, best_threshold


def threshold_by_otsu(pred_vessels, masks, flatten=True):
    # cut by otsu threshold
    threshold = filters.threshold_otsu(pred_vessels[masks == 1])
    pred_vessels_bin = np.zeros(pred_vessels.shape)
    pred_vessels_bin[pred_vessels >= threshold] = 1

    if flatten:
        return pred_vessels_bin[masks == 1].flatten()
    else:
        return pred_vessels_bin


def dice_coefficient_in_train(true_vessel_arr, pred_vessel_arr):
    true_vessel_arr = true_vessel_arr.astype(np.bool)
    pred_vessel_arr = pred_vessel_arr.astype(np.bool)

    intersection = np.count_nonzero(true_vessel_arr & pred_vessel_arr)

    size1 = np.count_nonzero(true_vessel_arr)
    size2 = np.count_nonzero(pred_vessel_arr)

    try:
        dc = 2. * intersection / float(size1 + size2)
    except ZeroDivisionError:
        dc = 0.0

    return dc


def misc_measures(true_vessel_arr, pred_vessel_arr):
    cm = confusion_matrix(true_vessel_arr, pred_vessel_arr)
    acc = 1. * (cm[0, 0] + cm[1, 1]) / np.sum(cm)
    sensitivity = 1. * cm[1, 1] / (cm[1, 0] + cm[1, 1])
    specificity = 1. * cm[0, 0] / (cm[0, 1] + cm[0, 0])
    return acc, sensitivity, specificity


def difference_map(ori_vessel, pred_vessel, mask):
    # ori_vessel : an RGB image
    thresholded_vessel = threshold_by_f1(np.expand_dims(ori_vessel, axis=0),
                                         np.expand_dims(pred_vessel, axis=0),
                                         np.expand_dims(mask, axis=0), flatten=False)

    thresholded_vessel = np.squeeze(thresholded_vessel, axis=0)
    diff_map = np.zeros((ori_vessel.shape[0], ori_vessel.shape[1], 3))

    # Green (overlapping)
    diff_map[(ori_vessel == 1) & (thresholded_vessel == 1)] = (0, 255, 0)
    # Red (false negative, missing in pred)
    diff_map[(ori_vessel == 1) & (thresholded_vessel != 1)] = (255, 0, 0)
    # Blue (false positive)
    diff_map[(ori_vessel != 1) & (thresholded_vessel == 1)] = (0, 0, 255)

    # compute dice coefficient for a given image
    overlap = len(diff_map[(ori_vessel == 1) & (thresholded_vessel == 1)])
    fn = len(diff_map[(ori_vessel == 1) & (thresholded_vessel != 1)])
    fp = len(diff_map[(ori_vessel != 1) & (thresholded_vessel == 1)])

    return diff_map, 2. * overlap / (2 * overlap + fn + fp)


def operating_pts_human_experts(gt_vessels, pred_vessels, masks):
    gt_vessels_in_mask, pred_vessels_in_mask = pixel_values_in_mask(
        gt_vessels, pred_vessels, masks, split_by_img=True)

    n = gt_vessels_in_mask.shape[0]
    op_pts_roc, op_pts_pr = [], []
    for i in range(n):
        cm = confusion_matrix(gt_vessels_in_mask[i], pred_vessels_in_mask[i])
        fpr = 1 - 1. * cm[0, 0] / (cm[0, 1] + cm[0, 0])
        tpr = 1. * cm[1, 1] / (cm[1, 0] + cm[1, 1])
        prec = 1. * cm[1, 1] / (cm[0, 1] + cm[1, 1])
        recall = tpr
        op_pts_roc.append((fpr, tpr))
        op_pts_pr.append((recall, prec))

    return op_pts_roc, op_pts_pr


def misc_measures_evaluation(true_vessels, pred_vessels, masks):
    thresholded_vessel_arr, f1_score = threshold_by_f1(true_vessels, pred_vessels, masks, f1_score=True)
    true_vessel_arr = true_vessels[masks == 1].flatten()

    cm = confusion_matrix(true_vessel_arr, thresholded_vessel_arr)
    acc = 1. * (cm[0, 0] + cm[1, 1]) / np.sum(cm)
    sensitivity = 1. * cm[1, 1] / (cm[1, 0] + cm[1, 1])
    specificity = 1. * cm[0, 0] / (cm[0, 1] + cm[0, 0])
    return f1_score, acc, sensitivity, specificity


def dice_coefficient(true_vessels, pred_vessels, masks):
    thresholded_vessels = threshold_by_f1(true_vessels, pred_vessels, masks, flatten=False)

    true_vessels = true_vessels.astype(np.bool)
    thresholded_vessels = thresholded_vessels.astype(np.bool)

    intersection = np.count_nonzero(true_vessels & thresholded_vessels)

    size1 = np.count_nonzero(true_vessels)
    size2 = np.count_nonzero(thresholded_vessels)

    try:
        dc = 2. * intersection / float(size1 + size2)
    except ZeroDivisionError:
        dc = 0.0

    return dc


def save_obj(true_vessel_arr, pred_vessel_arr, auc_roc_file_name, auc_pr_file_name):
    fpr, tpr, _ = roc_curve(true_vessel_arr, pred_vessel_arr)  # roc_curve: sklearn function

    precision, recall, _ = precision_recall_curve(true_vessel_arr.flatten(),
                                                  pred_vessel_arr.flatten(), pos_label=1)

    with open(auc_roc_file_name, 'wb') as f:
        pickle.dump({"fpr": fpr, "tpr": tpr}, f, pickle.HIGHEST_PROTOCOL)

    with open(auc_pr_file_name, 'wb') as f:
        pickle.dump({"precision": precision, "recall": recall}, f, pickle.HIGHEST_PROTOCOL)


def print_metrics(itr, kargs):
    print("*** Iteration {}  ====> ".format(itr))
    for name, value in kargs.items():
        print("{} : {:.6}, ".format(name, value))
    print("")
    sys.stdout.flush()


# noinspection PyPep8Naming
def plot_AUC_ROC(fprs, tprs, method_names, fig_dir, op_pts):
    # set font style
    font = {'family': 'serif'}
    matplotlib.rc('font', **font)

    # sort the order of plots manually for eye-pleasing plots
    colors = ['r', 'b', 'y', 'g', '#7e7e7e', 'm', 'c', 'k'] if len(fprs) == 8 \
        else ['r', 'y', 'm', 'g', 'k']
    indices = [7, 2, 5, 3, 4, 6, 1, 0] if len(fprs) == 8 else [4, 1, 2, 3, 0]

    # print auc
    print("****** ROC AUC ******")
    print("CAVEAT : AUC of V-GAN with 8bit images might be lower than the floating point array "
          "(check <home>/pretrained/auc_roc*.npy)")

    for index in indices:
        if method_names[index] != 'CRFs' and method_names[index] != '2nd_manual':
            print("{} : {:.4}".format(method_names[index], auc(fprs[index], tprs[index])))

    # plot results
    for index in indices:
        if method_names[index] == 'CRFs':
            plt.plot(fprs[index], tprs[index], colors[index] + '*', label=method_names[index].replace("_", " "))
        elif method_names[index] == '2nd_manual':
            plt.plot(fprs[index], tprs[index], colors[index] + '*', label='Human')
        else:
            plt.step(fprs[index], tprs[index], colors[index], where='post',
                     label=method_names[index].replace("_", " "), linewidth=1.5)

    # plot individual operation points
    for op_pt in op_pts:
        plt.plot(op_pt[0], op_pt[1], 'r.')

    plt.title('ROC Curve')
    plt.xlabel("1-Specificity")
    plt.ylabel("Sensitivity")
    plt.xlim(0, 0.3)
    plt.ylim(0.7, 1.0)
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(fig_dir, "ROC.png"))
    plt.close()


# noinspection PyPep8Naming
def plot_AUC_PR(precisions, recalls, method_names, fig_dir, op_pts):
    # set font style
    font = {'family': 'serif'}
    matplotlib.rc('font', **font)

    # sort the order of plots manually for eye-pleasing plots
    colors = ['r', 'b', 'y', 'g', '#7e7e7e', 'm', 'c', 'k'] if len(precisions) == 8 \
        else ['r', 'y', 'm', 'g', 'k']
    indices = [7, 2, 5, 3, 4, 6, 1, 0] if len(precisions) == 8 else [4, 1, 2, 3, 0]

    # print auc
    print("****** Precision Recall AUC ******")
    print("CAVEAT : AUC of V-GAN with 8bit images might be lower than the floating point array "
          "(check <home>/pretrained/auc_pr*.npy)")

    for index in indices:
        if method_names[index] != 'CRFs' and method_names[index] != '2nd_manual':
            print("{} : {:.4}".format(method_names[index], auc(recalls[index], precisions[index])))

    # plot results
    for index in indices:
        if method_names[index] == 'CRFs':
            plt.plot(recalls[index], precisions[index], colors[index] + '*',
                     label=method_names[index].replace("_", " "))
        elif method_names[index] == '2nd_manual':
            plt.plot(recalls[index], precisions[index], colors[index] + '*', label='Human')
        else:
            plt.step(recalls[index], precisions[index], colors[index], where='post',
                     label=method_names[index].replace("_", " "), linewidth=1.5)

    # plot individual operation points
    for op_pt in op_pts:
        plt.plot(op_pt[0], op_pt[1], 'r.')

    plt.title('Precision Recall Curve')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim(0.5, 1.0)
    plt.ylim(0.5, 1.0)
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(fig_dir, "Precision_recall.png"))
    plt.close()
