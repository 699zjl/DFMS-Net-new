import os
import argparse
import pickle
import cv2
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from PIL import Image
from ruamel.yaml import safe_load
from torchvision.transforms import Grayscale, Normalize, ToTensor
from utils.helpers import dir_exists, remove_files


def data_process(data_path, name, patch_size, stride, mode):
    save_path = os.path.join(data_path, f"{mode}_pro")   #'datasets/DRIVE/training_pro'
    dir_exists(save_path)
    remove_files(save_path)
    if name == "DRIVE":
        img_path = os.path.join(data_path, mode, "images")  #'datasets/DRIVE/training/images'
        gt_path = os.path.join(data_path, mode, "1st_manual")   #'datasets/DRIVE/training/1st_manual'
        file_list = list(sorted(os.listdir(img_path)))  #['21_training.tif', '22_training.tif', '23_training.tif', '24_training.tif', '25_training.tif', '26_training.tif', '27_training.tif', '28_training.tif', '29_training.tif', '30_training.tif', '31_training.tif', '32_training.tif', '33_training.tif', '34_training.tif', '35_training.tif', '36_training.tif', '37_training.tif', '38_training.tif', '39_training.tif', '40_training.tif']
    elif name == "CHASEDB1":
        file_list = list(sorted(os.listdir(data_path)))
    elif name == "STARE":
        img_path = os.path.join(data_path, "stare-images")
        gt_path = os.path.join(data_path, "labels-ah")
        file_list = list(sorted(os.listdir(img_path)))
    elif name == "DCA1":
        data_path = os.path.join(data_path, "Database_134_Angiograms")
        file_list = list(sorted(os.listdir(data_path)))
    elif name == "CHUAC":
        img_path = os.path.join(data_path, "Original")
        gt_path = os.path.join(data_path, "Photoshop")
        file_list = list(sorted(os.listdir(img_path)))
    img_list = []
    gt_list = []
    for i, file in enumerate(file_list):   # i=0， file='21_training.tif'
        if name == "DRIVE":
            img = Image.open(os.path.join(img_path, file))
            gt = Image.open(os.path.join(gt_path, file[0:2] + "_manual1.gif"))
            img = Grayscale(1)(img)   #RGB图像转换为灰度图像
            img_list.append(ToTensor()(img))
            gt_list.append(ToTensor()(gt))

        elif name == "CHASEDB1":
            if len(file) == 13:
                if mode == "training" and int(file[6:8]) <= 10:
                    img = Image.open(os.path.join(data_path, file))
                    gt = Image.open(os.path.join(
                        data_path, file[0:9] + '_1stHO.png'))
                    img = Grayscale(1)(img)
                    img_list.append(ToTensor()(img))
                    gt_list.append(ToTensor()(gt))
                elif mode == "test" and int(file[6:8]) > 10:
                    img = Image.open(os.path.join(data_path, file))
                    gt = Image.open(os.path.join(
                        data_path, file[0:9] + '_1stHO.png'))
                    img = Grayscale(1)(img)
                    img_list.append(ToTensor()(img))
                    gt_list.append(ToTensor()(gt))
        elif name == "DCA1":
            if len(file) <= 7:
                if mode == "training" and int(file[:-4]) <= 100:
                    img = cv2.imread(os.path.join(data_path, file), 0)
                    gt = cv2.imread(os.path.join(
                        data_path, file[:-4] + '_gt.pgm'), 0)
                    gt = np.where(gt >= 100, 255, 0).astype(np.uint8)
                    img_list.append(ToTensor()(img))
                    gt_list.append(ToTensor()(gt))
                elif mode == "test" and int(file[:-4]) > 100:
                    img = cv2.imread(os.path.join(data_path, file), 0)
                    gt = cv2.imread(os.path.join(
                        data_path, file[:-4] + '_gt.pgm'), 0)
                    gt = np.where(gt >= 100, 255, 0).astype(np.uint8)
                    img_list.append(ToTensor()(img))
                    gt_list.append(ToTensor()(gt))
        elif name == "CHUAC":
            if mode == "training" and int(file[:-4]) <= 20:
                img = cv2.imread(os.path.join(img_path, file), 0)
                if int(file[:-4]) <= 17 and int(file[:-4]) >= 11:
                    tail = "PNG"
                else:
                    tail = "png"
                gt = cv2.imread(os.path.join(
                    gt_path, "angio"+file[:-4] + "ok."+tail), 0)
                gt = np.where(gt >= 100, 255, 0).astype(np.uint8)
                img = cv2.resize(
                    img, (512, 512), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(f"save_picture/{i}img.png", img)
                cv2.imwrite(f"save_picture/{i}gt.png", gt)
                img_list.append(ToTensor()(img))
                gt_list.append(ToTensor()(gt))
            elif mode == "test" and int(file[:-4]) > 20:
                img = cv2.imread(os.path.join(img_path, file), 0)
                gt = cv2.imread(os.path.join(
                    gt_path, "angio"+file[:-4] + "ok.png"), 0)
                gt = np.where(gt >= 100, 255, 0).astype(np.uint8)
                img = cv2.resize(
                    img, (512, 512), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(f"save_picture/{i}img.png", img)
                cv2.imwrite(f"save_picture/{i}gt.png", gt)
                img_list.append(ToTensor()(img))
                gt_list.append(ToTensor()(gt))
        elif name == "STARE":
            if not file.endswith("gz"):
                img = Image.open(os.path.join(img_path, file))
                gt = Image.open(os.path.join(gt_path, file[0:6] + '.ah.ppm'))
                cv2.imwrite(f"save_picture/{i}img.png", np.array(img))
                cv2.imwrite(f"save_picture/{i}gt.png", np.array(gt))
                img = Grayscale(1)(img)
                img_list.append(ToTensor()(img))
                gt_list.append(ToTensor()(gt))
    img_list = normalization(img_list)
    if mode == "training":
        img_patch = get_patch(img_list, patch_size, stride)   #160160个图像块
        gt_patch = get_patch(gt_list, patch_size, stride)   #同上
        save_patch(img_patch, save_path, "img_patch", name)
        save_patch(gt_patch, save_path, "gt_patch", name)
    elif mode == "test":
        if name != "CHUAC":
            img_list = get_square(img_list, name)   #torch.Size([1, 592, 592])  ，list:20
            gt_list = get_square(gt_list, name)
        save_each_image(img_list, save_path, "img", name)
        save_each_image(gt_list, save_path, "gt", name)


def get_square(img_list, name):
    img_s = []
    if name == "DRIVE":
        shape = 592
    elif name == "CHASEDB1":
        shape = 1008
    elif name == "DCA1":
        shape = 320
    _, h, w = img_list[0].shape
    pad = nn.ConstantPad2d((0, shape-w, 0, shape-h), 0)
    for i in range(len(img_list)):
        img = pad(img_list[i])     #torch.Size([1, 592, 592])
        img_s.append(img)

    return img_s


def get_patch(imgs_list, patch_size, stride):
    image_list = []
    _, h, w = imgs_list[0].shape     #—_:1   h:584   w:565
    pad_h = stride - (h - patch_size) % stride   # pad_w: 5, pad_h: 4
    pad_w = stride - (w - patch_size) % stride
    for sub1 in imgs_list:
        image = F.pad(sub1, (0, pad_w, 0, pad_h), "constant", 0)   #(0, pad_w, 0, pad_h) 是一个四元组，指定了填充的尺寸。这个四元组的顺序是 (left, right, top, bottom)，分别对应图像的左侧、右侧、顶部和底部需要填充的像素数
        image = image.unfold(1, patch_size, stride).unfold(
            2, patch_size, stride).permute(1, 2, 0, 3, 4)     #将图像分割成小块（patches）  torch.Size([91, 88, 1, 48, 48])
        image = image.contiguous().view(               #重新排列和重塑 torch.Size([8008, 1, 48, 48])
            image.shape[0] * image.shape[1], image.shape[2], patch_size, patch_size)
        for sub2 in image:
            image_list.append(sub2)    #存储图像块
    return image_list


def save_patch(imgs_list, path, type, name):
    for i, sub in enumerate(imgs_list):
        with open(file=os.path.join(path, f'{type}_{i}.pkl'), mode='wb') as file:
            pickle.dump(np.array(sub), file)
            print(f'save {name} {type} : {type}_{i}.pkl')


def save_each_image(imgs_list, path, type, name):
    for i, sub in enumerate(imgs_list):
        with open(file=os.path.join(path, f'{type}_{i}.pkl'), mode='wb') as file:
            pickle.dump(np.array(sub), file)
            print(f'save {name} {type} : {type}_{i}.pkl')


def normalization(imgs_list):
    imgs = torch.cat(imgs_list, dim=0)
    mean = torch.mean(imgs)
    std = torch.std(imgs)
    normal_list = []
    for i in imgs_list:
        n = Normalize([mean], [std])(i)
        n = (n - torch.min(n)) / (torch.max(n) - torch.min(n))
        normal_list.append(n)
    return normal_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--dataset_path', default="datasets/DRIVE", type=str,
                        help='the path of dataset',required=True)
    parser.add_argument('-dn', '--dataset_name', default="DRIVE", type=str,
                        help='the name of dataset',choices=['DRIVE','CHASEDB1','STARE','CHUAC','DCA1'],required=True)
    parser.add_argument('-ps', '--patch_size', default=48,
                        help='the size of patch for image partition')
    parser.add_argument('-s', '--stride', default=6,
                        help='the stride of image partition')
    args = parser.parse_args()
    with open('config.yaml', encoding='utf-8') as file:
        CFG = safe_load(file)  # 为字典类型

    #data_process(args.dataset_path, args.dataset_name,
    #             args.patch_size, args.stride, "training")
    data_process(args.dataset_path, args.dataset_name,
                 args.patch_size, args.stride, "test")
