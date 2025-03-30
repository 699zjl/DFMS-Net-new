import os
import pickle
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomVerticalFlip
from utils.helpers import Fix_RandomRotation


class vessel_dataset(Dataset):
    def __init__(self, path, mode, is_val=False, split=None):

        self.mode = mode
        self.is_val = is_val
        self.data_path = os.path.join(path, f"{mode}_pro")
        self.data_file = os.listdir(self.data_path)
        #self.img_file = self._select_img(self.data_file)
        ######增加的
        # 过滤掉标签全为1或全为0的图像
        self.img_file = self._filter_images(self.data_file)
        #####
        if split is not None and mode == "training":
            assert split > 0 and split < 1
            if not is_val:
                self.img_file = self.img_file[:int(split*len(self.img_file))]
            else:
                self.img_file = self.img_file[int(split*len(self.img_file)):]
        self.transforms = Compose([
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            Fix_RandomRotation(),
        ])

    def __getitem__(self, idx):
        img_file = self.img_file[idx]
        with open(file=os.path.join(self.data_path, img_file), mode='rb') as file:
            img = torch.from_numpy(pickle.load(file)).float()
        gt_file = "gt" + img_file[3:]
        with open(file=os.path.join(self.data_path, gt_file), mode='rb') as file:
            gt = torch.from_numpy(pickle.load(file)).float()
        if self.mode == "training" and not self.is_val:
            seed = torch.seed()
            torch.manual_seed(seed)
            img = self.transforms(img)
            torch.manual_seed(seed)
            gt = self.transforms(gt)
        return img, gt

    def _filter_images(self, file_list):
        img_list = []
        for file in file_list:
            if file[:3] == "img":
                gt_file = "gt" + file[3:]
                gt_path = os.path.join(self.data_path, gt_file)
                with open(gt_path, 'rb') as f:
                    gt = pickle.load(f)
                    # 将 NumPy 数组转换为 PyTorch 张量
                    gt_tensor = torch.from_numpy(gt).float()
                    # 检查标签是否全为1或全为0
                    if not (torch.all(gt_tensor == 1) or torch.all(gt_tensor == 0)):
                        img_list.append(file)
        return img_list


    def _select_img(self, file_list):
        img_list = []
        for file in file_list:
            if file[:3] == "img":
                img_list.append(file)

        return img_list

    def __len__(self):
        return len(self.img_file)
