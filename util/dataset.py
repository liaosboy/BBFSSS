from util import config

import torch.utils.data as data
import torch
import os
import cv2
import random
import numpy as np

def bboxlabel2xywh(label, img_shape):
    x, y, w, h = label
    imgh, imgw = img_shape
    x = int(float(x) * imgw)
    y = int(float(y) * imgh)
    w = int(float(w) * imgw)
    h = int(float(h) * imgh)
    x1 = int(x - (w / 2))
    y1 = int(y - (h / 2))

    return x1, y1, w, h

def generate_support_mask(shape, bboxes):
    support_mask= torch.zeros(shape[0], shape[1]).cuda()
    for bbox in bboxes:
        x, y, w, h = bboxlabel2xywh(bbox, shape)
        if x == y == w == h:
            break
        support_mask[y:y+h, x:x+w] = 1
    return support_mask


class BBPFEDataset(data.Dataset):
    def __init__(self, data_root: str, data_classes:list, split:int, shot:int, mode:str, fold_num = 4, img_size = 481, que_transform=None, sup_transform = None) -> None:
        assert mode in ['train', 'val']
        assert split < fold_num and len(data_classes) % fold_num == 0

        classes_num = len(data_classes)
        fold_fize = int(classes_num / fold_num)
        val_start_idx = fold_fize * split

        self.data_root = data_root
        self.data_classes = data_classes
        self.shot = shot
        self.img_size = img_size
        self.max_num_labels = 25
        self.que_transform = que_transform
        self.sup_transform = sup_transform

        val_classes = range(val_start_idx, val_start_idx + fold_fize)
        train_classes = list(set(range(classes_num)) - set(val_classes))

        if mode == "val":
            print(f"Validation classes : {list(val_classes)}")
        elif mode == "train":
            print(f"Training classes : {train_classes}")

        if mode == "train":
            self.data_list, self.sup_list = self.load_data(train_classes)
        else:
            self.data_list, self.sup_list = self.load_data(val_classes)

    def load_data(self, sub_classes:list):
        data_list = []
        sup_list = {}
        for cls in sub_classes:
            cls_name = self.data_classes[cls]
            img_dir_path = self.data_root + "images/"
            gt_dir_path = self.data_root + "groundtruth/" + cls_name + "/"
            bbox_dir_path = self.data_root + "bbox/" + cls_name + "/"

            for root, _, files in os.walk(gt_dir_path):
                for file in files:
                    basename_no_ext = os.path.splitext(file)[0]
                    img_path = img_dir_path + basename_no_ext + ".jpg"
                    gt_path = gt_dir_path + file
                    bbox_path = bbox_dir_path + basename_no_ext + ".txt"
                    if os.path.exists(img_path) and os.path.exists(gt_path):
                        if  os.path.exists(bbox_path):
                            if self.filter_sup_file(bbox_path):
                                item = (img_path, gt_path, bbox_path)
                                if cls in sup_list.keys():
                                    sup_list[cls].append(item)
                                else:
                                    sup_list[cls] = [item]
                            item = (cls, img_path, bbox_path)
                            data_list.append(item)

        return data_list, sup_list

    def filter_sup_file(self, bbox_path):
        f = open(bbox_path, 'r')
        one_acreage = 0
        for label in f.readlines():
            data = label .split(' ')
            w = float(data[-2]) 
            h = float(data[-1]) 
            one_acreage += (w * h) * 100
        
        if one_acreage > config.sup_bbox_threshold:
            return True

        return False
        

    def load_bbox_label(self, bbox_file_path):
        bbox = []
        f = open(bbox_file_path, 'r')
        for label in f.readlines():
            _, x, y, w, h = label.split()
            item = (float(x), float(y), float(w), float(h))
            bbox.append(item)

        return bbox

    def flip_bbox(self, bboxlabel):
        new_bbox_list = []
        for bbox in bboxlabel:
            x, y, w, h = bbox           
            x = 1 - x
            new_bbox_list.append([x, y, w, h])
        return new_bbox_list

    def random_horizontal_flip(self, image, semlabel, bblabel):
        if random.random() < 0.5:
            image = cv2.flip(image, 1)
            semlabel = cv2.flip(semlabel, 1)
            bblabel = self.flip_bbox(bblabel)
        return image, semlabel, bblabel

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        cls, img_path, bbox_path = self.data_list[idx]

        bgrimg = cv2.imread(img_path, cv2.IMREAD_COLOR)
        rgbimg = cv2.cvtColor(bgrimg, cv2.COLOR_BGR2RGB)
        

        rgbimg = cv2.resize(rgbimg, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        label = self.load_bbox_label(bbox_path)
        label = generate_support_mask((self.img_size, self.img_size), label)

        if self.que_transform is not None:
            rgbimg, _label = self.que_transform(rgbimg, label.cpu().numpy())


        sup_file_list =  random.sample(self.sup_list[cls], self.shot)

        sup_img_path_list = []
        sup_semlabel_path_list = []
        sup_bblabel_path_list = []

        for img_path, gt_path, bbox_path in sup_file_list:
            sup_img_path_list.append(img_path)
            sup_semlabel_path_list.append(gt_path)
            sup_bblabel_path_list.append(bbox_path)

        sup_img_list = torch.zeros(self.shot, 3, self.img_size, self.img_size)
        sup_bblabel_list = torch.zeros(self.shot, self.img_size, self.img_size)
        sup_semlabel_list = torch.zeros(self.shot, self.img_size, self.img_size)
        subcls_list = torch.zeros(self.shot)
        
        for k in range(self.shot):
            sup_img_path = sup_img_path_list[k]
            sup_img = cv2.imread(sup_img_path, cv2.IMREAD_COLOR)
            sup_img = cv2.cvtColor(sup_img, cv2.COLOR_BGR2RGB)
            sup_img = cv2.resize(sup_img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)

            sup_semlabel_path = sup_semlabel_path_list[k]
            sup_semlabel = cv2.imread(sup_semlabel_path, cv2.IMREAD_GRAYSCALE)
            sup_semlabel = cv2.resize(sup_semlabel, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

            sup_bblabel_path = sup_bblabel_path_list[k]
            bboxes = self.load_bbox_label(sup_bblabel_path)

            sup_img, sup_semlabel, bboxes = self.random_horizontal_flip(sup_img, sup_semlabel, bboxes)
            if self.sup_transform is not None:
                sup_img, sup_semlabel = self.sup_transform(sup_img, sup_semlabel)

            sup_img_list[k] = sup_img
            sup_semlabel_list[k] = sup_semlabel
            subcls_list[k] = int(cls)

            sup_bblabel_list[k] = generate_support_mask((self.img_size, self.img_size), bboxes)

        return rgbimg, label, sup_img_list, sup_bblabel_list, sup_semlabel_list, subcls_list






