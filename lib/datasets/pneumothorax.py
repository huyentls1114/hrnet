# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import cv2
import torch
from tqdm import tqdm

import os

import cv2
import numpy as np
from PIL import Image

import torch
from torch.nn import functional as F

from .base_dataset import BaseDataset
import random

class Pneumothorax(BaseDataset):
    def __init__(self, root, 
                       list_path,
                       weight_positive = 1,
                       base_size = (256),
                       crop_size = (256, 256),
                       multi_scale = True,
                       flip=True, 
                       mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225],
                       downsample_rate = 1,
                       scale_factor = 16,
                       ignore_label = -1,
                       num_classes = 1,
                       num_samples = None):
        import pdb; pdb.set_trace()
        super(Pneumothorax, self).__init__(ignore_label, base_size,
                crop_size, downsample_rate, scale_factor, mean, std)
        self.input_folder = root
        self.weight_positive = weight_positive
        self.multi_scale = multi_scale
        self.flip = flip
        self.num_classes = num_classes

        self.image_folder = os.path.join(self.input_folder, "images")
        self.mask_folder = os.path.join(self.input_folder, "masks")
        
        #data process
        self.df_img_all = self.read_txt(list_path)
        if "train" in list_path:
            mode = "train"
        else:
            mode = "test"
        self.mode = mode
        if mode == "train":
            self.df_img = self.downsample_data(self.df_img_all, weight_positive)
            self.df_img = self.df_img.sample(frac=1)
        else:
            self.df_img = self.df_img_all
        self.list_img_name =self.df_img["img_name"].values

        self.class_weights = None
    
    def update_train_ds(self, weight_positive = 0.8):
        if self.mode == "train":
            self.df_img = self.downsample_data(self.df_img_all, weight_positive)
            self.df_img = self.df_img.sample(frac=1)
            self.list_img_name =self.df_img["img_name"].values

    def downsample_data(self, df, weight_positive):
        df_label0 = df[df["label"] == 0]
        df_label1 = df[df["label"] == 1]

        positive_length = len(df_label1)
        negative_length = int(positive_length*(1-weight_positive)/weight_positive)
        df_label0 = df_label0.sample(negative_length)
        return pd.concat([df_label0, df_label1])

    def __len__(self):
        return len(self.list_img_name)
    
    def __getitem__(self, idx):
        img_name = self.list_img_name[idx]
        img_path = os.path.join(self.image_folder, img_name)
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        size = image.shape

        mask_path = os.path.join(self.mask_folder, img_name)
        label = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if self.flip :
            flip = random.randint(0, 1)
        else:
            flip = 0

        image, label = self.gen_sample(image, label, 
                                self.multi_scale, flip)
                                
        return image.copy(), label.copy(), np.array(size), img_name
        
    def read_txt(self, txt_file):        
        file_ = open(txt_file, "r")
        list_ = file_.readlines()
        list_img_name = []
        list_label = []
        for line in list_:
            img_name, label = line.replace("\n","").split(",")
            list_img_name.append(img_name)
            list_label.append(int(label))
        file_.close()
        return pd.DataFrame({"img_name": list_img_name, "label":list_label})

