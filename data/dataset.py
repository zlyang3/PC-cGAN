import random
import torch
from torch.utils.data import Dataset
import os
import numpy as np


class ShapeNetDataset(Dataset):
    def __init__(self, root='./shapenetcore_partanno_segmentation_benchmark_v0', num_points=2048, class_choice=None):
        super().__init__()
        self.root = root
        self.num_points = num_points
        self.catfile = os.path.join(root, "synsetoffset2category.txt")
        self.cat = {}
        self.class_choice = class_choice

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]

        if class_choice is not None:
            self.cat = {class_choice: self.cat[class_choice]}

        self.meta = {}
        for item in self.cat:
            self.meta[item] = []
            dir_point = os.path.join(root, self.cat[item], "points")
            dir_seg = os.path.join(root, self.cat[item], "points_label")

            fns = sorted(os.listdir(dir_point))
            for fn in fns:
                token = os.path.splitext(os.path.basename(fn))[0]
                self.meta[item].append((os.path.join(dir_point, token+".pts"), os.path.join(dir_seg, token+".seg")))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn[0], fn[1]))

        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))

    def __getitem__(self, index):
        fn = self.datapath[index]
        cls = self.classes[self.datapath[index][0]]
        point = np.loadtxt(fn[1]).astype(np.float)
        seg = np.loadtxt(fn[2]).astype(np.int)

        # assert len(seg) >= self.num_points
        choice = random.choices(range(len(seg)), k=self.num_points)
        point = torch.from_numpy(point[choice, :])
        seg = torch.from_numpy(seg[choice])
        cls = torch.from_numpy(np.array([cls]).astype(np.int))

        return point, seg, cls

    def __len__(self):
        return len(self.datapath)

class Generated_ShapeNetDataset(Dataset):
    def __init__(self, root='/home/dell/Data/yzl/shapenetcore_partanno_segmentation_benchmark_v0', num_points=2048, class_choice=None):
        super().__init__()
        self.root = root
        self.num_points = num_points
        self.catfile = os.path.join(root, "synsetoffset2category.txt")
        self.cat = {}
        self.class_choice = class_choice

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]

        if class_choice is not None:
            self.cat = {class_choice: self.cat[class_choice]}

        self.meta = {}
        for item in self.cat:
            self.meta[item] = []
            dir_point = os.path.join(root, self.cat[item], "points")

            fns = sorted(os.listdir(dir_point))
            for fn in fns:
                token = os.path.splitext(os.path.basename(fn))[0]
                self.meta[item].append((os.path.join(dir_point, token+".pts")))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))

    def __getitem__(self, index):
        fn = self.datapath[index]
        cls = self.classes[self.datapath[index][0]]
        point = np.loadtxt(fn[1]).astype(np.float)

        # assert len(seg) >= self.num_points
        choice = random.choices(range(len(point)), k=self.num_points)
        point = torch.from_numpy(point[choice, :])
        cls = torch.from_numpy(np.array([cls]).astype(np.int))

        return point, cls

    def __len__(self):
        return len(self.datapath)

if __name__ == "__main__":
    datasets = ShapeNetDataset()
    point, seg, cls = datasets[-1]
    print(point.shape)  # num_point, 3