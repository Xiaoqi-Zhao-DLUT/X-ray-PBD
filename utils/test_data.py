import os
import numpy as np

class test_dataset:
    def __init__(self, neg_pre_root, pos_pre_root, neg_gt_root, pos_gt_root):
        self.neg_pre_list = [os.path.splitext(f)[0] for f in os.listdir(neg_pre_root) if f.endswith('.npy')]
        self.pos_pre_list = [os.path.splitext(f)[0] for f in os.listdir(pos_pre_root) if f.endswith('.npy')]
        self.neg_gt_list = [os.path.splitext(f)[0] for f in os.listdir(neg_gt_root) if f.endswith('.npy')]
        self.pos_gt_list = [os.path.splitext(f)[0] for f in os.listdir(pos_gt_root) if f.endswith('.npy')]
        self.img_list = list(set(self.neg_pre_list).intersection(set(self.neg_gt_list)))

        self.neg_pre_root = neg_pre_root
        self.pos_pre_root = pos_pre_root
        self.neg_gt_root = neg_gt_root
        self.pos_gt_root = pos_gt_root
        self.size = len(self.img_list)
        self.index = 0

    def load_data(self):
        #image = self.rgb_loader(self.images[self.index])
        data = np.load(os.path.join(self.neg_pre_root,self.img_list[self.index]+ '.npy'))
        neg_pre = []
        for i in data:
            neg_pre.append(list(i))
        data =  np.load(os.path.join(self.pos_pre_root, self.img_list[self.index] + '.npy'))
        pos_pre = []  # 直接将文件中按行读到list里，效果与方法2一样
        for i in data:
            pos_pre.append(list(i))

        data = np.load(os.path.join(self.neg_gt_root, self.img_list[self.index] + '.npy'))
        neg_gt =  []
        for i in data:
            neg_gt.append(list(i))

        data = np.load(os.path.join(self.pos_gt_root, self.img_list[self.index] + '.npy'))
        pos_gt =  []
        for i in data:
            pos_gt.append(list(i))
        name = self.img_list[self.index]
        self.index += 1

        return neg_pre, pos_pre, neg_gt, pos_gt,name

