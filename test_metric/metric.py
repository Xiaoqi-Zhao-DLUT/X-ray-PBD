import math
import numpy as np


class cal_neg_num_error(object):
    # mean absolute error
    def __init__(self):
        self.prediction = []
    def update(self, pred, gt):
        score = self.cal(pred, gt)
        self.prediction.append(score)
    def cal(self, pred, gt):
        num_pred = len(pred)
        num_gt = len(gt)
        return np.abs(num_pred - num_gt)
    def show(self):
        return np.mean(self.prediction)


class cal_pos_num_error(object):
    # mean absolute error
    def __init__(self):
        self.prediction = []
    def update(self, pred, gt):
        score = self.cal(pred, gt)
        self.prediction.append(score)
    def cal(self, pred, gt):
        num_pred = len(pred)
        num_gt = len(gt)
        return np.abs(num_pred - num_gt)
    def show(self):
        return np.mean(self.prediction)


class cal_neg_num_acc(object):
    # mean absolute error
    def __init__(self):
        self.prediction = []
    def update(self, pred_neg, gt_neg):
        score = self.cal(pred_neg, gt_neg)
        self.prediction.append(score)
    def cal(self, pred_neg, gt_neg):
        num_pred_neg = len(pred_neg)
        num_gt_neg = len(gt_neg)
        if num_pred_neg==num_gt_neg:
            return 1
        else:
            return 0
    def show(self):
        return np.mean(self.prediction)


class cal_pos_num_acc(object):
    # mean absolute error
    def __init__(self):
        self.prediction = []
    def update(self, pred_pos, gt_pos):
        score = self.cal(pred_pos, gt_pos)
        self.prediction.append(score)
    def cal(self, pred_pos, gt_pos):
        num_pred_pos = len(pred_pos)
        num_gt_pos = len(gt_pos)
        if num_pred_pos==num_gt_pos:
            return 1
        else:
            return 0
    def show(self):
        return np.mean(self.prediction)


class cal_neg_pos_num_acc(object):
    # mean absolute error
    def __init__(self):
        self.prediction = []
    def update(self, pred_neg, gt_neg, pred_pos, gt_pos):
        score = self.cal(pred_neg, gt_neg, pred_pos, gt_pos)
        self.prediction.append(score)
    def cal(self, pred_neg, gt_neg, pred_pos, gt_pos):
        num_pred_neg = len(pred_neg)
        num_pred_pos = len(pred_pos)
        num_gt_neg = len(gt_neg)
        num_gt_pos = len(gt_pos)
        if num_pred_neg==num_gt_neg and num_pred_pos == num_gt_pos:
            return 1
        else:
            return 0
    def show(self):
        return np.mean(self.prediction)


###一定是neg和pos数量都准确的情况下，这个指标才有意义。叠片的neg要比pos多一个，一个pos需要与两个neg算两个overhang
class cal_neg_pos_overhang(object):
    # mean absolute error
    def __init__(self):
        self.prediction = []
    def update(self, pred_neg, gt_neg, pred_pos, gt_pos):
        score = self.cal(pred_neg, gt_neg, pred_pos, gt_pos)
        self.prediction.append(score)
    def cal(self, pred_neg, gt_neg, pred_pos, gt_pos):
        overhang = []
        for i in range(len(pred_pos)):
            pred_ovrehang_left = abs(pred_neg[i][0]-pred_pos[i][0])
            pred_ovrehang_right = abs(pred_neg[i+1][0]-pred_pos[i][0])
            gt_ovrehang_left = abs(gt_neg[i][0] - gt_pos[i][0])
            gt_ovrehang_right = abs(gt_neg[i + 1][0] - gt_pos[i][0])
            mae_overhang_left = abs(pred_ovrehang_left - gt_ovrehang_left)
            mae_overhang_right = abs(pred_ovrehang_right-gt_ovrehang_right)
            overhang.append(mae_overhang_left)
            overhang.append(mae_overhang_right)
        return np.mean(overhang)
    def show(self):
        return np.mean(self.prediction)



class cal_neg_location(object):
    # mean absolute error
    def __init__(self):
        self.prediction = []
    def update(self, pred_neg, gt_neg):
        score = self.cal(pred_neg, gt_neg)
        self.prediction.append(score)
    def cal(self, pred_neg, gt_neg):
        location = []
        for i in range(len(pred_neg)):
            distance = math.sqrt(math.pow(pred_neg[i][0]-gt_neg[i][0],2)+math.pow(pred_neg[i][1]-gt_neg[i][1],2))
            location.append(distance)
        return np.mean(location)
    def show(self):
        return np.mean(self.prediction)



class cal_pos_location(object):
    # mean absolute error
    def __init__(self):
        self.prediction = []
    def update(self, pred_pos, gt_pos):
        score = self.cal(pred_pos, gt_pos)
        self.prediction.append(score)
    def cal(self, pred_pos, gt_pos):
        location = []
        for i in range(len(pred_pos)):
            distance = math.sqrt(math.pow(pred_pos[i][0]-gt_pos[i][0],2)+math.pow(pred_pos[i][1]-gt_pos[i][1],2))
            location.append(distance)
        return np.mean(location)
    def show(self):
        return np.mean(self.prediction)


if __name__ == "__main__":
    gt_pos = [[623, 281], [620, 340], [617, 377], [612, 412], [611, 447], [614, 474]]
    gt_neg = [[511, 407], [502, 443], [500, 474], [500, 509], [503, 542], [505, 552], [508, 562]]
    pred_pos = [[624, 281], [621, 340], [618, 377], [613, 412], [612, 447], [615, 474]]
    pred_neg = [[511, 407], [502, 443], [500, 474], [500, 509], [503, 542], [505, 552], [508, 562]]
    neg_num = cal_pos_location()
    neg_num.update(pred_pos, gt_pos)
    print(neg_num.show())