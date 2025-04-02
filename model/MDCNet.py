#!/usr/bin/python3
#coding=utf-8
import torch.nn as nn
import torch.nn.functional as F
from model.MS_module import ASPP
from model.dynamic_conv import Dynamic_conv2d
import torch
import timm


####Region seg
class Region_seg(nn.Module):
    def __init__(self):
        super(Region_seg, self).__init__()
        self.bkbone = timm.create_model('resnet50d', features_only=True, pretrained=True)
        ###############################Transition Layer########################################
        self.dem1 = ASPP(2048,64)
        # self.dem1 = nn.Sequential(nn.Conv2d(2048, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.dem2 = nn.Sequential(nn.Conv2d(1024, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.dem3 = nn.Sequential(nn.Conv2d(512, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.dem4 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.dem5 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        ################################FPN branch#######################################
        self.output1_1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output2_1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output3_1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output4_1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output5_1 = nn.Sequential(nn.Conv2d(64, 1, kernel_size=3, padding=1))

    def forward(self, x):
        input = x
        B, _, _, _ = input.size()
        E1, E2, E3, E4, E5 = self.bkbone(x)
        ################################Transition Layer#######################################
        T5 = self.dem1(E5)
        T4 = self.dem2(E4)
        T3 = self.dem3(E3)
        T2 = self.dem4(E2)
        T1 = self.dem5(E1)
        ################################Decoder Layer#######################################
        D4_1 = self.output2_1(F.upsample(T5, size=E4.size()[2:], mode='bilinear')+T4)
        D3_1 = self.output3_1(F.upsample(D4_1, size=E3.size()[2:], mode='bilinear')+T3)
        D2_1 = self.output4_1(F.upsample(D3_1, size=E2.size()[2:], mode='bilinear')+T2)
        D1_1 = self.output5_1(F.upsample(D2_1, size=E1.size()[2:], mode='bilinear')+T1)
        ################################Gated Parallel&Dual branch residual fuse#######################################
        output_fpn_p = F.upsample(D1_1, size=input.size()[2:], mode='bilinear')
        #######################################################################
        if self.training:
            return output_fpn_p
        # return F.sigmoid(output_fpn_p)
        return output_fpn_p




#####MDCNet_point_line_counting
class MDCNet_fcn(nn.Module):
    def __init__(self):
        super(MDCNet_fcn, self).__init__()
        self.bkbone_rgb = timm.create_model('resnet50d', features_only=True, pretrained=True)
        self.bkbone_anchor = timm.create_model('resnet50d', features_only=True, pretrained=True)
        ###############################Transition Layer########################################
        self.dem1 = ASPP(2048,64)
        # self.dem1 = nn.Sequential(nn.Conv2d(2048, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.dem2 = nn.Sequential(nn.Conv2d(1024, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.dem3 = nn.Sequential(nn.Conv2d(512, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.dem4 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.dem5 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.dem1_anchor = nn.Sequential(nn.Conv2d(2048, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.dem2_anchor = nn.Sequential(nn.Conv2d(1024, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.dem3_anchor = nn.Sequential(nn.Conv2d(512, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))


        ################################FPN branch#######################################
        self.output1_1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output2_1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output3_1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output4_1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output5_1 = nn.Sequential(nn.Conv2d(64, 2, kernel_size=3, padding=1))

        self.dynamic_conv3 = Dynamic_conv2d(in_planes=64, out_planes=64, kernel_size=3, ratio=0.25, padding=1,)
        self.dynamic_conv4 = Dynamic_conv2d(in_planes=64, out_planes=64, kernel_size=3, ratio=0.25, padding=1,)
        self.dynamic_conv5 = Dynamic_conv2d(in_planes=64, out_planes=64, kernel_size=3, ratio=0.25, padding=1,)

        self.regressor_fcn_neg = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, padding=0), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.regressor_fcn_pos = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, padding=0), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.regressor_neg = nn.Sequential(nn.Conv2d(64, 1, kernel_size=1, padding=0), nn.ReLU(inplace=True))
        self.regressor_pos = nn.Sequential(nn.Conv2d(64, 1, kernel_size=1, padding=0), nn.ReLU(inplace=True))

        self.T2_T1_fusion = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                  nn.ReLU(inplace=True))

        self.neg_line_pre = nn.Sequential(nn.Conv2d(64, 1, kernel_size=3, padding=1))
        self.pos_line_pre = nn.Sequential(nn.Conv2d(64, 1, kernel_size=3, padding=1))

    def forward(self, x, anchor_input):
        input = x
        B, _, _, _ = input.size()
        _, _, E3_anchor, E4_anchor, E5_anchor = self.bkbone_anchor(anchor_input)
        E1, E2, E3, E4, E5 = self.bkbone_rgb(x)
        ################################Transition Layer#######################################
        T5_rgb = self.dem1(E5)
        T4_rgb = self.dem2(E4)
        T3_rgb = self.dem3(E3)
        T2 = self.dem4(E2)
        T1 = self.dem5(E1)

        T5_anchor = self.dem1(E5_anchor)
        T4_anchor = self.dem2(E4_anchor)
        T3_anchor = self.dem3(E3_anchor)

        T5 = self.dynamic_conv5(T5_rgb,T5_anchor)
        T4 = self.dynamic_conv4(T4_rgb,T4_anchor)
        T3 = self.dynamic_conv3(T3_rgb,T3_anchor)
        ################################Decoder Layer#######################################
        D4_1 = self.output2_1(F.upsample(T5, size=E4.size()[2:], mode='bilinear')+T4)
        D3_1 = self.output3_1(F.upsample(D4_1, size=E3.size()[2:], mode='bilinear')+T3)
        D2_1 = self.output4_1(F.upsample(D3_1, size=E2.size()[2:], mode='bilinear')+T2)
        D1_1 = self.output5_1(F.upsample(D2_1, size=E1.size()[2:], mode='bilinear')+T1)

        ################################Gated Parallel&Dual branch residual fuse#######################################
        output_fpn_p = F.upsample(D1_1, size=input.size()[2:], mode='bilinear')
        neg_map = F.sigmoid(output_fpn_p[:, 0, :, :].unsqueeze(1))
        pos_map = F.sigmoid(output_fpn_p[:, 1, :, :].unsqueeze(1))
        neg_input = neg_map
        pos_input = pos_map

        regression_features_neg = self.regressor_fcn_neg(F.adaptive_avg_pool2d(T5 * F.upsample(neg_input,size=T5.size()[2:], mode='bilinear'),1))
        regression_features_pos = self.regressor_fcn_pos(F.adaptive_avg_pool2d(T5 * F.upsample(pos_input,size=T5.size()[2:], mode='bilinear'),1))


        regression_pos = self.regressor_pos(regression_features_pos)
        regression_neg = self.regressor_neg(regression_features_neg)

        T2_1 = self.T2_T1_fusion(F.upsample(T2,size=T1.size()[2:], mode='bilinear')+T1)
        T2_1 = F.upsample(T2_1, size=input.size()[2:], mode='bilinear')
        line_neg = self.neg_line_pre(T2_1*neg_map + T2_1)
        line_pos = self.pos_line_pre(T2_1*pos_map + T2_1)
        #######################################################################
        if self.training:

            return output_fpn_p,regression_neg,regression_pos,line_neg,line_pos
        # return F.sigmoid(pre_sal)
        return output_fpn_p



if __name__ == "__main__":
    model = MDCNet_fcn().eval()
    input = torch.randn(1, 3, 352, 352)
    anchor_input = torch.randn(1, 3, 352, 352)
    output = model(input,anchor_input)
    print(output.shape)
