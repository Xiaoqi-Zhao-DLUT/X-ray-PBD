import os
import torch
from torch.autograd import Variable
from torchvision import transforms
from utils.config import dataset_root_test
from utils.misc import check_mkdir
from model.MDCNeXt import Region_seg, MDCNeXt
import ttach as tta
import torch.nn.functional as F
import cv2
import numpy as np
torch.manual_seed(2018)
torch.cuda.set_device(0)

ckpt_path = './' # model_pth
anchor_img_path = './YXTP00574-NG-20210705081014963-1_duofuduo_sangdun-clear.jpg' # input_anchor_path
args = {
    'snapshot1': 'PBD5K_Crop',
    'snapshot2': 'MDCNeXt',
    'crf_refine': False,
    'save_results': True
}

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

])

depth_transform = transforms.ToTensor()
target_transform = transforms.ToTensor()
to_pil = transforms.ToPILImage()

to_test = {'PBD5K':dataset_root_test}
transforms = tta.Compose(
    [
        tta.HorizontalFlip(),
        tta.Scale(scales=[0.75,1,1.25], interpolation='bilinear', align_corners=False),
        tta.Scale(scales=[1], interpolation='bilinear', align_corners=False),
    ]
)

def Resize(image,W, H):
    image = cv2.resize(image, dsize=(W, H), interpolation=cv2.INTER_LINEAR)
    return image


def get_coonected_componet_status(img):
    img = np.array(img, dtype="uint8")
    img = img // 255
    retval, labels_cv, stats, centroids = cv2.connectedComponentsWithStats(
        img, ltype=cv2.CV_32S
    )
    return stats

def get_pts_from_xywh(xywh_list):
    centroids = []
    for item in xywh_list:
        x = int(item[0] + item[2] / 2)
        y = int(item[1] + item[3] / 2)
        centroids.append([x, y])
    centroids = np.asarray(centroids, dtype=np.int32).reshape(-1, 2)
    centroids = centroids[np.argsort(centroids[:, 1])]
    return centroids.tolist()

def cal_maxS_ringS(bubble_img):
    bubble_img_np = np.array(bubble_img,dtype = "uint8")
    bubble_img_np = bubble_img_np // 255
    retval, labels_cv, stats, centroids = cv2.connectedComponentsWithStats(bubble_img_np, ltype=cv2.CV_32S) #计算阈，包括背景
    counts_bubble_pixel_fromconnected_regin = [x[-1] for x in stats]
    max_index = counts_bubble_pixel_fromconnected_regin.index(max(counts_bubble_pixel_fromconnected_regin[1:]))
    x,y, w, h ,s= stats[max_index]
    return x,y, w, h ,s

def Normalize(image, mean, std):
    image = (image - mean) / std
    return image


def main():
    net1 = Region_seg().cuda()
    net2 = MDCNeXt().cuda()
    print ('load snapshot \'%s\' for testing' % args['snapshot2'])
    net1.load_state_dict(torch.load(os.path.join(ckpt_path, args['snapshot1']+'.pth'),map_location={'cuda:1': 'cuda:1'}))
    net2.load_state_dict(torch.load(os.path.join(ckpt_path, args['snapshot2']+'.pth'),map_location={'cuda:1': 'cuda:1'}))
    net1.eval()
    net2.eval()
    with torch.no_grad():
        for name, root in to_test.items():
            check_mkdir(os.path.join(ckpt_path, '%s_%s' % (name, args['snapshot2'])))
            root1 = os.path.join(root, 'img')
            img_list = [f for f in os.listdir(root1)]
            for idx, img_name in enumerate(img_list):
                print ('predicting for %s: %d / %d' % (name, idx + 1, len(img_list)))
                rgb_path = os.path.join(root, 'img', img_name)
                img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)[:, :, ::-1]
                anchor_img = cv2.imread(anchor_img_path, cv2.IMREAD_COLOR)[:, :, ::-1] # anchor

                w_,h_,_ = img.shape
                w_anchor,h_anchor,_ = anchor_img.shape
                img_resize = Resize(img,1024,1024)
                anchor_img_resize = Resize(anchor_img,1024,1024)
                img_var = Variable(img_transform(img_resize).unsqueeze(0), volatile=True).cuda()
                anchor_img_var = Variable(img_transform(anchor_img_resize).unsqueeze(0), volatile=True).cuda()

                n, c, h, w = img_var.size()
                model_output = net1(img_var)
                anchor_output = net1(anchor_img_var)

                prediction = model_output.sigmoid()
                anchor_output = anchor_output.sigmoid()

                res = F.upsample(prediction, size=[w_, h_], mode='bilinear', align_corners=False)
                res = res.data.cpu().numpy().squeeze()
                res = 255 * (res - res.min()) / (res.max() - res.min() + 1e-8)
                res[res >= 128] = 255
                res[res != 255] = 0

                anchor_output = F.upsample(anchor_output, size=[w_anchor, h_anchor], mode='bilinear', align_corners=False)
                anchor_output = anchor_output.data.cpu().numpy().squeeze()
                anchor_output = 255 * (anchor_output - anchor_output.min()) / (anchor_output.max() - anchor_output.min() + 1e-8)
                anchor_output[anchor_output >= 128] = 255
                anchor_output[anchor_output != 255] = 0


                
                crop_x, crop_y, crop_w, crop_h, s = cal_maxS_ringS(res)
                crop_x_anchor, crop_y_anchor, crop_w_anchor, crop_h_anchor, s_anchor = cal_maxS_ringS(anchor_output)
                img_crop = img[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
                anchor_img_crop = anchor_img[crop_y_anchor:crop_y_anchor + crop_h_anchor, crop_x_anchor:crop_x_anchor + crop_w_anchor]


                w_crop, h_crop, _ = img_crop.shape
                img_crop_resize = Resize(img_crop, 512,512 )
                anchor_img_crop_resize = Resize(anchor_img_crop, 512, 512)
                img_crop_var = Variable(img_transform(img_crop_resize).unsqueeze(0), volatile=True).cuda()
                anchor_img_crop_var = Variable(img_transform(anchor_img_crop_resize).unsqueeze(0), volatile=True).cuda()
                n, c, h, w = img_crop_var.size()

                mask_neg = []
                mask_pos = []
                for transformer in transforms:  # custom transforms or e.g. tta.aliases.d4_transform()
                    rgb_trans = transformer.augment_image(img_crop_var)
                    rgb_trans_anchor = transformer.augment_image(anchor_img_crop_var)
                    print(rgb_trans.shape,rgb_trans_anchor.shape)
                    model_output = net2(rgb_trans,rgb_trans_anchor)
                    model_output_neg = model_output[:, 0, :, :].unsqueeze(0)
                    model_output_pos = model_output[:, 1, :, :].unsqueeze(0)
                    deaug_mask_neg = transformer.deaugment_mask(model_output_neg)
                    deaug_mask_pos = transformer.deaugment_mask(model_output_pos)
                    mask_neg.append(deaug_mask_neg)
                    mask_pos.append(deaug_mask_pos)
                output_fpn_neg = torch.mean(torch.stack(mask_neg, dim=0), dim=0)
                output_fpn_pos = torch.mean(torch.stack(mask_pos, dim=0), dim=0)



                prediction_neg_crop = output_fpn_neg.sigmoid()
                prediction_pos_crop = output_fpn_pos.sigmoid()

                prediction_neg_crop = F.upsample(prediction_neg_crop, size=[w_crop, h_crop], mode='bilinear', align_corners=False)
                prediction_pos_crop = F.upsample(prediction_pos_crop, size=[w_crop, h_crop], mode='bilinear', align_corners=False)
                prediction_neg_crop = prediction_neg_crop.data.cpu().numpy().squeeze()
                prediction_pos_crop = prediction_pos_crop.data.cpu().numpy().squeeze()

                prediction_neg_crop = 255 * prediction_neg_crop
                prediction_pos_crop = 255 * prediction_pos_crop

                prediction_neg_crop[prediction_neg_crop > 128] = 255
                prediction_neg_crop[prediction_neg_crop != 255] = 0

                prediction_pos_crop[prediction_pos_crop > 128] = 255
                prediction_pos_crop[prediction_pos_crop != 255] = 0

                # #
                prediction_original_neg = np.zeros((img.shape[0], img.shape[1]), np.uint8)
                prediction_original_pos = np.zeros((img.shape[0], img.shape[1]), np.uint8)
                prediction_original_neg[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w] = prediction_neg_crop
                prediction_original_neg[res != 255] = 0

                prediction_original_pos[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w] = prediction_pos_crop
                prediction_original_pos[res != 255] = 0


                prediction_neg_location = prediction_original_neg.copy()
                prediction_pos_location = prediction_original_pos.copy()
                prediction_neg_location_stats = get_coonected_componet_status(prediction_neg_location)
                prediction_pos_location_stats = get_coonected_componet_status(prediction_pos_location)
                neg_list = get_pts_from_xywh(prediction_neg_location_stats[1:])
                pos_list = get_pts_from_xywh(prediction_pos_location_stats[1:])

                if args['save_results']:
                    check_mkdir(os.path.join(ckpt_path,args['snapshot2']+'epoch',name,'neg_location'))
                    check_mkdir(os.path.join(ckpt_path,args['snapshot2']+'epoch',name,'pos_location'))
                    check_mkdir(os.path.join(ckpt_path,args['snapshot2']+'epoch',name,'neg_point_mask'))
                    check_mkdir(os.path.join(ckpt_path,args['snapshot2']+'epoch',name,'pos_point_mask'))
                    check_mkdir(os.path.join(ckpt_path,args['snapshot2']+'epoch',name,'crop_neg_point_mask'))
                    check_mkdir(os.path.join(ckpt_path,args['snapshot2']+'epoch',name,'crop_pos_point_mask'))
                    cv2.imwrite(os.path.join(ckpt_path ,args['snapshot2']+'epoch',name, 'crop_neg_point_mask',img_name[:-4] + '.png'), prediction_neg_crop)
                    cv2.imwrite(os.path.join(ckpt_path ,args['snapshot2']+'epoch',name, 'crop_pos_point_mask',img_name[:-4] + '.png'), prediction_pos_crop)
                    cv2.imwrite(os.path.join(ckpt_path ,args['snapshot2']+'epoch',name, 'neg_point_mask',img_name[:-4] + '.png'), prediction_original_neg)
                    cv2.imwrite(os.path.join(ckpt_path ,args['snapshot2']+'epoch',name, 'pos_point_mask',img_name[:-4] + '.png'), prediction_original_pos)
                    np.save(os.path.join(ckpt_path ,args['snapshot2']+'epoch',name,'neg_location', img_name[:-4] + '.npy'), neg_list)
                    np.save(os.path.join(ckpt_path ,args['snapshot2']+'epoch',name, 'pos_location',img_name[:-4] + '.npy'), pos_list)


if __name__ == '__main__':
    main()
