import os
from utils.test_data import test_dataset
from PBD_metrics.metric import cal_neg_num_error,cal_pos_num_error,cal_neg_pos_num_acc,cal_neg_location,cal_pos_location,cal_neg_pos_overhang,cal_neg_num_acc,cal_pos_num_acc
from utils.config import test_datasets
from utils.config import Models
from tqdm import tqdm

def main(log_path):
    for method_name,method_prediction_root in Models.items():
        for name, root in test_datasets.items():
            neg_pre_root = method_prediction_root + '/neg_location'
            pos_pre_root = method_prediction_root + '/pos_location'

            neg_gt_root = root +'/neg_location'+'/all'

            pos_gt_root = root +'/pos_location'+'/all'
            img_root = root + '/img'
            if os.path.exists(neg_pre_root):
                test_loader = test_dataset(neg_pre_root, pos_pre_root, neg_gt_root, pos_gt_root,img_root)
                neg_num_error,pos_num_error,neg_num_acc,pos_num_acc,neg_pos_num_acc,neg_location_error,pos_location_error,neg_pos_overhang_error= cal_neg_num_error(),cal_pos_num_error(),cal_neg_num_acc(),cal_pos_num_acc(),cal_neg_pos_num_acc(),cal_neg_location(),cal_pos_location(),cal_neg_pos_overhang()
                for i in tqdm(range(test_loader.size)):
                    neg_pre, pos_pre, neg_gt, pos_gt,_name,h,w = test_loader.load_data()
                    neg_pre = sorted(neg_pre, key=lambda x: x[1])
                    pos_pre = sorted(pos_pre, key=lambda x: x[1])

                    neg_num_error.update(neg_pre, neg_gt)
                    pos_num_error.update(pos_pre, pos_gt)
                    neg_num_acc.update(neg_pre,neg_gt)
                    pos_num_acc.update(pos_pre,pos_gt)
                    neg_pos_num_acc.update(neg_pre, neg_gt, pos_pre, pos_gt)
                    if len(neg_pre) == len(neg_gt):
                        neg_location_error.update(neg_pre, neg_gt, h,w)
                    if len(pos_pre) == len(pos_gt):
                        pos_location_error.update(pos_pre, pos_gt,h,w)
                    if len(neg_pre) == len(neg_gt) and len(pos_pre) == len(pos_gt) and  len(pos_pre)+1==len(neg_pre):
                        neg_pos_overhang_error.update(neg_pre, neg_gt, pos_pre, pos_gt,h,w)


                neg_num_error = neg_num_error.show()
                pos_num_error = pos_num_error.show()
                neg_num_acc = neg_num_acc.show()
                pos_num_acc = pos_num_acc.show()
                neg_pos_num_acc = neg_pos_num_acc.show()
                neg_location_error = neg_location_error.show()
                pos_location_error = pos_location_error.show()
                neg_pos_overhang_error = neg_pos_overhang_error.show()
                log = 'method_name: {} dataset: {} neg_num_MAE: {:.4f} pos_num_MAE: {:.4f} neg_num_Acc: {:.4f} pos_num_Acc: {:.4f} neg_pos_num_Acc: {:.4f} neg_location_MAE: {:.10f} pos_location_MAE: {:.10f} neg_pos_overhang_MAE: {:.10f}'.format(method_name,name,neg_num_error,pos_num_error,neg_num_acc,pos_num_acc,neg_pos_num_acc,neg_location_error,pos_location_error,neg_pos_overhang_error)
                open(log_path, 'a').write(log + '\n')
                print(log)


if __name__ == '__main__':
    log_path = os.path.join('./model_results' + '.txt')
    main(log_path)