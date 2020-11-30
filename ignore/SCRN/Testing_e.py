import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
import cv2
from scripts.SCRN.lib.ResNet_models import SCRN
from utils.data_SCRN import get_loader, test_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--gpu_id', type=str, default='1', help='select gpu id')
parser.add_argument('--snapshot', type=str,
                    default='./snapshot/1021_SCRN_e/Net_epoch_best.pth')
parser.add_argument('--test_path', type=str, default='/media/nercms/NERCMS/Dataset/SOD/TestDataset/', help='test dataset path')
opt = parser.parse_args()

dataset_path = opt.test_path

# load the model
model = SCRN(channel=32).cuda()
model.load_state_dict(torch.load(opt.snapshot))
model.cuda()
model.eval()

# test
test_datasets = ['DUT-OMRON', 'DUTS-test', 'ECSSD', 'HKU-IS', 'Judd-A', 'MSRA-B-test', 'MSRA10K',
                 'PASCAL-S', 'SED2', 'SOC-test', 'SOD', 'THUR15K']
for dataset in test_datasets:
    save_path = './res/{}/'.format(opt.snapshot.split('/')[-2]) + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    image_root = dataset_path + 'SalMap/' + dataset + '/'
    gt_root = dataset_path + 'GT/' + dataset + '/'
    # depth_root = dataset_path + dataset + '/depth/'

    test_loader = test_dataset(image_root, gt_root, gt_root, opt.testsize)
    for i in range(test_loader.size):
        image, gt, _, name, image_for_post = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        # depth = depth.cuda()

        res = model(image)

        res = F.upsample(res[0], size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        print('Save Img To: ', save_path + name)
        cv2.imwrite(save_path + name, res * 255)
    print('Test Done!')
