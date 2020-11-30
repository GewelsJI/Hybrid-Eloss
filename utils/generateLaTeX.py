import os

# 1. for camouflaged object detection (SINet)
# model_lst = ['1011_SINet_wce', '1012_SINet_wiou', '1012_SINet_e', '1011_SINet_bei']
# dataset_lst = ['CHAMELEON', 'CAMO', 'COD10K']
# metric_idx = [0, 8, 5, 2]

# 2. for polyp segmentation
model_lst = ['1020_PraNet_wiou_90'] #['1013_PraNet_wce', '1013_PraNet_wiou', '1013_PraNet_e', '1012_PraNet_bei']
dataset_lst = ['CVC-300', 'CVC-ClinicDB', 'Kvasir']
metric_idx = [0, 1, 2, 5]

for model_i in model_lst:
    print('% {}'.format(model_i))
    for dataset_i in dataset_lst:
        txt_path = '../scripts/{}/eval/EvaluateResults3/{}_result.txt'.format(model_lst[0].split('_')[1], dataset_i)
        with open(txt_path) as f:
            # print('% {}'.format(model_i))
            for line in f.readlines():
                if model_i in line:
                    for idx in metric_idx:
                        metric_value = line[:-2].split(') ')[1].split(';')[idx].strip().split(':')[1]
                        # print(idx, metric_value)
                        print('& {}'.format(metric_value), end=' ')
        print('\n')
