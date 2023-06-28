##############
''' IMPORT '''
##############
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from collections import defaultdict
from pandas import DataFrame
from tqdm.auto import tqdm

import models
import datasets
import preprocessing

import warnings
warnings.filterwarnings(action='ignore')

######################
''' HYPERPARAMETER '''
######################
is_file = True

#####################
''' PSEUDO LABEL '''
#####################
def makelabel(model, videos):
    videos = videos.to(device)
    logit = model(videos)
    preds = F.softmax(logit, dim=1)
    return preds

''' cuda '''
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
torch.cuda.empty_cache()  # GPU MEM Cache 제거

''' data load '''
train_df = preprocessing.load('.', 'data', 'train.csv')
unlabeled_df = preprocessing.unlabeled(df=train_df, is_file=is_file, filename='pseudo_weather_best1')

unlabeled_dataset = datasets.CustomDataset2(unlabeled_df['sample_id'].values, unlabeled_df['video_path'].values, None, 50, 128, 'test', None)
unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=1, shuffle=False, num_workers=0)

''' model load'''
weather_model = models.ResNet18_3D(3)
print(f'Load Model : weather_infer_model')
weather_model.load_state_dict(torch.load(f'/home/oem/Desktop/SnowyLittleDrop/model/weather_infer_model.pt'))
weather_model.to(device)
weather_model.eval()

# timing_model = models.ResNet18_3D(2)
# print(f'Load Model : timing_model')
# timing_model.load_state_dict(torch.load(f'./pseudo/first/timing_infer_model.pt'))
# timing_model.to(device)
# timing_model.eval()

''' make pseudo label '''
# weather
pseudo_df = defaultdict(list)
normal_df = defaultdict(list)
count = 0
with torch.no_grad():
    for videos, name, idx in tqdm(iter(unlabeled_loader)):
    # for videos, name, idx in iter(unlabeled_loader):
        videos = videos.to(device)
        preds = makelabel(weather_model, videos)
        if preds[0][0] >= 0.9:
            normal_df['sample_id'].append(idx[0])
            normal_df['video_path'].append(name[0])
            normal_df['label'].append(0)
        elif preds[0][1] >= 0.8:
            pseudo_df['sample_id'].append(idx[0])
            pseudo_df['video_path'].append(name[0])
            pseudo_df['label'].append(1)
        elif preds[0][2] >= 0.86:
            count+=1
            pseudo_df['sample_id'].append(idx[0])
            pseudo_df['video_path'].append(name[0])
            pseudo_df['label'].append(2)
        # if preds[0][2] > preds[0][0] and preds[0][2] > preds[0][1]:
        # if preds[0][0] >= 0.9:
        #     print(preds)
        #     print(name)

# timing
# day_df = defaultdict(list)
# count = 0
# with torch.no_grad():
#     for videos, name, idx in tqdm(iter(unlabeled_loader)):
#     # for videos, name, idx in iter(unlabeled_loader):
#         videos = videos.to(device)
#         preds = makelabel(timing_model, videos)
#         if preds[0][0] >= 0.9:
#             day_df['sample_id'].append(idx[0])
#             day_df['video_path'].append(name[0])
#             day_df['label'].append(0)
#         if preds[0][1] >= 0.9:
#             count+=1
#             pseudo_df['sample_id'].append(idx[0])
#             pseudo_df['video_path'].append(name[0])
#             pseudo_df['label'].append(1)
#         # if preds[0][1] >= 0.9:
#         #     print(preds)
#         #     print(name)

print(count)
df1 = DataFrame(pseudo_df)
df2 = DataFrame(normal_df)
# # print(df)
df1.to_csv('./data/pseudo_weather_best2.csv', index=None)
df2.to_csv('./data/pseudo_normal2.csv', index=None)