import cv2
import torch
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import DataLoader

from collections import defaultdict
from pandas import DataFrame
from tqdm.auto import tqdm

import models
import datasets
import evaluating
import preprocessing

import warnings
warnings.filterwarnings(action='ignore')


''' cuda '''
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
torch.cuda.empty_cache()  # GPU MEM Cache 제거

''' data load '''
test_df = preprocessing.load('.', 'data', 'test.csv')
test_dataset = datasets.CustomDataset(test_df['video_path'].values, None, 50, 128, 'test', None)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

''' inference '''
crash_model = models.ResNet18_3D(2)
print(f'Load Model : crash_infer_model')
crash_model.load_state_dict(torch.load(f'./model/best/crash_infer_model.pt'))
crash_model.to(device)
crash_model.eval()

ego_model = models.ResNet18_3D(2)
print(f'Load Model : ego_infer_model')
ego_model.load_state_dict(torch.load(f'./model/best/ego_infer_model.pt'))
ego_model.to(device)
ego_model.eval()

weather_model = models.ResNet18_3D(3)
print(f'Load Model : ego_infer_model')
weather_model.load_state_dict(torch.load(f'./model/best/weather_infer_model.pt'))
weather_model.to(device)
weather_model.eval()

timing_model = models.ResNet18_3D(2)
print(f'Load Model : timing_infer_model')
timing_model.load_state_dict(torch.load(f'./model/best/timing_infer_model.pt'))
timing_model.to(device)
timing_model.eval()

preds = []
with torch.no_grad():
    for videos in tqdm(iter(test_loader)):
        videos = videos.to(device)
        # print(videos)
        crash_preds = evaluating.inference(crash_model, videos)
        # print(crash_preds)
        for c_pred in crash_preds:
            if c_pred == 1:
                ego_preds = evaluating.inference(ego_model, videos)
                weather_preds = evaluating.inference(weather_model, videos)
                timing_preds = evaluating.inference(timing_model, videos)
                # print(ego_preds)
                # print(weather_preds)
                # print(timing_preds)
                for e_pred in ego_preds:
                    if e_pred == 0:
                        for w_pred in weather_preds:
                            if w_pred == 0:
                                for t_pred in timing_preds:
                                    if t_pred == 0:
                                        preds.append(1)
                                    else:
                                        preds.append(2)
                            elif w_pred == 1:
                                for t_pred in timing_preds:
                                    if t_pred == 0:
                                        preds.append(3)
                                    else:
                                        preds.append(4)
                            elif w_pred == 2:
                                for t_pred in timing_preds:
                                    if t_pred == 0:
                                        preds.append(5)
                                    else:
                                        preds.append(6)
                    else:
                        for w_pred in weather_preds:
                            if w_pred == 0:
                                for t_pred in timing_preds:
                                    if t_pred == 0:
                                        preds.append(7)
                                    else:
                                        preds.append(8)
                            elif w_pred == 1:
                                for t_pred in timing_preds:
                                    if t_pred == 0:
                                        preds.append(9)
                                    else:
                                        preds.append(10)
                            elif w_pred == 2:
                                for t_pred in timing_preds:
                                    if t_pred == 0:
                                        preds.append(11)
                                    else:
                                        preds.append(12)
            else:
                preds.append(crash_preds[0])

# print(preds)
# df = pd.DataFrame(preds)
# df.to_csv('./data/prediction.csv')
# print(preds[27])
red = (0, 0, 255)
font = cv2.FONT_HERSHEY_PLAIN
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_path_list = test_df['video_path'].values
for i in range(len(video_path_list)):
    frames = []
    cap = cv2.VideoCapture(video_path_list[i])
    out = cv2.VideoWriter(f'./result/{i}.mp4', fourcc, 30.0, (int(1028), int(720)))
    if preds[i] == 0:
        for _ in range(50):
            _, img = cap.read()
            img = cv2.resize(img, (1028, 720))
            img = cv2.putText(img, "Label:0(Crash:No/Ego:--/Weather:--/Timing:--)", (50, 100), font, 2, red, 2,
                              cv2.LINE_AA)
            frames.append(img)
    elif preds[i] == 1:
        for _ in range(50):
            _, img = cap.read()
            img = cv2.resize(img, (1028, 720))
            img = cv2.putText(img, "Label:1(Crash:Yes/Ego:Yes/Weather:Normal/Timing:Day)", (50, 100), font, 2, red, 2,
                              cv2.LINE_AA)
            frames.append(img)
    elif preds[i] == 2:
        for _ in range(50):
            _, img = cap.read()
            img = cv2.resize(img, (1028, 720))
            img = cv2.putText(img, "Label:2(Crash:Yes/Ego:Yes/Weather:Normal/Timing:Night)", (50, 100), font, 2, red, 2,
                              cv2.LINE_AA)
            frames.append(img)
    elif preds[i] == 3:
        for _ in range(50):
            _, img = cap.read()
            img = cv2.resize(img, (1028, 720))
            img = cv2.putText(img, "Label:3(Crash:Yes/Ego:Yes/Weather:Snowy/Timing:Day)", (50, 100), font, 2, red, 2,
                              cv2.LINE_AA)
            frames.append(img)
    elif preds[i] == 4:
        for _ in range(50):
            _, img = cap.read()
            img = cv2.resize(img, (1028, 720))
            img = cv2.putText(img, "Label:4(Crash:Yes/Ego:Yes/Weather:Snowy/Timing:Night)", (50, 100), font, 2, red, 2,
                              cv2.LINE_AA)
            frames.append(img)
    elif preds[i] == 5:
        for _ in range(50):
            _, img = cap.read()
            img = cv2.resize(img, (1028, 720))
            img = cv2.putText(img, "Label:5(Crash:Yes/Ego:Yes/Weather:Rainy/Timing:Day)", (50, 100), font, 2, red, 2,
                              cv2.LINE_AA)
            frames.append(img)
    elif preds[i] == 6:
        for _ in range(50):
            _, img = cap.read()
            img = cv2.resize(img, (1028, 720))
            img = cv2.putText(img, "Label:6(Crash:Yes/Ego:Yes/Weather:Rainy/Timing:Night)", (50, 100), font, 2, red, 2,
                              cv2.LINE_AA)
            frames.append(img)
    elif preds[i] == 7:
        for _ in range(50):
            _, img = cap.read()
            img = cv2.resize(img, (1028, 720))
            img = cv2.putText(img, "Label:7(Crash:Yes/Ego:No/Weather:Normal/Timing:Day)", (50, 100), font, 2, red, 2,
                              cv2.LINE_AA)
            frames.append(img)
    elif preds[i] == 8:
        for _ in range(50):
            _, img = cap.read()
            img = cv2.resize(img, (1028, 720))
            img = cv2.putText(img, "Label:8(Crash:Yes/Ego:No/Weather:Normal/Timing:Night)", (50, 100), font, 2, red, 2,
                              cv2.LINE_AA)
            frames.append(img)
    elif preds[i] == 9:
        for _ in range(50):
            _, img = cap.read()
            img = cv2.resize(img, (1028, 720))
            img = cv2.putText(img, "Label:9(Crash:Yes/Ego:No/Weather:Snowy/Timing:Day)", (50, 100), font, 2, red, 2,
                              cv2.LINE_AA)
            frames.append(img)
    elif preds[i] == 10:
        for _ in range(50):
            _, img = cap.read()
            img = cv2.resize(img, (1028, 720))
            img = cv2.putText(img, "Label:7(Crash:Yes/Ego:No/Weather:Snowy/Timing:Night)", (50, 100), font, 2, red, 2,
                              cv2.LINE_AA)
            frames.append(img)
    elif preds[i] == 11:
        for _ in range(50):
            _, img = cap.read()
            img = cv2.resize(img, (1028, 720))
            img = cv2.putText(img, "Label:11(Crash:Yes/Ego:No/Weather:Rainy/Timing:Day)", (50, 100), font, 2, red, 2,
                              cv2.LINE_AA)
            frames.append(img)
    elif preds[i] == 12:
        for _ in range(50):
            _, img = cap.read()
            img = cv2.resize(img, (1028, 720))
            img = cv2.putText(img, "Label:12(Crash:Yes/Ego:No/Weather:Rainy/Timing:Night)", (50, 100), font, 2, red, 2,
                              cv2.LINE_AA)
            frames.append(img)

    for i in range(50):
        f_cur = frames[i]
        out.write(f_cur)
        # cv2.imwrite('./result/%02d.jpg' % i, f_cur)
    out.release()