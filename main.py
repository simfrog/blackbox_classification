##############
''' IMPORT '''
##############
import os
import random
import argparse
import numpy as np
import pandas as pd

import torch
import torch.optim as optim
import torchvision.models as models

from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import models
import datasets
import training
import trainingfold
import evaluating
import oversampling
import preprocessing

import warnings
warnings.filterwarnings(action='ignore')


######################
''' HYPERPARAMETER '''
######################
is_fold = False
is_oversample = False
is_pesudo = True

CFG = {
    'FILE_NAME': 'ResNet18_3D_rSwDlSpLyV_translate2',
    'VIDEO_LENGTH': 50, # 10 Frame * 5 seconds
    'IMG_SIZE': 256,
    'EPOCHS': 30,
    'FOLD': 3,
    'LEARNING_RATE': 1e-4,
    'BATCH_SIZE': 4,
    'SEED': 42
}

########################
''' FIXED RANDOMSEED '''
########################
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

###################
''' SET WEIGHTS '''
###################
def make_weights(labels, nclasses):
    labels = np.array(labels)
    weight_arr = np.zeros_like(labels)
    _, counts = np.unique(labels, return_counts=True)
    for cls in range(nclasses):
        weight_arr = np.where(labels == cls, 1/counts[cls], weight_arr)
    return weight_arr

############
''' MAIN '''
############
def main(args):
    ''' set hyperparameter '''
    CFG['FILE_NAME'] = args.filename
    CFG['EPOCHS'] = args.epoch
    CFG['FOLD'] = args.fold
    CFG['BATCH_SIZE'] = args.bs
    CFG['LEARNING_RATE'] = args.lr
    CFG['IMG_SIZE'] = args.size
    CFG['SEED'] = args.seed

    ''' fix seed '''
    print(f"Fix Random Seed : {CFG['SEED']}\n")
    seed_everything(CFG['SEED'])

    ''' cuda '''
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.cuda.empty_cache()  # Remove GPU MEM Cache

    ''' data load '''
    crash_df = preprocessing.load('.', 'data', 'train.csv')
    crash_df = preprocessing.separate(crash_df, 'crash')

    ego_df = preprocessing.load('.', 'data', 'train.csv')
    ego_df = preprocessing.separate(ego_df, 'ego_involve')

    weather_df = preprocessing.load('.', 'data', 'train.csv')
    weather_df = preprocessing.separate(weather_df, 'weather')
    if is_pesudo == True:
        pseudo_df1 = pd.read_csv('./data/pseudo_weather_best1.csv')
        weather_df = pd.concat([weather_df, pseudo_df1], ignore_index=True)
    # df1 = pd.DataFrame(weather_df)
    # df1.to_csv('./data/drop_weather.csv', index=None)

    timing_df = preprocessing.load('.', 'data', 'train.csv')
    timing_df = preprocessing.separate(timing_df, 'timing')
    # if is_pesudo == True:
    #     pseudo_df = pd.read_csv('./data/pseudo_night1.csv')
    #     timing_df = pd.concat([timing_df, pseudo_df], ignore_index=True)

    test_df = preprocessing.load('.', 'data', 'test.csv')
    print(weather_df.tail())
    print(len(weather_df['label']))

    ''' over sampling '''
    if is_oversample == True:
        cover_df = oversampling.addition(crash_df, 'crash')
        wover_df = oversampling.addition(weather_df, 'weather')
        tover_df = oversampling.addition(timing_df, 'timing')

    ''' train/validation split '''
    if is_fold == False:
        crash_train, crash_val, _, _ = train_test_split(crash_df, crash_df['label'], test_size=0.2, random_state=CFG['SEED'], stratify=crash_df['label'])
        ego_train, ego_val, _, _ = train_test_split(ego_df, ego_df['label'], test_size=0.2, random_state=CFG['SEED'], stratify=ego_df['label'])
        weather_train, weather_val, _, _ = train_test_split(weather_df, weather_df['label'], test_size=0.2, random_state=CFG['SEED'], stratify=weather_df['label'])
        timing_train, timing_val, _, _ = train_test_split(timing_df, timing_df['label'], test_size=0.2, random_state=CFG['SEED'], stratify=timing_df['label'])
    if is_oversample == True:
        crash_train, crash_val, _, _ = train_test_split(cover_df, cover_df['label'], test_size=0.2, random_state=CFG['SEED'])
        ego_train, ego_val, _, _ = train_test_split(ego_df, ego_df['label'], test_size=0.2, random_state=CFG['SEED'])
        weather_train, weather_val, _, _ = train_test_split(wover_df, wover_df['label'], test_size=0.2, random_state=CFG['SEED'])
        timing_train, timing_val, _, _ = train_test_split(tover_df, tover_df['label'], test_size=0.2, random_state=CFG['SEED'])

    ''' weights '''
    crash_weights = make_weights(crash_train['label'].values, len(np.unique(crash_train['label'])))
    crash_weights = torch.DoubleTensor(crash_weights)
    crash_sampler = torch.utils.data.sampler.WeightedRandomSampler(crash_weights, len(crash_weights))

    ego_weights = make_weights(ego_train['label'].values, len(np.unique(ego_train['label'])))
    ego_weights = torch.DoubleTensor(ego_weights)
    ego_sampler = torch.utils.data.sampler.WeightedRandomSampler(ego_weights, len(ego_weights))

    timing_weights = make_weights(timing_train['label'].values, len(np.unique(timing_train['label'])))
    timing_weights = torch.DoubleTensor(timing_weights)
    timing_sampler = torch.utils.data.sampler.WeightedRandomSampler(timing_weights, len(timing_weights))

    weather_weights = make_weights(weather_train['label'].values, len(np.unique(weather_train['label'])))
    weather_weights = torch.DoubleTensor(weather_weights)
    weather_sampler = torch.utils.data.sampler.WeightedRandomSampler(weather_weights, len(weather_weights))

    ''' custom dataset '''
    if is_fold == False:
        # crash
        cT_dataset = datasets.CustomDataset(crash_train['video_path'].values, crash_train['label'].values,
                                              CFG['VIDEO_LENGTH'], CFG['IMG_SIZE'], 'train', 'crash')
        cT_loader = DataLoader(cT_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, sampler=crash_sampler, num_workers=0)
        cV_dataset = datasets.CustomDataset(crash_val['video_path'].values, crash_val['label'].values,
                                            CFG['VIDEO_LENGTH'], CFG['IMG_SIZE'], 'test', 'crash')
        cV_loader = DataLoader(cV_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)
        # ego_involve
        eT_dataset = datasets.CustomDataset(ego_train['video_path'].values, ego_train['label'].values,
                                            CFG['VIDEO_LENGTH'], CFG['IMG_SIZE'], 'train', 'ego_involve')
        eT_loader = DataLoader(eT_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, sampler=ego_sampler, num_workers=0)
        eV_dataset = datasets.CustomDataset(ego_val['video_path'].values, ego_val['label'].values,
                                            CFG['VIDEO_LENGTH'], CFG['IMG_SIZE'], 'test', 'ego_involve')
        eV_loader = DataLoader(eV_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)
        # weather
        wT_dataset = datasets.CustomDataset(weather_train['video_path'].values, weather_train['label'].values,
                                            CFG['VIDEO_LENGTH'], CFG['IMG_SIZE'], 'train', 'weather')
        wT_loader = DataLoader(wT_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, sampler=weather_sampler, num_workers=0)
        wV_dataset = datasets.CustomDataset(weather_val['video_path'].values, weather_val['label'].values,
                                            CFG['VIDEO_LENGTH'], CFG['IMG_SIZE'], 'test', 'weather')
        wV_loader = DataLoader(wV_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)
        # timing
        tT_dataset = datasets.CustomDataset(timing_train['video_path'].values, timing_train['label'].values,
                                            CFG['VIDEO_LENGTH'], CFG['IMG_SIZE'], 'train', 'timing')
        tT_loader = DataLoader(tT_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, sampler=timing_sampler, num_workers=0)
        tV_dataset = datasets.CustomDataset(timing_val['video_path'].values, timing_val['label'].values,
                                            CFG['VIDEO_LENGTH'], CFG['IMG_SIZE'], 'test', 'timing')
        tV_loader = DataLoader(tV_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

    # test
    test_dataset = datasets.CustomDataset(test_df['video_path'].values, None, CFG['VIDEO_LENGTH'], CFG['IMG_SIZE'], 'test', None)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    ''' model : slowfast '''
    # model = models.SlowFast(CFG['IMG_HEIGHT'])
    # model.eval()
    # best_model = None
    # optimizer = optim.AdamW(params=model.parameters(), lr=CFG["LEARNING_RATE"], weight_decay=0.02)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[12], gamma=0.1)
    # model = model.to(device)
    #
    # infer_model = training.train(model, optimizer, train_loader, val_loader, scheduler, device, CFG['EPOCHS'], CFG['FILE_NAME'])
    # torch.save(infer_model.state_dict(), f'./model/{CFG["FILE_NAME"]}.pt')

    ''' model : 3D ResNet18'''
    # crash
    print("Training Crash Model")
    crash_classes = 2
    if is_fold == False:
        crash_model = models.ResNet18_3D(crash_classes)
        crash_model.eval()

        len_params = len(list(crash_model.parameters()))
        idx_start = len_params // 4
        for idx, param in enumerate(crash_model.parameters()):
            if idx < idx_start:
                param.requires_grad = False
            else:
                param.requires_grad = True
        optimizer = optim.AdamW(params=list(crash_model.parameters())[idx_start:], lr=CFG["LEARNING_RATE"], weight_decay=0.02)
        # optimizer = optim.AdamW(params=crash_model.parameters(), lr=CFG["LEARNING_RATE"], weight_decay=0.02)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, threshold_mode='abs', min_lr=1e-8, verbose=True)

        # crash_infer_model, crash_best_score = training.train2(crash_model, optimizer, cT_loader, scheduler,
        #                                                      device, crash_classes, 10, 'crash_infer_model')
        crash_infer_model, crash_best_score = training.train(crash_model, optimizer, cT_loader, cV_loader, scheduler,
                                                             device, crash_classes, 10, 'crash_infer_model')
    else:
        crash_infer_model, crash_best_score = trainingfold.train(crash_df, CFG, device, crash_classes, 10, 'crash_infer_model')

    torch.save(crash_infer_model.state_dict(), f'./model/crash_infer_model.pt')
    print(f"Save Success. Best F1 Score : {crash_best_score}")

    del crash_model
    del crash_infer_model
    torch.cuda.empty_cache()

    # ego_involve
    print("Training Ego_Involve Model")
    ego_classes = 2
    if is_fold == False:
        ego_model = models.ResNet18_3D(ego_classes)
        ego_model.eval()

        len_params = len(list(ego_model.parameters()))
        idx_start = len_params // 4
        for idx, param in enumerate(ego_model.parameters()):
            if idx < idx_start:
                param.requires_grad = False
            else:
                param.requires_grad = True
        optimizer = optim.AdamW(params=list(ego_model.parameters())[idx_start:], lr=CFG["LEARNING_RATE"], weight_decay=0.02)
        # optimizer = optim.AdamW(params=ego_model.parameters(), lr=CFG["LEARNING_RATE"], weight_decay=0.02)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, threshold_mode='abs', min_lr=1e-8, verbose=True)

        # ego_infer_model, ego_best_score = training.train2(ego_model, optimizer, eT_loader, scheduler,
        #                                                  device, ego_classes, CFG['EPOCHS'], 'ego_infer_model')
        ego_infer_model, ego_best_score = training.train(ego_model, optimizer, eT_loader, eV_loader, scheduler,
                                                         device, ego_classes, CFG['EPOCHS'], 'ego_infer_model')
    else:
        ego_path = [x for x in ego_df['video_path']]
        ego_label = [x for x in ego_df['label']]
        ego_infer_model, ego_best_score = trainingfold.train(ego_df, CFG, device, ego_classes, CFG['EPOCHS'], 'ego_infer_model')

    torch.save(ego_infer_model.state_dict(), f'./model/ego_infer_model.pt')
    print(f"Save Success. Best F1 Score : {ego_best_score}")

    del ego_model
    del ego_infer_model
    torch.cuda.empty_cache()

    # weather
    print("Training Weather Model")
    weather_classes = 3
    if is_fold == False:
        weather_model = models.ResNet18_3D(weather_classes)
        weather_model.eval()

        len_params = len(list(weather_model.parameters()))
        idx_start = len_params // 4
        for idx, param in enumerate(weather_model.parameters()):
            if idx < idx_start:
                param.requires_grad = False
            else:
                param.requires_grad = True
        optimizer = optim.AdamW(params=list(weather_model.parameters())[idx_start:], lr=CFG["LEARNING_RATE"],
                                weight_decay=0.02)
        # optimizer = optim.AdamW(params=weather_model.parameters(), lr=CFG["LEARNING_RATE"], weight_decay=0.02)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, threshold_mode='abs', min_lr=1e-8, verbose=True)

        # weather_infer_model, weather_best_score = training.train2(weather_model, optimizer, wT_loader, scheduler,
        #                                                          device, weather_classes, 50, 'weather_infer_model')
        weather_infer_model, weather_best_score = training.train(weather_model, optimizer, wT_loader, wV_loader,
                                                                 scheduler, device, weather_classes, 50, 'weather_infer_model')
    else:
        weather_path = [x for x in weather_df['video_path']]
        weather_label = [x for x in weather_df['label']]
        weather_infer_model, weather_best_score = trainingfold.train(weather_df, CFG, device, weather_classes, 50, 'weather_infer_model')
    torch.save(weather_infer_model.state_dict(), f'./model/weather_infer_model.pt')
    print(f"Save Success. Best F1 Score : {weather_best_score}")

    del weather_model
    del weather_infer_model
    torch.cuda.empty_cache()

    # timing
    print("Training Timing Model")
    timing_classes = 2
    if is_fold == False:
        timing_model = models.ResNet18_3D(timing_classes)
        timing_model.eval()

        len_params = len(list(timing_model.parameters()))
        idx_start = len_params // 4
        for idx, param in enumerate(timing_model.parameters()):
            if idx < idx_start:
                param.requires_grad = False
            else:
                param.requires_grad = True
        optimizer = optim.AdamW(params=list(timing_model.parameters())[idx_start:], lr=CFG["LEARNING_RATE"], weight_decay=0.02)
        # optimizer = optim.AdamW(params=timing_model.parameters(), lr=CFG["LEARNING_RATE"], weight_decay=0.02)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2,
                                                         threshold_mode='abs', min_lr=1e-8, verbose=True)

        # timing_infer_model, timing_best_score = training.train2(timing_model, optimizer, tT_loader, scheduler,
        #                                                        device, timing_classes, CFG['EPOCHS'], 'timing_infer_model')
        timing_infer_model, timing_best_score = training.train(timing_model, optimizer, tT_loader, tV_loader, scheduler,
                                                               device, timing_classes, CFG['EPOCHS'],
                                                               'timing_infer_model')
    else:
        timing_path = [x for x in timing_df['video_path']]
        timing_label = [x for x in timing_df['label']]
        timing_infer_model, timing_best_score = trainingfold.train(timing_df, CFG, device, timing_classes, CFG['EPOCHS'], 'timing_infer_model')

    torch.save(timing_infer_model.state_dict(), f'./model/timing_infer_model.pt')
    print(f"Save Success. Best F1 Score : {timing_best_score}")

    del timing_model
    del timing_infer_model
    torch.cuda.empty_cache()

    ''' inference '''
    crash_model = models.ResNet18_3D(crash_classes)
    print(f'Load Model : crash_infer_model')
    crash_model.load_state_dict(torch.load(f'./model/crash_infer_model.pt'))
    crash_model.to(device)
    crash_model.eval()

    ego_model = models.ResNet18_3D(ego_classes)
    print(f'Load Model : ego_infer_model')
    ego_model.load_state_dict(torch.load(f'./model/ego_infer_model.pt'))
    ego_model.to(device)
    ego_model.eval()

    weather_model = models.ResNet18_3D(weather_classes)
    print(f'Load Model : ego_infer_model')
    weather_model.load_state_dict(torch.load(f'./model/weather_infer_model.pt'))
    weather_model.to(device)
    weather_model.eval()

    timing_model = models.ResNet18_3D(timing_classes)
    print(f'Load Model : timing_infer_model')
    timing_model.load_state_dict(torch.load(f'./model/timing_infer_model.pt'))
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

    print(len(preds))
    # print(preds)

    ''' submission '''
    submit = pd.read_csv('./data/sample_submission.csv')

    submit['label'] = preds
    print(submit.head())

    submit.to_csv(f'./submit/{CFG["FILE_NAME"]}.csv', index=False)

    ''' slowfast submit '''
    # submit['crash'] = crash_preds
    # submit['weather'] = weather_preds
    # submit['timing'] = timing_preds
    #
    # submit['label'] = 1 + (submit['crash'] - 1) * 6 + submit['weather'] * 2 + submit['timing']
    # submit['label'][submit['crash'] == 0] = 0
    # submit[['sample_id', 'label']].head(10)
    #
    # submit[['sample_id', 'label']].to_csv(f'./submit/{CFG["FILE_NAME"]}.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Argparse Tutorial')
    parser.add_argument('--epoch', type=int, default=CFG['EPOCHS'])
    parser.add_argument('--fold', type=int, default=CFG['FOLD'])
    parser.add_argument('--bs', type=int, default=CFG['BATCH_SIZE'])
    parser.add_argument('--lr', type=float, default=CFG['LEARNING_RATE'])
    parser.add_argument('--size', type=int, default=CFG['IMG_SIZE'])
    parser.add_argument('--seed', type=int, default=CFG['SEED'])
    parser.add_argument('--filename', default=CFG['FILE_NAME'])

    args = parser.parse_args()
    main(args=args)