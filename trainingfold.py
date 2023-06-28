##############
''' IMPORT '''
##############
import numpy as np
import pandas as pd
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pytorchtools import EarlyStopping

from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

import models
import datasets

##############
''' CUMTIX '''
##############
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def cutmix(data, target, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_target = target[indices]

    lam = np.clip(np.random.beta(alpha, alpha),0.3,0.4)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    new_data = data.clone()
    new_data[:, :, bby1:bby2, bbx1:bbx2] = data[indices, :, bby1:bby2, bbx1:bbx2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
    targets = (target, shuffled_target, lam)

    return new_data, targets

#########################################
''' LABEL SMOOTHING CROSSENTROPY LOSS '''
#########################################
class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, classes, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

#######################
''' ASYMMETRIC LOSS '''
#######################
class ASLSingleLabel(nn.Module):
    '''
    This loss is intended for single-label classification problems
    '''
    def __init__(self, gamma_pos=0, gamma_neg=4, eps: float = 0.1, reduction='mean'):
        super(ASLSingleLabel, self).__init__()

        self.eps = eps
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.targets_classes = []
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.reduction = reduction

    def forward(self, inputs, target):
        '''
        "input" dimensions: - (batch_size,number_classes)
        "target" dimensions: - (batch_size)
        '''
        num_classes = inputs.size()[-1]
        log_preds = self.logsoftmax(inputs)
        self.targets_classes = torch.zeros_like(inputs).scatter_(1, target.long().unsqueeze(1), 1)

        # ASL weights
        targets = self.targets_classes
        anti_targets = 1 - targets
        xs_pos = torch.exp(log_preds)
        xs_neg = 1 - xs_pos
        xs_pos = xs_pos * targets
        xs_neg = xs_neg * anti_targets
        asymmetric_w = torch.pow(1 - xs_pos - xs_neg,
                                 self.gamma_pos * targets + self.gamma_neg * anti_targets)
        log_preds = log_preds * asymmetric_w

        if self.eps > 0:  # label smoothing
            self.targets_classes = self.targets_classes.mul(1 - self.eps).add(self.eps / num_classes)

        # loss calculation
        loss = - self.targets_classes.mul(log_preds)

        loss = loss.sum(dim=-1)
        if self.reduction == 'mean':
            loss = loss.mean()

        return loss

#######################
''' TRAIN WITH FOLD '''
#######################
def train(df, targets, CFG, device, num_classes, epochs, filename):
    criterion = LabelSmoothingLoss(classes=num_classes).to(device)
    # criterion = ASLSingleLabel().to(device)

    best_val_score = 0
    best_model = None

    data = defaultdict(list)

    skf = StratifiedKFold(n_splits=CFG['FOLD'], random_state=CFG['SEED'], shuffle=True)

    paths = [x for x in df['video_path']]
    labels = [x for x in df['label']]

    for fold, (train_idx, val_idx) in enumerate(skf.split(paths, targets)):
        print(f"{fold+1}-fold")

        model = models.ResNet18_3D(num_classes)
        model.eval()
        model.to(device)

        len_params = len(list(model.parameters()))
        idx_start = len_params // 4
        for idx, param in enumerate(model.parameters()):
            if idx < idx_start:
                param.requires_grad = False
            else:
                param.requires_grad = True
        optimizer = optim.AdamW(params=list(model.parameters())[idx_start:], lr=CFG["LEARNING_RATE"],
                                weight_decay=0.02)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2,
                                                         threshold_mode='abs', min_lr=1e-8, verbose=True)

        train_set = [paths[i] for i in train_idx]
        train_label = [targets[i] for i in train_idx]
        val_set = [paths[i] for i in val_idx]
        val_label = [targets[i] for i in val_idx]

        if filename == 'crash_infer_model':
            train_dataset = datasets.CustomDataset(train_set, train_label, CFG['VIDEO_LENGTH'], CFG['IMG_SIZE'], 'train', 'crash')
            train_loader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True, num_workers=0)
            valid_dataset = datasets.CustomDataset(val_set, val_label, CFG['VIDEO_LENGTH'], CFG['IMG_SIZE'], 'test', 'crash')
            valid_loader = DataLoader(valid_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)
        elif filename == 'ego_infer_model':
            train_dataset = datasets.CustomDataset(train_set, train_label, CFG['VIDEO_LENGTH'], CFG['IMG_SIZE'], 'train', 'ego_involve')
            train_loader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True, num_workers=0)
            valid_dataset = datasets.CustomDataset(val_set, val_label, CFG['VIDEO_LENGTH'], CFG['IMG_SIZE'], 'test', 'ego_involve')
            valid_loader = DataLoader(valid_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)
        elif filename == 'weather_infer_model':
            train_dataset = datasets.CustomDataset(train_set, train_label, CFG['VIDEO_LENGTH'], CFG['IMG_SIZE'], 'train', 'weather')
            train_loader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True, num_workers=0)
            valid_dataset = datasets.CustomDataset(val_set, val_label, CFG['VIDEO_LENGTH'], CFG['IMG_SIZE'], 'test', 'weather')
            valid_loader = DataLoader(valid_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)
        elif filename == 'timing_infer_model':
            train_dataset = datasets.CustomDataset(train_set, train_label, CFG['VIDEO_LENGTH'], CFG['IMG_SIZE'], 'train', 'timing')
            train_loader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True, num_workers=0)
            valid_dataset = datasets.CustomDataset(val_set, val_label, CFG['VIDEO_LENGTH'], CFG['IMG_SIZE'], 'test', 'timing')
            valid_loader = DataLoader(valid_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

        for epoch in range(1, epochs+1):
            print(f"\n++++++++++++++++++++ Start Epoch {epoch} ++++++++++++++++++++")
            model.train()
            train_loss = []
            for videos, labels in tqdm(iter(train_loader)):
                videos = videos.to(device)
                labels = labels.to(device)

                mix_decision = np.random.rand()

                if mix_decision < 0.5:
                    videos, videos_labels = cutmix(videos, labels, 1.)

                optimizer.zero_grad()

                output = model(videos)
                if mix_decision < 0.5:
                    loss = criterion(output, videos_labels[0]) * videos_labels[2] + criterion(output,videos_labels[1]) * (1. - videos_labels[2])
                else:
                    loss = criterion(output, labels)
                # loss = criterion(output, labels)

                loss.backward()
                optimizer.step()

                train_loss.append(loss.item())

            _val_loss, _val_score = validation(model, criterion, valid_loader, device)
            _train_loss = np.mean(train_loss)
            print(f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}] Val F1 : [{_val_score:.5f}]')

            data['train_loss'].append(_train_loss)
            data['val_loss'].append(_val_loss)
            data['val_f1'].append(_val_score)

            if scheduler is not None:
                scheduler.step(_val_score)

            if best_val_score < _val_score:
                best_val_score = _val_score
                best_model = model

    data_df = pd.DataFrame(data)
    data_df.to_csv(f"./process/{filename}.csv")

    return best_model, best_val_score

def validation(model, criterion, val_loader, device):
    model.eval()
    val_loss = []
    preds, trues = [], []

    with torch.no_grad():
        for videos, labels in tqdm(iter(val_loader)):
            videos = videos.to(device)
            labels = labels.to(device)

            logit = model(videos)
            loss = criterion(logit, labels)

            val_loss.append(loss.item())

            preds += logit.argmax(1).detach().cpu().numpy().tolist()
            trues += labels.detach().cpu().numpy().tolist()

        _val_loss = np.mean(val_loss)

    _val_score = f1_score(trues, preds, average='macro')
    return _val_loss, _val_score

# SlowFast Train
'''
def convert_to_label(crash_ego,weather,timing):
  labels = 1 + (crash_ego-1)*6 + weather*2 + timing
  labels[crash_ego==0] = 0
  return labels

def train(model, optimizer, train_loader, val_loader, scheduler, device, epochs, filename):
    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    best_val_score = 0
    best_model = None

    data = defaultdict(list)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = []
        for slow_videos, fast_videos, labels in tqdm(iter(train_loader)):
            slow_videos = slow_videos.to(device)
            fast_videos = fast_videos.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            crash_ego, weather, timing = model([slow_videos, fast_videos])

            crashego_loss = criterion(crash_ego, labels[:, 0])
            weather_loss = criterion(weather, labels[:, 1])
            timing_loss = criterion(timing, labels[:, 2])
            loss = crashego_loss + weather_loss + timing_loss

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        val_losses, val_scores, acc = validation(model, criterion, val_loader, device)
        # _val_loss, _val_score, acc, _val_lossE1, _val_lossE2, _val_lossE3 = validation(model, criterion, val_loader, device)
        _train_loss = np.mean(train_loss)
        # print(
        #     f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}] Val F1 : [{_val_score:.5f}]\n'
        #     f'Crash&Ego Val Loss L [{_val_lossE1:.5f}], Weather Val Loss L [{_val_lossE2:.5f}], Timing Val Loss L [{_val_lossE3:.5f}]')
        print(
            f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Val Loss : [{val_losses[0]:.5f}] Val F1 : [{val_scores[0]:.5f}]\n'
            f'Crash&Ego Val Loss L [{val_losses[1]:.5f}], Weather Val Loss L [{val_losses[2]:.5f}], Timing Val Loss L [{val_losses[3]:.5f}]')

        data['train_loss'].append(_train_loss)
        data['val_loss'].append(val_losses[0])
        data['val_crashego_loss'].append(val_losses[1])
        data['val_weather_loss'].append(val_losses[2])
        data['val_timing_loss'].append(val_losses[3])
        data['val_f1'].append(val_losses[0])
        data['val_crashego_f1'].append(val_scores[1])
        data['val_weather_f1'].append(val_scores[2])
        data['val_timing_f1'].append(val_scores[3])
        data['accuracy'].append(acc[0])
        data['crashego_accuracy'].append(acc[1])
        data['weather_accuracy'].append(acc[2])
        data['timing_accuracy'].append(acc[3])

        if scheduler is not None:
            scheduler.step(val_scores[0])

        if best_val_score < val_scores[0]:
            best_val_score = val_scores[0]
            best_model = model

    data_df = pd.DataFrame(data)
    data_df.to_csv(f"./process/{filename}.csv")

    return best_model


def validation(model, criterion, val_loader, device):
    model.eval()
    val_loss = []
    val_lossE1, val_lossE2, val_lossE3 = [], [], []
    crashego_preds, weather_preds, timing_preds = [], [], []
    crashego_trues, weather_trues, timing_trues = [], [], []
    accuracies, losses, scores = [], [], []

    with torch.no_grad():
        for fast_videos, videos, labels in tqdm(iter(val_loader)):
            fast_videos = fast_videos.to(device)
            videos = videos.to(device)
            labels = labels.to(device)

            crash_ego, weather, timing = model([fast_videos, videos])
            crashego_loss = criterion(crash_ego, labels[:, 0])
            weather_loss = criterion(weather, labels[:, 1])
            timing_loss = criterion(timing, labels[:, 2])
            loss = crashego_loss + weather_loss + timing_loss

            val_lossE1.append(crashego_loss.item())
            val_lossE2.append(weather_loss.item())
            val_lossE3.append(timing_loss.item())
            val_loss.append(loss.item())

            crashego_preds += crash_ego.argmax(1).detach().cpu().numpy().tolist()
            weather_preds += weather.argmax(1).detach().cpu().numpy().tolist()
            timing_preds += timing.argmax(1).detach().cpu().numpy().tolist()

            crashego_trues += labels[:, 0].detach().cpu().numpy().tolist()
            weather_trues += labels[:, 1].detach().cpu().numpy().tolist()
            timing_trues += labels[:, 2].detach().cpu().numpy().tolist()

        _val_lossE1 = np.mean(val_lossE1)
        _val_lossE2 = np.mean(val_lossE2)
        _val_lossE3 = np.mean(val_lossE3)
        _val_loss = np.mean(val_loss)
        losses.append(_val_loss)
        losses.append(_val_lossE1)
        losses.append(_val_lossE2)
        losses.append(_val_lossE3)

    origin_trues = convert_to_label(np.array(crashego_trues), np.array(weather_trues),
                                    np.array(timing_trues))
    origin_preds = convert_to_label(np.array(crashego_preds), np.array(weather_preds),
                                    np.array(timing_preds))
    _val_score = f1_score(origin_trues, origin_preds, average='macro')
    crashego_score = f1_score(crashego_trues, crashego_preds, average='macro')
    weather_score = f1_score(weather_trues, weather_preds, average='macro')
    timing_score = f1_score(timing_trues, timing_preds, average='macro')
    scores.append(_val_score)
    scores.append(crashego_score)
    scores.append(weather_score)
    scores.append(timing_score)

    acc = accuracy_score(origin_trues, origin_preds)
    crashego_acc = accuracy_score(crashego_trues, crashego_preds)
    weather_acc = accuracy_score(weather_trues, weather_preds)
    timing_acc = accuracy_score(timing_trues, timing_preds)
    accuracies.append(acc)
    accuracies.append(crashego_acc)
    accuracies.append(weather_acc)
    accuracies.append(timing_acc)

    return losses, scores, accuracies
    # return _val_loss, _val_score, acc, _val_lossE1, _val_lossE2, _val_lossE3
'''