##############
''' IMPORT '''
##############
import cv2
import random
import numpy as np

import torch
from torch.utils.data import Dataset

import PIL
from PIL import Image
from vidaug import augmentors as va
from torchvision.transforms import Compose, ToTensor, Normalize

''' Probability '''
sometimes1 = lambda aug: va.Sometimes(0.3, aug) # Used to apply augmentor with 50% probability
sometimes2 = lambda aug: va.Sometimes(0.4, aug)
sometimes3 = lambda aug: va.Sometimes(0.5, aug)

''' Normalize '''
train_transform = Compose([
        ToTensor(),
        Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])])

########################
''' RANDOM TRANSLATE '''
########################
class RandomTranslate(object):
    """
      Shifting video in X and Y coordinates.

        Args:
            x (int) : Translate in x direction, selected
            randomly from [-x, +x] pixels.

            y (int) : Translate in y direction, selected
            randomly from [-y, +y] pixels.
    """
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __call__(self, clip):
        x_move = random.randint(-self.x, +self.x+1)
        y_move = random.randint(-self.y, +self.y+1)

        if isinstance(clip[0], np.ndarray):
            rows, cols, ch = clip[0].shape
            transform_mat = np.float32([[1, 0, x_move], [0, 1, y_move]])
            return [cv2.warpAffine(img, transform_mat, (cols, rows), None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(128,128,128)) for img in clip]
        elif isinstance(clip[0], PIL.Image.Image):
            return [img.transform(img.size, PIL.Image.AFFINE, (1, 0, x_move, 0, 1, y_move), fillcolor=(128, 128, 128)) for img in clip]
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))

##########################
''' VIDEO AUGMENTATION '''
##########################
def video_transform(frame):
    randnum = random.randint(1, 2)
    rand_x = random.randint(0, 32)
    rand_y = random.randint(0, 32)
    pmAdd = random.randint(1, 2)
    randadd = random.randint(1, 50)
    if pmAdd == 1:
        parm = randadd
    else:
        parm = randadd * (-1)
    if randnum == 1:
        seq = va.Sequential([
            sometimes1(RandomTranslate(x=rand_x, y=rand_y)),
            sometimes3(va.Add(parm)),
            sometimes2(va.Downsample(0.8))
        ])
        return seq(frame)
    elif randnum == 2:
        seq = va.Sequential([
            sometimes1(RandomTranslate(x=rand_x, y=rand_y)),
            sometimes3(va.Add(parm)),
            sometimes2(va.Upsample(2.0))
        ])
        return seq(frame)

##############################
''' CV2 IMAGE TO PIL IMAGE '''
##############################
def cv2_to_pil(image):
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

#########################
''' DATASET FOR TRAIN '''
#########################
class CustomDataset(Dataset):
    def __init__(self, video_path_list, label_list, video_length, size, mode, label_name):
        self.video_path_list = video_path_list
        self.label_list = label_list
        self.video_length = video_length
        self.size = size
        self.mode = mode
        self.label_name = label_name

    def __getitem__(self, index):
        frames = self.get_video(self.video_path_list[index])
        if frames.shape[1] > 40:
            if frames.shape[1] == 50:
                frames = frames[:, 10:, :, :]
            else:
                # print("first:",frames.shape[1])
                frames = frames[:, 60:, :, :]
                # print("second:",frames.shape[1])

        if self.label_list is not None:
            label = self.label_list[index]
            return frames, label
        else:
            return frames

    def __len__(self):
        return len(self.video_path_list)

    def get_video(self, path):
        frames = []
        cap = cv2.VideoCapture(path)
        for _ in range(self.video_length):
            _, img = cap.read()
            img = cv2.resize(img, (self.size, self.size))
            img = cv2_to_pil(img)
            frames.append(img)
        if self.mode == 'train':
            frames = video_transform(frames)
        return torch.FloatTensor(((np.stack(frames) / 255. - 0.45) / 0.225)).permute(3,0,1,2)

###################################
''' DATASET FOR PSEUDO LABELING '''
###################################
class CustomDataset2(Dataset):
    def __init__(self, sample_id_list, video_path_list, label_list, video_length, size, mode, label_name):
        self.sample_id_list = sample_id_list
        self.video_path_list = video_path_list
        self.label_list = label_list
        self.video_length = video_length
        self.size = size
        self.mode = mode
        self.label_name = label_name

    def __getitem__(self, index):
        frames = self.get_video(self.video_path_list[index])
        if frames.shape[1] > 40:
            if frames.shape[1] == 50:
                frames = frames[:, 10:, :, :]
            else:
                frames = frames[:, 60:, :, :]

        if self.label_list is not None:
            label = self.label_list[index]
            return frames, label
        else:
            return frames, self.video_path_list[index], self.sample_id_list[index]

    def __len__(self):
        return len(self.video_path_list)

    def get_video(self, path):
        frames = []
        cap = cv2.VideoCapture(path)
        for _ in range(self.video_length):
            _, img = cap.read()
            img = cv2.resize(img, (self.size, self.size))
            img = cv2_to_pil(img)
            frames.append(img)
        if self.mode == 'train':
            frames = video_transform(frames)
        return torch.FloatTensor(((np.stack(frames) / 255. - 0.45) / 0.225)).permute(3,0,1,2)

# SlowFast
# class CustomDataset1(Dataset):
#     def __init__(self, video_path_list, label_list, video_length, height, width):
#         self.video_path_list = video_path_list
#         self.label_list = label_list
#         self.video_length = video_length
#         self.height = height
#         self.width = width
#         self.resizing = False
#         self.norm = Normalize(mean, std)
#
#     def __getitem__(self, index):
#         frames = self.get_video(self.video_path_list[index])[:, 2:, :, :]
#         low_frames = torch.index_select(
#             frames,
#             1,
#             torch.linspace(
#                 0, frames.shape[1] - 1, frames.shape[1] // 4
#             ).long(),
#         )
#         # frames = self.get_video(self.video_path_list[index])[:, 2:, :, :]
#         # low_frames = frames[:, [i * 4 + 3 for i in range(12)], :, :]
#
#         if self.label_list is not None:
#             label = self.label_list[index]
#             return low_frames, frames, label
#         else:
#             return low_frames, frames
#
#     def __len__(self):
#         return len(self.video_path_list)
#
#     def get_video(self, path):
#         frames = []
#         cap = cv2.VideoCapture(path)
#         for _ in range(self.video_length):
#             _, img = cap.read()
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             img = cv2.resize(img, (self.height, self.width))
#             img = img / 255.
#             frames.append(img)
#         return self.norm(torch.FloatTensor(np.array(frames)).permute(3, 0, 1, 2))
