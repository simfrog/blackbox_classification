##############
''' IMPORT '''
##############
import torch.nn as nn
import torchvision.models as models

# model_name = "slowfast_r50"
# slowfast_r50 = torch.hub.load("facebookresearch/pytorchvideo", model=model_name,pretrained=True)
# slowfast_r50_layers = slowfast_r50.blocks

##################
''' BASE MODEL '''
##################
class ResNet18_3D(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18_3D, self).__init__()
        self.feature_extract = models.video.r3d_18(pretrained=True)
        self.classifier = nn.Sequential(
                            nn.Linear(400, 256),
                            nn.ReLU(),
                            nn.Dropout(0.3),
                            nn.Linear(256, num_classes),
                            )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.feature_extract(x)
        x = x.view(batch_size, -1)
        x = self.classifier(x)
        return x

# class SlowFast(nn.Module):
#     def __init__(self, img_size):
#         super(SlowFast, self).__init__()
#         self.pool_size = img_size // 32 - 1
#
#         self.layer_1 = slowfast_r50_layers[0]
#         self.layer_2 = slowfast_r50_layers[1]
#         self.layer_3 = slowfast_r50_layers[2]
#         self.layer_4 = slowfast_r50_layers[3]
#         self.layer_5 = slowfast_r50_layers[4]
#         self.fast_pool = nn.AvgPool3d(kernel_size=(12, self.pool_size, self.pool_size), stride=(1, 1, 1),
#                                       padding=(0, 0, 0))
#         self.slow_pool = nn.AvgPool3d(kernel_size=(48, self.pool_size, self.pool_size), stride=(1, 1, 1),
#                                       padding=(0, 0, 0))
#
#         self.avgpool = nn.AdaptiveAvgPool3d(1)
#         self.crashego_fc = nn.Linear(2304, 3)
#         self.weather_fc = nn.Linear(2304, 3)
#         self.timing_fc = nn.Linear(2304, 2)
#         for para in self.layer_1.parameters():
#             para.requires_grad = False
#         for para in self.layer_2.parameters():
#             para.requires_grad = False
#
#     def forward(self, x):
#         batch_size = x[0].size(0)
#         x = self.layer_1(x)
#         x = self.layer_2(x)
#         x = self.layer_3(x)
#         x = self.layer_4(x)
#         x = self.layer_5(x)
#
#         x_1 = self.fast_pool(x[0])
#         x_2 = self.slow_pool(x[1])
#         x = torch.concat([x_1, x_2], axis=1)
#         x = self.avgpool(x).view(batch_size, -1)
#
#         crash_ego = self.crashego_fc(x)
#         weather = self.weather_fc(x)
#         timing = self.timing_fc(x)
#         return crash_ego, weather, timing