#################
''' INFERENCE '''
#################
''' 3D ResNet18 '''
def inference(model, videos):
    logit = model(videos)
    preds = logit.argmax(1).detach().cpu().numpy().tolist()
    return preds

''' SlowFast '''
# def inference(model, test_loader, device):
#     model.to(device)
#     model.eval()
#     crash_preds, weather_preds, timing_preds = [], [], []
#     with torch.no_grad():
#         for slow_videos, fast_videos in tqdm(iter(test_loader)):
#             fast_videos = fast_videos.to(device)
#             slow_videos = slow_videos.to(device)
#
#             crash, weather, timing = model([slow_videos, fast_videos])
#
#             crash_preds += crash.argmax(1).detach().cpu().numpy().tolist()
#             weather_preds += weather.argmax(1).detach().cpu().numpy().tolist()
#             timing_preds += timing.argmax(1).detach().cpu().numpy().tolist()
#
#     return crash_preds, weather_preds, timing_preds